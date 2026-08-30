from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


CHECKPOINT_SCHEMA = "urbanfly-helsinki-observation-policy-v1"
ACTION_NAMES = ("forward_mps", "left_mps", "up_mps", "yaw_rate_rps")
ACTION_LIMITS = np.asarray([6.0, 6.0, 3.0, np.deg2rad(60.0)], dtype=np.float32)
STATE_FEATURE_NAMES = (
    "local_goal_body_forward_m",
    "local_goal_body_left_m",
    "local_goal_body_up_m",
    "linear_velocity_body_forward_mps",
    "linear_velocity_body_left_mps",
    "linear_velocity_body_up_mps",
    "angular_velocity_body_forward_rps",
    "angular_velocity_body_left_rps",
    "angular_velocity_body_up_rps",
    "gravity_body_forward",
    "gravity_body_left",
    "gravity_body_up",
    "previous_action_forward_normalized",
    "previous_action_left_normalized",
    "previous_action_up_normalized",
    "previous_action_yaw_normalized",
)


@dataclass(frozen=True)
class ObservationPolicyConfig:
    history_frames: int = 2
    image_height: int = 48
    image_width: int = 80
    depth_max_m: float = 50.0
    hidden_dim: int = 192
    state_dim: int = len(STATE_FEATURE_NAMES)

    def __post_init__(self) -> None:
        if self.history_frames < 1:
            raise ValueError("history_frames must be positive")
        if self.image_height < 16 or self.image_width < 16:
            raise ValueError("policy image size is too small")
        if self.depth_max_m <= 0:
            raise ValueError("depth_max_m must be positive")
        if self.state_dim != len(STATE_FEATURE_NAMES):
            raise ValueError("state_dim does not match the public policy feature contract")


class HelsinkiObservationPolicy(nn.Module):
    """Compact RGB-D + public-state velocity policy.

    Inputs deliberately exclude the global route, global position, collision
    geometry, clearance labels, task identity and expert planner state.  The
    only navigation target is the receding-horizon Local Goal in body FLU.
    """

    def __init__(
        self,
        config: ObservationPolicyConfig | None = None,
        *,
        action_mean: np.ndarray | torch.Tensor | None = None,
        action_std: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ObservationPolicyConfig()
        channels = self.config.history_frames * 4
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(self.config.hidden_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 4),
        )
        mean = torch.as_tensor(
            np.zeros(4, dtype=np.float32) if action_mean is None else action_mean,
            dtype=torch.float32,
        )
        std = torch.as_tensor(
            np.ones(4, dtype=np.float32) if action_std is None else action_std,
            dtype=torch.float32,
        )
        if mean.shape != (4,) or std.shape != (4,) or torch.any(std <= 0):
            raise ValueError("action normalization must contain four positive scales")
        self.register_buffer("action_mean", mean)
        self.register_buffer("action_std", std)
        self.register_buffer(
            "state_scale",
            torch.tensor(
                [20.0, 20.0, 20.0, 6.0, 6.0, 3.0, 2.0, 2.0, 2.0]
                + [1.0] * 7,
                dtype=torch.float32,
            ),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _visual_input(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor,
        depth_valid: torch.Tensor,
    ) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[1] != self.config.history_frames or rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape [batch, history, height, width, 3]")
        if depth_m.shape != rgb.shape[:-1] or depth_valid.shape != depth_m.shape:
            raise ValueError("depth tensors must align with RGB history")
        rgb_value = rgb.float().permute(0, 1, 4, 2, 3).reshape(
            len(rgb), self.config.history_frames * 3, rgb.shape[2], rgb.shape[3]
        ) / 255.0
        valid = depth_valid.bool() & torch.isfinite(depth_m) & (depth_m > 0.0)
        depth_value = torch.where(valid, depth_m.float(), torch.zeros_like(depth_m.float()))
        depth_value = depth_value.clamp(0.0, self.config.depth_max_m) / self.config.depth_max_m
        depth_value = depth_value.reshape(
            len(rgb), self.config.history_frames, depth_m.shape[2], depth_m.shape[3]
        )
        visual = torch.cat((rgb_value, depth_value), dim=1)
        return F.interpolate(
            visual,
            size=(self.config.image_height, self.config.image_width),
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor,
        depth_valid: torch.Tensor,
        public_state: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encode(rgb, depth_m, depth_valid, public_state)
        normalized_action = self.head(latent)
        physical = normalized_action * self.action_std + self.action_mean
        limits = torch.as_tensor(ACTION_LIMITS, device=physical.device, dtype=physical.dtype)
        return torch.maximum(torch.minimum(physical, limits), -limits)

    def encode(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor,
        depth_valid: torch.Tensor,
        public_state: torch.Tensor,
    ) -> torch.Tensor:
        """Return the public-observation latent used by the action head.

        This is an auditable representation interface for the separately
        trained action-conditioned dynamics model. It does not add privileged
        inputs or change the policy checkpoint parameterization.
        """

        if public_state.ndim != 2 or public_state.shape[1] != self.config.state_dim:
            raise ValueError("public_state has the wrong shape")
        visual = self.visual_encoder(self._visual_input(rgb, depth_m, depth_valid))
        state = self.state_encoder(public_state.float() / self.state_scale)
        return torch.cat((visual, state), dim=1)

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_normalized = (predicted - self.action_mean) / self.action_std
        target_normalized = (target - self.action_mean) / self.action_std
        return F.smooth_l1_loss(predicted_normalized, target_normalized, beta=0.5)


def save_observation_policy_checkpoint(
    path: Path,
    model: HelsinkiObservationPolicy,
    *,
    metadata: dict[str, Any],
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": asdict(model.config),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
        "action_names": list(ACTION_NAMES),
        "state_feature_names": list(STATE_FEATURE_NAMES),
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_observation_policy_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[HelsinkiObservationPolicy, dict[str, Any]]:
    payload = torch.load(path.resolve(), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported observation-policy checkpoint schema")
    if tuple(payload.get("state_feature_names", ())) != STATE_FEATURE_NAMES:
        raise ValueError("checkpoint public-state feature contract does not match runtime")
    config = ObservationPolicyConfig(**payload["config"])
    state = payload["model"]
    model = HelsinkiObservationPolicy(
        config,
        action_mean=state["action_mean"],
        action_std=state["action_std"],
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(payload.get("metadata") or {})
