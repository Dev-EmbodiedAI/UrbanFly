from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from uav_wm_navigation.risk.cpa import DepthHistory
from uav_wm_navigation.types import BodyVelocityAction, EpisodeSpec, WorldModelObservation
from .continuous_protocol import episode_id_from_spec


class ARRFlyActorCritic(nn.Module):
    """15-frame depth CNN+TCN actor with a privileged asymmetric critic."""

    def __init__(self, proprio_dim: int = 16, privileged_dim: int = 36) -> None:
        super().__init__()
        self.frame = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 9)), nn.Flatten(), nn.Linear(32 * 2 * 9, 128), nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 160, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv1d(160, 160, 3, padding=4, dilation=4), nn.SiLU(),
        )
        self.public = nn.Sequential(nn.Linear(160 + proprio_dim, 192), nn.LayerNorm(192), nn.SiLU())
        self.actor_mean = nn.Linear(192, 4)
        self.actor_log_std = nn.Parameter(torch.full((4,), -0.5))
        self.critic = nn.Sequential(nn.Linear(192 + privileged_dim, 256), nn.SiLU(), nn.Linear(256, 1))

    def encode_public(self, depth_history: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if depth_history.ndim != 4 or depth_history.shape[1:] != (15, 6, 34):
            raise ValueError("depth_history must have shape [B,15,6,34]")
        batch = depth_history.shape[0]
        frames = self.frame(depth_history.reshape(batch * 15, 1, 6, 34)).reshape(batch, 15, -1)
        temporal = self.temporal(frames.transpose(1, 2))[..., :15].mean(-1)
        return self.public(torch.cat([temporal, proprio], dim=-1))

    def distribution(self, depth_history: torch.Tensor, proprio: torch.Tensor) -> Normal:
        public = self.encode_public(depth_history, proprio)
        return Normal(torch.tanh(self.actor_mean(public)), self.actor_log_std.exp().expand_as(self.actor_mean(public)))

    def value(self, depth_history: torch.Tensor, proprio: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([self.encode_public(depth_history, proprio), privileged], dim=-1)).squeeze(-1)


def asymmetric_ppo_loss(
    model: ARRFlyActorCritic,
    batch: dict[str, torch.Tensor],
    *,
    clip_ratio: float = 0.2,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    distribution = model.distribution(batch["depth_history"], batch["proprio"])
    log_probability = distribution.log_prob(batch["action"]).sum(-1)
    ratio = torch.exp(log_probability - batch["old_log_probability"])
    advantage = batch["advantage"]
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage).mean()
    value = nn.functional.mse_loss(model.value(batch["depth_history"], batch["proprio"], batch["critic_privileged"]), batch["return"])
    entropy = distribution.entropy().sum(-1).mean()
    total = policy + value_weight * value - entropy_weight * entropy
    return total, {"policy": policy, "value": value, "entropy": entropy}


class ARRFlyPPOPolicy:
    def __init__(self, checkpoint: str | Path, *, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema") != "urbanfly-world-model-v3" or payload.get("family") != "arr_fly_ppo" or int(payload.get("training_steps", 0)) <= 0:
            raise ValueError("checkpoint lacks trained ARR-Fly PPO provenance")
        self.model = ARRFlyActorCritic().to(self.device)
        self.model.load_state_dict(payload["model"]); self.model.eval()
        self.checkpoint, self.history = str(path), DepthHistory()
        self._episode_id, self._proprio, self._diagnostics = "", None, {}

    def reset(self, episode: str | EpisodeSpec) -> None:
        self._episode_id = episode_id_from_spec(episode); self.history.reset(); self._proprio = None

    def observe(self, observation: WorldModelObservation) -> None:
        if self._episode_id and observation.episode_id != self._episode_id:
            raise ValueError("observation belongs to a different ARR-Fly episode")
        self.history.append(observation.depth_m, observation.depth_valid_mask)
        self._proprio = np.concatenate([
            np.clip(observation.goal_body_flu_m / 120.0, -1, 1),
            observation.linear_velocity_body_flu_mps / 6.0,
            observation.angular_velocity_body_flu_rps / 2.0,
            observation.gravity_body_flu, observation.previous_action,
        ]).astype(np.float32)

    def act(self, deterministic: bool = True) -> BodyVelocityAction:
        if self._proprio is None:
            raise RuntimeError("observe must be called before ARR-Fly act")
        depth = torch.from_numpy(self.history.array()[None]).to(self.device)
        proprio = torch.from_numpy(self._proprio[None]).to(self.device)
        with torch.inference_mode():
            distribution = self.model.distribution(depth, proprio)
            action = distribution.mean if deterministic else distribution.sample()
        normalized = action.clamp(-1.0, 1.0)[0].cpu().numpy()
        self._diagnostics = {"family": "arr_fly_ppo", "checkpoint": self.checkpoint, "depth_history_frames": 15}
        return BodyVelocityAction(normalized)

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)
