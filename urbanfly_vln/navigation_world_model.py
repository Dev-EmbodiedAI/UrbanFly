from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .observation_policy import ACTION_LIMITS


CHECKPOINT_SCHEMA = "urbanfly-helsinki-latent-world-model-v1"
PHYSICAL_TARGET_NAMES = (
    "delta_forward_m",
    "delta_left_m",
    "delta_up_m",
    "route_progress_delta_m",
    "next_clearance_m",
)


@dataclass(frozen=True)
class NavigationWorldModelConfig:
    latent_dim: int = 192
    hidden_dim: int = 256
    ensemble_size: int = 3


class LatentDynamicsMember(nn.Module):
    def __init__(self, config: NavigationWorldModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.latent_dim + 4, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 5 + config.latent_dim),
        )

    def forward(self, latent: torch.Tensor, normalized_action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((latent, normalized_action), dim=-1))


class HelsinkiLatentWorldModel(nn.Module):
    """Action-conditioned ensemble over the observation-policy latent state."""

    def __init__(
        self,
        config: NavigationWorldModelConfig | None = None,
        *,
        target_mean: np.ndarray | torch.Tensor | None = None,
        target_std: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NavigationWorldModelConfig()
        self.members = nn.ModuleList(
            LatentDynamicsMember(self.config) for _ in range(self.config.ensemble_size)
        )
        width = 5 + self.config.latent_dim
        mean = torch.as_tensor(
            np.zeros(width, np.float32) if target_mean is None else target_mean,
            dtype=torch.float32,
        )
        std = torch.as_tensor(
            np.ones(width, np.float32) if target_std is None else target_std,
            dtype=torch.float32,
        )
        if mean.shape != (width,) or std.shape != (width,) or torch.any(std <= 0):
            raise ValueError("world-model target statistics have the wrong shape")
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_std", std)
        self.register_buffer("action_limits", torch.as_tensor(ACTION_LIMITS.copy()))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def member_prediction(
        self, member_index: int, latent: torch.Tensor, action_physical: torch.Tensor
    ) -> torch.Tensor:
        normalized_action = torch.clamp(action_physical / self.action_limits, -1.0, 1.0)
        return self.members[member_index](latent, normalized_action)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.target_std + self.target_mean

    @torch.inference_mode()
    def predict(self, latent: torch.Tensor, candidate_actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_dim:
            raise ValueError("latent must have shape [batch, latent_dim]")
        if candidate_actions.ndim != 3 or candidate_actions.shape[0] != latent.shape[0] or candidate_actions.shape[2] != 4:
            raise ValueError("candidate_actions must have shape [batch, candidates, 4]")
        batch, candidates, _ = candidate_actions.shape
        expanded_latent = latent[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        flattened_actions = candidate_actions.reshape(batch * candidates, 4)
        predictions = []
        for member_index in range(self.config.ensemble_size):
            normalized = self.member_prediction(member_index, expanded_latent, flattened_actions)
            predictions.append(self.denormalize(normalized).reshape(batch, candidates, -1))
        members = torch.stack(predictions)
        mean = members.mean(dim=0)
        std = members.std(dim=0, unbiased=False)
        return {
            "member_predictions": members,
            "physical_mean": mean[..., :5],
            "physical_std": std[..., :5],
            "next_latent_mean": latent[:, None] + mean[..., 5:],
            "next_latent_std": std[..., 5:],
        }


def save_navigation_world_model_checkpoint(
    path: Path,
    model: HelsinkiLatentWorldModel,
    *,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": asdict(model.config),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
        "physical_target_names": list(PHYSICAL_TARGET_NAMES),
    }
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def load_navigation_world_model_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[HelsinkiLatentWorldModel, dict[str, Any]]:
    payload = torch.load(path.resolve(), map_location=device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported navigation world-model checkpoint schema")
    if tuple(payload.get("physical_target_names", ())) != PHYSICAL_TARGET_NAMES:
        raise ValueError("navigation world-model target contract mismatch")
    config = NavigationWorldModelConfig(**payload["config"])
    state = payload["model"]
    model = HelsinkiLatentWorldModel(
        config,
        target_mean=state["target_mean"],
        target_std=state["target_std"],
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(payload.get("metadata") or {})
