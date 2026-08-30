from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import torch
from torch import nn


# Canonical learned-planning state. Positions and velocities are world-NWU,
# yaw follows the existing positive-counter-clockwise convention, and depth is
# metric forward clearance from the latest camera frame.
POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
YAW = 6
YAW_RATE = 7
FORWARD_CLEARANCE = 8
PLANNING_STATE_DIM = 9


class WorldModelBase(nn.Module, ABC):
    """Model-agnostic contract consumed by receding-horizon planners.

    Candidate trajectories are represented by the leading batch dimension.
    Implementations must never loop over that dimension; a loop over the short
    rollout horizon is acceptable for recurrent dynamics.
    """

    action_dim: int = 4
    action_representation: str = "normalized_body_flu"

    def planner_action_to_normalized(self, action: torch.Tensor) -> torch.Tensor:
        """Convert planner-space actions to the environment's normalized API."""

        return action.clamp(-1.0, 1.0)

    @abstractmethod
    def encode(
        self,
        observation: Mapping[str, torch.Tensor],
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the current observation into ``[B, latent_dim]``."""

    @abstractmethod
    def predict_step(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        dt: float | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict one action-conditioned latent/state/risk transition."""

    def rollout(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        action_sequence: torch.Tensor,
        *,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Vectorized multi-step imagination over ``[N, H, action_dim]``."""

        if action_sequence.ndim != 3 or action_sequence.shape[-1] != self.action_dim:
            raise ValueError("action_sequence must have shape [N, H, action_dim]")
        count, horizon, _ = action_sequence.shape
        if latent.ndim != 2 or latent.shape[0] not in {1, count}:
            raise ValueError("latent must have shape [1|N, latent_dim]")
        if state.ndim != 2 or state.shape != (state.shape[0], PLANNING_STATE_DIM):
            raise ValueError(f"state must have shape [B, {PLANNING_STATE_DIM}]")
        if state.shape[0] not in {1, count}:
            raise ValueError("state leading dimension must be 1 or match candidates")
        current_latent = latent.expand(count, -1)
        current_state = state.expand(count, -1)
        latents: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        risks: list[torch.Tensor] = []
        for step in range(horizon):
            prediction = self.predict_step(
                current_latent,
                current_state,
                action_sequence[:, step],
                dt=dt,
            )
            current_latent = prediction["latent"]
            current_state = prediction["state"]
            latents.append(current_latent)
            states.append(current_state)
            risks.append(prediction["collision_probability"])
        return {
            "latent": torch.stack(latents, dim=1),
            "state": torch.stack(states, dim=1),
            "position": torch.stack(states, dim=1)[..., POSITION],
            "velocity": torch.stack(states, dim=1)[..., VELOCITY],
            "collision_probability": torch.stack(risks, dim=1),
        }

    def predict_cost_features(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Optional current-step features exposed to a planner."""

        return {
            "position": state[..., POSITION],
            "velocity": state[..., VELOCITY],
        }


def validate_planning_state(state: torch.Tensor) -> None:
    if state.ndim != 2 or state.shape[-1] != PLANNING_STATE_DIM:
        raise ValueError(f"planning state must have shape [B, {PLANNING_STATE_DIM}]")
    if not torch.isfinite(state).all():
        raise ValueError("planning state contains non-finite values")
