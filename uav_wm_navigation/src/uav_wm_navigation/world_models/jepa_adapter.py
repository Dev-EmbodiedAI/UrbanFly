from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .base import (
    FORWARD_CLEARANCE,
    PLANNING_STATE_DIM,
    POSITION,
    VELOCITY,
    YAW,
    YAW_RATE,
    WorldModelBase,
    validate_planning_state,
)
from .factory import build_world_model
from .jepa_world_model import ActionConditionedJEPAWorldModel


def _residual_mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
    for _ in range(max(layers - 1, 0)):
        modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class JEPAWorldModelAdapter(WorldModelBase):
    """Step-wise action-conditioned adapter around the existing JEPA encoder.

    The kinematic backbone reflects UrbanFly's high-level body-FLU velocity
    interface. Learned residual dynamics and probes model tracking error and
    collision risk without reconstructing RGB or depth pixels.
    """

    def __init__(
        self,
        core: ActionConditionedJEPAWorldModel,
        *,
        latent_dim: int,
        hidden_dim: int = 256,
        dynamics_layers: int = 2,
        action_scale: tuple[float, float, float, float] = (6.0, 6.0, 3.0, 1.0471976),
        physics_residual_scale: float = 0.25,
        clearance_margin_m: float = 0.75,
        clearance_temperature_m: float = 0.35,
    ) -> None:
        super().__init__()
        self.core = core
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.physics_residual_scale = float(physics_residual_scale)
        self.clearance_margin_m = float(clearance_margin_m)
        self.clearance_temperature_m = float(clearance_temperature_m)
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        joined_dim = self.latent_dim + PLANNING_STATE_DIM + self.action_dim
        self.latent_dynamics = _residual_mlp(joined_dim, hidden_dim, self.latent_dim, dynamics_layers)
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.physics_probe = _residual_mlp(joined_dim, hidden_dim, 6, max(dynamics_layers - 1, 1))
        self.safety_probe = _residual_mlp(joined_dim, hidden_dim, 1, max(dynamics_layers - 1, 1))
        # Stable zero-residual initialization makes the untrained adapter an
        # honest velocity-command kinematic model rather than a random policy.
        nn.init.zeros_(self.physics_probe[-1].weight)
        nn.init.zeros_(self.physics_probe[-1].bias)
        nn.init.zeros_(self.safety_probe[-1].weight)
        nn.init.constant_(self.safety_probe[-1].bias, math.log(0.02 / 0.98))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(
        self,
        observation: Mapping[str, torch.Tensor],
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        required = ("depth", "state_history", "goal_body")
        missing = [name for name in required if name not in observation]
        if missing:
            raise ValueError(f"JEPA observation is missing {missing}")
        latent = self.core.context(
            observation["depth"],
            observation["state_history"],
            observation["goal_body"],
        )
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("JEPA encoder returned an invalid latent shape")
        if not torch.isfinite(latent).all():
            raise FloatingPointError("JEPA encoder returned a non-finite latent")
        return latent

    @staticmethod
    def _body_velocity_to_world(physical_action: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        forward, left, up = physical_action[:, 0], physical_action[:, 1], physical_action[:, 2]
        return torch.stack(
            [cosine * forward - sine * left, sine * forward + cosine * left, up],
            dim=-1,
        )

    def predict_step(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        dt: float | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        validate_planning_state(state)
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent has the wrong shape")
        if action.ndim != 2 or action.shape != (latent.shape[0], self.action_dim):
            raise ValueError("action must have shape [B, 4]")
        dt_tensor = torch.as_tensor(dt, dtype=latent.dtype, device=latent.device)
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.expand(latent.shape[0])
        if dt_tensor.shape != (latent.shape[0],):
            raise ValueError("dt must be scalar or have shape [B]")
        if not torch.isfinite(action).all() or not torch.isfinite(dt_tensor).all() or (dt_tensor <= 0.0).any():
            raise ValueError("action/dt must be finite and dt must be positive")
        dt_column = dt_tensor[:, None]
        action = action.clamp(-1.0, 1.0)
        joined = torch.cat([latent, state, action], dim=-1)
        next_latent = self.latent_norm(latent + 0.1 * self.latent_dynamics(joined))
        physical = action * self.action_scale.to(action)
        next_yaw = state[:, YAW] + physical[:, 3] * dt_tensor
        command_velocity = self._body_velocity_to_world(physical, next_yaw)
        residual = self.physics_probe(joined) * self.physics_residual_scale
        predicted_velocity = command_velocity + residual[:, 3:]
        predicted_position = state[:, POSITION] + predicted_velocity * dt_column + residual[:, :3] * dt_column
        next_state = state.clone()
        next_state[:, POSITION] = predicted_position
        next_state[:, VELOCITY] = predicted_velocity
        next_state[:, YAW] = next_yaw
        next_state[:, YAW_RATE] = physical[:, 3]
        forward_travel = torch.relu(physical[:, 0]) * dt_tensor
        next_clearance = torch.clamp(state[:, FORWARD_CLEARANCE] - forward_travel, min=0.0)
        next_state[:, FORWARD_CLEARANCE] = next_clearance
        learned_logit = self.safety_probe(joined).squeeze(-1)
        clearance_logit = (
            self.clearance_margin_m - next_clearance
        ) / max(self.clearance_temperature_m, 1e-3)
        # Noisy-or combines a learned probe and an auditable near-field prior.
        learned_probability = torch.sigmoid(learned_logit)
        clearance_probability = torch.sigmoid(clearance_logit)
        collision_probability = 1.0 - (1.0 - learned_probability) * (1.0 - clearance_probability)
        return {
            "latent": next_latent,
            "state": next_state,
            "state_prediction": {
                "position": predicted_position,
                "velocity": predicted_velocity,
            },
            "collision_logits": learned_logit,
            "collision_probability": collision_probability.clamp(0.0, 1.0),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        checkpoint: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple["JEPAWorldModelAdapter", dict[str, Any]]:
        payload: dict[str, Any] | None = None
        path: Path | None = None
        if checkpoint is not None:
            path = Path(checkpoint).expanduser().resolve()
            payload = torch.load(path, map_location=map_location, weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError("checkpoint payload must be a mapping")
        model_config = dict(config.get("jepa", config))
        if payload is not None and isinstance(payload.get("config"), Mapping):
            checkpoint_config = payload["config"]
            checkpoint_jepa = checkpoint_config.get("jepa", checkpoint_config)
            if isinstance(checkpoint_jepa, Mapping) and checkpoint_jepa.get("model", "jepa") == "jepa":
                # Architecture stored with a checkpoint wins over runtime
                # defaults; planner/cost settings remain caller-controlled.
                model_config.update(checkpoint_jepa)
        model_config.setdefault("model", "jepa")
        model_config.setdefault("state_dim", 13)
        model_config.setdefault("trajectory_dim", 9)
        model_config.setdefault("latent_dim", int(config.get("latent_dim", 96)))
        model_config.setdefault("dropout", 0.15)
        core = build_world_model(model_config)
        if not isinstance(core, ActionConditionedJEPAWorldModel):
            raise TypeError("JEPA adapter requires an ActionConditionedJEPAWorldModel")
        adapter_config = dict(config.get("adapter", {}))
        if payload is not None and "adapter_state" in payload:
            checkpoint_adapter = payload.get("config", {}).get("adapter", {})
            if isinstance(checkpoint_adapter, Mapping):
                adapter_config.update(checkpoint_adapter)
        adapter = cls(core, latent_dim=int(model_config["latent_dim"]), **adapter_config)
        provenance: dict[str, Any] = {"trained": False, "checkpoint": None, "format": "untrained"}
        if payload is not None and path is not None:
            if "adapter_state" in payload:
                adapter.load_state_dict(payload["adapter_state"])
                provenance.update({"trained": int(payload.get("training_steps", 0)) > 0, "format": "jepa_wam_mpc"})
            elif "model" in payload:
                adapter.core.load_state_dict(payload["model"])
                provenance.update({
                    "trained": bool(payload.get("training_history")) or int(payload.get("training_steps", 0)) > 0,
                    "format": "candidate_jepa_encoder",
                })
            else:
                raise ValueError("checkpoint contains neither adapter_state nor model")
            provenance["checkpoint"] = str(path)
        return adapter, provenance
