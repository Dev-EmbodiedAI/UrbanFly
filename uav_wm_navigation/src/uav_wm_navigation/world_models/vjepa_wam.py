from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
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
from .encoders import DepthFrameEncoder


SAFETY_MODES = {"geometry_only", "learned_only", "combined"}


@dataclass(frozen=True, slots=True)
class VJEPAWAMLossWeights:
    latent: float = 1.0
    position: float = 2.0
    velocity: float = 1.0
    collision: float = 5.0


class VJEPAWorldModelAdapter(WorldModelBase):
    """Recursive UAV WAM on top of a pretrained temporal ViT encoder.

    Planner/model actions are physical body-FLU values in m/s and rad/s.
    Pixel reconstruction is deliberately absent.
    """

    action_representation = "physical_body_flu_mps_rps"

    def __init__(
        self,
        visual_encoder: nn.Module,
        *,
        encoder_dim: int,
        latent_dim: int = 256,
        proprio_dim: int = 16,
        proprio_hidden_dim: int = 128,
        depth_auxiliary: bool = True,
        depth_dim: int = 96,
        predictor_hidden_dim: int = 384,
        action_scale: tuple[float, float, float, float] = (6.0, 6.0, 3.0, 1.0471976),
        physics_residual_scale: float = 0.25,
        safety_mode: str = "combined",
        clearance_margin_m: float = 0.75,
        clearance_temperature_m: float = 0.35,
    ) -> None:
        super().__init__()
        if safety_mode not in SAFETY_MODES:
            raise ValueError(f"safety_mode must be one of {sorted(SAFETY_MODES)}")
        self.visual_encoder = visual_encoder
        self.encoder_dim = int(encoder_dim)
        self.latent_dim = int(latent_dim)
        self.depth_auxiliary = bool(depth_auxiliary)
        self.safety_mode = str(safety_mode)
        self.physics_residual_scale = float(physics_residual_scale)
        self.clearance_margin_m = float(clearance_margin_m)
        self.clearance_temperature_m = float(clearance_temperature_m)
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.depth_encoder = DepthFrameEncoder(depth_dim) if self.depth_auxiliary else None
        self.proprio_encoder = nn.GRU(proprio_dim, proprio_hidden_dim, batch_first=True)
        self.goal_encoder = nn.Sequential(nn.Linear(3, 64), nn.LayerNorm(64), nn.SiLU())
        context_dim = self.encoder_dim + proprio_hidden_dim + 64 + (depth_dim if self.depth_auxiliary else 0)
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, predictor_hidden_dim), nn.LayerNorm(predictor_hidden_dim), nn.SiLU(),
            nn.Linear(predictor_hidden_dim, self.latent_dim), nn.LayerNorm(self.latent_dim),
        )
        self.action_encoder = nn.Sequential(nn.Linear(4, 128), nn.LayerNorm(128), nn.SiLU())
        self.state_encoder = nn.Sequential(nn.Linear(5, 96), nn.LayerNorm(96), nn.SiLU())
        self.predictor = nn.GRUCell(128 + 96, self.latent_dim)
        probe_input = self.latent_dim + 128 + 96
        self.physics_probe = nn.Sequential(
            nn.Linear(probe_input, predictor_hidden_dim), nn.SiLU(), nn.Linear(predictor_hidden_dim, 6)
        )
        self.safety_probe = nn.Sequential(
            nn.Linear(probe_input, predictor_hidden_dim), nn.SiLU(), nn.Linear(predictor_hidden_dim, 1)
        )
        nn.init.zeros_(self.physics_probe[-1].weight)
        nn.init.zeros_(self.physics_probe[-1].bias)
        nn.init.zeros_(self.safety_probe[-1].weight)
        nn.init.constant_(self.safety_probe[-1].bias, math.log(0.02 / 0.98))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def planner_action_to_normalized(self, action: torch.Tensor) -> torch.Tensor:
        return (action / self.action_scale.to(action)).clamp(-1.0, 1.0)

    def _visual_features(self, video: torch.Tensor) -> torch.Tensor:
        output = self.visual_encoder(video)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        if output.ndim > 2:
            output = output.reshape(output.shape[0], -1, output.shape[-1]).mean(dim=1)
        if output.shape != (video.shape[0], self.encoder_dim):
            raise ValueError(
                f"visual encoder must return [B,{self.encoder_dim}], got {tuple(output.shape)}"
            )
        return output

    def encode(
        self,
        observation: Mapping[str, torch.Tensor],
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        required = ("rgb_video", "proprio_history", "goal_body")
        missing = [name for name in required if name not in observation]
        if missing:
            raise ValueError(f"V-JEPA observation is missing {missing}")
        video = observation["rgb_video"]
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("rgb_video must have shape [B,T,3,H,W]")
        features = [self._visual_features(video)]
        _, proprio = self.proprio_encoder(observation["proprio_history"])
        features.append(proprio[-1])
        features.append(self.goal_encoder(observation["goal_body"] / 120.0))
        if self.depth_auxiliary:
            if "depth_video" not in observation:
                raise ValueError("depth_video is required when depth_auxiliary=true")
            features.append(self.depth_encoder(observation["depth_video"][:, -1]))
        latent = self.context_projection(torch.cat(features, dim=-1))
        if not torch.isfinite(latent).all():
            raise FloatingPointError("V-JEPA encoder/fusion returned non-finite latent")
        return latent

    def _state_features(self, state: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                state[:, VELOCITY.start] / 6.0,
                state[:, VELOCITY.start + 1] / 6.0,
                state[:, VELOCITY.start + 2] / 3.0,
                state[:, YAW_RATE] / float(self.action_scale[3]),
                state[:, FORWARD_CLEARANCE] / 120.0,
            ],
            dim=-1,
        )

    @staticmethod
    def _body_to_world(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        return torch.stack(
            [cosine * vector[:, 0] - sine * vector[:, 1], sine * vector[:, 0] + cosine * vector[:, 1], vector[:, 2]],
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
        if action.ndim != 2 or action.shape != (latent.shape[0], 4):
            raise ValueError("physical action must have shape [B,4]")
        dt_tensor = torch.as_tensor(dt, dtype=latent.dtype, device=latent.device)
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.expand(latent.shape[0])
        if dt_tensor.shape != (latent.shape[0],) or (dt_tensor <= 0).any():
            raise ValueError("dt must be positive scalar or [B]")
        bounded_action = torch.maximum(
            torch.minimum(action, self.action_scale.to(action)), -self.action_scale.to(action)
        )
        action_feature = self.action_encoder(bounded_action / self.action_scale.to(action))
        state_feature = self.state_encoder(self._state_features(state))
        predictor_input = torch.cat([action_feature, state_feature], dim=-1)
        next_latent = self.predictor(predictor_input, latent)
        probe_input = torch.cat([next_latent, action_feature, state_feature], dim=-1)
        residual = self.physics_probe(probe_input) * self.physics_residual_scale
        dt_column = dt_tensor[:, None]
        delta_body = bounded_action[:, :3] * dt_column + residual[:, :3]
        velocity_body = bounded_action[:, :3] + residual[:, 3:]
        next_yaw = state[:, YAW] + bounded_action[:, 3] * dt_tensor
        delta_world = self._body_to_world(delta_body, next_yaw)
        velocity_world = self._body_to_world(velocity_body, next_yaw)
        next_state = state.clone()
        next_state[:, POSITION] = state[:, POSITION] + delta_world
        next_state[:, VELOCITY] = velocity_world
        next_state[:, YAW] = next_yaw
        next_state[:, YAW_RATE] = bounded_action[:, 3]
        next_clearance = torch.clamp(
            state[:, FORWARD_CLEARANCE] - torch.relu(delta_body[:, 0]), min=0.0
        )
        next_state[:, FORWARD_CLEARANCE] = next_clearance
        learned_logit = self.safety_probe(probe_input).squeeze(-1)
        learned_probability = torch.sigmoid(learned_logit)
        geometry_probability = torch.sigmoid(
            (self.clearance_margin_m - next_clearance) / max(self.clearance_temperature_m, 1e-3)
        )
        if self.safety_mode == "geometry_only":
            collision_probability = geometry_probability
        elif self.safety_mode == "learned_only":
            collision_probability = learned_probability
        else:
            collision_probability = 1.0 - (1.0 - learned_probability) * (1.0 - geometry_probability)
        return {
            "latent": next_latent,
            "state": next_state,
            "state_prediction": {
                "delta_position_body_flu": delta_body,
                "position": next_state[:, POSITION],
                "velocity": velocity_world,
            },
            "collision_logits": learned_logit,
            "learned_collision_probability": learned_probability,
            "geometry_collision_probability": geometry_probability,
            "collision_probability": collision_probability.clamp(0.0, 1.0),
        }


def vjepa_wam_multistep_loss(
    model: VJEPAWorldModelAdapter,
    batch: Mapping[str, torch.Tensor],
    *,
    history_frames: int,
    rollout_steps: int,
    weights: VJEPAWAMLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or VJEPAWAMLossWeights()
    rgb = batch["rgb"].to(next(model.parameters()).device)
    depth = batch["depth"].to(rgb.device)
    proprio = batch["proprio"].to(rgb.device)
    goal = batch["goal_body"].to(rgb.device)
    state = batch["planning_state"].to(rgb.device)
    actions = batch["action_physical"].to(rgb.device)[:, :rollout_steps]
    dt = batch["dt"].to(rgb.device)[:, :rollout_steps]
    if rgb.shape[1] < history_frames + rollout_steps:
        raise ValueError("batch sequence is shorter than history + rollout")
    initial = model.encode({
        "rgb_video": rgb[:, :history_frames],
        "depth_video": depth[:, :history_frames],
        "proprio_history": proprio[:, :history_frames],
        "goal_body": goal[:, history_frames - 1],
    })
    current_latent, current_state = initial, state[:, 0]
    predicted_latent, predicted_position, predicted_velocity, collision_logits = [], [], [], []
    for step in range(rollout_steps):
        prediction = model.predict_step(
            current_latent, current_state, actions[:, step], dt=dt[:, step]
        )
        current_latent, current_state = prediction["latent"], prediction["state"]
        predicted_latent.append(current_latent)
        predicted_position.append(prediction["state_prediction"]["position"])
        predicted_velocity.append(prediction["state_prediction"]["velocity"])
        collision_logits.append(prediction["collision_logits"])
    predicted_latent_tensor = torch.stack(predicted_latent, dim=1)
    windows_rgb = torch.stack(
        [rgb[:, step + 1 : step + 1 + history_frames] for step in range(rollout_steps)], dim=1
    )
    windows_depth = torch.stack(
        [depth[:, step + 1 : step + 1 + history_frames] for step in range(rollout_steps)], dim=1
    )
    windows_proprio = torch.stack(
        [proprio[:, step + 1 : step + 1 + history_frames] for step in range(rollout_steps)], dim=1
    )
    batch_size = rgb.shape[0]
    with torch.no_grad():
        target_latent = model.encode({
            "rgb_video": windows_rgb.flatten(0, 1),
            "depth_video": windows_depth.flatten(0, 1),
            "proprio_history": windows_proprio.flatten(0, 1),
            "goal_body": goal[:, history_frames : history_frames + rollout_steps].reshape(batch_size * rollout_steps, 3),
        }).reshape(batch_size, rollout_steps, -1)
    latent_loss = (
        2.0 - 2.0 * nn.functional.cosine_similarity(predicted_latent_tensor, target_latent, dim=-1)
    ).mean()
    position_loss = nn.functional.smooth_l1_loss(
        torch.stack(predicted_position, dim=1), batch["target_position"].to(rgb.device)[:, :rollout_steps]
    )
    velocity_loss = nn.functional.smooth_l1_loss(
        torch.stack(predicted_velocity, dim=1), batch["target_velocity"].to(rgb.device)[:, :rollout_steps]
    )
    collision_loss = nn.functional.binary_cross_entropy_with_logits(
        torch.stack(collision_logits, dim=1), batch["collision"].to(rgb.device)[:, :rollout_steps]
    )
    total = (
        weights.latent * latent_loss + weights.position * position_loss
        + weights.velocity * velocity_loss + weights.collision * collision_loss
    )
    return total, {
        "latent": latent_loss, "position": position_loss,
        "velocity": velocity_loss, "collision": collision_loss,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_vjepa_wam_checkpoint(
    path: str | Path,
    model: VJEPAWorldModelAdapter,
    *,
    config: Mapping[str, Any],
    official_checkpoint: str | Path,
    training_steps: int,
    encoder_model_name: str,
) -> Path:
    if int(training_steps) <= 0:
        raise ValueError("refusing to save an untrained V-JEPA WAM checkpoint")
    output = Path(path).expanduser().resolve()
    official = Path(official_checkpoint).expanduser().resolve()
    if not official.is_file():
        raise FileNotFoundError("official V-JEPA checkpoint is required")
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("visual_encoder.")
    }
    payload = {
        "schema": "urbanfly-vjepa-wam-v1", "family": "vjepa_wam",
        "training_steps": int(training_steps), "config": dict(config), "model": state,
        "official_checkpoint": str(official), "official_checkpoint_sha256": _sha256(official),
        "official_model": encoder_model_name, "parameter_count": model.parameter_count,
        "trainable_parameter_count": model.trainable_parameter_count,
        "pixel_reconstruction": False, "action_representation": model.action_representation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output


def load_vjepa_wam_checkpoint(
    model: VJEPAWorldModelAdapter,
    path: str | Path,
    *,
    official_checkpoint: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    official = Path(official_checkpoint).expanduser().resolve()
    if not official.is_file():
        raise FileNotFoundError("official V-JEPA checkpoint is required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("family") != "vjepa_wam" or int(payload.get("training_steps", 0)) <= 0:
        raise ValueError("checkpoint is not a trained V-JEPA UAV WAM")
    if _sha256(official) != payload.get("official_checkpoint_sha256"):
        raise ValueError("official V-JEPA checkpoint hash does not match WAM provenance")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    encoder_missing = [name for name in missing if name.startswith("visual_encoder.")]
    non_encoder_missing = [name for name in missing if not name.startswith("visual_encoder.")]
    if non_encoder_missing or unexpected:
        raise ValueError(f"V-JEPA WAM checkpoint mismatch: missing={non_encoder_missing}, unexpected={unexpected}")
    payload["expected_encoder_state_keys"] = len(encoder_missing)
    return payload
