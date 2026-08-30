from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from uav_wm_navigation.types import BodyVelocityAction, WorldModelObservation
from uav_wm_navigation.world_models.continuous_protocol import ContinuousWorldModelPolicy, episode_id_from_spec, validate_action_sequences


class RGBDEncoder(nn.Module):
    def __init__(self, output_dim: int = 192) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.Mish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.Mish(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.Mish(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.Mish(),
            nn.AdaptiveAvgPool2d((4, 7)), nn.Flatten(),
            nn.Linear(128 * 4 * 7, output_dim), nn.LayerNorm(output_dim), nn.Mish(),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or depth.ndim != 4 or valid.ndim != 4:
            raise ValueError("RGB-D tensors must have shape [B,C,H,W]")
        return self.network(torch.cat([rgb, depth, valid], dim=1))


class TDMPC2VisualNetwork(nn.Module):
    """Task-oriented pixel TD-MPC2 model for v3 RGB-D sequences."""

    def __init__(self, proprio_dim: int = 16, latent_dim: int = 192, action_dim: int = 4) -> None:
        super().__init__()
        self.proprio_dim, self.latent_dim, self.action_dim = int(proprio_dim), int(latent_dim), int(action_dim)
        self.visual = RGBDEncoder(latent_dim)
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 96), nn.LayerNorm(96), nn.Mish())
        self.encoder = nn.Sequential(nn.Linear(latent_dim + 96, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 384), nn.LayerNorm(384), nn.Mish(), nn.Linear(384, latent_dim)
        )
        joined = latent_dim + action_dim
        def head(output: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Linear(joined, 192), nn.Mish(), nn.Linear(192, output))
        self.reward, self.q1, self.q2, self.risk = head(), head(), head(), head()
        self.clearance, self.progress, self.continuation = head(), head(), head()
        self.uncertainty, self.state_delta = head(), head(3)
        self.policy = nn.Sequential(nn.Linear(latent_dim, 192), nn.Mish(), nn.Linear(192, action_dim), nn.Tanh())

    def encode(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([self.visual(rgb, depth, valid), self.proprio(proprio)], dim=-1))

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent + 0.25 * self.dynamics(torch.cat([latent, action], dim=-1)))

    def predict_heads(self, latent: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        joined = torch.cat([latent, action], dim=-1)
        return {
            "reward": self.reward(joined).squeeze(-1),
            "q": torch.minimum(self.q1(joined), self.q2(joined)).squeeze(-1),
            "risk": torch.sigmoid(self.risk(joined).squeeze(-1)),
            "clearance": 120.0 * torch.sigmoid(self.clearance(joined).squeeze(-1)),
            "progress": self.progress(joined).squeeze(-1),
            "continuation": torch.sigmoid(self.continuation(joined).squeeze(-1)),
            "uncertainty": torch.sigmoid(self.uncertainty(joined).squeeze(-1)),
            "state_delta": self.state_delta(joined),
        }


def observation_visual_tensors(observation: WorldModelObservation, image_size: tuple[int, int] = (128, 224)) -> tuple[torch.Tensor, ...]:
    height, width = image_size
    rgb = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    depth = cv2.resize(observation.depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(observation.depth_valid_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    depth = np.where(valid, np.clip(depth / 120.0, 0.0, 1.0), 0.0).astype(np.float32)
    proprio = np.concatenate([
        np.clip(observation.goal_body_flu_m / 120.0, -1.0, 1.0),
        observation.linear_velocity_body_flu_mps / 6.0,
        observation.angular_velocity_body_flu_rps / 2.0,
        observation.gravity_body_flu,
        observation.previous_action,
    ]).astype(np.float32)
    return (
        torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0),
        torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(proprio).unsqueeze(0),
    )


class TDMPC2VisualPolicy(ContinuousWorldModelPolicy):
    def __init__(
        self, *, checkpoint: str | Path | None = None, device: str | torch.device | None = None,
        horizon: int = 15, candidates: int = 512, elites: int = 64, iterations: int = 5,
        discount: float = 0.97, risk_weight: float = 8.0, allow_untrained_for_tests: bool = False,
        seed: int = 20260731, image_size: tuple[int, int] = (128, 224),
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = TDMPC2VisualNetwork().to(self.device)
        self.horizon, self.candidates, self.elites, self.iterations = map(int, (horizon, candidates, elites, iterations))
        self.discount, self.risk_weight = float(discount), float(risk_weight)
        self.allow_untrained_for_tests, self.image_size = bool(allow_untrained_for_tests), tuple(image_size)
        self.rng = torch.Generator(device=self.device).manual_seed(int(seed))
        self._latent: torch.Tensor | None = None
        self._episode_id, self._checkpoint = "", None
        self._diagnostics: dict[str, Any] = {"family": "tdmpc2_visual", "trained": False, "status": "checkpoint_required"}
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        self.model.eval()

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema") != "urbanfly-world-model-v3" or payload.get("family") != "tdmpc2_visual" or int(payload.get("training_steps", 0)) <= 0:
            raise ValueError("checkpoint lacks trained v3 TD-MPC2 visual provenance")
        self.model.load_state_dict(payload["model"])
        self._checkpoint = str(path)
        self._diagnostics.update({"trained": True, "status": "ready", "checkpoint": str(path)})

    def reset(self, episode_id) -> None:
        self._episode_id, self._latent = episode_id_from_spec(episode_id), None

    def observe(self, observation: WorldModelObservation) -> None:
        if self._episode_id and observation.episode_id != self._episode_id:
            raise ValueError("observation belongs to a different episode")
        self._episode_id = observation.episode_id
        tensors = [tensor.to(self.device) for tensor in observation_visual_tensors(observation, self.image_size)]
        with torch.inference_mode():
            self._latent = self.model.encode(*tensors)

    def _require_ready(self) -> torch.Tensor:
        if self._latent is None:
            raise RuntimeError("observe must be called before prediction")
        if self._checkpoint is None and not self.allow_untrained_for_tests:
            raise RuntimeError("trained v3 checkpoint required; random policy is disabled")
        return self._latent

    def _rollout(self, action_sequences: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, horizon, _ = action_sequences.shape
        latent = self._require_ready().expand(batch, -1)
        names = ("reward", "q", "risk", "clearance", "progress", "continuation", "uncertainty", "latent_norm")
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        predicted_position = action_sequences.new_zeros(batch, 3)
        positions: list[torch.Tensor] = []
        for step in range(horizon):
            action = action_sequences[:, step]
            heads = self.model.predict_heads(latent, action)
            for name in names:
                outputs[name].append(torch.linalg.vector_norm(latent, dim=-1) if name == "latent_norm" else heads[name])
            predicted_position = predicted_position + heads["state_delta"]
            positions.append(predicted_position)
            latent = self.model.transition(latent, action)
        stacked = {name: torch.stack(values, dim=1) for name, values in outputs.items()}
        indices = [min(max(seconds * 5 - 1, 0), horizon - 1) for seconds in (1, 2, 3)]
        stacked["predicted_state_1s_2s_3s"] = torch.stack([positions[index] for index in indices], dim=1)
        return stacked

    def predict(self, action_sequences: np.ndarray) -> dict[str, np.ndarray]:
        actions = torch.from_numpy(validate_action_sequences(action_sequences)).to(self.device)
        with torch.inference_mode():
            output = self._rollout(actions)
        return {name: value.cpu().numpy() for name, value in output.items()}

    def act(self, deterministic: bool = True) -> BodyVelocityAction:
        latent, started = self._require_ready(), time.perf_counter()
        with torch.inference_mode():
            mean = self.model.policy(latent)[0].repeat(self.horizon, 1)
            std = torch.full_like(mean, 0.65)
            best_score, best_actions = torch.tensor(float("-inf"), device=self.device), mean.unsqueeze(0)
            discounts = self.discount ** torch.arange(self.horizon, device=self.device)
            for _ in range(self.iterations):
                noise = torch.randn((self.candidates, self.horizon, 4), generator=self.rng, device=self.device)
                actions = torch.clamp(mean[None] + std[None] * noise, -1.0, 1.0)
                prediction = self._rollout(actions)
                score = ((prediction["reward"] - self.risk_weight * prediction["risk"]) * discounts).sum(1) + discounts[-1] * prediction["q"][:, -1]
                indices = torch.topk(score, min(self.elites, self.candidates)).indices
                elite, elite_score = actions[indices], score[indices]
                weights = torch.softmax((elite_score - elite_score.max()) / 0.5, dim=0)
                mean = torch.einsum("n,nha->ha", weights, elite)
                variance = torch.einsum("n,nha->ha", weights, (elite - mean) ** 2)
                std = torch.sqrt(variance + 1e-4).clamp(0.05, 0.8)
                best_actions, best_score = elite[:1], elite_score[0]
            selected = mean[0] if deterministic else best_actions[0, 0]
            prediction = self._rollout(mean[None])
        self._diagnostics = {
            "family": "tdmpc2_visual", "trained": self._checkpoint is not None,
            "status": "ready" if self._checkpoint else "untrained_test_mode", "checkpoint": self._checkpoint,
            "horizon_steps": self.horizon, "candidate_count": self.candidates,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "predicted_return": float(best_score), "maximum_predicted_risk": float(prediction["risk"].max()),
            "maximum_uncertainty": float(prediction["uncertainty"].max()),
        }
        return BodyVelocityAction(selected.cpu().numpy().astype(np.float32))

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)
