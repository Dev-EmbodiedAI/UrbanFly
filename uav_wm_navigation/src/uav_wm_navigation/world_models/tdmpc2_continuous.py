from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from uav_wm_navigation.types import BodyVelocityAction, WorldModelObservation
from uav_wm_navigation.world_models.continuous_protocol import (
    ContinuousWorldModelPolicy,
    validate_action_sequences,
    episode_id_from_spec,
)


def observation_features(observation: WorldModelObservation) -> np.ndarray:
    """Small deterministic RGB-D/state stem used by the compact validation model."""

    rgb = observation.rgb.astype(np.float32) / 255.0
    valid_depth = observation.depth_m[observation.depth_valid_mask]
    if valid_depth.size:
        depth_features = np.asarray(
            [
                valid_depth.mean() / 120.0,
                valid_depth.std() / 120.0,
                valid_depth.min() / 120.0,
                np.percentile(valid_depth, 10) / 120.0,
                np.percentile(valid_depth, 50) / 120.0,
            ],
            dtype=np.float32,
        )
    else:
        depth_features = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate(
        [
            rgb.mean(axis=(0, 1)),
            rgb.std(axis=(0, 1)),
            depth_features,
            np.clip(observation.goal_body_flu_m / 120.0, -1.0, 1.0),
            observation.linear_velocity_body_flu_mps / 6.0,
            observation.angular_velocity_body_flu_rps / 2.0,
            observation.gravity_body_flu,
            observation.previous_action,
        ]
    ).astype(np.float32)


class TDMPC2Network(nn.Module):
    """Compact task-oriented latent dynamics model following TD-MPC2's structure."""

    def __init__(self, feature_dim: int = 27, latent_dim: int = 96, action_dim: int = 4):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 192),
            nn.LayerNorm(192),
            nn.Mish(),
            nn.Linear(192, latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, latent_dim),
        )
        self.reward = nn.Sequential(nn.Linear(latent_dim + action_dim, 128), nn.Mish(), nn.Linear(128, 1))
        self.q1 = nn.Sequential(nn.Linear(latent_dim + action_dim, 128), nn.Mish(), nn.Linear(128, 1))
        self.q2 = nn.Sequential(nn.Linear(latent_dim + action_dim, 128), nn.Mish(), nn.Linear(128, 1))
        self.risk = nn.Sequential(nn.Linear(latent_dim + action_dim, 128), nn.Mish(), nn.Linear(128, 1))
        self.policy = nn.Sequential(nn.Linear(latent_dim, 128), nn.Mish(), nn.Linear(128, action_dim), nn.Tanh())

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(features))

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.dynamics(torch.cat([latent, action], dim=-1))
        return torch.tanh(latent + 0.25 * delta)

    def heads(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joined = torch.cat([latent, action], dim=-1)
        reward = self.reward(joined).squeeze(-1)
        value = torch.minimum(self.q1(joined), self.q2(joined)).squeeze(-1)
        risk = torch.sigmoid(self.risk(joined).squeeze(-1))
        return reward, value, risk


class TDMPC2ContinuousPolicy(ContinuousWorldModelPolicy):
    """Batched short-horizon CEM planner over a learned task latent model."""

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
        horizon: int = 15,
        candidates: int = 512,
        elites: int = 64,
        iterations: int = 5,
        discount: float = 0.97,
        risk_weight: float = 8.0,
        allow_untrained_for_tests: bool = False,
        seed: int = 20260731,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = TDMPC2Network().to(self.device)
        self.horizon = int(horizon)
        self.candidates = int(candidates)
        self.elites = int(elites)
        self.iterations = int(iterations)
        self.discount = float(discount)
        self.risk_weight = float(risk_weight)
        self.allow_untrained_for_tests = bool(allow_untrained_for_tests)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))
        self._observation: WorldModelObservation | None = None
        self._latent: torch.Tensor | None = None
        self._episode_id = ""
        self._loaded_checkpoint: str | None = None
        self._last_action = np.zeros(4, dtype=np.float32)
        self._last_diagnostics: dict[str, Any] = {
            "family": "tdmpc2_continuous",
            "trained": False,
            "status": "checkpoint_required",
        }
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        self.model.eval()

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if (
            not isinstance(payload, dict)
            or "model" not in payload
            or payload.get("schema") != "urbanfly-world-model-v2"
            or payload.get("family") != "tdmpc2_continuous"
            or int(payload.get("training_steps", 0)) <= 0
        ):
            raise ValueError(
                "checkpoint lacks trained UrbanFly TD-MPC2 provenance metadata"
            )
        state_dict = payload["model"]
        self.model.load_state_dict(state_dict)
        self._loaded_checkpoint = str(path)
        self._last_diagnostics.update(
            {"trained": True, "status": "ready", "checkpoint": str(path)}
        )

    def reset(self, episode_id) -> None:
        self._episode_id = episode_id_from_spec(episode_id)
        self._observation = None
        self._latent = None
        self._last_action.fill(0.0)

    def observe(self, observation: WorldModelObservation) -> None:
        if self._episode_id and observation.episode_id != self._episode_id:
            raise ValueError("observation belongs to a different episode")
        self._episode_id = observation.episode_id
        features = torch.from_numpy(observation_features(observation)).to(self.device)
        with torch.inference_mode():
            self._latent = self.model.encode(features.unsqueeze(0))
        self._observation = observation

    def _require_ready(self) -> torch.Tensor:
        if self._latent is None:
            raise RuntimeError("observe must be called before planning")
        if self._loaded_checkpoint is None and not self.allow_untrained_for_tests:
            raise RuntimeError(
                "TD-MPC2 checkpoint required; refusing to present random weights as a policy"
            )
        return self._latent

    def _rollout(self, action_sequences: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, horizon, _ = action_sequences.shape
        latent = self._require_ready().expand(batch, -1)
        rewards, risks, values, latent_norms = [], [], [], []
        for index in range(horizon):
            action = action_sequences[:, index]
            reward, value, risk = self.model.heads(latent, action)
            rewards.append(reward)
            risks.append(risk)
            values.append(value)
            latent = self.model.transition(latent, action)
            latent_norms.append(torch.linalg.vector_norm(latent, dim=-1))
        return {
            "reward": torch.stack(rewards, dim=1),
            "risk": torch.stack(risks, dim=1),
            "q": torch.stack(values, dim=1),
            "latent_norm": torch.stack(latent_norms, dim=1),
        }

    def predict(self, action_sequences: np.ndarray) -> dict[str, np.ndarray]:
        actions = validate_action_sequences(action_sequences)
        tensor = torch.from_numpy(actions).to(self.device)
        with torch.inference_mode():
            result = self._rollout(tensor)
        return {name: value.detach().cpu().numpy() for name, value in result.items()}

    def act(self, deterministic: bool = True) -> BodyVelocityAction:
        latent = self._require_ready()
        started = time.perf_counter()
        with torch.inference_mode():
            prior = self.model.policy(latent)[0]
            mean = prior.repeat(self.horizon, 1)
            mean[:-1] = 0.65 * mean[:-1] + 0.35 * mean[1:]
            std = torch.full_like(mean, 0.65)
            best_actions = mean.unsqueeze(0)
            best_score = torch.tensor(float("-inf"), device=self.device)
            discounts = self.discount ** torch.arange(
                self.horizon, device=self.device
            )
            for _ in range(self.iterations):
                noise = torch.randn(
                    (self.candidates, self.horizon, 4),
                    generator=self.rng,
                    device=self.device,
                )
                actions = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * noise, -1, 1)
                prediction = self._rollout(actions)
                score = (
                    (prediction["reward"] - self.risk_weight * prediction["risk"])
                    * discounts
                ).sum(dim=1) + discounts[-1] * prediction["q"][:, -1]
                elite_count = min(self.elites, self.candidates)
                elite_indices = torch.topk(score, elite_count).indices
                elite_actions = actions[elite_indices]
                elite_scores = score[elite_indices]
                weights = torch.softmax(
                    (elite_scores - elite_scores.max()) / 0.5, dim=0
                )
                mean = torch.einsum("n,nha->ha", weights, elite_actions)
                variance = torch.einsum(
                    "n,nha->ha", weights, (elite_actions - mean) ** 2
                )
                std = torch.sqrt(variance + 1e-4).clamp(0.05, 0.8)
                best_actions = elite_actions[:1]
                best_score = elite_scores[0]
            selected = mean[0] if deterministic else best_actions[0, 0]
            prediction = self._rollout(mean.unsqueeze(0))
        self._last_action = selected.cpu().numpy().astype(np.float32)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_diagnostics = {
            "family": "tdmpc2_continuous",
            "trained": self._loaded_checkpoint is not None,
            "status": (
                "ready"
                if self._loaded_checkpoint is not None
                else "untrained_test_mode"
            ),
            "checkpoint": self._loaded_checkpoint,
            "horizon_steps": self.horizon,
            "horizon_s": self.horizon / 5.0,
            "candidate_count": self.candidates,
            "iterations": self.iterations,
            "latency_ms": latency_ms,
            "predicted_return": float(best_score.cpu()),
            "maximum_predicted_risk": float(prediction["risk"].max().cpu()),
            "raw_action_normalized": self._last_action.tolist(),
        }
        return BodyVelocityAction(self._last_action)

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)
