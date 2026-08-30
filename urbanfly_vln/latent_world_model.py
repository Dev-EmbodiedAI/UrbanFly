from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .risk_world_model import instruction_embedding


STATE_DIM = 6
ACTION_DIM = 4
OUTPUT_DIM = 7


class DynamicsMLP(nn.Module):
    """Action-conditioned latent dynamics and risk predictor."""

    def __init__(
        self,
        hidden_dim: int = 128,
        input_dim: int = STATE_DIM + ACTION_DIM,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dropout = float(dropout)
        self.layer_norm = bool(layer_norm)
        if not layer_norm and self.dropout <= 0.0:
            # Preserve the original parameter names so v1/v2 checkpoints remain loadable.
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, OUTPUT_DIM),
            )
        else:
            first_norm: nn.Module = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
            second_norm: nn.Module = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                first_norm,
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim, hidden_dim),
                second_norm,
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim, OUTPUT_DIM),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass
class CandidatePrediction:
    progress_m: float
    risk_probability: float
    uncertainty: float
    score: float
    delta_position: np.ndarray
    safe: bool = True


@dataclass
class CandidateSelection:
    selected_index: int
    predictions: list[CandidatePrediction]
    used_fallback: bool
    reason: str


class LatentWorldModelEnsemble:
    def __init__(
        self,
        models: list[DynamicsMLP],
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        y_mean: np.ndarray,
        y_scale: np.ndarray,
        device: torch.device,
        language_dimensions: int = 0,
        risk_temperature: float = 1.0,
        checkpoint_format: str = "urbanfly-latent-world-model-v1",
    ) -> None:
        self.models = models
        self.x_mean = x_mean.astype(np.float32)
        self.x_scale = x_scale.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_scale = y_scale.astype(np.float32)
        self.device = device
        self.language_dimensions = int(language_dimensions)
        self.risk_temperature = max(float(risk_temperature), 1e-3)
        self.checkpoint_format = checkpoint_format
        for model in self.models:
            model.to(device).eval()

    @classmethod
    def load(cls, checkpoint: Path, device: torch.device | None = None) -> "LatentWorldModelEnsemble":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        models: list[DynamicsMLP] = []
        input_dim = int(payload.get("input_dim", len(payload["x_mean"])))
        language_dimensions = int(payload.get("language_dimensions", max(0, input_dim - STATE_DIM - ACTION_DIM)))
        architecture = payload.get("architecture", {})
        dropout = float(architecture.get("dropout", 0.0))
        layer_norm = bool(architecture.get("layer_norm", False))
        for state_dict in payload["model_state_dicts"]:
            model = DynamicsMLP(
                hidden_dim=int(payload["hidden_dim"]),
                input_dim=input_dim,
                dropout=dropout,
                layer_norm=layer_norm,
            )
            model.load_state_dict(state_dict)
            models.append(model)
        return cls(
            models=models,
            x_mean=np.asarray(payload["x_mean"], dtype=np.float32),
            x_scale=np.asarray(payload["x_scale"], dtype=np.float32),
            y_mean=np.asarray(payload["y_mean"], dtype=np.float32),
            y_scale=np.asarray(payload["y_scale"], dtype=np.float32),
            device=device,
            language_dimensions=language_dimensions,
            risk_temperature=float(payload.get("risk_temperature", 1.0)),
            checkpoint_format=str(payload.get("format", "urbanfly-latent-world-model-v1")),
        )

    def _predict_members(
        self, state: np.ndarray, action: np.ndarray, instruction: str = ""
    ) -> tuple[np.ndarray, np.ndarray]:
        parts = [state, action]
        if self.language_dimensions:
            parts.append(instruction_embedding(instruction, self.language_dimensions).astype(np.float32))
        features = np.concatenate(parts).astype(np.float32)
        if features.shape != self.x_mean.shape:
            raise ValueError(f"checkpoint expects {len(self.x_mean)} features, got {len(features)}")
        normalized = (features - self.x_mean) / self.x_scale
        tensor = torch.from_numpy(normalized[None, :]).to(self.device)
        continuous: list[np.ndarray] = []
        risks: list[float] = []
        with torch.inference_mode():
            for model in self.models:
                output = model(tensor)[0].detach().cpu().numpy()
                continuous.append(output[:6] * self.y_scale + self.y_mean)
                risks.append(float(torch.sigmoid(torch.tensor(output[6] / self.risk_temperature)).item()))
        return np.stack(continuous), np.asarray(risks, dtype=np.float32)

    def predict_members(
        self, state: np.ndarray, action: np.ndarray, instruction: str = ""
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-member physical predictions and calibrated risk probabilities."""
        return self._predict_members(state, action, instruction)

    def predict_feature_matrix(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict an offline feature matrix for evaluation.

        Returns ensemble-mean continuous predictions, calibrated risk probabilities,
        and an epistemic uncertainty proxy based on member disagreement.
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != len(self.x_mean):
            raise ValueError(f"expected an N x {len(self.x_mean)} feature matrix")
        normalized = (features - self.x_mean) / self.x_scale
        tensor = torch.from_numpy(normalized).to(self.device)
        continuous: list[np.ndarray] = []
        risks: list[np.ndarray] = []
        with torch.inference_mode():
            for model in self.models:
                output = model(tensor).detach().cpu().numpy()
                continuous.append(output[:, :6] * self.y_scale + self.y_mean)
                risks.append(1.0 / (1.0 + np.exp(-np.clip(output[:, 6] / self.risk_temperature, -30.0, 30.0))))
        member_continuous = np.stack(continuous)
        member_risks = np.stack(risks)
        uncertainty = np.std(member_continuous[:, :, 5], axis=0) + 0.25 * np.linalg.norm(
            np.std(member_continuous[:, :, :3], axis=0), axis=1
        )
        return member_continuous.mean(axis=0), member_risks.mean(axis=0), uncertainty

    def rollout_candidate(
        self,
        state: np.ndarray,
        action_vector: np.ndarray,
        horizon: int = 3,
        risk_weight: float = 12.0,
        uncertainty_weight: float = 2.0,
        instruction: str = "",
        max_risk_probability: float = 1.0,
        max_uncertainty: float = float("inf"),
    ) -> CandidatePrediction:
        current_state = state.astype(np.float32).copy()
        remaining = action_vector.astype(np.float32).copy()
        total_progress = 0.0
        survival = 1.0
        uncertainty_sum = 0.0
        delta_sum = np.zeros(3, dtype=np.float32)

        for _ in range(max(horizon, 1)):
            distance = float(np.linalg.norm(remaining))
            if distance < 1e-3:
                break
            action = np.concatenate([remaining, [distance]]).astype(np.float32)
            members, risks = self._predict_members(current_state, action, instruction)
            mean = members.mean(axis=0)
            delta = mean[:3]
            projected_progress = max(0.0, float(mean[5]))
            step_uncertainty = float(np.std(members[:, 5]) + 0.25 * np.linalg.norm(np.std(members[:, :3], axis=0)))
            step_risk = float(np.mean(risks))

            total_progress += projected_progress
            survival *= 1.0 - np.clip(step_risk, 0.0, 1.0)
            uncertainty_sum += step_uncertainty
            delta_sum += delta

            direction = remaining / max(distance, 1e-6)
            remaining -= direction * min(projected_progress, distance)
            current_state[0] = max(0.0, float(mean[3]))
            current_state[1] = float(delta[2])
            current_state[2] = max(0.0, float(mean[4]))
            current_state[3] = max(0.0, distance - projected_progress)
            current_state[4] = max(0.0, current_state[4] - 0.5 * projected_progress)

        risk_probability = 1.0 - survival
        score = total_progress - risk_weight * risk_probability - uncertainty_weight * uncertainty_sum
        return CandidatePrediction(
            progress_m=total_progress,
            risk_probability=risk_probability,
            uncertainty=uncertainty_sum,
            score=score,
            delta_position=delta_sum,
            safe=risk_probability <= max_risk_probability and uncertainty_sum <= max_uncertainty,
        )

    def rank_candidates(
        self,
        state: np.ndarray,
        action_vectors: Iterable[np.ndarray],
        horizon: int = 3,
        risk_weight: float = 12.0,
        uncertainty_weight: float = 2.0,
        instruction: str = "",
        max_risk_probability: float = 1.0,
        max_uncertainty: float = float("inf"),
    ) -> list[CandidatePrediction]:
        actions = list(action_vectors)
        states = np.asarray(state)
        if states.ndim == 1:
            candidate_states = [states] * len(actions)
        elif states.ndim == 2 and len(states) == len(actions):
            candidate_states = list(states)
        else:
            raise ValueError("state must be one state vector or one state vector per candidate")
        return [
            self.rollout_candidate(
                candidate_state,
                action,
                horizon,
                risk_weight,
                uncertainty_weight,
                instruction,
                max_risk_probability,
                max_uncertainty,
            )
            for candidate_state, action in zip(candidate_states, actions)
        ]

    def select_candidate(
        self,
        state: np.ndarray,
        action_vectors: Iterable[np.ndarray],
        *,
        instruction: str = "",
        horizon: int = 3,
        risk_weight: float = 12.0,
        uncertainty_weight: float = 2.0,
        max_risk_probability: float = 0.65,
        max_uncertainty: float = float("inf"),
    ) -> CandidateSelection:
        predictions = self.rank_candidates(
            state,
            action_vectors,
            horizon,
            risk_weight,
            uncertainty_weight,
            instruction,
            max_risk_probability,
            max_uncertainty,
        )
        if not predictions:
            raise ValueError("at least one candidate action is required")
        safe_indices = [index for index, prediction in enumerate(predictions) if prediction.safe]
        if safe_indices:
            selected = max(safe_indices, key=lambda index: predictions[index].score)
            return CandidateSelection(selected, predictions, False, "highest_safe_score")
        selected = min(
            range(len(predictions)),
            key=lambda index: (predictions[index].risk_probability, predictions[index].uncertainty),
        )
        return CandidateSelection(selected, predictions, True, "no_candidate_passed_safety_gate")
