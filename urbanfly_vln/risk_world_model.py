from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np

from .schema import Episode


def instruction_embedding(text: str, dimensions: int = 16) -> np.ndarray:
    """Deterministic hashing baseline; replace with a frozen VLM/text encoder in the full model."""
    if dimensions < 0:
        raise ValueError("embedding dimensions must be non-negative")
    output = np.zeros(dimensions, dtype=np.float64)
    if dimensions == 0:
        return output
    tokens: list[str] = []
    for chunk in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.extend(chunk[index : index + 2] for index in range(max(len(chunk) - 1, 1)))
        else:
            tokens.append(chunk)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        output[value % dimensions] += 1.0 if (value >> 8) % 2 else -1.0
    norm = np.linalg.norm(output)
    return output / norm if norm > 0 else output


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class ModelMetrics:
    transition_rmse: float
    risk_accuracy: float
    risk_brier: float
    examples: int


class LinearRiskWorldModel:
    """Small executable baseline for the proposed action-conditioned risk world model.

    It predicts next navigation state and near-future risk. It is intentionally simple:
    the research model should replace it with a recurrent latent dynamics network and
    calibrated ensemble uncertainty while preserving this input/output contract.
    """

    def __init__(self, language_dimensions: int = 16, ridge: float = 1e-3) -> None:
        self.language_dimensions = language_dimensions
        self.ridge = ridge
        self.transition_weights: np.ndarray | None = None
        self.risk_weights: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None

    def _examples(self, episodes: list[Episode], risk_horizon: int, risk_depth_m: float):
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        risks: list[float] = []
        for episode in episodes:
            language = instruction_embedding(episode.instruction, self.language_dimensions)
            for index, (step, next_step) in enumerate(zip(episode.steps[:-1], episode.steps[1:])):
                state = np.array(step.velocity + [step.goal_distance_m, step.p05_depth_m], dtype=np.float64)
                action = np.array(step.action_delta, dtype=np.float64)
                features.append(np.concatenate([state, action, language, [1.0]]))
                targets.append(
                    np.array(next_step.velocity + [next_step.goal_distance_m, next_step.p05_depth_m], dtype=np.float64)
                )
                future = episode.steps[index + 1 : index + 1 + risk_horizon]
                risks.append(float(any(item.collision or item.p05_depth_m < risk_depth_m for item in future)))
        if not features:
            raise ValueError("at least one episode with two steps is required")
        return np.vstack(features), np.vstack(targets), np.asarray(risks, dtype=np.float64)

    def fit(self, episodes: list[Episode], risk_horizon: int = 5, risk_depth_m: float = 2.0) -> ModelMetrics:
        x_raw, y, risk = self._examples(episodes, risk_horizon, risk_depth_m)
        self.feature_mean = x_raw.mean(axis=0)
        self.feature_scale = x_raw.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-6] = 1.0
        # Preserve the explicit bias column.
        self.feature_mean[-1] = 0.0
        self.feature_scale[-1] = 1.0
        x = (x_raw - self.feature_mean) / self.feature_scale
        identity = np.eye(x.shape[1], dtype=np.float64)
        identity[-1, -1] = 0.0
        self.transition_weights = np.linalg.solve(x.T @ x + self.ridge * identity, x.T @ y)

        weights = np.zeros(x.shape[1], dtype=np.float64)
        positive_weight = float((len(risk) - risk.sum()) / max(risk.sum(), 1.0))
        for _ in range(800):
            probability = _sigmoid(x @ weights)
            sample_weight = np.where(risk > 0.5, positive_weight, 1.0)
            gradient = x.T @ ((probability - risk) * sample_weight) / len(risk) + self.ridge * weights
            weights -= 0.08 * gradient
        self.risk_weights = weights
        return self.evaluate(episodes, risk_horizon, risk_depth_m)

    def evaluate(self, episodes: list[Episode], risk_horizon: int = 5, risk_depth_m: float = 2.0) -> ModelMetrics:
        if self.transition_weights is None or self.risk_weights is None:
            raise RuntimeError("fit must be called before evaluate")
        x_raw, y, risk = self._examples(episodes, risk_horizon, risk_depth_m)
        x = (x_raw - self.feature_mean) / self.feature_scale
        prediction = x @ self.transition_weights
        probability = _sigmoid(x @ self.risk_weights)
        return ModelMetrics(
            transition_rmse=float(math.sqrt(np.mean((prediction - y) ** 2))),
            risk_accuracy=float(np.mean((probability >= 0.5) == (risk >= 0.5))),
            risk_brier=float(np.mean((probability - risk) ** 2)),
            examples=len(risk),
        )
