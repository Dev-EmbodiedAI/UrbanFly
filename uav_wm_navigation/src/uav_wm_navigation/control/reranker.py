from __future__ import annotations

import time

import numpy as np

from uav_wm_navigation.types import CandidatePrediction, CandidateTrajectory, RerankDecision, RiskPrediction


def _normalize(values: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if bounds is None:
        lower, upper = float(np.min(values)), float(np.max(values))
    else:
        lower, upper = map(float, bounds)
    return np.clip((values - lower) / max(upper - lower, 1e-8), 0.0, 1.0)


class RiskReranker:
    def __init__(self, weights: dict[str, float], normalization: dict[str, list[float]] | None = None) -> None:
        self.weights = weights
        self.normalization = normalization or {}

    def rank(
        self,
        candidates: list[CandidateTrajectory],
        predictions: list[RiskPrediction],
        latency_ms: float = 0.0,
        timeout_ms: float | None = None,
        max_risk: float = 0.75,
    ) -> RerankDecision:
        started = time.perf_counter()
        original = [int(index) for index in np.argsort([candidate.yopo_cost for candidate in candidates])]
        if len(candidates) != len(predictions) or not candidates:
            raise ValueError("candidates and predictions must be non-empty and have equal length")
        raw = np.array([[c.yopo_cost, p.collision_probability, p.goal_progress, p.minimum_clearance,
                         p.failure_probability, p.uncertainty] for c, p in zip(candidates, predictions)], dtype=np.float64)
        if not np.isfinite(raw).all() or (timeout_ms is not None and latency_ms > timeout_ms):
            return RerankDecision(original[0], original, original, [float(c.yopo_cost) for c in candidates], [],
                                  "model_invalid_or_timeout", True, latency_ms)
        normalized_yopo = _normalize(raw[:, 0], self.normalization.get("yopo_cost"))
        normalized_progress = _normalize(raw[:, 2], self.normalization.get("goal_progress"))
        normalized_clearance = _normalize(raw[:, 3], self.normalization.get("minimum_clearance"))
        normalized_uncertainty = _normalize(raw[:, 5], self.normalization.get("uncertainty"))
        score = (
            self.weights.get("yopo", 1.0) * normalized_yopo
            + self.weights.get("collision", 4.0) * raw[:, 1]
            - self.weights.get("progress", 1.0) * normalized_progress
            - self.weights.get("clearance", 1.0) * normalized_clearance
            + self.weights.get("failure", 2.0) * raw[:, 4]
            + self.weights.get("uncertainty", 1.0) * normalized_uncertainty
        )
        ranking = [int(index) for index in np.argsort(score)]
        all_risky = bool(np.all((raw[:, 1] >= max_risk) | (raw[:, 4] >= max_risk)))
        components = [{
            "yopo": float(normalized_yopo[i]), "collision": float(raw[i, 1]),
            "progress": float(normalized_progress[i]), "clearance": float(normalized_clearance[i]),
            "failure": float(raw[i, 4]), "uncertainty": float(normalized_uncertainty[i]),
        } for i in range(len(candidates))]
        elapsed = latency_ms + (time.perf_counter() - started) * 1000.0
        return RerankDecision(-1 if all_risky else ranking[0], original, ranking, score.tolist(), components,
                              "all_candidates_high_risk" if all_risky else "world_model_score", all_risky, elapsed)


class CandidateRerankerV3:
    """Shared preregistered scorer for TD-MPC2, RSSM and V-JEPA assistants."""

    DEFAULT_WEIGHTS = {
        "yopo": 1.0, "collision": 4.0, "cpa": 2.0, "failure": 2.0,
        "uncertainty": 1.0, "progress": 1.0, "clearance": 1.0, "value": 0.5,
    }

    def __init__(self, weights: dict[str, float] | None = None, normalization: dict[str, list[float]] | None = None) -> None:
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.normalization = normalization or {}

    def rank(
        self,
        candidates: list[CandidateTrajectory],
        predictions: list[CandidatePrediction],
        *,
        latency_ms: float = 0.0,
        timeout_ms: float = 150.0,
        max_risk: float = 0.75,
    ) -> RerankDecision:
        started = time.perf_counter()
        if not candidates or len(candidates) != len(predictions):
            raise ValueError("candidates and predictions must be non-empty and aligned")
        original = [int(index) for index in np.argsort([candidate.yopo_cost for candidate in candidates])]
        raw = np.asarray([[
            candidate.yopo_cost, prediction.collision_probability, prediction.cpa_risk,
            prediction.failure_probability, prediction.epistemic_uncertainty,
            prediction.goal_progress, prediction.minimum_clearance, prediction.terminal_value,
        ] for candidate, prediction in zip(candidates, predictions)], dtype=np.float64)
        if not np.isfinite(raw).all() or latency_ms > timeout_ms:
            return RerankDecision(original[0], original, original, raw[:, 0].tolist(), [], "model_invalid_or_timeout", True, latency_ms)
        names = ("yopo_cost", "collision", "cpa", "failure", "uncertainty", "goal_progress", "minimum_clearance", "terminal_value")
        normalized = {name: _normalize(raw[:, index], self.normalization.get(name)) for index, name in enumerate(names)}
        score = (
            self.weights["yopo"] * normalized["yopo_cost"]
            + self.weights["collision"] * raw[:, 1]
            + self.weights["cpa"] * raw[:, 2]
            + self.weights["failure"] * raw[:, 3]
            + self.weights["uncertainty"] * normalized["uncertainty"]
            - self.weights["progress"] * normalized["goal_progress"]
            - self.weights["clearance"] * normalized["minimum_clearance"]
            - self.weights["value"] * normalized["terminal_value"]
        )
        ranking = [int(index) for index in np.argsort(score)]
        all_risky = bool(np.all(np.maximum(raw[:, 1], np.maximum(raw[:, 2], raw[:, 3])) >= max_risk))
        components = [{name: float(normalized[name][index]) for name in names} for index in range(len(candidates))]
        elapsed = latency_ms + (time.perf_counter() - started) * 1000.0
        return RerankDecision(
            -1 if all_risky else ranking[0], original, ranking, score.tolist(), components,
            "all_candidates_high_risk" if all_risky else "world_model_v3_score", all_risky, elapsed,
        )
