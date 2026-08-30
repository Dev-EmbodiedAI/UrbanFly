from __future__ import annotations

import numpy as np

from uav_wm_navigation.types import CandidateTrajectory

from .route_manager import RouteManager


def rank_route_consistent_candidates(
    candidates: list[CandidateTrajectory], route: RouteManager, base_scores: np.ndarray,
    config: dict, previous_trajectory_id: str | None = None,
) -> tuple[int, list[int], list[float], list[dict[str, float]], str]:
    """Apply a hard monotonic route gate and a soft smooth-progress preference."""
    if not candidates:
        return -1, [], [], [], "no_candidates"
    base = np.asarray(base_scores, dtype=np.float64)
    if base.shape != (len(candidates),):
        raise ValueError("base_scores must match candidates")
    finite = np.isfinite(base)
    if finite.any():
        low, high = float(base[finite].min()), float(base[finite].max())
        normalized_base = (np.nan_to_num(base, nan=high, posinf=high, neginf=low) - low) / max(high - low, 1e-8)
    else:
        normalized_base = np.ones(len(candidates), dtype=np.float64)

    minimum_endpoint = float(config.get("minimum_endpoint_progress_m", 0.5))
    maximum_regression = float(config.get("maximum_regression_m", 0.25))
    maximum_lateral = float(config.get("maximum_lateral_m", 2.5))
    minimum_efficiency = float(config.get("minimum_progress_efficiency", 0.30))
    metrics: list[dict[str, float]] = []
    eligible = np.zeros(len(candidates), dtype=bool)
    endpoint_progress = np.zeros(len(candidates), dtype=np.float64)
    minimum_delta = np.zeros(len(candidates), dtype=np.float64)
    lateral = np.zeros(len(candidates), dtype=np.float64)
    efficiency = np.zeros(len(candidates), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        projections = [route.project_nearest(point, route.progress_m) for point in candidate.positions]
        projected = np.asarray([item[0] for item in projections])
        cross_track = np.asarray([item[1] for item in projections])
        endpoint_progress[index] = projected[-1] - route.progress_m
        minimum_delta[index] = float(projected.min() - route.progress_m)
        lateral[index] = float(cross_track.max())
        path_length = float(np.linalg.norm(np.diff(candidate.positions, axis=0), axis=1).sum())
        efficiency[index] = endpoint_progress[index] / max(path_length, 1e-6)
        eligible[index] = bool(
            endpoint_progress[index] >= minimum_endpoint
            and minimum_delta[index] >= -maximum_regression
            and lateral[index] <= maximum_lateral
            and efficiency[index] >= minimum_efficiency
        )
        metrics.append({
            "endpoint_progress_m": float(endpoint_progress[index]),
            "minimum_progress_delta_m": float(minimum_delta[index]),
            "maximum_lateral_m": float(lateral[index]),
            "progress_efficiency": float(efficiency[index]),
            "eligible": float(eligible[index]),
        })

    positive_progress = np.clip(endpoint_progress, 0.0, None)
    progress_normalized = positive_progress / max(float(positive_progress.max()), 1e-6)
    adjusted = (
        normalized_base
        - float(config.get("progress_reward", 0.8)) * progress_normalized
        + float(config.get("lateral_penalty", 0.35)) * np.clip(lateral / max(maximum_lateral, 1e-6), 0.0, 2.0)
        + float(config.get("inefficiency_penalty", 0.45)) * np.clip(1.0 - efficiency, 0.0, 2.0)
    )
    if previous_trajectory_id is not None:
        for index, candidate in enumerate(candidates):
            if candidate.trajectory_id == previous_trajectory_id:
                adjusted[index] -= float(config.get("hysteresis_bonus", 0.08))
    adjusted[~eligible] = np.inf
    reason = "route_consistent"
    if not eligible.any():
        relaxed = (
            (endpoint_progress > 0.0)
            & (minimum_delta >= -2.0 * maximum_regression)
            & (lateral <= 1.25 * maximum_lateral)
        )
        adjusted[relaxed] = normalized_base[relaxed] - 0.5 * progress_normalized[relaxed]
        reason = "route_gate_relaxed" if relaxed.any() else "route_gate_hover"
    ranking = [int(index) for index in np.argsort(adjusted) if np.isfinite(adjusted[index])]
    return (-1 if not ranking else ranking[0]), ranking, adjusted.tolist(), metrics, reason
