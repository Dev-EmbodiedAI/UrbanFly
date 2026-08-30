from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CpaResult:
    time_to_cpa_s: float
    distance_at_cpa_m: float
    risk: float


def cpa_risk(
    relative_position_m: np.ndarray,
    relative_velocity_mps: np.ndarray,
    *,
    safe_radius_m: float = 4.0,
    horizon_s: float = 3.0,
    alpha: float = 0.35,
    beta: float = 2.0,
) -> CpaResult:
    """ARR-Fly-style closest-point-of-approach risk.

    Inputs are in the ego/body frame and may be two- or three-dimensional.
    Risk is zero when the predicted CPA is outside the safety radius or when
    the actor is moving away beyond the finite prediction horizon.
    """

    position = np.asarray(relative_position_m, dtype=np.float64)
    velocity = np.asarray(relative_velocity_mps, dtype=np.float64)
    if position.shape not in {(2,), (3,)} or velocity.shape != position.shape:
        raise ValueError("relative position and velocity must share shape (2,) or (3,)")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("CPA inputs must be finite")
    speed_squared = float(velocity @ velocity)
    unconstrained = (
        -float(position @ velocity) / speed_squared if speed_squared > 1e-9 else 0.0
    )
    time_to_cpa = float(np.clip(unconstrained, 0.0, horizon_s))
    distance = float(np.linalg.norm(position + velocity * time_to_cpa))
    spatial = max(0.0, 1.0 - distance / max(float(safe_radius_m), 1e-6))
    risk = float(np.exp(-alpha * time_to_cpa) * spatial**beta)
    return CpaResult(time_to_cpa, distance, float(np.clip(risk, 0.0, 1.0)))


def cpa_risk_map(
    relative_positions_body_flu_m: np.ndarray,
    relative_velocities_body_flu_mps: np.ndarray,
    *,
    azimuth_bins: int = 34,
    safe_radius_m: float = 4.0,
    horizon_s: float = 3.0,
    lse_temperature: float = 0.12,
) -> np.ndarray:
    """Compress privileged actor CPA risks into a one-dimensional azimuth map."""

    positions = np.asarray(relative_positions_body_flu_m, dtype=np.float64)
    velocities = np.asarray(relative_velocities_body_flu_mps, dtype=np.float64)
    if positions.size == 0:
        return np.zeros(azimuth_bins, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or velocities.shape != positions.shape:
        raise ValueError("actor positions and velocities must have shape [actors, 3]")
    per_bin: list[list[float]] = [[] for _ in range(azimuth_bins)]
    for position, velocity in zip(positions, velocities):
        # FLU azimuth: forward=0, left=+pi/2.
        azimuth = float(np.arctan2(position[1], position[0]))
        index = min(
            azimuth_bins - 1,
            int(np.floor((azimuth + np.pi) / (2.0 * np.pi) * azimuth_bins)),
        )
        per_bin[index].append(
            cpa_risk(
                position,
                velocity,
                safe_radius_m=safe_radius_m,
                horizon_s=horizon_s,
            ).risk
        )
    result = np.zeros(azimuth_bins, dtype=np.float32)
    temperature = max(float(lse_temperature), 1e-6)
    for index, values in enumerate(per_bin):
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        maximum = float(array.max())
        pooled = maximum + temperature * np.log(
            np.exp((array - maximum) / temperature).sum()
        )
        result[index] = np.clip(pooled, 0.0, 1.0)
    return result


def depth_to_spherical_range_map(
    depth_m: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    elevation_bins: int = 6,
    azimuth_bins: int = 34,
    max_range_m: float = 120.0,
) -> np.ndarray:
    """Min-pool a perspective depth image to ARR-Fly's 34x6 range layout."""

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = (
        np.isfinite(depth) & (depth > 0.0)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    )
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("depth and valid mask must share shape [height, width]")
    rows = np.array_split(np.arange(depth.shape[0]), elevation_bins)
    columns = np.array_split(np.arange(depth.shape[1]), azimuth_bins)
    result = np.full((elevation_bins, azimuth_bins), max_range_m, dtype=np.float32)
    for row_index, row_ids in enumerate(rows):
        for column_index, column_ids in enumerate(columns):
            block = depth[np.ix_(row_ids, column_ids)]
            block_valid = valid[np.ix_(row_ids, column_ids)]
            if block_valid.any():
                result[row_index, column_index] = min(
                    float(block[block_valid].min()), float(max_range_m)
                )
    return result


class DepthHistory:
    """Fixed 15-frame causal buffer used by the ARR-Fly baseline."""

    def __init__(self, frames: int = 15, elevation_bins: int = 6, azimuth_bins: int = 34):
        self.frames = int(frames)
        self.elevation_bins = int(elevation_bins)
        self.azimuth_bins = int(azimuth_bins)
        self._items: deque[np.ndarray] = deque(maxlen=self.frames)

    def reset(self) -> None:
        self._items.clear()

    def append(self, depth_m: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
        self._items.append(
            depth_to_spherical_range_map(
                depth_m,
                valid_mask,
                elevation_bins=self.elevation_bins,
                azimuth_bins=self.azimuth_bins,
            )
        )

    def array(self) -> np.ndarray:
        if not self._items:
            raise RuntimeError("depth history is empty")
        padding = [self._items[0]] * (self.frames - len(self._items))
        return np.stack([*padding, *self._items]).astype(np.float32)
