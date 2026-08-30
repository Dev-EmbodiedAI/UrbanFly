"""Small, inspectable world-model MPC for RGB-D corridor flight.

The model deliberately stays lightweight: it predicts the multirotor's short
horizon response to body-frame velocity commands and scores those imagined
trajectories against a local obstacle cloud reconstructed from the live depth
camera.  It is therefore a model-based controller, not a DreamerV3 claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DepthCameraModel:
    width: int
    height: int
    horizontal_fov_deg: float = 90.0

    @property
    def fx(self) -> float:
        return 0.5 * self.width / np.tan(np.deg2rad(self.horizontal_fov_deg) * 0.5)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return (self.width - 1) * 0.5

    @property
    def cy(self) -> float:
        return (self.height - 1) * 0.5


@dataclass(frozen=True)
class WorldModelConfig:
    horizon_s: float = 3.0
    rollout_dt_s: float = 0.20
    velocity_response_tau_s: float = 0.35
    vehicle_radius_m: float = 0.48
    collision_margin_m: float = 0.32
    comfort_clearance_m: float = 1.05
    max_depth_m: float = 18.0
    min_depth_m: float = 0.20
    vertical_keep_m: float = 1.15
    point_stride: int = 4
    progress_weight: float = 5.0
    center_weight: float = 0.65
    clearance_weight: float = 18.0
    collision_penalty: float = 600.0
    smoothness_weight: float = 0.55

    @property
    def collision_clearance_m(self) -> float:
        return self.vehicle_radius_m + self.collision_margin_m


@dataclass(frozen=True)
class CandidateEvaluation:
    command_body_mps: np.ndarray
    score: float
    min_clearance_m: float
    predicted_collision: bool
    trajectory_body_m: np.ndarray


@dataclass(frozen=True)
class ObstacleMemoryConfig:
    voxel_size_m: float = 0.22
    keep_behind_m: float = 3.0
    keep_ahead_m: float = 20.0
    keep_lateral_m: float = 8.0
    occlusion_depth_m: float = 1.5
    occlusion_step_m: float = 0.5


class LocalObstacleMemory:
    """Persistent 2-D occupancy memory built only from past depth observations.

    A depth camera observes the first surface but cannot see the object's rear.
    Points are conservatively extruded a short distance along their viewing ray,
    which prevents the vehicle from forgetting an obstacle as the camera passes
    its leading edge.
    """

    def __init__(self, config: ObstacleMemoryConfig | None = None) -> None:
        self.config = config or ObstacleMemoryConfig()
        self._points_world_xy = np.empty((0, 2), dtype=np.float32)

    @property
    def point_count(self) -> int:
        return int(self._points_world_xy.shape[0])

    def reset(self) -> None:
        self._points_world_xy = np.empty((0, 2), dtype=np.float32)

    def update(
        self,
        points_body_m: np.ndarray,
        position_world_m: np.ndarray,
        yaw_rad: float,
    ) -> None:
        points = np.asarray(points_body_m, dtype=np.float32)
        position = np.asarray(position_world_m, dtype=np.float32)
        cosine, sine = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
        rotation_body_to_world = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=np.float32
        )
        new_world_chunks: list[np.ndarray] = []
        if points.size:
            max_steps = max(
                0,
                int(np.floor(self.config.occlusion_depth_m / self.config.occlusion_step_m)),
            )
            ray_xy = points[:, :2]
            ray_norm = np.linalg.norm(ray_xy, axis=1, keepdims=True)
            ray_unit = ray_xy / np.maximum(ray_norm, 1e-6)
            for step in range(max_steps + 1):
                extended_body = ray_xy + ray_unit * (step * self.config.occlusion_step_m)
                new_world_chunks.append(
                    extended_body @ rotation_body_to_world.T + position[None, :2]
                )

        chunks = [self._points_world_xy]
        chunks.extend(new_world_chunks)
        combined = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=np.float32)
        if combined.size == 0:
            self._points_world_xy = combined
            return

        relative_world = combined - position[None, :2]
        relative_body = relative_world @ rotation_body_to_world
        keep = (
            (relative_body[:, 0] >= -self.config.keep_behind_m)
            & (relative_body[:, 0] <= self.config.keep_ahead_m)
            & (np.abs(relative_body[:, 1]) <= self.config.keep_lateral_m)
        )
        combined = combined[keep]
        voxel_keys = np.floor(combined / self.config.voxel_size_m).astype(np.int32)
        _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
        self._points_world_xy = combined[np.sort(unique_indices)].astype(np.float32)

    def points_body(self, position_world_m: np.ndarray, yaw_rad: float) -> np.ndarray:
        if not self._points_world_xy.size:
            return np.empty((0, 3), dtype=np.float32)
        position = np.asarray(position_world_m, dtype=np.float32)
        cosine, sine = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
        rotation_body_to_world = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=np.float32
        )
        relative_world = self._points_world_xy - position[None, :2]
        relative_body = relative_world @ rotation_body_to_world
        return np.column_stack(
            (relative_body, np.zeros(relative_body.shape[0], dtype=np.float32))
        ).astype(np.float32)


def depth_to_body_points(
    depth_planar_m: np.ndarray,
    camera: DepthCameraModel,
    config: WorldModelConfig,
    camera_translation_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    camera_pitch_deg: float = 0.0,
) -> np.ndarray:
    """Project planar-depth pixels into the camera/body NED frame.

    The forward camera uses x-forward, y-right, z-down. The configured
    camera has only a small pitch/translation offset, so the local MPC treats
    camera and body frames as coincident over its short horizon.
    """

    depth = np.asarray(depth_planar_m, dtype=np.float32)
    if depth.shape != (camera.height, camera.width):
        raise ValueError(
            f"Depth shape {depth.shape} does not match camera "
            f"({camera.height}, {camera.width})."
        )

    rows = np.arange(0, camera.height, config.point_stride, dtype=np.int32)
    cols = np.arange(0, camera.width, config.point_stride, dtype=np.int32)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    sampled = depth[vv, uu]
    valid = (
        np.isfinite(sampled)
        & (sampled >= config.min_depth_m)
        & (sampled <= config.max_depth_m)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    x = sampled[valid]
    y = (uu[valid].astype(np.float32) - camera.cx) * x / camera.fx
    z = (vv[valid].astype(np.float32) - camera.cy) * x / camera.fy
    points = np.column_stack((x, y, z)).astype(np.float32)
    pitch = np.deg2rad(float(camera_pitch_deg))
    cosine, sine = float(np.cos(pitch)), float(np.sin(pitch))
    rotation_camera_to_body = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )
    points = points @ rotation_camera_to_body.T
    points += np.asarray(camera_translation_body_m, dtype=np.float32)[None, :]
    return points[np.abs(points[:, 2]) <= config.vertical_keep_m]


def robust_front_clearance(depth_planar_m: np.ndarray, max_depth_m: float) -> float:
    """Return a noise-resistant near-obstacle distance in the central view."""

    depth = np.asarray(depth_planar_m, dtype=np.float32)
    height, width = depth.shape
    crop = depth[
        int(height * 0.35) : max(int(height * 0.70), int(height * 0.35) + 1),
        int(width * 0.30) : max(int(width * 0.70), int(width * 0.30) + 1),
    ]
    valid = crop[np.isfinite(crop) & (crop > 0.05) & (crop <= max_depth_m)]
    return float(np.percentile(valid, 5.0)) if valid.size else float(max_depth_m)


class DepthWorldModelMPC:
    """Imagine short trajectories and choose the safest useful command."""

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        self.config = config or WorldModelConfig()

    def rollout(
        self,
        command_body_mps: np.ndarray,
        measured_velocity_body_mps: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        command = np.asarray(command_body_mps, dtype=np.float32)[:2]
        velocity = np.asarray(measured_velocity_body_mps, dtype=np.float32)[:2].copy()
        position = np.zeros(2, dtype=np.float32)
        positions: list[np.ndarray] = []
        alpha = float(np.clip(cfg.rollout_dt_s / cfg.velocity_response_tau_s, 0.0, 1.0))
        steps = max(1, int(np.ceil(cfg.horizon_s / cfg.rollout_dt_s)))
        for _ in range(steps):
            velocity += alpha * (command - velocity)
            position += velocity * cfg.rollout_dt_s
            positions.append(position.copy())
        return np.asarray(positions, dtype=np.float32)

    def evaluate(
        self,
        command_body_mps: np.ndarray,
        measured_velocity_body_mps: np.ndarray,
        obstacle_points_body_m: np.ndarray,
        current_corridor_y_m: float,
        previous_command_body_mps: np.ndarray,
    ) -> CandidateEvaluation:
        cfg = self.config
        trajectory = self.rollout(command_body_mps, measured_velocity_body_mps)
        points = np.asarray(obstacle_points_body_m, dtype=np.float32)

        if points.size:
            deltas = trajectory[:, None, :2] - points[None, :, :2]
            clearance_by_step = np.sqrt(np.sum(deltas * deltas, axis=2)).min(axis=1)
        else:
            clearance_by_step = np.full(trajectory.shape[0], cfg.max_depth_m, dtype=np.float32)

        min_clearance = float(np.min(clearance_by_step))
        predicted_collision = bool(np.any(clearance_by_step < cfg.collision_clearance_m))
        comfort_violation = np.maximum(cfg.comfort_clearance_m - clearance_by_step, 0.0)
        clearance_cost = float(np.mean(comfort_violation * comfort_violation))
        absolute_y = current_corridor_y_m + trajectory[:, 1]
        center_cost = float(np.mean(np.abs(absolute_y)))
        progress = float(trajectory[-1, 0])
        smoothness = float(
            np.linalg.norm(
                np.asarray(command_body_mps, dtype=np.float32)[:2]
                - np.asarray(previous_command_body_mps, dtype=np.float32)[:2]
            )
        )
        score = (
            cfg.progress_weight * progress
            - cfg.center_weight * center_cost
            - cfg.clearance_weight * clearance_cost
            - cfg.smoothness_weight * smoothness
            - (cfg.collision_penalty if predicted_collision else 0.0)
        )
        return CandidateEvaluation(
            command_body_mps=np.asarray(command_body_mps, dtype=np.float32)[:2],
            score=float(score),
            min_clearance_m=min_clearance,
            predicted_collision=predicted_collision,
            trajectory_body_m=trajectory,
        )

    def choose_action(
        self,
        obstacle_points_body_m: np.ndarray,
        measured_velocity_body_mps: np.ndarray,
        current_corridor_y_m: float,
        previous_command_body_mps: np.ndarray | None = None,
        candidates: Iterable[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        previous = (
            np.zeros(2, dtype=np.float32)
            if previous_command_body_mps is None
            else np.asarray(previous_command_body_mps, dtype=np.float32)[:2]
        )
        if candidates is None:
            candidates = (
                (forward, lateral)
                for forward in (2.2, 1.5, 0.8)
                for lateral in (-1.25, -0.85, -0.45, 0.0, 0.45, 0.85, 1.25)
            )
            candidates = list(candidates) + [(0.0, -0.8), (0.0, 0.0), (0.0, 0.8), (-0.35, 0.0)]

        evaluations = [
            self.evaluate(
                np.asarray(candidate, dtype=np.float32),
                measured_velocity_body_mps,
                obstacle_points_body_m,
                current_corridor_y_m,
                previous,
            )
            for candidate in candidates
        ]
        if not evaluations:
            raise ValueError("At least one candidate action is required.")
        evaluations.sort(key=lambda item: item.score, reverse=True)
        best = evaluations[0]
        safe_count = sum(not item.predicted_collision for item in evaluations)
        diagnostics: dict[str, object] = {
            "score": best.score,
            "min_predicted_clearance_m": best.min_clearance_m,
            "predicted_collision": best.predicted_collision,
            "safe_candidate_count": safe_count,
            "candidate_count": len(evaluations),
            "trajectory_body_m": best.trajectory_body_m.tolist(),
            "top_candidates": [
                {
                    "command_body_mps": item.command_body_mps.tolist(),
                    "score": item.score,
                    "min_clearance_m": item.min_clearance_m,
                    "predicted_collision": item.predicted_collision,
                }
                for item in evaluations[:5]
            ],
        }
        return best.command_body_mps.copy(), diagnostics
