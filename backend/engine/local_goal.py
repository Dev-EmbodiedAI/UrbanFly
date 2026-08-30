"""Global-route to local-goal interface for future local navigation policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .helsinki_frames import backend_delta_to_body_flu


@dataclass(frozen=True)
class LocalGoalSelection:
    current_position_world: np.ndarray
    current_velocity_world: np.ndarray
    yaw_degrees: float
    lookahead_distance_m: float
    local_goal_world: np.ndarray
    local_goal_body_flu: np.ndarray
    remaining_global_path: np.ndarray
    route_progress_m: float
    remaining_distance_m: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "current_position_world": self.current_position_world.tolist(),
            "current_velocity_world": self.current_velocity_world.tolist(),
            "yaw_degrees": self.yaw_degrees,
            "lookahead_distance_m": self.lookahead_distance_m,
            "local_goal_world": self.local_goal_world.tolist(),
            "local_goal_body_flu": self.local_goal_body_flu.tolist(),
            "remaining_global_path": self.remaining_global_path.tolist(),
            "route_progress_m": self.route_progress_m,
            "remaining_distance_m": self.remaining_distance_m,
        }


class LocalGoalSelector:
    """Select an arc-length lookahead target from a frozen global route.

    World coordinates are the Helsinki backend/renderer frame
    ``[east x, up y, south z]``; geographic north is negative Z.
    Body output is FLU ``[forward, left, up]`` relative to the current UAV yaw.
    """

    def __init__(self, lookahead_distance_m: float = 20.0):
        lookahead = float(lookahead_distance_m)
        if not 10.0 <= lookahead <= 30.0:
            raise ValueError("lookahead_distance_m must be within 10--30 m")
        self.lookahead_distance_m = lookahead

    @staticmethod
    def _validate_path(global_path: np.ndarray) -> np.ndarray:
        path = np.asarray(global_path, dtype=float)
        if path.ndim != 2 or path.shape[1:] != (3,) or len(path) < 2:
            raise ValueError("global_path must have shape (N, 3), N >= 2")
        if not np.isfinite(path).all():
            raise ValueError("global_path contains non-finite coordinates")
        return path

    @staticmethod
    def _point_at_distance(path: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
        target = float(np.clip(distance, 0.0, cumulative[-1]))
        segment = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment = min(max(segment, 0), len(path) - 2)
        length = cumulative[segment + 1] - cumulative[segment]
        alpha = 0.0 if length <= 1e-9 else (target - cumulative[segment]) / length
        return path[segment] + alpha * (path[segment + 1] - path[segment])

    @staticmethod
    def world_delta_to_body_flu(delta_world: np.ndarray, yaw_degrees: float) -> np.ndarray:
        return backend_delta_to_body_flu(delta_world, yaw_degrees)

    def select(
        self,
        current_position_world: np.ndarray,
        current_velocity_world: np.ndarray,
        yaw_degrees: float,
        global_path: np.ndarray,
    ) -> LocalGoalSelection:
        path = self._validate_path(global_path)
        position = np.asarray(current_position_world, dtype=float)
        velocity = np.asarray(current_velocity_world, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("current position and velocity must be 3-vectors")
        segments = path[1:] - path[:-1]
        lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.r_[0.0, np.cumsum(lengths)]

        best_distance = float("inf")
        best_segment = 0
        best_alpha = 0.0
        best_projection = path[0]
        for index, (start, segment, length) in enumerate(zip(path[:-1], segments, lengths)):
            if length <= 1e-9:
                alpha = 0.0
            else:
                alpha = float(np.clip(np.dot(position - start, segment) / length**2, 0.0, 1.0))
            projection = start + alpha * segment
            distance = float(np.linalg.norm(position - projection))
            if distance < best_distance:
                best_distance = distance
                best_segment = index
                best_alpha = alpha
                best_projection = projection

        progress = float(cumulative[best_segment] + best_alpha * lengths[best_segment])
        goal_distance = min(float(cumulative[-1]), progress + self.lookahead_distance_m)
        local_goal_world = self._point_at_distance(path, cumulative, goal_distance)
        local_goal_body = self.world_delta_to_body_flu(
            local_goal_world - position,
            yaw_degrees,
        )
        remaining = [best_projection]
        remaining.extend(path[best_segment + 1 :])
        if np.linalg.norm(np.asarray(remaining[-1]) - path[-1]) > 1e-9:
            remaining.append(path[-1])
        return LocalGoalSelection(
            current_position_world=position.copy(),
            current_velocity_world=velocity.copy(),
            yaw_degrees=float(yaw_degrees),
            lookahead_distance_m=self.lookahead_distance_m,
            local_goal_world=local_goal_world,
            local_goal_body_flu=local_goal_body,
            remaining_global_path=np.asarray(remaining, dtype=float),
            route_progress_m=progress,
            remaining_distance_m=float(cumulative[-1] - progress),
        )
