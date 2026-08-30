from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RouteProjection:
    """Immutable route state used by planners, monitors and visualizers."""

    progress_m: float
    nearest_progress_m: float
    cross_track_error_m: float
    segment_index: int
    remaining_m: float
    lookahead_m: float
    local_goal_nwu: np.ndarray
    distance_to_turn_m: float | None


class PolylineRoute:
    """Monotonic polyline tracking with speed-adaptive local goals.

    The route is a geometric reference, not a second local planner.  It only
    supplies the local goal that forms the final three entries of YOPO's
    nine-dimensional observation.
    """

    def __init__(
        self,
        waypoints_nwu: np.ndarray,
        normal_lookahead_m: float = 12.0,
        turn_lookahead_m: float = 12.0,
        turn_threshold_degrees: float = 30.0,
        turn_awareness_distance_m: float = 20.0,
        lookahead_speed_gain_s: float = 1.5,
        maximum_lookahead_m: float = 22.0,
    ) -> None:
        self.waypoints = np.asarray(waypoints_nwu, dtype=np.float64)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3 or len(self.waypoints) < 2:
            raise ValueError("route requires at least two finite [3] waypoints")
        if not np.isfinite(self.waypoints).all(): raise ValueError("route contains non-finite waypoints")
        segment = np.diff(self.waypoints, axis=0); length = np.linalg.norm(segment, axis=1)
        if (length < 1e-4).any(): raise ValueError("route contains duplicate adjacent waypoints")
        self.segment = segment; self.length = length
        self.cumulative = np.r_[0.0, np.cumsum(length)]
        self.progress_m = 0.0
        self.normal_lookahead_m = float(normal_lookahead_m)
        self.turn_lookahead_m = float(turn_lookahead_m)
        self.turn_threshold = np.deg2rad(turn_threshold_degrees)
        self.turn_awareness_distance_m = float(turn_awareness_distance_m)
        self.lookahead_speed_gain_s = float(lookahead_speed_gain_s)
        self.maximum_lookahead_m = float(maximum_lookahead_m)
        self.route_id = hashlib.sha256(self.waypoints.astype(np.float32).tobytes()).hexdigest()[:16]

    def project_progress(self, position_nwu: np.ndarray, minimum_progress: float | None = None) -> float:
        position = np.asarray(position_nwu, dtype=np.float64)
        floor = self.progress_m if minimum_progress is None else float(minimum_progress)
        best_progress, best_distance = floor, float("inf")
        for index, (start, delta, length) in enumerate(zip(self.waypoints[:-1], self.segment, self.length)):
            segment_start, segment_end = self.cumulative[index], self.cumulative[index + 1]
            if segment_end < floor - 12.0 or segment_start > floor + 35.0:
                continue
            alpha = np.clip(np.dot(position - start, delta) / (length * length), 0.0, 1.0)
            projected = start + alpha * delta; distance = float(np.linalg.norm(position - projected))
            progress = float(self.cumulative[index] + alpha * length)
            if distance < best_distance:
                best_progress, best_distance = progress, distance
        return max(floor, best_progress)

    def project_nearest(
        self, position_nwu: np.ndarray, reference_progress: float | None = None,
        backward_window_m: float = 12.0, forward_window_m: float = 35.0,
    ) -> tuple[float, float]:
        """Return unconstrained arc length and cross-track distance near the current route section."""
        position = np.asarray(position_nwu, dtype=np.float64)
        reference = self.progress_m if reference_progress is None else float(reference_progress)
        best_progress, best_distance = reference, float("inf")
        for index, (start, delta, length) in enumerate(zip(self.waypoints[:-1], self.segment, self.length)):
            segment_start, segment_end = self.cumulative[index], self.cumulative[index + 1]
            if segment_end < reference - backward_window_m or segment_start > reference + forward_window_m:
                continue
            alpha = np.clip(np.dot(position - start, delta) / (length * length), 0.0, 1.0)
            projected = start + alpha * delta
            distance = float(np.linalg.norm(position - projected))
            progress = float(segment_start + alpha * length)
            if distance < best_distance:
                best_progress, best_distance = progress, distance
        return best_progress, best_distance

    def update(self, position_nwu: np.ndarray) -> float:
        self.progress_m = self.project_progress(position_nwu)
        return self.progress_m

    def _interpolate(self, progress: float) -> np.ndarray:
        progress = float(np.clip(progress, 0.0, self.cumulative[-1]))
        index = min(int(np.searchsorted(self.cumulative, progress, side="right") - 1), len(self.segment) - 1)
        alpha = (progress - self.cumulative[index]) / self.length[index]
        return self.waypoints[index] + alpha * self.segment[index]

    def _segment_index(self, progress_m: float) -> int:
        return min(
            max(int(np.searchsorted(self.cumulative, progress_m, side="right") - 1), 0),
            len(self.segment) - 1,
        )

    def _next_significant_turn(self, progress_m: float) -> tuple[float, int] | None:
        current = self._segment_index(progress_m)
        first = self.segment[current] / self.length[current]
        for index in range(current + 1, len(self.segment)):
            distance = max(float(self.cumulative[index] - progress_m), 0.0)
            if distance > self.turn_awareness_distance_m:
                break
            second = self.segment[index] / self.length[index]
            angle = float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))
            if angle >= self.turn_threshold:
                return distance, index
        return None

    def lookahead_for_speed(self, speed_mps: float, progress_m: float | None = None) -> tuple[float, float | None]:
        progress = self.progress_m if progress_m is None else float(progress_m)
        lookahead = float(np.clip(
            self.normal_lookahead_m + self.lookahead_speed_gain_s * max(float(speed_mps), 0.0),
            self.normal_lookahead_m,
            self.maximum_lookahead_m,
        ))
        next_turn = self._next_significant_turn(progress)
        distance_to_turn = None if next_turn is None else next_turn[0]
        if distance_to_turn is not None and distance_to_turn <= self.turn_awareness_distance_m:
            lookahead = self.turn_lookahead_m
        return lookahead, distance_to_turn

    def local_goal(self, speed_mps: float = 0.0) -> np.ndarray:
        lookahead, _ = self.lookahead_for_speed(speed_mps)
        return self._interpolate(self.progress_m + lookahead).astype(np.float32)

    def observe(self, position_nwu: np.ndarray, speed_mps: float = 0.0) -> RouteProjection:
        """Update monotonic progress and return a complete route observation."""
        self.update(position_nwu)
        nearest_progress, cross_track = self.project_nearest(position_nwu)
        lookahead, distance_to_turn = self.lookahead_for_speed(speed_mps)
        return RouteProjection(
            progress_m=float(self.progress_m),
            nearest_progress_m=float(nearest_progress),
            cross_track_error_m=float(cross_track),
            segment_index=self._segment_index(nearest_progress),
            remaining_m=max(float(self.cumulative[-1] - self.progress_m), 0.0),
            lookahead_m=float(lookahead),
            local_goal_nwu=self._interpolate(self.progress_m + lookahead).astype(np.float32),
            distance_to_turn_m=distance_to_turn,
        )

    @property
    def completion(self) -> float:
        return float(self.progress_m / self.cumulative[-1])

    @property
    def total_length_m(self) -> float:
        return float(self.cumulative[-1])

    def reached(self, position_nwu: np.ndarray, tolerance_m: float = 1.0) -> bool:
        return bool(np.linalg.norm(np.asarray(position_nwu) - self.waypoints[-1]) <= tolerance_m)


# Backward-compatible public name used by the dataset collector and older
# baseline runner.  New realtime code uses the explicit PolylineRoute name.
RouteManager = PolylineRoute
