from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_wm_navigation.types import SensorFrame, VehicleState


@dataclass(slots=True)
class SafetyResult:
    velocity: np.ndarray
    yaw_rate: float
    mode: str
    reasons: list[str]


class SafetyFilter:
    def __init__(self, config: dict[str, float | list[float]]) -> None:
        self.max_speed = float(config.get("max_speed_mps", 5.0))
        self.max_acceleration = float(config.get("max_acceleration_mps2", 4.0))
        self.max_yaw_rate = float(config.get("max_yaw_rate_rps", 1.2))
        self.emergency_depth = float(config.get("emergency_depth_m", 1.0))
        self.slow_depth = float(config.get("slow_depth_m", 3.0))
        self.clearance_percentile = float(config.get("clearance_percentile", 5.0))
        self.depth_roi_fraction = np.asarray(config.get("depth_roi_fraction", [0.0, 1.0, 0.0, 1.0]), dtype=np.float64)
        if self.depth_roi_fraction.shape != (4,):
            raise ValueError("depth_roi_fraction must be [top, bottom, left, right]")
        self.min_altitude = float(config.get("min_altitude_m", 0.5))
        self.max_altitude = float(config.get("max_altitude_m", 60.0))
        self.target_altitude = config.get("target_altitude_m")
        self.altitude_kp = float(config.get("altitude_kp", 1.5))
        self.bounds_min = np.asarray(config.get("bounds_min_nwu", [-1e6, -1e6, -1e6]), dtype=np.float64)
        self.bounds_max = np.asarray(config.get("bounds_max_nwu", [1e6, 1e6, 1e6]), dtype=np.float64)
        self.collision_debounce = int(config.get("collision_debounce_steps", 2))
        self.acceleration_reference = str(config.get("acceleration_reference", "measured"))
        if self.acceleration_reference not in {"measured", "commanded"}:
            raise ValueError("acceleration_reference must be 'measured' or 'commanded'")
        self._last_command_velocity: np.ndarray | None = None
        self._collision_hits = 0

    def apply(
        self,
        desired_velocity: np.ndarray,
        yaw_rate: float,
        state: VehicleState,
        sensor: SensorFrame,
        dt: float,
        predicted_risk: float = 0.0,
        collision: bool = False,
    ) -> SafetyResult:
        velocity = np.asarray(desired_velocity, dtype=np.float64).copy()
        reasons: list[str] = []
        if velocity.shape != (3,) or not np.isfinite(velocity).all() or not np.isfinite(yaw_rate):
            self._last_command_velocity = np.zeros(3, dtype=np.float64)
            return SafetyResult(np.zeros(3), 0.0, "hover", ["invalid_command"])
        self._collision_hits = self._collision_hits + 1 if collision else 0
        if self._collision_hits >= self.collision_debounce:
            self._last_command_velocity = np.zeros(3, dtype=np.float64)
            return SafetyResult(np.zeros(3), 0.0, "hover", ["collision_debounced"])
        height, width = sensor.depth_m.shape
        top, bottom, left, right = self.depth_roi_fraction
        y0, y1 = int(np.clip(top, 0.0, 1.0) * height), int(np.clip(bottom, 0.0, 1.0) * height)
        x0, x1 = int(np.clip(left, 0.0, 1.0) * width), int(np.clip(right, 0.0, 1.0) * width)
        y1, x1 = max(y1, y0 + 1), max(x1, x0 + 1)
        roi_depth = sensor.depth_m[y0:y1, x0:x1]
        roi_mask = sensor.valid_mask[y0:y1, x0:x1]
        valid_depth = roi_depth[roi_mask]
        if valid_depth.size == 0:
            self._last_command_velocity = np.zeros(3, dtype=np.float64)
            return SafetyResult(np.zeros(3), 0.0, "hover", ["missing_depth"])
        clearance = float(np.percentile(valid_depth, self.clearance_percentile))
        if clearance <= self.emergency_depth or predicted_risk >= 0.9:
            self._last_command_velocity = np.zeros(3, dtype=np.float64)
            return SafetyResult(np.zeros(3), 0.0, "hover", ["emergency_risk_or_clearance"])
        if clearance < self.slow_depth or predicted_risk >= 0.65:
            scale = min(
                (clearance - self.emergency_depth) / max(self.slow_depth - self.emergency_depth, 1e-6),
                max(0.0, 1.0 - predicted_risk),
            )
            velocity *= float(np.clip(scale, 0.0, 1.0))
            reasons.append("risk_slowdown")
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed:
            velocity *= self.max_speed / speed
            reasons.append("speed_clamped")
        if self.target_altitude is not None:
            altitude_error = float(self.target_altitude) - float(state.position[2])
            velocity[2] = float(np.clip(self.altitude_kp * altitude_error, -1.5, 1.5))
            if abs(altitude_error) > 0.15:
                reasons.append("altitude_hold")
        acceleration_base = state.linear_velocity
        if self.acceleration_reference == "commanded" and self._last_command_velocity is not None:
            acceleration_base = self._last_command_velocity
        delta = velocity - acceleration_base
        max_delta = self.max_acceleration * max(float(dt), 1e-6)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta:
            velocity = acceleration_base + delta * max_delta / delta_norm
            reasons.append("acceleration_clamped")
        next_position = state.position + velocity * dt
        if next_position[2] < self.min_altitude:
            recovery = np.clip((self.min_altitude - float(state.position[2])) / max(float(dt), 1e-6), 0.0, 1.5)
            velocity[2] = max(velocity[2], float(recovery))
            reasons.append("minimum_altitude")
        if next_position[2] > self.max_altitude:
            velocity[2] = min(velocity[2], 0.0)
            reasons.append("maximum_altitude")
        for axis in range(3):
            moving_farther_below = next_position[axis] < self.bounds_min[axis] and velocity[axis] <= 0.0
            moving_farther_above = next_position[axis] > self.bounds_max[axis] and velocity[axis] >= 0.0
            if moving_farther_below or moving_farther_above:
                velocity[axis] = 0.0
                reasons.append(f"boundary_axis_{axis}")
        clipped_yaw = float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))
        if clipped_yaw != yaw_rate:
            reasons.append("yaw_rate_clamped")
        self._last_command_velocity = velocity.copy()
        return SafetyResult(velocity.astype(np.float32), clipped_yaw, "safe" if not reasons else "filtered", reasons)
