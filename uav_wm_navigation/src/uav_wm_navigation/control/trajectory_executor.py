from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from uav_wm_navigation.control.safety_filter import SafetyFilter
from uav_wm_navigation.simulators.base import SimulatorAdapter
from uav_wm_navigation.types import ActionLimits, CandidateTrajectory, SensorFrame


class TrajectoryExecutor:
    def __init__(
        self, simulator: SimulatorAdapter, safety_filter: SafetyFilter, control_dt: float = 0.1,
        position_kp: float = 1.2, yaw_kp: float | None = None,
        yaw_deadband_degrees: float = 0.0, yaw_rate_smoothing_alpha: float = 1.0,
        velocity_smoothing_alpha: float = 1.0, route_lateral_velocity_scale: float = 1.0,
        action_noise_std: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        action_noise_bound: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        action_limits: ActionLimits | None = None,
        seed: int = 0,
    ) -> None:
        self.simulator = simulator
        self.safety_filter = safety_filter
        self.control_dt = float(control_dt)
        self.position_kp = float(position_kp)
        self.yaw_kp = float(1.0 / max(self.control_dt, 1e-6) if yaw_kp is None else yaw_kp)
        self.yaw_deadband = float(np.deg2rad(yaw_deadband_degrees))
        self.yaw_rate_smoothing_alpha = float(np.clip(yaw_rate_smoothing_alpha, 0.0, 1.0))
        self.velocity_smoothing_alpha = float(np.clip(velocity_smoothing_alpha, 0.0, 1.0))
        self.route_lateral_velocity_scale = float(np.clip(route_lateral_velocity_scale, 0.0, 1.0))
        self.action_noise_std = np.asarray(action_noise_std, dtype=np.float64)
        self.action_noise_bound = np.asarray(action_noise_bound, dtype=np.float64)
        if self.action_noise_std.shape != (4,) or self.action_noise_bound.shape != (4,):
            raise ValueError("action noise std/bound must have shape [4]")
        if (self.action_noise_std < 0).any() or (self.action_noise_bound < 0).any():
            raise ValueError("action noise std/bound must be non-negative")
        self.action_limits = action_limits or ActionLimits()
        self.rng = np.random.default_rng(int(seed))
        self._smoothed_velocity: np.ndarray | None = None
        self._smoothed_yaw_rate: float | None = None

    def reset(self) -> None:
        """Reset command-filter state at an episode boundary."""
        self._smoothed_velocity = None
        self._smoothed_yaw_rate = None

    def execute_prefix(
        self, trajectory: CandidateTrajectory, duration: float, predicted_risk: float = 0.0,
        sensor: SensorFrame | None = None, heading_target_nwu: np.ndarray | None = None,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        max_steps = max(1, int(np.ceil(float(duration) / self.control_dt)))
        valid_indices = np.flatnonzero(trajectory.valid_mask)
        if valid_indices.size > 1:
            valid_indices = valid_indices[1:]
        for index in valid_indices[:max_steps]:
            state = self.simulator.get_kinematics()
            cycle_sensor = sensor if sensor is not None else self.simulator.get_depth()
            collision = bool(self.simulator.get_collision_info().get("has_collided", False))
            position_error = trajectory.positions[index] - state.position
            desired_velocity = trajectory.velocities[index] + self.position_kp * position_error
            current_yaw = float(Rotation.from_quat(state.orientation_xyzw).as_euler("ZYX")[0])
            heading_vector = desired_velocity
            if heading_target_nwu is not None:
                heading_vector = np.asarray(heading_target_nwu, dtype=np.float64) - state.position
                route_direction = heading_vector[:2]
                route_norm = float(np.linalg.norm(route_direction))
                if route_norm > 1e-6 and self.route_lateral_velocity_scale < 1.0:
                    forward = route_direction / route_norm
                    left = np.array([-forward[1], forward[0]], dtype=np.float64)
                    horizontal = desired_velocity[:2]
                    desired_velocity = np.asarray(desired_velocity, dtype=np.float64).copy()
                    desired_velocity[:2] = (
                        float(horizontal @ forward) * forward
                        + self.route_lateral_velocity_scale * float(horizontal @ left) * left
                    )
            if self._smoothed_velocity is None:
                self._smoothed_velocity = np.asarray(desired_velocity, dtype=np.float64).copy()
            else:
                alpha = self.velocity_smoothing_alpha
                self._smoothed_velocity = alpha * desired_velocity + (1.0 - alpha) * self._smoothed_velocity
            desired_velocity = self._smoothed_velocity.copy()
            desired_yaw = float(np.arctan2(heading_vector[1], heading_vector[0]))
            yaw_error = (desired_yaw - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
            raw_yaw_rate = 0.0 if abs(yaw_error) <= self.yaw_deadband else self.yaw_kp * yaw_error
            cosine, sine = np.cos(current_yaw), np.sin(current_yaw)
            expert_body = np.asarray([
                cosine * desired_velocity[0] + sine * desired_velocity[1],
                -sine * desired_velocity[0] + cosine * desired_velocity[1],
                desired_velocity[2], raw_yaw_rate,
            ], dtype=np.float64)
            noise = self.rng.normal(0.0, self.action_noise_std)
            noise = np.clip(noise, -self.action_noise_bound, self.action_noise_bound)
            perturbed_body = expert_body + noise
            desired_velocity = np.asarray([
                cosine * perturbed_body[0] - sine * perturbed_body[1],
                sine * perturbed_body[0] + cosine * perturbed_body[1],
                perturbed_body[2],
            ], dtype=np.float64)
            raw_yaw_rate = float(perturbed_body[3])
            if self._smoothed_yaw_rate is None:
                self._smoothed_yaw_rate = raw_yaw_rate
            else:
                alpha = self.yaw_rate_smoothing_alpha
                self._smoothed_yaw_rate = alpha * raw_yaw_rate + (1.0 - alpha) * self._smoothed_yaw_rate
            yaw_rate = float(self._smoothed_yaw_rate)
            result = self.safety_filter.apply(
                desired_velocity, yaw_rate, state, cycle_sensor, self.control_dt, predicted_risk, collision
            )
            self.simulator.execute_velocity_command(result.velocity, result.yaw_rate, self.control_dt)
            executed_body = np.asarray([
                cosine * result.velocity[0] + sine * result.velocity[1],
                -sine * result.velocity[0] + cosine * result.velocity[1],
                result.velocity[2], result.yaw_rate,
            ], dtype=np.float64)
            normalized = np.clip(executed_body / self.action_limits.vector, -1.0, 1.0)
            records.append({
                "index": int(index), "mode": result.mode, "reasons": result.reasons,
                "desired_velocity_nwu": np.asarray(desired_velocity).tolist(),
                "position_error_nwu": np.asarray(position_error).tolist(),
                "command_velocity_nwu": np.asarray(result.velocity).tolist(),
                "current_yaw_rad": current_yaw,
                "desired_yaw_rad": desired_yaw,
                "yaw_error_rad": yaw_error,
                "raw_yaw_rate_rps": raw_yaw_rate,
                "command_yaw_rate_rps": float(result.yaw_rate),
                "expert_action_physical_body_flu": expert_body.tolist(),
                "perturbed_action_physical_body_flu": perturbed_body.tolist(),
                "executed_action_physical_body_flu": executed_body.tolist(),
                "executed_action_normalized": normalized.tolist(),
            })
            if result.mode == "hover":
                break
        return records
