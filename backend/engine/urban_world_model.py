"""Receding-horizon urban flight world model.

This module is deliberately explicit about what is and is not learned.  The
online belief is reconstructed from a sensor-limited view of the active static
collision surface.  Candidate controls are then imagined with cloned instances
of the same multirotor dynamics used by the simulator.  A learned RGB-D/RSSM
backend can replace the belief predictor later, but is not enabled across the
unvalidated source-simulator -> target-city domain gap.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, Optional

import numpy as np

from .multirotor_dynamics import (
    GRAVITY,
    MAX_PHYSICS_SUBSTEP,
    MultirotorDynamics,
)


@dataclass(frozen=True)
class UrbanWorldModelConfig:
    planning_interval_s: float = 0.50
    horizon_s: float = 3.0
    rollout_dt_s: float = 0.25
    sensor_range_m: float = 78.0
    sensor_horizontal_fov_deg: float = 110.0
    sensor_ray_count: int = 61
    belief_resolution_m: float = 2.0
    belief_max_age_s: float = 45.0
    belief_keep_radius_m: float = 110.0
    collision_clearance_m: float = 3.0
    comfort_clearance_m: float = 6.0
    lookahead_s: float = 1.0
    cruise_speeds_mps: tuple[float, ...] = (4.0, 7.0, 10.0)
    heading_offsets_deg: tuple[float, ...] = (
        -70.0,
        -40.0,
        -20.0,
        0.0,
        20.0,
        40.0,
        70.0,
    )
    climb_rates_mps: tuple[float, ...] = (-2.0, 0.0, 2.5)
    progress_weight: float = 7.0
    terminal_distance_weight: float = 0.10
    clearance_weight: float = 9.0
    unknown_weight: float = 4.0
    smoothness_weight: float = 0.45
    climb_weight: float = 0.16
    collision_penalty: float = 12000.0

    @classmethod
    def from_dict(cls, values: Optional[dict]) -> "UrbanWorldModelConfig":
        values = dict(values or {})
        allowed = {item.name for item in fields(cls)}
        cleaned = {key: value for key, value in values.items() if key in allowed}
        for key in ("cruise_speeds_mps", "heading_offsets_deg", "climb_rates_mps"):
            if key in cleaned:
                cleaned[key] = tuple(float(value) for value in cleaned[key])
        return cls(**cleaned)


@dataclass
class BeliefCell:
    surface_height_m: float
    observed_at_s: float


class LocalHeightBelief:
    """Persistent, sensor-limited height belief in the local ENU frame."""

    def __init__(self, collision_map, config: UrbanWorldModelConfig):
        self.collision_map = collision_map
        self.config = config
        self.cells: Dict[tuple[int, int], BeliefCell] = {}
        self.observation_count = 0

    def reset(self) -> None:
        self.cells.clear()
        self.observation_count = 0

    def _key(self, x: float, z: float) -> tuple[int, int]:
        resolution = self.config.belief_resolution_m
        return int(np.floor(x / resolution)), int(np.floor(z / resolution))

    def observe(
        self,
        position_world_m: np.ndarray,
        yaw_degrees: float,
        simulation_time_s: float,
    ) -> None:
        """Reveal cells along forward range rays, stopping at blocking roofs."""

        if self.collision_map is None or not hasattr(
            self.collision_map,
            "surface_height",
        ):
            return

        cfg = self.config
        position = np.asarray(position_world_m, dtype=float)
        yaw = np.radians(float(yaw_degrees))
        angles = yaw + np.radians(
            np.linspace(
                -cfg.sensor_horizontal_fov_deg * 0.5,
                cfg.sensor_horizontal_fov_deg * 0.5,
                cfg.sensor_ray_count,
            )
        )
        distances = np.arange(
            cfg.belief_resolution_m,
            cfg.sensor_range_m + cfg.belief_resolution_m * 0.5,
            cfg.belief_resolution_m,
        )
        new_observations = 0
        for angle in angles:
            direction = np.array([np.cos(angle), 0.0, np.sin(angle)])
            for distance in distances:
                point = position + direction * distance
                surface = self.collision_map.surface_height(point, 0.0)
                if not np.isfinite(surface):
                    break
                key = self._key(point[0], point[2])
                self.cells[key] = BeliefCell(float(surface), simulation_time_s)
                new_observations += 1
                if surface + cfg.collision_clearance_m >= position[1]:
                    break

        # The cell underneath the aircraft is always observable from the depth
        # camera's small downward pitch.
        surface = self.collision_map.surface_height(position, 0.0)
        if np.isfinite(surface):
            self.cells[self._key(position[0], position[2])] = BeliefCell(
                float(surface),
                simulation_time_s,
            )
        self.observation_count += new_observations
        self._prune(position, simulation_time_s)

    def _prune(self, position: np.ndarray, simulation_time_s: float) -> None:
        cfg = self.config
        resolution = cfg.belief_resolution_m
        keep = {}
        for key, cell in self.cells.items():
            center = np.array(
                [(key[0] + 0.5) * resolution, (key[1] + 0.5) * resolution]
            )
            distance = np.linalg.norm(center - position[[0, 2]])
            age = simulation_time_s - cell.observed_at_s
            if distance <= cfg.belief_keep_radius_m and age <= cfg.belief_max_age_s:
                keep[key] = cell
        self.cells = keep

    def surface_height(self, position_world_m: np.ndarray) -> Optional[float]:
        key = self._key(
            float(position_world_m[0]),
            float(position_world_m[2]),
        )
        radius_cells = max(
            1,
            int(
                np.ceil(
                    self.config.collision_clearance_m
                    / self.config.belief_resolution_m
                )
            ),
        )
        nearby = [
            cell.surface_height_m
            for dx in range(-radius_cells, radius_cells + 1)
            for dz in range(-radius_cells, radius_cells + 1)
            if (cell := self.cells.get((key[0] + dx, key[1] + dz))) is not None
        ]
        return max(nearby) if nearby else None


@dataclass
class CandidateRollout:
    command_world_mps: np.ndarray
    trajectory_world_m: np.ndarray
    score: float
    minimum_clearance_m: float
    observed_ratio: float
    predicted_collision: bool


class UrbanWorldModelMPC:
    """Sensor belief + cloned 6-DOF rollouts + first-action MPC."""

    backend_name = "city_belief_multirotor_mpc_v1"

    def __init__(
        self,
        collision_map,
        config: UrbanWorldModelConfig | None = None,
    ) -> None:
        self.config = config or UrbanWorldModelConfig()
        self.belief = LocalHeightBelief(collision_map, self.config)
        self.previous_command = np.zeros(3, dtype=float)
        self.last_plan_time = -float("inf")
        self.last_goal = None
        self.last_decision: Optional[dict] = None
        self.decision_sequence = 0

    @staticmethod
    def _clone_dynamics(model: MultirotorDynamics) -> MultirotorDynamics:
        clone = MultirotorDynamics(model.parameters)
        clone.orientation = model.orientation.copy()
        clone.angular_velocity = model.angular_velocity.copy()
        clone.motor_omega = model.motor_omega.copy()
        clone.initialized = model.initialized
        clone.last_power_w = model.last_power_w
        clone.last_total_thrust = model.last_total_thrust
        clone.last_motor_thrusts = model.last_motor_thrusts.copy()
        return clone

    def _commands(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        fallback_yaw_degrees: float,
    ) -> Iterable[np.ndarray]:
        cfg = self.config
        horizontal = goal[[0, 2]] - position[[0, 2]]
        horizontal_distance = float(np.linalg.norm(horizontal))
        vertical_error = float(goal[1] - position[1])
        if horizontal_distance < 2.0:
            climb_candidates = sorted(
                set(
                    [
                        -min(2.0, max(0.0, -vertical_error)),
                        0.0,
                        min(2.5, max(0.0, vertical_error)),
                    ]
                )
            )
            for climb in climb_candidates:
                yield np.array([0.0, climb, 0.0], dtype=float)
            return

        base_heading = np.degrees(np.arctan2(horizontal[1], horizontal[0]))
        if not np.isfinite(base_heading):
            base_heading = fallback_yaw_degrees
        for speed in cfg.cruise_speeds_mps:
            for offset in cfg.heading_offsets_deg:
                heading = np.radians(base_heading + offset)
                for climb in cfg.climb_rates_mps:
                    yield np.array(
                        [
                            np.cos(heading) * speed,
                            climb,
                            np.sin(heading) * speed,
                        ],
                        dtype=float,
                    )
        yield np.zeros(3, dtype=float)

    def _rollout(
        self,
        *,
        command: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        yaw_degrees: float,
        dynamics_model: MultirotorDynamics,
        wind_velocity: np.ndarray,
        payload_mass: float,
        max_acceleration: float,
    ) -> np.ndarray:
        cfg = self.config
        model = self._clone_dynamics(dynamics_model)
        simulated_position = np.asarray(position, dtype=float).copy()
        simulated_velocity = np.asarray(velocity, dtype=float).copy()
        target = simulated_position + command * cfg.horizon_s
        horizontal = np.hypot(command[0], command[2])
        desired_yaw = (
            np.degrees(np.arctan2(command[2], command[0]))
            if horizontal > 0.2
            else yaw_degrees
        )
        points = []
        steps = max(1, int(np.ceil(cfg.horizon_s / cfg.rollout_dt_s)))
        for _ in range(steps):
            frame = model.step(
                position=simulated_position,
                velocity=simulated_velocity,
                target_position=target,
                target_velocity=command,
                desired_yaw_degrees=desired_yaw,
                wind_velocity=wind_velocity,
                payload_mass=payload_mass,
                max_acceleration=max_acceleration,
                dt=cfg.rollout_dt_s,
            )
            simulated_position = frame["position"]
            simulated_velocity = frame["velocity"]
            points.append(simulated_position.copy())
        return np.asarray(points, dtype=float)

    @staticmethod
    def _normalize_rows(
        values: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        output = values / np.maximum(norms, 1e-9)
        invalid = norms[:, 0] < 1e-9
        if np.any(invalid):
            output[invalid] = fallback
        return output

    @staticmethod
    def _quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
        q = UrbanWorldModelMPC._normalize_rows(
            quaternions,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        w, x, y, z = q.T
        matrices = np.empty((len(q), 3, 3), dtype=float)
        matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
        matrices[:, 0, 1] = 2 * (x * y - z * w)
        matrices[:, 0, 2] = 2 * (x * z + y * w)
        matrices[:, 1, 0] = 2 * (x * y + z * w)
        matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
        matrices[:, 1, 2] = 2 * (y * z - x * w)
        matrices[:, 2, 0] = 2 * (x * z - y * w)
        matrices[:, 2, 1] = 2 * (y * z + x * w)
        matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return matrices

    def _rollout_batch(
        self,
        *,
        commands: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        yaw_degrees: float,
        dynamics_model: MultirotorDynamics,
        wind_velocity: np.ndarray,
        payload_mass: float,
        max_acceleration: float,
    ) -> np.ndarray:
        """Vectorized equivalent of cloned 6-DOF candidate rollouts."""

        cfg = self.config
        commands = np.asarray(commands, dtype=float)
        candidate_count = len(commands)
        params = dynamics_model.parameters
        positions = np.repeat(
            np.asarray(position, dtype=float)[None],
            candidate_count,
            axis=0,
        )
        velocities = np.repeat(
            np.asarray(velocity, dtype=float)[None],
            candidate_count,
            axis=0,
        )
        targets = positions + commands * cfg.horizon_s
        wind = np.asarray(wind_velocity, dtype=float)[None]
        horizontal = np.hypot(commands[:, 0], commands[:, 2])
        desired_yaws = np.where(
            horizontal > 0.2,
            np.degrees(np.arctan2(commands[:, 2], commands[:, 0])),
            float(yaw_degrees),
        )

        if dynamics_model.initialized:
            orientations = np.repeat(
                dynamics_model.orientation[None],
                candidate_count,
                axis=0,
            )
            angular_velocities = np.repeat(
                dynamics_model.angular_velocity[None],
                candidate_count,
                axis=0,
            )
            motor_omega = np.repeat(
                dynamics_model.motor_omega[None],
                candidate_count,
                axis=0,
            )
        else:
            half_yaw = -np.radians(desired_yaws) * 0.5
            orientations = np.column_stack(
                (
                    np.cos(half_yaw),
                    np.zeros(candidate_count),
                    np.sin(half_yaw),
                    np.zeros(candidate_count),
                )
            )
            angular_velocities = np.zeros((candidate_count, 3), dtype=float)
            hover_thrust = params.mass * GRAVITY / 4.0
            hover_omega = np.sqrt(
                hover_thrust / params.thrust_coefficient
            )
            motor_omega = np.full(
                (candidate_count, 4),
                hover_omega,
                dtype=float,
            )

        total_mass = params.mass + max(0.0, payload_mass)
        gravity = np.array([0.0, -GRAVITY, 0.0])
        rollout_steps = max(
            1,
            int(np.ceil(cfg.horizon_s / cfg.rollout_dt_s)),
        )
        physics_steps = max(
            1,
            int(np.ceil(cfg.rollout_dt_s / MAX_PHYSICS_SUBSTEP)),
        )
        sub_dt = cfg.rollout_dt_s / physics_steps
        motor_alpha = 1.0 - np.exp(
            -sub_dt / max(params.motor_time_constant, 1e-3)
        )
        trajectories = np.empty(
            (candidate_count, rollout_steps, 3),
            dtype=float,
        )

        for rollout_index in range(rollout_steps):
            for _ in range(physics_steps):
                position_error = targets - positions
                velocity_error = commands - velocities
                acceleration_command = (
                    params.position_gain * position_error
                    + params.velocity_gain * velocity_error
                )
                horizontal_command = acceleration_command[:, [0, 2]]
                horizontal_magnitude = np.linalg.norm(
                    horizontal_command,
                    axis=1,
                )
                scale = np.minimum(
                    1.0,
                    max_acceleration
                    / np.maximum(horizontal_magnitude, 1e-9),
                )
                acceleration_command[:, 0] *= scale
                acceleration_command[:, 2] *= scale
                acceleration_command[:, 1] = np.clip(
                    acceleration_command[:, 1],
                    -max_acceleration,
                    max_acceleration,
                )

                required_specific_force = acceleration_command - gravity
                desired_up = self._normalize_rows(
                    required_specific_force,
                    np.array([0.0, 1.0, 0.0]),
                )
                yaw_radians = np.radians(desired_yaws)
                desired_forward_flat = np.column_stack(
                    (
                        np.cos(yaw_radians),
                        np.zeros(candidate_count),
                        np.sin(yaw_radians),
                    )
                )
                projection = np.sum(
                    desired_forward_flat * desired_up,
                    axis=1,
                    keepdims=True,
                )
                desired_forward = self._normalize_rows(
                    desired_forward_flat - desired_up * projection,
                    np.array([1.0, 0.0, 0.0]),
                )
                desired_side = self._normalize_rows(
                    np.cross(desired_forward, desired_up),
                    np.array([0.0, 0.0, 1.0]),
                )
                desired_rotation = np.stack(
                    (desired_forward, desired_up, desired_side),
                    axis=2,
                )

                rotation = self._quaternion_matrices(orientations)
                attitude_matrix_error = 0.5 * (
                    np.matmul(
                        np.transpose(desired_rotation, (0, 2, 1)),
                        rotation,
                    )
                    - np.matmul(
                        np.transpose(rotation, (0, 2, 1)),
                        desired_rotation,
                    )
                )
                attitude_error = np.column_stack(
                    (
                        attitude_matrix_error[:, 2, 1],
                        attitude_matrix_error[:, 0, 2],
                        attitude_matrix_error[:, 1, 0],
                    )
                )
                desired_moment = (
                    -params.attitude_gain * attitude_error
                    - params.angular_rate_gain * angular_velocities
                )
                desired_collective = total_mass * np.linalg.norm(
                    required_specific_force,
                    axis=1,
                )
                desired_collective = np.clip(
                    desired_collective,
                    0.0,
                    params.max_thrust_per_motor * 4.0,
                )
                desired_wrench = np.column_stack(
                    (desired_collective, desired_moment)
                )
                desired_motor_thrusts = (
                    desired_wrench @ dynamics_model._mixer_inverse.T
                )
                desired_motor_thrusts = np.clip(
                    desired_motor_thrusts,
                    0.0,
                    params.max_thrust_per_motor,
                )
                desired_omega = np.sqrt(
                    desired_motor_thrusts / params.thrust_coefficient
                )
                motor_omega += (
                    desired_omega - motor_omega
                ) * motor_alpha

                motor_thrusts = (
                    params.thrust_coefficient * motor_omega**2
                )
                actual_wrench = motor_thrusts @ dynamics_model._mixer.T
                total_thrust = actual_wrench[:, 0]
                body_moment = actual_wrench[:, 1:4]
                relative_air_velocity = velocities - wind
                drag_force_world = (
                    -params.linear_drag
                    * relative_air_velocity
                    * np.abs(relative_air_velocity)
                )
                thrust_world = (
                    rotation[:, :, 1] * total_thrust[:, None]
                )
                acceleration = (
                    thrust_world + drag_force_world
                ) / total_mass + gravity

                angular_acceleration = (
                    body_moment
                    - np.cross(
                        angular_velocities,
                        params.inertia * angular_velocities,
                    )
                    - params.angular_drag * angular_velocities
                ) / params.inertia
                angular_velocities += angular_acceleration * sub_dt
                angular_velocities = np.clip(
                    angular_velocities,
                    -8.0,
                    8.0,
                )

                w, x, y, z = orientations.T
                wx, wy, wz = angular_velocities.T
                quaternion_derivative = 0.5 * np.column_stack(
                    (
                        -x * wx - y * wy - z * wz,
                        w * wx + y * wz - z * wy,
                        w * wy + z * wx - x * wz,
                        w * wz + x * wy - y * wx,
                    )
                )
                orientations = self._normalize_rows(
                    orientations + quaternion_derivative * sub_dt,
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )
                velocities += acceleration * sub_dt
                positions += velocities * sub_dt

            trajectories[:, rollout_index] = positions

        return trajectories

    def _evaluate(
        self,
        command: np.ndarray,
        trajectory: np.ndarray,
        position: np.ndarray,
        goal: np.ndarray,
    ) -> CandidateRollout:
        cfg = self.config
        clearances = []
        observed = 0
        collision = False
        clearance_cost = 0.0
        for point in trajectory:
            surface = self.belief.surface_height(point)
            if surface is None:
                continue
            observed += 1
            clearance = float(point[1] - surface)
            clearances.append(clearance)
            collision = collision or clearance < cfg.collision_clearance_m
            violation = max(0.0, cfg.comfort_clearance_m - clearance)
            clearance_cost += violation * violation

        observed_ratio = observed / max(len(trajectory), 1)
        minimum_clearance = min(clearances) if clearances else float("inf")
        initial_distance = float(np.linalg.norm(goal - position))
        terminal_distance = float(np.linalg.norm(goal - trajectory[-1]))
        progress = initial_distance - terminal_distance
        unknown_ratio = 1.0 - observed_ratio
        smoothness = float(np.linalg.norm(command - self.previous_command))
        score = (
            cfg.progress_weight * progress
            - cfg.terminal_distance_weight * terminal_distance
            - cfg.clearance_weight * clearance_cost / max(observed, 1)
            - cfg.unknown_weight * unknown_ratio
            - cfg.smoothness_weight * smoothness
            - cfg.climb_weight * abs(float(command[1]))
            - (cfg.collision_penalty if collision else 0.0)
        )
        return CandidateRollout(
            command_world_mps=command.copy(),
            trajectory_world_m=trajectory,
            score=float(score),
            minimum_clearance_m=float(minimum_clearance),
            observed_ratio=float(observed_ratio),
            predicted_collision=bool(collision),
        )

    def plan(
        self,
        *,
        simulation_time_s: float,
        position_world_m: np.ndarray,
        velocity_world_mps: np.ndarray,
        yaw_degrees: float,
        goal_world_m: np.ndarray,
        dynamics_model: MultirotorDynamics,
        wind_velocity: np.ndarray,
        payload_mass: float,
        max_acceleration: float,
    ) -> dict:
        cfg = self.config
        goal = np.asarray(goal_world_m, dtype=float)
        due = (
            simulation_time_s - self.last_plan_time >= cfg.planning_interval_s
            or self.last_decision is None
            or self.last_goal is None
            or np.linalg.norm(goal - self.last_goal) > 1.0
        )
        if not due:
            return self.last_decision

        position = np.asarray(position_world_m, dtype=float)
        velocity = np.asarray(velocity_world_mps, dtype=float)
        self.belief.observe(position, yaw_degrees, simulation_time_s)
        commands = np.asarray(
            list(self._commands(position, goal, yaw_degrees)),
            dtype=float,
        )
        trajectories = self._rollout_batch(
            commands=commands,
            position=position,
            velocity=velocity,
            yaw_degrees=yaw_degrees,
            dynamics_model=dynamics_model,
            wind_velocity=np.asarray(wind_velocity, dtype=float),
            payload_mass=payload_mass,
            max_acceleration=max_acceleration,
        )
        evaluations = [
            self._evaluate(command, trajectory, position, goal)
            for command, trajectory in zip(commands, trajectories)
        ]

        evaluations.sort(key=lambda item: item.score, reverse=True)
        best = evaluations[0]
        safe_count = sum(not item.predicted_collision for item in evaluations)
        lookahead_index = min(
            len(best.trajectory_world_m) - 1,
            max(0, int(round(cfg.lookahead_s / cfg.rollout_dt_s)) - 1),
        )
        local_target = best.trajectory_world_m[lookahead_index]
        finite_clearance = np.isfinite(best.minimum_clearance_m)
        self.previous_command = best.command_world_mps.copy()
        self.last_plan_time = simulation_time_s
        self.last_goal = goal.copy()
        self.decision_sequence += 1
        self.last_decision = {
            "enabled": True,
            "backend": self.backend_name,
            "learned_backend": "disabled_unvalidated_source_to_city_domain",
            "decision_sequence": self.decision_sequence,
            "planning_time_s": round(float(simulation_time_s), 3),
            "horizon_s": cfg.horizon_s,
            "rollout_dt_s": cfg.rollout_dt_s,
            "candidate_count": len(evaluations),
            "safe_candidate_count": safe_count,
            "belief_cell_count": len(self.belief.cells),
            "belief_observations": self.belief.observation_count,
            "command_world_mps": best.command_world_mps.tolist(),
            "local_target_world_m": local_target.tolist(),
            "goal_world_m": goal.tolist(),
            "score": round(best.score, 4),
            "minimum_predicted_clearance_m": (
                round(best.minimum_clearance_m, 3)
                if finite_clearance
                else None
            ),
            "observed_ratio": round(best.observed_ratio, 4),
            "predicted_collision": best.predicted_collision,
            "selected_trajectory_world_m": best.trajectory_world_m.tolist(),
            "top_candidates": [
                {
                    "command_world_mps": item.command_world_mps.tolist(),
                    "score": round(item.score, 4),
                    "minimum_clearance_m": (
                        round(item.minimum_clearance_m, 3)
                        if np.isfinite(item.minimum_clearance_m)
                        else None
                    ),
                    "observed_ratio": round(item.observed_ratio, 4),
                    "predicted_collision": item.predicted_collision,
                    "trajectory_world_m": item.trajectory_world_m.tolist(),
                }
                for item in evaluations[:5]
            ],
        }
        return self.last_decision
