from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from uav_wm_navigation.control.transparent_safety import (
    TransparentSafetyLayer,
    body_flu_to_world_nwu,
    world_nwu_to_body_flu,
)
from uav_wm_navigation.simulators.base import SimulatorAdapter
from uav_wm_navigation.types import (
    ActionLimits,
    BodyVelocityAction,
    SensorFrame,
    EpisodeSpec,
    WorldModelObservation,
)


@dataclass(frozen=True, slots=True)
class UAVEnvConfig:
    physics_hz: int = 50
    sensor_hz: int = 10
    policy_hz: int = 5
    success_radius_m: float = 3.0
    success_dwell_s: float = 2.0
    max_episode_s: float = 180.0
    depth_clip_m: float = 120.0
    action_limits: ActionLimits = ActionLimits()

    def __post_init__(self) -> None:
        if self.physics_hz % self.sensor_hz or self.sensor_hz % self.policy_hz:
            raise ValueError("physics_hz, sensor_hz and policy_hz must divide exactly")


class UAVWorldModelEnv:
    """Simulator-independent deterministic single-UAV environment."""

    metadata = {
        "schema": "urbanfly-world-model-v3",
        "supported_schemas": ("urbanfly-world-model-v2", "urbanfly-world-model-v3"),
    }

    def __init__(
        self,
        simulator: SimulatorAdapter,
        *,
        config: UAVEnvConfig | None = None,
        safety_layer: TransparentSafetyLayer | None = None,
        seed: int = 20260731,
    ) -> None:
        self.simulator = simulator
        self.config = config or UAVEnvConfig()
        self.safety_layer = safety_layer or TransparentSafetyLayer(
            enabled=True, limits=self.config.action_limits
        )
        self.seed = int(seed)
        self.episode_id = ""
        self.goal_nwu = np.zeros(3, dtype=np.float32)
        self.step_id = 0
        self.previous_action = np.zeros(4, dtype=np.float32)
        self._last_distance = 0.0
        self._success_dwell = 0.0
        self._last_acceleration = np.zeros(3, dtype=np.float32)
        self._connected = False
        self.sensor_frames: list[SensorFrame] = []

    @staticmethod
    def _yaw_from_xyzw(quaternion: np.ndarray) -> float:
        x, y, z, w = np.asarray(quaternion, dtype=np.float64)
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def reset(
        self,
        *,
        goal_nwu: np.ndarray,
        episode_id: str | None = None,
        scenario: str = "MixedUrban",
        difficulty: str = "medium",
        episode_spec: EpisodeSpec | None = None,
    ) -> tuple[WorldModelObservation, dict]:
        if not self._connected:
            self.simulator.connect()
            self._connected = True
        self.simulator.reset()
        if episode_spec is not None:
            if episode_id is not None and episode_id != episode_spec.episode_id:
                raise ValueError("episode_id conflicts with episode_spec")
            episode_id = episode_spec.episode_id
            scenario = episode_spec.scenario
            self.seed = int(episode_spec.seed)
            if not np.allclose(np.asarray(goal_nwu, dtype=np.float32), episode_spec.goal_nwu_m):
                raise ValueError("goal_nwu conflicts with episode_spec.goal_nwu_m")
            self.simulator.set_initial_pose(np.asarray(episode_spec.start_nwu_m, dtype=np.float32))
        self.simulator.configure_scenario(scenario, difficulty, self.seed)
        self.goal_nwu = np.asarray(goal_nwu, dtype=np.float32)
        if self.goal_nwu.shape != (3,) or not np.isfinite(self.goal_nwu).all():
            raise ValueError("goal_nwu must be a finite vector with shape (3,)")
        self.simulator.set_goal(self.goal_nwu)
        self.simulator.takeoff()
        self.episode_id = episode_id or f"urbanfly-{uuid4().hex[:12]}"
        self.step_id = 0
        self.previous_action.fill(0.0)
        self._success_dwell = 0.0
        self._last_acceleration.fill(0.0)
        self.sensor_frames.clear()
        self.safety_layer.reset()
        observation, state, sensor = self._capture_observation()
        self._last_distance = float(np.linalg.norm(self.goal_nwu - state.position))
        return observation, {
            "episode_id": self.episode_id,
            "schema": self.metadata["schema"] if episode_spec is not None else "urbanfly-world-model-v2",
            "episode_spec": episode_spec,
            "state": state,
            "sensor": sensor,
            "actor_states": self.simulator.get_actor_states(),
        }

    def _capture_observation(self) -> tuple[WorldModelObservation, object, SensorFrame]:
        sensor = self.simulator.get_depth()
        state = self.simulator.get_kinematics()
        rgb = self.simulator.get_rgb()
        if rgb is None:
            rgb = np.zeros((*sensor.depth_m.shape, 3), dtype=np.uint8)
        yaw = self._yaw_from_xyzw(state.orientation_xyzw)
        goal_body = world_nwu_to_body_flu(self.goal_nwu - state.position, yaw)
        velocity_body = world_nwu_to_body_flu(state.linear_velocity, yaw)
        angular_body = world_nwu_to_body_flu(state.angular_velocity, yaw)
        height, width = sensor.depth_m.shape
        if sensor.camera_intrinsics is None:
            focal = width / (2.0 * np.tan(np.deg2rad(90.0) / 2.0))
            intrinsics = np.asarray(
                [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0, 0, 1]],
                dtype=np.float32,
            )
        else:
            intrinsics = sensor.camera_intrinsics
        clipped_depth = np.clip(sensor.depth_m, 0.0, self.config.depth_clip_m)
        observation = WorldModelObservation(
            episode_id=self.episode_id,
            step_id=self.step_id,
            sim_time=float(self.simulator.get_timestamp()),
            rgb=rgb,
            depth_m=clipped_depth,
            depth_valid_mask=sensor.valid_mask,
            goal_body_flu_m=goal_body,
            linear_velocity_body_flu_mps=velocity_body,
            angular_velocity_body_flu_rps=angular_body,
            gravity_body_flu=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            previous_action=self.previous_action.copy(),
            sensor_timestamp=float(sensor.timestamp),
            state_timestamp=float(state.timestamp),
            camera_intrinsics=intrinsics,
            camera_extrinsics_body=np.eye(4, dtype=np.float32),
        )
        self.sensor_frames.append(sensor)
        return observation, state, sensor

    def step(
        self,
        action_normalized: np.ndarray,
        *,
        predicted_risk: float = 0.0,
        shield_enabled: bool | None = None,
    ) -> tuple[WorldModelObservation, float, bool, bool, dict]:
        if not self.episode_id:
            raise RuntimeError("reset must be called before step")
        action = BodyVelocityAction(action_normalized, self.config.action_limits)
        if shield_enabled is not None:
            self.safety_layer.enabled = bool(shield_enabled)
        state = self.simulator.get_kinematics()
        sensor = self.simulator.get_depth()
        yaw = self._yaw_from_xyzw(state.orientation_xyzw)
        collision_before = bool(
            self.simulator.get_collision_info().get("has_collided", False)
        )
        velocity_body, yaw_rate, audit = self.safety_layer.apply(
            action,
            state=state,
            sensor=sensor,
            yaw_rad=yaw,
            episode_id=self.episode_id,
            step_id=self.step_id,
            sim_time=float(self.simulator.get_timestamp()),
            dt=1.0 / self.config.policy_hz,
            predicted_risk=predicted_risk,
            collision=collision_before,
        )
        world_velocity = body_flu_to_world_nwu(velocity_body, yaw)
        physics_dt = 1.0 / self.config.physics_hz
        physics_per_sensor = self.config.physics_hz // self.config.sensor_hz
        sensor_observation = None
        for physics_index in range(self.config.physics_hz // self.config.policy_hz):
            self.simulator.execute_velocity_command(world_velocity, yaw_rate, physics_dt)
            if (physics_index + 1) % physics_per_sensor == 0:
                sensor_observation = self._capture_observation()
        assert sensor_observation is not None
        self.step_id += 1
        self.previous_action = action.normalized.copy()
        observation, next_state, next_sensor = sensor_observation
        observation.step_id = self.step_id
        observation.previous_action = self.previous_action.copy()
        collision = bool(self.simulator.get_collision_info().get("has_collided", False))
        distance = float(np.linalg.norm(self.goal_nwu - next_state.position))
        progress = self._last_distance - distance
        self._last_distance = distance
        within_goal = distance <= self.config.success_radius_m
        policy_dt = 1.0 / self.config.policy_hz
        self._success_dwell = self._success_dwell + policy_dt if within_goal else 0.0
        success = self._success_dwell >= self.config.success_dwell_s
        acceleration = np.asarray(next_state.linear_acceleration, dtype=np.float32)
        jerk = float(np.linalg.norm(acceleration - self._last_acceleration) / policy_dt)
        self._last_acceleration = acceleration
        valid_depth = next_sensor.depth_m[next_sensor.valid_mask]
        clearance = float(valid_depth.min()) if valid_depth.size else 0.0
        reward = (
            1.0 * progress
            + (50.0 if success else 0.0)
            - (40.0 if collision else 0.0)
            - 0.02 * jerk
            - 0.02
            - max(0.0, 3.0 - clearance) * 0.25
        )
        truncated = float(self.simulator.get_timestamp()) >= self.config.max_episode_s
        info = {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "sim_time": float(self.simulator.get_timestamp()),
            "success": success,
            "collision": collision,
            "goal_distance_m": distance,
            "minimum_clearance_m": clearance,
            "progress_m": progress,
            "jerk_mps3": jerk,
            "raw_action": audit.raw_action_physical.copy(),
            "executed_action": audit.executed_action_physical.copy(),
            "safety_audit": audit,
            "safety_interventions": self.safety_layer.intervention_count,
            "state": next_state,
            "sensor": next_sensor,
            "actor_states": self.simulator.get_actor_states(),
        }
        return observation, float(reward), bool(success or collision), bool(truncated), info

    def close(self) -> None:
        self.simulator.close()
        self._connected = False


# Compatibility names retained for all existing scripts/checkpoints.
UrbanFlyEnvConfig = UAVEnvConfig


class UrbanFlyWorldModelEnv(UAVWorldModelEnv):
    """Compatibility wrapper for the former UrbanFly-specific class name."""
