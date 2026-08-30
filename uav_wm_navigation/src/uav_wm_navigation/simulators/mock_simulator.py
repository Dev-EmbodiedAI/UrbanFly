from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from uav_wm_navigation.simulators.base import SimulatorAdapter
from uav_wm_navigation.types import ActorState, SensorFrame, VehicleState


@dataclass(slots=True)
class SphereObstacle:
    center: np.ndarray
    radius: float
    dynamic_velocity: np.ndarray | None = None


class MockSimulator(SimulatorAdapter):
    def __init__(
        self,
        seed: int = 0,
        depth_shape: tuple[int, int] = (96, 160),
        depth_max_m: float = 20.0,
        control_dt: float = 0.1,
        scenario: str = "StaticObstacle",
        vehicle_name: str = "SimpleFlight",
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.depth_shape = depth_shape
        self.depth_max_m = float(depth_max_m)
        self.control_dt = float(control_dt)
        self.scenario = scenario
        self.vehicle_name = vehicle_name
        self.connected = False
        self.paused = False
        self.goal = np.array([12.0, 0.0, 2.0], dtype=np.float64)
        self._start_monotonic = time.monotonic()
        self._build_scenario()
        self.reset()

    def _build_scenario(self) -> None:
        base = [] if self.scenario == "OpenSpace" else [SphereObstacle(np.array([6.0, 1.8, 2.0]), 1.0)]
        if self.scenario in {"NarrowPassage", "MixedUrban", "DenseMixedUrban", "StreetCanyon"}:
            base.extend(
                [SphereObstacle(np.array([7.0, -1.7, 2.0]), 1.2), SphereObstacle(np.array([7.0, 1.7, 2.0]), 1.2)]
            )
        if self.scenario in {"DynamicCrossing", "OccludedCrossing", "MixedUrban", "DenseMixedUrban"}:
            base.append(SphereObstacle(np.array([8.0, -4.0, 2.0]), 0.8, np.array([0.0, 0.8, 0.0])))
        self.obstacles = base

    def connect(self) -> None:
        self.connected = True

    def reset(self) -> None:
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0
        self.landed = True
        self.collided = False
        self.sim_time = 0.0
        self.paused = False

    def takeoff(self) -> None:
        if not self.connected:
            raise RuntimeError("simulator is not connected")
        self.position[2] = 2.0
        self.landed = False

    def land(self) -> None:
        self.position[2] = 0.0
        self.velocity.fill(0.0)
        self.landed = True

    def _nearest_forward_depth(self) -> float:
        heading = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        best = self.depth_max_m
        for obstacle in self.obstacles:
            relative = obstacle.center - self.position
            forward = float(relative @ heading)
            lateral = float(np.linalg.norm(relative - forward * heading))
            if forward > 0.0 and lateral <= obstacle.radius + 2.0:
                best = min(best, max(0.05, forward - obstacle.radius))
        return best

    def get_depth(self) -> SensorFrame:
        depth = np.full(self.depth_shape, self.depth_max_m, dtype=np.float32)
        nearest = self._nearest_forward_depth()
        h, w = self.depth_shape
        depth[h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = nearest
        if self.scenario in {"SensorCorruption", "MixedUrban", "DenseMixedUrban"}:
            depth += self.rng.normal(0.0, 0.03, depth.shape).astype(np.float32)
            if int(self.sim_time / max(self.control_dt, 1e-6)) % 11 == 7:
                depth[h // 3 : h // 2, w // 4 : w // 2] = np.nan
        valid = np.isfinite(depth) & (depth > 0.0) & (depth <= self.depth_max_m)
        return SensorFrame(self.get_timestamp(), depth, valid)

    def get_rgb(self) -> np.ndarray:
        return np.zeros((*self.depth_shape, 3), dtype=np.uint8)

    def get_kinematics(self) -> VehicleState:
        orientation = Rotation.from_euler("z", self.yaw).as_quat().astype(np.float32)
        return VehicleState(
            timestamp=self.get_timestamp(),
            position=self.position.copy(),
            orientation_xyzw=orientation,
            linear_velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            linear_acceleration=self.acceleration.copy(),
            vehicle_name=self.vehicle_name,
        )

    def get_collision_info(self) -> dict[str, object]:
        return {"has_collided": self.collided, "object_name": "mock_obstacle" if self.collided else ""}

    def get_actor_states(self) -> list[ActorState]:
        actors = []
        for index, obstacle in enumerate(self.obstacles):
            actors.append(ActorState(
                actor_id=index,
                actor_type="dynamic_sphere" if obstacle.dynamic_velocity is not None else "static_sphere",
                position=obstacle.center.copy(),
                velocity=np.zeros(3) if obstacle.dynamic_velocity is None else obstacle.dynamic_velocity.copy(),
                bbox_extent=np.full(3, obstacle.radius, dtype=np.float32),
                timestamp=self.get_timestamp(),
                scripted=obstacle.dynamic_velocity is not None,
            ))
        return actors

    def get_timestamp(self) -> float:
        return self.sim_time

    def set_goal(self, goal_nwu: np.ndarray) -> None:
        goal = np.asarray(goal_nwu, dtype=np.float64)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("goal must be a finite [3] vector")
        self.goal = goal

    def set_initial_pose(self, position_nwu: np.ndarray) -> None:
        self.position = np.asarray(position_nwu, dtype=np.float64).copy()

    def configure_scenario(self, scenario: str, difficulty: str, seed: int) -> None:
        self.scenario = str(scenario)
        self.rng = np.random.default_rng(int(seed))
        self._build_scenario()
        scale = {"easy": 0.85, "medium": 1.0, "hard": 1.2}.get(str(difficulty), 1.0)
        for obstacle in self.obstacles:
            obstacle.radius *= scale
            if obstacle.dynamic_velocity is not None:
                obstacle.dynamic_velocity *= scale

    def execute_velocity_command(self, velocity_nwu: np.ndarray, yaw_rate: float, duration: float) -> None:
        if self.paused:
            return
        velocity = np.asarray(velocity_nwu, dtype=np.float64)
        duration = float(duration)
        if velocity.shape != (3,) or duration <= 0:
            raise ValueError("velocity must be [3] and duration must be positive")
        previous_velocity = self.velocity.copy()
        steps = max(1, int(np.ceil(duration / self.control_dt)))
        dt = duration / steps
        for _ in range(steps):
            self.velocity = velocity
            self.position += self.velocity * dt
            self.yaw += float(yaw_rate) * dt
            self.sim_time += dt
            for obstacle in self.obstacles:
                if obstacle.dynamic_velocity is not None:
                    obstacle.center += obstacle.dynamic_velocity * dt
                if np.linalg.norm(self.position - obstacle.center) <= obstacle.radius + 0.25:
                    self.collided = True
        self.acceleration = (self.velocity - previous_velocity) / duration
        self.angular_velocity = np.array([0.0, 0.0, yaw_rate])

    def pause(self) -> None:
        self.paused = True

    def continue_simulation(self) -> None:
        self.paused = False
