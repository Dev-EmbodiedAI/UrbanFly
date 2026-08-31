"""Swarm ``cf_swarm_autopilot`` 的环境无关 policy contract。

本模块不依赖 Swarm 仓库。它把 UrbanFly Helsinki 的坐标、RGB-D 和多机
运动状态显式编码为 Swarm 的 ``depth[N,128,128,1]``、``state[N,190]``，
并把同一个 policy 的动作转回 UrbanFly 的世界坐标顺序。几何、动力学和
碰撞仍由各自环境负责，避免把两个仿真器的内部实现硬拼在一起。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


MIN_DRONES = 2
MAX_DRONES = 8
DEPTH_SHAPE = (128, 128, 1)
STATE_WIDTH = 190
ACTION_WIDTH = 5
ACTION_HISTORY_LENGTH = 25
NEIGHBOUR_SLOTS = 7


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,):
        raise ValueError(f"{name} 必须是 shape=({size},)，实际为 {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 包含非有限值")
    return result


def urbanfly_world_to_enu(value: Sequence[float]) -> np.ndarray:
    """UrbanFly ``[east, up, north]`` → policy ``[east, north, up]``。"""

    east_up_north = _finite_vector(value, 3, "UrbanFly 世界向量")
    return east_up_north[[0, 2, 1]]


def enu_to_urbanfly_world(value: Sequence[float]) -> np.ndarray:
    """Policy ``[east, north, up]`` → UrbanFly ``[east, up, north]``。"""

    east_north_up = _finite_vector(value, 3, "ENU 向量")
    return east_north_up[[0, 2, 1]]


@dataclass(frozen=True, slots=True)
class CanonicalDroneState:
    """已转换到 ENU/radian 的单机运动状态。"""

    drone_id: str
    position_enu_m: np.ndarray
    orientation_rpy_rad: np.ndarray
    linear_velocity_enu_mps: np.ndarray
    angular_velocity_rpy_radps: np.ndarray
    altitude_distance_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_enu_m", _finite_vector(self.position_enu_m, 3, "position_enu_m"))
        object.__setattr__(self, "orientation_rpy_rad", _finite_vector(self.orientation_rpy_rad, 3, "orientation_rpy_rad"))
        object.__setattr__(self, "linear_velocity_enu_mps", _finite_vector(self.linear_velocity_enu_mps, 3, "linear_velocity_enu_mps"))
        object.__setattr__(self, "angular_velocity_rpy_radps", _finite_vector(self.angular_velocity_rpy_radps, 3, "angular_velocity_rpy_radps"))
        if not self.drone_id:
            raise ValueError("drone_id 不能为空")
        if not math.isfinite(self.altitude_distance_m) or self.altitude_distance_m < 0.0:
            raise ValueError("altitude_distance_m 必须是非负有限值")

    @classmethod
    def from_urbanfly(
        cls,
        *,
        drone_id: str,
        position_eun_m: Sequence[float],
        roll_rad: float,
        pitch_rad: float,
        yaw_degrees: float,
        velocity_eun_mps: Sequence[float],
        angular_velocity_eun_radps: Sequence[float],
        altitude_distance_m: float,
    ) -> "CanonicalDroneState":
        angular_eun = urbanfly_world_to_enu(angular_velocity_eun_radps)
        return cls(
            drone_id=drone_id,
            position_enu_m=urbanfly_world_to_enu(position_eun_m),
            orientation_rpy_rad=np.asarray(
                [roll_rad, pitch_rad, math.radians(yaw_degrees)], dtype=np.float32
            ),
            linear_velocity_enu_mps=urbanfly_world_to_enu(velocity_eun_mps),
            angular_velocity_rpy_radps=angular_eun,
            altitude_distance_m=float(altitude_distance_m),
        )


@dataclass(frozen=True, slots=True)
class SwarmPolicyObservation:
    depth: np.ndarray
    state: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float32)
        state = np.asarray(self.state, dtype=np.float32)
        n = state.shape[0] if state.ndim == 2 else -1
        if not MIN_DRONES <= n <= MAX_DRONES:
            raise ValueError(f"无人机数量必须在 {MIN_DRONES}–{MAX_DRONES}，实际为 {n}")
        if depth.shape != (n, *DEPTH_SHAPE):
            raise ValueError(f"depth shape 应为 {(n, *DEPTH_SHAPE)}，实际为 {depth.shape}")
        if state.shape != (n, STATE_WIDTH):
            raise ValueError(f"state shape 应为 {(n, STATE_WIDTH)}，实际为 {state.shape}")
        if not np.all(np.isfinite(depth)) or depth.min() < 0.0 or depth.max() > 1.0:
            raise ValueError("depth 必须是 [0,1] 内有限值")
        if not np.all(np.isfinite(state)):
            raise ValueError("state 包含非有限值")
        object.__setattr__(self, "depth", np.ascontiguousarray(depth))
        object.__setattr__(self, "state", np.ascontiguousarray(state))

    def as_policy_dict(self) -> dict[str, np.ndarray]:
        return {"depth": self.depth, "state": self.state}


class SwarmPolicyEncoder:
    """构造与上游 `submission_zip.v1` 对齐的批量观测。"""

    def __init__(self, *, max_depth_m: float = 20.0, max_altitude_ray_m: float = 20.0):
        if max_depth_m <= 0.0 or max_altitude_ray_m <= 0.0:
            raise ValueError("深度量程必须为正数")
        self.max_depth_m = float(max_depth_m)
        self.max_altitude_ray_m = float(max_altitude_ray_m)

    def encode(
        self,
        *,
        depth_m: np.ndarray,
        drones: Sequence[CanonicalDroneState],
        shared_clue_enu_m: Sequence[float],
        action_history: np.ndarray | None = None,
    ) -> SwarmPolicyObservation:
        n = len(drones)
        if not MIN_DRONES <= n <= MAX_DRONES:
            raise ValueError(f"无人机数量必须在 {MIN_DRONES}–{MAX_DRONES}")
        if len({drone.drone_id for drone in drones}) != n:
            raise ValueError("drone_id 必须唯一")

        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape == (n, 128, 128):
            depth = depth[..., None]
        if depth.shape != (n, *DEPTH_SHAPE):
            raise ValueError(f"metric depth shape 应为 {(n, *DEPTH_SHAPE)}")
        finite_depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, self.max_depth_m)
        normalized_depth = np.clip(finite_depth / self.max_depth_m, 0.0, 1.0).astype(np.float32)

        clue = _finite_vector(shared_clue_enu_m, 3, "shared_clue_enu_m")
        if action_history is None:
            history = np.zeros((n, ACTION_HISTORY_LENGTH, ACTION_WIDTH), dtype=np.float32)
        else:
            history = np.asarray(action_history, dtype=np.float32)
            expected = (n, ACTION_HISTORY_LENGTH, ACTION_WIDTH)
            if history.shape != expected:
                raise ValueError(f"action_history shape 应为 {expected}，实际为 {history.shape}")
            if not np.all(np.isfinite(history)):
                raise ValueError("action_history 包含非有限值")

        positions = np.stack([drone.position_enu_m for drone in drones])
        velocities = np.stack([drone.linear_velocity_enu_mps for drone in drones])
        rows = np.zeros((n, STATE_WIDTH), dtype=np.float32)
        for index, drone in enumerate(drones):
            rows[index, 0:3] = drone.position_enu_m
            rows[index, 3:6] = drone.orientation_rpy_rad
            rows[index, 6:9] = drone.linear_velocity_enu_mps
            rows[index, 9:12] = drone.angular_velocity_rpy_radps
            rows[index, 12:137] = history[index].reshape(-1)
            rows[index, 137] = np.clip(
                drone.altitude_distance_m / self.max_altitude_ray_m, 0.0, 1.0
            )
            rows[index, 138:141] = clue - drone.position_enu_m

            neighbours = []
            for other_index in range(n):
                if other_index == index:
                    continue
                relative_position = positions[other_index] - positions[index]
                relative_velocity = velocities[other_index] - velocities[index]
                distance_squared = float(np.dot(relative_position, relative_position))
                neighbours.append(
                    (distance_squared, other_index, relative_position, relative_velocity)
                )
            neighbours.sort(key=lambda item: (item[0], item[1]))
            for slot, (_, _, relative_position, relative_velocity) in enumerate(neighbours):
                start = 141 + slot * 7
                rows[index, start : start + 3] = relative_position
                rows[index, start + 3 : start + 6] = relative_velocity
                rows[index, start + 6] = 1.0

        return SwarmPolicyObservation(depth=normalized_depth, state=rows)


def normalized_swarm_action_to_urbanfly(action: np.ndarray) -> dict[str, np.ndarray]:
    """校验 Swarm action，并转成 UrbanFly 世界方向和绝对 yaw（degree）。"""

    values = np.asarray(action, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_WIDTH:
        raise ValueError("action 必须是 [N,5]")
    if not MIN_DRONES <= values.shape[0] <= MAX_DRONES:
        raise ValueError("action 的无人机数量必须在 2–8")
    if not np.all(np.isfinite(values)):
        raise ValueError("action 包含非有限值")
    if np.any(values[:, 0:3] < -1.0) or np.any(values[:, 0:3] > 1.0):
        raise ValueError("方向分量必须在 [-1,1]")
    if np.any(values[:, 3] < 0.0) or np.any(values[:, 3] > 1.0):
        raise ValueError("speed 必须在 [0,1]")
    if np.any(values[:, 4] < -1.0) or np.any(values[:, 4] > 1.0):
        raise ValueError("yaw 必须在 [-1,1]")
    directions = np.stack([enu_to_urbanfly_world(row) for row in values[:, 0:3]])
    return {
        "direction_eun": directions.astype(np.float32),
        "speed_fraction": values[:, 3].copy(),
        "absolute_yaw_degrees": (values[:, 4] * 180.0).astype(np.float32),
    }
