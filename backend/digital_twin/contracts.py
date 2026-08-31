"""实景与程序化数字孪生共享的任务、观测和反馈契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _finite_array(value, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须是 {ndim} 维有限数组")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class DigitalTwinMission:
    environment_id: str
    episode_id: str
    starts_enu_m: np.ndarray
    goals_enu_m: np.ndarray
    agent_provider: str
    privileged_goal_mode: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        starts = _finite_array(self.starts_enu_m, ndim=2, name="starts_enu_m")
        goals = _finite_array(self.goals_enu_m, ndim=2, name="goals_enu_m")
        if starts.shape != goals.shape or starts.shape[1:] != (3,):
            raise ValueError("start/goal 必须具有相同的 [N,3] shape")
        if not self.environment_id or not self.episode_id or not self.agent_provider:
            raise ValueError("mission 标识和 Agent provider 不能为空")
        object.__setattr__(self, "starts_enu_m", starts)
        object.__setattr__(self, "goals_enu_m", goals)


@dataclass(frozen=True, slots=True)
class DigitalTwinObservation:
    environment_id: str
    episode_id: str
    sequence: int
    timestamp_s: float
    depth: np.ndarray
    state: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        depth = _finite_array(self.depth, ndim=4, name="depth")
        state = _finite_array(self.state, ndim=2, name="state")
        if depth.shape[0] != state.shape[0] or depth.shape[-1] != 1:
            raise ValueError("depth/state 的无人机 batch 不一致")
        if depth.min() < 0.0 or depth.max() > 1.0:
            raise ValueError("depth 必须归一化到 [0,1]")
        if self.sequence < 0 or not np.isfinite(self.timestamp_s):
            raise ValueError("observation sequence/timestamp 无效")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "state", state)

    @property
    def drone_count(self) -> int:
        return int(self.state.shape[0])

    @property
    def positions_enu_m(self) -> np.ndarray:
        return self.state[:, 0:3]


@dataclass(frozen=True, slots=True)
class DigitalTwinFeedback:
    observation: DigitalTwinObservation
    reward: float
    terminated: bool
    truncated: bool
    per_drone_success: tuple[bool, ...]
    per_drone_collision: tuple[bool, ...]
    per_drone_failure_reason: tuple[str, ...]
    raw_info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.observation.drone_count
        if not (
            len(self.per_drone_success)
            == len(self.per_drone_collision)
            == len(self.per_drone_failure_reason)
            == n
        ):
            raise ValueError("per-drone 反馈长度必须与观测 batch 一致")
