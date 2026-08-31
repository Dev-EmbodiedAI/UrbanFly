"""Helsinki 实景数字孪生的因果执行适配器。

该层把现有 WebSocket/RGB-D adapter 收敛为与 Swarm 相同的
``reset → observation → action → fresh feedback`` 生命周期。它不改变冻结的
规划器、控制器或安全真值，只负责接口、时间戳和执行反馈的 fail-closed 检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class HelsinkiPlatformObservation:
    sequence: int
    timestamp_s: float
    frame: Any
    kinematics: Any

    def __post_init__(self) -> None:
        rgb = np.asarray(self.frame.rgb)
        depth = np.asarray(self.frame.depth_m)
        valid = np.asarray(self.frame.valid_mask)
        position = np.asarray(self.kinematics.position, dtype=float)
        orientation = np.asarray(self.kinematics.orientation_xyzw, dtype=float)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Helsinki RGB 必须是 [H,W,3]")
        if depth.shape != rgb.shape[:2] or valid.shape != depth.shape:
            raise ValueError("Helsinki RGB/depth/valid mask shape 不一致")
        if not np.isfinite(depth).all() or not np.isfinite(position).all():
            raise ValueError("Helsinki observation 含非有限值")
        if position.shape != (3,) or orientation.shape != (4,):
            raise ValueError("Helsinki pose shape 无效")
        if self.sequence < 0 or not np.isfinite(self.timestamp_s):
            raise ValueError("Helsinki sequence/timestamp 无效")


@dataclass(frozen=True, slots=True)
class HelsinkiPlatformFeedback:
    observation: HelsinkiPlatformObservation
    factual_action: Mapping[str, Any]
    collision: Mapping[str, Any]

    @property
    def stale_action(self) -> bool:
        return bool(self.factual_action.get("stale_action", False))

    @property
    def safety_intervened(self) -> bool:
        return bool(self.factual_action.get("safety_intervened", False))

    @property
    def has_collided(self) -> bool:
        return bool(self.collision.get("has_collided", False))


class HelsinkiDigitalTwinAdapter:
    """单一 Helsinki 传感器表面的强顺序数字孪生 session。"""

    def __init__(self, config: Mapping[str, Any] | None = None, *, raw_adapter: Any = None) -> None:
        if raw_adapter is None:
            from uav_wm_navigation.simulators.helsinki_websocket_adapter import (
                HelsinkiWebSocketAdapter,
            )

            raw_adapter = HelsinkiWebSocketAdapter(dict(config or {}))
        self.raw = raw_adapter
        self.sequence = 0
        self.last_timestamp_s: float | None = None
        self.active = False

    def connect_and_reset(
        self,
        *,
        task_type: str,
        split: str,
        seed: int,
        start_enu_m: np.ndarray,
        goal_enu_m: np.ndarray,
    ) -> HelsinkiPlatformObservation:
        if self.active:
            raise RuntimeError("Helsinki adapter session 已启动")
        self.raw.connect()
        self.raw.reset()
        self.raw.configure_scenario(task_type, split, int(seed))
        self.raw.set_initial_pose(np.asarray(start_enu_m, dtype=float))
        self.raw.set_goal(np.asarray(goal_enu_m, dtype=float))
        self.raw.takeoff()
        self.sequence = 0
        self.last_timestamp_s = None
        self.active = True
        return self._observe()

    def step_velocity(
        self,
        command_world_enu: np.ndarray,
        yaw_rate_rad_s: float,
        duration_s: float,
        *,
        inference_latency_ms: float,
        predicted_risk: float,
    ) -> HelsinkiPlatformFeedback:
        if not self.active:
            raise RuntimeError("Helsinki adapter 必须先 reset")
        factual = self.raw.execute_velocity_command(
            np.asarray(command_world_enu, dtype=float),
            float(yaw_rate_rad_s),
            float(duration_s),
            inference_latency_ms=float(inference_latency_ms),
            predicted_risk=float(predicted_risk),
        )
        self.sequence += 1
        observation = self._observe()
        collision = dict(self.raw.get_collision_info())
        return HelsinkiPlatformFeedback(
            observation=observation,
            factual_action=dict(factual),
            collision=collision,
        )
    def publish_policy_visualization(self, payload: Mapping[str, Any]) -> None:
        self.raw.publish_policy_visualization(dict(payload))

    def start_synchronized_recording(self, *args, **kwargs) -> Any:
        return self.raw.start_synchronized_recording(*args, **kwargs)

    def stop_synchronized_recording(self) -> Any:
        return self.raw.stop_synchronized_recording()

    def close(self) -> None:
        try:
            self.raw.close()
        finally:
            self.active = False

    def _observe(self) -> HelsinkiPlatformObservation:
        frame = self.raw.get_depth()
        kinematics = self.raw.get_kinematics()
        timestamp = float(frame.timestamp)
        if self.last_timestamp_s is not None and timestamp <= self.last_timestamp_s:
            raise RuntimeError("Helsinki feedback timestamp 未前进")
        self.last_timestamp_s = timestamp
        return HelsinkiPlatformObservation(
            sequence=self.sequence,
            timestamp_s=timestamp,
            frame=frame,
            kinematics=kinematics,
        )
