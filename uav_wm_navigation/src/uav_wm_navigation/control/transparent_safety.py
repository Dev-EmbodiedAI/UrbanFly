from __future__ import annotations

import numpy as np

from uav_wm_navigation.control.safety_filter import SafetyFilter
from uav_wm_navigation.types import (
    ActionLimits,
    BodyVelocityAction,
    SafetyAudit,
    SensorFrame,
    VehicleState,
)


def body_flu_to_world_nwu(vector: np.ndarray, yaw_rad: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.asarray(
        [
            cosine * value[0] - sine * value[1],
            sine * value[0] + cosine * value[1],
            value[2],
        ],
        dtype=np.float32,
    )


def world_nwu_to_body_flu(vector: np.ndarray, yaw_rad: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.asarray(
        [
            cosine * value[0] + sine * value[1],
            -sine * value[0] + cosine * value[1],
            value[2],
        ],
        dtype=np.float32,
    )


class TransparentSafetyLayer:
    """Auditable last-mile shield that never rewrites the policy output in place."""

    def __init__(
        self,
        enabled: bool = True,
        limits: ActionLimits | None = None,
        filter_config: dict | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.limits = limits or ActionLimits()
        config = {
            "max_speed_mps": float(
                np.hypot(self.limits.forward_mps, self.limits.lateral_mps)
            ),
            "max_yaw_rate_rps": self.limits.yaw_rate_rps,
            "max_acceleration_mps2": 4.0,
            "emergency_depth_m": 1.5,
            "slow_depth_m": 5.0,
            "clearance_percentile": 2.0,
            **(filter_config or {}),
        }
        self.filter = SafetyFilter(config)
        self.audits: list[SafetyAudit] = []

    @property
    def intervention_count(self) -> int:
        return sum(item.intervened for item in self.audits)

    def reset(self) -> None:
        self.audits.clear()
        self.filter._last_command_velocity = None
        self.filter._collision_hits = 0

    def apply(
        self,
        action: BodyVelocityAction,
        *,
        state: VehicleState,
        sensor: SensorFrame,
        yaw_rad: float,
        episode_id: str,
        step_id: int,
        sim_time: float,
        dt: float,
        predicted_risk: float = 0.0,
        collision: bool = False,
    ) -> tuple[np.ndarray, float, SafetyAudit]:
        raw = action.physical.astype(np.float32)
        raw_world = body_flu_to_world_nwu(raw[:3], yaw_rad)
        if self.enabled:
            filtered = self.filter.apply(
                raw_world,
                float(raw[3]),
                state,
                sensor,
                dt,
                predicted_risk=predicted_risk,
                collision=collision,
            )
            executed_velocity_body = world_nwu_to_body_flu(filtered.velocity, yaw_rad)
            executed = np.r_[executed_velocity_body, filtered.yaw_rate].astype(np.float32)
            reasons = tuple(filtered.reasons)
        else:
            executed = raw.copy()
            reasons = ()
        difference = float(np.linalg.norm(executed - raw))
        valid_depth = sensor.depth_m[sensor.valid_mask]
        audit = SafetyAudit(
            episode_id=episode_id,
            step_id=step_id,
            sim_time=sim_time,
            raw_action_normalized=action.normalized.copy(),
            raw_action_physical=raw.copy(),
            executed_action_physical=executed.copy(),
            intervened=bool(difference > 1e-5),
            reasons=reasons,
            action_delta_l2=difference,
            minimum_depth_m=(
                float(valid_depth.min()) if valid_depth.size else float("nan")
            ),
            predicted_risk=float(predicted_risk),
        )
        self.audits.append(audit)
        return executed[:3], float(executed[3]), audit
