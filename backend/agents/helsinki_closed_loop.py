"""Agent/World-Model/Helsinki 闭环的可审计运行时状态机。

本模块只负责慢速任务语义、动作授权和因果反馈审计。高频动作仍由已训练
policy 提议，World Model 只在局部候选中重排，最终执行与独立安全门仍由
Helsinki adapter/backend 负责。这样 Agent 不会用自由文本绕过控制链。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np


class ClosedLoopViolation(RuntimeError):
    """闭环因果关系、数值或安全门被破坏。"""


class AgentStatus(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


def _finite_vector(value: Sequence[float], *, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ClosedLoopViolation(f"{name} must be a finite vector with shape ({size},)")
    return vector


@dataclass(frozen=True, slots=True)
class SemanticMissionPlan:
    """通过确定性 gate 后，Agent 可以使用的低频语义任务计划。"""

    provider: str
    model: str
    api_called: bool
    waypoint_order: tuple[str, ...]
    ordered_waypoints_backend: tuple[tuple[float, float, float], ...]
    control_authority: str = "semantic waypoint ordering only; no flight actions"

    @classmethod
    def from_report(cls, payload: Mapping[str, Any]) -> "SemanticMissionPlan":
        if payload.get("status") != "PASS":
            raise ClosedLoopViolation("semantic mission report has not passed")
        if payload.get("deterministic_gate") != "PASS":
            raise ClosedLoopViolation("semantic mission deterministic gate has not passed")
        authority = str(payload.get("control_authority", ""))
        if "no flight actions" not in authority.lower():
            raise ClosedLoopViolation("semantic mission authority is not safely bounded")
        order_raw = payload.get("waypoint_order")
        points_raw = payload.get("ordered_waypoints_backend")
        if not isinstance(order_raw, list) or not isinstance(points_raw, list):
            raise ClosedLoopViolation("semantic mission order/waypoints are missing")
        order = tuple(str(item) for item in order_raw)
        if not order or len(order) != len(set(order)) or len(order) != len(points_raw):
            raise ClosedLoopViolation("semantic mission waypoint order is not unique/complete")
        points = tuple(
            tuple(float(item) for item in _finite_vector(point, size=3, name="semantic waypoint"))
            for point in points_raw
        )
        return cls(
            provider=str(payload.get("provider", "unknown")),
            model=str(payload.get("model", "unknown")),
            api_called=bool(payload.get("api_called", False)),
            waypoint_order=order,
            ordered_waypoints_backend=points,
            control_authority=authority,
        )

    @classmethod
    def deterministic(
        cls, ordered_waypoints_backend: Sequence[Sequence[float]]
    ) -> "SemanticMissionPlan":
        points = tuple(
            tuple(float(item) for item in _finite_vector(point, size=3, name="waypoint"))
            for point in ordered_waypoints_backend
        )
        return cls(
            provider="deterministic_mission_input",
            model="none",
            api_called=False,
            waypoint_order=tuple(f"W{index + 1}" for index in range(len(points))),
            ordered_waypoints_backend=points,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["waypoint_order"] = list(self.waypoint_order)
        result["ordered_waypoints_backend"] = [list(point) for point in self.ordered_waypoints_backend]
        return result


@dataclass(frozen=True, slots=True)
class WorldModelActionDecision:
    step: int
    selected_index: int
    candidate_count: int
    predicted_risk: float
    uncertainty: float
    action_body_flu: tuple[float, float, float, float]

    @classmethod
    def create(
        cls,
        *,
        step: int,
        selected_index: int,
        candidate_count: int,
        predicted_risk: float,
        uncertainty: float,
        action_body_flu: Sequence[float],
    ) -> "WorldModelActionDecision":
        action = _finite_vector(action_body_flu, size=4, name="World Model action")
        risk = float(predicted_risk)
        spread = float(uncertainty)
        if step < 0 or candidate_count <= 0 or not 0 <= selected_index < candidate_count:
            raise ClosedLoopViolation("invalid World Model candidate selection")
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ClosedLoopViolation("predicted risk must be finite and in [0, 1]")
        if not math.isfinite(spread) or spread < 0.0:
            raise ClosedLoopViolation("World Model uncertainty must be finite and non-negative")
        return cls(
            step=int(step),
            selected_index=int(selected_index),
            candidate_count=int(candidate_count),
            predicted_risk=risk,
            uncertainty=spread,
            action_body_flu=tuple(float(item) for item in action),
        )


@dataclass(frozen=True, slots=True)
class AgentDirective:
    step: int
    status: AgentStatus
    active_waypoint_index: int
    active_waypoint_enu: tuple[float, float, float]
    reason: str


class HelsinkiAgentWorldModelRuntime:
    """强制执行 Agent→WM→动作→Helsinki→新观测的单步因果顺序。"""

    def __init__(
        self,
        mission: SemanticMissionPlan,
        mission_waypoints_enu: Sequence[Sequence[float]],
        *,
        semantic_waypoint_tolerance_m: float = 25.0,
        final_goal_tolerance_m: float = 3.0,
        audit_limit: int = 96,
    ) -> None:
        self.mission = mission
        self.waypoints = tuple(
            _finite_vector(point, size=3, name="mission waypoint ENU")
            for point in mission_waypoints_enu
        )
        if not self.waypoints:
            raise ValueError("at least one mission waypoint is required")
        if semantic_waypoint_tolerance_m <= 0 or final_goal_tolerance_m <= 0:
            raise ValueError("waypoint tolerances must be positive")
        self.semantic_waypoint_tolerance_m = float(semantic_waypoint_tolerance_m)
        self.final_goal_tolerance_m = float(final_goal_tolerance_m)
        self.audit_limit = int(audit_limit)
        self.status = AgentStatus.READY
        self.active_waypoint_index = 0
        self.current_step = 0
        self.last_observation_timestamp_s: float | None = None
        self.last_observation_position_enu: np.ndarray | None = None
        self.pending_decision: WorldModelActionDecision | None = None
        self.agent_observations = 0
        self.world_model_decisions = 0
        self.actions_authorized = 0
        self.executions = 0
        self.fresh_feedbacks = 0
        self.safety_interventions = 0
        self.waypoints_reached = 0
        self.abort_reason: str | None = None
        self.audit: list[dict[str, Any]] = []

    def begin(
        self, *, observation_timestamp_s: float, position_enu: Sequence[float]
    ) -> AgentDirective:
        if self.status is not AgentStatus.READY:
            raise ClosedLoopViolation("closed loop has already started")
        timestamp = self._timestamp(observation_timestamp_s, "initial observation")
        self.last_observation_timestamp_s = timestamp
        self.last_observation_position_enu = _finite_vector(
            position_enu, size=3, name="initial position"
        )
        self.agent_observations = 1
        self.status = AgentStatus.RUNNING
        self._advance_waypoints(self.last_observation_position_enu)
        self._append_audit("agent_observation", step=0, timestamp_s=timestamp, initial=True)
        return self.directive("initial observation accepted")

    def authorize_world_model(self, decision: WorldModelActionDecision) -> AgentDirective:
        if self.status is not AgentStatus.RUNNING:
            raise ClosedLoopViolation(f"Agent cannot authorize action while {self.status.value}")
        if decision.step != self.current_step:
            return self._abort_and_raise(
                f"World Model step mismatch: expected {self.current_step}, got {decision.step}"
            )
        if self.pending_decision is not None:
            return self._abort_and_raise("more than one World Model decision in one cycle")
        self.pending_decision = decision
        self.world_model_decisions += 1
        self.actions_authorized += 1
        self._append_audit(
            "world_model_action_authorized",
            step=decision.step,
            selected_index=decision.selected_index,
            candidate_count=decision.candidate_count,
            predicted_risk=decision.predicted_risk,
            uncertainty=decision.uncertainty,
        )
        return self.directive("gated World Model action authorized")

    def accept_execution_feedback(
        self,
        *,
        step: int,
        feedback_timestamp_s: float,
        position_enu: Sequence[float],
        executed_action_body_flu: Sequence[float],
        stale_action: bool,
        collision: bool,
        safety_intervened: bool,
    ) -> AgentDirective:
        if self.status is not AgentStatus.RUNNING or self.pending_decision is None:
            raise ClosedLoopViolation("execution feedback arrived without an authorized action")
        if step != self.current_step or self.pending_decision.step != step:
            return self._abort_and_raise("execution feedback step does not match authorization")
        timestamp = self._timestamp(feedback_timestamp_s, "execution feedback")
        if timestamp <= float(self.last_observation_timestamp_s):
            return self._abort_and_raise("feedback timestamp is not newer than its action observation")
        position = _finite_vector(position_enu, size=3, name="feedback position")
        _finite_vector(executed_action_body_flu, size=4, name="executed action")
        if stale_action:
            return self._abort_and_raise("stale action in Helsinki execution feedback")
        if collision:
            return self._abort_and_raise("collision in Helsinki execution feedback")
        self.executions += 1
        self.fresh_feedbacks += 1
        self.agent_observations += 1
        self.safety_interventions += int(bool(safety_intervened))
        self.pending_decision = None
        self.last_observation_timestamp_s = timestamp
        self.last_observation_position_enu = position
        self.current_step += 1
        self._advance_waypoints(position)
        self._append_audit(
            "fresh_execution_feedback",
            step=step,
            timestamp_s=timestamp,
            safety_intervened=bool(safety_intervened),
            active_waypoint_index=self.active_waypoint_index,
        )
        return self.directive("fresh Helsinki observation fed back to Agent and next policy step")

    def directive(self, reason: str) -> AgentDirective:
        index = min(self.active_waypoint_index, len(self.waypoints) - 1)
        return AgentDirective(
            step=self.current_step,
            status=self.status,
            active_waypoint_index=index,
            active_waypoint_enu=tuple(float(item) for item in self.waypoints[index]),
            reason=reason,
        )

    def snapshot(self) -> dict[str, Any]:
        causal_complete = (
            self.pending_decision is None
            and self.world_model_decisions == self.actions_authorized
            and self.actions_authorized == self.executions
            and self.executions == self.fresh_feedbacks
            and self.agent_observations == self.fresh_feedbacks + 1
        )
        return {
            "schema": "urbanfly-agent-world-model-helsinki-loop-v1",
            "status": self.status.value,
            "abort_reason": self.abort_reason,
            "mission": self.mission.to_dict(),
            "control_authority": {
                "agent": "semantic mission sequencing and fail-closed action authorization",
                "policy": "high-rate local action proposal",
                "world_model": "bounded local candidate reranking",
                "executor": "Helsinki 6DoF adapter plus independent backend safety shield",
            },
            "mission_waypoints": len(self.waypoints),
            "waypoints_reached": self.waypoints_reached,
            "active_waypoint_index": min(self.active_waypoint_index, len(self.waypoints) - 1),
            "agent_observations": self.agent_observations,
            "world_model_decisions": self.world_model_decisions,
            "actions_authorized": self.actions_authorized,
            "executions": self.executions,
            "fresh_feedbacks": self.fresh_feedbacks,
            "feedback_to_next_policy": max(0, self.fresh_feedbacks - int(self.status is AgentStatus.COMPLETE)),
            "safety_interventions": self.safety_interventions,
            "causal_chain_complete": causal_complete,
            "audit_tail": list(self.audit),
        }

    def abort(self, reason: str) -> None:
        """由外部独立安全门将本轮闭环标记为失败并保存原因。"""
        if self.status is AgentStatus.COMPLETE:
            raise ClosedLoopViolation("a completed mission cannot be aborted")
        self.status = AgentStatus.ABORTED
        self.abort_reason = str(reason)
        self.pending_decision = None
        self._append_audit("agent_abort", step=self.current_step, reason=self.abort_reason)

    def _advance_waypoints(self, position: np.ndarray) -> None:
        while self.active_waypoint_index < len(self.waypoints):
            is_final = self.active_waypoint_index == len(self.waypoints) - 1
            tolerance = (
                self.final_goal_tolerance_m if is_final else self.semantic_waypoint_tolerance_m
            )
            distance = float(np.linalg.norm(position - self.waypoints[self.active_waypoint_index]))
            if distance > tolerance:
                break
            reached = self.active_waypoint_index
            self.active_waypoint_index += 1
            self.waypoints_reached += 1
            self._append_audit(
                "agent_waypoint_reached", waypoint_index=reached, distance_m=distance
            )
            if is_final:
                self.status = AgentStatus.COMPLETE
                break

    @staticmethod
    def _timestamp(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ClosedLoopViolation(f"{name} timestamp must be finite")
        return result

    def _abort_and_raise(self, reason: str):
        self.abort(reason)
        raise ClosedLoopViolation(reason)

    def _append_audit(self, kind: str, **fields: Any) -> None:
        self.audit.append({"kind": kind, **fields})
        if len(self.audit) > self.audit_limit:
            del self.audit[: len(self.audit) - self.audit_limit]
