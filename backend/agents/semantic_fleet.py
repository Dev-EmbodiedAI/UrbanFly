"""Validated semantic-event and small-fleet coordination layer.

The module implements the slow, auditable half of a hierarchical UAV agent:

* a VLM (Qwen-VL through an OpenAI-compatible local endpoint) proposes events;
* a deterministic gate validates schema, freshness, confidence and provenance;
* a deterministic coordinator reallocates tasks for a 3--5 UAV fleet;
* the existing controller/local policy remains the only component allowed to
  produce high-rate flight commands.

No model weights are downloaded by this module.  A deterministic interpreter
is provided for repeatable simulation and CI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
import os
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib import request
import uuid

import numpy as np
from scipy.optimize import linear_sum_assignment


class SemanticEventType(str, Enum):
    TEMPORARY_OBSTACLE = "temporary_obstacle"
    NO_FLY_ZONE = "no_fly_zone"
    WEATHER_HAZARD = "weather_hazard"
    DRONE_FAILURE = "drone_failure"
    GOAL_LANDMARK = "goal_landmark"


_HAZARD_TYPES = {
    SemanticEventType.TEMPORARY_OBSTACLE,
    SemanticEventType.NO_FLY_ZONE,
    SemanticEventType.WEATHER_HAZARD,
}


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain finite values")
    return vector


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    event_id: str
    event_type: SemanticEventType
    timestamp_s: float
    source_drone_id: str
    position: tuple[float, float, float]
    radius_m: float
    confidence: float
    severity: float
    ttl_s: float
    evidence: str
    affected_task_ids: tuple[str, ...] = ()
    source: str = "vlm"
    temporal_support_count: int = 0
    depth_support: bool = False
    telemetry_support: bool = False
    authoritative_notice: bool = False

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        default_timestamp_s: float,
        default_source_drone_id: str,
        default_source: str = "vlm",
    ) -> "SemanticEvent":
        event_type = SemanticEventType(str(payload["event_type"]))
        return cls(
            event_id=str(payload.get("event_id") or uuid.uuid4().hex[:12]),
            event_type=event_type,
            timestamp_s=float(payload.get("timestamp_s", default_timestamp_s)),
            source_drone_id=str(
                payload.get("source_drone_id", default_source_drone_id)
            ),
            position=_vec3(payload.get("position", (0.0, 0.0, 0.0)), "position"),
            radius_m=float(payload.get("radius_m", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            severity=float(payload.get("severity", 0.0)),
            ttl_s=float(payload.get("ttl_s", 30.0)),
            evidence=str(payload.get("evidence", "")).strip(),
            affected_task_ids=tuple(
                str(item) for item in payload.get("affected_task_ids", ())
            ),
            source=str(payload.get("source", default_source)),
            temporal_support_count=int(payload.get("temporal_support_count", 0)),
            depth_support=bool(payload.get("depth_support", False)),
            telemetry_support=bool(payload.get("telemetry_support", False)),
            authoritative_notice=bool(payload.get("authoritative_notice", False)),
        )

    @property
    def expires_at_s(self) -> float:
        return self.timestamp_s + self.ttl_s

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["event_type"] = self.event_type.value
        result["position"] = list(self.position)
        result["affected_task_ids"] = list(self.affected_task_ids)
        return result


@dataclass(frozen=True, slots=True)
class ObservationPacket:
    timestamp_s: float
    drone_id: str
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    semantic_cues: tuple[Mapping[str, Any], ...] = ()
    frame_data_urls: tuple[str, ...] = ()


class SemanticInterpreter(Protocol):
    def analyze(self, packet: ObservationPacket) -> list[SemanticEvent]: ...


class DeterministicSemanticInterpreter:
    """Turn simulator/perception cues into events without a network or VLM."""

    def analyze(self, packet: ObservationPacket) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        for cue in packet.semantic_cues:
            events.append(
                SemanticEvent.from_mapping(
                    cue,
                    default_timestamp_s=packet.timestamp_s,
                    default_source_drone_id=packet.drone_id,
                    default_source="simulator",
                )
            )
        return events


class OpenAICompatibleQwenVLClient:
    """Qwen-VL semantic client for an OpenAI-compatible API endpoint.

    The endpoint is expected to expose ``/v1/chat/completions``.  Only semantic
    proposals are returned; generated text can never enter the flight-control
    command path.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        model: str = "qwen3-vl-plus",
        api_key: str | None = None,
        timeout_s: float = 8.0,
    ) -> None:
        self.endpoint = (
            endpoint or os.environ.get("URBANFLY_QWEN_ENDPOINT", "")
        ).rstrip("/")
        self.model = model
        self.api_key = (
            api_key
            or os.environ.get("URBANFLY_QWEN_API_KEY", "")
            or os.environ.get("DASHSCOPE_API_KEY", "")
        )
        self.timeout_s = float(timeout_s)

    def analyze(self, packet: ObservationPacket) -> list[SemanticEvent]:
        if not self.endpoint:
            raise RuntimeError(
                "Qwen endpoint is not configured; set URBANFLY_QWEN_ENDPOINT"
            )
        content: list[dict[str, Any]] = []
        for image_url in packet.frame_data_urls[-4:]:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append(
            {
                "type": "text",
                "text": self._prompt(packet),
            }
        )
        body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        completion_url = (
            f"{self.endpoint}/chat/completions"
            if self.endpoint.endswith("/v1")
            else f"{self.endpoint}/v1/chat/completions"
        )
        req = request.Request(
            completion_url,
            data=encoded,
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        with request.urlopen(req, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        message = payload["choices"][0]["message"]["content"]
        raw = self._content_text(message)
        parsed = self._parse_json_object(raw)
        proposals = parsed.get("events", [])
        if not isinstance(proposals, list):
            raise ValueError("Qwen response field 'events' must be a list")
        events = [
            SemanticEvent.from_mapping(
                item,
                default_timestamp_s=packet.timestamp_s,
                default_source_drone_id=packet.drone_id,
            )
            for item in proposals
        ]
        self.last_latency_ms = elapsed_ms
        return events

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping)
            )
        raise ValueError("Unsupported Qwen response content")

    @staticmethod
    def _parse_json_object(raw: str) -> Mapping[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, Mapping):
            raise ValueError("Qwen response must be a JSON object")
        return parsed

    @staticmethod
    def _prompt(packet: ObservationPacket) -> str:
        telemetry = json.dumps(packet.telemetry, ensure_ascii=False, separators=(",", ":"))
        return (
            "You are the slow semantic safety layer of a UAV fleet. Inspect up to four "
            "time-ordered frames and telemetry. Detect only clearly supported NEW events. "
            "Static buildings, trees, roads, parked objects, shadows, and ordinary depth "
            "edges are not temporary obstacles. Require corroborating change across at "
            "least two frames, an authoritative notice, or matching health/weather telemetry. "
            "Never output motor, attitude, velocity, waypoint, or free-form flight commands. "
            "Return one JSON object with key events and no markdown. Each event must contain event_type "
            "(temporary_obstacle|no_fly_zone|weather_hazard|drone_failure|goal_landmark), "
            "position [east,up,north] metres, numeric radius_m, numeric confidence [0,1], "
            "numeric severity [0,1], numeric "
            "ttl_s, evidence, and affected_task_ids. Independent temporal/depth/telemetry "
            "support is computed by the deterministic backend and cannot be self-asserted. "
            "Use events=[] when evidence is weak. "
            f"Observer={packet.drone_id}; timestamp_s={packet.timestamp_s:.3f}; "
            f"telemetry={telemetry}"
        )


@dataclass(frozen=True, slots=True)
class GateDecision:
    accepted: bool
    reason: str
    event: SemanticEvent


class SemanticEventGate:
    """Fail-closed validation for VLM and simulator event proposals."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.70,
        maximum_age_s: float = 3.0,
        maximum_radius_m: float = 500.0,
        maximum_ttl_s: float = 300.0,
    ) -> None:
        self.minimum_confidence = float(minimum_confidence)
        self.maximum_age_s = float(maximum_age_s)
        self.maximum_radius_m = float(maximum_radius_m)
        self.maximum_ttl_s = float(maximum_ttl_s)

    def validate(
        self,
        event: SemanticEvent,
        *,
        now_s: float,
        known_drone_ids: set[str],
    ) -> GateDecision:
        numeric = (
            event.timestamp_s,
            event.radius_m,
            event.confidence,
            event.severity,
            event.ttl_s,
            *event.position,
        )
        if not all(math.isfinite(value) for value in numeric):
            return GateDecision(False, "non_finite_value", event)
        if event.source_drone_id not in known_drone_ids and event.source != "backend":
            return GateDecision(False, "unknown_source_drone", event)
        if event.timestamp_s > now_s + 0.25:
            return GateDecision(False, "future_timestamp", event)
        if now_s - event.timestamp_s > self.maximum_age_s:
            return GateDecision(False, "stale_event", event)
        if not 0.0 <= event.confidence <= 1.0:
            return GateDecision(False, "confidence_out_of_range", event)
        if event.confidence < self.minimum_confidence:
            return GateDecision(False, "low_confidence", event)
        if not 0.0 <= event.severity <= 1.0:
            return GateDecision(False, "severity_out_of_range", event)
        if not 0.0 < event.ttl_s <= self.maximum_ttl_s:
            return GateDecision(False, "invalid_ttl", event)
        if event.event_type in _HAZARD_TYPES:
            if not 0.1 <= event.radius_m <= self.maximum_radius_m:
                return GateDecision(False, "invalid_hazard_radius", event)
            if not event.evidence:
                return GateDecision(False, "missing_evidence", event)
        if event.source == "vlm":
            if event.event_type == SemanticEventType.TEMPORARY_OBSTACLE and not (
                event.temporal_support_count >= 2 and event.depth_support
            ):
                return GateDecision(False, "insufficient_rgbd_temporal_support", event)
            if event.event_type == SemanticEventType.GOAL_LANDMARK and not (
                event.temporal_support_count >= 2 and event.depth_support
            ):
                return GateDecision(False, "insufficient_landmark_support", event)
            if (
                event.event_type == SemanticEventType.NO_FLY_ZONE
                and not event.authoritative_notice
            ):
                return GateDecision(False, "missing_authoritative_notice", event)
            if event.event_type in (
                SemanticEventType.WEATHER_HAZARD,
                SemanticEventType.DRONE_FAILURE,
            ) and not event.telemetry_support:
                return GateDecision(False, "missing_telemetry_support", event)
        return GateDecision(True, "accepted", event)


@dataclass(frozen=True, slots=True)
class FleetDrone:
    drone_id: str
    position: tuple[float, float, float]
    battery_pct: float
    max_payload_kg: float
    available: bool = True
    comm_neighbor_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vec3(self.position, "position"))


@dataclass(frozen=True, slots=True)
class FleetTask:
    task_id: str
    pickup: tuple[float, float, float]
    delivery: tuple[float, float, float]
    payload_kg: float
    priority: int
    required_comms: bool = False
    assigned_to: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pickup", _vec3(self.pickup, "pickup"))
        object.__setattr__(self, "delivery", _vec3(self.delivery, "delivery"))


@dataclass(frozen=True, slots=True)
class FleetPlan:
    generated_at_s: float
    assignments: Mapping[str, tuple[str, ...]]
    blocked_task_ids: tuple[str, ...]
    replan_task_ids: tuple[str, ...]
    active_event_ids: tuple[str, ...]
    total_cost: float
    reasons: Mapping[str, str]
    task_routes: Mapping[str, tuple[tuple[float, float, float], ...]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_s": self.generated_at_s,
            "assignments": {key: list(value) for key, value in self.assignments.items()},
            "blocked_task_ids": list(self.blocked_task_ids),
            "replan_task_ids": list(self.replan_task_ids),
            "active_event_ids": list(self.active_event_ids),
            "total_cost": self.total_cost,
            "reasons": dict(self.reasons),
            "task_routes": {
                task_id: [list(point) for point in route]
                for task_id, route in self.task_routes.items()
            },
        }


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    delta = end - start
    norm_squared = float(np.dot(delta, delta))
    if norm_squared <= 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, delta) / norm_squared)
    projection = start + np.clip(ratio, 0.0, 1.0) * delta
    return float(np.linalg.norm(point - projection))


def _horizontal_point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    return _point_segment_distance(point[[0, 2]], start[[0, 2]], end[[0, 2]])


class FleetCoordinator:
    """Deterministic, hazard-aware assignment for a small UAV fleet."""

    _HARD_COST = 1.0e12

    def __init__(self, *, max_tasks_per_drone: int = 3) -> None:
        self.max_tasks_per_drone = int(max_tasks_per_drone)

    def allocate(
        self,
        drones: Sequence[FleetDrone],
        tasks: Sequence[FleetTask],
        events: Sequence[SemanticEvent],
        *,
        now_s: float,
    ) -> FleetPlan:
        drones = sorted(drones, key=lambda item: item.drone_id)
        tasks = sorted(tasks, key=lambda item: (item.priority, item.task_id))
        active_events = sorted(
            (event for event in events if event.expires_at_s >= now_s),
            key=lambda item: item.event_id,
        )
        assignments: dict[str, list[str]] = {drone.drone_id: [] for drone in drones}
        if not drones or not tasks:
            return FleetPlan(
                now_s,
                {key: tuple(value) for key, value in assignments.items()},
                tuple(task.task_id for task in tasks),
                (),
                tuple(event.event_id for event in active_events),
                0.0,
                {},
                {},
            )

        slots = [
            (drone, slot)
            for drone in drones
            for slot in range(self.max_tasks_per_drone)
        ]
        costs = np.full((len(slots), len(tasks)), self._HARD_COST, dtype=float)
        replan_tasks: set[str] = set()
        reasons: dict[str, str] = {}
        candidate_routes: dict[tuple[int, int], tuple[tuple[float, float, float], ...]] = {}
        for row, (drone, slot_index) in enumerate(slots):
            for column, task in enumerate(tasks):
                cost, replan, reason, route = self._assignment_cost(
                    drone, task, active_events, slot_index
                )
                costs[row, column] = cost
                candidate_routes[(row, column)] = route
                if replan:
                    replan_tasks.add(task.task_id)
                if reason:
                    reasons[f"{drone.drone_id}:{task.task_id}"] = reason

        row_indices, column_indices = linear_sum_assignment(costs)
        assigned_tasks: set[str] = set()
        task_routes: dict[str, tuple[tuple[float, float, float], ...]] = {}
        total_cost = 0.0
        for row, column in zip(row_indices.tolist(), column_indices.tolist()):
            cost = float(costs[row, column])
            task = tasks[column]
            if cost >= self._HARD_COST:
                continue
            drone = slots[row][0]
            assignments[drone.drone_id].append(task.task_id)
            assigned_tasks.add(task.task_id)
            task_routes[task.task_id] = candidate_routes[(row, column)]
            total_cost += cost
        blocked = tuple(task.task_id for task in tasks if task.task_id not in assigned_tasks)
        return FleetPlan(
            generated_at_s=float(now_s),
            assignments={key: tuple(value) for key, value in assignments.items()},
            blocked_task_ids=blocked,
            replan_task_ids=tuple(sorted(replan_tasks & assigned_tasks)),
            active_event_ids=tuple(event.event_id for event in active_events),
            total_cost=round(total_cost, 3),
            reasons=reasons,
            task_routes=task_routes,
        )

    def _assignment_cost(
        self,
        drone: FleetDrone,
        task: FleetTask,
        events: Sequence[SemanticEvent],
        slot_index: int,
    ) -> tuple[float, bool, str, tuple[tuple[float, float, float], ...]]:
        if not drone.available or drone.battery_pct < 0.15:
            return self._HARD_COST, False, "drone_unavailable", ()
        if task.payload_kg > drone.max_payload_kg:
            return self._HARD_COST, False, "payload_infeasible", ()
        if task.required_comms and drone.comm_neighbor_count < 1:
            return self._HARD_COST, False, "communication_infeasible", ()

        start = np.asarray(drone.position, dtype=float)
        pickup = np.asarray(task.pickup, dtype=float)
        delivery = np.asarray(task.delivery, dtype=float)
        if self._point_in_no_fly_zone(start, events):
            return self._HARD_COST, True, "drone_inside_no_fly_zone", ()
        route: list[tuple[float, float, float]] = [drone.position]
        route.extend(self._dynamic_detours(start, pickup, events))
        route.append(task.pickup)
        route.extend(self._dynamic_detours(pickup, delivery, events))
        route.append(task.delivery)
        if self._endpoint_in_no_fly_zone(task, events):
            return self._HARD_COST, True, "task_endpoint_inside_no_fly_zone", ()
        distance = sum(
            float(np.linalg.norm(np.asarray(route[index + 1]) - np.asarray(route[index])))
            for index in range(len(route) - 1)
        )
        cost = distance * (1.0 + 0.20 * slot_index)
        cost += (1.0 - float(np.clip(drone.battery_pct, 0.0, 1.0))) * 120.0
        cost += max(0, task.priority) * 2.0
        # Mission continuity is a first-class cost.  A new semantic event may
        # justify moving a task, but ordinary position changes must not cause
        # repeated assignment swaps while aircraft are already flying.
        if task.assigned_to:
            if task.assigned_to == drone.drone_id:
                cost = max(0.0, cost - 240.0)
            else:
                cost += 240.0

        replan = False
        hazard_reasons: list[str] = []
        for event in events:
            if event.event_type == SemanticEventType.DRONE_FAILURE:
                if event.source_drone_id == drone.drone_id:
                    return self._HARD_COST, False, "reported_drone_failure", ()
                continue
            if event.event_type not in _HAZARD_TYPES:
                continue
            center = np.asarray(event.position, dtype=float)
            distance_fn = (
                _horizontal_point_segment_distance
                if event.event_type == SemanticEventType.NO_FLY_ZONE
                else _point_segment_distance
            )
            intersects = (
                distance_fn(center, start, pickup) <= event.radius_m
                or distance_fn(center, pickup, delivery) <= event.radius_m
                or task.task_id in event.affected_task_ids
            )
            if not intersects:
                continue
            if event.event_type == SemanticEventType.NO_FLY_ZONE:
                replan = True
                cost += 80.0 * max(0.1, event.severity)
                hazard_reasons.append(event.event_type.value)
                continue
            replan = True
            scale = 450.0 if event.event_type == SemanticEventType.TEMPORARY_OBSTACLE else 260.0
            cost += scale * max(0.1, event.severity)
            hazard_reasons.append(event.event_type.value)
        return cost, replan, ",".join(hazard_reasons), tuple(route)

    @staticmethod
    def _point_in_no_fly_zone(
        point: np.ndarray, events: Sequence[SemanticEvent]
    ) -> bool:
        horizontal = np.asarray(point, dtype=float)[[0, 2]]
        for event in events:
            if event.event_type not in (
                SemanticEventType.NO_FLY_ZONE,
                SemanticEventType.TEMPORARY_OBSTACLE,
            ):
                continue
            center = np.asarray(event.position, dtype=float)[[0, 2]]
            if float(np.linalg.norm(horizontal - center)) <= event.radius_m:
                return True
        return False

    @staticmethod
    def _endpoint_in_no_fly_zone(
        task: FleetTask, events: Sequence[SemanticEvent]
    ) -> bool:
        for event in events:
            if event.event_type != SemanticEventType.NO_FLY_ZONE:
                continue
            center = np.asarray(event.position, dtype=float)[[0, 2]]
            for endpoint in (task.pickup, task.delivery):
                if float(np.linalg.norm(np.asarray(endpoint, dtype=float)[[0, 2]] - center)) <= event.radius_m:
                    return True
        return False

    @staticmethod
    def _dynamic_detours(
        start: np.ndarray,
        end: np.ndarray,
        events: Sequence[SemanticEvent],
        *,
        clearance_m: float = 5.0,
    ) -> list[tuple[float, float, float]]:
        """Create deterministic dog-leg waypoints around cylindrical no-fly zones.

        This is a high-level constraint adapter, not a replacement for the
        frozen collision-aware planner.  The resulting waypoints must still be
        checked against the static map by that planner.
        """
        detours: list[tuple[float, float, float]] = []
        leg_start = np.asarray(start, dtype=float)
        leg_end = np.asarray(end, dtype=float)
        for event in sorted(events, key=lambda item: item.event_id):
            if event.event_type != SemanticEventType.NO_FLY_ZONE:
                continue
            center = np.asarray(event.position, dtype=float)
            distance = (
                _horizontal_point_segment_distance(center, leg_start, leg_end)
                if event.event_type == SemanticEventType.NO_FLY_ZONE
                else _point_segment_distance(center, leg_start, leg_end)
            )
            if distance > event.radius_m:
                continue
            horizontal = leg_end[[0, 2]] - leg_start[[0, 2]]
            norm = float(np.linalg.norm(horizontal))
            if norm <= 1e-9:
                continue
            direction = horizontal / norm
            normal = np.array([-direction[1], direction[0]])
            offset = event.radius_m + clearance_m
            center_2d = center[[0, 2]]
            before_2d = center_2d - direction * offset
            after_2d = center_2d + direction * offset
            candidates: list[tuple[float, list[tuple[float, float, float]]]] = []
            altitude = max(float(leg_start[1]), float(leg_end[1]))
            for sign in (-1.0, 1.0):
                side = normal * offset * sign
                points_2d = (before_2d + side, after_2d + side)
                points = [
                    (float(point[0]), altitude, float(point[1]))
                    for point in points_2d
                ]
                length = float(np.linalg.norm(np.asarray(points[0]) - leg_start))
                length += float(np.linalg.norm(np.asarray(points[1]) - np.asarray(points[0])))
                length += float(np.linalg.norm(leg_end - np.asarray(points[1])))
                candidates.append((length, points))
            _, chosen = min(candidates, key=lambda item: (item[0], item[1][0]))
            detours.extend(chosen)
        return detours


class SemanticFleetRuntime:
    """Stateful, auditable composition of interpreter, gate and coordinator."""

    def __init__(
        self,
        interpreter: SemanticInterpreter,
        *,
        gate: SemanticEventGate | None = None,
        coordinator: FleetCoordinator | None = None,
        audit_limit: int = 256,
    ) -> None:
        self.interpreter = interpreter
        self.gate = gate or SemanticEventGate()
        self.coordinator = coordinator or FleetCoordinator()
        self.audit_limit = int(audit_limit)
        self.active_events: dict[str, SemanticEvent] = {}
        self.audit: list[dict[str, Any]] = []
        self.last_plan: FleetPlan | None = None

    def ingest(
        self,
        packet: ObservationPacket,
        *,
        known_drone_ids: set[str],
        now_s: float | None = None,
    ) -> list[GateDecision]:
        validation_time_s = packet.timestamp_s if now_s is None else float(now_s)
        decisions: list[GateDecision] = []
        for event in self.interpreter.analyze(packet):
            decision = self.gate.validate(
                event,
                now_s=validation_time_s,
                known_drone_ids=known_drone_ids,
            )
            decisions.append(decision)
            if decision.accepted:
                self.active_events[event.event_id] = event
            self._append_audit(
                {
                    "kind": "event_gate",
                    "timestamp_s": validation_time_s,
                    "observation_timestamp_s": packet.timestamp_s,
                    "event_id": event.event_id,
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                }
            )
        self.expire(validation_time_s)
        return decisions

    def expire(self, now_s: float) -> None:
        expired = [
            event_id
            for event_id, event in self.active_events.items()
            if event.expires_at_s < now_s
        ]
        for event_id in expired:
            del self.active_events[event_id]
            self._append_audit(
                {"kind": "event_expired", "timestamp_s": now_s, "event_id": event_id}
            )

    def reallocate(
        self,
        drones: Sequence[FleetDrone],
        tasks: Sequence[FleetTask],
        *,
        now_s: float,
    ) -> FleetPlan:
        self.expire(now_s)
        self.last_plan = self.coordinator.allocate(
            drones,
            tasks,
            list(self.active_events.values()),
            now_s=now_s,
        )
        self._append_audit(
            {
                "kind": "fleet_plan",
                "timestamp_s": now_s,
                "assignments": {
                    key: list(value)
                    for key, value in self.last_plan.assignments.items()
                },
                "blocked_task_ids": list(self.last_plan.blocked_task_ids),
            }
        )
        return self.last_plan

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_events": [
                event.to_dict()
                for event in sorted(
                    self.active_events.values(), key=lambda item: item.event_id
                )
            ],
            "last_plan": self.last_plan.to_dict() if self.last_plan else None,
            "audit_tail": self.audit[-20:],
            "control_authority": "semantic_only",
        }

    def _append_audit(self, entry: Mapping[str, Any]) -> None:
        self.audit.append(dict(entry))
        if len(self.audit) > self.audit_limit:
            del self.audit[: len(self.audit) - self.audit_limit]
