"""Adapter between the semantic fleet runtime and the existing Simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from .semantic_fleet import (
    DeterministicSemanticInterpreter,
    FleetCoordinator,
    FleetDrone,
    FleetTask,
    ObservationPacket,
    SemanticFleetRuntime,
)
from ..engine.models import (
    DroneState,
    EventType,
    SimulationEvent,
    TaskStatus,
    Waypoint,
)
from ..engine.dynamic_actors import ScriptedDynamicActor

if TYPE_CHECKING:  # pragma: no cover
    from ..engine.simulator import Simulator


@dataclass(slots=True)
class TimelineEvent:
    time_s: float
    observer_drone_id: str
    cue: Mapping[str, Any]


class SemanticFleetSimulatorBridge:
    """Runs semantic events as a slow supervisory loop.

    It owns task assignment and mission-level detour waypoints only while a
    scenario explicitly enables ``semantic_agent``.  It never produces motor,
    attitude, body-rate, velocity, or local-policy actions.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.runtime = SemanticFleetRuntime(
            DeterministicSemanticInterpreter(),
            coordinator=FleetCoordinator(max_tasks_per_drone=1),
        )
        self.timeline: list[TimelineEvent] = []
        self.next_event_index = 0
        self.last_allocation_s = -float("inf")
        self.reallocation_interval_s = 5.0
        self.applied_plan_count = 0
        self.path_validation_failures: list[dict[str, Any]] = []
        self.failure_recoveries: list[dict[str, Any]] = []
        self.failed_drone_ids: set[str] = set()
        self.accept_external_proposals = False
        self.external_proposal_count = 0

    def configure(self, scenario_def, simulator: "Simulator") -> bool:
        config = dict(getattr(scenario_def, "semantic_agent", {}) or {})
        self.enabled = bool(config.get("enabled", False))
        self.runtime = SemanticFleetRuntime(
            DeterministicSemanticInterpreter(),
            coordinator=FleetCoordinator(
                max_tasks_per_drone=int(config.get("max_tasks_per_drone", 1))
            ),
        )
        self.reallocation_interval_s = float(
            config.get("reallocation_interval_s", 5.0)
        )
        self.accept_external_proposals = bool(
            config.get("accept_external_proposals", False)
        )
        self.timeline = []
        for raw in getattr(scenario_def, "events", []) or []:
            cue = raw.get("semantic_event")
            if not isinstance(cue, Mapping):
                continue
            observer = str(
                raw.get("observer_drone_id")
                or cue.get("source_drone_id")
                or (simulator.drones[0].id if simulator.drones else "backend")
            )
            self.timeline.append(
                TimelineEvent(float(raw.get("time_s", 0.0)), observer, dict(cue))
            )
        self.timeline.sort(key=lambda item: (item.time_s, str(item.cue.get("event_id", ""))))
        self.next_event_index = 0
        self.last_allocation_s = -float("inf")
        self.applied_plan_count = 0
        self.path_validation_failures = []
        self.failure_recoveries = []
        self.failed_drone_ids = set()
        self.external_proposal_count = 0
        return self.enabled

    def ingest_external_proposals(
        self,
        simulator: "Simulator",
        *,
        observer_drone_id: str,
        observation_timestamp_s: float,
        proposals: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Gate external VLM proposals at the simulator clock and apply accepts."""

        if not self.enabled or not self.accept_external_proposals:
            raise RuntimeError("external semantic proposals are disabled")
        if len(proposals) > 16:
            raise ValueError("at most 16 semantic proposals are accepted per observation")
        known_drone_ids = {drone.id for drone in simulator.drones}
        # Corroboration is a trusted-backend property. Never accept support
        # booleans self-asserted by generative model output.
        cues = tuple({
            **dict(item),
            "source": "vlm",
            "temporal_support_count": 0,
            "depth_support": False,
            "telemetry_support": False,
            "authoritative_notice": False,
        } for item in proposals)
        packet = ObservationPacket(
            timestamp_s=float(observation_timestamp_s),
            drone_id=str(observer_drone_id),
            telemetry={"received_at_sim_time_s": float(simulator.time)},
            semantic_cues=cues,
        )
        decisions = self.runtime.ingest(
            packet,
            known_drone_ids=known_drone_ids,
            now_s=float(simulator.time),
        )
        accepted = []
        for decision in decisions:
            self.external_proposal_count += 1
            accepted.append({
                "event_id": decision.event.event_id,
                "accepted": decision.accepted,
                "reason": decision.reason,
            })
            simulator.events.append(
                SimulationEvent(
                    time=simulator.time,
                    event_type=EventType.PATH_REPLANNED,
                    drone_id=str(observer_drone_id),
                    message=(
                        f"Qwen语义事件 {decision.event.event_type.value}: "
                        f"{'ACCEPT' if decision.accepted else 'REJECT'} "
                        f"({decision.reason})"
                    ),
                    metadata={
                        "semantic_event": decision.event.to_dict(),
                        "gate_accepted": decision.accepted,
                        "gate_reason": decision.reason,
                    },
                )
            )
            if decision.accepted:
                self._apply_event_effect(simulator, decision.event)
        if any(decision.accepted for decision in decisions):
            self._apply_weather_effect(simulator)
            self.reallocate(simulator)
        return accepted

    def update(self, simulator: "Simulator") -> None:
        if not self.enabled:
            return
        accepted_any = False
        active_before = set(self.runtime.active_events)
        known_drone_ids = {drone.id for drone in simulator.drones}
        while (
            self.next_event_index < len(self.timeline)
            and self.timeline[self.next_event_index].time_s <= simulator.time + 1e-9
        ):
            item = self.timeline[self.next_event_index]
            self.next_event_index += 1
            packet = ObservationPacket(
                timestamp_s=float(simulator.time),
                drone_id=item.observer_drone_id,
                telemetry={"sim_time_s": simulator.time},
                semantic_cues=(item.cue,),
            )
            decisions = self.runtime.ingest(
                packet,
                known_drone_ids=known_drone_ids,
            )
            for decision in decisions:
                simulator.events.append(
                    SimulationEvent(
                        time=simulator.time,
                        event_type=EventType.PATH_REPLANNED,
                        drone_id=item.observer_drone_id,
                        message=(
                            f"语义事件 {decision.event.event_type.value}: "
                            f"{'ACCEPT' if decision.accepted else 'REJECT'} "
                            f"({decision.reason})"
                        ),
                        metadata={
                            "semantic_event": decision.event.to_dict(),
                            "gate_accepted": decision.accepted,
                            "gate_reason": decision.reason,
                        },
                    )
                )
                accepted_any = accepted_any or decision.accepted
                if decision.accepted:
                    self._apply_event_effect(simulator, decision.event)
        self.runtime.expire(float(simulator.time))
        expired_any = active_before - set(self.runtime.active_events)
        self._apply_weather_effect(simulator)
        unassigned_work = any(
            task.status == TaskStatus.PENDING and task.assigned_to is None
            for task in simulator.tasks
        )
        retry_unassigned = (
            unassigned_work
            and simulator.time - self.last_allocation_s >= self.reallocation_interval_s
        )
        if accepted_any or expired_any or retry_unassigned:
            self.reallocate(simulator)

    def _apply_event_effect(self, simulator: "Simulator", event) -> None:
        if event.event_type.value == "temporary_obstacle":
            next_id = max(
                (actor.actor_id for actor in simulator.dynamic_actor_field.actors),
                default=0,
            ) + 1
            position = np.asarray(event.position, dtype=float)
            actor = ScriptedDynamicActor(
                actor_id=next_id,
                actor_type="temporary_obstacle",
                origin=position.copy(),
                direction=np.array([1.0, 0.0, 0.0]),
                speed_mps=0.05,
                half_extent=np.array([4.0, 5.0, 4.0]),
                travel_m=0.25,
                phase_s=0.0,
                zone_type="semantic_hazard",
                position=position.copy(),
                velocity=np.zeros(3),
            )
            simulator.dynamic_actor_field.actors.append(actor)
        elif event.event_type.value == "drone_failure":
            for drone in simulator.drones:
                if drone.id == event.source_drone_id:
                    self.failed_drone_ids.add(drone.id)
                    failed_task_id = drone.current_task_id
                    recovery_position = np.asarray(drone.position, dtype=float).copy()
                    carried_payload = float(drone.payload_current)
                    if failed_task_id:
                        task = next(
                            (item for item in simulator.tasks if item.id == failed_task_id),
                            None,
                        )
                        if task is not None and task.status not in (
                            TaskStatus.COMPLETED,
                            TaskStatus.CANCELLED,
                            TaskStatus.FAILED,
                        ):
                            # A replacement aircraft must recover a payload at
                            # the failed vehicle, never pretend it remained at
                            # the original depot or was transferred invisibly.
                            if carried_payload > 1e-6:
                                task.pickup_pos = recovery_position.copy()
                            task.status = TaskStatus.PENDING
                            task.assigned_to = None
                            self.failure_recoveries.append({
                                "time_s": float(simulator.time),
                                "failed_drone_id": drone.id,
                                "task_id": task.id,
                                "payload_was_onboard": carried_payload > 1e-6,
                                "recovery_pickup": recovery_position.tolist(),
                            })
                    drone.state = DroneState.EMERGENCY
                    drone.path = []
                    drone.assigned_tasks = []
                    drone.current_task_id = None
                    drone.payload_current = 0.0
                    break

    def _apply_weather_effect(self, simulator: "Simulator") -> None:
        severity = max(
            (
                event.severity
                for event in self.runtime.active_events.values()
                if event.event_type.value == "weather_hazard"
            ),
            default=0.0,
        )
        simulator._episode_wind_offset = np.array(
            [4.0 * severity, 0.0, -2.5 * severity], dtype=float
        )

    def reallocate(self, simulator: "Simulator") -> None:
        if not self.enabled:
            return
        picked_up_by = {
            drone.current_task_id: drone.id
            for drone in simulator.drones
            if drone.current_task_id and drone.payload_current > 1e-6
        }
        active_tasks = [
            task
            for task in simulator.tasks
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED)
        ]
        fleet_drones = [
            FleetDrone(
                drone_id=drone.id,
                position=tuple(float(value) for value in drone.position),
                battery_pct=float(drone.battery_pct),
                max_payload_kg=float(drone.max_payload),
                available=drone.id not in self.failed_drone_ids and drone.state not in (
                    DroneState.EMERGENCY,
                    DroneState.RETURNING,
                    DroneState.CHARGING,
                ),
                comm_neighbor_count=len(drone.comm_neighbors),
            )
            for drone in simulator.drones
        ]
        fleet_tasks = [
            FleetTask(
                task_id=task.id,
                pickup=tuple(float(value) for value in task.pickup_pos),
                delivery=tuple(float(value) for value in task.delivery_pos),
                payload_kg=float(task.payload_weight),
                priority=int(task.priority),
                required_comms=bool(task.required_comms),
                assigned_to=task.assigned_to,
            )
            for task in active_tasks
        ]
        plan = self.runtime.reallocate(
            fleet_drones,
            fleet_tasks,
            now_s=float(simulator.time),
        )
        task_by_id = {task.id: task for task in active_tasks}
        for task in active_tasks:
            task.status = TaskStatus.PENDING
            task.assigned_to = None
        for drone in simulator.drones:
            assigned = list(plan.assignments.get(drone.id, ()))
            drone.assigned_tasks = assigned
            drone.current_task_id = assigned[0] if assigned else None
            drone.current_path_index = 0
            drone.path = []
            for task_id in assigned:
                task = task_by_id[task_id]
                picked_up = picked_up_by.get(task_id) == drone.id
                task.status = (
                    TaskStatus.EN_ROUTE_DELIVERY if picked_up else TaskStatus.ASSIGNED
                )
                task.assigned_to = drone.id
                route = plan.task_routes.get(task_id, ())
                if picked_up:
                    current = np.asarray(drone.position, dtype=float)
                    delivery = np.asarray(task.delivery_pos, dtype=float)
                    detours = self.runtime.coordinator._dynamic_detours(
                        current,
                        delivery,
                        list(self.runtime.active_events.values()),
                    )
                    route = (
                        tuple(float(value) for value in current),
                        *detours,
                        tuple(float(value) for value in delivery),
                    )
                candidate = [
                    Waypoint(
                        position=np.asarray(point, dtype=float),
                        action=(
                            "delivery"
                            if np.allclose(point, task.delivery_pos, atol=1e-6)
                            else (
                                "pickup"
                                if not picked_up
                                and np.allclose(point, task.pickup_pos, atol=1e-6)
                                else None
                            )
                        ),
                        metadata={
                            "semantic_agent": True,
                            "task_id": task_id,
                            "dynamic_replan": task_id in plan.replan_task_ids,
                            "active_event_ids": list(plan.active_event_ids),
                        },
                    )
                    for point in route[1:]
                ]
                try:
                    drone.path = simulator._validate_and_repair_static_path(
                        drone, candidate
                    )
                except Exception as error:  # fail closed and preserve audit
                    drone.path = []
                    drone.assigned_tasks = []
                    drone.current_task_id = None
                    task.status = TaskStatus.PENDING
                    task.assigned_to = None
                    self.path_validation_failures.append({
                        "time_s": simulator.time,
                        "drone_id": drone.id,
                        "task_id": task_id,
                        "error": repr(error),
                    })
        self.last_allocation_s = float(simulator.time)
        self.applied_plan_count += 1
        simulator.events.append(
            SimulationEvent(
                time=simulator.time,
                event_type=EventType.TASK_ASSIGNED,
                message=(
                    f"语义协调重分配 #{self.applied_plan_count}: "
                    f"blocked={len(plan.blocked_task_ids)}, "
                    f"dynamic_replan={len(plan.replan_task_ids)}"
                ),
                metadata={"semantic_fleet_plan": plan.to_dict()},
            )
        )

    def snapshot(self) -> dict[str, Any]:
        data = self.runtime.snapshot()
        data.update({
            "enabled": self.enabled,
            "provider": "deterministic_simulator_cues",
            "timeline_total": len(self.timeline),
            "timeline_consumed": self.next_event_index,
            "applied_plan_count": self.applied_plan_count,
            "path_validation_failure_count": len(self.path_validation_failures),
            "path_validation_failures": self.path_validation_failures[-10:],
            "failure_recovery_count": len(self.failure_recoveries),
            "failure_recoveries": self.failure_recoveries[-10:],
            "failed_drone_ids": sorted(self.failed_drone_ids),
            "accept_external_proposals": self.accept_external_proposals,
            "external_proposal_count": self.external_proposal_count,
        })
        return data
