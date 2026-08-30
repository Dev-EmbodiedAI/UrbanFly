"""Repeatable component-level QA for the semantic fleet coordinator.

This benchmark is deliberately broader than the four-UAV demonstration.  It
samples fleet size, payload, battery, communication, failures and spatial
hazards, checks safety invariants, and repeats every allocation to verify
determinism.  It does not claim RGB-D/VLM or full-flight validation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.semantic_fleet import (  # noqa: E402
    FleetCoordinator,
    FleetDrone,
    FleetTask,
    SemanticEvent,
    SemanticEventGate,
)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _segment_distance_2d(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    point = point[[0, 2]]
    start = start[[0, 2]]
    end = end[[0, 2]]
    delta = end - start
    denom = float(np.dot(delta, delta))
    ratio = 0.0 if denom <= 1e-12 else float(np.dot(point - start, delta) / denom)
    projection = start + np.clip(ratio, 0.0, 1.0) * delta
    return float(np.linalg.norm(point - projection))


def _make_event(payload: dict, timestamp_s: float = 10.0) -> SemanticEvent:
    return SemanticEvent.from_mapping(
        payload,
        default_timestamp_s=timestamp_s,
        default_source_drone_id=payload.get("source_drone_id", "UAV-00"),
    )


def _sample_case(rng: np.random.Generator, index: int):
    fleet_size = int(rng.integers(3, 6))
    task_count = int(rng.integers(fleet_size, fleet_size * 2 + 1))
    drones = []
    for drone_index in range(fleet_size):
        drones.append(FleetDrone(
            drone_id=f"UAV-{drone_index:02d}",
            position=tuple(rng.uniform((-220, 25, -220), (220, 65, 220))),
            battery_pct=float(rng.uniform(0.12, 1.0)),
            max_payload_kg=float(rng.choice((3.0, 5.0, 8.0))),
            available=True,
            comm_neighbor_count=int(rng.integers(0, fleet_size)),
        ))
    tasks = []
    for task_index in range(task_count):
        tasks.append(FleetTask(
            task_id=f"CASE-{index:04d}-TASK-{task_index:02d}",
            pickup=tuple(rng.uniform((-180, 30, -180), (180, 60, 180))),
            delivery=tuple(rng.uniform((-180, 30, -180), (180, 60, 180))),
            payload_kg=float(rng.uniform(0.2, 8.5)),
            priority=int(rng.integers(0, 4)),
            required_comms=bool(rng.random() < 0.25),
        ))
    events = []
    if rng.random() < 0.60:
        task = tasks[int(rng.integers(0, len(tasks)))]
        midpoint = (np.asarray(task.pickup) + np.asarray(task.delivery)) * 0.5
        midpoint[[0, 2]] += rng.uniform(-8.0, 8.0, size=2)
        events.append(_make_event({
            "event_id": f"CASE-{index:04d}-NOFLY",
            "event_type": "no_fly_zone",
            "timestamp_s": 10.0,
            "source_drone_id": drones[0].drone_id,
            "position": midpoint.tolist(),
            "radius_m": float(rng.uniform(8.0, 30.0)),
            "confidence": 0.98,
            "severity": 1.0,
            "ttl_s": 120.0,
            "evidence": "authoritative simulated airspace notice",
        }))
    if rng.random() < 0.55:
        task = tasks[int(rng.integers(0, len(tasks)))]
        midpoint = (np.asarray(task.pickup) + np.asarray(task.delivery)) * 0.5
        events.append(_make_event({
            "event_id": f"CASE-{index:04d}-WEATHER",
            "event_type": "weather_hazard",
            "timestamp_s": 10.0,
            "source_drone_id": drones[-1].drone_id,
            "position": midpoint.tolist(),
            "radius_m": float(rng.uniform(18.0, 55.0)),
            "confidence": 0.86,
            "severity": float(rng.uniform(0.35, 0.9)),
            "ttl_s": 90.0,
            "evidence": "wind estimator corroborated the visual motion cue",
        }))
    if rng.random() < 0.25:
        failed_index = int(rng.integers(0, len(drones)))
        failed = drones[failed_index]
        events.append(_make_event({
            "event_id": f"CASE-{index:04d}-FAILURE",
            "event_type": "drone_failure",
            "timestamp_s": 10.0,
            "source_drone_id": failed.drone_id,
            "position": list(failed.position),
            "radius_m": 1.0,
            "confidence": 1.0,
            "severity": 1.0,
            "ttl_s": 120.0,
            "evidence": "health monitor fault latch",
        }))
        drones[failed_index] = replace(failed, available=False)
    return drones, tasks, events


def _check_plan(drones, tasks, events, plan) -> list[str]:
    failures: list[str] = []
    drone_by_id = {drone.drone_id: drone for drone in drones}
    task_by_id = {task.task_id: task for task in tasks}
    assigned = [task_id for bundle in plan.assignments.values() for task_id in bundle]
    if len(assigned) != len(set(assigned)):
        failures.append("duplicate_task_assignment")
    if set(assigned) | set(plan.blocked_task_ids) != set(task_by_id):
        failures.append("task_accounting_mismatch")
    for drone_id, task_ids in plan.assignments.items():
        drone = drone_by_id[drone_id]
        for task_id in task_ids:
            task = task_by_id[task_id]
            if not drone.available or drone.battery_pct < 0.15:
                failures.append("unavailable_drone_assigned")
            if task.payload_kg > drone.max_payload_kg + 1e-9:
                failures.append("payload_constraint_violated")
            if task.required_comms and drone.comm_neighbor_count < 1:
                failures.append("communication_constraint_violated")
    no_fly_events = [event for event in events if event.event_type.value == "no_fly_zone"]
    for task_id, route in plan.task_routes.items():
        points = [np.asarray(point, dtype=float) for point in route]
        for event in no_fly_events:
            center = np.asarray(event.position, dtype=float)
            distances = [
                _segment_distance_2d(center, start, end)
                for start, end in zip(points[:-1], points[1:])
            ]
            task = task_by_id[task_id]
            direct_distance = min(
                _segment_distance_2d(center, np.asarray(task.pickup), np.asarray(task.delivery)),
                _segment_distance_2d(center, np.asarray(route[0]), np.asarray(task.pickup)),
            )
            if direct_distance <= event.radius_m and min(distances) < event.radius_m - 1e-6:
                failures.append("no_fly_clearance_violated")
    return failures


def run_benchmark(*, cases: int = 500, seed: int = 20260829) -> dict:
    rng = np.random.default_rng(seed)
    coordinator = FleetCoordinator(max_tasks_per_drone=3)
    latencies_ms: list[float] = []
    invariant_failures: list[dict] = []
    determinism_failures = 0
    total_tasks = 0
    assigned_tasks = 0
    blocked_tasks = 0
    replan_tasks = 0
    changed_cases = 0
    gate = SemanticEventGate()
    gate_valid_accepts = 0
    gate_invalid_rejects = 0
    gate_trials = 0

    for case_index in range(cases):
        drones, tasks, events = _sample_case(rng, case_index)
        baseline = coordinator.allocate(drones, tasks, [], now_s=10.0)
        started = time.perf_counter()
        plan = coordinator.allocate(drones, tasks, events, now_s=10.0)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        repeated = coordinator.allocate(drones, tasks, events, now_s=10.0)
        if plan.to_dict() != repeated.to_dict():
            determinism_failures += 1
        failures = sorted(set(_check_plan(drones, tasks, events, plan)))
        if failures:
            invariant_failures.append({"case": case_index, "failures": failures})
        total_tasks += len(tasks)
        assigned = sum(len(bundle) for bundle in plan.assignments.values())
        assigned_tasks += assigned
        blocked_tasks += len(plan.blocked_task_ids)
        replan_tasks += len(plan.replan_task_ids)
        if plan.assignments != baseline.assignments:
            changed_cases += 1

        if events:
            gate_trials += 1
            valid = gate.validate(
                events[0], now_s=10.0, known_drone_ids={item.drone_id for item in drones}
            )
            gate_valid_accepts += int(valid.accepted)
            invalid = replace(events[0], confidence=0.1)
            rejected = gate.validate(
                invalid, now_s=10.0, known_drone_ids={item.drone_id for item in drones}
            )
            gate_invalid_rejects += int(not rejected.accepted)

    p95 = _percentile(latencies_ms, 95)
    status = "PASS" if (
        not invariant_failures
        and determinism_failures == 0
        and p95 < 20.0
        and gate_valid_accepts == gate_trials
        and gate_invalid_rejects == gate_trials
    ) else "FAIL"
    return {
        "benchmark": "urbanfly_semantic_fleet_component_qa_v1",
        "status": status,
        "scope": "semantic event gate + dynamic constraint adapter + fleet assignment",
        "explicitly_not_tested": [
            "Qwen model inference",
            "RGB-D visual event recognition",
            "static-map path-planner validation of detour waypoints",
            "multi-UAV 6-DOF closed-loop flight",
            "frontend visualization",
        ],
        "seed": seed,
        "cases": cases,
        "fleet_size_range": [3, 5],
        "total_tasks": total_tasks,
        "assigned_tasks": assigned_tasks,
        "blocked_tasks": blocked_tasks,
        "assignment_rate": assigned_tasks / total_tasks if total_tasks else 0.0,
        "replan_tasks": replan_tasks,
        "event_changed_assignment_cases": changed_cases,
        "invariant_failure_count": len(invariant_failures),
        "invariant_failure_examples": invariant_failures[:10],
        "determinism_failures": determinism_failures,
        "gate_valid_accepts": gate_valid_accepts,
        "gate_invalid_rejects": gate_invalid_rejects,
        "gate_trials": gate_trials,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "p50": _percentile(latencies_ms, 50),
            "p95": p95,
            "max": max(latencies_ms),
        },
        "acceptance": {
            "invariant_failure_count": 0,
            "determinism_failures": 0,
            "gate_invalid_rejection_rate": 1.0,
            "latency_p95_ms_max": 20.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "qwen_fleet_system_v1" / "semantic_coordinator_qa.json",
    )
    args = parser.parse_args()
    report = run_benchmark(cases=args.cases, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "cases": report["cases"],
        "invariant_failures": report["invariant_failure_count"],
        "determinism_failures": report["determinism_failures"],
        "latency_p95_ms": report["latency_ms"]["p95"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
