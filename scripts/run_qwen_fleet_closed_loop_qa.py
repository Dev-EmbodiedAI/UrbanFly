"""Headless 6-DOF Helsinki QA for the four-UAV semantic fleet scenario.

This validates simulator integration, event timing, assignment, fail-closed
static path repair, dynamics, collisions and completion.  Qwen inference and
RGB-D semantic recognition remain explicitly outside this test. The scenario
runs for 300 simulated seconds so three surviving aircraft can recover the
failed aircraft's in-flight payload and still finish all four missions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_navigation import HelsinkiNavigationStack  # noqa: E402
from backend.engine.models import TaskStatus  # noqa: E402
from backend.engine.scenario import ScenarioEngine  # noqa: E402
from backend.engine.simulator import Simulator  # noqa: E402


def _clearance(collision_map, position: np.ndarray, radius: float) -> float:
    try:
        return float(collision_map.clearance(position, radius))
    except TypeError:
        return float(collision_map.clearance(position))


def run_qa(scene: Path) -> dict:
    load_started = time.perf_counter()
    stack = HelsinkiNavigationStack.load(scene)
    load_s = time.perf_counter() - load_started
    scenario = ScenarioEngine.create_default().get_scenario("qwen_semantic_fleet")
    if scenario is None:
        raise RuntimeError("qwen_semantic_fleet scenario is unavailable")
    simulator = Simulator(
        planner=stack.global_planner,
        static_collision_map=stack.collision_map,
    )
    simulator.initialize_scenario(scenario)
    minimum_clearance = {drone.id: float("inf") for drone in simulator.drones}
    minimum_separation = float("inf")
    trajectory_samples = {drone.id: 0 for drone in simulator.drones}
    run_started = time.perf_counter()
    while simulator.state == "running":
        simulator.step()
        if simulator._step_count % 5 != 0:
            continue
        for drone in simulator.drones:
            minimum_clearance[drone.id] = min(
                minimum_clearance[drone.id],
                _clearance(stack.collision_map, drone.position, drone.safety_radius),
            )
            trajectory_samples[drone.id] += 1
        for first_index, first in enumerate(simulator.drones):
            for second in simulator.drones[first_index + 1:]:
                minimum_separation = min(
                    minimum_separation,
                    float(np.linalg.norm(first.position - second.position)),
                )
    wall_s = time.perf_counter() - run_started
    bridge = simulator.semantic_fleet_bridge
    audit = bridge.runtime.audit
    gate_entries = [item for item in audit if item.get("kind") == "event_gate"]
    completed = [task.id for task in simulator.tasks if task.status == TaskStatus.COMPLETED]
    collision_counts = dict(simulator._static_collision_counts)
    failures = []
    if bridge.next_event_index != len(bridge.timeline):
        failures.append("timeline_not_fully_consumed")
    if len(gate_entries) != len(bridge.timeline) or not all(
        item.get("accepted") for item in gate_entries
    ):
        failures.append("semantic_event_gate_failure")
    if bridge.path_validation_failures:
        failures.append("static_path_validation_failure")
    if not bridge.failure_recoveries:
        failures.append("active_mission_failure_recovery_not_exercised")
    if any(value > 0 for value in collision_counts.values()):
        failures.append("collision_detected")
    if min(minimum_clearance.values()) < 2.5:
        failures.append("minimum_clearance_below_2p5m")
    if minimum_separation < 3.0:
        failures.append("fleet_separation_below_3m")
    if len(completed) != len(simulator.tasks):
        failures.append("task_completion_incomplete")
    if simulator.time < scenario.duration:
        failures.append("simulation_ended_early")
    return {
        "qa": "urbanfly_qwen_semantic_fleet_helsinki_6dof_v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scope": [
            "4-UAV Helsinki mission-level closed loop",
            "50 Hz six-degree-of-freedom dynamics",
            "scripted semantic event ingestion and gate",
            "temporary obstacle/no-fly/weather/failure effects",
            "deterministic task reallocation",
            "static collision-map route readback",
        ],
        "explicitly_not_tested": [
            "Qwen model inference",
            "RGB-D visual event recognition",
            "frontend visual rendering",
            "real PX4 hardware",
        ],
        "scene": str(scene.resolve()),
        "scene_load_s": load_s,
        "simulated_duration_s": simulator.time,
        "physics_steps": simulator._step_count,
        "wall_time_s": wall_s,
        "realtime_factor": simulator.time / wall_s if wall_s > 0 else None,
        "fleet_size": len(simulator.drones),
        "task_count": len(simulator.tasks),
        "completed_tasks": completed,
        "completion_rate": len(completed) / len(simulator.tasks),
        "semantic_events_total": len(bridge.timeline),
        "semantic_events_consumed": bridge.next_event_index,
        "semantic_events_accepted": sum(bool(item.get("accepted")) for item in gate_entries),
        "applied_plan_count": bridge.applied_plan_count,
        "path_validation_failure_count": len(bridge.path_validation_failures),
        "failure_recovery_count": len(bridge.failure_recoveries),
        "failure_recoveries": bridge.failure_recoveries,
        "collision_counts": collision_counts,
        "minimum_clearance_m": minimum_clearance,
        "fleet_minimum_separation_m": minimum_separation,
        "trajectory_samples": trajectory_samples,
        "drones": {
            drone.id: {
                "state": drone.state.value,
                "position": drone.position.tolist(),
                "battery_pct": drone.battery_pct,
                "tasks_completed": drone.tasks_completed,
                "distance_m": drone.total_distance_traveled,
                "energy_wh": drone.total_energy_consumed,
            }
            for drone in simulator.drones
        },
        "final_semantic_snapshot": bridge.snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "qwen_fleet_system_v1" / "helsinki_6dof_qa.json",
    )
    args = parser.parse_args()
    report = run_qa(args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "failures": report["failures"],
        "completion_rate": report["completion_rate"],
        "minimum_clearance_m": min(report["minimum_clearance_m"].values()),
        "collision_total": sum(report["collision_counts"].values()),
        "wall_time_s": report["wall_time_s"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
