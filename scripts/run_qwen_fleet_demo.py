"""Run the repeatable four-UAV semantic coordination demonstration.

The default mode is offline and deterministic.  It exercises exactly the same
event gate and fleet coordinator used by the optional Qwen-VL client, without
requiring model downloads or a network connection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.semantic_fleet import (
    DeterministicSemanticInterpreter,
    FleetDrone,
    FleetTask,
    ObservationPacket,
    SemanticFleetRuntime,
)


def build_fleet(*, failed_drone: str | None = None) -> list[FleetDrone]:
    definitions = [
        ("UAV-SCOUT", (-180.0, 42.0, -80.0), 0.82, 3.0),
        ("UAV-ALPHA", (-120.0, 38.0, 60.0), 0.75, 8.0),
        ("UAV-BRAVO", (100.0, 45.0, -60.0), 0.88, 8.0),
        ("UAV-RESERVE", (170.0, 35.0, 90.0), 0.94, 5.0),
    ]
    return [
        FleetDrone(
            drone_id=drone_id,
            position=position,
            battery_pct=battery,
            max_payload_kg=payload,
            available=drone_id != failed_drone,
            comm_neighbor_count=3,
        )
        for drone_id, position, battery, payload in definitions
    ]


def build_tasks() -> list[FleetTask]:
    return [
        FleetTask("INSPECT-WEST", (-80, 40, -40), (60, 45, -20), 1.0, 1),
        FleetTask("MEDICAL-NORTH", (-30, 35, 90), (140, 40, 120), 2.0, 0, True),
        FleetTask("INSPECT-EAST", (80, 45, -80), (190, 45, -120), 1.0, 2),
        FleetTask("DELIVERY-SOUTH", (40, 35, 20), (-130, 38, 100), 4.0, 2),
    ]


def run_demo() -> dict:
    runtime = SemanticFleetRuntime(DeterministicSemanticInterpreter())
    tasks = build_tasks()
    fleet = build_fleet()
    timeline: list[dict] = []

    timeline.append({"time_s": 0.0, "plan": runtime.reallocate(fleet, tasks, now_s=0.0).to_dict()})
    scripted_events = [
        (20.0, "UAV-SCOUT", {
            "event_id": "evt-crane",
            "event_type": "temporary_obstacle",
            "position": [-10, 42, -30],
            "radius_m": 28,
            "confidence": 0.94,
            "severity": 0.72,
            "ttl_s": 120,
            "evidence": "RGB-D crane boom occupies the planned corridor",
            "affected_task_ids": ["INSPECT-WEST"],
        }),
        (40.0, "UAV-SCOUT", {
            "event_id": "evt-nofly",
            "event_type": "no_fly_zone",
            "position": [80, 40, 105],
            "radius_m": 38,
            "confidence": 0.99,
            "severity": 1.0,
            "ttl_s": 180,
            "evidence": "authoritative temporary airspace notice",
            "affected_task_ids": ["MEDICAL-NORTH"],
        }),
        (60.0, "UAV-BRAVO", {
            "event_id": "evt-gust",
            "event_type": "weather_hazard",
            "position": [120, 45, -90],
            "radius_m": 55,
            "confidence": 0.88,
            "severity": 0.65,
            "ttl_s": 90,
            "evidence": "wind estimator and image motion agree on a strong gust front",
            "affected_task_ids": ["INSPECT-EAST"],
        }),
        (80.0, "UAV-ALPHA", {
            "event_id": "evt-failure",
            "event_type": "drone_failure",
            "position": [-120, 38, 60],
            "radius_m": 1,
            "confidence": 1.0,
            "severity": 1.0,
            "ttl_s": 220,
            "evidence": "flight computer health monitor declared actuator fault",
        }),
    ]
    for timestamp_s, source, cue in scripted_events:
        packet = ObservationPacket(
            timestamp_s=timestamp_s,
            drone_id=source,
            semantic_cues=(cue,),
        )
        decisions = runtime.ingest(
            packet,
            known_drone_ids={item.drone_id for item in fleet},
        )
        if cue["event_type"] == "drone_failure":
            fleet = build_fleet(failed_drone=source)
        plan = runtime.reallocate(fleet, tasks, now_s=timestamp_s)
        timeline.append({
            "time_s": timestamp_s,
            "event": cue,
            "gate": [
                {"accepted": item.accepted, "reason": item.reason}
                for item in decisions
            ],
            "plan": plan.to_dict(),
        })
    return {
        "demo": "urbanfly_qwen_semantic_fleet_v1",
        "fleet_size": len(fleet),
        "qwen_required": False,
        "control_authority": "existing_local_policy_only",
        "timeline": timeline,
        "final_runtime": runtime.snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional canonical JSON report")
    args = parser.parse_args()
    report = run_demo()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered)


if __name__ == "__main__":
    main()
