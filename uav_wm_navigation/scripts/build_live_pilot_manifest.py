from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import h5py


def main() -> int:
    parser = argparse.ArgumentParser(description="Build repeatable live-route Pilot manifest from audited episode metadata.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--calibration-dir", type=Path)
    args = parser.parse_args()
    templates = []
    for path in sorted(args.metadata_dir.glob("*.metadata.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        waypoints = item.get("route_nwu") or []
        if len(waypoints) < 2:
            continue
        dx, dy = waypoints[1][0] - waypoints[0][0], waypoints[1][1] - waypoints[0][1]
        templates.append({
            "corridor_id": item["corridor_id"], "spatial_zone": item.get("spatial_zone", "pilot"),
            "scenario": item["scenario"], "difficulty": item["difficulty"],
            "start_nwu": waypoints[0], "goal_nwu": waypoints[-1], "waypoints_nwu": waypoints,
            "initial_yaw_nwu_deg": math.degrees(math.atan2(dy, dx)),
        })
    if not templates:
        raise RuntimeError("no usable audited route metadata found")
    calibrated_risk: dict[int, float] = {}
    if args.calibration_dir:
        samples: dict[int, list[float]] = {}
        for path in sorted(args.calibration_dir.glob("*.h5")):
            try:
                episode_index = int(path.stem.rsplit("_", 1)[-1]) % len(templates)
            except ValueError:
                continue
            with h5py.File(path, "r") as handle:
                risk = float(handle["labels/candidate_collision"][:].mean())
            samples.setdefault(episode_index, []).append(risk)
        calibrated_risk = {index: float(np.mean(values)) for index, values in samples.items()}
    routes = []
    for index in range(args.routes):
        route = dict(templates[index % len(templates)])
        waypoints = np.asarray(route["waypoints_nwu"], dtype=float)
        middle_index = min(max(len(waypoints) // 2, 1), len(waypoints) - 2)
        anchor = waypoints[middle_index].copy()
        tangent = waypoints[middle_index + 1, :2] - waypoints[middle_index - 1, :2]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        left = np.array([-tangent[1], tangent[0]])
        template_index = index % len(templates)
        measured_risk = calibrated_risk.get(template_index)
        # Move the small obstacle towards low-risk corridors and away from a
        # corridor whose first calibration was already saturated.  The sign
        # alternates so the data do not encode a fixed avoidance side.
        offset_m = 0.3 if measured_risk is None or measured_risk < 0.08 else (0.8 if measured_risk <= 0.25 else 2.0)
        route["scripted_obstacles"] = []
        for obstacle_number, waypoint_index in enumerate((min(2, len(waypoints) - 2), min(5, len(waypoints) - 2))):
            obstacle_anchor = waypoints[waypoint_index].copy()
            before_index, after_index = max(waypoint_index - 1, 0), min(waypoint_index + 1, len(waypoints) - 1)
            obstacle_tangent = waypoints[after_index, :2] - waypoints[before_index, :2]
            obstacle_tangent /= max(float(np.linalg.norm(obstacle_tangent)), 1e-6)
            obstacle_left = np.array([-obstacle_tangent[1], obstacle_tangent[0]])
            side = 1.0 if (index + obstacle_number) % 2 == 0 else -1.0
            obstacle_anchor[:2] += side * offset_m * obstacle_left
            route["scripted_obstacles"].append({
                "local_nwu": obstacle_anchor.tolist(),
                "yaw_degrees": -math.degrees(math.atan2(obstacle_tangent[1], obstacle_tangent[0])),
                "blueprint": "walker.pedestrian.0001",
            })
        if route["scenario"] in {"DynamicCrossing", "OccludedCrossing", "DenseMixedUrban"}:
            start_crossing, end_crossing = anchor.copy(), anchor.copy()
            start_crossing[:2] -= 5.0 * left
            end_crossing[:2] += 5.0 * left
            route["scripted_crossing"] = {
                "enabled": True, "start_local_nwu": start_crossing.tolist(),
                "end_local_nwu": end_crossing.tolist(), "yaw_degrees": 90.0,
                "delay_s": 2.0, "duration_s": 4.0,
                "blueprint": "walker.pedestrian.0001",
            }
        route.update({
            "route_id": f"Town10HD_wm_pilot_{index:04d}", "split": None,
            "seed": args.seed + index, "calibrated_obstacle_offset_m": offset_m,
            "calibration_danger_fraction": measured_risk,
        })
        routes.append(route)
    payload = {
        "format": "uav-wm-live-pilot-recollection-v1", "map": "Town10HD",
        "source_metadata_dir": str(args.metadata_dir.resolve()),
        "calibration_dir": None if args.calibration_dir is None else str(args.calibration_dir.resolve()),
        "calibrated_corridors": len(calibrated_risk), "routes": routes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "routes": len(routes), "unique_corridors": len(templates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
