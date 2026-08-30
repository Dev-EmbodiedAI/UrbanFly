from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_midterm_benchmark as bench
from backend.engine.allocator.cbba import CBBAAllocator


OUTPUT_PATH = ROOT / "data" / "stc_assignment_view.json"
TASK_COUNT = 84
DYNAMIC_BIAS = 1.50
MAX_SAMPLE_ROUTES = 18


def build_assignment():
    layout = bench.load_city_layout()
    buildings = layout["buildings"]
    density_meta = bench.build_density_grid(buildings)
    hotspots = bench.make_hotspots(layout)
    drones = bench.build_drones()
    tasks = bench.build_tasks(layout, hotspots, TASK_COUNT, dynamic_bias=DYNAMIC_BIAS)
    comm_graph = bench.make_comm_graph(drones, buildings, "occlusion")

    allocator = CBBAAllocator(
        max_iterations=5,
        max_bundle_size=9,
        use_priority_term=True,
        use_corridor_term=True,
        use_robust_consensus=True,
        use_residual_repair=True,
    )

    t0 = time.perf_counter()
    assignments = allocator.allocate(
        bench.clone_drones(drones),
        bench.clone_tasks(tasks),
        comm_graph,
        current_time=bench.CURRENT_TIME,
    )
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    loader = bench.load_planner_if_available
    try:
        bench.load_planner_if_available = lambda: None
        metrics = bench.evaluate_assignments(
            assignments,
            bench.clone_drones(drones),
            bench.clone_tasks(tasks),
            density_meta,
            comm_graph,
            "STC-RCBBA",
            "dense_occlusion",
        )
    finally:
        bench.load_planner_if_available = loader

    metrics["runtime_ms"] = runtime_ms
    return layout, density_meta, drones, tasks, assignments, metrics


def select_sample_tasks(
    drones,
    task_map: Dict[str, object],
    assignments: Dict[str, List[str]],
    max_samples: int,
):
    ordered_drones = [d for d in drones if assignments.get(d.id)]
    selected = []
    seen = set()
    round_idx = 0

    while len(selected) < max_samples:
        progress = False
        ordered_drones.sort(
            key=lambda d: (
                len(assignments.get(d.id, [])) <= round_idx,
                d.drone_type,
                d.id,
            )
        )
        for drone in ordered_drones:
            bundle = assignments.get(drone.id, [])
            if round_idx >= len(bundle):
                continue
            task_id = bundle[round_idx]
            if task_id in seen:
                continue
            selected.append((drone, round_idx, task_map[task_id]))
            seen.add(task_id)
            progress = True
            if len(selected) >= max_samples:
                break
        if not progress:
            break
        round_idx += 1
    return selected


def planner_sample_routes(layout, density_meta, drones, tasks, assignments):
    planner = bench.load_planner_if_available()
    if planner is None:
        raise RuntimeError("未找到可用的路径规划场景缓存，无法导出真实平滑轨迹。")

    task_map = {task.id: task for task in tasks}
    samples = select_sample_tasks(drones, task_map, assignments, MAX_SAMPLE_ROUTES)
    route_rows = []

    for drone, task_rank, task in samples:
        bundle = assignments[drone.id]
        start_pos = drone.position.copy()
        departure_time = bench.CURRENT_TIME

        for prev_task_id in bundle[:task_rank]:
            prev_task = task_map[prev_task_id]
            layer = prev_task.airspace_level or "L1_street_canyon"
            leg1 = bench.route_distance_proxy(start_pos, prev_task.pickup_pos, layer, density_meta)
            leg2 = bench.route_distance_proxy(prev_task.pickup_pos, prev_task.delivery_pos, layer, density_meta)
            departure_time += leg1 / max(drone.cruise_speed, 0.1)
            departure_time = max(departure_time, prev_task.time_window[0])
            departure_time += prev_task.pickup_service_time
            departure_time += leg2 / max(drone.cruise_speed, 0.1)
            departure_time += prev_task.delivery_service_time
            start_pos = prev_task.delivery_pos.copy()

        sample = bench.build_planner_route_sample(
            planner=planner,
            drone=drone,
            start_pos=start_pos,
            task=task,
            departure_time=departure_time,
        )
        route_rows.append(
            {
                "drone_id": drone.id,
                "drone_type": drone.drone_type,
                "task_id": task.id,
                "task_type": task.task_type,
                "layer": sample["layer"],
                "points": sample["points"],
                "pickup_point": sample["pickup_point"],
                "delivery_point": sample["delivery_point"],
            }
        )

    return route_rows


def main():
    layout, density_meta, drones, tasks, assignments, metrics = build_assignment()
    routes = planner_sample_routes(layout, density_meta, drones, tasks, assignments)

    payload = {
        "algorithm": "STC-RCBBA",
        "meta": {
            "city_source": layout["source_path"],
            "task_count": len(tasks),
            "building_count": len(layout["buildings"]),
            "dynamic_bias": DYNAMIC_BIAS,
            "sample_route_count": len(routes),
        },
        "summary": {
            "completed_tasks": int(metrics["completed_tasks"]),
            "total_tasks": int(len(tasks)),
            "assignment_rate": float(metrics["assignment_rate"]),
            "weighted_completion_rate": float(metrics["weighted_completion_rate"]),
            "time_window_rate": float(metrics["time_window_rate"]),
            "corridor_conflicts": int(metrics["corridor_conflicts"]),
            "runtime_ms": float(metrics["runtime_ms"]),
            "avg_completion_time_s": float(metrics["avg_completion_time_s"]),
            "sample_chain_count": int(len(routes)),
            "sample_active_drone_count": int(len({row["drone_id"] for row in routes})),
        },
        "benchmark_metrics": metrics,
        "sample_routes": routes,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"assignment view written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
