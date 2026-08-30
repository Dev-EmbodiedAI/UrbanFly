#!/usr/bin/env python3
"""Urban-core sampling visualization and unseen 100-task final regression."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_navigation import HelsinkiNavigationStack  # noqa: E402
from backend.engine.helsinki_urban_sampling import HelsinkiUrbanDensity  # noqa: E402
from backend.engine.planner import PlanningError  # noqa: E402
from scripts.verify_helsinki_low_altitude_expert import (  # noqa: E402
    DifficultTaskSampler,
    LowAltitudeTask,
    TASK_TYPES,
    _jsonable,
    _run_task,
)


SCENE = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
STRATUM_COUNTS_PER_TYPE = {
    "dense_core": 15,
    "peripheral_mixed": 3,
    "cross_city": 2,
}


def _generate_tasks(stack, density, seed: int) -> List[LowAltitudeTask]:
    sampler = DifficultTaskSampler(stack, seed, urban_density=density)
    tasks = []
    index = 0
    for stratum, per_type in STRATUM_COUNTS_PER_TYPE.items():
        mask = density.mask_for_stratum(stratum)
        for repeat in range(per_type):
            for task_type in TASK_TYPES:
                distance_override = (400.0, 700.0) if stratum == "cross_city" else None
                task = sampler.sample(
                    index,
                    task_type,
                    spatial_stratum=stratum,
                    start_spatial_mask=mask,
                    goal_spatial_mask=mask,
                    distance_range_override=distance_override,
                )
                tasks.append(task)
                index += 1
                print(
                    f"sample {index}/100 stratum={stratum} type={task_type} "
                    f"density={task.local_obstacle_density:.3f} distance={task.planar_distance_m:.1f}m",
                    flush=True,
                )
    return tasks


def _plan_candidate(stack, task: LowAltitudeTask):
    start = np.asarray(task.start, dtype=float)
    goal = np.asarray(task.goal, dtype=float)
    record = asdict(task)
    try:
        plan = stack.plan(
            start,
            goal,
            expert_mode="low_altitude_3d",
            altitude_min_m=task.altitude_min_m,
            altitude_max_m=task.altitude_max_m,
        )
    except PlanningError as error:
        record.update(
            planning_success=False,
            triangle_validation_success=False,
            result="PLANNING_FAILED",
            error=str(error),
        )
        return record, None
    vectors = np.diff(plan.simplified_path, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    angles = []
    for incoming, outgoing, a, b in zip(vectors[:-1], vectors[1:], lengths[:-1], lengths[1:]):
        if a < 1e-6 or b < 1e-6:
            continue
        angle = float(np.degrees(np.arccos(np.clip(np.dot(incoming, outgoing) / (a * b), -1.0, 1.0))))
        if angle >= 20.0:
            angles.append(angle)
    record.update(
        planning_success=True,
        triangle_validation_success=bool(
            plan.triangle_validation is not None
            and not plan.triangle_validation["collision"]
        ),
        result="PLANNED",
        path_length_m=plan.path_length_m,
        planning_time_ms=plan.planning_time_ms,
        minimum_clearance_m=plan.validation["minimum_clearance_m"],
        triangle_minimum_distance_m=(
            plan.triangle_validation["minimum_distance_m"]
            if plan.triangle_validation is not None
            else None
        ),
        max_altitude_m=float(np.max(plan.trajectory[:, 1])),
        turn_count=len(angles),
        total_turn_angle_degrees=float(sum(angles)),
        path_curvature_degrees_per_m=float(sum(angles) / max(plan.path_length_m, 1e-6)),
    )
    return record, plan.trajectory


def _plot_overview(stack, density, tasks, records, paths, output: Path):
    height = stack.collision_map.height
    extent = [
        stack.collision_map.origin_x,
        stack.collision_map.origin_x + (height.shape[1] - 1) * stack.collision_map.resolution,
        stack.collision_map.origin_z,
        stack.collision_map.maximum_z,
    ]
    score_image = density.urban_density_score.T[::-1, :]
    core_image = density.dense_urban_core_mask.T[::-1, :]
    edge_image = (density.distance_to_boundary_m >= density.edge_exclusion_m).T[::-1, :]
    colors = {
        "dense_core": "#ff3b30",
        "peripheral_mixed": "#00b7ff",
        "cross_city": "#ffd60a",
    }
    figure, axes = plt.subplots(1, 2, figsize=(17, 8.2), constrained_layout=True)
    axes[0].imshow(height, extent=extent, origin="upper", cmap="terrain", vmin=-2, vmax=46)
    density_layer = axes[0].imshow(
        score_image,
        extent=extent,
        origin="upper",
        cmap="magma",
        alpha=0.55,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].contour(core_image.astype(float), levels=[0.5], colors=["#00ff80"], linewidths=1.2, extent=extent, origin="upper")
    axes[0].contour(edge_image.astype(float), levels=[0.5], colors=["#ffffff"], linewidths=1.0, linestyles="--", extent=extent, origin="upper")
    axes[0].set(title="Geometry-derived urban density and dense-core mask", xlabel="east x (m)", ylabel="north z (m)")
    axes[0].set_aspect("equal")
    figure.colorbar(density_layer, ax=axes[0], label="urban density score", shrink=0.82)

    axes[1].imshow(height, extent=extent, origin="upper", cmap="terrain", vmin=-2, vmax=46)
    shown = set()
    for task, record, path in zip(tasks, records, paths):
        if path is None:
            continue
        label = task.spatial_stratum if task.spatial_stratum not in shown else None
        shown.add(task.spatial_stratum)
        color = colors[task.spatial_stratum]
        axes[1].plot(path[:, 0], path[:, 2], color=color, lw=1.0, alpha=0.68, label=label)
        axes[1].scatter(task.start[0], task.start[2], color=color, s=9, alpha=0.85)
        axes[1].scatter(task.goal[0], task.goal[2], color=color, s=12, marker="x", alpha=0.9)
    axes[1].contour(core_image.astype(float), levels=[0.5], colors=["#00ff80"], linewidths=0.8, extent=extent, origin="upper")
    axes[1].contour(edge_image.astype(float), levels=[0.5], colors=["#ffffff"], linewidths=0.8, linestyles="--", extent=extent, origin="upper")
    axes[1].set(title="100 stratified candidate tasks and planned paths", xlabel="east x (m)", ylabel="north z (m)")
    axes[1].set_aspect("equal")
    axes[1].legend(loc="best", fontsize=8)
    figure.savefig(output, dpi=175)
    plt.close(figure)


def _spatial_metrics(tasks: Sequence[dict], records: Sequence[dict]) -> dict:
    obstacle = [float(task["local_obstacle_density"]) for task in tasks]
    boundary = [float(task["mean_boundary_distance_m"]) for task in tasks]
    blockers = [int(task["number_of_blocking_obstacles"]) for task in tasks]
    lengths = [float(record["path_length_m"]) for record in records if record.get("planning_success")]
    turns = [float(record["turn_count"]) for record in records if record.get("planning_success")]
    counts = Counter(task["spatial_stratum"] for task in tasks)
    denominator = max(1, len(tasks))
    return {
        "tasks": len(tasks),
        "dense_core_task_ratio": counts["dense_core"] / denominator,
        "peripheral_mixed_task_ratio": counts["peripheral_mixed"] / denominator,
        "cross_city_task_ratio": counts["cross_city"] / denominator,
        "mean_local_obstacle_density": float(np.mean(obstacle)),
        "median_local_obstacle_density": float(np.median(obstacle)),
        "mean_distance_to_map_boundary_m": float(np.mean(boundary)),
        "mean_number_of_blocking_obstacles": float(np.mean(blockers)),
        "mean_path_length_m": float(np.mean(lengths)) if lengths else None,
        "p95_path_length_m": float(np.percentile(lengths, 95)) if lengths else None,
        "mean_turn_count": float(np.mean(turns)) if turns else None,
    }


def _previous_distribution(density, previous_tasks_path: Path, previous_records_path: Path):
    if not previous_tasks_path.exists() or not previous_records_path.exists():
        return None
    tasks = json.loads(previous_tasks_path.read_text(encoding="utf-8"))
    records = json.loads(previous_records_path.read_text(encoding="utf-8"))
    converted = []
    for task in tasks:
        start = np.asarray(task["start"], dtype=float)
        goal = np.asarray(task["goal"], dtype=float)
        start_cell = density.world_to_cell(start)
        goal_cell = density.world_to_cell(goal)
        planar = float(np.linalg.norm((goal - start)[[0, 2]]))
        if density.dense_urban_core_mask[start_cell] and density.dense_urban_core_mask[goal_cell]:
            stratum = "dense_core"
        elif planar >= 400.0:
            stratum = "cross_city"
        else:
            stratum = "peripheral_mixed"
        converted.append(
            {
                **task,
                "spatial_stratum": stratum,
                "local_obstacle_density": 0.5 * (
                    density.obstacle_density_at(start) + density.obstacle_density_at(goal)
                ),
                "mean_boundary_distance_m": 0.5 * (
                    density.boundary_distance_at(start) + density.boundary_distance_at(goal)
                ),
                "number_of_blocking_obstacles": 1,
            }
        )
    previous_lengths = [float(item["path_length_m"]) for item in records if item.get("planning_success")]
    counts = Counter(item["spatial_stratum"] for item in converted)
    return {
        "tasks": len(converted),
        "inferred_dense_core_task_ratio": counts["dense_core"] / max(1, len(converted)),
        "inferred_peripheral_mixed_task_ratio": counts["peripheral_mixed"] / max(1, len(converted)),
        "inferred_cross_city_task_ratio": counts["cross_city"] / max(1, len(converted)),
        "mean_local_obstacle_density": float(np.mean([item["local_obstacle_density"] for item in converted])),
        "median_local_obstacle_density": float(np.median([item["local_obstacle_density"] for item in converted])),
        "mean_distance_to_map_boundary_m": float(np.mean([item["mean_boundary_distance_m"] for item in converted])),
        "mean_path_length_m": float(np.mean(previous_lengths)),
        "p95_path_length_m": float(np.percentile(previous_lengths, 95)),
        "note": "old tasks are reclassified by the new geometry-derived masks; old turn/blocker metrics were not recorded",
    }


def _regression_summary(records: Sequence[dict]) -> dict:
    count = max(1, len(records))
    planned = [item for item in records if item.get("planning_success")]
    triangle_valid = [
        item
        for item in planned
        if item.get("triangle_validation") is not None
        and not item["triangle_validation"]["collision"]
    ]
    successful = [item for item in records if item.get("execution_success")]
    ceiling_violations = [
        item for item in records if int(item.get("executed_height_violation_samples", 0)) > 0
    ]
    minimum_clearance = [float(item["minimum_clearance_m"]) for item in planned]
    tracking = [float(item["tracking_rmse_m"]) for item in records if item.get("tracking_rmse_m") is not None]
    triangle_distances = [
        float(item["triangle_validation"]["minimum_distance_m"])
        for item in triangle_valid
    ]
    return {
        "tasks": len(records),
        "planning_success_rate": len(planned) / count,
        "triangle_validation_success_rate": len(triangle_valid) / count,
        "execution_success_rate": len(successful) / count,
        "collision_rate": sum(bool(item.get("collision")) for item in records) / count,
        "controller_failure_rate": sum(item.get("result") == "CONTROLLER_TRACKING_FAILED" for item in records) / count,
        "ceiling_violation_rate": len(ceiling_violations) / count,
        "minimum_heightmap_clearance_m": min(minimum_clearance) if minimum_clearance else None,
        "minimum_triangle_distance_m": min(triangle_distances) if triangle_distances else None,
        "tracking_rmse_mean_m": float(np.mean(tracking)) if tracking else None,
        "results": dict(Counter(str(item.get("result")) for item in records)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "helsinki_urban_core_regression",
    )
    parser.add_argument("--candidate-seed", type=int, default=20260830)
    parser.add_argument("--regression-seed", type=int, default=20260930)
    parser.add_argument("--overview-only", action="store_true")
    parser.add_argument(
        "--regression-only",
        action="store_true",
        help="reuse the saved candidate set/overview and run only the independent unseen set",
    )
    args = parser.parse_args()
    if args.overview_only and args.regression_only:
        parser.error("--overview-only and --regression-only are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stack = HelsinkiNavigationStack.load(args.scene, enable_triangle_geometry=True)
    density = HelsinkiUrbanDensity(stack.grid)
    started = time.perf_counter()

    if args.regression_only:
        candidate_task_dicts = json.loads(
            (args.output_dir / "candidate_tasks.json").read_text(encoding="utf-8")
        )
        candidate_records = json.loads(
            (args.output_dir / "candidate_records.json").read_text(encoding="utf-8")
        )
        candidate_spatial = _spatial_metrics(candidate_task_dicts, candidate_records)
    else:
        candidate_tasks = _generate_tasks(stack, density, args.candidate_seed)
        candidate_records = []
        candidate_paths = []
        for index, task in enumerate(candidate_tasks):
            record, path = _plan_candidate(stack, task)
            candidate_records.append(record)
            candidate_paths.append(path)
            print(f"candidate plan {index + 1}/100 result={record['result']}", flush=True)
        candidate_task_dicts = [asdict(task) for task in candidate_tasks]
        (args.output_dir / "candidate_tasks.json").write_text(
            json.dumps(candidate_task_dicts, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.output_dir / "candidate_records.json").write_text(
            json.dumps(_jsonable(candidate_records), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez_compressed(
            args.output_dir / "candidate_paths.npz",
            **{f"path_{index:03d}": path for index, path in enumerate(candidate_paths) if path is not None},
        )
        _plot_overview(
            stack,
            density,
            candidate_tasks,
            candidate_records,
            candidate_paths,
            args.output_dir / "urban_core_100_tasks_overview.png",
        )
        candidate_spatial = _spatial_metrics(candidate_task_dicts, candidate_records)
    previous = _previous_distribution(
        density,
        ROOT / "outputs" / "helsinki_low_altitude_expert" / "qualification_tasks.json",
        ROOT / "outputs" / "helsinki_low_altitude_expert" / "qualification_200" / "records.json",
    )

    regression_summary = None
    if not args.overview_only:
        regression_tasks = _generate_tasks(stack, density, args.regression_seed)
        regression_dir = args.output_dir / "unseen_100"
        regression_dir.mkdir(exist_ok=True)
        paths_dir = regression_dir / "paths"
        paths_dir.mkdir(exist_ok=True)
        regression_records = []
        for index, task in enumerate(regression_tasks):
            record = _run_task(stack, task, paths_dir / f"task_{index:03d}.npz")
            regression_records.append(record)
            print(
                f"unseen run {index + 1}/100 stratum={task.spatial_stratum} "
                f"result={record.get('result')} wall={record.get('wall_time_s', 0):.2f}s",
                flush=True,
            )
            if (index + 1) % 10 == 0:
                (regression_dir / "checkpoint.json").write_text(
                    json.dumps(_jsonable(regression_records), indent=2, ensure_ascii=False), encoding="utf-8"
                )
        regression_summary = _regression_summary(regression_records)
        regression_summary["spatial_distribution"] = _spatial_metrics(
            [asdict(task) for task in regression_tasks], regression_records
        )
        regression_summary["qualified"] = bool(
            regression_summary["planning_success_rate"] > 0.95
            and regression_summary["triangle_validation_success_rate"] > 0.95
            and regression_summary["execution_success_rate"] > 0.95
            and regression_summary["collision_rate"] < 0.02
            and regression_summary["ceiling_violation_rate"] < 0.02
        )
        (regression_dir / "tasks.json").write_text(
            json.dumps([asdict(task) for task in regression_tasks], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (regression_dir / "records.json").write_text(
            json.dumps(_jsonable(regression_records), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (regression_dir / "summary.json").write_text(
            json.dumps(_jsonable(regression_summary), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    report = {
        "scene": str(args.scene.resolve()),
        "triangle_geometry": {
            "source": str(stack.local_triangle_geometry.source_path.resolve()),
            "triangles": stack.local_triangle_geometry.triangle_count,
            "acceleration_structure": stack.local_triangle_geometry.acceleration_structure,
            "build_time_s": stack.local_triangle_geometry.build_time_s,
        },
        "urban_density": density.summary(),
        "candidate_100_spatial_metrics": candidate_spatial,
        "previous_200_spatial_metrics": previous,
        "unseen_100_regression": regression_summary,
        "elapsed_wall_time_s": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return 0 if regression_summary is None or regression_summary.get("qualified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
