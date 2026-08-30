#!/usr/bin/env python3
"""Deterministic real-Helsinki planner/execution qualification.

No procedural city and no learned policy are used.  The script first runs the
three mandatory deterministic tests.  Only if all pass does it run the
requested random-task qualification batch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_navigation import (  # noqa: E402
    HelsinkiNavigationStack,
    NavigationResult,
    PlanningError,
    polyline_length,
)


SCENE = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"


def _world_from_cell(stack, row: int, column: int, altitude: float) -> np.ndarray:
    collision = stack.collision_map
    x = collision.origin_x + (column + 0.5) * collision.resolution
    z = collision.maximum_z - (row + 0.5) * collision.resolution
    return np.array([x, altitude, z], dtype=float)


def _surface_endpoint(stack, row: int, column: int, extra: float = 2.0) -> np.ndarray:
    probe = _world_from_cell(stack, row, column, 0.0)
    surface = stack.collision_map.surface_height(probe, stack.planning_clearance)
    probe[1] = surface + stack.planning_clearance + extra
    return probe


def _find_free_space_test(stack) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260817)
    height = stack.collision_map.height
    rows, columns = np.where(np.isfinite(height) & (height < 6.0))
    order = rng.permutation(len(rows))
    candidates = list(zip(rows[order[:6000]], columns[order[:6000]]))
    for row_a, col_a in candidates:
        start = _world_from_cell(stack, int(row_a), int(col_a), 28.0)
        if not stack.is_valid_start(start)["valid"]:
            continue
        for _ in range(12):
            row_b, col_b = candidates[int(rng.integers(0, len(candidates)))]
            goal = _world_from_cell(stack, int(row_b), int(col_b), 28.0)
            distance = float(np.linalg.norm((goal - start)[[0, 2]]))
            if not 90.0 <= distance <= 180.0:
                continue
            if not stack.is_valid_goal(goal)["valid"]:
                continue
            if stack.validate_path(np.vstack((start, goal)))["path_valid"]:
                return start, goal
    raise RuntimeError("could not find deterministic free-space task")


def _component_crossings(mask: np.ndarray):
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    component_ids = np.argsort(sizes[1:])[::-1] + 1
    for component_id in component_ids:
        size = int(sizes[component_id])
        if not 200 <= size <= 20_000:
            continue
        rows, columns = np.where(labels == component_id)
        row_counts = np.bincount(rows, minlength=mask.shape[0])
        row = int(np.argmax(row_counts))
        component_columns = np.where(labels[row] == component_id)[0]
        if len(component_columns) >= 5:
            yield "x", row, int(component_columns.min()), int(component_columns.max()), size
        column_counts = np.bincount(columns, minlength=mask.shape[1])
        column = int(np.argmax(column_counts))
        component_rows = np.where(labels[:, column] == component_id)[0]
        if len(component_rows) >= 5:
            yield "z", column, int(component_rows.min()), int(component_rows.max()), size


def _find_building_test(stack):
    height = stack.collision_map.height
    altitude = 28.0
    obstacle = np.isfinite(height) & (height >= altitude - stack.required_clearance)
    padding_cells = int(math.ceil(12.0 / stack.collision_map.resolution))
    for axis, fixed, minimum, maximum, size in _component_crossings(obstacle):
        low = minimum - padding_cells
        high = maximum + padding_cells
        if low < 20 or high >= height.shape[0] - 20:
            continue
        if axis == "x":
            start = _world_from_cell(stack, fixed, low, altitude)
            goal = _world_from_cell(stack, fixed, high, altitude)
        else:
            start = _world_from_cell(stack, high, fixed, altitude)
            goal = _world_from_cell(stack, low, fixed, altitude)
        if not stack.is_valid_start(start)["valid"] or not stack.is_valid_goal(goal)["valid"]:
            continue
        straight = stack.validate_path(np.vstack((start, goal)))
        if straight["path_valid"]:
            continue
        try:
            plan = stack.plan(
                start,
                goal,
                flight_level="L2_transition",
                allow_layer_transitions=False,
            )
        except PlanningError:
            continue
        planar_deviation = float(
            np.max(
                np.abs(
                    plan.trajectory[:, 2 if axis == "x" else 0]
                    - start[2 if axis == "x" else 0]
                )
            )
        )
        if planar_deviation < stack.required_clearance:
            continue
        return start, goal, plan, {
            "component_size_cells": size,
            "crossing_axis": axis,
            "straight_line_validation": straight,
            "planar_detour_m": planar_deviation,
        }
    raise RuntimeError("could not find a building-blockage task with a valid planar detour")


def _find_rooftop_test(stack) -> tuple[np.ndarray, np.ndarray, dict]:
    height = stack.collision_map.height
    roof_mask = np.isfinite(height) & (height >= 18.0)
    local_range = ndimage.maximum_filter(height, size=9) - ndimage.minimum_filter(height, size=9)
    roof_rows, roof_columns = np.where(roof_mask & (local_range < 2.5))
    if len(roof_rows) == 0:
        roof_rows, roof_columns = np.where(roof_mask)
    ranking = np.argsort(height[roof_rows, roof_columns])[::-1]

    ground_rows, ground_columns = np.where(np.isfinite(height) & (height < 5.0))
    rng = np.random.default_rng(20260818)
    ground_order = rng.permutation(len(ground_rows))
    for roof_index in ranking[:3000]:
        row = int(roof_rows[roof_index])
        column = int(roof_columns[roof_index])
        start = _surface_endpoint(stack, row, column, extra=2.0)
        roof_surface = float(stack.collision_map.surface_height(start, stack.required_clearance))
        if start[1] > float(stack.config["maximum_altitude_m"]):
            continue
        for ground_index in ground_order[:5000]:
            goal_row = int(ground_rows[ground_index])
            goal_column = int(ground_columns[ground_index])
            goal = _surface_endpoint(stack, goal_row, goal_column, extra=2.0)
            planar_distance = float(np.linalg.norm((goal - start)[[0, 2]]))
            if not 180.0 <= planar_distance <= 420.0:
                continue
            if not stack.is_valid_start(start)["valid"] or not stack.is_valid_goal(goal)["valid"]:
                continue
            return start, goal, {
                "roof_surface_m": roof_surface,
                "ground_surface_m": float(
                    stack.collision_map.surface_height(goal, stack.required_clearance)
                ),
                "planar_distance_m": planar_distance,
            }
    raise RuntimeError("could not find deterministic rooftop-to-ground task")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _plot_test(stack, name, plan, execution, output, extra):
    height = stack.collision_map.height
    extent = [
        stack.collision_map.origin_x,
        stack.collision_map.origin_x + (height.shape[1] - 1) * stack.collision_map.resolution,
        stack.collision_map.origin_z,
        stack.collision_map.maximum_z,
    ]
    executed = execution["executed_trajectory"]
    all_x = np.concatenate((plan.trajectory[:, 0], executed[:, 0]))
    all_z = np.concatenate((plan.trajectory[:, 2], executed[:, 2]))
    margin = 35.0
    xlim = (max(extent[0], float(all_x.min()) - margin), min(extent[1], float(all_x.max()) + margin))
    zlim = (max(extent[2], float(all_z.min()) - margin), min(extent[3], float(all_z.max()) + margin))

    reference_altitude = float(np.median(plan.trajectory[:, 1]))
    obstacle = np.isfinite(height) & (height >= reference_altitude - stack.required_clearance)
    inflate_cells = int(math.ceil(stack.required_clearance / stack.collision_map.resolution))
    inflated = ndimage.binary_dilation(obstacle, iterations=inflate_cells)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    axis = axes[0]
    image = axis.imshow(height, extent=extent, origin="upper", cmap="terrain", vmin=-2, vmax=48)
    axis.contour(
        inflated.astype(float),
        levels=[0.5],
        colors=["#ff3b30"],
        linewidths=0.65,
        extent=extent,
        origin="upper",
    )
    axis.plot(plan.global_path[:, 0], plan.global_path[:, 2], "--", color="#242424", lw=1.1, label="A* global")
    axis.plot(plan.trajectory[:, 0], plan.trajectory[:, 2], color="#00b7ff", lw=2.4, label="validated trajectory")
    axis.plot(executed[:, 0], executed[:, 2], color="#ffcc00", lw=1.5, label="6-DOF executed")
    axis.scatter(plan.start[0], plan.start[2], c="#00df81", s=60, edgecolors="black", label="start")
    axis.scatter(plan.goal[0], plan.goal[2], c="#ff4fb3", s=70, marker="*", edgecolors="black", label="goal")
    axis.set(xlim=xlim, ylim=zlim, xlabel="local east x (m)", ylabel="local north z (m)")
    axis.set_title(f"{name}: real Helsinki height surface + inflated obstacle")
    axis.set_aspect("equal")
    axis.legend(loc="best", fontsize=8)
    figure.colorbar(image, ax=axis, label="surface height (m)", shrink=0.82)

    profile = axes[1]
    planned_distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(plan.trajectory, axis=0), axis=1))]
    planned_surface = np.asarray(
        [stack.collision_map.surface_height(point, stack.required_clearance) for point in plan.trajectory]
    )
    executed_distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(executed, axis=0), axis=1))]
    profile.plot(planned_distance, planned_surface, color="#654321", lw=1.5, label="inflated surface")
    profile.fill_between(planned_distance, planned_surface, alpha=0.25, color="#8b5a2b")
    profile.plot(planned_distance, plan.trajectory[:, 1], color="#00b7ff", lw=2.2, label="planned altitude")
    profile.plot(executed_distance, executed[:, 1], color="#ff9900", lw=1.2, label="executed altitude")
    profile.set(xlabel="path distance (m)", ylabel="height y (m)")
    profile.set_title(
        f"clearance={plan.validation['minimum_clearance_m']:.2f} m, "
        f"result={execution['result']}"
    )
    profile.grid(alpha=0.25)
    profile.legend(fontsize=8)
    figure.suptitle(json.dumps(extra, ensure_ascii=False, default=str)[:180], fontsize=8)
    figure.savefig(output, dpi=165)
    plt.close(figure)


def _run_case(stack, name, start, goal, output_dir, plan=None, extra=None, **plan_kwargs):
    extra = dict(extra or {})
    try:
        if plan is None:
            plan = stack.plan(start, goal, **plan_kwargs)
    except PlanningError as error:
        return {
            "name": name,
            "pass": False,
            "result": NavigationResult.PLANNING_FAILED.value,
            "error": str(error),
            "start": start,
            "goal": goal,
            **extra,
        }
    execution = stack.execute(plan)
    passed = bool(
        plan.validation["path_valid"]
        and execution["success"]
        and not execution["collision"]
        and execution["executed_validation"]["path_valid"]
    )
    image_path = output_dir / f"{name.lower()}_debug.png"
    _plot_test(stack, name, plan, execution, image_path, extra)
    np.savez_compressed(
        output_dir / f"{name.lower()}_paths.npz",
        global_path=plan.global_path,
        planned_trajectory=plan.trajectory,
        executed_trajectory=execution["executed_trajectory"],
        start=start,
        goal=goal,
    )
    execution_summary = {key: value for key, value in execution.items() if key != "executed_trajectory"}
    return {
        "name": name,
        "pass": passed,
        "result": execution["result"],
        "start": start,
        "goal": goal,
        "path_length_m": plan.path_length_m,
        "minimum_clearance_m": plan.validation["minimum_clearance_m"],
        "minimum_clearance_margin_m": plan.validation["minimum_clearance_margin_m"],
        "planning_time_ms": plan.planning_time_ms,
        "planner_mode": plan.planner_mode,
        "collision": execution["collision"],
        "execution": execution_summary,
        "path_validation": plan.validation,
        "debug_image": str(image_path),
        **extra,
    }


def _endpoint_pools(stack):
    height = stack.collision_map.height
    local_range = ndimage.maximum_filter(height, size=7) - ndimage.minimum_filter(height, size=7)
    ground = np.column_stack(np.where(np.isfinite(height) & (height < 6.0) & (local_range < 2.0)))
    rooftop = np.column_stack(np.where(np.isfinite(height) & (height > 15.0) & (local_range < 2.5)))
    return ground, rooftop


def _qualification(stack, count, seed, output_dir):
    rng = np.random.default_rng(seed)
    ground, rooftop = _endpoint_pools(stack)
    task_types = (
        "ground_to_ground",
        "rooftop_to_ground",
        "ground_to_rooftop",
        "rooftop_to_rooftop",
        "long_range_urban",
    )
    records = []
    minimum_clearances = []
    planning_times = []
    path_lengths = []
    tracking_errors = []
    for task_index in range(count):
        task_started = time.perf_counter()
        task_type = task_types[task_index % len(task_types)]
        pools = {
            "ground_to_ground": (ground, ground),
            "rooftop_to_ground": (rooftop, ground),
            "ground_to_rooftop": (ground, rooftop),
            "rooftop_to_rooftop": (rooftop, rooftop),
            "long_range_urban": (ground, ground),
        }[task_type]
        task_record = {"index": task_index, "task_type": task_type}
        selected = None
        for _ in range(500):
            start_cell = pools[0][int(rng.integers(0, len(pools[0])))]
            goal_cell = pools[1][int(rng.integers(0, len(pools[1])))]
            start = _surface_endpoint(stack, int(start_cell[0]), int(start_cell[1]), extra=2.0)
            goal = _surface_endpoint(stack, int(goal_cell[0]), int(goal_cell[1]), extra=2.0)
            distance = float(np.linalg.norm((goal - start)[[0, 2]]))
            minimum_distance = 300.0 if task_type == "long_range_urban" else 60.0
            maximum_distance = 500.0 if task_type == "long_range_urban" else 450.0
            if not minimum_distance <= distance <= maximum_distance:
                continue
            if stack.is_valid_start(start)["valid"] and stack.is_valid_goal(goal)["valid"]:
                selected = (start, goal)
                break
        if selected is None:
            task_record["result"] = NavigationResult.INVALID_GOAL.value
            records.append(task_record)
            continue
        start, goal = selected
        task_record.update(start=start.tolist(), goal=goal.tolist())
        try:
            plan = stack.plan(start, goal, flight_level="L2_transition", allow_layer_transitions=True)
        except PlanningError as error:
            task_record.update(result=NavigationResult.PLANNING_FAILED.value, error=str(error))
            records.append(task_record)
            continue
        task_record["planning_success"] = True
        task_record["planner_mode"] = plan.planner_mode
        planning_times.append(plan.planning_time_ms)
        path_lengths.append(plan.path_length_m)
        minimum_clearances.append(plan.validation["minimum_clearance_m"])
        timeout = float(stack.config["execution_timeout_s"])
        execution = stack.execute(plan, timeout_s=timeout)
        tracking_errors.append(execution["tracking_error_mean_m"])
        task_record.update(
            result=execution["result"],
            execution_success=execution["success"],
            collision=execution["collision"],
            planning_time_ms=plan.planning_time_ms,
            path_length_m=plan.path_length_m,
            minimum_clearance_m=plan.validation["minimum_clearance_m"],
            tracking_error_mean_m=execution["tracking_error_mean_m"],
            final_error_m=execution["final_error_m"],
            wall_time_s=time.perf_counter() - task_started,
        )
        records.append(task_record)
        print(
            f"qualification {task_index + 1}/{count} "
            f"mode={plan.planner_mode} result={execution['result']} "
            f"wall={task_record['wall_time_s']:.2f}s",
            flush=True,
        )
        if (task_index + 1) % 10 == 0:
            (output_dir / "qualification_checkpoint.json").write_text(
                json.dumps(_jsonable(records), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    planning_success = sum(bool(record.get("planning_success")) for record in records)
    execution_success = sum(bool(record.get("execution_success")) for record in records)
    collisions = sum(bool(record.get("collision")) for record in records)
    timeouts = sum(record.get("result") in {"TIMEOUT", "ACTION_TIMEOUT"} for record in records)
    invalid_paths = sum(record.get("result") == "PATH_COLLISION" for record in records)
    denominator = max(1, len(records))
    summary = {
        "tasks": len(records),
        "seed": seed,
        "planning_success_rate": planning_success / denominator,
        "execution_success_rate": execution_success / denominator,
        "collision_rate": collisions / denominator,
        "timeout_rate": timeouts / denominator,
        "invalid_path_rate": invalid_paths / denominator,
        "minimum_clearance_m": float(min(minimum_clearances)) if minimum_clearances else None,
        "path_length_mean_m": float(np.mean(path_lengths)) if path_lengths else None,
        "planning_time_mean_ms": float(np.mean(planning_times)) if planning_times else None,
        "planning_time_p95_ms": float(np.percentile(planning_times, 95)) if planning_times else None,
        "tracking_error_mean_m": float(np.mean(tracking_errors)) if tracking_errors else None,
    }
    summary["qualified"] = bool(
        summary["planning_success_rate"] > 0.98
        and summary["execution_success_rate"] > 0.95
        and summary["collision_rate"] < 0.02
        and summary["invalid_path_rate"] <= 1e-9
    )
    (output_dir / "qualification_records.json").write_text(
        json.dumps(_jsonable(records), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "helsinki_privileged_planner",
    )
    parser.add_argument("--qualification-tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--skip-qualification", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stack = HelsinkiNavigationStack.load(args.scene)
    started = time.perf_counter()

    start_a, goal_a = _find_free_space_test(stack)
    result_a = _run_case(
        stack,
        "Test_A",
        start_a,
        goal_a,
        args.output_dir,
        flight_level="L2_transition",
        allow_layer_transitions=False,
    )
    print(json.dumps(_jsonable(result_a), ensure_ascii=False), flush=True)

    try:
        start_b, goal_b, plan_b, extra_b = _find_building_test(stack)
        result_b = _run_case(
            stack,
            "Test_B",
            start_b,
            goal_b,
            args.output_dir,
            plan=plan_b,
            extra=extra_b,
        )
    except Exception as error:
        result_b = {"name": "Test_B", "pass": False, "result": "PLANNING_FAILED", "error": str(error)}
    print(json.dumps(_jsonable(result_b), ensure_ascii=False), flush=True)

    try:
        start_c, goal_c, extra_c = _find_rooftop_test(stack)
        result_c = _run_case(
            stack,
            "Test_C",
            start_c,
            goal_c,
            args.output_dir,
            extra=extra_c,
            flight_level="L2_transition",
            allow_layer_transitions=True,
        )
    except Exception as error:
        result_c = {"name": "Test_C", "pass": False, "result": "PLANNING_FAILED", "error": str(error)}
    print(json.dumps(_jsonable(result_c), ensure_ascii=False), flush=True)

    abc_pass = all(result.get("pass", False) for result in (result_a, result_b, result_c))
    qualification = None
    if abc_pass and not args.skip_qualification:
        qualification = _qualification(
            stack,
            max(200, min(500, int(args.qualification_tasks))),
            args.seed,
            args.output_dir,
        )

    report = {
        "scene": str(args.scene),
        "map": {
            "collision_type": "conservative_heightmap_from_real_L18_triangle_mesh",
            "collision_shape": list(stack.collision_map.shape),
            "collision_resolution_m": stack.collision_map.resolution,
            "planner_grid_shape": list(stack.grid.shape),
            "planner_resolution_m": stack.grid.resolution,
            "drone_radius_m": stack.drone_radius,
            "safety_margin_m": stack.safety_margin,
            "required_clearance_m": stack.required_clearance,
            "planning_clearance_m": stack.planning_clearance,
            "distance_slice_altitudes_m": [item[0] for item in stack.distance_slices.slices],
            "coordinate_frame": "local x=east, y=up, z=north; Helsinki source north maps to -visual-z before runtime conversion",
        },
        "stack": {
            "global_planner": "existing PathPlanner multi-layer 2.5D A* (fail-closed)",
            "trajectory_smoothing": "line-of-sight shortcut + validated quadratic corner arcs",
            "controller": "heuristic 6-DOF geometric waypoint controller (not MPC)",
            "safety_filter": "independent 0.25 m swept-heightmap validation + runtime swept collision guard",
        },
        "tests": [result_a, result_b, result_c],
        "abc_pass": abc_pass,
        "qualification": qualification,
        "verdict": (
            "QUALIFIED AS PRIVILEGED EXPERT"
            if qualification and qualification.get("qualified")
            else "NOT QUALIFIED"
        ),
        "elapsed_wall_time_s": time.perf_counter() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False), flush=True)
    return 0 if abc_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
