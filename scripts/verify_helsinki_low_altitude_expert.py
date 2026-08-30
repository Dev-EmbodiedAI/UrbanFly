#!/usr/bin/env python3
"""Strict low-altitude 3-D qualification on the real Helsinki surface.

The benchmark never invokes the high-altitude overflight candidate.  Every
task has a hard altitude interval and a straight line blocked by geometry that
is too tall to overfly below its task ceiling.  Feasibility is established
independently by connected free space in the ceiling slice before the planner
is called.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
)
from backend.engine.planner import PlanningError  # noqa: E402


SCENE = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
TASK_TYPES = (
    "building_blocked",
    "street_canyon",
    "rooftop_to_ground",
    "ground_to_rooftop",
    "rooftop_to_rooftop",
)
DEBUG_DISTRIBUTION = {task_type: 4 for task_type in TASK_TYPES}
QUALIFICATION_DISTRIBUTION = {
    "building_blocked": 60,
    "street_canyon": 40,
    "rooftop_to_ground": 40,
    "ground_to_rooftop": 30,
    "rooftop_to_rooftop": 30,
}


@dataclass
class LowAltitudeTask:
    index: int
    task_type: str
    start: List[float]
    goal: List[float]
    altitude_min_m: float
    altitude_max_m: float
    planar_distance_m: float
    direct_max_inflated_surface_m: float
    direct_overflight_required_m: float
    ceiling_component: int
    start_surface_class: str
    goal_surface_class: str
    spatial_stratum: str = "global"
    start_density_score: float = float("nan")
    goal_density_score: float = float("nan")
    local_obstacle_density: float = float("nan")
    mean_boundary_distance_m: float = float("nan")
    number_of_blocking_obstacles: int = 0


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _height_distribution(stack: HelsinkiNavigationStack) -> dict:
    height = stack.collision_map.height
    finite = height[np.isfinite(height)]
    percentiles = {
        str(value): float(np.percentile(finite, value))
        for value in (0, 1, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 100)
    }
    return {
        "source": "real HelsinkiCentral1km 0.5 m triangle-mesh-derived highest-surface heightmap",
        "shape": list(height.shape),
        "resolution_m": stack.collision_map.resolution,
        "finite_ratio": float(np.isfinite(height).mean()),
        "percentiles_m": percentiles,
        "fraction_above_height": {
            str(threshold): float(np.mean(finite > threshold))
            for threshold in (5, 10, 15, 20, 25, 30, 35, 40)
        },
        "selected_ceiling_bins_m": [15, 20, 25, 30, 35, 40],
    }


class DifficultTaskSampler:
    """Generate blocked-but-feasible tasks from the real conservative map."""

    def __init__(self, stack: HelsinkiNavigationStack, seed: int, urban_density=None):
        self.stack = stack
        self.rng = np.random.default_rng(seed)
        self.height = np.asarray(stack.low_altitude_grid.heightmap, dtype=float)
        self.resolution = float(stack.low_altitude_grid.resolution)
        self.clearance = float(stack.low_altitude_planning_clearance)
        self.vertical_tracking_buffer = float(
            stack.config["low_altitude_vertical_tracking_buffer_m"]
        )
        self.ground = self.height < 6.0
        self.roof = (self.height >= 12.0) & (self.height <= 35.0)
        tall = self.height >= 12.0
        distance_to_tall = ndimage.distance_transform_edt(~tall) * self.resolution
        self.street = self.ground & (distance_to_tall >= 5.0) & (distance_to_tall <= 25.0)
        self.roof_labels, _ = ndimage.label(self.roof, structure=np.ones((3, 3), dtype=np.uint8))
        self.obstacle_labels, _ = ndimage.label(
            self.height >= 6.0,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        self.urban_density = urban_density
        self._ceiling_components: Dict[float, Tuple[np.ndarray, int]] = {}
        self._used: set[Tuple] = set()

    def _components(self, ceiling: float) -> Tuple[np.ndarray, int]:
        key = float(ceiling)
        if key not in self._ceiling_components:
            free = self.height + self.clearance + 0.25 <= key
            labels, count = ndimage.label(free, structure=np.ones((3, 3), dtype=np.uint8))
            self._ceiling_components[key] = (labels, int(count))
        return self._ceiling_components[key]

    def _pool(
        self,
        surface_class: str,
        ceiling: float,
        spatial_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        base = {
            "ground": self.ground,
            "street": self.street,
            "roof": self.roof,
        }[surface_class]
        search_ceiling = ceiling - self.vertical_tracking_buffer
        mask = base & (self.height + self.clearance + 0.5 <= search_ceiling)
        if spatial_mask is not None:
            mask &= np.asarray(spatial_mask, dtype=bool)
        # Exclude the outer 25 m; endpoint connectors at the physical map edge
        # are an unrelated failure mode.
        border = max(1, int(math.ceil(25.0 / self.resolution)))
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
        return np.column_stack(np.where(mask))

    def _world_endpoint(self, cell: np.ndarray, surface_class: str) -> np.ndarray:
        gx, gz = int(cell[0]), int(cell[1])
        surface = float(self.height[gx, gz])
        altitude = surface + self.clearance + 0.5
        if surface_class in {"ground", "street"}:
            altitude = max(5.0, altitude)
        return self.stack.low_altitude_grid.grid_to_world_xz(gx, gz, altitude)

    def _direct_peak(self, start: np.ndarray, goal: np.ndarray) -> float:
        planar = float(np.linalg.norm((goal - start)[[0, 2]]))
        count = max(2, int(math.ceil(planar / 1.0)))
        maximum = -float("inf")
        for alpha in np.linspace(0.0, 1.0, count + 1):
            point = start + float(alpha) * (goal - start)
            maximum = max(
                maximum,
                self.stack.collision_map.surface_height(point, self.clearance),
            )
        return float(maximum)

    def _spec(self, task_type: str):
        if task_type == "building_blocked":
            return "ground", "ground", (15.0, 20.0, 25.0), (60.0, 260.0)
        if task_type == "street_canyon":
            return "street", "street", (15.0, 20.0, 25.0), (70.0, 240.0)
        if task_type == "rooftop_to_ground":
            return "roof", "ground", (25.0, 30.0, 35.0, 40.0), (80.0, 340.0)
        if task_type == "ground_to_rooftop":
            return "ground", "roof", (25.0, 30.0, 35.0, 40.0), (80.0, 340.0)
        if task_type == "rooftop_to_rooftop":
            return "roof", "roof", (25.0, 30.0, 35.0, 40.0), (80.0, 340.0)
        raise ValueError(task_type)

    def sample(
        self,
        index: int,
        task_type: str,
        max_attempts: int = 50000,
        spatial_stratum: str = "global",
        start_spatial_mask: Optional[np.ndarray] = None,
        goal_spatial_mask: Optional[np.ndarray] = None,
        distance_range_override: Optional[Tuple[float, float]] = None,
    ) -> LowAltitudeTask:
        start_class, goal_class, ceilings, distance_range = self._spec(task_type)
        if distance_range_override is not None:
            distance_range = distance_range_override
        ceiling_order = np.asarray(ceilings, dtype=float)
        for attempt in range(max_attempts):
            ceiling = float(ceiling_order[int(self.rng.integers(0, len(ceiling_order)))])
            start_pool = self._pool(start_class, ceiling, start_spatial_mask)
            goal_pool = self._pool(goal_class, ceiling, goal_spatial_mask)
            if len(start_pool) == 0 or len(goal_pool) == 0:
                continue
            start_cell = start_pool[int(self.rng.integers(0, len(start_pool)))]
            goal_cell = goal_pool[int(self.rng.integers(0, len(goal_pool)))]
            signature = (
                task_type,
                int(start_cell[0]),
                int(start_cell[1]),
                int(goal_cell[0]),
                int(goal_cell[1]),
                ceiling,
            )
            if signature in self._used:
                continue
            distance = float(np.linalg.norm((goal_cell - start_cell) * self.resolution))
            if not distance_range[0] <= distance <= distance_range[1]:
                continue
            labels, _ = self._components(ceiling - self.vertical_tracking_buffer)
            start_component = int(labels[tuple(start_cell)])
            if start_component == 0 or int(labels[tuple(goal_cell)]) != start_component:
                continue
            if task_type == "rooftop_to_rooftop":
                start_roof = int(self.roof_labels[tuple(start_cell)])
                goal_roof = int(self.roof_labels[tuple(goal_cell)])
                if start_roof == 0 or goal_roof == 0 or start_roof == goal_roof:
                    continue
            start = self._world_endpoint(start_cell, start_class)
            goal = self._world_endpoint(goal_cell, goal_class)
            altitude_min = max(
                float(self.stack.config["minimum_altitude_m"]),
                min(float(start[1]), float(goal[1]), 5.0) - 0.5,
            )
            if not self.stack.is_valid_start(start)["valid"] or not self.stack.is_valid_goal(goal)["valid"]:
                continue
            direct_validation = self.stack.validate_path(
                np.vstack((start, goal)),
                required_clearance=self.clearance,
                altitude_min_m=altitude_min,
                altitude_max_m=ceiling,
            )
            if direct_validation["path_valid"]:
                continue
            peak = self._direct_peak(start, goal)
            required_overflight = peak + self.clearance
            # The blocker must be too tall to overfly under the task ceiling,
            # so every feasible solution must detour in plan view.
            if required_overflight <= ceiling + 0.25:
                continue
            line_count = max(2, int(math.ceil(distance / self.resolution)))
            line_cells = np.rint(
                start_cell[None]
                + np.linspace(0.0, 1.0, line_count + 1)[:, None]
                * (goal_cell - start_cell)[None]
            ).astype(int)
            line_cells[:, 0] = np.clip(line_cells[:, 0], 0, self.height.shape[0] - 1)
            line_cells[:, 1] = np.clip(line_cells[:, 1], 0, self.height.shape[1] - 1)
            blocking_labels = np.unique(
                self.obstacle_labels[line_cells[:, 0], line_cells[:, 1]]
            )
            blocking_count = int(np.sum(blocking_labels > 0))
            if self.urban_density is not None:
                start_density = self.urban_density.score_at(start)
                goal_density = self.urban_density.score_at(goal)
                obstacle_density = 0.5 * (
                    self.urban_density.obstacle_density_at(start)
                    + self.urban_density.obstacle_density_at(goal)
                )
                boundary_distance = 0.5 * (
                    self.urban_density.boundary_distance_at(start)
                    + self.urban_density.boundary_distance_at(goal)
                )
            else:
                start_density = goal_density = obstacle_density = boundary_distance = float("nan")
            self._used.add(signature)
            return LowAltitudeTask(
                index=index,
                task_type=task_type,
                start=start.tolist(),
                goal=goal.tolist(),
                altitude_min_m=float(altitude_min),
                altitude_max_m=ceiling,
                planar_distance_m=distance,
                direct_max_inflated_surface_m=peak,
                direct_overflight_required_m=required_overflight,
                ceiling_component=start_component,
                start_surface_class=start_class,
                goal_surface_class=goal_class,
                spatial_stratum=spatial_stratum,
                start_density_score=float(start_density),
                goal_density_score=float(goal_density),
                local_obstacle_density=float(obstacle_density),
                mean_boundary_distance_m=float(boundary_distance),
                number_of_blocking_obstacles=blocking_count,
            )
        raise RuntimeError(
            f"unable to sample {task_type} after {max_attempts} attempts"
        )

    def sample_distribution(self, distribution: Dict[str, int]) -> List[LowAltitudeTask]:
        tasks: List[LowAltitudeTask] = []
        schedule: List[str] = []
        remaining = dict(distribution)
        while any(value > 0 for value in remaining.values()):
            for task_type in TASK_TYPES:
                if remaining.get(task_type, 0) > 0:
                    schedule.append(task_type)
                    remaining[task_type] -= 1
        for index, task_type in enumerate(schedule):
            task = self.sample(index, task_type)
            tasks.append(task)
            print(
                f"sample {index + 1}/{len(schedule)} type={task_type} "
                f"ceiling={task.altitude_max_m:.0f}m distance={task.planar_distance_m:.1f}m",
                flush=True,
            )
        return tasks


def _sample_polyline(path: np.ndarray, step: float = 0.25) -> np.ndarray:
    points = np.asarray(path, dtype=float)
    samples: List[np.ndarray] = []
    for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        count = max(1, int(math.ceil(float(np.linalg.norm(end - start)) / step)))
        segment = start[None] + np.linspace(0.0, 1.0, count + 1)[:, None] * (end - start)[None]
        samples.extend(segment if segment_index == 0 else segment[1:])
    return np.asarray(samples, dtype=float)


def _clearance_stats(stack: HelsinkiNavigationStack, path: np.ndarray) -> Dict[str, float]:
    samples = _sample_polyline(path, float(stack.config["validation_step_m"]))
    clearances = np.asarray(
        [stack.collision_map.clearance(point, stack.required_clearance) for point in samples],
        dtype=float,
    )
    return {
        "minimum": float(np.min(clearances)),
        "mean": float(np.mean(clearances)),
        "p10": float(np.percentile(clearances, 10)),
        "median": float(np.median(clearances)),
        "p90": float(np.percentile(clearances, 90)),
    }


def _run_task(stack: HelsinkiNavigationStack, task: LowAltitudeTask, save_path: Optional[Path] = None) -> dict:
    started = time.perf_counter()
    start = np.asarray(task.start, dtype=float)
    goal = np.asarray(task.goal, dtype=float)
    record = asdict(task)
    record.update(
        straight_line_blocked=True,
        high_altitude_fallback_disabled=True,
        planning_success=False,
        execution_success=False,
        collision=False,
        timeout=False,
        invalid_path=False,
        high_altitude_escape=False,
    )
    try:
        plan = stack.plan(
            start,
            goal,
            expert_mode="low_altitude_3d",
            altitude_min_m=task.altitude_min_m,
            altitude_max_m=task.altitude_max_m,
            allow_layer_transitions=True,
        )
    except PlanningError as error:
        record.update(
            result=NavigationResult.PLANNING_FAILED.value,
            error=str(error),
            wall_time_s=time.perf_counter() - started,
        )
        return record

    clearance_stats = _clearance_stats(stack, plan.trajectory)
    planned_altitude = plan.trajectory[:, 1]
    segment_vectors = np.diff(plan.simplified_path, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    turn_angles = []
    for incoming, outgoing, incoming_length, outgoing_length in zip(
        segment_vectors[:-1],
        segment_vectors[1:],
        segment_lengths[:-1],
        segment_lengths[1:],
    ):
        if incoming_length < 1e-6 or outgoing_length < 1e-6:
            continue
        cosine = float(
            np.clip(
                np.dot(incoming, outgoing) / (incoming_length * outgoing_length),
                -1.0,
                1.0,
            )
        )
        turn_angles.append(float(np.degrees(np.arccos(cosine))))
    meaningful_turns = [angle for angle in turn_angles if angle >= 20.0]
    record.update(
        planning_success=True,
        planner_mode=plan.planner_mode,
        planning_time_ms=plan.planning_time_ms,
        path_length_m=plan.path_length_m,
        minimum_clearance_m=clearance_stats["minimum"],
        mean_clearance_m=clearance_stats["mean"],
        clearance_p10_m=clearance_stats["p10"],
        clearance_median_m=clearance_stats["median"],
        clearance_p90_m=clearance_stats["p90"],
        turn_count=len(meaningful_turns),
        total_turn_angle_degrees=float(sum(meaningful_turns)),
        path_curvature_degrees_per_m=float(
            sum(meaningful_turns) / max(plan.path_length_m, 1e-6)
        ),
        mean_altitude_m=float(np.mean(planned_altitude)),
        max_altitude_m=float(np.max(planned_altitude)),
        min_altitude_m=float(np.min(planned_altitude)),
        vertical_travel_m=float(np.abs(np.diff(planned_altitude)).sum()),
        invalid_path=not bool(plan.validation["path_valid"]),
        high_altitude_escape=bool(np.max(planned_altitude) > task.altitude_max_m + 1e-6),
        planned_path_points=int(len(plan.trajectory)),
        expanded_states=int(stack.low_altitude_planner._last_debug.get("expanded_states", 0)),
        altitude_levels_m=stack.low_altitude_planner._last_debug.get("altitudes_m", []),
        triangle_validation=plan.triangle_validation,
    )
    execution = stack.execute(plan)
    executed = execution.pop("executed_trajectory")
    executed_summary = _jsonable(execution)
    record.update(
        result=execution["result"],
        execution_success=bool(execution["success"]),
        collision=bool(execution["collision"]),
        timeout=execution["result"] in {
            NavigationResult.TIMEOUT.value,
            NavigationResult.ACTION_TIMEOUT.value,
        },
        tracking_rmse_m=float(execution["tracking_error_rmse_m"]),
        flight_time_s=float(execution["sim_time_s"]),
        executed_max_altitude_m=float(np.max(executed[:, 1])),
        executed_vertical_travel_m=float(np.abs(np.diff(executed[:, 1])).sum()),
        executed_height_violation_samples=int(
            execution["executed_validation"]["number_of_height_violations"]
        ),
        action_saturation_ratio=max(
            float(execution["speed_saturation_ratio"]),
            float(execution["climb_saturation_ratio"]),
            float(execution["acceleration_saturation_ratio"]),
        ),
        execution=executed_summary,
        wall_time_s=time.perf_counter() - started,
    )
    if save_path is not None:
        np.savez_compressed(
            save_path,
            start=start,
            goal=goal,
            global_path=plan.global_path,
            planned_trajectory=plan.trajectory,
            executed_trajectory=executed,
            altitude_min_m=np.asarray(task.altitude_min_m),
            altitude_max_m=np.asarray(task.altitude_max_m),
        )
    return record


def _summarize(records: Sequence[dict]) -> dict:
    count = max(1, len(records))
    planned = [item for item in records if item.get("planning_success")]
    executed = [item for item in records if item.get("execution_success")]
    numeric = lambda key, rows=planned: [float(item[key]) for item in rows if item.get(key) is not None]
    minimum_clearances = numeric("minimum_clearance_m")
    mean_clearances = numeric("mean_clearance_m")
    return {
        "tasks": len(records),
        "planning_success_rate": len(planned) / count,
        "execution_success_rate": len(executed) / count,
        "collision_rate": sum(bool(item.get("collision")) for item in records) / count,
        "timeout_rate": sum(bool(item.get("timeout")) for item in records) / count,
        "invalid_path_rate": sum(bool(item.get("invalid_path")) for item in records) / count,
        "straight_line_blocked_ratio": sum(bool(item.get("straight_line_blocked")) for item in records) / count,
        "high_altitude_escape_ratio": sum(bool(item.get("high_altitude_escape")) for item in records) / count,
        "executed_ceiling_violation_episode_ratio": sum(
            int(item.get("executed_height_violation_samples", 0)) > 0
            for item in records
        ) / count,
        "minimum_clearance_m": min(minimum_clearances) if minimum_clearances else None,
        "mean_clearance_m": float(np.mean(mean_clearances)) if mean_clearances else None,
        "mean_path_length_m": float(np.mean(numeric("path_length_m"))) if planned else None,
        "mean_flight_time_s": float(np.mean(numeric("flight_time_s", records))) if records else None,
        "mean_planning_time_ms": float(np.mean(numeric("planning_time_ms"))) if planned else None,
        "p95_planning_time_ms": float(np.percentile(numeric("planning_time_ms"), 95)) if planned else None,
        "mean_altitude_m": float(np.mean(numeric("mean_altitude_m"))) if planned else None,
        "maximum_altitude_m": max(numeric("max_altitude_m")) if planned else None,
        "mean_vertical_travel_m": float(np.mean(numeric("vertical_travel_m"))) if planned else None,
        "mean_tracking_rmse_m": float(np.mean(numeric("tracking_rmse_m", records))) if records else None,
        "mean_action_saturation_ratio": float(np.mean(numeric("action_saturation_ratio", records))) if records else None,
        "results": {
            result: sum(item.get("result") == result for item in records)
            for result in sorted({str(item.get("result")) for item in records})
        },
    }


def _qualified(summary: dict, debug: bool = False) -> bool:
    return bool(
        summary["straight_line_blocked_ratio"] >= 1.0
        and summary["high_altitude_escape_ratio"] <= 0.0
        and summary["invalid_path_rate"] <= 1e-9
        and summary["planning_success_rate"] > 0.95
        and summary["execution_success_rate"] >= (0.90 if debug else 0.90)
        and summary["collision_rate"] < (0.05 if debug else 0.03)
    )


def _plot_case(stack: HelsinkiNavigationStack, task: dict, paths_file: Path, output: Path):
    with np.load(paths_file, allow_pickle=False) as data:
        planned = np.asarray(data["planned_trajectory"], dtype=float)
        executed = np.asarray(data["executed_trajectory"], dtype=float)
    height = stack.collision_map.height
    extent = [
        stack.collision_map.origin_x,
        stack.collision_map.origin_x + (height.shape[1] - 1) * stack.collision_map.resolution,
        stack.collision_map.origin_z,
        stack.collision_map.maximum_z,
    ]
    all_x = np.r_[planned[:, 0], executed[:, 0]]
    all_z = np.r_[planned[:, 2], executed[:, 2]]
    margin = 30.0
    xlim = (max(extent[0], all_x.min() - margin), min(extent[1], all_x.max() + margin))
    zlim = (max(extent[2], all_z.min() - margin), min(extent[3], all_z.max() + margin))
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    image = axes[0].imshow(height, extent=extent, origin="upper", cmap="terrain", vmin=-2, vmax=46)
    blocked_at_ceiling = height + stack.planning_clearance > float(task["altitude_max_m"])
    axes[0].contour(
        blocked_at_ceiling.astype(float), levels=[0.5], colors=["#d62728"],
        linewidths=0.7, extent=extent, origin="upper",
    )
    axes[0].plot(planned[:, 0], planned[:, 2], color="#00b7ff", lw=2.3, label="planned XYZ")
    axes[0].plot(executed[:, 0], executed[:, 2], color="#ffb000", lw=1.4, label="executed 6-DOF")
    axes[0].plot(
        [task["start"][0], task["goal"][0]], [task["start"][2], task["goal"][2]],
        ":", color="#ffffff", lw=1.2, label="blocked straight line",
    )
    axes[0].scatter(task["start"][0], task["start"][2], c="#00df81", s=60, edgecolors="black", label="start")
    axes[0].scatter(task["goal"][0], task["goal"][2], c="#ff4fb3", s=75, marker="*", edgecolors="black", label="goal")
    axes[0].set(xlim=xlim, ylim=zlim, xlabel="east x (m)", ylabel="north z (m)")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"{task['task_type']} — ceiling-obstacle outline")
    axes[0].legend(fontsize=8, loc="best")
    figure.colorbar(image, ax=axes[0], label="highest surface (m)", shrink=0.82)

    planned_distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(planned, axis=0), axis=1))]
    executed_distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(executed, axis=0), axis=1))]
    surface = np.asarray(
        [stack.collision_map.surface_height(point, stack.required_clearance) for point in planned]
    )
    axes[1].fill_between(planned_distance, surface, color="#8b5a2b", alpha=0.3, label="inflated surface")
    axes[1].plot(planned_distance, planned[:, 1], color="#00b7ff", lw=2.2, label="planned altitude")
    axes[1].plot(executed_distance, executed[:, 1], color="#ff9900", lw=1.2, label="executed altitude")
    axes[1].axhline(float(task["altitude_max_m"]), color="#d62728", ls="--", lw=1.5, label="hard ceiling")
    axes[1].axhline(float(task["altitude_min_m"]), color="#9467bd", ls=":", lw=1.2, label="hard floor")
    axes[1].set(xlabel="path distance (m)", ylabel="altitude y (m)")
    axes[1].set_title(
        f"result={task['result']}, min clearance={task.get('minimum_clearance_m', float('nan')):.2f} m"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_debug_overview(stack: HelsinkiNavigationStack, records: Sequence[dict], paths_dir: Path, output: Path):
    height = stack.collision_map.height
    extent = [
        stack.collision_map.origin_x,
        stack.collision_map.origin_x + (height.shape[1] - 1) * stack.collision_map.resolution,
        stack.collision_map.origin_z,
        stack.collision_map.maximum_z,
    ]
    figure, axis = plt.subplots(figsize=(10.5, 10), constrained_layout=True)
    image = axis.imshow(height, extent=extent, origin="upper", cmap="terrain", vmin=-2, vmax=46)
    colors = dict(zip(TASK_TYPES, ("#00b7ff", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")))
    labels = set()
    for record in records:
        file = paths_dir / f"task_{record['index']:03d}.npz"
        if not file.exists():
            continue
        with np.load(file, allow_pickle=False) as data:
            planned = data["planned_trajectory"]
        label = record["task_type"] if record["task_type"] not in labels else None
        labels.add(record["task_type"])
        axis.plot(planned[:, 0], planned[:, 2], color=colors[record["task_type"]], lw=1.3, alpha=0.9, label=label)
        axis.scatter(record["start"][0], record["start"][2], color=colors[record["task_type"]], s=12)
    axis.set(xlabel="east x (m)", ylabel="north z (m)", title="20 deterministic blocked low-altitude XYZ tasks")
    axis.set_aspect("equal")
    axis.legend(loc="best", fontsize=8)
    figure.colorbar(image, ax=axis, label="highest surface (m)", shrink=0.8)
    figure.savefig(output, dpi=175)
    plt.close(figure)


def _run_phase(
    stack: HelsinkiNavigationStack,
    tasks: Sequence[LowAltitudeTask],
    phase_dir: Path,
    visualize_cases: bool,
) -> Tuple[List[dict], dict]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    paths_dir = phase_dir / "paths"
    paths_dir.mkdir(exist_ok=True)
    records: List[dict] = []
    first_success_by_type: Dict[str, dict] = {}
    for task_index, task in enumerate(tasks):
        record = _run_task(stack, task, paths_dir / f"task_{task.index:03d}.npz")
        records.append(record)
        if record.get("planning_success") and task.task_type not in first_success_by_type:
            first_success_by_type[task.task_type] = record
        print(
            f"run {task_index + 1}/{len(tasks)} type={task.task_type} "
            f"plan={record.get('planning_success')} result={record.get('result')} "
            f"max={record.get('max_altitude_m')} ceiling={task.altitude_max_m} "
            f"wall={record['wall_time_s']:.2f}s",
            flush=True,
        )
        if (task_index + 1) % 5 == 0:
            (phase_dir / "checkpoint.json").write_text(
                json.dumps(_jsonable(records), indent=2, ensure_ascii=False), encoding="utf-8"
            )
    summary = _summarize(records)
    (phase_dir / "records.json").write_text(
        json.dumps(_jsonable(records), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (phase_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if visualize_cases:
        _plot_debug_overview(stack, records, paths_dir, phase_dir / "all_20_tasks.png")
        cases_dir = phase_dir / "typical_cases"
        cases_dir.mkdir(exist_ok=True)
        for task_type, record in first_success_by_type.items():
            _plot_case(
                stack,
                record,
                paths_dir / f"task_{record['index']:03d}.npz",
                cases_dir / f"{task_type}.png",
            )
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "helsinki_low_altitude_expert",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--debug-only", action="store_true")
    parser.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stack = HelsinkiNavigationStack.load(args.scene)
    height_stats = _height_distribution(stack)
    (args.output_dir / "height_distribution.json").write_text(
        json.dumps(height_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    started = time.perf_counter()
    debug_summary = None
    qualification_summary = None
    sampler = DifficultTaskSampler(stack, args.seed)

    if not args.qualification_only:
        debug_tasks = sampler.sample_distribution(DEBUG_DISTRIBUTION)
        (args.output_dir / "debug_tasks.json").write_text(
            json.dumps([asdict(task) for task in debug_tasks], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _, debug_summary = _run_phase(
            stack, debug_tasks, args.output_dir / "debug_20", visualize_cases=True
        )
        debug_summary["passed_gate"] = _qualified(debug_summary, debug=True)
        (args.output_dir / "debug_20" / "summary.json").write_text(
            json.dumps(_jsonable(debug_summary), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not debug_summary["passed_gate"]:
            print("20-task debug gate failed; 200-task qualification is intentionally not started", flush=True)

    may_qualify = args.qualification_only or (
        debug_summary is not None and debug_summary.get("passed_gate")
    )
    if may_qualify and not args.debug_only:
        qualification_sampler = DifficultTaskSampler(stack, args.seed + 1)
        qualification_tasks = qualification_sampler.sample_distribution(QUALIFICATION_DISTRIBUTION)
        (args.output_dir / "qualification_tasks.json").write_text(
            json.dumps([asdict(task) for task in qualification_tasks], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        qualification_records, qualification_summary = _run_phase(
            stack,
            qualification_tasks,
            args.output_dir / "qualification_200",
            visualize_cases=False,
        )
        qualification_summary["by_task_type"] = {
            task_type: _summarize(
                [record for record in qualification_records if record["task_type"] == task_type]
            )
            for task_type in TASK_TYPES
        }
        qualification_summary["qualified"] = _qualified(qualification_summary, debug=False)
        (args.output_dir / "qualification_200" / "summary.json").write_text(
            json.dumps(_jsonable(qualification_summary), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    verdict = (
        "QUALIFIED AS LOW-ALTITUDE 3D PRIVILEGED EXPERT"
        if qualification_summary and qualification_summary.get("qualified")
        else "NOT QUALIFIED AS LOW-ALTITUDE 3D PRIVILEGED EXPERT"
    )
    report = {
        "scene": str(args.scene.resolve()),
        "height_distribution": height_stats,
        "planner_reality": {
            "legacy_state": "(grid_x, grid_z, altitude_layer): four coarse 2.5-D bands",
            "low_altitude_state": "(grid_x, grid_z, metric_altitude_index): hard-bounded XYZ lattice",
            "low_altitude_step_m": float(stack.config["low_altitude_step_m"]),
            "neighbors": "10-connected: 8 same-altitude planar moves plus explicit up/down transitions",
            "vertical_cost": "3-D metric edge length plus climb_penalty_factor",
            "height_constraint": "hard task altitude_min_m <= y <= altitude_max_m",
            "collision_source": "0.5 m Helsinki triangle-mesh-derived highest-surface map",
            "clearance_source": "4.0 m planning inflation; 2.5 m independent physical swept validation",
            "high_altitude_fallback_in_low_mode": False,
        },
        "debug_20": debug_summary,
        "qualification_200": qualification_summary,
        "verdict": verdict,
        "geometry_limitations": {
            "building": "represented conservatively by highest surface columns",
            "terrain": "represented by highest surface columns",
            "tree": "only represented where photogrammetry contributes to the highest surface; semantic/tree-complete safety is not proven",
            "bridge": "heightmap fills space below the highest deck and cannot represent fly-under free space",
            "overhang": "heightmap fills space below the highest surface and cannot represent under-overhang free space",
            "thin_structure": "may be lost or broadened by 0.5 m rasterization and 5 m conservative global pooling",
        },
        "elapsed_wall_time_s": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False), flush=True)
    return 0 if verdict.startswith("QUALIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
