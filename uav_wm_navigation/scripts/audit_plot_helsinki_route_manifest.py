#!/usr/bin/env python3
"""Audit and plot spatial coverage for a prepared Helsinki route manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

import _bootstrap  # noqa: F401

ROOT = _bootstrap.PROJECT_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_frames import backend_world_to_enu  # noqa: E402
from backend.engine.helsinki_navigation import HelsinkiNavigationStack  # noqa: E402
from backend.engine.helsinki_spatial_split import HelsinkiSpatialSplit  # noqa: E402
from backend.engine.helsinki_urban_sampling import HelsinkiUrbanDensity  # noqa: E402
from scripts.verify_helsinki_low_altitude_expert import TASK_TYPES  # noqa: E402


SCENE_ROOT = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
MAP_MIN_M = -500.0
MAP_MAX_M = 500.0
CELL_SIZE_M = 50.0
TASK_COLORS = {
    "building_blocked": "#0072B2",
    "street_canyon": "#E69F00",
    "rooftop_to_ground": "#009E73",
    "ground_to_rooftop": "#CC79A7",
    "rooftop_to_rooftop": "#D55E00",
}


def _cell(point_enu: np.ndarray) -> tuple[int, int]:
    return tuple(np.floor(np.asarray(point_enu)[0:2] / CELL_SIZE_M).astype(int))


def _cells(points_enu: np.ndarray) -> set[tuple[int, int]]:
    return {_cell(point) for point in np.asarray(points_enu)}


def _minimum_pair_distance(points: np.ndarray) -> float | None:
    return float(pdist(points).min()) if len(points) > 1 else None


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def audit_manifest(manifest_path: Path) -> tuple[dict, list[dict]]:
    records = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    stack = HelsinkiNavigationStack.load(SCENE_ROOT, enable_triangle_geometry=True)
    density = HelsinkiUrbanDensity(stack.low_altitude_grid)
    partition = HelsinkiSpatialSplit()
    train_mask = partition.masks(stack.low_altitude_grid)["train"]
    urban_mask = train_mask & density.non_open_mask

    eligible_train_cells: set[tuple[int, int]] = set()
    eligible_urban_cells: set[tuple[int, int]] = set()
    for ix, iz in zip(*np.where(train_mask)):
        point = stack.low_altitude_grid.grid_to_world_xz(int(ix), int(iz), 0.0)
        point_enu = backend_world_to_enu(point)
        eligible_train_cells.add(_cell(point_enu))
        if urban_mask[ix, iz]:
            eligible_urban_cells.add(_cell(point_enu))

    starts = []
    goals = []
    route_cells: set[tuple[int, int]] = set()
    task_counts = {task_type: 0 for task_type in TASK_TYPES}
    task_lengths = {task_type: [] for task_type in TASK_TYPES}
    train_leakage_count = 0
    triangle_collision_count = 0
    triangle_clearances = []
    lengths = []
    for record in records:
        route_backend = np.asarray(record["route_backend"], dtype=np.float64)
        route_enu = np.asarray(record["route_enu"], dtype=np.float64)
        task_type = str(record["task"]["task_type"])
        starts.append(route_enu[0])
        goals.append(route_enu[-1])
        route_cells.update(_cells(route_enu))
        length = float(np.linalg.norm(np.diff(route_enu, axis=0), axis=1).sum())
        lengths.append(length)
        task_counts[task_type] += 1
        task_lengths[task_type].append(length)
        train_leakage_count += int(partition.assign_backend_route(route_backend) != "train")
        triangle = dict(record.get("triangle_validation") or {})
        triangle_collision_count += int(bool(triangle.get("collision", True)))
        triangle_clearance = triangle.get(
            "minimum_clearance_m", triangle.get("minimum_distance_m")
        )
        if triangle_clearance is not None:
            triangle_clearances.append(float(triangle_clearance))

    starts = np.asarray(starts, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    start_cells = _cells(starts)
    goal_cells = _cells(goals)
    combined_endpoint_cells = start_cells | goal_cells
    urban_route_hit = route_cells & eligible_urban_cells
    train_route_hit = route_cells & eligible_train_cells
    exact_per_task = len(records) // len(TASK_TYPES) if records else 0
    gate_checks = {
        "episode_count_500": len(records) == 500,
        "task_balance_100_each": all(value == 100 for value in task_counts.values()),
        "train_split_leakage_zero": train_leakage_count == 0,
        "triangle_collision_zero": triangle_collision_count == 0,
        "urban_train_route_cell_coverage_at_least_90_percent": bool(
            eligible_urban_cells
            and len(urban_route_hit) / len(eligible_urban_cells) >= 0.90
        ),
    }
    report = {
        "status": "PASS" if all(gate_checks.values()) else "FAIL",
        "manifest": str(manifest_path.resolve()),
        "coordinate_frame": "canonical ENU metres",
        "scope": "Dataset v1 train partition; test/validation bands remain held out",
        "episode_count": len(records),
        "task_counts": task_counts,
        "expected_per_task": exact_per_task,
        "route_length_m": {
            "minimum": float(np.min(lengths)),
            "median": float(np.median(lengths)),
            "maximum": float(np.max(lengths)),
            "by_task_median": {
                key: float(np.median(values)) for key, values in task_lengths.items()
            },
        },
        "endpoint_coverage": {
            "start_minimum_pair_distance_m": _minimum_pair_distance(starts),
            "goal_minimum_pair_distance_m": _minimum_pair_distance(goals),
            "start_unique_50m_cells": len(start_cells),
            "goal_unique_50m_cells": len(goal_cells),
            "combined_unique_50m_cells": len(combined_endpoint_cells),
            "eligible_urban_train_50m_cells": len(eligible_urban_cells),
            "combined_endpoint_urban_cell_ratio": float(
                len(combined_endpoint_cells & eligible_urban_cells)
                / len(eligible_urban_cells)
            ),
            "start_bounds_enu": {
                "minimum": starts.min(axis=0).tolist(),
                "maximum": starts.max(axis=0).tolist(),
            },
            "goal_bounds_enu": {
                "minimum": goals.min(axis=0).tolist(),
                "maximum": goals.max(axis=0).tolist(),
            },
        },
        "route_coverage": {
            "visited_unique_50m_cells": len(route_cells),
            "eligible_train_50m_cells": len(eligible_train_cells),
            "eligible_urban_train_50m_cells": len(eligible_urban_cells),
            "train_cell_ratio": float(len(train_route_hit) / len(eligible_train_cells)),
            "urban_train_cell_ratio": float(len(urban_route_hit) / len(eligible_urban_cells)),
        },
        "geometry": {
            "train_split_leakage_count": train_leakage_count,
            "triangle_collision_count": triangle_collision_count,
            "minimum_planned_triangle_clearance_m": (
                float(np.min(triangle_clearances)) if triangle_clearances else None
            ),
        },
        "gate_checks": gate_checks,
    }
    return report, records


def plot_manifest(records: list[dict], report: dict, output_path: Path) -> None:
    edges = np.arange(MAP_MIN_M, MAP_MAX_M + CELL_SIZE_M, CELL_SIZE_M)
    route_points = np.concatenate(
        [np.asarray(record["route_enu"], dtype=np.float64) for record in records], axis=0
    )
    starts = np.asarray([record["route_enu"][0] for record in records], dtype=np.float64)
    goals = np.asarray([record["route_enu"][-1] for record in records], dtype=np.float64)
    route_hist, _, _ = np.histogram2d(route_points[:, 0], route_points[:, 1], bins=(edges, edges))
    start_hist, _, _ = np.histogram2d(starts[:, 0], starts[:, 1], bins=(edges, edges))
    goal_hist, _, _ = np.histogram2d(goals[:, 0], goals[:, 1], bins=(edges, edges))

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.45, 1.0))
    route_ax = fig.add_subplot(grid[:, 0])
    route_ax.axhspan(-280.0, 80.0, color="#E8F3E8", zorder=0, label="Train partition")
    route_ax.axhspan(120.0, 280.0, color="#EEEEEE", zorder=0, label="Held-out test")
    route_ax.axhspan(-480.0, -320.0, color="#E2E2E2", zorder=0, label="Held-out validation")
    route_ax.axhspan(300.0, 500.0, color="#F5F5F5", zorder=0, label="Excluded open/water band")
    seen = set()
    for record in records:
        route = np.asarray(record["route_enu"], dtype=np.float64)
        task_type = str(record["task"]["task_type"])
        label = task_type if task_type not in seen else None
        seen.add(task_type)
        route_ax.plot(
            route[:, 0], route[:, 1], color=TASK_COLORS[task_type], alpha=0.24,
            linewidth=0.75, label=label,
        )
    route_ax.scatter(starts[:, 0], starts[:, 1], s=7, c="#111111", alpha=0.45, marker="o", label="Start")
    route_ax.scatter(goals[:, 0], goals[:, 1], s=9, c="#111111", alpha=0.45, marker="x", label="Goal")
    route_ax.set_title("500 planned routes across the Dataset v1 train partition")
    route_ax.set_xlabel("East (m)")
    route_ax.set_ylabel("North (m)")
    route_ax.set_xlim(MAP_MIN_M, MAP_MAX_M)
    route_ax.set_ylim(MAP_MIN_M, MAP_MAX_M)
    route_ax.set_aspect("equal")
    route_ax.grid(alpha=0.15)
    route_ax.legend(loc="upper left", ncol=2, fontsize=8)

    def heatmap(axis, values, title, cmap):
        image = axis.imshow(
            values.T,
            extent=(MAP_MIN_M, MAP_MAX_M, MAP_MIN_M, MAP_MAX_M),
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            aspect="equal",
        )
        axis.axhline(-280.0, color="white", linewidth=0.8, alpha=0.8)
        axis.axhline(80.0, color="white", linewidth=0.8, alpha=0.8)
        axis.set_xlim(MAP_MIN_M, MAP_MAX_M)
        axis.set_ylim(-500.0, 300.0)
        axis.set_title(title)
        axis.set_xlabel("East (m)")
        axis.set_ylabel("North (m)")
        fig.colorbar(image, ax=axis, shrink=0.78, label="Count per 50 m cell")

    heatmap(fig.add_subplot(grid[0, 1]), route_hist, "Route corridor coverage", "viridis")
    endpoint_ax = fig.add_subplot(grid[1, 1])
    endpoint_total = start_hist + goal_hist
    heatmap(endpoint_ax, endpoint_total, "Start + goal coverage", "magma")
    coverage = report["route_coverage"]["urban_train_cell_ratio"]
    minimum_clearance = report["geometry"]["minimum_planned_triangle_clearance_m"]
    fig.suptitle(
        f"Helsinki Dataset v1 route plan — {report['status']} | "
        f"urban train coverage {coverage:.1%} | "
        f"planned triangle clearance min {minimum_clearance:.2f} m",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    report, records = audit_manifest(args.manifest)
    _atomic_json(args.output_json.resolve(), report)
    plot_manifest(records, report, args.output_png.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
