#!/usr/bin/env python3
"""Qualify a dense, low-altitude Helsinki building-canyon route."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import heapq
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_navigation import HelsinkiNavigationStack  # noqa: E402
from backend.engine.helsinki_urban_sampling import HelsinkiUrbanDensity  # noqa: E402
from backend.engine.planner import PlanningError  # noqa: E402
from scripts.verify_helsinki_low_altitude_expert import DifficultTaskSampler  # noqa: E402


SCENE = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"


def _turn_metrics(points: np.ndarray, minimum_leg_m: float = 0.0) -> tuple[int, float]:
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    angles = []
    for incoming, outgoing, a, b in zip(vectors[:-1], vectors[1:], lengths[:-1], lengths[1:]):
        if a < max(1e-6, minimum_leg_m) or b < max(1e-6, minimum_leg_m):
            continue
        angle = math.degrees(math.acos(float(np.clip(np.dot(incoming, outgoing) / (a * b), -1.0, 1.0))))
        if angle >= 20.0:
            angles.append(angle)
    return len(angles), float(sum(angles))


def _bounded_safe_shortcut(stack, points: np.ndarray, maximum_grid_steps: int = 10) -> np.ndarray:
    """Collision-checked shortcut that keeps successive city-street turns."""

    values = np.asarray(points, dtype=float)
    result = [values[0]]
    index = 0
    while index < len(values) - 1:
        next_index = index + 1
        furthest = min(len(values) - 1, index + maximum_grid_steps)
        for candidate in range(furthest, index, -1):
            if stack._segment_valid(values[index], values[candidate]):
                next_index = candidate
                break
        result.append(values[next_index])
        index = next_index
    return np.asarray(result, dtype=float)


def _sample_indices(points: np.ndarray, spacing_m: float = 5.0) -> np.ndarray:
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    targets = np.arange(0.0, distance[-1] + 1e-6, spacing_m)
    return np.unique(np.searchsorted(distance, targets, side="left").clip(0, len(points) - 1))


def _topology_metrics(stack, trajectory: np.ndarray) -> dict:
    """Measure whether a route is a genuine one-way path instead of a tour."""

    points = np.asarray(trajectory, dtype=float)
    planar_length = float(np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1).sum())
    displacement = float(np.linalg.norm(points[-1, [0, 2]] - points[0, [0, 2]]))
    sampled = points[_sample_indices(points, spacing_m=float(stack.grid.resolution))]
    sampled_cells = [tuple(stack.grid.world_to_grid_xz(point)) for point in sampled]
    collapsed_cells = []
    for cell in sampled_cells:
        if not collapsed_cells or cell != collapsed_cells[-1]:
            collapsed_cells.append(cell)
    undirected_edges = [
        tuple(sorted((a, b))) for a, b in zip(collapsed_cells[:-1], collapsed_cells[1:])
    ]
    return {
        "straight_planar_distance_m": displacement,
        "path_to_displacement_ratio": planar_length / max(displacement, 1e-6),
        "trajectory_reentered_grid_cell_count": int(
            len(collapsed_cells) - len(set(collapsed_cells))
        ),
        "trajectory_repeated_edge_count": int(
            len(undirected_edges) - len(set(undirected_edges))
        ),
    }


def _encountered_building_count(stack, building_mask: np.ndarray, trajectory: np.ndarray) -> int:
    """Count distinct large connected building masses encountered within 35 m."""

    labels, _ = ndimage.label(building_mask, structure=np.ones((3, 3), dtype=np.uint8))
    radius_cells = int(math.ceil(35.0 / float(stack.grid.resolution)))
    encountered = set()
    for point in np.asarray(trajectory)[_sample_indices(trajectory, spacing_m=10.0)]:
        gx, gz = stack.grid.world_to_grid_xz(point)
        x0, x1 = max(0, gx - radius_cells), min(labels.shape[0], gx + radius_cells + 1)
        z0, z1 = max(0, gz - radius_cells), min(labels.shape[1], gz + radius_cells + 1)
        local = labels[x0:x1, z0:z1]
        encountered.update(int(value) for value in np.unique(local) if value > 0)
    return len(encountered)


def _bilateral_building_ratio(stack, building_mask: np.ndarray, trajectory: np.ndarray) -> float:
    """Route fraction with large building mass within 30 m on both sides."""

    points = np.asarray(trajectory, dtype=float)
    hits = 0
    tested = 0
    for index in _sample_indices(points):
        before = max(0, index - 2)
        after = min(len(points) - 1, index + 2)
        tangent = points[after, [0, 2]] - points[before, [0, 2]]
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            continue
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float) / norm
        side_hits = []
        for sign in (-1.0, 1.0):
            found = False
            for distance_m in np.arange(5.0, 31.0, 2.5):
                probe = points[index].copy()
                probe[[0, 2]] += sign * distance_m * normal
                try:
                    gx, gz = stack.grid.world_to_grid_xz(probe)
                except (IndexError, ValueError):
                    continue
                if bool(building_mask[gx, gz]):
                    found = True
                    break
            side_hits.append(found)
        tested += 1
        hits += int(all(side_hits))
    return float(hits / tested) if tested else 0.0


def _route_metrics(stack, density, strict_mask, building_coverage, building_mask, task, plan) -> dict:
    trajectory = np.asarray(plan.trajectory, dtype=float)
    samples = trajectory[_sample_indices(trajectory)]
    cells = [density.world_to_cell(point) for point in samples]
    core = [bool(density.dense_urban_core_mask[cell]) for cell in cells]
    strict = [bool(strict_mask[cell]) for cell in cells]
    building_context = [float(building_coverage[cell]) for cell in cells]
    obstacle = [density.obstacle_density_at(point) for point in samples]
    scores = [density.score_at(point) for point in samples]
    turn_count, turn_angle = _turn_metrics(np.asarray(plan.simplified_path, dtype=float))
    triangle = plan.triangle_validation or {}
    return {
        **asdict(task),
        "path_length_m": float(plan.path_length_m),
        "trajectory_points": int(len(trajectory)),
        "minimum_heightmap_clearance_m": float(plan.validation["minimum_clearance_m"]),
        "minimum_triangle_distance_m": float(triangle.get("minimum_distance_m", -1.0)),
        "triangle_collision": bool(triangle.get("collision", True)),
        "minimum_altitude_m": float(np.min(trajectory[:, 1])),
        "maximum_altitude_m": float(np.max(trajectory[:, 1])),
        "dense_core_route_ratio": float(np.mean(core)),
        "strict_building_district_route_ratio": float(np.mean(strict)),
        "mean_large_building_coverage": float(np.mean(building_context)),
        "mean_route_obstacle_coverage": float(np.mean(obstacle)),
        "p25_route_obstacle_coverage": float(np.percentile(obstacle, 25)),
        "mean_route_density_score": float(np.mean(scores)),
        "bilateral_building_ratio": _bilateral_building_ratio(stack, building_mask, trajectory),
        "turn_count": turn_count,
        "total_turn_angle_degrees": turn_angle,
        "planner_mode": plan.planner_mode,
        "mission_waypoints_backend": np.asarray(plan.simplified_path, dtype=float).tolist(),
    }


def _qualified(item: dict) -> bool:
    return bool(
        item["path_length_m"] >= 300.0
        and item.get("straight_planar_distance_m", 0.0) >= 150.0
        and item.get("path_to_displacement_ratio", float("inf")) <= 2.7
        and item.get("source_graph_repeated_cell_count", 1) == 0
        and item.get("source_graph_repeated_edge_count", 1) == 0
        and item.get("trajectory_reentered_grid_cell_count", 1) == 0
        and item.get("trajectory_repeated_edge_count", 1) == 0
        and item["maximum_altitude_m"] <= 30.0 + 1e-6
        and not item["triangle_collision"]
        and item["minimum_triangle_distance_m"] >= 7.5
        and item["minimum_heightmap_clearance_m"] >= 6.0
        and item["strict_building_district_route_ratio"] >= 0.60
        and item["mean_large_building_coverage"] >= 0.25
        and item["mean_route_obstacle_coverage"] >= 0.12
        and item["bilateral_building_ratio"] >= 0.30
        # Adjacent Helsinki photogrammetry buildings are frequently welded
        # into one connected mesh mass; require two independent masses plus
        # bilateral context instead of pretending every facade is segmented.
        and item.get("encountered_large_building_count", 0) >= 2
        and item.get("turn_count", 0) >= 10
        and item["total_turn_angle_degrees"] >= 60.0
    )


def _score(item: dict) -> float:
    return float(
        min(item["path_length_m"], 650.0) / 650.0
        + 1.8 * item["dense_core_route_ratio"]
        + 2.5 * item["strict_building_district_route_ratio"]
        + 2.5 * item["bilateral_building_ratio"]
        + 1.5 * item["mean_route_obstacle_coverage"]
        + min(item["total_turn_angle_degrees"], 360.0) / 360.0
        - 0.025 * item["maximum_altitude_m"]
    )


def _chain_metrics(stack, density, strict_mask, building_coverage, building_mask, plans, waypoints) -> tuple[dict, np.ndarray]:
    trajectory = np.concatenate(
        [plan.trajectory if index == 0 else plan.trajectory[1:] for index, plan in enumerate(plans)],
        axis=0,
    )
    simplified = np.concatenate(
        [plan.simplified_path if index == 0 else plan.simplified_path[1:] for index, plan in enumerate(plans)],
        axis=0,
    )
    samples = trajectory[_sample_indices(trajectory)]
    cells = [density.world_to_cell(point) for point in samples]
    obstacle = [density.obstacle_density_at(point) for point in samples]
    scores = [density.score_at(point) for point in samples]
    turns, turn_angle = _turn_metrics(simplified)
    triangle_distances = [
        float(plan.triangle_validation["minimum_distance_m"])
        for plan in plans
        if plan.triangle_validation is not None
    ]
    metrics = {
        "index": "multi_segment_chain",
        "task_type": "street_canyon_chain",
        "start": np.asarray(waypoints[0], dtype=float).tolist(),
        "goal": np.asarray(waypoints[-1], dtype=float).tolist(),
        "altitude_min_m": 4.5,
        "altitude_max_m": 25.0,
        "spatial_stratum": "strict_building_district",
        "path_length_m": float(sum(plan.path_length_m for plan in plans)),
        "trajectory_points": int(len(trajectory)),
        "minimum_heightmap_clearance_m": float(
            min(plan.validation["minimum_clearance_m"] for plan in plans)
        ),
        "minimum_triangle_distance_m": min(triangle_distances, default=-1.0),
        "triangle_collision": any(
            plan.triangle_validation is None or plan.triangle_validation["collision"]
            for plan in plans
        ),
        "minimum_altitude_m": float(np.min(trajectory[:, 1])),
        "maximum_altitude_m": float(np.max(trajectory[:, 1])),
        "dense_core_route_ratio": float(np.mean([density.dense_urban_core_mask[cell] for cell in cells])),
        "strict_building_district_route_ratio": float(np.mean([strict_mask[cell] for cell in cells])),
        "mean_large_building_coverage": float(np.mean([building_coverage[cell] for cell in cells])),
        "mean_route_obstacle_coverage": float(np.mean(obstacle)),
        "p25_route_obstacle_coverage": float(np.percentile(obstacle, 25)),
        "mean_route_density_score": float(np.mean(scores)),
        "bilateral_building_ratio": _bilateral_building_ratio(stack, building_mask, trajectory),
        "turn_count": turns,
        "total_turn_angle_degrees": turn_angle,
        "planner_mode": "bounded_xyz_astar_multi_segment",
        "segment_count": len(plans),
        "mission_waypoints_backend": [np.asarray(point, dtype=float).tolist() for point in waypoints],
    }
    metrics["qualified"] = _qualified(metrics) and len(plans) >= 5
    metrics["rank_score"] = _score(metrics) + 0.2 * min(len(plans), 8)
    return metrics, trajectory


def _search_chain(stack, density, strict_mask, building_coverage, building_mask, sampler, seed: int, attempts: int = 4):
    """Force a long mission through successive qualified street-canyon legs."""

    rng = np.random.default_rng(seed + 7001)
    pool = sampler._pool("street", 25.0, strict_mask)
    if not len(pool):
        return []
    chains = []
    for attempt in range(attempts):
        start_cell = pool[int(rng.integers(0, len(pool)))]
        current = sampler._world_endpoint(start_cell, "street")
        waypoints = [current]
        plans = []
        previous_direction = None
        for segment_index in range(6):
            deltas = (pool - start_cell[None]) * sampler.resolution
            distances = np.linalg.norm(deltas, axis=1)
            eligible = np.flatnonzero((distances >= 75.0) & (distances <= 155.0))
            if len(eligible) == 0:
                break
            rng.shuffle(eligible)
            best = None
            for pool_index in eligible[:6]:
                goal_cell = pool[pool_index]
                goal = sampler._world_endpoint(goal_cell, "street")
                if any(np.linalg.norm((goal - old)[[0, 2]]) < 45.0 for old in waypoints[:-1]):
                    continue
                direction = (goal - current)[[0, 2]]
                if previous_direction is not None:
                    cosine = float(np.clip(
                        np.dot(previous_direction, direction)
                        / max(np.linalg.norm(previous_direction) * np.linalg.norm(direction), 1e-6),
                        -1.0,
                        1.0,
                    ))
                    turn = math.degrees(math.acos(cosine))
                    if turn < 25.0 or turn > 145.0:
                        continue
                try:
                    plan = stack.plan(
                        current,
                        goal,
                        expert_mode="low_altitude_3d",
                        altitude_min_m=4.5,
                        altitude_max_m=25.0,
                        allow_layer_transitions=True,
                    )
                except PlanningError:
                    continue
                segment = np.asarray(plan.trajectory, dtype=float)
                sample_cells = [density.world_to_cell(point) for point in segment[_sample_indices(segment)]]
                strict_ratio = float(np.mean([strict_mask[cell] for cell in sample_cells]))
                bilateral = _bilateral_building_ratio(stack, building_mask, segment)
                if strict_ratio < 0.58 or bilateral < 0.25:
                    continue
                value = strict_ratio + 1.5 * bilateral - abs(plan.path_length_m - 120.0) / 500.0
                if best is None or value > best[0]:
                    best = (value, goal_cell, goal, direction, plan)
            if best is None:
                break
            _, start_cell, current, previous_direction, plan = best
            plans.append(plan)
            waypoints.append(current)
            print(
                f"chain {attempt + 1}/{attempts} accepted segment {segment_index + 1}/6 "
                f"length={plan.path_length_m:.1f}m",
                flush=True,
            )
        if len(plans) >= 3:
            metrics, trajectory = _chain_metrics(
                stack, density, strict_mask, building_coverage, building_mask, plans, waypoints
            )
            chains.append((metrics, trajectory))
            print(
                f"chain {attempt + 1}/{attempts} segments={len(plans)} length={metrics['path_length_m']:.1f} "
                f"strict={metrics['strict_building_district_route_ratio']:.0%} "
                f"bilateral={metrics['bilateral_building_ratio']:.0%} qualified={metrics['qualified']}",
                flush=True,
            )
    return chains


def _dijkstra(mask: np.ndarray, start: tuple[int, int], resolution: float, coverage: np.ndarray):
    distances = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]
    farthest = start
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost != distances.get(cell):
            continue
        if cost > distances[farthest]:
            farthest = cell
        for dx, dz in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            neighbor = (cell[0] + dx, cell[1] + dz)
            if (
                neighbor[0] < 0
                or neighbor[1] < 0
                or neighbor[0] >= mask.shape[0]
                or neighbor[1] >= mask.shape[1]
                or not mask[neighbor]
            ):
                continue
            step = (
                resolution
                * math.hypot(dx, dz)
                * (1.0 + 1.5 * max(0.0, 0.45 - float(coverage[neighbor])))
            )
            candidate = cost + step
            if candidate + 1e-9 < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = cell
                heapq.heappush(queue, (candidate, neighbor))
    return farthest, distances, previous


def _search_corridor_graph(stack, density, strict_mask, building_coverage, building_mask):
    """Find the long geodesic of a low-altitude street component."""

    # Search connectivity conservatively at 20 m, then execute the exact same
    # corridor at 24 m.  The 30--45 m block roofs remain above the vehicle,
    # while the trained controller gains vertical margin over low clutter.
    # connections between several internal street rows that are disconnected
    # by conservative photogrammetry clutter at 18 m.
    cruise_altitude = 20.0
    execution_altitude = 27.0
    resolution = float(stack.grid.resolution)
    district_window = max(3, int(round(250.0 / resolution)))
    if district_window % 2 == 0:
        district_window += 1
    district_score = ndimage.uniform_filter(
        building_mask.astype(np.float32), size=district_window, mode="nearest"
    )
    district_score = district_score.copy()
    district_score[density.distance_to_boundary_m < density.edge_exclusion_m] = -1.0
    district_center = np.unravel_index(int(np.argmax(district_score)), district_score.shape)
    gx, gz = np.indices(district_score.shape)
    district_radius_m = 350.0
    district_mask = (
        np.hypot(gx - district_center[0], gz - district_center[1]) * resolution
        <= district_radius_m
    )
    free = (
        np.asarray(stack.grid.heightmap, dtype=float)
        + float(stack.low_altitude_planning_clearance)
        + 0.5
        <= cruise_altitude
    )
    planar_center_clearance = (
        ndimage.distance_transform_edt(free) * resolution
    )
    context_threshold = 0.16
    corridor = (
        free
        & density.non_open_mask
        & (building_coverage >= context_threshold)
        & (density.distance_to_boundary_m >= density.edge_exclusion_m)
        & district_mask
        & (planar_center_clearance >= 5.0)
    )
    # Reserve real triangle-mesh margin for tracking error, not just the UAV
    # sphere.  This removes narrow passages that are geometrically collision-
    # free on the exact line but infeasible for the existing learned policy.
    corridor_cells = np.column_stack(np.where(corridor))
    triangle_center_clearance = np.zeros_like(building_coverage, dtype=np.float32)
    if len(corridor_cells):
        corridor_points = np.asarray(
            [
                stack.grid.grid_to_world_xz(int(gx), int(gz), execution_altitude)
                for gx, gz in corridor_cells
            ],
            dtype=float,
        )
        _, corridor_distances, _ = stack.local_triangle_geometry._closest(corridor_points)
        triangle_center_clearance[
            corridor_cells[:, 0], corridor_cells[:, 1]
        ] = corridor_distances.astype(np.float32)
    corridor &= triangle_center_clearance >= 8.0
    # Work on a one-cell-wide medial skeleton.  Walking the full-width free
    # mask creates grid zig-zags inside a street; the skeleton represents the
    # actual street topology and keeps the line visually and kinematically clean.
    corridor = skeletonize(corridor)
    labels, count = ndimage.label(corridor, structure=np.ones((3, 3), dtype=np.uint8))
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    component_order = np.argsort(sizes[1:])[::-1] + 1
    best = None
    for component_id in component_order[:8]:
        component = labels == component_id
        if int(sizes[component_id]) < 40:
            continue
        first_cell = tuple(np.column_stack(np.where(component))[0])
        endpoint_a, _, _ = _dijkstra(component, first_cell, resolution, building_coverage)
        endpoint_b, distances, previous = _dijkstra(component, endpoint_a, resolution, building_coverage)
        cells = [endpoint_b]
        while cells[-1] != endpoint_a:
            parent = previous.get(cells[-1])
            if parent is None:
                break
            cells.append(parent)
        cells.reverse()
        diameter_cells = cells
        component_cells = np.column_stack(np.where(component))
        anchor_indices = {
            int(np.argmin(component_cells[:, 0])),
            int(np.argmax(component_cells[:, 0])),
            int(np.argmin(component_cells[:, 1])),
            int(np.argmax(component_cells[:, 1])),
            int(np.argmin(component_cells[:, 0] + component_cells[:, 1])),
            int(np.argmax(component_cells[:, 0] + component_cells[:, 1])),
        }
        anchors = [tuple(component_cells[index]) for index in sorted(anchor_indices)]
        # Search only self-avoiding walks.  Repeated cells or edges are never
        # allowed to inflate distance: the final route must enter the urban
        # maze once and leave it at a genuinely different endpoint.
        rng = np.random.default_rng(20260831 + int(component_id))
        component_set = {tuple(cell) for cell in component_cells}
        best_simple = []
        best_simple_score = -float("inf")
        starts = list(dict.fromkeys([endpoint_a, endpoint_b, *anchors]))
        for trial in range(12000):
            current = starts[trial % len(starts)]
            walk = [current]
            visited = {current}
            previous_direction = None
            while True:
                options = []
                for dx, dz in (
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                ):
                    neighbor = (current[0] + dx, current[1] + dz)
                    if neighbor not in component_set or neighbor in visited:
                        continue
                    onward = 0
                    for ex, ez in (
                        (-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1),
                    ):
                        probe = (neighbor[0] + ex, neighbor[1] + ez)
                        onward += int(probe in component_set and probe not in visited)
                    direction = (dx, dz)
                    turn_penalty = int(
                        previous_direction is not None and direction != previous_direction
                    )
                    options.append((onward + 0.18 * turn_penalty + float(rng.random()) * 0.45, neighbor, direction))
                if not options:
                    break
                _, current, previous_direction = min(options)
                walk.append(current)
                visited.add(current)
            length_m = float(
                sum(
                    math.hypot(b[0] - a[0], b[1] - a[1]) * resolution
                    for a, b in zip(walk[:-1], walk[1:])
                )
            )
            separation_m = float(
                np.linalg.norm((np.asarray(walk[-1]) - np.asarray(walk[0])) * resolution)
            )
            stretch = length_m / max(separation_m, 1e-6)
            raw_turns = sum(
                (walk[index][0] - walk[index - 1][0], walk[index][1] - walk[index - 1][1])
                != (walk[index + 1][0] - walk[index][0], walk[index + 1][1] - walk[index][1])
                for index in range(1, len(walk) - 1)
            )
            valid_one_way = (
                length_m >= 300.0
                and separation_m >= 150.0
                and stretch <= 2.7
                and raw_turns >= 12
            )
            score = (
                length_m
                + 0.65 * separation_m
                + 12.0 * min(raw_turns, 14)
                - 120.0 * max(0.0, stretch - 2.0)
            )
            if valid_one_way and score > best_simple_score:
                best_simple = walk
                best_simple_score = score
        if best_simple:
            cells = best_simple
        else:
            # A component diameter is non-repeating by construction and is a
            # valid fallback only if it independently satisfies the same Gate.
            cells = diameter_cells
        raw = np.asarray(
            [stack.grid.grid_to_world_xz(gx, gz, cruise_altitude) for gx, gz in cells],
            dtype=float,
        )
        if len(raw) < 2:
            continue
        # Preserve the self-avoiding street topology.  Only redundant
        # collinear samples are removed; a global shortcut would erase the
        # maze turns and turn the route back into a trivial perimeter path.
        simplified = _bounded_safe_shortcut(stack, raw, maximum_grid_steps=8)
        smoothed = stack._corner_smooth(
            simplified,
            radius_override=20.0,
        )
        trajectory = stack._densify(smoothed)
        trajectory[:, 1] = execution_altitude
        simplified = np.asarray(simplified, dtype=float).copy()
        simplified[:, 1] = execution_altitude
        validation = stack.validate_path(
            trajectory,
            required_clearance=stack.required_clearance,
            altitude_min_m=4.5,
            altitude_max_m=30.0,
        )
        if not validation["path_valid"]:
            print(
                f"corridor component {component_id} rejected by heightmap audit "
                f"{validation}",
                flush=True,
            )
            continue
        triangle = stack.local_triangle_geometry.trajectory_query(
            trajectory, stack.required_clearance
        ).as_dict()
        if triangle["collision"]:
            print(
                f"corridor component {component_id} rejected by triangle audit "
                f"distance={triangle['minimum_distance_m']:.3f}m",
                flush=True,
            )
            continue
        samples = trajectory[_sample_indices(trajectory)]
        sample_cells = [density.world_to_cell(point) for point in samples]
        geometric_turns, geometric_turn_angle = _turn_metrics(simplified)
        turns, turn_angle = _turn_metrics(simplified, minimum_leg_m=18.0)
        kinematic_leg_lengths = np.linalg.norm(
            np.diff(simplified[:, [0, 2]], axis=0), axis=1
        )
        source_edges = [tuple(sorted((a, b))) for a, b in zip(cells[:-1], cells[1:])]
        topology = _topology_metrics(stack, trajectory)
        # Explicit mission points at all real corners plus a maximum spacing of
        # about 55 m.  This prevents the runtime planner from escaping to a
        # visually open peripheral route between sparse semantic waypoints.
        waypoint_indices = set(_sample_indices(trajectory, spacing_m=40.0).tolist())
        waypoint_indices.update({0, len(trajectory) - 1})
        waypoint_candidates = trajectory[sorted(waypoint_indices)]
        retained = [waypoint_candidates[0]]
        for point in waypoint_candidates[1:-1]:
            if float(np.linalg.norm(point - retained[-1])) >= 20.0:
                retained.append(point)
        final_point = waypoint_candidates[-1]
        if len(retained) > 1 and float(np.linalg.norm(final_point - retained[-1])) < 12.0:
            retained[-1] = final_point
        else:
            retained.append(final_point)
        waypoints = np.asarray(retained, dtype=float)
        metrics = {
            "index": f"corridor_component_{component_id}",
            "task_type": "street_canyon_graph",
            "start": trajectory[0].tolist(),
            "goal": trajectory[-1].tolist(),
            "altitude_min_m": 4.5,
            "altitude_max_m": 30.0,
            "spatial_stratum": "large_building_street_component",
            "path_length_m": float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()),
            **topology,
            "source_graph_repeated_cell_count": int(len(cells) - len(set(cells))),
            "source_graph_repeated_edge_count": int(len(source_edges) - len(set(source_edges))),
            "trajectory_points": int(len(trajectory)),
            "minimum_heightmap_clearance_m": float(validation["minimum_clearance_m"]),
            "minimum_triangle_distance_m": float(triangle["minimum_distance_m"]),
            "triangle_collision": bool(triangle["collision"]),
            "minimum_altitude_m": float(np.min(trajectory[:, 1])),
            "maximum_altitude_m": float(np.max(trajectory[:, 1])),
            "dense_core_route_ratio": float(np.mean([density.dense_urban_core_mask[cell] for cell in sample_cells])),
            "strict_building_district_route_ratio": float(np.mean([strict_mask[cell] for cell in sample_cells])),
            "mean_large_building_coverage": float(np.mean([building_coverage[cell] for cell in sample_cells])),
            "mean_route_obstacle_coverage": float(np.mean([density.obstacle_density_at(point) for point in samples])),
            "p25_route_obstacle_coverage": float(np.percentile([density.obstacle_density_at(point) for point in samples], 25)),
            "mean_route_density_score": float(np.mean([density.score_at(point) for point in samples])),
            "bilateral_building_ratio": _bilateral_building_ratio(stack, building_mask, trajectory),
            "encountered_large_building_count": _encountered_building_count(
                stack, building_mask, trajectory
            ),
            "turn_count": turns,
            "total_turn_angle_degrees": turn_angle,
            "geometric_turn_count_including_short_grid_corrections": geometric_turns,
            "geometric_turn_angle_degrees": geometric_turn_angle,
            "kinematic_turn_minimum_incoming_and_outgoing_leg_m": 18.0,
            "minimum_kinematic_leg_m": float(
                np.min(kinematic_leg_lengths) if len(kinematic_leg_lengths) else 0.0
            ),
            "mean_kinematic_leg_m": float(
                np.mean(kinematic_leg_lengths) if len(kinematic_leg_lengths) else 0.0
            ),
            "minimum_graph_triangle_center_clearance_m": 8.0,
            "planner_mode": "geometry_weighted_street_graph_then_frozen_safe_smoothing",
            "segment_count": int(len(waypoints) - 1),
            "mission_waypoints_backend": waypoints.tolist(),
            "corridor_component_cells": int(sizes[component_id]),
            "corridor_context_threshold": context_threshold,
            "weighted_geodesic_cost": float(distances[endpoint_b]),
            "geometry_selected_district_center_cell": [
                int(district_center[0]), int(district_center[1])
            ],
            "geometry_selected_district_center_backend": stack.grid.grid_to_world_xz(
                int(district_center[0]), int(district_center[1]), cruise_altitude
            ).tolist(),
            "geometry_selected_district_radius_m": district_radius_m,
            "geometry_selected_district_building_coverage": float(
                district_score[district_center]
            ),
        }
        metrics["qualified"] = _qualified(metrics)
        metrics["rank_score"] = _score(metrics)
        if best is None or metrics["rank_score"] > best[0]["rank_score"]:
            best = (metrics, trajectory)
    return best


def _plot(stack, density, strict_mask, trajectory: np.ndarray, result: dict, output: Path) -> None:
    height = np.asarray(stack.collision_map.height, dtype=float)
    extent = [
        stack.collision_map.origin_x,
        stack.collision_map.origin_x + (height.shape[1] - 1) * stack.collision_map.resolution,
        stack.collision_map.origin_z,
        stack.collision_map.maximum_z,
    ]
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.4), constrained_layout=True)
    image = None
    for axis in axes:
        image = axis.imshow(height, extent=extent, origin="upper", cmap="terrain", vmin=-2, vmax=46)
        axis.plot(trajectory[:, 0], trajectory[:, 2], color="#00e5ff", lw=2.8, label="单向规划主线")
        axis.scatter(trajectory[0, 0], trajectory[0, 2], c="#00df81", s=70, edgecolors="black", label="起点")
        axis.scatter(trajectory[-1, 0], trajectory[-1, 2], c="#ff4fb3", s=90, marker="*", edgecolors="black", label="终点")
        axis.set_aspect("equal")
        axis.set(xlabel="东向 x (m)", ylabel="北向 z (m)")
    axes[0].set_title("Helsinki 全地图 · 城市核心单向穿梭主线")
    margin = 45.0
    axes[1].set_xlim(float(np.min(trajectory[:, 0]) - margin), float(np.max(trajectory[:, 0]) + margin))
    axes[1].set_ylim(float(np.min(trajectory[:, 2]) - margin), float(np.max(trajectory[:, 2]) + margin))
    axes[1].set_title(
        f"楼宇迷宫单向路线 · {result['path_length_m']:.0f} m · 位移比 {result['path_to_displacement_ratio']:.2f}"
    )
    axes[1].legend(loc="best", fontsize=9)
    figure.colorbar(image, ax=axes, label="最高表面高度 (m)", shrink=0.82)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--distance-min-m", type=float, default=300.0)
    parser.add_argument("--distance-max-m", type=float, default=520.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    stack = HelsinkiNavigationStack.load(args.scene, enable_triangle_geometry=True)
    density = HelsinkiUrbanDensity(stack.grid)
    # Photogrammetry has no semantic labels.  Reject tree rows and isolated
    # high clutter by retaining only connected >=12 m surfaces whose footprint
    # is at least 500 m², then require sustained local coverage by those large
    # masses.  This is a geometry-derived building-district proxy.
    high_surface = np.asarray(stack.grid.heightmap, dtype=float) >= 12.0
    labels, count = ndimage.label(high_surface, structure=np.ones((3, 3), dtype=np.uint8))
    sizes_m2 = np.bincount(labels.ravel()) * float(stack.grid.resolution) ** 2
    keep = sizes_m2 >= 500.0
    if len(keep):
        keep[0] = False
    building_mask = keep[labels] if count else np.zeros_like(high_surface)
    window = max(3, int(round(75.0 / float(stack.grid.resolution))))
    if window % 2 == 0:
        window += 1
    building_coverage = ndimage.uniform_filter(
        building_mask.astype(np.float32), size=window, mode="nearest"
    )
    strict_threshold = 0.16
    strict_mask = (
        (building_coverage >= strict_threshold)
        & (density.distance_to_boundary_m >= density.edge_exclusion_m)
    )
    sampler = DifficultTaskSampler(stack, args.seed, urban_density=density)
    records = []
    paths = []
    failures = []
    for index in range(args.candidates):
        task_type = "street_canyon" if index % 2 else "building_blocked"
        try:
            task = sampler.sample(
                index,
                task_type,
                spatial_stratum="dense_core",
                start_spatial_mask=strict_mask,
                goal_spatial_mask=strict_mask,
                distance_range_override=(args.distance_min_m, args.distance_max_m),
            )
            plan = stack.plan(
                np.asarray(task.start, dtype=float),
                np.asarray(task.goal, dtype=float),
                expert_mode="low_altitude_3d",
                altitude_min_m=task.altitude_min_m,
                altitude_max_m=task.altitude_max_m,
                allow_layer_transitions=True,
            )
            record = _route_metrics(
                stack, density, strict_mask, building_coverage, building_mask, task, plan
            )
            record["qualified"] = _qualified(record)
            record["rank_score"] = _score(record)
            records.append(record)
            paths.append(np.asarray(plan.trajectory, dtype=float))
            print(
                f"{index + 1}/{args.candidates} {task_type} length={record['path_length_m']:.1f} "
                f"strict={record['strict_building_district_route_ratio']:.0%} "
                f"bilateral={record['bilateral_building_ratio']:.0%} "
                f"turns={record['turn_count']} qualified={record['qualified']}",
                flush=True,
            )
        except (PlanningError, RuntimeError) as error:
            failures.append({"index": index, "task_type": task_type, "error": str(error)})
            print(f"{index + 1}/{args.candidates} {task_type} FAIL {error}", flush=True)

    corridor_result = _search_corridor_graph(
        stack, density, strict_mask, building_coverage, building_mask
    )
    if corridor_result is not None:
        corridor_record, corridor_path = corridor_result
        records.append(corridor_record)
        paths.append(corridor_path)
        print(
            f"corridor graph length={corridor_record['path_length_m']:.1f} "
            f"strict={corridor_record['strict_building_district_route_ratio']:.0%} "
            f"bilateral={corridor_record['bilateral_building_ratio']:.0%} "
            f"turns={corridor_record['turn_count']} qualified={corridor_record['qualified']}",
            flush=True,
        )

    ranked = sorted(range(len(records)), key=lambda i: records[i]["rank_score"], reverse=True)
    qualified = [i for i in ranked if records[i]["qualified"]]
    selected_index = qualified[0] if qualified else (ranked[0] if ranked else None)
    if selected_index is None:
        report = {"status": "NOT_READY", "reason": "no route planned", "failures": failures}
        (args.output_dir / "route_qualification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(2)

    selected = records[selected_index]
    trajectory = paths[selected_index]
    report = {
        "schema": "urbanfly-building-canyon-route-qualification-v1",
        "status": "PASS" if selected["qualified"] else "NOT_READY",
        "selection_gate": {
            "minimum_path_length_m": 300.0,
            "minimum_straight_planar_distance_m": 150.0,
            "maximum_path_to_displacement_ratio": 2.7,
            "maximum_source_graph_repeated_cell_count": 0,
            "maximum_source_graph_repeated_edge_count": 0,
            "maximum_trajectory_reentered_grid_cell_count": 0,
            "maximum_trajectory_repeated_edge_count": 0,
            "maximum_altitude_m": 30.0,
            "minimum_triangle_distance_m": 7.5,
            "minimum_graph_triangle_center_clearance_m": 8.0,
            "minimum_heightmap_clearance_m": 6.0,
            "minimum_strict_building_district_route_ratio": 0.60,
            "minimum_mean_large_building_coverage": 0.25,
            "minimum_mean_obstacle_coverage": 0.12,
            "minimum_bilateral_building_ratio": 0.30,
            "minimum_encountered_large_building_count": 2,
            "minimum_kinematic_turn_count": 10,
            "kinematic_turn_minimum_incoming_and_outgoing_leg_m": 18.0,
            "minimum_total_turn_angle_degrees": 60.0,
        },
        "selected": selected,
        "candidate_count_requested": args.candidates,
        "candidate_count_planned": len(records),
        "qualified_candidate_count": len(qualified),
        "failures": failures,
        "density_summary": density.summary(),
        "building_proxy": {
            "minimum_connected_high_surface_area_m2": 500.0,
            "local_coverage_window_m": window * float(stack.grid.resolution),
            "strict_coverage_threshold": strict_threshold,
            "strict_mask_cell_ratio": float(np.mean(strict_mask)),
        },
    }
    (args.output_dir / "route_qualification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "selected_route.npz", planned_trajectory=trajectory)
    _plot(stack, density, strict_mask, trajectory, selected, args.output_dir / "building_canyon_route_overview.png")
    print(json.dumps({"status": report["status"], "selected": selected}, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
