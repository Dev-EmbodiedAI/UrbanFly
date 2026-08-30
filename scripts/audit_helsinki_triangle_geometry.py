#!/usr/bin/env python3
"""Runtime audit of the real Helsinki triangle collision surface."""

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
from backend.engine.triangle_geometry import TriangleMeshLocalCollision  # noqa: E402


SCENE = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"


def _vertical_cavities(query, heightmap, spacing_m=10.0):
    xs = np.arange(-450.0, 450.0 + 1e-6, spacing_m)
    zs = np.arange(-450.0, 450.0 + 1e-6, spacing_m)
    origins = np.asarray([[x, -5.0, z] for x in xs for z in zs], dtype=float)
    directions = np.tile([0.0, 1.0, 0.0], (len(origins), 1))
    locations, rays, _ = query.mesh.ray.intersects_location(
        origins,
        directions,
        multiple_hits=True,
    )
    grouped = {}
    for location, ray in zip(locations, rays):
        grouped.setdefault(int(ray), []).append(float(location[1]))
    raw_candidates = []
    for ray, values in grouped.items():
        levels = sorted(set(round(value, 3) for value in values))
        for lower, upper in zip(levels[:-1], levels[1:]):
            gap = upper - lower
            if gap < 2.5:
                continue
            x, _, z = origins[ray]
            midpoint = np.array([x, (lower + upper) * 0.5, z], dtype=float)
            raw_candidates.append((midpoint, lower, upper, gap))
    if not raw_candidates:
        return []
    bridge_raw = sorted(
        (item for item in raw_candidates if item[2] <= 12.0),
        key=lambda item: item[3],
        reverse=True,
    )[:200]
    overhang_raw = sorted(
        (item for item in raw_candidates if item[2] > 12.0),
        key=lambda item: item[3],
        reverse=True,
    )[:300]
    raw_candidates = bridge_raw + overhang_raw
    midpoints = np.asarray([item[0] for item in raw_candidates], dtype=float)
    _, distances, _ = query._closest(midpoints)
    probe_offsets = np.asarray(
        [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, -6.0], [0.0, 0.0, 6.0]],
        dtype=float,
    )
    probes = (midpoints[:, None, :] + probe_offsets[None, :, :]).reshape(-1, 3)
    _, probe_distances, _ = query._closest(probes)
    probe_distances = probe_distances.reshape(len(midpoints), 4)
    candidates = []
    for candidate_index, ((midpoint, lower, upper, gap), distance) in enumerate(
        zip(raw_candidates, distances)
    ):
            if float(distance) <= 0.75:
                continue
            x, _, z = midpoint
            horizontal = {
                "east_west": bool(np.all(probe_distances[candidate_index, :2] > 0.75)),
                "north_south": bool(np.all(probe_distances[candidate_index, 2:] > 0.75)),
            }
            heightmap_surface = heightmap.surface_height(midpoint, 0.5)
            candidates.append(
                {
                    "position": midpoint.tolist(),
                    "lower_surface_m": lower,
                    "upper_surface_m": upper,
                    "gap_m": gap,
                    "triangle_distance_m": float(distance),
                    "heightmap_surface_m": heightmap_surface,
                    "heightmap_treats_midpoint_as_occupied": bool(
                        midpoint[1] < heightmap_surface + 0.5
                    ),
                    "horizontal_free_axes": horizontal,
                }
            )
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "helsinki_triangle_geometry_audit.json",
    )
    args = parser.parse_args()
    stack = HelsinkiNavigationStack.load(args.scene)
    manifest = json.loads((args.scene / "manifest.json").read_text(encoding="utf-8"))
    query = TriangleMeshLocalCollision.load(args.scene / manifest["collision"]["uri"])
    rng = np.random.default_rng(20260831)

    height = stack.collision_map.height
    rows, columns = np.where(np.isfinite(height) & (height >= 12.0))
    selected = rng.choice(len(rows), size=min(300, len(rows)), replace=False)
    surface_points = []
    for item in selected:
        row, column = int(rows[item]), int(columns[item])
        x = stack.collision_map.origin_x + (column + 0.5) * stack.collision_map.resolution
        z = stack.collision_map.maximum_z - (row + 0.5) * stack.collision_map.resolution
        surface_points.append([x, float(height[row, column]), z])
    _, building_distances, _ = query._closest(np.asarray(surface_points, dtype=float))

    cavities = _vertical_cavities(query, stack.collision_map)
    bridge_candidates = [
        item
        for item in cavities
        if item["lower_surface_m"] <= 2.5
        and item["upper_surface_m"] <= 12.0
        and any(item["horizontal_free_axes"].values())
    ]
    overhang_candidates = [
        item
        for item in cavities
        if item["upper_surface_m"] > 12.0
        and any(item["horizontal_free_axes"].values())
    ]

    areas = np.asarray(query.mesh.area_faces)
    centroids = np.asarray(query.mesh.triangles_center)
    thin_indices = np.flatnonzero((areas < 0.05) & (centroids[:, 1] > 2.0))
    if len(thin_indices) > 20000:
        thin_indices = rng.choice(thin_indices, size=20000, replace=False)
    thin_misses = []
    for index in thin_indices:
        point = centroids[int(index)]
        surface = stack.collision_map.surface_height(point, 0.0)
        if point[1] - surface > 0.75:
            thin_misses.append(
                {
                    "triangle_index": int(index),
                    "centroid": point.tolist(),
                    "triangle_area_m2": float(areas[int(index)]),
                    "heightmap_surface_m": float(surface),
                    "heightmap_vertical_gap_m": float(point[1] - surface),
                    "triangle_detected": query.is_collision(point, 0.05),
                }
            )
            if len(thin_misses) >= 20:
                break

    report = {
        "triangle_mesh_source": str(query.source_path.resolve()),
        "triangles": query.triangle_count,
        "vertices": query.vertex_count,
        "watertight": bool(query.mesh.is_watertight),
        "acceleration_structure": query.acceleration_structure,
        "index_build_time_s": query.build_time_s,
        "building": {
            "samples": len(surface_points),
            "median_heightmap_surface_to_triangle_distance_m": float(np.median(building_distances)),
            "p95_heightmap_surface_to_triangle_distance_m": float(np.percentile(building_distances, 95)),
            "within_0p75m_ratio": float(np.mean(building_distances <= 0.75)),
            "within_2m_ratio": float(np.mean(building_distances <= 2.0)),
            "verdict": "PASS" if float(np.mean(building_distances <= 2.0)) >= 0.95 else "LIMITATION",
            "note": "0.5 m heightmap is conservative raster coverage, so cell maxima need not lie exactly on a triangle",
        },
        "bridge": {
            "candidate_count": len(bridge_candidates),
            "examples": bridge_candidates[:10],
            "verdict": "PASS" if bridge_candidates else "LIMITATION",
            "note": "geometric cavity classification; no semantic bridge labels exist in the source asset",
        },
        "overhang": {
            "candidate_count": len(overhang_candidates),
            "examples": overhang_candidates[:10],
            "verdict": "PASS" if overhang_candidates else "LIMITATION",
            "note": "geometric under-surface cavity, not a semantic asset label",
        },
        "thin_structure": {
            "heightmap_miss_candidates": len(thin_misses),
            "examples": thin_misses[:10],
            "verdict": "PASS" if thin_misses else "LIMITATION",
        },
        "tree": {
            "verdict": "LIMITATION",
            "note": "TREE COLLISION COMPLETENESS NOT GUARANTEED: photogrammetry mesh has no semantic vegetation inventory",
        },
        "inside_outside_limitation": (
            "mesh is non-watertight; distance is unsigned surface distance. "
            "Swept validation is reliable for trajectories starting in known free space, "
            "but arbitrary deep-inside point classification is not guaranteed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
