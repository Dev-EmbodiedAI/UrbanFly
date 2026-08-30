"""
Build a benchmark city directly from the real actor layout exported from
the SimWorld `/Game/Maps/Empty` map.

Outputs:
    data/scene_simworld_dense/city_layout.json
    data/scene_simworld_dense/buildings.json
    data/scene_simworld_dense/occupancy_grid.npz
    data/scene_simworld_dense/heightmap.npz
    data/scene_simworld_dense/scene_summary.json
    data/scene_simworld_dense/simworld_city.obj
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from backend.engine.models import BuildingInfo
from preprocess.occupancy_grid import OccupancyGridBuilder


SOURCE_EXPORT = ROOT / "data" / "scene_simworld_dense" / "empty_map_actors.json"
OUTPUT_DIR = ROOT / "data" / "scene_simworld_dense"
CITY_LAYOUT_PATH = OUTPUT_DIR / "city_layout.json"
BUILDINGS_PATH = OUTPUT_DIR / "buildings.json"
SUMMARY_PATH = OUTPUT_DIR / "scene_summary.json"
OBJ_PATH = OUTPUT_DIR / "simworld_city.obj"

CM_TO_M = 0.01
GROUND_LABEL = "Arena_Env_Ground"
ROAD_PREFIXES = ("Road_", "Lines_", "StreetLight_", "Secondary_")
DISTRICT_KEYS = ("industrial", "mixed", "cbd", "park", "residential", "plaza")


def style_from_height(height_m: float) -> str:
    if height_m >= 120.0:
        return "landmark"
    if height_m >= 80.0:
        return "highrise"
    if height_m >= 40.0:
        return "midrise"
    return "lowrise"


def rect_from_bounds(x_min: float, x_max: float, z_min: float, z_max: float) -> list[list[float]]:
    return [
        [round(x_min, 4), round(z_min, 4)],
        [round(x_max, 4), round(z_min, 4)],
        [round(x_max, 4), round(z_max, 4)],
        [round(x_min, 4), round(z_max, 4)],
    ]


def centroid_from_group(items: list[dict]) -> np.ndarray:
    return np.array(
        [
            sum(item["x"] for item in items) / len(items),
            sum(item["z"] for item in items) / len(items),
        ],
        dtype=float,
    )


def bounds_from_group(items: list[dict], margin_x: float = 12.0, margin_z: float = 12.0) -> tuple[float, float, float, float]:
    x_min = min(item["x"] - item["w"] * 0.5 for item in items) - margin_x
    x_max = max(item["x"] + item["w"] * 0.5 for item in items) + margin_x
    z_min = min(item["z"] - item["d"] * 0.5 for item in items) - margin_z
    z_max = max(item["z"] + item["d"] * 0.5 for item in items) + margin_z
    return x_min, x_max, z_min, z_max


def district_seed_from_label(label: str) -> str | None:
    upper = label.upper()
    if upper.startswith(("CBD_", "GLASS_CBD_", "ROOF_CBD_", "PODIUM_CBD_", "FRONT_CBD_")):
        return "cbd"
    if upper.startswith(("MIX_", "ROOF_MIX_", "PODIUM_MIX_", "FRONT_MIX_")):
        return "mixed"
    if upper.startswith(("RES_", "ROOF_RES_", "FRONT_RES_")):
        return "residential"
    if upper.startswith(("CIV_", "ROOF_CIV_", "FRONT_CIV_", "PLAZA_")):
        return "plaza"
    if upper.startswith("FRONT_N_RING"):
        return "industrial"
    if upper.startswith("FRONT_S_RING"):
        return "park"
    return None


def load_actor_export() -> dict:
    if not SOURCE_EXPORT.exists():
        raise FileNotFoundError(
            f"Missing exported actor file: {SOURCE_EXPORT}. "
            f"Run scripts/export_empty_map_actors.py first."
        )
    return json.loads(SOURCE_EXPORT.read_text(encoding="utf-8"))


def road_footprint(actor: dict) -> tuple[float, float]:
    bounds = actor.get("bounds") or {}
    size = bounds.get("size") or [0.0, 0.0, 0.0]
    size_x = float(size[0]) * CM_TO_M
    size_y = float(size[1]) * CM_TO_M
    return max(size_x, size_y), max(min(size_x, size_y), 1.0)


def parse_roads(actors: list[dict]) -> list[dict]:
    roads: list[dict] = []
    for actor in actors:
        label = actor["label"]
        if label == GROUND_LABEL or not any(label.startswith(prefix) for prefix in ROAD_PREFIXES):
            continue
        length_m, width_m = road_footprint(actor)
        roads.append(
            {
                "label": label,
                "category": (
                    "main_road" if label.startswith("Road_") else
                    "lane_marking" if label.startswith("Lines_") else
                    "secondary_road" if label.startswith("Secondary_") else
                    "street_fixture"
                ),
                "x": round(float(actor["location"][0]) * CM_TO_M, 4),
                "z": round(float(actor["location"][1]) * CM_TO_M, 4),
                "length": round(length_m, 4),
                "width": round(width_m, 4),
                "yaw": round(float(actor["rotation"][1]), 4),
            }
        )
    return roads


def is_building_actor(actor: dict) -> bool:
    if actor["class"] != "/Script/Engine.StaticMeshActor":
        return False
    label = actor["label"]
    if label == GROUND_LABEL or any(label.startswith(prefix) for prefix in ROAD_PREFIXES):
        return False
    bounds = actor.get("bounds") or {}
    size = bounds.get("size") or [0.0, 0.0, 0.0]
    if float(size[2]) < 600.0:
        return False
    if max(float(size[0]), float(size[1])) < 300.0:
        return False
    return True


def actor_to_building(actor: dict, building_id: int) -> dict:
    bounds = actor["bounds"]
    size = bounds["size"]
    x = float(actor["location"][0]) * CM_TO_M
    z = float(actor["location"][1]) * CM_TO_M
    w = float(size[0]) * CM_TO_M
    d = float(size[1]) * CM_TO_M
    h = float(size[2]) * CM_TO_M
    center_y = float(actor["location"][2]) * CM_TO_M
    bottom_y = max(0.0, center_y - h * 0.5)
    top_y = bottom_y + h

    generated_facade = any(
        "M_HighriseGlass_Proc" in material
        for comp in actor.get("mesh_components", [])
        for material in (comp.get("materials") or [])
    )

    return {
        "id": building_id,
        "label": actor["label"],
        "x": round(x, 4),
        "z": round(z, 4),
        "w": round(w, 4),
        "d": round(d, 4),
        "h": round(h, 4),
        "bottom_y": round(bottom_y, 4),
        "top_y": round(top_y, 4),
        "yaw": round(float(actor["rotation"][1]), 4),
        "style": style_from_height(h),
        "district": None,
        "uses_generated_facade": generated_facade,
        "source_class": actor["class"],
        "mesh_path": next(
            (comp.get("mesh_path", "") for comp in actor.get("mesh_components", []) if comp.get("mesh_path")),
            "",
        ),
    }


def building_extents(buildings: list[dict]) -> tuple[float, float, float, float]:
    x_min = min(item["x"] - item["w"] * 0.5 for item in buildings)
    x_max = max(item["x"] + item["w"] * 0.5 for item in buildings)
    z_min = min(item["z"] - item["d"] * 0.5 for item in buildings)
    z_max = max(item["z"] + item["d"] * 0.5 for item in buildings)
    return x_min, x_max, z_min, z_max


def derive_district_geometry(buildings: list[dict]) -> tuple[dict[str, list[list[float]]], dict[str, list[float]]]:
    x_min, x_max, z_min, z_max = building_extents(buildings)
    seed_groups: dict[str, list[dict]] = {key: [] for key in DISTRICT_KEYS}
    remainder: list[dict] = []

    for item in buildings:
        seed = district_seed_from_label(item["label"])
        if seed is None:
            remainder.append(item)
            continue
        item["district"] = seed
        seed_groups[seed].append(item)

    centroids = {
        key: centroid_from_group(group)
        for key, group in seed_groups.items()
        if group
    }
    north_seed = seed_groups["industrial"]
    south_seed = seed_groups["park"]
    north_gate = min(item["z"] - item["d"] * 0.5 for item in north_seed) - 6.0 if north_seed else z_max - 30.0
    south_gate = max(item["z"] + item["d"] * 0.5 for item in south_seed) + 6.0 if south_seed else z_min + 30.0

    for item in remainder:
        if item["z"] >= north_gate:
            item["district"] = "industrial"
            continue
        if item["z"] <= south_gate:
            item["district"] = "park"
            continue

        candidate_keys = ("mixed", "cbd", "industrial") if item["z"] >= 0.0 else ("residential", "plaza", "park")
        best_key = None
        best_dist = float("inf")
        for key in candidate_keys:
            center = centroids.get(key)
            if center is None:
                continue
            dist = math.hypot(item["x"] - center[0], item["z"] - center[1])
            if dist < best_dist:
                best_dist = dist
                best_key = key
        item["district"] = best_key or ("mixed" if item["x"] < 0.0 else "cbd")

    district_groups: dict[str, list[dict]] = {key: [] for key in DISTRICT_KEYS}
    for item in buildings:
        district_groups[item["district"]].append(item)

    hotspots: dict[str, list[float]] = {}
    polygons: dict[str, list[list[float]]] = {}
    for key in DISTRICT_KEYS:
        core_group = seed_groups[key] or district_groups[key]
        center = centroid_from_group(core_group)
        hotspots[key] = [round(float(center[0]), 4), 10.0, round(float(center[1]), 4)]

        bx0, bx1, bz0, bz1 = bounds_from_group(core_group, margin_x=14.0, margin_z=14.0)
        if key in {"industrial", "park"}:
            bx0 = x_min - 14.0
            bx1 = x_max + 14.0
        polygons[key] = rect_from_bounds(bx0, bx1, bz0, bz1)

    return polygons, hotspots


def write_buildings_json(buildings: list[dict]) -> None:
    data = []
    for item in buildings:
        data.append(
            {
                "id": item["id"],
                "original_group": item["label"],
                "bounds_min": [
                    item["x"] - item["w"] * 0.5,
                    item["bottom_y"],
                    item["z"] - item["d"] * 0.5,
                ],
                "bounds_max": [
                    item["x"] + item["w"] * 0.5,
                    item["top_y"],
                    item["z"] + item["d"] * 0.5,
                ],
                "height": item["h"],
                "num_faces_original": 12,
                "num_faces_simplified": 12,
                "style": item["style"],
                "district": item["district"],
                "uses_generated_facade": item["uses_generated_facade"],
                "yaw": item["yaw"],
            }
        )
    BUILDINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_occupancy(buildings: list[dict]) -> None:
    building_info = [
        BuildingInfo(
            id=item["id"],
            original_group=item["label"],
            bounds_min=np.array(
                [item["x"] - item["w"] * 0.5, item["bottom_y"], item["z"] - item["d"] * 0.5],
                dtype=float,
            ),
            bounds_max=np.array(
                [item["x"] + item["w"] * 0.5, item["top_y"], item["z"] + item["d"] * 0.5],
                dtype=float,
            ),
            num_faces_original=12,
            num_faces_simplified=12,
        )
        for item in buildings
    ]
    builder = OccupancyGridBuilder(grid_resolution=5.0, safety_margin=2.0)
    grid, heightmap, origin = builder.build(building_info)
    np.savez_compressed(OUTPUT_DIR / "occupancy_grid.npz", grid=grid, origin=origin, resolution=5.0)
    np.savez_compressed(OUTPUT_DIR / "heightmap.npz", heightmap=heightmap, origin=origin, resolution=5.0)


def obj_box_vertices(x: float, z: float, w: float, d: float, y0: float, y1: float) -> list[tuple[float, float, float]]:
    x0, x1 = x - w * 0.5, x + w * 0.5
    z0, z1 = z - d * 0.5, z + d * 0.5
    return [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y0, z1),
        (x0, y0, z1),
        (x0, y1, z0),
        (x1, y1, z0),
        (x1, y1, z1),
        (x0, y1, z1),
    ]


def write_box_obj(lines: list[str], vertex_offset: int, name: str, x: float, z: float, w: float, d: float, y0: float, y1: float) -> int:
    verts = obj_box_vertices(x, z, w, d, y0, y1)
    lines.append(f"o {name}")
    for vx, vy, vz in verts:
        lines.append(f"v {vx:.4f} {vy:.4f} {vz:.4f}")
    faces = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 4, 8, 7),
        (4, 1, 5, 8),
    ]
    for a, b, c, d_face in faces:
        lines.append(f"f {vertex_offset + a} {vertex_offset + b} {vertex_offset + c} {vertex_offset + d_face}")
    return vertex_offset + 8


def write_city_obj(buildings: list[dict], roads: list[dict]) -> None:
    lines = ["# SimWorld Empty.umap derived city geometry"]
    vertex_offset = 0

    x_min, x_max, z_min, z_max = building_extents(buildings)
    ground_w = (x_max - x_min) + 40.0
    ground_d = (z_max - z_min) + 40.0
    ground_x = (x_min + x_max) * 0.5
    ground_z = (z_min + z_max) * 0.5
    vertex_offset = write_box_obj(lines, vertex_offset, "ground", ground_x, ground_z, ground_w, ground_d, -0.2, 0.0)

    for road in roads:
        if road["category"] not in {"main_road", "secondary_road"}:
            continue
        road_w = road["length"] if abs(road["yaw"]) < 45.0 or abs(road["yaw"]) > 135.0 else road["width"]
        road_d = road["width"] if abs(road["yaw"]) < 45.0 or abs(road["yaw"]) > 135.0 else road["length"]
        vertex_offset = write_box_obj(
            lines,
            vertex_offset,
            road["label"],
            road["x"],
            road["z"],
            road_w,
            road_d,
            -0.05,
            0.05,
        )

    for item in buildings:
        vertex_offset = write_box_obj(
            lines,
            vertex_offset,
            item["label"],
            item["x"],
            item["z"],
            item["w"],
            item["d"],
            item["bottom_y"],
            item["top_y"],
        )

    OBJ_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_city_layout(
    buildings: list[dict],
    roads: list[dict],
    polygons: dict[str, list[list[float]]],
    hotspots: dict[str, list[float]],
) -> dict:
    x_min, x_max, z_min, z_max = building_extents(buildings)
    return {
        "scene_name": "dense_city_scene",
        "source_path": str(CITY_LAYOUT_PATH),
        "source_map": "/Game/Maps/Empty",
        "source_asset": "/Game/GeneratedFacade/M_HighriseGlass_Proc",
        "geometry_mode": "empty_umap_actor_export_to_obj",
        "source_notes": [
            "Geometry is reconstructed from the current Empty.umap actor layout.",
            "Building footprints and heights come from exported StaticMeshActor bounds.",
            "Material usage of M_HighriseGlass_Proc is preserved in building metadata.",
            "District centers are derived from real block labels rather than synthetic ellipses.",
        ],
        "total_x": round((x_max - x_min) + 28.0, 4),
        "total_z": round((z_max - z_min) + 28.0, 4),
        "max_height": round(max(item["top_y"] for item in buildings), 4),
        "district_hotspots": hotspots,
        "district_polygons": polygons,
        "roads": roads,
        "buildings": buildings,
        "obj_path": str(OBJ_PATH),
    }


def write_summary(buildings: list[dict], roads: list[dict], city_layout: dict) -> None:
    district_counts: dict[str, int] = {key: 0 for key in DISTRICT_KEYS}
    for item in buildings:
        district_counts[item["district"]] += 1

    summary = {
        "scene_name": city_layout["scene_name"],
        "source_map": city_layout["source_map"],
        "source_asset": city_layout["source_asset"],
        "building_count": len(buildings),
        "generated_facade_count": sum(1 for item in buildings if item["uses_generated_facade"]),
        "road_count": len(roads),
        "district_counts": district_counts,
        "height_stats": {
            "min_m": round(min(item["h"] for item in buildings), 4),
            "mean_m": round(sum(item["h"] for item in buildings) / len(buildings), 4),
            "max_m": round(max(item["h"] for item in buildings), 4),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export = load_actor_export()
    actors = export["actors"]

    roads = parse_roads(actors)
    buildings = [
        actor_to_building(actor, idx)
        for idx, actor in enumerate((item for item in actors if is_building_actor(item)), start=1)
    ]
    polygons, hotspots = derive_district_geometry(buildings)

    write_buildings_json(buildings)
    write_occupancy(buildings)
    write_city_obj(buildings, roads)
    city_layout = build_city_layout(buildings, roads, polygons, hotspots)
    CITY_LAYOUT_PATH.write_text(json.dumps(city_layout, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(buildings, roads, city_layout)

    print(f"Built city from Empty.umap actor export: {CITY_LAYOUT_PATH}")
    print(f"Building count: {len(buildings)}")
    print(f"Road count: {len(roads)}")
    print(f"OBJ: {OBJ_PATH}")


if __name__ == "__main__":
    main()
