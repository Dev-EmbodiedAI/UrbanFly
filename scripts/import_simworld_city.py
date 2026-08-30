"""
从 SimWorld-Studio 场景图导入 UrbanFly 城市底座。

输入：
    D:\\AI\\SimWorld-Studio\\test_map_scene_graph.json

输出：
    data/scene_simworld/
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_city import export_buildings_json, export_city_json, export_occupancy_grid


SIMWORLD_SCENE = Path(r"D:\AI\SimWorld-Studio\test_map_scene_graph.json")
OUTPUT_DIR = Path("data/scene_simworld")


def _height_from_actor(actor: dict, width_m: float, depth_m: float) -> float:
    seed = hashlib.md5(actor["name"].encode("utf-8")).hexdigest()
    bucket = int(seed[:8], 16) % 7
    base = 18.0 + bucket * 8.0
    footprint_term = 0.55 * (width_m + depth_m)
    return max(18.0, min(135.0, base + footprint_term))


def load_simworld_buildings(scene_path: Path):
    with open(scene_path, "r", encoding="utf-8") as f:
        actors = json.load(f)

    buildings = []
    for idx, actor in enumerate(actors):
        name = actor.get("name", "")
        cls = actor.get("class", "")
        if "Building" not in name and "Building" not in cls:
            continue

        center = actor.get("center", {})
        size = actor.get("size", {})
        cx = float(center.get("x", 0.0)) * 0.01
        cz = float(center.get("y", 0.0)) * 0.01
        width_m = max(8.0, float(size.get("width", 0.0)) * 0.01)
        depth_m = max(8.0, float(size.get("height", 0.0)) * 0.01)
        height_m = _height_from_actor(actor, width_m, depth_m)

        if width_m < 6.0 or depth_m < 6.0:
            continue

        style = "midrise"
        if height_m > 75:
            style = "skyscraper"
        elif height_m > 42:
            style = "highrise"
        elif height_m > 18:
            style = "midrise"
        else:
            style = "lowrise"

        buildings.append(
            {
                "id": len(buildings) + 1,
                "original_group": name or f"simworld_{idx}",
                "x": cx,
                "z": cz,
                "w": width_m,
                "d": depth_m,
                "h": height_m,
                "style": style,
            }
        )

    return buildings


def main():
    if not SIMWORLD_SCENE.exists():
        raise FileNotFoundError(f"未找到 SimWorld 场景图: {SIMWORLD_SCENE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings = load_simworld_buildings(SIMWORLD_SCENE)
    if len(buildings) < 12:
        raise RuntimeError(f"可用建筑数量过少: {len(buildings)}")

    xs = [b["x"] for b in buildings]
    zs = [b["z"] for b in buildings]
    total_x = max(xs) - min(xs) + 80.0
    total_z = max(zs) - min(zs) + 80.0

    export_buildings_json(buildings, str(OUTPUT_DIR))
    export_city_json(buildings, total_x, total_z, str(OUTPUT_DIR))
    export_occupancy_grid(buildings, str(OUTPUT_DIR))

    cfg = {
        "name": "SimWorld Imported City",
        "source": str(SIMWORLD_SCENE),
        "bounds_center": [0.0, 0.0, 0.0],
        "bounds_size": [total_x, max(b["h"] for b in buildings) + 12.0, total_z],
        "num_buildings": len(buildings),
        "grid_resolution": 5.0,
    }
    with open(OUTPUT_DIR / "scene_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"导入完成: {OUTPUT_DIR}")
    print(f"建筑数量: {len(buildings)}")


if __name__ == "__main__":
    main()
