"""
生成高密度城市版本，作为中期报告 2.2 节默认实验底座。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.config import DENSE_CITY
from scripts import generate_city as base_city


def main():
    output_dir = "data/scene_dense"
    paris_dir = "C:/Users/caste/Desktop/paris"
    os.makedirs(output_dir, exist_ok=True)

    old_cfg = dict(base_city.CITY_CONFIG)
    base_city.CITY_CONFIG.update({
        "blocks_x": DENSE_CITY["blocks_x"],
        "blocks_z": DENSE_CITY["blocks_z"],
        "block_size": DENSE_CITY["block_size"],
        "street_width": DENSE_CITY["street_width"],
        "height_min": DENSE_CITY["height_min"],
        "height_max": DENSE_CITY["height_max"],
        "height_center_bias": DENSE_CITY["height_center_bias"],
        "vacancy_rate": DENSE_CITY["vacancy_rate"],
        "park_rate": DENSE_CITY["park_rate"],
        "plaza_center": DENSE_CITY["plaza_center"],
        "plaza_radius": DENSE_CITY["plaza_radius"],
    })

    try:
        if os.path.isdir(paris_dir):
            base_city.create_texture_atlas(paris_dir, output_dir)
        else:
            base_city._make_procedural_atlas(output_dir, 2048, 256)
        buildings, total_x, total_z = base_city.generate_city()
        base_city.export_buildings_json(buildings, output_dir)
        base_city.export_city_json(buildings, total_x, total_z, output_dir)
        base_city.export_occupancy_grid(buildings, output_dir)

        cfg = {
            "name": "Dense Generated City",
            "bounds_center": [0, 0, 0],
            "bounds_size": [total_x + 20, max(b["h"] for b in buildings) + 10, total_z + 20],
            "num_buildings": len(buildings),
            "grid_resolution": 5.0,
            "density_profile": "dense_midterm",
        }
        with open(os.path.join(output_dir, "scene_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        print(f"高密度城市已生成: {output_dir}")
        print(f"建筑数量: {len(buildings)}")
    finally:
        base_city.CITY_CONFIG.clear()
        base_city.CITY_CONFIG.update(old_cfg)


if __name__ == "__main__":
    main()
