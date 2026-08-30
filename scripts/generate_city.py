"""
程序化城市生成器
===============
1. 从 Paris BMP 贴图生成纹理图集
2. 生成城市布局 JSON
3. 生成建筑包围盒 + 占据网格（给后端路径规划用）

前端直接用 Three.js BoxGeometry + 贴图渲染，不做复杂 UV。
"""

import numpy as np
import os
import sys
import json
import random
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CITY_CONFIG = {
    "blocks_x": 16,
    "blocks_z": 16,
    "block_size": 44,
    "street_width": 10,
    "height_min": 6,
    "height_max": 85,
    "height_center_bias": 1.8,
    "vacancy_rate": 0.06,
    "park_rate": 0.04,
    "plaza_center": True,
    "plaza_radius": 2,
}

# 建筑类型 → 颜色基调
BUILDING_STYLES = {
    "skyscraper": {"color": [180,190,200], "window_ratio": 0.6},
    "highrise":   {"color": [195,185,170], "window_ratio": 0.4},
    "midrise":    {"color": [210,200,185], "window_ratio": 0.35},
    "lowrise":    {"color": [220,210,195], "window_ratio": 0.3},
}


def create_texture_atlas(paris_dir, output_dir, atlas_size=2048, tile_size=256):
    """从 Paris BMP 创建纹理图集"""
    print("[Atlas] Creating texture atlas from Paris BMPs...")
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 diffuse 贴图
    all_files = []
    for f in os.listdir(paris_dir):
        if f.endswith('_D.bmp'):
            all_files.append(os.path.join(paris_dir, f))

    if not all_files:
        print("[Atlas] No BMP textures found, generating procedural")
        return _make_procedural_atlas(output_dir, atlas_size, tile_size)

    # 按色彩饱和度排序，挑最好的
    scored = []
    for fp in all_files:
        try:
            img = Image.open(fp).convert('RGB')
            arr = np.array(img.resize((64, 64)))
            richness = float(np.std(arr, axis=(0,1)).mean())
            scored.append((richness, fp))
        except Exception:
            continue

    scored.sort(key=lambda x: -x[0])
    tiles_per_row = atlas_size // tile_size
    max_tiles = tiles_per_row * tiles_per_row
    num_tiles = min(len(scored), max_tiles)

    print(f"[Atlas] {len(scored)} textures → using {num_tiles} best")

    atlas = Image.new('RGB', (atlas_size, atlas_size), (190, 185, 175))

    for i, (_, fp) in enumerate(scored[:num_tiles]):
        try:
            img = Image.open(fp).convert('RGB')
            img = img.resize((tile_size, tile_size), Image.LANCZOS)
            row = i // tiles_per_row
            col = i % tiles_per_row
            atlas.paste(img, (col * tile_size, row * tile_size))
        except Exception:
            continue

    # 为屋顶添加专用行
    roof_start_row = (num_tiles // tiles_per_row) + 1
    for ci in range(min(8, tiles_per_row)):
        roof_img = _make_roof_tile(tile_size, seed=ci)
        atlas.paste(roof_img, (ci * tile_size, roof_start_row * tile_size))

    path = os.path.join(output_dir, "city_atlas.jpg")
    atlas.save(path, quality=88)
    print(f"[Atlas] Saved: {path} ({atlas_size}x{atlas_size})")
    return path, num_tiles + min(8, tiles_per_row), tile_size, atlas_size


def _make_roof_tile(size, seed):
    """生成屋顶纹理 tile"""
    rng = random.Random(seed + 999)
    img = Image.new('RGB', (size, size), (
        130 + rng.randint(-20, 20),
        105 + rng.randint(-15, 15),
        85 + rng.randint(-15, 15),
    ))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for _ in range(80):
        x, y = rng.randint(0, size), rng.randint(0, size)
        s = rng.randint(2, 6)
        c = (rng.randint(100, 140), rng.randint(75, 110), rng.randint(55, 90))
        draw.rectangle([x, y, x+s, y+s], fill=c)
    return img


def _make_procedural_atlas(output_dir, atlas_size, tile_size):
    """生成纯程序化纹理图集"""
    tiles_per_row = atlas_size // tile_size
    atlas = Image.new('RGB', (atlas_size, atlas_size), (200, 195, 185))
    for i in range(tiles_per_row * tiles_per_row):
        row = i // tiles_per_row
        col = i % tiles_per_row
        img = _make_facade_tile(tile_size, seed=i)
        atlas.paste(img, (col * tile_size, row * tile_size))
    path = os.path.join(output_dir, "city_atlas.jpg")
    atlas.save(path, quality=85)
    return path, tiles_per_row * tiles_per_row, tile_size, atlas_size


def _make_facade_tile(size, seed):
    """生成程序化建筑立面 tile"""
    rng = random.Random(seed)
    base = [180 + rng.randint(-25, 25), 170 + rng.randint(-20, 20), 155 + rng.randint(-20, 20)]
    img = Image.new('RGB', (size, size), tuple(base))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    story_h = size // 6
    for y in range(story_h, size, story_h):
        draw.line([(0, y), (size, y)], fill=(130, 120, 110), width=2)
    win_w = size // 5
    win_h = int(story_h * 0.7)
    for row in range(6):
        for col in range(5):
            wx = col * (size // 5) + 4
            wy = row * story_h + story_h // 6
            wc = (30 + rng.randint(-10, 10), 50 + rng.randint(-15, 15), 60 + rng.randint(-15, 15))
            draw.rectangle([wx, wy, wx + win_w - 6, wy + win_h], fill=wc)
    return img


def generate_city():
    """生成城市布局"""
    cfg = CITY_CONFIG
    bx, bz  = cfg["blocks_x"], cfg["blocks_z"]
    block_s = cfg["block_size"]
    street  = cfg["street_width"]
    total_x = bx * (block_s + street) + street
    total_z = bz * (block_s + street) + street
    rng = random.Random(42)
    buildings = []

    for ix in range(bx):
        for iz in range(bz):
            cx = -total_x/2 + street + ix*(block_s+street) + block_s/2
            cz = -total_z/2 + street + iz*(block_s+street) + block_s/2
            dist = np.sqrt((ix - bx/2)**2 + (iz - bz/2)**2)
            max_d = np.sqrt((bx/2)**2 + (bz/2)**2)
            center_f = max(0, 1 - dist/(max_d+1))

            if cfg["plaza_center"] and dist < cfg["plaza_radius"]:
                continue
            if rng.random() < cfg["vacancy_rate"]:
                continue

            n_bld = rng.randint(1, 4) if center_f > 0.6 else rng.randint(1, 3)
            plots = _carve_block(block_s, n_bld, rng)

            for (px, pz, pw, pd) in plots:
                w = max(5, min(pw - 1.5, 36))
                d = max(5, min(pd - 1.5, 36))
                h = cfg["height_min"] + (cfg["height_max"]-cfg["height_min"]) * center_f**cfg["height_center_bias"]
                h += rng.uniform(-10, 15)
                h = max(cfg["height_min"], min(cfg["height_max"], h))
                if h > 55:   style = "skyscraper"
                elif h > 28: style = "highrise"
                elif h > 12: style = "midrise"
                else:        style = "lowrise"

                buildings.append({
                    "x": cx - block_s/2 + px + pw/2,
                    "z": cz - block_s/2 + pz + pd/2,
                    "w": w, "d": d, "h": h,
                    "style": style,
                })

    styles_count = {}
    for b in buildings:
        styles_count[b["style"]] = styles_count.get(b["style"], 0) + 1

    print(f"[Layout] {len(buildings)} buildings in {total_x:.0f}m × {total_z:.0f}m")
    print(f"[Layout] Heights: {min(b['h'] for b in buildings):.0f}m — {max(b['h'] for b in buildings):.0f}m")
    print(f"[Layout] Styles: {styles_count}")
    return buildings, total_x, total_z


def _carve_block(block_s, n, rng):
    """将街区细分为建筑地块"""
    if n <= 1:
        return [(1, 1, block_s-2, block_s-2)]
    plots = []
    if rng.random() < 0.5:
        splits = sorted([rng.uniform(0.2, 0.8) for _ in range(n-1)])
        prev = 0
        for p in splits + [1.0]:
            sz = max(5, (p - prev) * (block_s-2))
            plots.append((1 + prev*(block_s-2), 1, sz, block_s-2))
            prev = p
    else:
        splits = sorted([rng.uniform(0.2, 0.8) for _ in range(n-1)])
        prev = 0
        for p in splits + [1.0]:
            sz = max(5, (p - prev) * (block_s-2))
            plots.append((1, 1 + prev*(block_s-2), block_s-2, sz))
            prev = p
    return plots


def export_buildings_json(buildings, output_dir):
    """导出建筑元数据 → buildings.json"""
    data = []
    for i, b in enumerate(buildings):
        data.append({
            "id": i + 1,
            "original_group": f"bld_{i+1}",
            "bounds_min": [b["x"]-b["w"]/2, 0, b["z"]-b["d"]/2],
            "bounds_max": [b["x"]+b["w"]/2, b["h"], b["z"]+b["d"]/2],
            "height": b["h"],
            "num_faces_original": 12,
            "num_faces_simplified": 12,
            "style": b["style"],
        })
    path = os.path.join(output_dir, "buildings.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Export] {path}")
    return path


def export_city_json(buildings, total_x, total_z, output_dir):
    """导出城市布局 → city_layout.json (前端用)"""
    data = {
        "total_x": total_x,
        "total_z": total_z,
        "max_height": max(b["h"] for b in buildings),
        "buildings": buildings,
    }
    path = os.path.join(output_dir, "city_layout.json")
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"[Export] {path}")
    return path


def export_occupancy_grid(buildings, output_dir):
    """生成占据网格"""
    from preprocess.occupancy_grid import OccupancyGridBuilder
    from backend.engine.models import BuildingInfo
    import numpy as np

    binfo = [BuildingInfo(
        id=b["id"], original_group=b["original_group"],
        bounds_min=np.array(b["bounds_min"]),
        bounds_max=np.array(b["bounds_max"]),
        num_faces_original=b["num_faces_original"]
    ) for b in json.load(open(os.path.join(output_dir, "buildings.json")))]

    builder = OccupancyGridBuilder(grid_resolution=5.0, safety_margin=2.0)
    grid, heightmap, origin = builder.build(binfo)
    np.savez_compressed(os.path.join(output_dir, "occupancy_grid.npz"),
                        grid=grid, origin=origin, resolution=5.0)
    np.savez_compressed(os.path.join(output_dir, "heightmap.npz"),
                        heightmap=heightmap, origin=origin, resolution=5.0)
    print(f"[Grid] Saved: {grid.shape}")


def main():
    output_dir = "data/scene"
    paris_dir = "C:/Users/caste/Desktop/paris"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 纹理图集
    atlas_path, num_tiles, tile_size, atlas_size = create_texture_atlas(
        paris_dir, output_dir
    )

    # 2. 城市布局
    buildings, total_x, total_z = generate_city()

    # 3. 导出 JSON
    export_buildings_json(buildings, output_dir)
    export_city_json(buildings, total_x, total_z, output_dir)

    # 4. 占据网格
    export_occupancy_grid(buildings, output_dir)

    # 5. 场景配置
    cfg = {
        "name": "Generated City",
        "bounds_center": [0, 0, 0],
        "bounds_size": [total_x + 20, max(b["h"] for b in buildings) + 10, total_z + 20],
        "num_buildings": len(buildings),
        "grid_resolution": 5.0,
    }
    with open(os.path.join(output_dir, "scene_config.json"), 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n✅ Done! Output: {output_dir}/")
    print(f"   city_atlas.jpg   — 巴黎贴图图集")
    print(f"   city_layout.json — 城市布局 (前端加载)")
    print(f"   buildings.json   — 建筑包围盒")
    print(f"   scene_config.json— 场景参数")
    print(f"   *.npz            — 占据网格+高度图")


if __name__ == "__main__":
    main()
