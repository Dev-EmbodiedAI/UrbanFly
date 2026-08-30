#!/usr/bin/env python3
"""Build a browser-ready 500 m Helsinki photogrammetry mesh scene.

The Helsinki OBJ release is a tiled, multi-LOD production dataset. This script
selects a contiguous region, converts the chosen visual LOD into one GLB per
250 m source tile, and derives a lower-detail triangle collision mesh and
0.5 m height/ESDF diagnostics from the same source geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt


LOD_PATTERN = re.compile(r"_L(\d+)")


@dataclass(frozen=True)
class SourceTile:
    name: str
    directory: Path
    minimum: np.ndarray
    maximum: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-name", default="Helsinki500")
    parser.add_argument("--origin-x", type=float, default=8750.0)
    parser.add_argument("--origin-y", type=float, default=5250.0)
    parser.add_argument("--size", type=float, default=500.0)
    parser.add_argument("--visual-lod", type=int, default=21)
    parser.add_argument("--overview-lod", type=int, default=18)
    parser.add_argument("--collision-lod", type=int, default=18)
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument(
        "--esdf-altitudes",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 30.0, 40.0],
    )
    parser.add_argument("--uav-clearance", type=float, default=2.0)
    return parser.parse_args()


def obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            xyz = np.fromstring(line[2:], sep=" ", dtype=np.float64, count=3)
            if xyz.size != 3:
                continue
            minimum = np.minimum(minimum, xyz)
            maximum = np.maximum(maximum, xyz)
    return minimum, maximum


def discover_source_tiles(source: Path) -> list[SourceTile]:
    tiles: list[SourceTile] = []
    for directory in sorted(path for path in source.iterdir() if path.is_dir()):
        coarse = next(directory.glob("*_L13.obj"), None)
        if coarse is None or coarse.stat().st_size == 0:
            continue
        minimum, maximum = obj_bounds(coarse)
        tiles.append(SourceTile(directory.name, directory, minimum, maximum))
    if not tiles:
        raise RuntimeError(f"No Helsinki L13 source tiles found below {source}")
    return tiles


def select_region(
    tiles: Iterable[SourceTile],
    origin_x: float,
    origin_y: float,
    size: float,
) -> list[SourceTile]:
    epsilon = 0.25
    maximum_x = origin_x + size
    maximum_y = origin_y + size
    selected = [
        tile
        for tile in tiles
        if tile.minimum[0] >= origin_x - epsilon
        and tile.maximum[0] <= maximum_x + epsilon
        and tile.minimum[1] >= origin_y - epsilon
        and tile.maximum[1] <= maximum_y + epsilon
    ]
    covered_area = sum(
        (tile.maximum[0] - tile.minimum[0])
        * (tile.maximum[1] - tile.minimum[1])
        for tile in selected
    )
    expected_area = size * size
    # Photogrammetry tile borders may overlap their nominal 250 m cell by a
    # few centimetres so adjacent meshes do not reveal cracks.
    if not math.isclose(covered_area, expected_area, rel_tol=0.0, abs_tol=20.0):
        raise RuntimeError(
            f"Selected source tiles cover {covered_area:.1f} m², "
            f"expected {expected_area:.1f} m²"
        )
    return sorted(selected, key=lambda tile: (tile.minimum[1], tile.minimum[0]))


def files_for_lod(directory: Path, lod: int) -> list[Path]:
    files = []
    for path in directory.glob("*.obj"):
        match = LOD_PATTERN.search(path.stem)
        if (
            match is not None
            and int(match.group(1)) == lod
            and path.stat().st_size > 0
        ):
            files.append(path)
    return sorted(files)


def validate_visual_source(paths: Iterable[Path]) -> None:
    errors: list[str] = []
    for obj_path in paths:
        mtl_path = obj_path.with_suffix(".mtl")
        texture_path = obj_path.with_name(f"{obj_path.stem}_0.jpg")
        if not mtl_path.is_file():
            errors.append(f"missing material: {mtl_path}")
        if not texture_path.is_file():
            errors.append(f"missing texture: {texture_path}")
    if errors:
        raise RuntimeError("\n".join(errors[:20]))


def world_to_local_transform(center_x: float, center_y: float) -> np.ndarray:
    # Helsinki OBJ: X east, Y north, Z up.
    # UrbanFly/Three.js: X east, Y up, -Z north.
    return np.array(
        [
            [1.0, 0.0, 0.0, -center_x],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, center_y],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def add_scene_geometry(
    destination: trimesh.Scene,
    source_scene: trimesh.Scene,
    transform: np.ndarray,
    prefix: str,
) -> tuple[int, int]:
    vertex_count = 0
    triangle_count = 0
    for index, (name, geometry) in enumerate(source_scene.geometry.items()):
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        mesh_name = f"{prefix}_{index}_{name}"
        destination.add_geometry(
            mesh,
            node_name=mesh_name,
            geom_name=mesh_name,
        )
        vertex_count += len(mesh.vertices)
        triangle_count += len(mesh.faces)
    return vertex_count, triangle_count


def build_visual_tile(
    source_tile: SourceTile,
    lod: int,
    transform: np.ndarray,
    output_path: Path,
) -> dict:
    paths = files_for_lod(source_tile.directory, lod)
    if not paths:
        raise RuntimeError(f"No L{lod} OBJ files in {source_tile.directory}")
    validate_visual_source(paths)

    output_scene = trimesh.Scene()
    vertex_count = 0
    triangle_count = 0
    for index, obj_path in enumerate(paths, start=1):
        print(
            f"  [{source_tile.name}] visual {index:02d}/{len(paths):02d} "
            f"{obj_path.name}"
        )
        source_scene = trimesh.load(
            obj_path,
            force="scene",
            process=False,
        )
        vertices, triangles = add_scene_geometry(
            output_scene,
            source_scene,
            transform,
            f"{source_tile.name}_{obj_path.stem}",
        )
        vertex_count += vertices
        triangle_count += triangles

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        trimesh.exchange.gltf.export_glb(
            output_scene,
            include_normals=True,
        )
    )
    bounds = output_scene.bounds
    return {
        "name": source_tile.name,
        "uri": output_path.name,
        "source_lod": lod,
        "source_objects": len(paths),
        "vertices": vertex_count,
        "triangles": triangle_count,
        "bytes": output_path.stat().st_size,
        "bounds": {
            "minimum": bounds[0].tolist(),
            "maximum": bounds[1].tolist(),
        },
    }


def load_collision_mesh(
    selected: Iterable[SourceTile],
    lod: int,
    transform: np.ndarray,
) -> tuple[trimesh.Trimesh, int]:
    meshes: list[trimesh.Trimesh] = []
    source_objects = 0
    for source_tile in selected:
        paths = files_for_lod(source_tile.directory, lod)
        for index, obj_path in enumerate(paths, start=1):
            print(
                f"  [{source_tile.name}] collision {index:02d}/{len(paths):02d} "
                f"{obj_path.name}"
            )
            scene = trimesh.load(
                obj_path,
                force="scene",
                process=False,
                skip_materials=True,
            )
            for geometry in scene.geometry.values():
                mesh = geometry.copy()
                mesh.apply_transform(transform)
                meshes.append(mesh)
            source_objects += 1
    if not meshes:
        raise RuntimeError(f"No collision geometry found at L{lod}")
    collision = trimesh.util.concatenate(meshes)
    collision.remove_unreferenced_vertices()
    return collision, source_objects


def export_collision_glb(mesh: trimesh.Trimesh, output_path: Path) -> None:
    collision = mesh.copy()
    collision.visual = trimesh.visual.ColorVisuals(
        mesh=collision,
        face_colors=np.array([255, 166, 70, 255], dtype=np.uint8),
    )
    scene = trimesh.Scene(collision)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    )


def rasterize_heightmap(
    mesh: trimesh.Trimesh,
    size: float,
    resolution: float,
) -> np.ndarray:
    cells = int(round(size / resolution)) + 1
    heightmap = np.full((cells, cells), -np.inf, dtype=np.float32)
    vertices = np.asarray(mesh.vertices)
    triangles = vertices[np.asarray(mesh.faces)]

    # Local X/Z are in [-size/2, size/2]. Image rows run from north to south.
    pixel_x = (triangles[:, :, 0] + size / 2.0) / resolution
    pixel_y = (size / 2.0 - triangles[:, :, 2]) / resolution
    projected = np.stack((pixel_x, pixel_y), axis=-1)
    heights = triangles[:, :, 1].max(axis=1)

    # Painting low surfaces first makes later roofs/trees conservatively win.
    for index in np.argsort(heights):
        points = np.rint(projected[index]).astype(np.int32)
        if (
            points[:, 0].max() < 0
            or points[:, 1].max() < 0
            or points[:, 0].min() >= cells
            or points[:, 1].min() >= cells
        ):
            continue
        points[:, 0] = np.clip(points[:, 0], 0, cells - 1)
        points[:, 1] = np.clip(points[:, 1], 0, cells - 1)
        cv2.fillConvexPoly(
            heightmap,
            points,
            float(heights[index]),
            lineType=cv2.LINE_8,
        )
    heightmap[~np.isfinite(heightmap)] = np.nan
    return heightmap


def esdf_preview(distance: np.ndarray, occupied: np.ndarray) -> Image.Image:
    finite = np.clip(distance, 0.0, 60.0) / 60.0
    rgba = np.zeros((*distance.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.where(occupied, 255, 30)
    rgba[..., 1] = np.where(occupied, 88, 150 + finite * 90)
    rgba[..., 2] = np.where(occupied, 55, 255)
    rgba[..., 3] = np.where(occupied, 205, 42 + (1.0 - finite) * 105)
    return Image.fromarray(rgba, mode="RGBA")


def build_esdf_products(
    heightmap: np.ndarray,
    output_dir: Path,
    size: float,
    resolution: float,
    altitudes: Iterable[float],
    clearance: float,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "heightmap_0p5m.npz",
        height_m=heightmap,
        resolution_m=np.float32(resolution),
        origin_x_m=np.float32(-size / 2.0),
        origin_z_m=np.float32(-size / 2.0),
    )

    slices = []
    for altitude in altitudes:
        occupied = np.isfinite(heightmap) & (
            heightmap + clearance >= float(altitude)
        )
        free_distance = distance_transform_edt(~occupied) * resolution
        obstacle_distance = distance_transform_edt(occupied) * resolution
        signed_distance = free_distance.astype(np.float32)
        signed_distance[occupied] = -obstacle_distance[occupied]

        label = f"{int(round(altitude)):03d}m"
        npz_name = f"esdf_{label}_0p5m.npz"
        png_name = f"esdf_{label}_0p5m.png"
        np.savez_compressed(
            output_dir / npz_name,
            signed_distance_m=signed_distance,
            occupied=occupied,
            altitude_m=np.float32(altitude),
            resolution_m=np.float32(resolution),
        )
        esdf_preview(free_distance, occupied).save(output_dir / png_name)
        slices.append(
            {
                "altitude_m": float(altitude),
                "npz": f"diagnostics/{npz_name}",
                "image": f"diagnostics/{png_name}",
                "occupied_cells": int(occupied.sum()),
            }
        )
    return slices


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    source_tiles = discover_source_tiles(source)
    selected = select_region(
        source_tiles,
        args.origin_x,
        args.origin_y,
        args.size,
    )
    center_x = args.origin_x + args.size / 2.0
    center_y = args.origin_y + args.size / 2.0
    transform = world_to_local_transform(center_x, center_y)

    print("Selected source tiles:", ", ".join(tile.name for tile in selected))
    visual_dir = output / "visual"
    visual_tiles = []
    for source_tile in selected:
        print(f"Building visual tile {source_tile.name}")
        visual_tiles.append(
            build_visual_tile(
                source_tile,
                args.visual_lod,
                transform,
                visual_dir / f"{source_tile.name}_L{args.visual_lod}.glb",
            )
        )

    overview_tiles = []
    if args.overview_lod is not None:
        overview_dir = output / "overview"
        for source_tile in selected:
            print(f"Building overview tile {source_tile.name}")
            overview_tiles.append(
                build_visual_tile(
                    source_tile,
                    args.overview_lod,
                    transform,
                    overview_dir
                    / f"{source_tile.name}_L{args.overview_lod}.glb",
                )
            )

    print(f"Building L{args.collision_lod} collision mesh")
    collision_mesh, collision_source_objects = load_collision_mesh(
        selected,
        args.collision_lod,
        transform,
    )
    collision_path = output / "collision" / (
        f"{args.scene_name}_collision_L{args.collision_lod}.glb"
    )
    export_collision_glb(collision_mesh, collision_path)

    print(f"Rasterizing {args.grid_resolution:.2f} m heightmap and ESDF")
    heightmap = rasterize_heightmap(
        collision_mesh,
        args.size,
        args.grid_resolution,
    )
    slices = build_esdf_products(
        heightmap,
        output / "diagnostics",
        args.size,
        args.grid_resolution,
        args.esdf_altitudes,
        args.uav_clearance,
    )

    finite_heights = heightmap[np.isfinite(heightmap)]
    manifest = {
        "schema_version": 1,
        "scene_name": args.scene_name,
        "source": {
            "title": "Helsinki 3D+ photogrammetric mesh (2017)",
            "license": "CC BY 4.0",
            "crs": "EPSG:3879+5773",
            "srs_origin": [25490000.0, 6668000.0, 0.0],
            "dataset_path": str(source),
            "selected_source_tiles": [tile.name for tile in selected],
        },
        "operation_size_m": args.size,
        "original_bounds": {
            "minimum": [args.origin_x, args.origin_y],
            "maximum": [
                args.origin_x + args.size,
                args.origin_y + args.size,
            ],
        },
        "local_frame": {
            "center_original_xy": [center_x, center_y],
            "axes": {
                "x": "east",
                "y": "up",
                "negative_z": "north",
            },
            "matrix_row_major": transform.tolist(),
        },
        "camera": {
            "position": [
                args.size * 0.66,
                args.size * 0.38,
                args.size * 0.70,
            ],
            "target": [0.0, 24.0, 0.0],
            "fov_degrees": 48.0,
        },
        "visual": {
            "format": "glTF 2.0 binary",
            "source_lod": args.visual_lod,
            "tiles": visual_tiles,
            "vertices": sum(tile["vertices"] for tile in visual_tiles),
            "triangles": sum(tile["triangles"] for tile in visual_tiles),
            "bytes": sum(tile["bytes"] for tile in visual_tiles),
            "overview": {
                "source_lod": args.overview_lod,
                "tiles": overview_tiles,
                "vertices": sum(tile["vertices"] for tile in overview_tiles),
                "triangles": sum(tile["triangles"] for tile in overview_tiles),
                "bytes": sum(tile["bytes"] for tile in overview_tiles),
            },
        },
        "collision": {
            "uri": f"collision/{collision_path.name}",
            "source_lod": args.collision_lod,
            "source_objects": collision_source_objects,
            "mode": "triangle-mesh BVH",
            "watertight": bool(collision_mesh.is_watertight),
            "vertices": int(len(collision_mesh.vertices)),
            "triangles": int(len(collision_mesh.faces)),
            "bytes": collision_path.stat().st_size,
            "bounds": {
                "minimum": collision_mesh.bounds[0].tolist(),
                "maximum": collision_mesh.bounds[1].tolist(),
            },
            "heightmap": {
                "uri": "diagnostics/heightmap_0p5m.npz",
                "resolution_m": args.grid_resolution,
                "shape": list(heightmap.shape),
                "minimum_height_m": float(finite_heights.min()),
                "maximum_height_m": float(finite_heights.max()),
            },
            "esdf_slices": slices,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
