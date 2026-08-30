"""Build a watertight collision mesh and signed distance field for CityGS.

The source height envelope is reconstructed from real Gaussian centers by
``build_citygs_collision_proxy.py``.  This stage turns that conservative
envelope into closed static solids, adds a ground slab, and computes a dense
1 m signed distance field.  Fine 0.25 m surface detail remains in the sparse
local layer and is combined with this field at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from skimage import measure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CityGS collision geometry.")
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        default=Path("data/citygs_collision/Residence"),
    )
    parser.add_argument("--esdf-truncation", type=float, default=64.0)
    parser.add_argument("--target-triangles", type=int, default=250_000)
    parser.add_argument("--slice-altitudes", type=float, nargs="+", default=[30.0, 60.0])
    return parser.parse_args()


def add_quad(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    points: tuple[tuple[float, float, float], ...],
) -> None:
    base = len(vertices)
    vertices.extend(points)
    faces.append((base, base + 1, base + 2))
    faces.append((base, base + 2, base + 3))


def heightfield_to_watertight_mesh(
    heightmap: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    floor_y: float = 0.0,
) -> trimesh.Trimesh:
    """Convert occupied height columns to a closed union-of-columns surface."""
    occupied = heightmap > floor_y + 1e-4
    nx, nz = heightmap.shape
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for gx, gz in zip(*np.nonzero(occupied)):
        x0 = float(origin[0] + gx * resolution)
        x1 = x0 + resolution
        z0 = float(origin[2] + gz * resolution)
        z1 = z0 + resolution
        top = float(heightmap[gx, gz])

        # Top (+Y) and bottom (-Y).
        add_quad(
            vertices,
            faces,
            ((x0, top, z0), (x0, top, z1), (x1, top, z1), (x1, top, z0)),
        )
        add_quad(
            vertices,
            faces,
            (
                (x0, floor_y, z0),
                (x1, floor_y, z0),
                (x1, floor_y, z1),
                (x0, floor_y, z1),
            ),
        )

        west = float(heightmap[gx - 1, gz]) if gx > 0 and occupied[gx - 1, gz] else floor_y
        east = float(heightmap[gx + 1, gz]) if gx + 1 < nx and occupied[gx + 1, gz] else floor_y
        north = float(heightmap[gx, gz - 1]) if gz > 0 and occupied[gx, gz - 1] else floor_y
        south = float(heightmap[gx, gz + 1]) if gz + 1 < nz and occupied[gx, gz + 1] else floor_y

        if top > west + 1e-4:  # -X
            add_quad(
                vertices,
                faces,
                ((x0, west, z0), (x0, west, z1), (x0, top, z1), (x0, top, z0)),
            )
        if top > east + 1e-4:  # +X
            add_quad(
                vertices,
                faces,
                ((x1, east, z0), (x1, top, z0), (x1, top, z1), (x1, east, z1)),
            )
        if top > north + 1e-4:  # -Z
            add_quad(
                vertices,
                faces,
                ((x0, north, z0), (x0, top, z0), (x1, top, z0), (x1, north, z0)),
            )
        if top > south + 1e-4:  # +Z
            add_quad(
                vertices,
                faces,
                ((x0, south, z1), (x1, south, z1), (x1, top, z1), (x0, top, z1)),
            )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        process=True,
        validate=True,
    )
    mesh.remove_unreferenced_vertices()
    return mesh


def occupancy_to_watertight_mesh(
    occupancy: np.ndarray,
    origin: np.ndarray,
    resolution: float,
) -> trimesh.Trimesh:
    """Extract a manifold isosurface from solid occupied voxels."""
    solid = occupancy.astype(bool, copy=False)
    nonzero = np.argwhere(solid)
    if not len(nonzero):
        return trimesh.Trimesh()
    lower = np.maximum(nonzero.min(axis=0) - 1, 0)
    upper = np.minimum(nonzero.max(axis=0) + 2, np.asarray(solid.shape))
    cropped = solid[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ]
    padded = np.pad(cropped, 1, mode="constant", constant_values=False)
    vertices, faces, _, _ = measure.marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        spacing=(resolution, resolution, resolution),
        gradient_direction="descent",
        allow_degenerate=False,
        method="lewiner",
    )
    world_offset = origin + (lower.astype(np.float32) - 0.5) * resolution
    vertices += world_offset
    mesh = trimesh.Trimesh(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        process=True,
        validate=True,
    )
    mesh.remove_unreferenced_vertices()
    if mesh.is_volume and mesh.volume < 0:
        mesh.invert()
    return mesh


def create_ground_slab(origin: np.ndarray, grid_shape: tuple[int, int, int], resolution: float) -> trimesh.Trimesh:
    extent_x = grid_shape[0] * resolution
    extent_z = grid_shape[2] * resolution
    slab = trimesh.creation.box(extents=(extent_x, 1.0, extent_z))
    slab.apply_translation(
        (
            float(origin[0] + extent_x / 2.0),
            -0.5,
            float(origin[2] + extent_z / 2.0),
        )
    )
    return slab


def compute_signed_esdf(
    occupancy: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    truncation: float,
) -> np.ndarray:
    collision = occupancy.astype(bool, copy=True)
    ground_index = int(np.clip(np.floor((0.0 - origin[1]) / resolution), 0, collision.shape[1] - 1))
    collision[:, : ground_index + 1, :] = True

    print(f"[Collision geometry] EDT outside over {collision.shape} ...")
    outside = ndimage.distance_transform_edt(~collision, sampling=resolution)
    np.minimum(outside, truncation, out=outside)
    signed = outside.astype(np.float16)
    del outside

    print("[Collision geometry] EDT inside ...")
    inside = ndimage.distance_transform_edt(collision, sampling=resolution)
    np.minimum(inside, truncation, out=inside)
    signed[collision] = -inside[collision].astype(np.float16)
    return signed


def esdf_slice_rgba(distance_slice: np.ndarray, max_visual_distance: float = 20.0) -> np.ndarray:
    distance = np.asarray(distance_slice, dtype=np.float32)
    normalized = np.clip(distance / max_visual_distance, 0.0, 1.0)
    near = np.clip(1.0 - normalized, 0.0, 1.0)
    rgba = np.zeros((*distance.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.rint(255.0 * near).astype(np.uint8)
    rgba[..., 1] = np.rint(220.0 * normalized).astype(np.uint8)
    rgba[..., 2] = np.rint(255.0 * np.sqrt(normalized)).astype(np.uint8)
    rgba[..., 3] = np.where(distance < 0.0, 215, np.rint(145.0 * near)).astype(np.uint8)
    # Image rows represent Z and columns represent X.
    return np.transpose(rgba, (1, 0, 2))[::-1]


def main() -> None:
    args = parse_args()
    grid_data = np.load(args.proxy_dir / "occupancy_grid.npz")
    height_data = np.load(args.proxy_dir / "heightmap.npz")
    occupancy = grid_data["grid"]
    origin = grid_data["origin"].astype(np.float32)
    resolution = float(grid_data["resolution"])
    heightmap = height_data["heightmap"].astype(np.float32)

    building_mesh_full = occupancy_to_watertight_mesh(occupancy, origin, resolution)
    building_mesh = building_mesh_full
    if len(building_mesh_full.faces) > args.target_triangles:
        candidate = building_mesh_full.simplify_quadric_decimation(
            face_count=int(args.target_triangles)
        )
        if candidate.is_watertight and candidate.is_winding_consistent:
            building_mesh = candidate
        else:
            print("[Collision geometry] simplification rejected: topology was not preserved")
    ground_mesh = create_ground_slab(origin, occupancy.shape, resolution)
    collision_scene = trimesh.Scene(
        {
            "city_collision_buildings": building_mesh,
            "city_collision_ground": ground_mesh,
        }
    )
    glb_path = args.proxy_dir / "city_collision.glb"
    collision_scene.export(glb_path)

    esdf = compute_signed_esdf(
        occupancy,
        origin,
        resolution,
        float(args.esdf_truncation),
    )
    np.savez_compressed(
        args.proxy_dir / "global_esdf.npz",
        distance=esdf,
        origin=origin,
        resolution=np.float32(resolution),
        truncation=np.float32(args.esdf_truncation),
    )

    slice_manifest = []
    for altitude in args.slice_altitudes:
        gy = int(np.clip(np.floor((altitude - origin[1]) / resolution), 0, esdf.shape[1] - 1))
        image_name = f"esdf_slice_{int(round(altitude))}m.png"
        Image.fromarray(esdf_slice_rgba(esdf[:, gy, :]), mode="RGBA").save(
            args.proxy_dir / image_name,
            optimize=True,
        )
        slice_manifest.append(
            {
                "altitude_m": float(altitude),
                "grid_y": gy,
                "image": image_name,
                "max_visual_distance_m": 20.0,
            }
        )

    mesh_stats = {
        "source": "CityGS conservative height envelope",
        "coordinate_frame": "Y-up ENU-like local metric frame",
        "buildings": {
            "vertices": int(len(building_mesh.vertices)),
            "triangles": int(len(building_mesh.faces)),
            "source_triangles": int(len(building_mesh_full.faces)),
            "watertight": bool(building_mesh.is_watertight),
            "winding_consistent": bool(building_mesh.is_winding_consistent),
            "components": int(len(building_mesh.split(only_watertight=False))),
            "bounds": building_mesh.bounds.tolist(),
        },
        "ground": {
            "vertices": int(len(ground_mesh.vertices)),
            "triangles": int(len(ground_mesh.faces)),
            "watertight": bool(ground_mesh.is_watertight),
        },
        "global_esdf": {
            "shape": list(esdf.shape),
            "resolution_m": resolution,
            "truncation_m": float(args.esdf_truncation),
            "dtype": str(esdf.dtype),
            "min_m": float(np.min(esdf)),
            "max_m": float(np.max(esdf)),
        },
        "local_esdf": {
            "resolution_m": 0.25,
            "storage": "runtime_lru_tiles",
            "block_size": 16,
            "block_extent_m": 4.0,
            "max_resident_blocks": 256,
            "max_resident_megabytes": 2.097152,
            "distance_source": "local_collision_sparse.npz",
            "reason": (
                "A dense 0.25 m field over the full volume would contain "
                "2.72 billion samples; only queried 4 m tiles are materialized."
            ),
        },
        "slices": slice_manifest,
    }
    (args.proxy_dir / "collision_geometry.json").write_text(
        json.dumps(mesh_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(mesh_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
