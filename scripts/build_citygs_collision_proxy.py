"""Build a multi-resolution collision proxy from a CityGS point cloud.

The Gaussian splat remains the visual source of truth.  Collision and routing
use three distinct products in the exact same centered, Y-up metric frame:

* 1.00 m dense global occupancy + height map for global route planning.
* 0.25 m sparse surface voxels for final local obstacle clearance checks.
* 5.00 m cost map for inexpensive fleet scheduling only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_citygs_splats import open_ply_memmap, sigmoid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CityGS collision hierarchy.")
    parser.add_argument(
        "--ply",
        type=Path,
        default=Path(
            r"C:\Users\caste\Downloads\Residence\residence_c20_r4_light_60_vq\point_cloud.ply"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/citygs_visualization/Residence/Residence_browser_config.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/citygs_collision/Residence"))
    parser.add_argument("--extent", type=float, default=500.0)
    parser.add_argument("--global-resolution", type=float, default=1.0)
    parser.add_argument("--local-resolution", type=float, default=0.25)
    parser.add_argument("--cost-resolution", type=float, default=5.0)
    parser.add_argument("--min-y", type=float, default=-20.0)
    parser.add_argument("--max-y", type=float, default=150.0)
    parser.add_argument("--metric-scale", type=float, default=0.5)
    parser.add_argument("--opacity", type=float, default=0.12)
    parser.add_argument("--min-points-per-global-voxel", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=300_000)
    return parser.parse_args()


def create_semantic_blocks(extent: float) -> list[dict]:
    districts = [
        "residential", "mixed", "park", "residential",
        "mixed", "cbd", "plaza", "mixed",
        "residential", "mixed", "cbd", "residential",
        "park", "residential", "mixed", "industrial",
    ]
    half = extent / 2.0
    step = extent / 4.0
    blocks = []
    for ix in range(4):
        for iz in range(4):
            x0 = -half + ix * step
            z0 = -half + iz * step
            block_id = ix * 4 + iz + 1
            blocks.append(
                {
                    "id": block_id,
                    "name": f"R-{ix + 1}{iz + 1}",
                    "district": districts[block_id - 1],
                    "polygon": [
                        [x0, z0],
                        [x0 + step, z0],
                        [x0 + step, z0 + step],
                        [x0, z0 + step],
                    ],
                    "area": step * step,
                    "num_buildings": 0,
                    "buildings": [],
                    "metadata": {"source": "CityGS metric partition"},
                }
            )
    return blocks


def decode_sparse_keys(keys: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Decode packed xyz keys to compact uint16 voxel coordinates."""
    ny = np.uint64(shape[1])
    nz = np.uint64(shape[2])
    yz = ny * nz
    x = keys // yz
    remainder = keys % yz
    y = remainder // nz
    z = remainder % nz
    return np.column_stack((x, y, z)).astype(np.uint16)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    vertices = open_ply_memmap(args.ply)

    extent = float(args.extent)
    half = extent / 2.0
    bounds_min = np.array([-half, args.min_y, -half], dtype=np.float32)
    bounds_max = np.array([half, args.max_y, half], dtype=np.float32)

    global_resolution = float(args.global_resolution)
    local_resolution = float(args.local_resolution)
    cost_resolution = float(args.cost_resolution)
    global_shape = np.ceil((bounds_max - bounds_min) / global_resolution).astype(np.int32)
    local_shape = np.ceil((bounds_max - bounds_min) / local_resolution).astype(np.int32)

    # uint16 is sufficient because only a small threshold is used downstream.
    global_counts = np.zeros(tuple(global_shape), dtype=np.uint16)
    local_key_chunks: list[np.ndarray] = []

    alignment_rotation = Rotation.from_quat(config["sceneQuaternion"]).as_matrix().astype(np.float32)
    alignment_translation = np.asarray(config["scenePosition"], dtype=np.float32)
    aligned_target = np.asarray(config["initialCameraLookAt"], dtype=np.float32)

    total_kept = 0
    for start in range(0, len(vertices), args.chunk_size):
        stop = min(len(vertices), start + args.chunk_size)
        chunk = vertices[start:stop]
        alpha_mask = sigmoid(np.asarray(chunk["opacity"], dtype=np.float32)) >= args.opacity
        if not np.any(alpha_mask):
            continue

        raw = np.column_stack(
            (
                np.asarray(chunk["x"], dtype=np.float32),
                np.asarray(chunk["y"], dtype=np.float32),
                np.asarray(chunk["z"], dtype=np.float32),
            )
        )[alpha_mask]
        aligned = raw @ alignment_rotation.T + alignment_translation
        world = np.column_stack(
            (
                aligned[:, 0] - aligned_target[0],
                aligned[:, 2],
                -(aligned[:, 1] - aligned_target[1]),
            )
        ) * float(args.metric_scale)
        inside = np.all((world >= bounds_min) & (world < bounds_max), axis=1)
        world = world[inside]
        if not len(world):
            continue

        global_indices = np.floor((world - bounds_min) / global_resolution).astype(np.int32)
        np.add.at(
            global_counts,
            (global_indices[:, 0], global_indices[:, 1], global_indices[:, 2]),
            1,
        )

        local_indices = np.floor((world - bounds_min) / local_resolution).astype(np.uint64)
        local_keys = (
            (local_indices[:, 0] * np.uint64(local_shape[1]) + local_indices[:, 1])
            * np.uint64(local_shape[2])
            + local_indices[:, 2]
        )
        local_key_chunks.append(local_keys)
        total_kept += len(world)
        print(
            f"[CityGS proxy] {stop:,}/{len(vertices):,} splats; "
            f"{total_kept:,} inside the 500 m operation area"
        )

    # The 1 m global layer is conservative: retain measured surfaces, expand one
    # cell, then solid-fill building columns using the recovered roof envelope.
    surface = global_counts >= int(args.min_points_per_global_voxel)
    surface = ndimage.binary_dilation(surface, iterations=1)
    has_surface = np.any(surface, axis=1)
    highest = global_shape[1] - 1 - np.argmax(surface[:, ::-1, :], axis=1)
    heightmap = bounds_min[1] + (highest.astype(np.float32) + 1.0) * global_resolution
    heightmap[~has_surface] = 0.0
    heightmap[heightmap < 6.0] = 0.0
    heightmap = ndimage.maximum_filter(heightmap, size=3)

    building_mask = heightmap > 12.0
    building_mask = ndimage.binary_opening(building_mask, structure=np.ones((2, 2)))
    safe_mask = ndimage.binary_dilation(building_mask, iterations=2)
    safe_heightmap = ndimage.maximum_filter(heightmap, size=5)
    safe_heightmap[~safe_mask] = 0.0

    occupancy = np.zeros(tuple(global_shape), dtype=np.uint8)
    ground_index = max(0, int(np.floor((0.0 - bounds_min[1]) / global_resolution)))
    for gx, gz in zip(*np.nonzero(safe_mask)):
        top_index = int(
            np.ceil((safe_heightmap[gx, gz] - bounds_min[1]) / global_resolution)
        )
        occupancy[gx, ground_index:min(top_index + 1, global_shape[1]), gz] = 1

    # Keep every observed 0.25 m surface voxel.  This is intentionally sparse:
    # a full 2000 x 680 x 2000 dense volume would be 2.72 billion cells.
    if local_key_chunks:
        local_keys = np.unique(np.concatenate(local_key_chunks))
        local_coords = decode_sparse_keys(local_keys, local_shape)
    else:
        local_coords = np.empty((0, 3), dtype=np.uint16)

    labels, _ = ndimage.label(building_mask)
    buildings = []
    for component_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        x_slice, z_slice = slices
        component_cells = labels[x_slice, z_slice] == component_id
        area_cells = int(np.count_nonzero(component_cells))
        if area_cells < 8:
            continue
        component_height = float(np.max(heightmap[labels == component_id]))
        buildings.append(
            {
                "id": len(buildings) + 1,
                "original_group": f"citygs_component_{component_id:04d}",
                "bounds_min": [
                    float(bounds_min[0] + x_slice.start * global_resolution),
                    0.0,
                    float(bounds_min[2] + z_slice.start * global_resolution),
                ],
                "bounds_max": [
                    float(bounds_min[0] + x_slice.stop * global_resolution),
                    component_height,
                    float(bounds_min[2] + z_slice.stop * global_resolution),
                ],
                "height": component_height,
                "num_faces_original": area_cells,
                "num_faces_simplified": 12,
                "style": "citygs_proxy_1m",
            }
        )

    blocks = create_semantic_blocks(extent)
    for building in buildings:
        center_x = (building["bounds_min"][0] + building["bounds_max"][0]) / 2.0
        center_z = (building["bounds_min"][2] + building["bounds_max"][2]) / 2.0
        for block in blocks:
            polygon = block["polygon"]
            if (
                polygon[0][0] <= center_x <= polygon[1][0]
                and polygon[0][1] <= center_z <= polygon[2][1]
            ):
                block["buildings"].append(building["id"])
                block["num_buildings"] += 1
                break

    # The 5 m map is explicitly a scheduling cost layer, not a collision map.
    cost_factor = int(round(cost_resolution / global_resolution))
    if not np.isclose(cost_factor * global_resolution, cost_resolution):
        raise ValueError("cost resolution must be an integer multiple of global resolution")
    cost_shape = (
        safe_heightmap.shape[0] // cost_factor,
        safe_heightmap.shape[1] // cost_factor,
    )
    trimmed = safe_heightmap[
        : cost_shape[0] * cost_factor,
        : cost_shape[1] * cost_factor,
    ]
    cost_heightmap = trimmed.reshape(
        cost_shape[0], cost_factor, cost_shape[1], cost_factor
    ).max(axis=(1, 3))
    scheduling_cost = 1.0 + np.clip(cost_heightmap / 80.0, 0.0, 2.0)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "occupancy_grid.npz",
        grid=occupancy,
        origin=bounds_min,
        resolution=np.float32(global_resolution),
    )
    np.savez_compressed(
        args.output / "heightmap.npz",
        heightmap=safe_heightmap.astype(np.float32),
        resolution=np.float32(global_resolution),
    )
    np.savez_compressed(
        args.output / "local_collision_sparse.npz",
        coords=local_coords,
        origin=bounds_min,
        resolution=np.float32(local_resolution),
        shape=local_shape,
    )
    np.savez_compressed(
        args.output / "cost_grid_5m.npz",
        cost=scheduling_cost.astype(np.float32),
        heightmap=cost_heightmap.astype(np.float32),
        origin=bounds_min[[0, 2]],
        resolution=np.float32(cost_resolution),
    )
    (args.output / "buildings.json").write_text(
        json.dumps(buildings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "road_network.json").write_text(
        json.dumps({"blocks": blocks, "roads": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scene_config = {
        "name": "Residence CityGS 500m Digital Twin",
        "source": str(args.ply),
        "visual_asset": config["modelAssetRelativePath"],
        "bounds_center": [0.0, (args.max_y + args.min_y) / 2.0, 0.0],
        "bounds_size": [extent, args.max_y - args.min_y, extent],
        "num_buildings": len(buildings),
        "grid_resolution": global_resolution,
        "global_collision_resolution": global_resolution,
        "local_collision_resolution": local_resolution,
        "cost_grid_resolution": cost_resolution,
        "default_task_count": 12,
        "coordinate_frame": "Y-up ENU-like local metric frame",
        "collision_hierarchy": {
            "global": {
                "file": "occupancy_grid.npz",
                "resolution_m": global_resolution,
                "purpose": "global route planning",
            },
            "local": {
                "file": "local_collision_sparse.npz",
                "resolution_m": local_resolution,
                "purpose": "final static obstacle clearance",
            },
            "cost": {
                "file": "cost_grid_5m.npz",
                "resolution_m": cost_resolution,
                "purpose": "fleet scheduling cost only",
            },
        },
        "alignment": {
            "metric_scale": args.metric_scale,
            "scene_position": config["scenePosition"],
            "scene_quaternion": config["sceneQuaternion"],
            "center_target": config["initialCameraLookAt"],
        },
        "proxy_stats": {
            "source_splats": len(vertices),
            "splats_in_operation_area": total_kept,
            "occupied_global_voxels": int(np.count_nonzero(occupancy)),
            "global_grid_shape": global_shape.tolist(),
            "local_surface_voxels": int(len(local_coords)),
            "local_grid_shape": local_shape.tolist(),
            "cost_grid_shape": list(cost_shape),
        },
    }
    (args.output / "scene_config.json").write_text(
        json.dumps(scene_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[CityGS proxy] wrote {args.output}: "
        f"global={tuple(global_shape)} @ {global_resolution:.2f} m, "
        f"local={len(local_coords):,} sparse voxels @ {local_resolution:.2f} m, "
        f"cost={cost_shape} @ {cost_resolution:.2f} m"
    )


if __name__ == "__main__":
    main()
