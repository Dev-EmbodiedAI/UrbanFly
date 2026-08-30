from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import ndimage

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_citygs_splats import (
    apply_alignment,
    estimate_alignment_transform,
    invert_alignment,
    load_cameras,
    open_ply_memmap,
    sample_scene_points,
)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    point_cloud_path: Path
    cameras_path: Path


DEFAULT_SCENES = [
    SceneSpec(
        name="Residence",
        point_cloud_path=Path(
            r"C:\Users\caste\Downloads\Residence\residence_c20_r4_light_60_vq\point_cloud.ply"
        ),
        cameras_path=Path(r"C:\Users\caste\Downloads\Residence\residence_c20_r4\cameras.json"),
    ),
]


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    m = rotation.astype(np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float32)
    quat /= np.linalg.norm(quat).clip(min=1e-8)
    return quat


def build_density_component(
    aligned_xyz: np.ndarray,
    grid_size: int,
    low_z_pct: float,
    high_z_pct: float,
    density_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zmin, zmax = np.percentile(aligned_xyz[:, 2], [low_z_pct, high_z_pct])
    mask = (aligned_xyz[:, 2] >= zmin) & (aligned_xyz[:, 2] <= zmax)
    pts = aligned_xyz[mask]

    mins = np.percentile(pts[:, :2], 1.0, axis=0)
    maxs = np.percentile(pts[:, :2], 99.0, axis=0)
    density, xedges, yedges = np.histogram2d(
        pts[:, 0],
        pts[:, 1],
        bins=grid_size,
        range=[[mins[0], maxs[0]], [mins[1], maxs[1]]],
    )
    density = ndimage.gaussian_filter(density, sigma=2.0)
    threshold = np.percentile(density[density > 0], density_pct)
    occupancy = density >= threshold
    labels, count = ndimage.label(occupancy)
    if count == 0:
        raise RuntimeError("No connected component found while extracting building footprint")
    peak_index = np.unravel_index(np.argmax(density), density.shape)
    peak_label = int(labels[peak_index])
    if peak_label == 0:
        component_sizes = ndimage.sum(occupancy, labels, index=np.arange(1, count + 1))
        peak_label = int(np.argmax(component_sizes) + 1)
    component_mask = labels == peak_label
    return pts, density, component_mask, xedges, yedges, np.array([zmin, zmax], dtype=np.float32)


def select_points_from_component(
    aligned_xyz: np.ndarray,
    component_mask: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
) -> np.ndarray:
    xbins = np.clip(np.digitize(aligned_xyz[:, 0], xedges) - 1, 0, component_mask.shape[0] - 1)
    ybins = np.clip(np.digitize(aligned_xyz[:, 1], yedges) - 1, 0, component_mask.shape[1] - 1)
    mask = component_mask[xbins, ybins]
    return aligned_xyz[mask]


def bbox_corners_aligned(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    hx, hy, hz = size / 2.0
    corners = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    return corners + center[None, :]


def bbox_edges(corners: np.ndarray) -> np.ndarray:
    indices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return np.stack([corners[[a, b]] for a, b in indices], axis=0)


def save_visualization(
    output_path: Path,
    scene_name: str,
    sample_xyz: np.ndarray,
    sample_rgb: np.ndarray,
    bbox_edges_aligned: np.ndarray,
    footprint_xy: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(13.6, 7.2), dpi=220, facecolor="white")

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        sample_xyz[:, 0],
        sample_xyz[:, 1],
        sample_xyz[:, 2],
        c=sample_rgb / 255.0,
        s=0.28,
        alpha=0.48,
        linewidths=0,
    )
    ax1.add_collection3d(Line3DCollection(bbox_edges_aligned, colors="#d62728", linewidths=2.0))
    ax1.set_title(f"{scene_name} - Building BBox", fontsize=13, pad=12)
    ax1.view_init(elev=22, azim=36)
    ax1.set_axis_off()

    mins = np.percentile(sample_xyz, 1, axis=0)
    maxs = np.percentile(sample_xyz, 99, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), 1.0)
    ax1.set_xlim(center[0] - radius, center[0] + radius)
    ax1.set_ylim(center[1] - radius, center[1] + radius)
    ax1.set_zlim(center[2] - radius, center[2] + radius)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(sample_xyz[:, 0], sample_xyz[:, 1], c=sample_rgb / 255.0, s=0.2, alpha=0.22, linewidths=0)
    ax2.plot(*np.vstack([footprint_xy, footprint_xy[:1]]).T, color="#d62728", linewidth=2.2)
    ax2.set_title(f"{scene_name} - Footprint", fontsize=13, pad=10)
    ax2.set_aspect("equal", adjustable="box")
    ax2.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_scene(
    scene: SceneSpec,
    output_root: Path,
    sample_points: int,
    grid_size: int,
    density_pct: float,
    seed: int,
) -> None:
    print(f"Extracting building bbox for {scene.name}")
    vertices = open_ply_memmap(scene.point_cloud_path)
    cameras = load_cameras(scene.cameras_path)
    camera_positions = np.asarray([cam["position"] for cam in cameras], dtype=np.float32)
    sample_xyz, sample_rgb = sample_scene_points(vertices, sample_points, seed)
    origin, basis = estimate_alignment_transform(sample_xyz, camera_positions)
    aligned_xyz = apply_alignment(sample_xyz, origin, basis)

    _, _, component_mask, xedges, yedges, _ = build_density_component(
        aligned_xyz=aligned_xyz,
        grid_size=grid_size,
        low_z_pct=5.0,
        high_z_pct=95.0,
        density_pct=density_pct,
    )
    selected = select_points_from_component(aligned_xyz, component_mask, xedges, yedges)
    if selected.shape[0] < 1000:
        raise RuntimeError("Too few points selected for building bbox extraction")

    mins = np.percentile(selected, 1.0, axis=0).astype(np.float32)
    maxs = np.percentile(selected, 99.0, axis=0).astype(np.float32)
    center_aligned = (mins + maxs) / 2.0
    size_aligned = maxs - mins
    corners_aligned = bbox_corners_aligned(center_aligned, size_aligned)
    corners_world = invert_alignment(corners_aligned, origin, basis)
    center_world = invert_alignment(center_aligned[None, :], origin, basis)[0]
    rotation_world = basis.T.astype(np.float32)
    quaternion_world = matrix_to_quaternion(rotation_world)

    footprint_xy = np.array(
        [
            [mins[0], mins[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]],
            [mins[0], maxs[1]],
        ],
        dtype=np.float32,
    )

    bbox_json = {
        "scene": scene.name,
        "source_point_cloud": str(scene.point_cloud_path),
        "source_cameras": str(scene.cameras_path),
        "selection_point_count": int(selected.shape[0]),
        "coordinate_system": {
            "aligned_origin_world": origin.tolist(),
            "aligned_basis_columns_world": basis.tolist(),
            "bbox_rotation_world_matrix": rotation_world.tolist(),
            "bbox_rotation_world_quaternion_xyzw": quaternion_world.tolist(),
        },
        "aligned_bbox": {
            "center": center_aligned.tolist(),
            "size_xyz": size_aligned.tolist(),
            "half_size_xyz": (size_aligned / 2.0).tolist(),
            "min_xyz": mins.tolist(),
            "max_xyz": maxs.tolist(),
            "corners": corners_aligned.tolist(),
        },
        "world_bbox": {
            "center": center_world.tolist(),
            "size_xyz_aligned_axes": size_aligned.tolist(),
            "half_size_xyz_aligned_axes": (size_aligned / 2.0).tolist(),
            "rotation_world_matrix": rotation_world.tolist(),
            "rotation_world_quaternion_xyzw": quaternion_world.tolist(),
            "corners": corners_world.tolist(),
        },
    }

    scene_dir = output_root / scene.name
    scene_dir.mkdir(parents=True, exist_ok=True)
    json_path = scene_dir / f"{scene.name}_building_bbox.json"
    txt_path = scene_dir / f"{scene.name}_building_bbox.txt"
    viz_path = scene_dir / f"{scene.name}_building_bbox.png"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(bbox_json, handle, indent=2)

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Scene: {scene.name}\n")
        handle.write(f"Source point cloud: {scene.point_cloud_path}\n")
        handle.write(f"Selected building points: {selected.shape[0]:,}\n")
        handle.write(
            "Aligned bbox center xyz: "
            + ", ".join(f"{value:.6f}" for value in center_aligned)
            + "\n"
        )
        handle.write(
            "Aligned bbox size xyz: "
            + ", ".join(f"{value:.6f}" for value in size_aligned)
            + "\n"
        )
        handle.write(
            "World bbox center xyz: "
            + ", ".join(f"{value:.6f}" for value in center_world)
            + "\n"
        )
        handle.write(
            "World bbox quaternion xyzw: "
            + ", ".join(f"{value:.6f}" for value in quaternion_world)
            + "\n"
        )

    bbox_edges_aligned = bbox_edges(corners_aligned)
    save_visualization(
        output_path=viz_path,
        scene_name=scene.name,
        sample_xyz=aligned_xyz,
        sample_rgb=sample_rgb,
        bbox_edges_aligned=bbox_edges_aligned,
        footprint_xy=footprint_xy,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {viz_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract building bbox from CityGS point cloud.")
    parser.add_argument("--scene", action="append", dest="scenes", help="Scene name to process.")
    parser.add_argument("--output-root", type=Path, default=Path("data/citygs_visualization"))
    parser.add_argument("--sample-points", type=int, default=500000)
    parser.add_argument("--grid-size", type=int, default=320)
    parser.add_argument("--density-pct", type=float, default=72.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_lookup = {scene.name.lower(): scene for scene in DEFAULT_SCENES}
    selected = [scene_lookup[name.lower()] for name in args.scenes] if args.scenes else DEFAULT_SCENES
    for scene in selected:
        process_scene(
            scene=scene,
            output_root=args.output_root,
            sample_points=args.sample_points,
            grid_size=args.grid_size,
            density_pct=args.density_pct,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
