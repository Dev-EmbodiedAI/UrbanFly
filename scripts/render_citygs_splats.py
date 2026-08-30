from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


@dataclass(frozen=True)
class SceneSpec:
    name: str
    point_cloud_path: Path
    cameras_path: Path


DEFAULT_SCENES = [
    SceneSpec(
        name="Residence",
        point_cloud_path=Path(
            r"C:\Users\caste\Downloads\Residence\residence_c20_r4\point_cloud\iteration_30000\point_cloud.ply"
        ),
        cameras_path=Path(r"C:\Users\caste\Downloads\Residence\residence_c20_r4\cameras.json"),
    ),
    SceneSpec(
        name="SciArt",
        point_cloud_path=Path(
            r"C:\Users\caste\Downloads\SciArt\sciart_c9_r4\point_cloud\iteration_30000\point_cloud.ply"
        ),
        cameras_path=Path(r"C:\Users\caste\Downloads\SciArt\sciart_c9_r4\cameras.json"),
    ),
]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def parse_ply_header(ply_path: Path) -> tuple[int, int, np.dtype]:
    props: list[tuple[str, str]] = []
    vertex_count = 0
    with ply_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading {ply_path}")
            decoded = line.decode("ascii", errors="strict").strip()
            if decoded.startswith("element vertex"):
                vertex_count = int(decoded.split()[2])
            elif decoded.startswith("property"):
                _, data_type, name = decoded.split()
                props.append((name, data_type))
            elif decoded == "end_header":
                header_end = handle.tell()
                break
    dtype_map = {
        "float": "<f4",
        "float32": "<f4",
        "uchar": "u1",
        "uint8": "u1",
        "int": "<i4",
        "int32": "<i4",
    }
    dtype_fields = []
    for name, data_type in props:
        if data_type not in dtype_map:
            raise ValueError(f"Unsupported property type {data_type!r} in {ply_path}")
        dtype_fields.append((name, dtype_map[data_type]))
    return vertex_count, header_end, np.dtype(dtype_fields)


def open_ply_memmap(ply_path: Path) -> np.memmap:
    vertex_count, header_end, dtype = parse_ply_header(ply_path)
    return np.memmap(
        ply_path,
        dtype=dtype,
        mode="r",
        offset=header_end,
        shape=(vertex_count,),
    )


def load_cameras(cameras_path: Path) -> list[dict]:
    with cameras_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_orientation_basis(
    rotation_matrices: np.ndarray,
    camera_positions: np.ndarray,
    scene_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_sets = []
    row_right = rotation_matrices[:, 0, :]
    row_up = -rotation_matrices[:, 1, :]
    row_forward = rotation_matrices[:, 2, :]
    col_right = rotation_matrices[:, :, 0]
    col_up = -rotation_matrices[:, :, 1]
    col_forward = rotation_matrices[:, :, 2]

    candidate_sets.append((row_right, row_up, row_forward))
    candidate_sets.append((row_right, row_up, -row_forward))
    candidate_sets.append((col_right, col_up, col_forward))
    candidate_sets.append((col_right, col_up, -col_forward))

    view_to_center = scene_center[None, :] - camera_positions
    view_norm = np.linalg.norm(view_to_center, axis=1, keepdims=True).clip(min=1e-6)
    view_to_center = view_to_center / view_norm

    best_score = -np.inf
    best_basis = None
    for right, up, forward in candidate_sets:
        normed_forward = forward / np.linalg.norm(forward, axis=1, keepdims=True).clip(min=1e-6)
        score = np.mean(np.sum(normed_forward * view_to_center, axis=1))
        if score > best_score:
            best_score = score
            best_basis = (right, up, normed_forward)
    assert best_basis is not None

    right, up, forward = best_basis
    right = right / np.linalg.norm(right, axis=1, keepdims=True).clip(min=1e-6)
    up = up / np.linalg.norm(up, axis=1, keepdims=True).clip(min=1e-6)
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True).clip(min=1e-6)
    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def build_camera_geometry(
    cameras: list[dict],
    scene_center: np.ndarray,
    bbox_diag: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([cam["position"] for cam in cameras], dtype=np.float32)
    rotations = np.asarray([cam["rotation"] for cam in cameras], dtype=np.float32)
    right, up, forward = infer_orientation_basis(rotations, positions, scene_center)

    step = max(len(cameras) // 48, 1)
    subset = np.arange(0, len(cameras), step, dtype=np.int32)
    if subset[-1] != len(cameras) - 1:
        subset = np.append(subset, len(cameras) - 1)

    frustum_depth = max(bbox_diag * 0.018, 0.8)
    frustum_width = frustum_depth * 0.55
    frustum_height = frustum_depth * 0.32

    segments: list[np.ndarray] = []
    for idx in subset:
        origin = positions[idx]
        fwd = forward[idx]
        side = right[idx]
        vertical = up[idx]
        face_center = origin + fwd * frustum_depth
        corners = [
            face_center + side * frustum_width + vertical * frustum_height,
            face_center - side * frustum_width + vertical * frustum_height,
            face_center - side * frustum_width - vertical * frustum_height,
            face_center + side * frustum_width - vertical * frustum_height,
        ]
        for corner in corners:
            segments.append(np.stack([origin, corner], axis=0))
        for a, b in zip(corners, corners[1:] + corners[:1]):
            segments.append(np.stack([a, b], axis=0))

    frustum_lines = (
        np.stack(segments, axis=0).astype(np.float32)
        if segments
        else np.empty((0, 2, 3), np.float32)
    )
    return positions.astype(np.float32), frustum_lines


def estimate_alignment_transform(
    xyz: np.ndarray,
    camera_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.median(xyz, axis=0).astype(np.float32)
    camera_mean = camera_positions.mean(axis=0).astype(np.float32)
    up_axis = camera_mean - origin
    up_norm = float(np.linalg.norm(up_axis))
    if up_norm < 1e-6:
        up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        up_axis = up_axis / up_norm

    centered = xyz - origin[None, :]
    projected = centered - np.outer(centered @ up_axis, up_axis)
    covariance = projected.T @ projected / max(projected.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    forward_axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    forward_axis = forward_axis - up_axis * float(np.dot(forward_axis, up_axis))
    forward_norm = float(np.linalg.norm(forward_axis))
    if forward_norm < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, up_axis))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        forward_axis = fallback - up_axis * float(np.dot(fallback, up_axis))
        forward_norm = float(np.linalg.norm(forward_axis))
    forward_axis = forward_axis / max(forward_norm, 1e-6)

    lateral_axis = np.cross(up_axis, forward_axis).astype(np.float32)
    lateral_norm = float(np.linalg.norm(lateral_axis))
    if lateral_norm < 1e-6:
        lateral_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        lateral_axis = lateral_axis - up_axis * float(np.dot(lateral_axis, up_axis))
        lateral_axis = lateral_axis / max(float(np.linalg.norm(lateral_axis)), 1e-6)
    else:
        lateral_axis = lateral_axis / lateral_norm

    forward_axis = np.cross(lateral_axis, up_axis).astype(np.float32)
    forward_axis = forward_axis / max(float(np.linalg.norm(forward_axis)), 1e-6)

    basis = np.stack([forward_axis, lateral_axis, up_axis], axis=1).astype(np.float32)
    return origin, basis


def apply_alignment(values: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
    original_shape = values.shape
    flattened = values.reshape(-1, 3).astype(np.float32)
    transformed = (flattened - origin[None, :]) @ basis
    return transformed.reshape(original_shape).astype(np.float32)


def invert_alignment(values: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
    original_shape = values.shape
    flattened = values.reshape(-1, 3).astype(np.float32)
    transformed = flattened @ basis.T + origin[None, :]
    return transformed.reshape(original_shape).astype(np.float32)


def sample_scene_points(
    vertices: np.memmap,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = len(vertices)
    rng = np.random.default_rng(seed)
    if total <= sample_size:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = rng.choice(total, size=sample_size, replace=False)
        indices.sort()
    xyz = np.stack(
        [vertices["x"][indices], vertices["y"][indices], vertices["z"][indices]],
        axis=1,
    ).astype(np.float32)
    rgb = np.stack(
        [vertices["f_dc_0"][indices], vertices["f_dc_1"][indices], vertices["f_dc_2"][indices]],
        axis=1,
    ).astype(np.float32)
    rgb = np.clip(0.5 + 0.28209479177387814 * rgb, 0.0, 1.0)
    return xyz, (rgb * 255.0).round().astype(np.uint8)


def collect_active_gaussians(
    vertices: np.memmap,
    opacity_threshold: float,
    origin: np.ndarray,
    basis: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    active_chunks: list[np.ndarray] = []
    aligned_chunks: list[np.ndarray] = []
    total = len(vertices)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        opacities = sigmoid(vertices["opacity"][start:end].astype(np.float32))
        mask = opacities > opacity_threshold
        if not np.any(mask):
            continue
        local_indices = np.flatnonzero(mask).astype(np.int64)
        global_indices = local_indices + start
        xyz = np.stack(
            [
                vertices["x"][global_indices],
                vertices["y"][global_indices],
                vertices["z"][global_indices],
            ],
            axis=1,
        ).astype(np.float32)
        aligned_xyz = apply_alignment(xyz, origin, basis)
        active_chunks.append(global_indices.astype(np.int32))
        aligned_chunks.append(aligned_xyz)
        kept = sum(chunk.shape[0] for chunk in active_chunks)
        print(f"  active preload {end:,}/{total:,}, kept {kept:,}")
    active_indices = np.concatenate(active_chunks, axis=0)
    aligned_means = np.concatenate(aligned_chunks, axis=0)
    return active_indices, aligned_means


def percentile_bounds(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
    mins = np.percentile(values, low, axis=0)
    maxs = np.percentile(values, high, axis=0)
    return mins.astype(np.float32), maxs.astype(np.float32)


def compute_orbit_config(
    aligned_xyz_sample: np.ndarray,
    aligned_camera_positions: np.ndarray,
    camera_intrinsics: list[dict],
) -> dict[str, float]:
    mins, maxs = percentile_bounds(aligned_xyz_sample, low=1.0, high=99.0)
    center_xy = ((mins[:2] + maxs[:2]) / 2.0).astype(np.float32)
    target_z = float(mins[2] * 0.2 + maxs[2] * 0.8)
    target = np.array([center_xy[0], center_xy[1], target_z], dtype=np.float32)

    scene_radius_xy = float(np.max(np.linalg.norm(aligned_xyz_sample[:, :2] - center_xy[None, :], axis=1)))
    scene_height = float(maxs[2] - mins[2])

    cam_xy_radius = np.linalg.norm(aligned_camera_positions[:, :2] - center_xy[None, :], axis=1)
    orbit_radius = float(max(np.percentile(cam_xy_radius, 60), scene_radius_xy * 1.45))
    orbit_height = float(
        max(
            np.percentile(aligned_camera_positions[:, 2], 55),
            maxs[2] + max(scene_height * 0.22, scene_radius_xy * 0.18),
        )
    )

    median_fx = float(np.median([cam["fx"] for cam in camera_intrinsics]))
    median_fy = float(np.median([cam["fy"] for cam in camera_intrinsics]))
    median_width = float(np.median([cam["width"] for cam in camera_intrinsics]))
    median_height = float(np.median([cam["height"] for cam in camera_intrinsics]))
    fov_y = 2.0 * math.atan(median_height / (2.0 * median_fy))
    fov_y = float(np.clip(fov_y, math.radians(32.0), math.radians(58.0)))
    fov_x = 2.0 * math.atan(median_width / (2.0 * median_fx))
    fov_x = float(np.clip(fov_x, math.radians(38.0), math.radians(72.0)))

    return {
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "target_z": float(target[2]),
        "orbit_radius": orbit_radius,
        "orbit_height": orbit_height,
        "fov_x": fov_x,
        "fov_y": fov_y,
        "scene_radius_xy": scene_radius_xy,
        "scene_height": scene_height,
    }


def make_intrinsics(width: int, height: int, fov_y: float) -> np.ndarray:
    fy = height / (2.0 * math.tan(fov_y / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize near-zero vector")
    return (vector / norm).astype(np.float32)


def make_orbit_pose(
    theta: float,
    orbit_radius: float,
    orbit_height: float,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.array(
        [
            target[0] + orbit_radius * math.cos(theta),
            target[1] + orbit_radius * math.sin(theta),
            orbit_height,
        ],
        dtype=np.float32,
    )
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = normalize(target - position)
    right = normalize(np.cross(up_world, forward))
    down = normalize(np.cross(right, forward))
    return position, right, down, forward


def world_to_camera_matrix(
    position: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    forward: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    position_world = invert_alignment(position[None, :], origin, basis)[0]
    right_world = basis @ right
    down_world = basis @ down
    forward_world = basis @ forward
    R = np.stack([right_world, down_world, forward_world], axis=0).astype(np.float32)
    t = -(R @ position_world.astype(np.float32))
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat


def cull_visible_indices(
    aligned_means: np.ndarray,
    active_indices: np.ndarray,
    position: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    forward: np.ndarray,
    tan_fov_x: float,
    tan_fov_y: float,
    near_plane: float,
    far_plane: float,
    chunk_size: int,
) -> np.ndarray:
    visible_chunks: list[np.ndarray] = []
    total = aligned_means.shape[0]
    margin = 1.18
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        points = aligned_means[start:end]
        rel = points - position[None, :]
        cam_x = rel @ right
        cam_y = rel @ down
        cam_z = rel @ forward
        mask = (cam_z > near_plane) & (cam_z < far_plane)
        if np.any(mask):
            x_over_z = np.abs(cam_x[mask] / cam_z[mask])
            y_over_z = np.abs(cam_y[mask] / cam_z[mask])
            submask = (x_over_z < tan_fov_x * margin) & (y_over_z < tan_fov_y * margin)
            if np.any(submask):
                visible_chunks.append(active_indices[start:end][mask][submask].astype(np.int32))
    if not visible_chunks:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(visible_chunks, axis=0)


def extract_gaussian_batch(vertices: np.memmap, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.stack(
        [vertices["x"][indices], vertices["y"][indices], vertices["z"][indices]],
        axis=1,
    ).astype(np.float32)
    quats = np.stack(
        [
            vertices["rot_0"][indices],
            vertices["rot_1"][indices],
            vertices["rot_2"][indices],
            vertices["rot_3"][indices],
        ],
        axis=1,
    ).astype(np.float32)
    scales = np.exp(
        np.stack(
            [
                vertices["scale_0"][indices],
                vertices["scale_1"][indices],
                vertices["scale_2"][indices],
            ],
            axis=1,
        ).astype(np.float32)
    )
    opacities = sigmoid(vertices["opacity"][indices].astype(np.float32))

    coeffs = np.empty((indices.shape[0], 16, 3), dtype=np.float32)
    coeffs[:, 0, 0] = vertices["f_dc_0"][indices]
    coeffs[:, 0, 1] = vertices["f_dc_1"][indices]
    coeffs[:, 0, 2] = vertices["f_dc_2"][indices]
    for sh_idx in range(15):
        base = sh_idx * 3
        coeffs[:, sh_idx + 1, 0] = vertices[f"f_rest_{base}"][indices]
        coeffs[:, sh_idx + 1, 1] = vertices[f"f_rest_{base + 1}"][indices]
        coeffs[:, sh_idx + 1, 2] = vertices[f"f_rest_{base + 2}"][indices]
    return means, quats, scales, opacities, coeffs


def set_axes_bounds(ax: plt.Axes, *arrays: np.ndarray) -> None:
    valid_arrays = [arr for arr in arrays if arr.size]
    if not valid_arrays:
        return
    stacked = np.concatenate(valid_arrays, axis=0)
    mins = np.percentile(stacked, 1.0, axis=0)
    maxs = np.percentile(stacked, 99.0, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


class PoseFrameRenderer:
    def __init__(
        self,
        scene_name: str,
        xyz: np.ndarray,
        rgb: np.ndarray,
        camera_path: np.ndarray,
        frustum_lines: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        dpi = 160
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self.ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=rgb / 255.0,
            s=0.38,
            alpha=0.64,
            linewidths=0,
        )
        self.ax.plot(
            camera_path[:, 0],
            camera_path[:, 1],
            camera_path[:, 2],
            color="#d62728",
            linewidth=1.85,
            alpha=0.98,
        )
        if frustum_lines.size:
            self.ax.add_collection3d(
                Line3DCollection(
                    frustum_lines,
                    colors="#1f77b4",
                    linewidths=0.75,
                    alpha=0.58,
                )
            )
        self.ax.set_title(f"{scene_name} - Camera Trajectory", fontsize=12, pad=8)
        set_axes_bounds(self.ax, xyz, camera_path)
        self.ax.set_axis_off()
        self.fig.tight_layout(pad=0.1)

    def render(self, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        self.ax.view_init(elev=float(elevation_deg), azim=float(azimuth_deg))
        self.fig.canvas.draw()
        frame = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        return frame.copy()

    def close(self) -> None:
        plt.close(self.fig)


def save_pose_overview(
    output_path: Path,
    scene_name: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    camera_path: np.ndarray,
    frustum_lines: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(13.5, 7.8), dpi=220, facecolor="white")
    views = [
        (1, 23.0, 38.0, "Perspective"),
        (2, 89.0, -90.0, "Top View"),
    ]
    for subplot_index, elev, azim, title in views:
        ax = fig.add_subplot(1, 2, subplot_index, projection="3d")
        ax.set_title(f"{scene_name} - {title}", fontsize=13, pad=12)
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=rgb / 255.0,
            s=0.34,
            alpha=0.62,
            linewidths=0,
        )
        ax.plot(
            camera_path[:, 0],
            camera_path[:, 1],
            camera_path[:, 2],
            color="#d62728",
            linewidth=1.75,
            alpha=0.98,
        )
        if frustum_lines.size:
            ax.add_collection3d(
                Line3DCollection(
                    frustum_lines,
                    colors="#1f77b4",
                    linewidths=0.72,
                    alpha=0.56,
                )
            )
        ax.view_init(elev=elev, azim=azim)
        set_axes_bounds(ax, xyz, camera_path)
        ax.set_axis_off()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_scene(
    scene: SceneSpec,
    output_root: Path,
    device: torch.device,
    width: int,
    height: int,
    fps: int,
    frames: int,
    opacity_threshold: float,
    preload_chunk: int,
    cull_chunk: int,
    sample_points: int,
    pose_points: int,
    gif_max_width: int,
    seed: int,
) -> None:
    print(f"\n=== {scene.name} ===")
    print(f"Loading cameras from {scene.cameras_path}")
    cameras = load_cameras(scene.cameras_path)
    vertices = open_ply_memmap(scene.point_cloud_path)
    print(f"PLY gaussians: {len(vertices):,}")

    sample_xyz, sample_rgb = sample_scene_points(vertices, sample_points, seed)
    camera_positions = np.asarray([cam["position"] for cam in cameras], dtype=np.float32)
    origin, basis = estimate_alignment_transform(sample_xyz, camera_positions)
    aligned_sample_xyz = apply_alignment(sample_xyz, origin, basis)
    aligned_camera_positions = apply_alignment(camera_positions, origin, basis)

    bbox_diag = float(np.linalg.norm(aligned_sample_xyz.max(axis=0) - aligned_sample_xyz.min(axis=0)))
    _, frustum_lines = build_camera_geometry(cameras, np.median(sample_xyz, axis=0).astype(np.float32), bbox_diag)
    aligned_frustum_lines = apply_alignment(frustum_lines, origin, basis) if frustum_lines.size else frustum_lines

    print(f"Preloading active gaussians with opacity > {opacity_threshold}")
    active_indices, aligned_means = collect_active_gaussians(
        vertices=vertices,
        opacity_threshold=opacity_threshold,
        origin=origin,
        basis=basis,
        chunk_size=preload_chunk,
    )
    print(f"Active gaussians kept: {active_indices.shape[0]:,}")

    rng = np.random.default_rng(seed + 17)
    pose_plot_count = min(pose_points, aligned_sample_xyz.shape[0])
    pose_indices = (
        rng.choice(aligned_sample_xyz.shape[0], size=pose_plot_count, replace=False)
        if aligned_sample_xyz.shape[0] > pose_plot_count
        else np.arange(aligned_sample_xyz.shape[0], dtype=np.int64)
    )
    pose_xyz = aligned_sample_xyz[pose_indices]
    pose_rgb = sample_rgb[pose_indices]

    orbit = compute_orbit_config(aligned_sample_xyz, aligned_camera_positions, cameras)
    target = np.array([orbit["target_x"], orbit["target_y"], orbit["target_z"]], dtype=np.float32)
    far_plane = max(orbit["orbit_radius"] * 4.0, orbit["scene_radius_xy"] * 5.0)
    near_plane = 0.25
    K = make_intrinsics(width, height, orbit["fov_y"])
    tan_fov_x = width / (2.0 * K[0, 0])
    tan_fov_y = height / (2.0 * K[1, 1])

    scene_dir = output_root / scene.name
    scene_dir.mkdir(parents=True, exist_ok=True)

    pose_overview_path = scene_dir / f"{scene.name}_camera_pose.png"
    save_pose_overview(
        pose_overview_path,
        scene.name,
        pose_xyz,
        pose_rgb,
        aligned_camera_positions,
        aligned_frustum_lines,
    )

    render_video_path = scene_dir / f"{scene.name}_splat_turntable.mp4"
    composite_video_path = scene_dir / f"{scene.name}_presentation.mp4"
    render_gif_path = scene_dir / f"{scene.name}_splat_turntable.gif"
    composite_gif_path = scene_dir / f"{scene.name}_presentation.gif"
    preview_path = scene_dir / f"{scene.name}_preview.png"
    meta_path = scene_dir / f"{scene.name}_render_meta.json"

    render_writer = imageio.get_writer(
        render_video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    composite_writer = imageio.get_writer(
        composite_video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )

    pose_renderer = PoseFrameRenderer(
        scene_name=scene.name,
        xyz=pose_xyz,
        rgb=pose_rgb,
        camera_path=aligned_camera_positions,
        frustum_lines=aligned_frustum_lines,
        width=width,
        height=max(height // 3, 280),
    )

    render_gif_frames: list[np.ndarray] = []
    composite_gif_frames: list[np.ndarray] = []

    background = torch.ones((1, 3), dtype=torch.float32, device=device)
    K_torch = torch.from_numpy(K).to(device=device)[None, :, :]

    preview_saved = False
    frame_stats: list[dict[str, float]] = []

    try:
        for frame_idx, theta in enumerate(np.linspace(0.0, math.tau, frames, endpoint=False)):
            position, right, down, forward = make_orbit_pose(
                theta=theta,
                orbit_radius=orbit["orbit_radius"],
                orbit_height=orbit["orbit_height"],
                target=target,
            )
            visible_indices = cull_visible_indices(
                aligned_means=aligned_means,
                active_indices=active_indices,
                position=position,
                right=right,
                down=down,
                forward=forward,
                tan_fov_x=tan_fov_x,
                tan_fov_y=tan_fov_y,
                near_plane=near_plane,
                far_plane=far_plane,
                chunk_size=cull_chunk,
            )
            print(f"  frame {frame_idx + 1:03d}/{frames:03d}: visible gaussians {visible_indices.shape[0]:,}")
            if visible_indices.shape[0] == 0:
                raise RuntimeError(f"No visible gaussians for frame {frame_idx} in {scene.name}")

            means_np, quats_np, scales_np, opacities_np, coeffs_np = extract_gaussian_batch(vertices, visible_indices)
            viewmat = world_to_camera_matrix(
                position=position,
                right=right,
                down=down,
                forward=forward,
                origin=origin,
                basis=basis,
            )

            means = torch.from_numpy(means_np).to(device=device)
            quats = torch.from_numpy(quats_np).to(device=device)
            scales = torch.from_numpy(scales_np).to(device=device)
            opacities = torch.from_numpy(opacities_np).to(device=device)
            coeffs = torch.from_numpy(coeffs_np).to(device=device)
            viewmats = torch.from_numpy(viewmat).to(device=device)[None, :, :]

            with torch.no_grad():
                renders, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=coeffs,
                    viewmats=viewmats,
                    Ks=K_torch,
                    width=width,
                    height=height,
                    sh_degree=3,
                    packed=True,
                    backgrounds=background,
                    render_mode="RGB",
                    rasterize_mode="antialiased",
                    near_plane=near_plane,
                    far_plane=far_plane,
                    radius_clip=0.0,
                )
            render_frame = torch.clamp(renders[0], 0.0, 1.0).mul(255.0).byte().cpu().numpy()
            if not preview_saved:
                imageio.imwrite(preview_path, render_frame)
                preview_saved = True

            pose_frame = pose_renderer.render(
                azimuth_deg=float(math.degrees(theta) + 42.0),
                elevation_deg=24.0,
            )
            pose_height = pose_frame.shape[0]
            canvas = np.full((pose_height + height, width, 3), 255, dtype=np.uint8)
            canvas[:pose_height, :, :] = pose_frame[:, :width, :]
            canvas[pose_height : pose_height + height, :, :] = render_frame

            render_writer.append_data(render_frame)
            composite_writer.append_data(canvas)

            if width > gif_max_width:
                gif_scale = gif_max_width / width
                gif_render_height = max(int(round(height * gif_scale)), 1)
                gif_pose_height = max(int(round(pose_height * gif_scale)), 1)
                gif_render = np.asarray(
                    Image.fromarray(render_frame).resize(
                        (gif_max_width, gif_render_height),
                        resample=Image.Resampling.LANCZOS,
                    )
                )
                gif_composite = np.asarray(
                    Image.fromarray(canvas).resize(
                        (gif_max_width, gif_pose_height + gif_render_height),
                        resample=Image.Resampling.LANCZOS,
                    )
                )
            else:
                gif_render = render_frame
                gif_composite = canvas
            render_gif_frames.append(gif_render)
            composite_gif_frames.append(gif_composite)

            frame_stats.append(
                {
                    "frame": frame_idx,
                    "theta_deg": float(math.degrees(theta)),
                    "visible_gaussians": int(visible_indices.shape[0]),
                }
            )

            del means, quats, scales, opacities, coeffs, viewmats, renders
            torch.cuda.empty_cache()
    finally:
        pose_renderer.close()
        render_writer.close()
        composite_writer.close()

    imageio.mimsave(render_gif_path, render_gif_frames, fps=fps)
    imageio.mimsave(composite_gif_path, composite_gif_frames, fps=fps)

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scene": scene.name,
                "point_cloud_path": str(scene.point_cloud_path),
                "cameras_path": str(scene.cameras_path),
                "width": width,
                "height": height,
                "fps": fps,
                "frames": frames,
                "opacity_threshold": opacity_threshold,
                "active_gaussians": int(active_indices.shape[0]),
                "orbit": orbit,
                "frame_stats": frame_stats,
            },
            handle,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render true Gaussian splat videos for CityGS scenes.")
    parser.add_argument("--scene", action="append", dest="scenes", help="Scene name to process. Can be repeated.")
    parser.add_argument("--output-root", type=Path, default=Path("data/citygs_visualization"))
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=810)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--opacity-threshold", type=float, default=0.005)
    parser.add_argument("--preload-chunk", type=int, default=600000)
    parser.add_argument("--cull-chunk", type=int, default=1200000)
    parser.add_argument("--sample-points", type=int, default=180000)
    parser.add_argument("--pose-points", type=int, default=70000)
    parser.add_argument("--gif-max-width", type=int, default=960)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_lookup = {scene.name.lower(): scene for scene in DEFAULT_SCENES}
    if args.scenes:
        selected_scenes = [scene_lookup[name.lower()] for name in args.scenes]
    else:
        selected_scenes = DEFAULT_SCENES

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for true Gaussian splat rendering.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for scene in selected_scenes:
        process_scene(
            scene=scene,
            output_root=args.output_root,
            device=device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frames=args.frames,
            opacity_threshold=args.opacity_threshold,
            preload_chunk=args.preload_chunk,
            cull_chunk=args.cull_chunk,
            sample_points=args.sample_points,
            pose_points=args.pose_points,
            gif_max_width=args.gif_max_width,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
