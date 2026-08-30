from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_citygs_splats import (
    PoseFrameRenderer,
    apply_alignment,
    build_camera_geometry,
    compute_orbit_config,
    estimate_alignment_transform,
    load_cameras,
    open_ply_memmap,
    sample_scene_points,
    save_pose_overview,
)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    point_cloud_path: Path
    cameras_path: Path
    model_asset_name: str


DEFAULT_SCENES = [
    SceneSpec(
        name="Residence",
        point_cloud_path=Path(
            r"C:\Users\caste\Downloads\Residence\residence_c20_r4\point_cloud\iteration_30000\point_cloud.ply"
        ),
        cameras_path=Path(r"C:\Users\caste\Downloads\Residence\residence_c20_r4\cameras.json"),
        model_asset_name="Residence_full_deg3.ksplat",
    ),
    SceneSpec(
        name="SciArt",
        point_cloud_path=Path(
            r"C:\Users\caste\Downloads\SciArt\sciart_c9_r4\point_cloud\iteration_30000\point_cloud.ply"
        ),
        cameras_path=Path(r"C:\Users\caste\Downloads\SciArt\sciart_c9_r4\cameras.json"),
        model_asset_name="SciArt_full_deg3.ksplat",
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


def encode_array(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def process_scene(
    scene: SceneSpec,
    output_root: Path,
    sample_points: int,
    pose_points: int,
    frames: int,
    fps: int,
    width: int,
    height: int,
    seed: int,
) -> None:
    print(f"\n=== Preparing {scene.name} ===")
    vertices = open_ply_memmap(scene.point_cloud_path)
    cameras = load_cameras(scene.cameras_path)
    camera_positions = np.asarray([cam["position"] for cam in cameras], dtype=np.float32)

    sample_xyz, sample_rgb = sample_scene_points(vertices, sample_points, seed)
    origin, basis = estimate_alignment_transform(sample_xyz, camera_positions)
    aligned_xyz = apply_alignment(sample_xyz, origin, basis)
    aligned_camera_positions = apply_alignment(camera_positions, origin, basis)

    bbox_diag = float(np.linalg.norm(aligned_xyz.max(axis=0) - aligned_xyz.min(axis=0)))
    _, frustum_lines = build_camera_geometry(
        cameras,
        np.median(sample_xyz, axis=0).astype(np.float32),
        bbox_diag,
    )
    aligned_frustum_lines = apply_alignment(frustum_lines, origin, basis) if frustum_lines.size else frustum_lines

    rng = np.random.default_rng(seed + 31)
    pose_count = min(pose_points, aligned_xyz.shape[0])
    pose_indices = (
        rng.choice(aligned_xyz.shape[0], size=pose_count, replace=False)
        if aligned_xyz.shape[0] > pose_count
        else np.arange(aligned_xyz.shape[0], dtype=np.int64)
    )
    pose_xyz = aligned_xyz[pose_indices]
    pose_rgb = sample_rgb[pose_indices]

    orbit = compute_orbit_config(aligned_xyz, aligned_camera_positions, cameras)
    rotation_matrix = basis.T.astype(np.float32)
    translation = (-rotation_matrix @ origin).astype(np.float32)
    quaternion = matrix_to_quaternion(rotation_matrix)

    scene_dir = output_root / scene.name
    scene_dir.mkdir(parents=True, exist_ok=True)

    overview_path = scene_dir / f"{scene.name}_camera_pose_real.png"
    pose_video_path = scene_dir / f"{scene.name}_camera_pose_turntable.mp4"
    pose_gif_path = scene_dir / f"{scene.name}_camera_pose_turntable.gif"
    config_path = scene_dir / f"{scene.name}_browser_config.json"
    pose_asset_path = scene_dir / f"{scene.name}_pose_asset.json"

    save_pose_overview(
        overview_path,
        scene.name,
        pose_xyz,
        pose_rgb,
        aligned_camera_positions,
        aligned_frustum_lines,
    )

    pose_renderer = PoseFrameRenderer(
        scene_name=scene.name,
        xyz=pose_xyz,
        rgb=pose_rgb,
        camera_path=aligned_camera_positions,
        frustum_lines=aligned_frustum_lines,
        width=width,
        height=height,
    )
    pose_frames: list[np.ndarray] = []
    try:
        for theta in np.linspace(0.0, math.tau, frames, endpoint=False):
            frame = pose_renderer.render(
                azimuth_deg=float(math.degrees(theta) + 40.0),
                elevation_deg=24.0,
            )
            pose_frames.append(frame)
    finally:
        pose_renderer.close()

    imageio.mimsave(pose_video_path, pose_frames, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
    imageio.mimsave(pose_gif_path, pose_frames, fps=fps)

    config = {
        "sceneName": scene.name,
        "sourcePointCloud": str(scene.point_cloud_path),
        "sourceCameras": str(scene.cameras_path),
        "modelAssetName": scene.model_asset_name,
        "modelAssetRelativePath": f"../assets/{scene.model_asset_name}",
        "scenePosition": translation.tolist(),
        "sceneQuaternion": quaternion.tolist(),
        "sceneScale": [1.0, 1.0, 1.0],
        "cameraUp": [0.0, 0.0, 1.0],
        "orbit": orbit,
        "initialCameraPosition": [
            orbit["target_x"] + orbit["orbit_radius"],
            orbit["target_y"],
            orbit["orbit_height"],
        ],
        "initialCameraLookAt": [
            orbit["target_x"],
            orbit["target_y"],
            orbit["target_z"],
        ],
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    pose_asset = {
        "sceneName": scene.name,
        "pointCount": int(pose_xyz.shape[0]),
        "cameraCount": int(aligned_camera_positions.shape[0]),
        "frustumCount": int(aligned_frustum_lines.shape[0]),
        "boundsMin": aligned_xyz.min(axis=0).astype(np.float32).tolist(),
        "boundsMax": aligned_xyz.max(axis=0).astype(np.float32).tolist(),
        "pointPositions": encode_array(pose_xyz.astype(np.float32)),
        "pointColors": encode_array(pose_rgb.astype(np.uint8)),
        "cameraPath": encode_array(aligned_camera_positions.astype(np.float32)),
        "frustumLines": encode_array(aligned_frustum_lines.astype(np.float32)),
        "orbit": orbit,
    }
    with pose_asset_path.open("w", encoding="utf-8") as handle:
        json.dump(pose_asset, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare browser-viewer assets for CityGS scenes.")
    parser.add_argument("--scene", action="append", dest="scenes", help="Scene name to process. Can be repeated.")
    parser.add_argument("--output-root", type=Path, default=Path("data/citygs_visualization"))
    parser.add_argument("--sample-points", type=int, default=180000)
    parser.add_argument("--pose-points", type=int, default=80000)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
            pose_points=args.pose_points,
            frames=args.frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
