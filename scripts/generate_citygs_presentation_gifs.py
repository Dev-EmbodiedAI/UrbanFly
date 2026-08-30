from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_citygs_viewers import (
    apply_alignment,
    build_camera_geometry,
    estimate_alignment_transform,
    load_cameras,
    read_gaussian_sample,
    set_axes_bounds,
)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    point_cloud_path: Path
    cameras_path: Path
    top_gif_path: Path
    output_gif_path: Path
    target_points: int


DEFAULT_SCENES = [
    SceneSpec(
        name="Residence",
        point_cloud_path=Path(r"C:\Users\caste\Downloads\Residence\residence_c20_r4_light_50_vq\point_cloud.ply"),
        cameras_path=Path(r"C:\Users\caste\Downloads\Residence\residence_c20_r4\cameras.json"),
        top_gif_path=Path("data/citygs_visualization/Residence/Residence_camera_pose_turntable.gif"),
        output_gif_path=Path("data/citygs_visualization/Residence/Residence_presentation_dense.gif"),
        target_points=5_000_000,
    ),
    SceneSpec(
        name="SciArt",
        point_cloud_path=Path(r"C:\Users\caste\Downloads\SciArt\sciart_c9_r4\point_cloud\iteration_30000\point_cloud.ply"),
        cameras_path=Path(r"C:\Users\caste\Downloads\SciArt\sciart_c9_r4\cameras.json"),
        top_gif_path=Path("data/citygs_visualization/SciArt/SciArt_camera_pose_turntable.gif"),
        output_gif_path=Path("data/citygs_visualization/SciArt/SciArt_presentation_dense.gif"),
        target_points=3_798_926,
    ),
]


def render_dense_bottom_frames(
    scene_name: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    width: int,
    height: int,
    frames: int,
    plot_points: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    plot_count = min(plot_points, xyz.shape[0])
    indices = (
        rng.choice(xyz.shape[0], size=plot_count, replace=False)
        if xyz.shape[0] > plot_count
        else np.arange(xyz.shape[0], dtype=np.int64)
    )
    indices.sort()
    sampled_xyz = xyz[indices]
    sampled_rgb = rgb[indices] / 255.0

    dpi = 160
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(
        sampled_xyz[:, 0],
        sampled_xyz[:, 1],
        sampled_xyz[:, 2],
        c=sampled_rgb,
        s=0.16,
        alpha=0.78,
        linewidths=0,
    )
    ax.set_title(f"{scene_name} - Dense Point Cloud", fontsize=14, pad=10)
    set_axes_bounds(ax, sampled_xyz)
    ax.set_axis_off()
    fig.tight_layout(pad=0.05)

    rendered_frames: list[np.ndarray] = []
    for azimuth in np.linspace(46.0, 406.0, frames, endpoint=False):
        ax.view_init(elev=18.0, azim=float(azimuth))
        fig.canvas.draw()
        rendered_frames.append(np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy())

    plt.close(fig)
    return rendered_frames


def combine_frames(
    top_frames: list[np.ndarray],
    bottom_frames: list[np.ndarray],
    width: int,
) -> list[np.ndarray]:
    frame_count = len(bottom_frames)
    top_indices = np.linspace(0, len(top_frames) - 1, frame_count).round().astype(np.int64)
    combined: list[np.ndarray] = []

    resized_top_frames = []
    for top_frame in top_frames:
        top_image = Image.fromarray(top_frame)
        top_height = max(int(round(top_image.height * (width / top_image.width))), 1)
        resized_top_frames.append(
            np.asarray(
                top_image.resize((width, top_height), resample=Image.Resampling.LANCZOS)
            )
        )

    for frame_index, bottom_frame in enumerate(bottom_frames):
        bottom_image = Image.fromarray(bottom_frame)
        if bottom_image.width != width:
            bottom_height = max(int(round(bottom_image.height * (width / bottom_image.width))), 1)
            bottom_frame = np.asarray(
                bottom_image.resize((width, bottom_height), resample=Image.Resampling.LANCZOS)
            )

        top_frame = resized_top_frames[int(top_indices[frame_index])]
        canvas = np.full(
            (top_frame.shape[0] + bottom_frame.shape[0], width, 3),
            255,
            dtype=np.uint8,
        )
        canvas[: top_frame.shape[0], :, :] = top_frame
        canvas[top_frame.shape[0] :, :, :] = bottom_frame
        combined.append(canvas)

    return combined


def process_scene(
    scene: SceneSpec,
    frames: int,
    fps: int,
    width: int,
    bottom_height: int,
    plot_points: int,
    chunk_size: int,
    seed: int,
) -> None:
    print(f"\n=== {scene.name} ===")
    xyz, rgb, _alpha = read_gaussian_sample(
        ply_path=scene.point_cloud_path,
        target_points=scene.target_points,
        opacity_min=0.0,
        oversample_factor=1.0,
        chunk_size=chunk_size,
        seed=seed,
    )
    cameras = load_cameras(scene.cameras_path)
    raw_camera_path, _raw_frustum_lines = build_camera_geometry(
        cameras=cameras,
        scene_center=xyz.mean(axis=0),
        bbox_diag=float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))),
    )
    alignment_origin, alignment_basis = estimate_alignment_transform(xyz, raw_camera_path)
    xyz = apply_alignment(xyz, alignment_origin, alignment_basis)

    bottom_frames = render_dense_bottom_frames(
        scene_name=scene.name,
        xyz=xyz,
        rgb=rgb,
        width=width,
        height=bottom_height,
        frames=frames,
        plot_points=plot_points,
        seed=seed + 17,
    )

    top_frames = [np.asarray(frame)[..., :3] for frame in imageio.mimread(scene.top_gif_path)]
    combined_frames = combine_frames(top_frames, bottom_frames, width)

    scene.output_gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(scene.output_gif_path, combined_frames, fps=fps)
    print(f"Wrote {scene.output_gif_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense CityGS presentation GIFs.")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--bottom-height", type=int, default=760)
    parser.add_argument("--plot-points", type=int, default=140000)
    parser.add_argument("--chunk-size", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for index, scene in enumerate(DEFAULT_SCENES):
        process_scene(
            scene=scene,
            frames=args.frames,
            fps=args.fps,
            width=args.width,
            bottom_height=args.bottom_height,
            plot_points=args.plot_points,
            chunk_size=args.chunk_size,
            seed=args.seed + index * 1000,
        )


if __name__ == "__main__":
    main()
