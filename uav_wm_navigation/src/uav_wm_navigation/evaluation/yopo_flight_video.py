from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .yopo_visualizer import depth_to_flu_points


def _project_world(
    trajectory_world: np.ndarray,
    camera_pose_nwu: np.ndarray,
    depth_intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> np.ndarray:
    """Project NWU points through the recorded camera calibration.

    ``camera_pose_nwu`` stores camera-FLU to world. The pinhole camera is
    forward/left/up in this representation, hence u=-y/x and v=-z/x.
    """
    height, width = image_shape
    depth_height, depth_width = depth_shape
    rotation_world_camera = camera_pose_nwu[:3, :3]
    camera_position = camera_pose_nwu[:3, 3]
    trajectory = (trajectory_world - camera_position[None]) @ rotation_world_camera
    forward = trajectory[:, 0]
    valid = forward > 0.15
    scale_x = width / max(depth_width, 1)
    scale_y = height / max(depth_height, 1)
    fx = float(depth_intrinsics[0, 0]) * scale_x
    fy = float(depth_intrinsics[1, 1]) * scale_y
    cx = (float(depth_intrinsics[0, 2]) + 0.5) * scale_x - 0.5
    cy = (float(depth_intrinsics[1, 2]) + 0.5) * scale_y - 0.5
    u = fx * (-trajectory[:, 1]) / np.maximum(forward, 0.15) + cx
    v = fy * (-trajectory[:, 2]) / np.maximum(forward, 0.15) + cy
    projected = np.column_stack([u, v])
    projected[~valid] = np.nan
    return projected


def _draw_candidates_on_rgb(
    rgb: np.ndarray, candidates_world: np.ndarray, selected: int, scores: np.ndarray,
    camera_pose_nwu: np.ndarray, camera_intrinsics: np.ndarray, depth_shape: tuple[int, int],
    method_label: str = "YOPO", collision_probability: np.ndarray | None = None,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    finite = scores[np.isfinite(scores)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    normalized = (
        np.nan_to_num(scores, nan=high, posinf=high, neginf=low) - low
    ) / max(high - low, 1e-6)
    for index, trajectory in enumerate(candidates_world):
        projected = _project_world(trajectory, camera_pose_nwu, camera_intrinsics, rgb.shape[:2], depth_shape)
        valid = np.isfinite(projected).all(axis=1)
        points = np.round(projected[valid]).astype(np.int32)
        inside = ((points[:, 0] >= 0) & (points[:, 0] < rgb.shape[1]) &
                  (points[:, 1] >= 0) & (points[:, 1] < rgb.shape[0])) if len(points) else np.zeros(0, bool)
        points = points[inside]
        if len(points) < 2:
            continue
        if index == selected:
            color, thickness = (60, 255, 60), 5
        elif not np.isfinite(scores[index]):
            color, thickness = (105, 105, 105), 1
        else:
            color = tuple(int(value) for value in cv2.applyColorMap(
                np.array([[int((1.0 - normalized[index]) * 255)]], dtype=np.uint8), cv2.COLORMAP_TURBO
            )[0, 0])
            thickness = 2
        cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[-1]), 4 if index == selected else 2, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, f"{method_label}: 15 candidates", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"selected #{selected}", (24, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 255, 60), 2, cv2.LINE_AA)
    if collision_probability is not None and 0 <= selected < len(collision_probability):
        risk = float(collision_probability[selected])
        if np.isfinite(risk):
            cv2.putText(canvas, f"predicted collision risk {risk:.3f}", (24, 108), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, (80, 220, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _render_frame(data: dict[str, np.ndarray], frame_index: int) -> np.ndarray:
    rgb = data["rgb"][frame_index]
    candidates = data["candidates"][frame_index]
    candidates_world = data.get("candidates_world", data["candidates"])[frame_index]
    selected = int(data["selected"][frame_index])
    scores = data["total_score"][frame_index]
    raw_method = str(data["method"][frame_index]) if "method" in data else "yopo"
    method_label = "YOPO" if raw_method == "yopo" else "YOPO + " + raw_method.removeprefix("yopo_").upper()
    collision_probability = data.get("collision_probability")
    frame_collision_probability = None if collision_probability is None else collision_probability[frame_index]
    if "camera_pose_nwu" in data and "camera_intrinsics" in data and "candidates_world" in data:
        overlay = _draw_candidates_on_rgb(
            rgb, candidates_world, selected, scores,
            data["camera_pose_nwu"][frame_index], data["camera_intrinsics"][frame_index],
            data["depth"][frame_index].shape,
            method_label, frame_collision_probability,
        )
    else:
        # Compatibility for old archives; new recordings always use calibrated projection.
        overlay = _draw_candidates_on_rgb(
            rgb, candidates, selected, scores, np.eye(4),
            np.array([[rgb.shape[1] / 2, 0, (rgb.shape[1] - 1) / 2],
                      [0, rgb.shape[1] / 2, (rgb.shape[0] - 1) / 2], [0, 0, 1]]), rgb.shape[:2],
            method_label, frame_collision_probability,
        )
    chase_frames = data.get("third_person_rgb")
    has_chase = chase_frames is not None and np.any(chase_frames)
    points = depth_to_flu_points(data["depth"][frame_index], stride=3)
    positions = data["position_nwu"][: frame_index + 1]
    goal = data["goal_nwu"][frame_index]
    if frame_index > 0 and "elapsed_s" in data:
        dt = max(float(data["elapsed_s"][frame_index] - data["elapsed_s"][frame_index - 1]), 1e-3)
        speed = float(np.linalg.norm(positions[-1] - positions[-2]) / dt)
    else:
        speed = float(np.linalg.norm(data["velocity_nwu"][frame_index]))

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#090d14")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.58, 1.0], hspace=0.22, wspace=0.12)
    if has_chase:
        chase_axis = fig.add_subplot(grid[0, 0])
        chase_axis.imshow(chase_frames[frame_index]); chase_axis.set_axis_off()
        chase_axis.set_title("Chase camera: visible UAV flight", color="white", fontsize=13, pad=8)
        camera_axis = fig.add_subplot(grid[1, 0])
    else:
        camera_axis = fig.add_subplot(grid[:, 0])
    camera_axis.imshow(overlay); camera_axis.set_axis_off()
    camera_axis.set_title("Live front camera + projected YOPO primitives", color="white", fontsize=13, pad=8)

    local_axis = fig.add_subplot(grid[0, 1], projection="3d")
    cloud_available = False
    if len(points):
        keep = (points[:, 0] < 16.0) & (np.abs(points[:, 1]) < 8.0)
        cloud = points[keep]
        cloud_available = len(cloud) > 0
        if cloud_available:
            if len(cloud) > 4500:
                cloud = cloud[np.linspace(0, len(cloud) - 1, 4500).astype(int)]
            local_axis.scatter(
                cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 0], cmap="winter",
                s=1.8, alpha=0.28, depthshade=False, rasterized=True,
            )
    for index, trajectory in enumerate(candidates):
        local_axis.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            color="#55ff55" if index == selected else "#ff8b3d",
            lw=4.0 if index == selected else 1.0, alpha=1.0 if index == selected else 0.62,
        )
    local_axis.scatter(0, 0, 0, marker="^", s=80, color="white", depthshade=False)
    local_axis.set(
        xlim=(-1, 15), ylim=(-7, 7), zlim=(-3, 5), xlabel="Forward / m", ylabel="Left / m",
        zlabel="Up / m", title="Metric 3D depth cloud + 15 YOPO primitives",
    )
    local_axis.view_init(elev=24, azim=-60)
    local_axis.set_box_aspect((1.55, 1.0, 0.72))

    world_axis = fig.add_subplot(grid[1, 1])
    world_axis.plot(positions[:, 0], positions[:, 1], color="#42d7ff", lw=3)
    world_axis.scatter(positions[-1, 0], positions[-1, 1], marker="^", s=100, color="#55ff55", label="UAV")
    world_axis.scatter(goal[0], goal[1], marker="*", s=140, color="#ffd84a", label="goal")
    margin = 4.0
    x_values = np.r_[positions[:, 0], goal[0]]; y_values = np.r_[positions[:, 1], goal[1]]
    world_axis.set_xlim(float(x_values.min() - margin), float(x_values.max() + margin))
    world_axis.set_ylim(float(y_values.min() - margin), float(y_values.max() + margin))
    world_axis.set(xlabel="NWU x / m", ylabel="NWU y / m", title=f"Executed flight · {speed:.2f} m/s")
    world_axis.legend(loc="upper left", fontsize=8)
    for axis in (local_axis, world_axis):
        axis.set_facecolor("#111827"); axis.tick_params(colors="white")
        axis.xaxis.label.set_color("white"); axis.yaxis.label.set_color("white")
        axis.title.set_color("white"); axis.grid(color="white", alpha=0.12)
        for spine in axis.spines.values(): spine.set_color("#4b5563")
    fig.suptitle(f"YOPO closed-loop flight · Town10HD · step {frame_index + 1}/{len(data['rgb'])}",
                 color="white", fontsize=16, y=0.985)
    world_axis.set_title(f"Executed trajectory | speed {speed:.2f} m/s")
    local_axis.zaxis.label.set_color("white")
    local_axis.zaxis.set_tick_params(colors="white")
    fig.suptitle(
        f"{method_label} closed-loop flight | Town10HD | step {frame_index + 1}/{len(data['rgb'])}",
        color="white", fontsize=16, y=0.985,
    )
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def render_yopo_flight_video(
    npz_path: str | Path, output_mp4: str | Path, output_gif: str | Path | None = None,
    fps: float = 5.0,
) -> tuple[Path, Path | None]:
    with np.load(Path(npz_path).resolve()) as archive:
        data = {name: archive[name] for name in archive.files}
    acceptance_path = Path(npz_path).resolve().with_name("acceptance.json")
    if acceptance_path.exists():
        import json
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if not acceptance.get("passed", False):
            failed = [name for name, passed in acceptance.get("checks", {}).items() if not passed]
            raise ValueError(f"recording failed visualization acceptance gates: {failed}")
    if "rgb" not in data or not np.any(data["rgb"]):
        raise ValueError("the run contains no real RGB frames; Mock data cannot produce the requested flight video")
    output = Path(output_mp4).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    gif_frames = []
    try:
        for frame_index in range(len(data["rgb"])):
            frame = _render_frame(data, frame_index)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if output_gif is not None:
                gif_frames.append(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA))
    finally:
        writer.release()
    gif_path = None
    if output_gif is not None:
        from PIL import Image
        gif_path = Path(output_gif).resolve(); gif_path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in gif_frames]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0,
                       optimize=True)
    return output, gif_path
