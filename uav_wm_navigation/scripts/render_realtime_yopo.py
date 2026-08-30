from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import numpy as np


OUTPUT_FPS = 60.0
OUTPUT_SIZE = (1920, 1080)
RVIZ_STYLE_VERSION = 3


def _metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_duration(metadata: dict) -> float:
    return float(metadata.get("simulation_duration_s", 0.0))


def time_correct_camera(source: Path, metadata: dict, output: Path, duration_s: float) -> dict:
    cap = cv2.VideoCapture(str(source))
    encoded_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    encoded_duration = max((frame_count - 1) / max(encoded_fps, 1e-6), 1e-6)
    time_scale = _source_duration(metadata) / encoded_duration
    target_frames = max(1, int(round(duration_s * OUTPUT_FPS)))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        f"setpts={time_scale:.10f}*PTS,"
        f"minterpolate=fps={OUTPUT_FPS:g}:mi_mode=blend,"
        "tpad=stop_mode=clone:stop_duration=0.100000,"
        f"trim=end_frame={target_frames},setpts=N/({OUTPUT_FPS:g}*TB),"
        f"scale={OUTPUT_SIZE[0]}:{OUTPUT_SIZE[1]},setsar=1"
    )
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-vf", filter_graph,
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        "-frames:v", str(target_frames), "-movflags", "+faststart", str(output),
    ], check=True)
    return {
        "source": str(source), "output": str(output), "source_frames": int(metadata["frame_count"]),
        "source_duration_s": _source_duration(metadata),
        "measured_source_fps": float(metadata["measured_simulation_fps"]),
        "encoded_source_fps": encoded_fps, "display_fps": OUTPUT_FPS,
        "display_duration_s": duration_s,
        "display_to_source_frame_ratio": duration_s * OUTPUT_FPS / max(int(metadata["frame_count"]), 1),
        "dropped_queue_frames": int(metadata.get("dropped_queue_frames", 0)),
        "temporal_processing": "timestamp correction plus optical blend interpolation; flight speed is unchanged",
    }


def _draw_text(frame: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.7,
               color: tuple[int, int, int] = (235, 242, 248), thickness: int = 1) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_DUPLEX, scale, (3, 8, 12), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def _project_flu(points: np.ndarray, origin: tuple[int, int], scale: float) -> np.ndarray:
    forward, left, up = points[:, 0], points[:, 1], points[:, 2]
    x = origin[0] - left * scale + forward * scale * 0.55
    y = origin[1] - up * scale - forward * scale * 0.28
    return np.column_stack([x, y]).astype(np.int32)


def _depth_points(depth_mm: np.ndarray, intrinsics: np.ndarray, stride: int = 5) -> tuple[np.ndarray, np.ndarray]:
    depth = depth_mm.astype(np.float32) * 0.001
    rows, cols = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    sampled = depth[::stride, ::stride]
    valid = (sampled > 0.15) & (sampled < 20.0)
    z = sampled[valid]
    u, v = cols[valid], rows[valid]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    # Camera RDF -> project convention FLU.
    points = np.column_stack([z, -(u - cx) * z / fx, -(v - cy) * z / fy])
    return points, z


def render_rviz(
    telemetry_path: Path, output: Path, duration_s: float, speed_target: float, method: str = "yopo"
) -> dict:
    with h5py.File(telemetry_path, "r") as handle:
        plans = {name: handle[f"plans/{name}"][:] for name in handle["plans"]}
        control = {name: handle[f"control/{name}"][:] for name in handle["control"]}
        goal = np.asarray(handle.attrs["goal_nwu"], dtype=np.float64)
    plan_times = np.asarray(plans["elapsed_s"], dtype=float)
    control_times = np.asarray(control["monotonic_s"], dtype=float)
    control_times -= control_times[0]
    actual_positions = np.asarray(control["actual_position_nwu"], dtype=float)
    actual_velocities = np.asarray(control["actual_velocity_nwu"], dtype=float)
    start = actual_positions[0]
    route = goal - start
    route_forward = route / max(float(np.linalg.norm(route)), 1e-9)
    route_left = np.array([-route_forward[1], route_forward[0], 0.0])
    route_left /= max(float(np.linalg.norm(route_left)), 1e-9)
    route_up = np.cross(route_forward, route_left)
    rotation = np.column_stack([route_forward, route_left, route_up])
    temporary = output.with_suffix(".mp4v.tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, OUTPUT_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"could not open RViz writer {temporary}")
    frame_total = max(1, int(round(duration_s * OUTPUT_FPS)))
    for frame_index in range(frame_total):
        elapsed = frame_index / OUTPUT_FPS
        plan_index = int(np.clip(np.searchsorted(plan_times, elapsed, side="right") - 1, 0, len(plan_times) - 1))
        control_index = int(np.clip(np.searchsorted(control_times, elapsed, side="right") - 1, 0, len(control_times) - 1))
        frame = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (17, 23, 29), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (OUTPUT_SIZE[0], 94), (10, 15, 20), -1)
        method_label = "YOPO" if method == "yopo" else f"YOPO + {method.upper()}"
        selection_label = "RAW COST ARGMIN" if method == "yopo" else "CALIBRATED RISK RERANK"
        _draw_text(frame, f"{method_label} REAL-TIME LOCAL PLANNING", (38, 45), 1.0, (246, 248, 250), 2)
        _draw_text(frame, f"REAL DEPTH  |  15 PRIMITIVES  |  {selection_label}", (40, 78), 0.60, (74, 225, 137), 1)
        depth_mm = np.asarray(plans["depth_mm"][plan_index])
        intrinsics = np.asarray(plans.get("camera_intrinsics", np.eye(3)[None])[plan_index])
        points, ranges = _depth_points(depth_mm, intrinsics)
        view_origin = (720, 920)
        view_scale = 70.0
        # Five-metre perspective grid, analogous to the spatial reference in
        # the official RViz view.
        for forward_m in (0.0, 5.0, 10.0, 15.0, 20.0):
            grid = np.array([[forward_m, -8.0, 0.0], [forward_m, 8.0, 0.0]])
            cv2.polylines(frame, [_project_flu(grid, view_origin, view_scale)], False, (40, 52, 62), 1, cv2.LINE_AA)
        for left_m in (-8.0, -4.0, 0.0, 4.0, 8.0):
            grid = np.array([[0.0, left_m, 0.0], [20.0, left_m, 0.0]])
            cv2.polylines(frame, [_project_flu(grid, view_origin, view_scale)], False, (40, 52, 62), 1, cv2.LINE_AA)
        point_pixels = _project_flu(points, view_origin, view_scale)
        inside = (
            (point_pixels[:, 0] >= 20) & (point_pixels[:, 0] < 1510)
            & (point_pixels[:, 1] >= 110) & (point_pixels[:, 1] < 1030)
        )
        colors = cv2.applyColorMap(
            np.clip(ranges / 20.0 * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
        ).reshape(-1, 3)
        for pixel, color in zip(point_pixels[inside][::2], colors[inside][::2]):
            cv2.circle(frame, tuple(pixel), 1, tuple(map(int, color)), -1, cv2.LINE_AA)
        planning_position = np.asarray(plans["planning_position_nwu"][plan_index], dtype=float)
        candidates_world = np.asarray(plans["candidate_positions_nwu"][plan_index], dtype=float)
        candidates_flu = (candidates_world - planning_position[None, None]) @ rotation
        selected = int(plans["selected_index"][plan_index])
        raw_selected = int(plans.get("raw_selected_index", plans["selected_index"])[plan_index])
        for index, candidate in enumerate(candidates_flu):
            pixels = _project_flu(candidate, view_origin, view_scale)
            if index == selected:
                cv2.polylines(frame, [pixels], False, (40, 70, 255), 7, cv2.LINE_AA)
            else:
                cv2.polylines(frame, [pixels], False, (50, 215, 245), 2, cv2.LINE_AA)
        trail_world = actual_positions[: control_index + 1]
        trail_flu = (trail_world - planning_position[None]) @ rotation
        if len(trail_flu) > 1:
            cv2.polylines(frame, [_project_flu(trail_flu, view_origin, view_scale)], False, (70, 230, 120), 4, cv2.LINE_AA)
        cv2.drawMarker(frame, view_origin, (245, 245, 245), cv2.MARKER_TRIANGLE_UP, 32, 3, cv2.LINE_AA)
        goal_flu = ((goal - planning_position) @ rotation)[None]
        goal_pixel = _project_flu(goal_flu, view_origin, view_scale)[0]
        if 0 <= goal_pixel[0] < 1510 and 100 <= goal_pixel[1] < 1050:
            cv2.drawMarker(frame, tuple(goal_pixel), (230, 70, 230), cv2.MARKER_STAR, 28, 3, cv2.LINE_AA)
        # Metric depth is shown honestly, with invalid pixels black.
        depth_vis = np.clip(depth_mm.astype(np.float32) / 20000.0 * 255.0, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        depth_vis[depth_mm == 0] = 0
        depth_vis = cv2.resize(depth_vis, (370, 235), interpolation=cv2.INTER_NEAREST)
        frame[130:365, 1520:1890] = depth_vis
        cv2.rectangle(frame, (1518, 128), (1892, 367), (205, 215, 225), 2)
        _draw_text(frame, "METRIC DEPTH (0-20 m)", (1530, 397), 0.55)
        speed = float(np.linalg.norm(actual_velocities[control_index]))
        latency = float(plans["planner_latency_ms"][plan_index])
        progress = float(np.clip(((actual_positions[control_index] - start) @ route_forward) / max(np.linalg.norm(route), 1e-9), 0, 1))
        cv2.rectangle(frame, (1518, 455), (1892, 825), (9, 14, 19), -1)
        _draw_text(frame, f"TIME       {elapsed:6.2f} s", (1542, 505), 0.66)
        _draw_text(frame, f"SPEED      {speed:6.2f} m/s", (1542, 555), 0.66, (75, 225, 140))
        _draw_text(frame, f"TARGET     {speed_target:6.2f} m/s", (1542, 605), 0.66)
        model_latency = float(plans.get("model_latency_ms", np.zeros_like(plans["planner_latency_ms"]))[plan_index])
        _draw_text(frame, f"TOTAL PRED {latency:6.1f} ms", (1542, 655), 0.66)
        _draw_text(frame, f"SELECTED   {selected:6d}", (1542, 705), 0.66, (60, 95, 255))
        _draw_text(frame, f"PROGRESS   {progress * 100:6.1f} %", (1542, 755), 0.66)
        if "collision_probability" in plans:
            collision_risk = float(plans["collision_probability"][plan_index, selected])
            _draw_text(frame, f"RISK {collision_risk:.3f} | RAW {raw_selected} | WM {model_latency:.1f} ms", (1542, 805), 0.42, (175, 190, 202))
        else:
            _draw_text(frame, "coordinate: local FLU / world NWU", (1542, 805), 0.46, (175, 190, 202))
        cv2.line(frame, (1540, 900), (1870, 900), (65, 75, 85), 14, cv2.LINE_AA)
        cv2.line(frame, (1540, 900), (1540 + int(330 * progress), 900), (65, 225, 125), 14, cv2.LINE_AA)
        _draw_text(frame, "cyan: candidates", (1530, 960), 0.52, (50, 215, 245))
        _draw_text(frame, "red: selected", (1530, 995), 0.52, (60, 95, 255))
        _draw_text(frame, "green: executed", (1530, 1030), 0.52, (70, 230, 120))
        writer.write(frame)
    writer.release()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary), "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)
    temporary.unlink(missing_ok=True)
    return {"output": str(output), "frames": frame_total, "fps": OUTPUT_FPS, "duration_s": duration_s,
            "data_source": str(telemetry_path), "candidate_count": 15, "synthetic_planning_data": False}


def compose(front: Path, chase: Path, rviz: Path, output: Path, duration_s: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    graph = (
        "[0:v]scale=1280:720[f];[1:v]scale=1280:360[c];[2:v]scale=640:1080[r];"
        "[f][c]vstack=inputs=2[left];[left][r]hstack=inputs=2,format=yuv420p[out]"
    )
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-i", str(front), "-i", str(chase), "-i", str(rviz),
        "-filter_complex", graph, "-map", "[out]", "-t", f"{duration_s:.8f}", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)


def render(run_dir: Path) -> dict:
    front_meta = _metadata(run_dir / "front_source.json")
    chase_meta = _metadata(run_dir / "chase_source.json")
    duration = min(_source_duration(front_meta), _source_duration(chase_meta))
    if duration <= 0:
        raise ValueError("camera metadata does not contain a positive synchronized duration")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    speed = float(summary["metrics"]["speed_target_mps"])
    method = str(summary["metrics"].get("method", "yopo")).removeprefix("yopo_")
    front = run_dir / "front_camera_60fps.mp4"
    chase = run_dir / "chase_camera_60fps.mp4"
    rviz = run_dir / "yopo_rviz_60fps.mp4"
    composite = run_dir / "yopo_three_view_60fps.mp4"
    expected_frames = max(1, int(round(duration * OUTPUT_FPS)))
    previous_manifest_path = run_dir / "recording_manifest.json"
    previous_manifest = _metadata(previous_manifest_path) if previous_manifest_path.exists() else {}
    existing_rviz = cv2.VideoCapture(str(rviz))
    existing_rviz_frames = int(existing_rviz.get(cv2.CAP_PROP_FRAME_COUNT)) if existing_rviz.isOpened() else 0
    existing_rviz.release()
    rviz_report = (
        {"output": str(rviz), "frames": expected_frames, "fps": OUTPUT_FPS, "duration_s": duration,
         "data_source": str(run_dir / "telemetry.h5"), "candidate_count": 15,
         "synthetic_planning_data": False, "reused_verified_render": True}
        if existing_rviz_frames == expected_frames and previous_manifest.get("rviz_style_version") == RVIZ_STYLE_VERSION else
        render_rviz(run_dir / "telemetry.h5", rviz, duration, speed, method)
    )
    report = {
        "front": time_correct_camera(run_dir / "front_source.mp4", front_meta, front, duration),
        "chase": time_correct_camera(run_dir / "chase_source.mp4", chase_meta, chase, duration),
        "rviz": rviz_report,
    }
    compose(front, chase, rviz, composite, duration)
    report.update({
        "composite": str(composite), "synchronized_duration_s": duration,
        "rviz_style_version": RVIZ_STYLE_VERSION,
        "synchronization_bound_frames": 1, "output_fps": OUTPUT_FPS,
        "flight_speed_time_scale": 1.0, "videos": [str(front), str(chase), str(rviz), str(composite)],
    })
    (run_dir / "recording_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Render synchronized front, chase and real-data YOPO RViz videos.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
