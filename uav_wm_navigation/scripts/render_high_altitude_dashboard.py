from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.evaluation.video_layout import CANVAS_4K, PANELS_4K, validate_layout


DISPLAY_FPS = 60.0
CANDIDATE_SIZE = (1280, 1440)
DEPTH_PANEL_SIZE = (1200, 720)
TELEMETRY_SIZE = (1360, 720)
STYLE_VERSION = 1


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_text(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (235, 242, 248),
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_DUPLEX, scale, (3, 8, 12), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def _video_info(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return result


def _find_source(run_dir: Path, stem: str, legacy: str) -> tuple[Path, Path]:
    video = run_dir / f"{stem}.mp4"
    metadata = run_dir / f"{stem}.json"
    if video.exists() and metadata.exists():
        return video, metadata
    return run_dir / f"{legacy}.mp4", run_dir / f"{legacy}.json"


def _timestamp_correct(
    source: Path,
    metadata: dict,
    output: Path,
    duration_s: float,
    interpolation: str,
) -> dict:
    info = _video_info(source)
    source_frames = int(info["frames"])
    encoded_duration = max((source_frames - 1) / max(float(info["fps"]), 1e-6), 1e-6)
    simulation_duration = float(metadata["simulation_duration_s"])
    time_scale = simulation_duration / encoded_duration
    frames = max(1, int(round(duration_s * DISPLAY_FPS)))
    temporal = (
        f"minterpolate=fps={DISPLAY_FPS:g}:mi_mode=blend"
        if interpolation == "blend"
        else f"fps=fps={DISPLAY_FPS:g}:round=near"
    )
    graph = (
        f"setpts={time_scale:.10f}*PTS,{temporal},"
        "tpad=stop_mode=clone:stop_duration=0.2,"
        f"trim=end_frame={frames},setpts=N/({DISPLAY_FPS:g}*TB),setsar=1"
    )
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
        "-vf", graph, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p", "-frames:v", str(frames), "-movflags", "+faststart", str(output),
    ], check=True)
    measured = float(metadata.get("measured_simulation_fps", 0.0))
    source_count = int(metadata.get("frame_count", source_frames))
    return {
        "source": str(source),
        "display_working_video": str(output),
        "source_frames": source_count,
        "source_duration_s": simulation_duration,
        "measured_source_fps": measured,
        "display_fps": DISPLAY_FPS,
        "display_frames": frames,
        "time_scale": 1.0,
        "timestamp_correction_factor": time_scale,
        "interpolation": interpolation,
        "display_to_source_frame_ratio": frames / max(source_count, 1),
        "interpolated_or_duplicated_fraction": max(0.0, 1.0 - source_count / max(frames, 1)),
        "dropped_queue_frames": int(metadata.get("dropped_queue_frames", 0)),
        "drop_fraction": int(metadata.get("dropped_queue_frames", 0)) / max(
            source_count + int(metadata.get("dropped_queue_frames", 0)), 1
        ),
    }


def _local_basis(local_goal: np.ndarray, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    forward = np.asarray(local_goal - position, dtype=np.float64)
    forward[2] = 0.0
    if np.linalg.norm(forward) < 0.5:
        forward = np.asarray(velocity, dtype=np.float64)
        forward[2] = 0.0
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    left = np.asarray([-forward[1], forward[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    return np.column_stack([forward, left, up])


def _project(points_flu: np.ndarray, width: int, height: int) -> np.ndarray:
    forward, left, up = points_flu[:, 0], points_flu[:, 1], points_flu[:, 2]
    origin = np.asarray([width * 0.50, height * 0.84])
    scale = min(width / 20.0, height / 23.0)
    x = origin[0] - left * scale + forward * scale * 0.16
    y = origin[1] - up * scale - forward * scale * 0.47
    return np.column_stack([x, y]).astype(np.int32)


def _point_cloud(depth_mm: np.ndarray, intrinsics: np.ndarray, stride: int = 10) -> tuple[np.ndarray, np.ndarray]:
    depth = depth_mm.astype(np.float32) * 0.001
    rows, columns = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    sampled = depth[::stride, ::stride]
    valid = np.isfinite(sampled) & (sampled > 0.2) & (sampled <= 20.0)
    ranges = sampled[valid]
    u, v = columns[valid], rows[valid]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    # Camera RDF to body FLU.
    points = np.column_stack([ranges, -(u - cx) * ranges / fx, -(v - cy) * ranges / fy])
    return points, ranges


def _transcode_mp4v(temporary: Path, output: Path) -> None:
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(temporary),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)
    temporary.unlink(missing_ok=True)


def _load_telemetry(path: Path) -> tuple[dict, dict, np.ndarray]:
    with h5py.File(path, "r") as handle:
        plans = {name: handle[f"plans/{name}"][:] for name in handle["plans"]}
        controls = {name: handle[f"control/{name}"][:] for name in handle["control"]}
        route = np.asarray(handle.attrs.get("route_nwu", [handle.attrs["goal_nwu"]]), dtype=np.float64)
    return plans, controls, route


def render_candidate_panel(telemetry: Path, output: Path, duration_s: float) -> dict:
    plans, controls, route = _load_telemetry(telemetry)
    width, height = CANDIDATE_SIZE
    temporary = output.with_suffix(".mp4v.tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), DISPLAY_FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer {temporary}")
    plan_times = np.asarray(plans["elapsed_s"], dtype=float)
    control_times = np.asarray(controls["monotonic_s"], dtype=float)
    control_times -= control_times[0]
    actual = np.asarray(controls["actual_position_nwu"], dtype=float)
    frames = max(1, int(round(duration_s * DISPLAY_FPS)))
    for frame_index in range(frames):
        elapsed = frame_index / DISPLAY_FPS
        pi = int(np.clip(np.searchsorted(plan_times, elapsed, side="right") - 1, 0, len(plan_times) - 1))
        ci = int(np.clip(np.searchsorted(control_times, elapsed, side="right") - 1, 0, len(control_times) - 1))
        frame = np.full((height, width, 3), (15, 22, 29), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (width, 126), (8, 13, 18), -1)
        _draw_text(frame, "YOPO / LOCAL PLANNING", (34, 52), 1.0, (246, 248, 250), 2)
        _draw_text(frame, "REAL DEPTH  |  15 PRIMITIVES  |  RAW COST ARGMIN", (36, 94), 0.53, (70, 225, 140))
        position = np.asarray(plans["planning_position_nwu"][pi], dtype=float)
        velocity = np.asarray(plans["planning_velocity_nwu"][pi], dtype=float)
        local_goal = np.asarray(plans.get("local_goal_nwu", np.repeat(route[-1][None], len(plan_times), 0))[pi])
        basis = _local_basis(local_goal, position, velocity)
        for forward in (0.0, 5.0, 10.0, 15.0, 20.0):
            line = np.asarray([[forward, -9.0, 0.0], [forward, 9.0, 0.0]])
            cv2.polylines(frame, [_project(line, width, height)], False, (39, 51, 61), 1, cv2.LINE_AA)
        for left in (-8.0, -4.0, 0.0, 4.0, 8.0):
            line = np.asarray([[0.0, left, 0.0], [20.0, left, 0.0]])
            cv2.polylines(frame, [_project(line, width, height)], False, (39, 51, 61), 1, cv2.LINE_AA)
        depth_mm = np.asarray(plans["depth_mm"][pi])
        intrinsics = np.asarray(plans["camera_intrinsics"][pi])
        points, ranges = _point_cloud(depth_mm, intrinsics)
        pixels = _project(points, width, height)
        inside = (
            (pixels[:, 0] >= 5) & (pixels[:, 0] < width - 5)
            & (pixels[:, 1] >= 130) & (pixels[:, 1] < height - 5)
        )
        if ranges.size:
            colors = cv2.applyColorMap(
                np.clip(ranges / 20.0 * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1),
                cv2.COLORMAP_TURBO,
            ).reshape(-1, 3)
            for pixel, color in zip(pixels[inside], colors[inside]):
                cv2.circle(frame, tuple(pixel), 2, tuple(int(value) for value in color), -1, cv2.LINE_AA)
        candidates_world = np.asarray(plans["candidate_positions_nwu"][pi], dtype=float)
        candidates_local = (candidates_world - position[None, None]) @ basis
        selected = int(plans["selected_index"][pi])
        for index, candidate in enumerate(candidates_local):
            color, thickness = ((45, 75, 255), 8) if index == selected else ((45, 215, 245), 2)
            cv2.polylines(frame, [_project(candidate, width, height)], False, color, thickness, cv2.LINE_AA)
        trail = (actual[: ci + 1] - position[None]) @ basis
        if len(trail) > 1:
            cv2.polylines(frame, [_project(trail, width, height)], False, (65, 230, 125), 5, cv2.LINE_AA)
        route_local = (route - position[None]) @ basis
        cv2.polylines(frame, [_project(route_local, width, height)], False, (185, 90, 210), 3, cv2.LINE_AA)
        origin = _project(np.zeros((1, 3)), width, height)[0]
        cv2.drawMarker(frame, tuple(origin), (250, 250, 250), cv2.MARKER_TRIANGLE_UP, 38, 3, cv2.LINE_AA)
        goal_pixel = _project(((local_goal - position) @ basis)[None], width, height)[0]
        if 0 <= goal_pixel[0] < width and 125 <= goal_pixel[1] < height:
            cv2.drawMarker(frame, tuple(goal_pixel), (220, 70, 225), cv2.MARKER_STAR, 34, 3, cv2.LINE_AA)
        _draw_text(frame, f"plan {int(plans['sequence_id'][pi]):04d}  selected {selected:02d}", (34, height - 42), 0.60)
        writer.write(frame)
    writer.release()
    _transcode_mp4v(temporary, output)
    return {"path": str(output), "frames": frames, "size": list(CANDIDATE_SIZE), "source": str(telemetry)}


def render_telemetry_panel(telemetry: Path, output: Path, duration_s: float, summary: dict) -> dict:
    plans, controls, _ = _load_telemetry(telemetry)
    width, height = TELEMETRY_SIZE
    temporary = output.with_suffix(".mp4v.tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), DISPLAY_FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer {temporary}")
    plan_times = np.asarray(plans["elapsed_s"], dtype=float)
    control_times = np.asarray(controls["monotonic_s"], dtype=float)
    control_times -= control_times[0]
    velocity = np.asarray(controls["actual_velocity_nwu"], dtype=float)
    frames = max(1, int(round(duration_s * DISPLAY_FPS)))
    target = float(summary["metrics"]["speed_target_mps"])
    route_length = float(summary["metrics"].get("route_length_m", 1.0))
    for frame_index in range(frames):
        elapsed = frame_index / DISPLAY_FPS
        pi = int(np.clip(np.searchsorted(plan_times, elapsed, side="right") - 1, 0, len(plan_times) - 1))
        ci = int(np.clip(np.searchsorted(control_times, elapsed, side="right") - 1, 0, len(control_times) - 1))
        frame = np.full((height, width, 3), (10, 16, 22), dtype=np.uint8)
        speed = float(np.linalg.norm(velocity[ci]))
        progress_m = float(plans.get("route_progress_s_m", np.zeros(len(plan_times)))[pi])
        progress = float(np.clip(progress_m / max(route_length, 1e-9), 0.0, 1.0))
        values = [
            ("TIME", f"{elapsed:6.2f} s", (235, 242, 248)),
            ("SPEED", f"{speed:6.2f} m/s", (70, 225, 140)),
            ("TARGET", f"{target:6.2f} m/s", (235, 242, 248)),
            ("AGL", f"{float(plans.get('agl_m', np.zeros(len(plan_times)))[pi]):6.2f} m", (235, 242, 248)),
            ("ROUTE", f"{progress_m:6.1f} / {route_length:.1f} m", (235, 242, 248)),
            ("CROSS TRACK", f"{float(plans.get('cross_track_error_m', np.zeros(len(plan_times)))[pi]):6.2f} m", (235, 242, 248)),
            ("PLAN LATENCY", f"{float(plans['planner_latency_ms'][pi]):6.1f} ms", (235, 242, 248)),
            ("CANDIDATE", f"{int(plans['selected_index'][pi]):6d} / 15", (45, 75, 255)),
        ]
        for row, (label, value, color) in enumerate(values):
            y = 64 + row * 66
            _draw_text(frame, label, (42, y), 0.56, (155, 173, 188))
            _draw_text(frame, value, (480, y), 0.74, color, 2)
        x0, x1, y = 42, width - 44, height - 82
        cv2.line(frame, (x0, y), (x1, y), (55, 68, 80), 24, cv2.LINE_AA)
        cv2.line(frame, (x0, y), (x0 + int((x1 - x0) * progress), y), (65, 225, 125), 24, cv2.LINE_AA)
        _draw_text(frame, f"PROGRESS {progress * 100:5.1f}%", (42, height - 28), 0.52, (205, 218, 228))
        writer.write(frame)
    writer.release()
    _transcode_mp4v(temporary, output)
    return {"path": str(output), "frames": frames, "size": list(TELEMETRY_SIZE), "source": str(telemetry)}


def render_depth_panel(corrected_depth: Path, metadata: dict, output: Path, duration_s: float) -> dict:
    capture = cv2.VideoCapture(str(corrected_depth))
    width, height = DEPTH_PANEL_SIZE
    temporary = output.with_suffix(".mp4v.tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), DISPLAY_FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer {temporary}")
    source_stats = list(metadata.get("frames", []))
    frame_total = max(1, int(round(duration_s * DISPLAY_FPS)))
    for frame_index in range(frame_total):
        ok, source = capture.read()
        if not ok:
            source = np.zeros((384, 640, 3), dtype=np.uint8)
        frame = cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 82), (5, 9, 13), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0.0, frame)
        source_index = min(int(frame_index / DISPLAY_FPS * max(float(metadata.get("measured_simulation_fps", 1.0)), 1.0)), len(source_stats) - 1)
        stats = source_stats[source_index] if source_stats else {}
        _draw_text(frame, "METRIC DEPTH / LOG SCALE / 1-80 m", (24, 36), 0.65, (246, 248, 250), 2)
        _draw_text(
            frame,
            f"valid {float(stats.get('valid_fraction', 0.0)) * 100:5.1f}%   "
            f"near {float(stats.get('minimum_depth_m', 0.0)):5.2f} m   "
            f"median {float(stats.get('median_depth_m', 0.0)):5.1f} m",
            (24, 69),
            0.46,
            (215, 228, 238),
        )
        legend_x0, legend_x1, legend_y0, legend_y1 = width - 58, width - 32, 106, height - 28
        gradient = np.linspace(255, 0, legend_y1 - legend_y0, dtype=np.uint8)[:, None]
        legend = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        frame[legend_y0:legend_y1, legend_x0:legend_x1] = legend
        for tick in (1, 5, 10, 20, 40, 80):
            ratio = (math.log1p(tick) - math.log1p(1.0)) / (math.log1p(80.0) - math.log1p(1.0))
            y = int(legend_y1 - ratio * (legend_y1 - legend_y0))
            cv2.line(frame, (legend_x0 - 6, y), (legend_x1 + 2, y), (245, 245, 245), 1)
            _draw_text(frame, f"{tick}", (legend_x0 - 54, y + 5), 0.35, (245, 245, 245))
        roi_x0, roi_x1 = int(width * 0.28), int(width * 0.72)
        roi_y0, roi_y1 = int(height * 0.25), int(height * 0.82)
        cv2.rectangle(frame, (roi_x0, roi_y0), (roi_x1, roi_y1), (245, 245, 245), 2, cv2.LINE_AA)
        _draw_text(frame, "SafetyFilter ROI", (roi_x0 + 8, roi_y0 + 28), 0.40, (245, 245, 245))
        writer.write(frame)
    capture.release()
    writer.release()
    _transcode_mp4v(temporary, output)
    return {"path": str(output), "frames": frame_total, "size": list(DEPTH_PANEL_SIZE)}


def compose_dashboard(
    front: Path,
    candidates: Path,
    chase: Path,
    depth: Path,
    telemetry: Path,
    output_4k: Path,
    output_1080: Path,
    frames: int,
) -> None:
    validate_layout(CANVAS_4K, PANELS_4K)
    graph = (
        "[0:v]scale=2560:1440:flags=lanczos,setsar=1[front];"
        "[1:v]scale=1280:1440:flags=lanczos,setsar=1[candidates];"
        "[2:v]scale=1280:720:flags=lanczos,setsar=1[chase];"
        "[3:v]scale=1200:720:flags=neighbor,setsar=1[depth];"
        "[4:v]scale=1360:720:flags=lanczos,setsar=1[telemetry];"
        "[front][candidates]hstack=inputs=2[top];"
        "[chase][depth][telemetry]hstack=inputs=3[bottom];"
        "[top][bottom]vstack=inputs=2,format=yuv420p[out]"
    )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(front), "-i", str(candidates), "-i", str(chase), "-i", str(depth), "-i", str(telemetry),
        "-filter_complex", graph, "-map", "[out]", "-an", "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_4k),
    ], check=True)
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-i", str(output_4k),
        "-vf", "scale=1920:1080:flags=lanczos,setsar=1", "-an", "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_1080),
    ], check=True)


def render(run_dir: Path, reuse_existing: bool = False) -> dict:
    front_source, front_meta_path = _find_source(run_dir, "front_camera_source", "front_source")
    chase_source, chase_meta_path = _find_source(run_dir, "chase_camera_source", "chase_source")
    depth_source, depth_meta_path = _find_source(run_dir, "metric_depth_source", "metric_depth_source")
    for path in (front_source, front_meta_path, chase_source, chase_meta_path, depth_source, depth_meta_path):
        if not path.exists():
            raise FileNotFoundError(path)
    front_meta, chase_meta, depth_meta = map(_read_json, (front_meta_path, chase_meta_path, depth_meta_path))
    duration = min(
        float(front_meta["simulation_duration_s"]),
        float(chase_meta["simulation_duration_s"]),
        float(depth_meta["simulation_duration_s"]),
    )
    if duration <= 0.0:
        raise ValueError("source streams do not have a positive synchronized duration")
    summary = _read_json(run_dir / "summary.json")
    working = run_dir / ".dashboard_work"
    working.mkdir(exist_ok=True)
    front_display = working / "front_60.mp4"
    chase_display = working / "chase_60.mp4"
    depth_display = working / "depth_60.mp4"
    frames = max(1, int(round(duration * DISPLAY_FPS)))
    stream_specs = (
        ("front", front_source, front_meta, front_display, "blend"),
        ("chase", chase_source, chase_meta, chase_display, "blend"),
        ("depth", depth_source, depth_meta, depth_display, "duplicate"),
    )
    streams = {}
    for name, source, metadata, display, interpolation in stream_specs:
        if reuse_existing and display.exists() and int(_video_info(display)["frames"]) == frames:
            streams[name] = {
                "source": str(source),
                "display_working_video": str(display),
                "source_frames": int(metadata.get("frame_count", _video_info(source)["frames"])),
                "source_duration_s": float(metadata["simulation_duration_s"]),
                "measured_source_fps": float(metadata.get("measured_simulation_fps", 0.0)),
                "display_fps": DISPLAY_FPS,
                "display_frames": frames,
                "time_scale": 1.0,
                "interpolation": interpolation,
                "reused": True,
                "dropped_queue_frames": int(metadata.get("dropped_queue_frames", 0)),
            }
        else:
            streams[name] = _timestamp_correct(source, metadata, display, duration, interpolation)
    candidates = run_dir / "yopo_rviz_native.mp4"
    telemetry = working / "telemetry_native.mp4"
    depth_panel = working / "depth_panel.mp4"
    candidate_report = (
        {"path": str(candidates), "frames": frames, "size": list(CANDIDATE_SIZE),
         "source": str(run_dir / "telemetry.h5"), "reused": True}
        if reuse_existing and candidates.exists() and int(_video_info(candidates)["frames"]) == frames
        else render_candidate_panel(run_dir / "telemetry.h5", candidates, duration)
    )
    telemetry_report = (
        {"path": str(telemetry), "frames": frames, "size": list(TELEMETRY_SIZE),
         "source": str(run_dir / "telemetry.h5"), "reused": True}
        if reuse_existing and telemetry.exists() and int(_video_info(telemetry)["frames"]) == frames
        else render_telemetry_panel(run_dir / "telemetry.h5", telemetry, duration, summary)
    )
    depth_report = render_depth_panel(depth_display, depth_meta, depth_panel, duration)
    output_4k = run_dir / "yopo_high_altitude_dashboard_4k_60fps.mp4"
    output_1080 = run_dir / "yopo_high_altitude_dashboard_1080p_60fps.mp4"
    compose_dashboard(
        front_display, candidates, chase_display, depth_panel, telemetry, output_4k, output_1080, frames
    )
    outputs = {
        "dashboard_4k": {**_video_info(output_4k), "path": str(output_4k)},
        "dashboard_1080p": {**_video_info(output_1080), "path": str(output_1080)},
        "candidate_view": candidate_report,
        "telemetry_panel": telemetry_report,
        "depth_panel": depth_report,
    }
    report = {
        "schema": "uav-wm-nav-high-altitude-dashboard-v1",
        "style_version": STYLE_VERSION,
        "source_streams": streams,
        "synchronized_duration_s": duration,
        "display_fps": DISPLAY_FPS,
        "display_frames": frames,
        "time_scale": 1.0,
        "panel_layout_4k": [
            {"name": panel.name, "x": panel.x, "y": panel.y, "width": panel.width, "height": panel.height}
            for panel in PANELS_4K
        ],
        "depth_display": {
            "source_resolution": list(_video_info(depth_source).values())[:2],
            "metric_range_m": [1, 80],
            "mapping": "fixed logarithmic Turbo; invalid black",
            "scaling_interpolation": "nearest",
            "panel_aspect_error_fraction": abs(DEPTH_PANEL_SIZE[0] / DEPTH_PANEL_SIZE[1] - 5 / 3) / (5 / 3),
        },
        "outputs": outputs,
    }
    (run_dir / "recording_manifest_high_altitude.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the native 4K/1080p high-altitude YOPO dashboard.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    render(args.run_dir.resolve(), reuse_existing=args.reuse_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
