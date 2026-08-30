from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import generate_midterm_figures as figmod
import render_priority_figures as global_fig
import run_midterm_benchmark as bench


FIG_DIR = ROOT / "thesis" / "figures"
MP4_PATH = FIG_DIR / "fig_2_11_global_assignment_order.mp4"
GIF_PATH = FIG_DIR / "fig_2_11_global_assignment_order.gif"

FPS_MP4 = 20
FPS_GIF = 10
INTRO_SECONDS = 0.8
ASSIGN_STEP_SECONDS = 0.30
EXEC_SECONDS = 7.6
OUTRO_SECONDS = 1.2
MP4_FIGSIZE = (9.6, 7.4)
GIF_FIGSIZE = (7.0, 5.4)
MP4_DPI = 120
GIF_DPI = 96


def stable_bend_sign(drone_id: str, task_id: str) -> float:
    token = f"{drone_id}:{task_id}".encode("utf-8")
    digest = hashlib.blake2s(token, digest_size=1).digest()[0]
    return 1.0 if digest % 2 == 0 else -1.0


def task_sort_key(task) -> tuple[int, float]:
    return int(task.priority), -float(task.reward)


def path_cumulative(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.array([0.0], dtype=float)
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def clip_polyline(points: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    if len(points) == 0:
        return points
    if distance <= 0.0:
        return points[:1].copy()
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    if distance >= total:
        return points.copy()

    idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
    idx = int(np.clip(idx, 0, len(points) - 2))
    d0 = float(cumulative[idx])
    d1 = float(cumulative[idx + 1])
    ratio = 0.0 if d1 - d0 <= 1e-9 else (distance - d0) / (d1 - d0)
    interp = points[idx] + (points[idx + 1] - points[idx]) * ratio
    return np.vstack([points[: idx + 1], interp])


def assign_tasks_with_sequence(drones, tasks):
    assignments = {drone.id: [] for drone in drones}
    loads = Counter()
    positions = {drone.id: np.array(drone.position, dtype=float) for drone in drones}
    events = []
    skipped = []

    for task in sorted(tasks, key=task_sort_key):
        feasible = [drone for drone in drones if drone.max_payload + 1e-6 >= task.payload_weight]
        if not feasible:
            skipped.append(task.id)
            continue

        best_drone = None
        best_key = None
        for drone in feasible:
            if loads[drone.id] >= global_fig.DISPLAY_MAX_TASKS_PER_DRONE:
                continue

            start = positions[drone.id]
            d1 = float(np.linalg.norm(task.pickup_pos[[0, 2]] - start[[0, 2]]))
            d2 = float(np.linalg.norm(task.delivery_pos[[0, 2]] - task.pickup_pos[[0, 2]]))
            load_penalty = 95.0 * loads[drone.id]
            type_penalty = 0.0
            if task.preferred_drone_types and drone.drone_type not in task.preferred_drone_types:
                type_penalty += 120.0
            if task.cold_chain and drone.drone_type == "heavy":
                type_penalty += 45.0
            if task.fragile and drone.drone_type == "heavy":
                type_penalty += 35.0

            key = (loads[drone.id], d1 + 0.85 * d2 + type_penalty + load_penalty, drone.id)
            if best_key is None or key < best_key:
                best_key = key
                best_drone = drone

        if best_drone is None:
            skipped.append(task.id)
            continue

        assignments[best_drone.id].append(task.id)
        loads[best_drone.id] += 1
        positions[best_drone.id] = np.array(task.delivery_pos, dtype=float)
        events.append(
            {
                "sequence": len(events) + 1,
                "task_id": task.id,
                "drone_id": best_drone.id,
                "drone_type": best_drone.drone_type,
                "task_type": task.task_type,
                "priority": int(task.priority),
                "reward": float(task.reward),
            }
        )

    return assignments, events, skipped


def build_stable_routes(drones, tasks, assignments):
    task_map = {task.id: task for task in tasks}
    routes = []
    for drone in drones:
        bundle = assignments.get(drone.id, [])
        current = np.array(drone.position, dtype=float)
        for task_id in bundle:
            task = task_map[task_id]
            cruise_alt = global_fig.layer_altitude(task)
            bend_sign = stable_bend_sign(drone.id, task.id)
            pickup_service = global_fig.clamp_service_point(task.pickup_pos, low=4.0, high=10.0)
            delivery_service = global_fig.clamp_service_point(task.delivery_pos, low=3.5, high=6.5)

            leg1 = global_fig.smooth_leg(current, pickup_service, cruise_alt * 0.92, bend_sign, bend_scale=0.82)
            leg2 = global_fig.smooth_leg(pickup_service, delivery_service, cruise_alt, -bend_sign, bend_scale=1.0)
            points = np.vstack([leg1, leg2[1:]])

            speed = max(float(drone.cruise_speed), 0.1)
            leg1_cum = path_cumulative(leg1)
            leg2_cum = path_cumulative(leg2)
            leg1_length = float(leg1_cum[-1])
            leg2_length = float(leg2_cum[-1])
            leg1_time = leg1_length / speed
            leg2_time = leg2_length / speed
            wait_before_pickup = max(0.0, float(task.time_window[0]) - bench.CURRENT_TIME - leg1_time)
            pickup_service_time = float(task.pickup_service_time)
            delivery_service_time = float(task.delivery_service_time)
            second_leg_start = leg1_time + wait_before_pickup + pickup_service_time
            complete_time = second_leg_start + leg2_time + delivery_service_time

            routes.append(
                {
                    "drone_id": drone.id,
                    "drone_type": drone.drone_type,
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "priority": int(task.priority),
                    "reward": float(task.reward),
                    "delivery_district": getattr(task, "delivery_district", ""),
                    "pickup_point": pickup_service,
                    "delivery_point": delivery_service,
                    "points": points,
                    "raw_points": np.array([current, pickup_service, delivery_service], dtype=float),
                    "leg1_points": leg1,
                    "leg2_points": leg2,
                    "leg1_cum": leg1_cum,
                    "leg2_cum": leg2_cum,
                    "speed": speed,
                    "leg1_time": leg1_time,
                    "leg2_time": leg2_time,
                    "wait_before_pickup": wait_before_pickup,
                    "pickup_service_time": pickup_service_time,
                    "delivery_service_time": delivery_service_time,
                    "second_leg_start": second_leg_start,
                    "complete_time": complete_time,
                }
            )
            current = delivery_service.copy()
    return routes


def build_animation_payload():
    city = global_fig.load_city()
    tasks = global_fig.select_balanced_tasks(city)
    drones = bench.build_drones()
    assignments, events, skipped = assign_tasks_with_sequence(drones, tasks)
    routes = build_stable_routes(drones, tasks, assignments)
    summary = global_fig.summarize_routes(routes, assignments, drones)
    route_by_task = {route["task_id"]: route for route in routes}

    ordered_routes = []
    for event in events:
        route = route_by_task.get(event["task_id"])
        if route is None:
            continue
        merged = dict(route)
        merged.update(event)
        ordered_routes.append(merged)

    max_exec_time = max((float(route["complete_time"]) for route in ordered_routes), default=1.0)
    return city, ordered_routes, summary, len(tasks), len(skipped), max_exec_time


def format_task_type(task_type: str) -> str:
    return task_type.replace("_", " ")


def build_scene(city, routes, figsize, dpi):
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#f7efe7")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    global_fig.draw_global_oblique(ax, city, routes)

    plan_lines = []
    trail_lines = []
    start_markers = []
    delivery_markers = []
    head_markers = []
    for route in routes:
        style = figmod.DRONE_STYLE[route["drone_type"]]
        (plan_line,) = ax.plot([], [], [], color=style["color"], linewidth=1.5, alpha=0.0, zorder=6)
        (trail_line,) = ax.plot([], [], [], color=style["color"], linewidth=2.9, alpha=0.0, zorder=8)
        (start_marker,) = ax.plot(
            [],
            [],
            [],
            linestyle="",
            marker=style["marker"],
            markersize=8.0,
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=9,
        )
        (delivery_marker,) = ax.plot(
            [],
            [],
            [],
            linestyle="",
            marker="s",
            markersize=5.8,
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=8,
        )
        (head_marker,) = ax.plot(
            [],
            [],
            [],
            linestyle="",
            marker="o",
            markersize=6.0,
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=10,
        )
        plan_lines.append(plan_line)
        trail_lines.append(trail_line)
        start_markers.append(start_marker)
        delivery_markers.append(delivery_marker)
        head_markers.append(head_marker)

    (highlight_halo,) = ax.plot([], [], [], color="#ffffff", linewidth=5.4, alpha=0.0, zorder=9)
    (highlight_line,) = ax.plot([], [], [], color="#2e86de", linewidth=3.4, alpha=0.0, zorder=10)

    overlay = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    overlay.set_axis_off()
    overlay.set_xlim(0.0, 1.0)
    overlay.set_ylim(0.0, 1.0)

    panel = patches.FancyBboxPatch(
        (0.030, 0.872),
        0.660,
        0.108,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        facecolor=(1.0, 1.0, 1.0, 0.84),
        edgecolor=(1.0, 1.0, 1.0, 0.0),
    )
    overlay.add_patch(panel)
    title_text = overlay.text(
        0.050,
        0.946,
        "Global Assignment Timeline",
        fontsize=17,
        fontweight="bold",
        color="#1f2937",
        ha="left",
        va="center",
    )
    phase_text = overlay.text(
        0.050,
        0.913,
        "",
        fontsize=12.1,
        fontweight="bold",
        color="#2e86de",
        ha="left",
        va="center",
    )
    detail_text = overlay.text(
        0.050,
        0.886,
        "",
        fontsize=10.9,
        color="#425466",
        ha="left",
        va="center",
    )
    footer_text = overlay.text(
        0.032,
        0.060,
        "",
        fontsize=10.2,
        color="#4b5563",
        ha="left",
        va="center",
    )
    progress_text = overlay.text(
        0.968,
        0.060,
        "",
        fontsize=10.5,
        color="#1f2937",
        ha="right",
        va="center",
        fontweight="bold",
    )
    progress_bg = patches.FancyBboxPatch(
        (0.032, 0.028),
        0.936,
        0.015,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        facecolor="#ddd2c3",
        edgecolor="none",
        alpha=0.95,
    )
    progress_fg = patches.FancyBboxPatch(
        (0.032, 0.028),
        0.001,
        0.015,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        facecolor="#2e86de",
        edgecolor="none",
        alpha=0.98,
    )
    overlay.add_patch(progress_bg)
    overlay.add_patch(progress_fg)

    return {
        "fig": fig,
        "plan_lines": plan_lines,
        "trail_lines": trail_lines,
        "start_markers": start_markers,
        "delivery_markers": delivery_markers,
        "head_markers": head_markers,
        "highlight_halo": highlight_halo,
        "highlight_line": highlight_line,
        "phase_text": phase_text,
        "detail_text": detail_text,
        "footer_text": footer_text,
        "progress_text": progress_text,
        "progress_fg": progress_fg,
        "progress_w": 0.936,
    }


def set_line_points(artist, points: np.ndarray):
    if points is None or len(points) == 0:
        artist.set_data_3d([], [], [])
        return
    artist.set_data_3d(points[:, 0], points[:, 2], points[:, 1])


def set_point(artist, point: np.ndarray | None):
    if point is None:
        artist.set_data_3d([], [], [])
        return
    arr = np.asarray(point, dtype=float).reshape(1, 3)
    artist.set_data_3d(arr[:, 0], arr[:, 2], arr[:, 1])


def route_state_at_time(route, sim_time: float):
    if sim_time <= 0.0:
        return route["points"][:1].copy(), route["points"][0], "queued"

    if sim_time < route["leg1_time"]:
        partial = clip_polyline(route["leg1_points"], route["leg1_cum"], sim_time * route["speed"])
        return partial, partial[-1], "to_pickup"

    if sim_time < route["leg1_time"] + route["wait_before_pickup"]:
        return route["leg1_points"].copy(), route["pickup_point"], "pickup_wait"

    if sim_time < route["leg1_time"] + route["wait_before_pickup"] + route["pickup_service_time"]:
        return route["leg1_points"].copy(), route["pickup_point"], "pickup_service"

    if sim_time < route["second_leg_start"] + route["leg2_time"]:
        second_leg_elapsed = sim_time - route["second_leg_start"]
        partial_leg2 = clip_polyline(route["leg2_points"], route["leg2_cum"], second_leg_elapsed * route["speed"])
        trail = np.vstack([route["leg1_points"], partial_leg2[1:]]) if len(partial_leg2) > 1 else route["leg1_points"].copy()
        return trail, trail[-1], "to_delivery"

    if sim_time < route["complete_time"]:
        return route["points"].copy(), route["delivery_point"], "delivery_service"

    return route["points"].copy(), None, "completed"


def render_frame(scene, routes, total_selected: int, skipped_count: int, max_exec_time: float, frame_idx: int, total_frames: int, fps: int):
    n_routes = len(routes)
    intro_frames = max(1, round(INTRO_SECONDS * fps))
    assign_step_frames = max(2, round(ASSIGN_STEP_SECONDS * fps))
    assign_frames = assign_step_frames * n_routes
    exec_frames = max(2, round(EXEC_SECONDS * fps))

    overall_progress = frame_idx / max(1, total_frames - 1)
    scene["progress_fg"].set_width(max(0.001, scene["progress_w"] * overall_progress))

    for idx, route in enumerate(routes):
        plan_line = scene["plan_lines"][idx]
        trail_line = scene["trail_lines"][idx]
        start_marker = scene["start_markers"][idx]
        delivery_marker = scene["delivery_markers"][idx]
        head_marker = scene["head_markers"][idx]

        set_line_points(plan_line, np.empty((0, 3), dtype=float))
        plan_line.set_alpha(0.0)
        set_line_points(trail_line, np.empty((0, 3), dtype=float))
        trail_line.set_alpha(0.0)
        set_point(start_marker, None)
        set_point(delivery_marker, None)
        set_point(head_marker, None)

    set_line_points(scene["highlight_halo"], np.empty((0, 3), dtype=float))
    scene["highlight_halo"].set_alpha(0.0)
    set_line_points(scene["highlight_line"], np.empty((0, 3), dtype=float))
    scene["highlight_line"].set_alpha(0.0)

    if frame_idx < intro_frames:
        scene["phase_text"].set_text("Phase 1/2  Assignment Order")
        scene["detail_text"].set_text("Preparing the global allocation view...")
        scene["footer_text"].set_text("This phase shows assignment order only. Concurrent flight starts in phase 2.")
        scene["progress_text"].set_text("warmup")
    elif frame_idx < intro_frames + assign_frames:
        active_idx = frame_idx - intro_frames
        current_idx = active_idx // assign_step_frames
        local_idx = active_idx % assign_step_frames
        completed = current_idx

        for idx, route in enumerate(routes):
            if idx > current_idx:
                continue
            plan_line = scene["plan_lines"][idx]
            set_line_points(plan_line, route["points"])
            plan_line.set_alpha(0.22 if idx < current_idx else 0.16)
            set_point(scene["start_markers"][idx], route["points"][0])
            set_point(scene["delivery_markers"][idx], route["delivery_point"])

        current_route = routes[min(current_idx, n_routes - 1)]
        pulse = 0.70 + 0.30 * np.sin(np.pi * (local_idx / max(1, assign_step_frames - 1)))
        set_line_points(scene["highlight_halo"], current_route["points"])
        scene["highlight_halo"].set_alpha(0.88 * pulse)
        set_line_points(scene["highlight_line"], current_route["points"])
        scene["highlight_line"].set_color(figmod.DRONE_STYLE[current_route["drone_type"]]["color"])
        scene["highlight_line"].set_alpha(0.96)

        scene["phase_text"].set_text("Phase 1/2  Assignment Order")
        scene["detail_text"].set_text(
            f"Step {current_route['sequence']:02d}/{n_routes:02d}  |  "
            f"{current_route['drone_id']} <- {current_route['task_id']}  |  "
            f"{format_task_type(current_route['task_type'])}  |  P{current_route['priority']}"
        )
        scene["footer_text"].set_text("This phase shows assignment order only. Concurrent flight starts in phase 2.")
        scene["progress_text"].set_text(f"assign {min(n_routes, completed + 1)}/{n_routes}")
    elif frame_idx < intro_frames + assign_frames + exec_frames:
        exec_idx = frame_idx - intro_frames - assign_frames
        exec_fraction = exec_idx / max(1, exec_frames - 1)
        sim_time = float(exec_fraction * max_exec_time)

        counts = Counter()
        for idx, route in enumerate(routes):
            plan_line = scene["plan_lines"][idx]
            trail_line = scene["trail_lines"][idx]
            set_line_points(plan_line, route["points"])
            plan_line.set_alpha(0.18)
            set_point(scene["start_markers"][idx], route["points"][0])
            set_point(scene["delivery_markers"][idx], route["delivery_point"])

            trail, head, status = route_state_at_time(route, sim_time)
            counts[status] += 1

            set_line_points(trail_line, trail)
            if status == "completed":
                trail_line.set_alpha(0.72)
            else:
                trail_line.set_alpha(0.96)
            set_point(scene["head_markers"][idx], head)

        airborne = counts["to_pickup"] + counts["to_delivery"]
        servicing = counts["pickup_wait"] + counts["pickup_service"] + counts["delivery_service"]
        completed = counts["completed"]
        scene["phase_text"].set_text("Phase 2/2  Concurrent Execution")
        scene["detail_text"].set_text(
            f"T+{sim_time:4.0f} s  |  airborne {airborne:02d}  |  service {servicing:02d}  |  done {completed:02d}"
        )
        scene["footer_text"].set_text("All assigned UAVs advance on the same global clock from CURRENT_TIME.")
        scene["progress_text"].set_text(f"exec T+{sim_time:.0f}s")
    else:
        for idx, route in enumerate(routes):
            plan_line = scene["plan_lines"][idx]
            trail_line = scene["trail_lines"][idx]
            set_line_points(plan_line, route["points"])
            plan_line.set_alpha(0.18)
            set_line_points(trail_line, route["points"])
            trail_line.set_alpha(0.76)
            set_point(scene["start_markers"][idx], route["points"][0])
            set_point(scene["delivery_markers"][idx], route["delivery_point"])
            set_point(scene["head_markers"][idx], None)

        scene["phase_text"].set_text("Phase 2/2  Concurrent Execution")
        scene["detail_text"].set_text(
            f"Final  |  {n_routes}/{total_selected} sampled tasks assigned  |  {n_routes} routes finished together"
        )
        scene["footer_text"].set_text("Phase 1 conveyed assignment order; phase 2 showed simultaneous UAV execution.")
        scene["progress_text"].set_text(f"done  {n_routes}/{n_routes}")

    scene["fig"].canvas.draw()
    buf = np.asarray(scene["fig"].canvas.buffer_rgba(), dtype=np.uint8)
    return buf[:, :, :3].copy()


def render_animation(output_path: Path, city, routes, total_selected: int, skipped_count: int, max_exec_time: float, fps: int, figsize, dpi):
    intro_frames = max(1, round(INTRO_SECONDS * fps))
    assign_frames = max(2, round(ASSIGN_STEP_SECONDS * fps)) * len(routes)
    exec_frames = max(2, round(EXEC_SECONDS * fps))
    outro_frames = max(1, round(OUTRO_SECONDS * fps))
    total_frames = intro_frames + assign_frames + exec_frames + outro_frames

    scene = build_scene(city, routes, figsize=figsize, dpi=dpi)
    writer_kwargs = {"fps": fps}
    if output_path.suffix.lower() == ".mp4":
        writer_kwargs.update({"codec": "libx264", "quality": 8, "macro_block_size": 1})
    elif output_path.suffix.lower() == ".gif":
        writer_kwargs.update({"loop": 0})

    writer = imageio.get_writer(output_path, **writer_kwargs)
    try:
        for frame_idx in range(total_frames):
            frame = render_frame(
                scene,
                routes=routes,
                total_selected=total_selected,
                skipped_count=skipped_count,
                max_exec_time=max_exec_time,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=fps,
            )
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(scene["fig"])

    return total_frames / float(fps)


def main():
    figmod.setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    city, routes, summary, total_selected, skipped_count, max_exec_time = build_animation_payload()
    mp4_duration = render_animation(
        MP4_PATH,
        city=city,
        routes=routes,
        total_selected=total_selected,
        skipped_count=skipped_count,
        max_exec_time=max_exec_time,
        fps=FPS_MP4,
        figsize=MP4_FIGSIZE,
        dpi=MP4_DPI,
    )
    gif_duration = render_animation(
        GIF_PATH,
        city=city,
        routes=routes,
        total_selected=total_selected,
        skipped_count=skipped_count,
        max_exec_time=max_exec_time,
        fps=FPS_GIF,
        figsize=GIF_FIGSIZE,
        dpi=GIF_DPI,
    )

    print(f"sampled routes: {len(routes)} / {total_selected}")
    print(f"active drones: {summary['active_drones']}")
    print(f"skipped tasks: {skipped_count}")
    print(f"simulated concurrent horizon: {max_exec_time:.1f}s")
    print(f"mp4 duration: {mp4_duration:.2f}s -> {MP4_PATH}")
    print(f"gif duration: {gif_duration:.2f}s -> {GIF_PATH}")


if __name__ == "__main__":
    main()
