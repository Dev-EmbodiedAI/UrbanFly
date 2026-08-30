from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from scipy.interpolate import splprep, splev
except Exception:  # pragma: no cover
    splprep = None
    splev = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_midterm_figures as figmod
import run_midterm_benchmark as bench


FIG_DIR = ROOT / "thesis" / "figures"
CITY_PATH = ROOT / "data" / "scene_simworld_dense" / "city_layout.json"
TASK_POOL_SIZE = 180
TASKS_PER_DISTRICT = 4
GLOBAL_ROUTE_LIMIT = 24
LOCAL_ROUTE_LIMIT = 4
DISPLAY_MAX_TASKS_PER_DRONE = 1


def load_city():
    return json.loads(CITY_PATH.read_text(encoding="utf-8"))


def draw_city_oblique_district(ax, city):
    core = [b for b in city["buildings"] if abs(b["x"]) <= 250 and abs(b["z"]) <= 230]
    buildings = sorted(core or city["buildings"], key=lambda b: (b["x"] + b["z"], b["h"]))
    x_min, x_max, z_min, z_max = figmod.city_bounds(buildings)
    max_h = max(b["top_y"] for b in buildings)

    for road in city.get("roads", []):
        if road["category"] not in {"main_road", "secondary_road"}:
            continue
        if not (x_min - 40 <= road["x"] <= x_max + 40 and z_min - 40 <= road["z"] <= z_max + 40):
            continue
        if abs(road["yaw"]) < 45 or abs(road["yaw"]) > 135:
            w = road["length"]
            d = road["width"]
        else:
            w = road["width"]
            d = road["length"]
        x0, x1 = road["x"] - w * 0.5, road["x"] + w * 0.5
        y0, y1 = road["z"] - d * 0.5, road["z"] + d * 0.5
        road_face = [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]]
        ax.add_collection3d(Poly3DCollection([road_face], facecolors="#d7dce6", edgecolors="none", alpha=1.0))

    for b in buildings:
        base_color = figmod.lighten(figmod.district_color(b["district"]), 0.08)
        face_colors = [
            figmod.lighten(base_color, 0.18),
            figmod.darken(base_color, 0.10),
            figmod.darken(base_color, 0.18),
            figmod.darken(base_color, 0.14),
            figmod.darken(base_color, 0.24),
        ]
        poly = Poly3DCollection(
            figmod.cuboid_faces(b),
            facecolors=face_colors,
            edgecolors="#f7f4ed",
            linewidths=0.15,
            alpha=0.98,
        )
        ax.add_collection3d(poly)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(0.0, max_h * 1.05)
    ax.view_init(elev=26, azim=-54)
    ax.set_proj_type("persp")
    ax.set_box_aspect((x_max - x_min, z_max - z_min, max_h * 1.08))
    ax.set_facecolor("#f7efe7")
    ax.set_axis_off()


def select_balanced_tasks(city):
    layout = dict(city)
    layout["source_path"] = str(CITY_PATH.relative_to(ROOT))
    hotspots = bench.make_hotspots(layout)
    pool = bench.build_tasks(layout, hotspots, TASK_POOL_SIZE, dynamic_bias=1.22)
    grouped = defaultdict(list)
    for task in pool:
        grouped[getattr(task, "delivery_district", "mixed")].append(task)

    selected = []
    for district in figmod.DISTRICT_ORDER:
        candidates = sorted(
            grouped[district],
            key=lambda t: (int(t.priority), -float(t.reward), float(np.linalg.norm(t.delivery_pos[[0, 2]] - t.pickup_pos[[0, 2]]))),
        )
        selected.extend(candidates[:TASKS_PER_DISTRICT])

    selected.sort(key=lambda t: (int(t.priority), getattr(t, "delivery_district", ""), -float(t.reward)))
    return selected[:GLOBAL_ROUTE_LIMIT]


def assign_balanced_tasks(drones, tasks):
    assignments = {drone.id: [] for drone in drones}
    loads = Counter()
    positions = {drone.id: np.array(drone.position, dtype=float) for drone in drones}

    for task in sorted(tasks, key=lambda t: (int(t.priority), -float(t.reward))):
        feasible = [drone for drone in drones if drone.max_payload + 1e-6 >= task.payload_weight]
        if not feasible:
            continue

        best_drone = None
        best_key = None
        for drone in feasible:
            if loads[drone.id] >= DISPLAY_MAX_TASKS_PER_DRONE:
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
            continue
        assignments[best_drone.id].append(task.id)
        loads[best_drone.id] += 1
        positions[best_drone.id] = np.array(task.delivery_pos, dtype=float)

    return assignments


def layer_altitude(task):
    mapping = {
        "emergency_medical": 168.0,
        "medical": 142.0,
        "fresh": 118.0,
        "regular": 96.0,
        "patrol": 84.0,
    }
    return mapping.get(task.task_type, 30.0)


def catmull_rom(points, samples=80):
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 4:
        return np.array(pts, dtype=float)
    padded = [pts[0], *pts, pts[-1]]
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for t in np.linspace(0.0, 1.0, max(12, samples // (len(pts) - 1)), endpoint=False):
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(point)
    out.append(pts[-1])
    return np.array(out, dtype=float)


def bspline_curve(points, samples=90):
    pts = np.array(points, dtype=float)
    if len(pts) <= 3:
        return pts
    if splprep is not None and splev is not None:
        try:
            k = min(3, len(pts) - 1)
            tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=max(1.0, len(pts) * 0.8), k=k)
            u_new = np.linspace(0.0, 1.0, samples)
            x_new, y_new, z_new = splev(u_new, tck)
            curve = np.column_stack([x_new, y_new, z_new])
            curve[0] = pts[0]
            curve[-1] = pts[-1]
            return curve
        except Exception:
            pass
    return catmull_rom(points, samples=samples)


def clamp_service_point(point, low=3.5, high=8.0):
    clipped = np.array(point, dtype=float)
    clipped[1] = float(np.clip(clipped[1], low, high))
    return clipped


def smooth_leg(start, goal, cruise_altitude, bend_sign, bend_scale=1.0):
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    vec = goal[[0, 2]] - start[[0, 2]]
    horiz_dist = float(np.linalg.norm(vec))
    if horiz_dist < 1e-6:
        return np.vstack([start, goal])

    perp = np.array([-vec[1], vec[0]], dtype=float)
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-6:
        perp = np.array([0.0, 1.0], dtype=float)
        perp_norm = 1.0
    perp = perp / perp_norm
    offset = bend_sign * bend_scale * float(np.clip(horiz_dist * 0.16, 18.0, 54.0))

    cp1 = np.array([
        start[0] + vec[0] * 0.20 + perp[0] * offset * 0.45,
        max(start[1] + 10.0, cruise_altitude * 0.72),
        start[2] + vec[1] * 0.20 + perp[1] * offset * 0.45,
    ])
    cp2 = np.array([
        start[0] + vec[0] * 0.48 + perp[0] * offset,
        cruise_altitude,
        start[2] + vec[1] * 0.48 + perp[1] * offset,
    ])
    cp3 = np.array([
        start[0] + vec[0] * 0.78 - perp[0] * offset * 0.35,
        max(goal[1] + 10.0, cruise_altitude * 0.78),
        start[2] + vec[1] * 0.78 - perp[1] * offset * 0.35,
    ])
    return bspline_curve([start, cp1, cp2, cp3, goal], samples=96)


def smooth_leg_map(start, goal, bend_sign, bend_scale=0.28):
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    vec = goal[[0, 2]] - start[[0, 2]]
    horiz_dist = float(np.linalg.norm(vec))
    if horiz_dist < 1e-6:
        return np.vstack([start, goal])

    perp = np.array([-vec[1], vec[0]], dtype=float)
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-6:
        perp = np.array([0.0, 1.0], dtype=float)
        perp_norm = 1.0
    perp = perp / perp_norm
    offset = bend_sign * bend_scale * float(np.clip(horiz_dist * 0.08, 6.0, 16.0))

    cp1 = np.array([
        start[0] + vec[0] * 0.30 + perp[0] * offset,
        start[1],
        start[2] + vec[1] * 0.30 + perp[1] * offset,
    ])
    cp2 = np.array([
        start[0] + vec[0] * 0.68 - perp[0] * offset * 0.55,
        goal[1],
        start[2] + vec[1] * 0.68 - perp[1] * offset * 0.55,
    ])
    return bspline_curve([start, cp1, cp2, goal], samples=36)


def build_route_geometry(drones, tasks, assignments):
    task_map = {task.id: task for task in tasks}
    routes = []
    for drone in drones:
        bundle = assignments.get(drone.id, [])
        current = np.array(drone.position, dtype=float)
        for order, task_id in enumerate(bundle):
            task = task_map[task_id]
            cruise_alt = layer_altitude(task)
            bend_sign = 1.0 if (hash(f"{drone.id}:{task.id}") % 2 == 0) else -1.0
            pickup_service = clamp_service_point(task.pickup_pos, low=4.0, high=10.0)
            delivery_service = clamp_service_point(task.delivery_pos, low=3.5, high=6.5)
            leg1 = smooth_leg(current, pickup_service, cruise_alt * 0.92, bend_sign, bend_scale=0.82)
            leg2 = smooth_leg(pickup_service, delivery_service, cruise_alt, -bend_sign, bend_scale=1.0)
            points = np.vstack([leg1, leg2[1:]])
            leg1_map = smooth_leg_map(current, pickup_service, bend_sign, bend_scale=0.22)
            leg2_map = smooth_leg_map(pickup_service, delivery_service, -bend_sign, bend_scale=0.26)
            map_points = np.vstack([leg1_map, leg2_map[1:]])
            routes.append(
                {
                    "drone_id": drone.id,
                    "drone_type": drone.drone_type,
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "delivery_district": getattr(task, "delivery_district", ""),
                    "pickup_district": getattr(task, "pickup_district", ""),
                    "layer": getattr(task, "airspace_level", "L2_transition"),
                    "pickup_point": pickup_service,
                    "delivery_point": delivery_service,
                    "points": points,
                    "map_points": map_points,
                    "raw_points": np.array([current, pickup_service, delivery_service], dtype=float),
                }
            )
            current = delivery_service.copy()
    return routes


def curvature_cost(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    total = 0.0
    count = 0
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 > 1e-6 and n2 > 1e-6:
            total += math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))) ** 2
            count += 1
    return total / max(count, 1)


def summarize_routes(routes, assignments, drones):
    drone_type_counts = Counter(route["drone_type"] for route in routes)
    task_type_counts = Counter(route["task_type"] for route in routes)
    district_counts = Counter(route["delivery_district"] for route in routes)
    load_counts = [len(assignments.get(drone.id, [])) for drone in drones]
    raw_curv = np.mean([curvature_cost(route["raw_points"]) for route in routes])
    smooth_curv = np.mean([curvature_cost(route["points"]) for route in routes])
    return {
        "drone_type_counts": drone_type_counts,
        "task_type_counts": task_type_counts,
        "district_counts": district_counts,
        "active_drones": sum(1 for drone in drones if assignments.get(drone.id)),
        "max_load": max(load_counts) if load_counts else 0,
        "min_load": min(load_counts) if load_counts else 0,
        "avg_load": float(np.mean(load_counts)) if load_counts else 0.0,
        "raw_curvature": raw_curv,
        "smooth_curvature": smooth_curv,
        "curvature_drop": 0.0 if raw_curv <= 1e-9 else max(0.0, 1.0 - smooth_curv / raw_curv),
    }


def pick_local_routes(routes):
    candidates = []
    for route in routes:
        points = route["points"]
        center_score = float(np.mean(np.linalg.norm(points[:, [0, 2]], axis=1)))
        district_bonus = 0 if route["delivery_district"] in {"cbd", "mixed", "plaza"} else 1
        candidates.append((district_bonus, center_score, route))
    chosen = [item[2] for item in sorted(candidates, key=lambda item: (item[0], item[1]))[:LOCAL_ROUTE_LIMIT]]
    return chosen


def draw_fig_2_1(city):
    fig = plt.figure(figsize=(13.4, 8.6), facecolor="#fcfaf5")
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.25], wspace=0.06, hspace=0.05)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_map = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 1])

    draw_city_oblique_district(ax3d, city)
    figmod.draw_roads(ax_map, city, alpha=0.95)
    figmod.draw_buildings_topdown(ax_map, city["buildings"], mode="district", alpha=0.94)
    figmod.add_district_markers(ax_map, {k: np.array(v, dtype=float) for k, v in bench.make_hotspots(city).items()})
    figmod.format_topdown(ax_map, city["buildings"])
    ax_map.set_title("业务分区与真实城市街区对应关系", fontsize=16, weight="bold", pad=10)
    figmod.add_short_district_legend(ax_legend)

    fig.suptitle("图2.1 城市密集低空配送场景示意图", fontsize=20, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_1_dense_city_scene.png", dpi=240)
    plt.close(fig)


def draw_fig_2_9(city, routes, summary):
    fig = plt.figure(figsize=(13.8, 8.2), facecolor="#fcfaf5")
    outer = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.62, 0.82], height_ratios=[1.0, 0.13], wspace=0.08, hspace=0.03)
    ax_map = fig.add_subplot(outer[0, 0])
    side = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 1], height_ratios=[0.46, 0.24, 0.30], hspace=0.28)
    ax_summary = fig.add_subplot(side[0, 0])
    ax_drone = fig.add_subplot(side[1, 0])
    ax_task = fig.add_subplot(side[2, 0])
    ax_leg = fig.add_subplot(outer[1, :])

    figmod.draw_roads(ax_map, city, alpha=0.94)
    figmod.draw_buildings_topdown(ax_map, city["buildings"], mode="height", alpha=0.34, edge="#faf6ef", linewidth=0.14)
    figmod.format_topdown(ax_map, city["buildings"])
    ax_map.set_title("全局多无人机任务分配图", fontsize=17, weight="bold", pad=12)

    seen_drones = set()
    for route in routes:
        style = figmod.DRONE_STYLE[route["drone_type"]]
        pts = route["map_points"]
        ax_map.plot(pts[:, 0], pts[:, 2], color=style["color"], linewidth=1.9, alpha=0.76, zorder=7)
        ax_map.scatter(route["pickup_point"][0], route["pickup_point"][2], s=12, color="#f39c12", edgecolor="white", linewidth=0.45, zorder=8, alpha=0.85)
        ax_map.scatter(route["delivery_point"][0], route["delivery_point"][2], s=16, marker="s", color="#1f78b4", edgecolor="white", linewidth=0.5, zorder=8)
        if route["drone_id"] not in seen_drones:
            ax_map.scatter(pts[0, 0], pts[0, 2], s=76, marker=style["marker"], color=style["color"], edgecolor="white", linewidth=0.9, zorder=9)
            seen_drones.add(route["drone_id"])

    ax_summary.axis("off")
    ax_summary.set_facecolor("#fffdf8")
    ax_summary.text(0.02, 0.93, "分配结果摘要", fontsize=17, weight="bold", color="#1f2937")
    lines = [
        ("展示任务样本", f"{len(routes)} 个"),
        ("活跃无人机", f"{summary['active_drones']} 架"),
        ("功能区覆盖", f"{len(summary['district_counts'])} / 6"),
        ("单机任务上限", f"{summary['max_load']}"),
        ("平均单机负载", f"{summary['avg_load']:.1f}"),
        ("曲率代价下降", f"{summary['curvature_drop'] * 100:.1f}%"),
    ]
    for idx, (label, value) in enumerate(lines):
        y = 0.80 - idx * 0.11
        ax_summary.text(0.03, y, label, fontsize=12, color="#4a5568")
        ax_summary.text(0.97, y, value, fontsize=12.7, color="#111827", ha="right", weight="bold")
    ax_summary.text(0.03, 0.06, "全局图改为均衡抽样与低偏移骨架曲线，重点展示分配格局而非局部飞行动作。", fontsize=9.5, color="#5b677a", va="bottom", wrap=True)

    drone_labels = ["重型机", "标准机", "轻型机"]
    drone_values = [summary["drone_type_counts"].get(key, 0) for key in ["heavy", "standard", "light"]]
    drone_colors = [figmod.DRONE_STYLE[key]["color"] for key in ["heavy", "standard", "light"]]
    figmod.draw_count_panel(ax_drone, "样本链路按机型分布", drone_labels, drone_values, drone_colors, xmax=max(drone_values) if drone_values else 1)

    task_keys = [key for key, _ in sorted(summary["task_type_counts"].items(), key=lambda item: (-item[1], item[0]))]
    task_labels = [figmod.TASK_TYPE_CN.get(key, key) for key in task_keys]
    task_values = [summary["task_type_counts"][key] for key in task_keys]
    task_colors = [figmod.TASK_TYPE_COLOR.get(key, "#7f8c8d") for key in task_keys]
    figmod.draw_count_panel(ax_task, "样本任务按业务分布", task_labels, task_values, task_colors, xmax=max(task_values) if task_values else 1)

    ax_leg.axis("off")
    legend_handles = [
        Line2D([0], [0], color=figmod.DRONE_STYLE["heavy"]["color"], lw=3, label="重型机链路"),
        Line2D([0], [0], color=figmod.DRONE_STYLE["standard"]["color"], lw=3, label="标准机链路"),
        Line2D([0], [0], color=figmod.DRONE_STYLE["light"]["color"], lw=3, label="轻型机链路"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#f39c12", markeredgecolor="white", markersize=7, label="取件点"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#1f78b4", markeredgecolor="white", markersize=7, label="送达点"),
    ]
    ax_leg.legend(handles=legend_handles, loc="center", ncol=5, frameon=False, fontsize=11)
    ax_leg.text(0.5, 0.08, "全局图仅保留图例与摘要文字，线路区不加标签。", ha="center", va="center", fontsize=10, color="#5b677a")

    fig.suptitle("图2.9 全局多无人机任务分配图", fontsize=20, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_9_assignment_2d.png", dpi=240)
    plt.close(fig)


def draw_local_oblique(ax, city, x_min, x_max, z_min, z_max):
    buildings = []
    for b in city["buildings"]:
        bx0 = b["x"] - b["w"] * 0.5
        bx1 = b["x"] + b["w"] * 0.5
        bz0 = b["z"] - b["d"] * 0.5
        bz1 = b["z"] + b["d"] * 0.5
        if bx1 < x_min - 24 or bx0 > x_max + 24 or bz1 < z_min - 24 or bz0 > z_max + 24:
            continue
        buildings.append(b)

    max_h = max(b["top_y"] for b in buildings)
    for road in city.get("roads", []):
        if road["category"] not in {"main_road", "secondary_road"}:
            continue
        if not (x_min - 30 <= road["x"] <= x_max + 30 and z_min - 30 <= road["z"] <= z_max + 30):
            continue
        if abs(road["yaw"]) < 45 or abs(road["yaw"]) > 135:
            w = road["length"]
            d = road["width"]
        else:
            w = road["width"]
            d = road["length"]
        x0, x1 = road["x"] - w * 0.5, road["x"] + w * 0.5
        y0, y1 = road["z"] - d * 0.5, road["z"] + d * 0.5
        road_face = [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]]
        ax.add_collection3d(Poly3DCollection([road_face], facecolors="#d7dce6", edgecolors="none", alpha=0.95))

    for b in buildings:
        base_color = "#98a8c6" if b.get("uses_generated_facade") else figmod.lighten(figmod.district_color(b["district"]), 0.32)
        poly = Poly3DCollection(
            figmod.cuboid_faces(b),
            facecolors=[
                figmod.lighten(base_color, 0.20),
                figmod.darken(base_color, 0.10),
                figmod.darken(base_color, 0.18),
                figmod.darken(base_color, 0.14),
                figmod.darken(base_color, 0.24),
            ],
            edgecolors="#f7f4ed",
            linewidths=0.14,
            alpha=0.28,
        )
        ax.add_collection3d(poly)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(0.0, max_h * 1.08)
    ax.view_init(elev=35, azim=-62)
    ax.set_proj_type("persp")
    ax.set_box_aspect((x_max - x_min, z_max - z_min, max_h * 1.06))
    ax.set_facecolor("#f7efe7")
    ax.set_axis_off()


def draw_global_oblique(ax, city, routes):
    buildings = sorted(city["buildings"], key=lambda b: (b["x"] + b["z"], b["h"]))
    x_min, x_max, z_min, z_max = figmod.city_bounds(buildings)
    max_h = max(b["top_y"] for b in buildings)
    route_max_h = max(float(np.max(route["points"][:, 1])) for route in routes) if routes else max_h

    for road in city.get("roads", []):
        if road["category"] not in {"main_road", "secondary_road"}:
            continue
        if abs(road["yaw"]) < 45 or abs(road["yaw"]) > 135:
            w = road["length"]
            d = road["width"]
        else:
            w = road["width"]
            d = road["length"]
        x0, x1 = road["x"] - w * 0.5, road["x"] + w * 0.5
        y0, y1 = road["z"] - d * 0.5, road["z"] + d * 0.5
        road_face = [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]]
        ax.add_collection3d(Poly3DCollection([road_face], facecolors="#d6dbe5", edgecolors="none", alpha=0.85))

    for b in buildings:
        base_color = figmod.lighten(figmod.district_color(b["district"]), 0.34)
        poly = Poly3DCollection(
            figmod.cuboid_faces(b),
            facecolors=[
                figmod.lighten(base_color, 0.18),
                figmod.darken(base_color, 0.08),
                figmod.darken(base_color, 0.16),
                figmod.darken(base_color, 0.12),
                figmod.darken(base_color, 0.20),
            ],
            edgecolors="#f7f4ed",
            linewidths=0.10,
            alpha=0.18,
        )
        ax.add_collection3d(poly)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(0.0, max(max_h, route_max_h) * 1.12)
    ax.view_init(elev=33, azim=-57)
    ax.set_proj_type("persp")
    ax.set_box_aspect((x_max - x_min, z_max - z_min, max(max_h, route_max_h) * 0.88))
    ax.set_facecolor("#f7efe7")
    ax.set_axis_off()


def draw_fig_2_10(city, routes, summary):
    local_routes = pick_local_routes(routes)
    pts = np.concatenate([route["points"][:, [0, 2]] for route in local_routes], axis=0)
    x_min, x_max = float(pts[:, 0].min() - 40.0), float(pts[:, 0].max() + 40.0)
    z_min, z_max = float(pts[:, 1].min() - 40.0), float(pts[:, 1].max() + 40.0)

    fig = plt.figure(figsize=(13.8, 8.2), facecolor="#fcfaf5")
    outer = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.58, 0.86], height_ratios=[1.0, 0.13], wspace=0.06, hspace=0.03)
    ax3d = fig.add_subplot(outer[0, 0], projection="3d")
    side = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 1], height_ratios=[0.42, 0.24, 0.34], hspace=0.28)
    ax_summary = fig.add_subplot(side[0, 0])
    ax_layer = fig.add_subplot(side[1, 0])
    ax_drone = fig.add_subplot(side[2, 0])
    ax_leg = fig.add_subplot(outer[1, :])

    draw_local_oblique(ax3d, city, x_min, x_max, z_min, z_max)

    seen = set()
    for route in local_routes:
        style = figmod.DRONE_STYLE[route["drone_type"]]
        pts3 = route["points"]
        ax3d.plot(pts3[:, 0], pts3[:, 2], pts3[:, 1], color=style["color"], linewidth=4.2, alpha=1.0, zorder=7)
        ax3d.scatter(route["pickup_point"][0], route["pickup_point"][2], route["pickup_point"][1], s=30, color="#f39c12", edgecolor="white", linewidth=0.55, depthshade=False, zorder=8)
        ax3d.scatter(route["delivery_point"][0], route["delivery_point"][2], route["delivery_point"][1], s=30, marker="s", color="#1f78b4", edgecolor="white", linewidth=0.55, depthshade=False, zorder=8)
        if route["drone_id"] not in seen:
            ax3d.scatter(pts3[0, 0], pts3[0, 2], pts3[0, 1], s=74, marker=style["marker"], color=style["color"], edgecolor="white", linewidth=0.85, depthshade=False, zorder=9)
            seen.add(route["drone_id"])

    ax_summary.axis("off")
    ax_summary.set_facecolor("#fffdf8")
    ax_summary.text(0.02, 0.94, "局部轨迹摘要", fontsize=17, weight="bold", color="#1f2937")
    lines = [
        ("局部平滑航迹", f"{len(local_routes)} 条"),
        ("平均采样点数", f"{int(np.mean([len(route['points']) for route in local_routes]))}"),
        ("平均曲率下降", f"{summary['curvature_drop'] * 100:.1f}%"),
        ("展示飞行层", "L2 / L3 / L4"),
        ("局部最大楼高", f"{max(b['h'] for b in city['buildings']):.0f} m"),
        ("局部说明", "仅展示密集街区"),
    ]
    for idx, (label, value) in enumerate(lines):
        y = 0.82 - idx * 0.115
        ax_summary.text(0.03, y, label, fontsize=12, color="#4a5568")
        ax_summary.text(0.97, y, value, fontsize=12.6, color="#111827", ha="right", weight="bold")
    ax_summary.text(0.03, 0.06, "三维局部图强调平滑航迹的连续转弯与分层穿行效果，不在航迹区叠加说明文字。", fontsize=9.3, color="#5b677a", va="bottom", wrap=True)

    layer_counts = Counter(route["layer"] for route in local_routes)
    layer_keys = [key for key, _ in sorted(layer_counts.items(), key=lambda item: (-item[1], item[0]))]
    layer_labels = [figmod.LAYER_CN.get(key, key) for key in layer_keys]
    layer_values = [layer_counts[key] for key in layer_keys]
    layer_colors = ["#4f7dd1", "#16a085", "#d94841", "#f39c12"][: len(layer_values)]
    figmod.draw_count_panel(ax_layer, "局部航迹按空域层分布", layer_labels, layer_values, layer_colors, xmax=max(layer_values) if layer_values else 1)

    drone_counts = Counter(route["drone_id"] for route in local_routes)
    drone_items = sorted(drone_counts.items(), key=lambda item: (-item[1], item[0]))
    drone_labels = [item[0] for item in drone_items]
    drone_values = [item[1] for item in drone_items]
    drone_colors = ["#5b7cfa"] * len(drone_items)
    figmod.draw_count_panel(ax_drone, "局部航迹按无人机归属", drone_labels, drone_values, drone_colors, xmax=max(drone_values) if drone_values else 1)

    ax_leg.axis("off")
    legend_handles = [
        Line2D([0], [0], color=figmod.DRONE_STYLE["heavy"]["color"], lw=3, label="重型机平滑航迹"),
        Line2D([0], [0], color=figmod.DRONE_STYLE["standard"]["color"], lw=3, label="标准机平滑航迹"),
        Line2D([0], [0], color=figmod.DRONE_STYLE["light"]["color"], lw=3, label="轻型机平滑航迹"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#f39c12", markeredgecolor="white", markersize=8, label="取件点"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#1f78b4", markeredgecolor="white", markersize=7, label="送达点"),
    ]
    ax_leg.legend(handles=legend_handles, loc="center", ncol=5, frameon=False, fontsize=11)
    ax_leg.text(0.5, 0.08, "三维局部图仅保留图例文字，避免遮挡建筑与航迹。", ha="center", va="center", fontsize=10, color="#5b677a")

    fig.suptitle("图2.10 三维局部平滑航迹图", fontsize=20, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_10_assignment_3d.png", dpi=240)
    plt.close(fig)


def draw_fig_2_11(city, routes, summary):
    fig = plt.figure(figsize=(14.2, 8.6), facecolor="#fcfaf5")
    outer = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.64, 0.82], height_ratios=[1.0, 0.13], wspace=0.05, hspace=0.03)
    ax3d = fig.add_subplot(outer[0, 0], projection="3d")
    side = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 1], height_ratios=[0.42, 0.24, 0.34], hspace=0.30)
    ax_summary = fig.add_subplot(side[0, 0])
    ax_layer = fig.add_subplot(side[1, 0])
    ax_task = fig.add_subplot(side[2, 0])
    ax_leg = fig.add_subplot(outer[1, :])

    draw_global_oblique(ax3d, city, routes)

    seen = set()
    for route in routes:
        style = figmod.DRONE_STYLE[route["drone_type"]]
        pts3 = route["points"]
        ax3d.plot(pts3[:, 0], pts3[:, 2], pts3[:, 1], color=style["color"], linewidth=2.6, alpha=0.92, zorder=7)
        if route["drone_id"] not in seen:
            ax3d.scatter(
                pts3[0, 0],
                pts3[0, 2],
                pts3[0, 1],
                s=44,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.65,
                depthshade=False,
                zorder=9,
            )
            seen.add(route["drone_id"])
        ax3d.scatter(route["delivery_point"][0], route["delivery_point"][2], route["delivery_point"][1], s=14, marker="s", color=style["color"], edgecolor="white", linewidth=0.35, depthshade=False, zorder=8)

    ax_summary.axis("off")
    ax_summary.set_facecolor("#fffdf8")
    ax_summary.text(0.02, 0.94, "全局三维摘要", fontsize=17, weight="bold", color="#1f2937")
    lines = [
        ("平滑航迹总数", f"{len(routes)} 条"),
        ("活跃无人机", f"{summary['active_drones']} 架"),
        ("功能区覆盖", f"{len(summary['district_counts'])} / 6"),
        ("平均单机负载", f"{summary['avg_load']:.1f}"),
        ("平均曲率下降", f"{summary['curvature_drop'] * 100:.1f}%"),
        ("最大楼高", f"{max(b['h'] for b in city['buildings']):.0f} m"),
    ]
    for idx, (label, value) in enumerate(lines):
        y = 0.82 - idx * 0.115
        ax_summary.text(0.03, y, label, fontsize=12, color="#4a5568")
        ax_summary.text(0.97, y, value, fontsize=12.6, color="#111827", ha="right", weight="bold")
    ax_summary.text(0.03, 0.06, "该图展示整城范围的平滑分配航迹，轨迹颜色按机型区分，楼体透明化以保留整体空间结构。", fontsize=9.2, color="#5b677a", va="bottom", wrap=True)

    layer_counts = Counter(route["layer"] for route in routes)
    layer_keys = [key for key, _ in sorted(layer_counts.items(), key=lambda item: (-item[1], item[0]))]
    layer_labels = [figmod.LAYER_CN.get(key, key) for key in layer_keys]
    layer_values = [layer_counts[key] for key in layer_keys]
    layer_colors = ["#4f7dd1", "#16a085", "#d94841", "#f39c12"][: len(layer_values)]
    figmod.draw_count_panel(ax_layer, "全局航迹按空域层分布", layer_labels, layer_values, layer_colors, xmax=max(layer_values) if layer_values else 1)

    task_keys = [key for key, _ in sorted(summary["task_type_counts"].items(), key=lambda item: (-item[1], item[0]))]
    task_labels = [figmod.TASK_TYPE_CN.get(key, key) for key in task_keys]
    task_values = [summary["task_type_counts"][key] for key in task_keys]
    task_colors = [figmod.TASK_TYPE_COLOR.get(key, "#7f8c8d") for key in task_keys]
    figmod.draw_count_panel(ax_task, "全局样本按业务分布", task_labels, task_values, task_colors, xmax=max(task_values) if task_values else 1)

    ax_leg.axis("off")
    legend_handles = [
        Line2D([0], [0], color=figmod.DRONE_STYLE["heavy"]["color"], lw=3, label="重型机平滑航迹"),
        Line2D([0], [0], color=figmod.DRONE_STYLE["standard"]["color"], lw=3, label="标准机平滑航迹"),
        Line2D([0], [0], color=figmod.DRONE_STYLE["light"]["color"], lw=3, label="轻型机平滑航迹"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor=figmod.DRONE_STYLE["heavy"]["color"], markeredgecolor="white", markersize=7, label="无人机起点"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#4f7dd1", markeredgecolor="white", markersize=6, label="送达点"),
    ]
    ax_leg.legend(handles=legend_handles, loc="center", ncol=5, frameon=False, fontsize=11)
    ax_leg.text(0.5, 0.08, "全局三维图仅保留图例和右侧摘要，不在航迹区域叠加文字。", ha="center", va="center", fontsize=10, color="#5b677a")

    fig.suptitle("图2.11 全局三维任务分配图", fontsize=20, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_11_global_assignment_3d.png", dpi=240)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figmod.setup_font()
    city = load_city()
    tasks = select_balanced_tasks(city)
    drones = bench.build_drones()
    assignments = assign_balanced_tasks(drones, tasks)
    routes = build_route_geometry(drones, tasks, assignments)
    summary = summarize_routes(routes, assignments, drones)

    draw_fig_2_1(city)
    draw_fig_2_9(city, routes, summary)
    draw_fig_2_10(city, routes, summary)
    draw_fig_2_11(city, routes, summary)
    print("Rendered figure files:")
    print(FIG_DIR / "fig_2_1_dense_city_scene.png")
    print(FIG_DIR / "fig_2_9_assignment_2d.png")
    print(FIG_DIR / "fig_2_10_assignment_3d.png")
    print(FIG_DIR / "fig_2_11_global_assignment_3d.png")


if __name__ == "__main__":
    main()
