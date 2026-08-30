"""
Generate the final figure set for Section 2.2 of the midterm report.
All figures are derived from the real Empty.umap-exported city geometry
and the optimized benchmark outputs.
"""

from __future__ import annotations

from collections import Counter
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from backend.config import DRONE_TYPES, FLIGHT_LEVELS


FIG_DIR = ROOT / "thesis" / "figures"
BENCH_PATH = ROOT / "data" / "midterm_benchmark.json"
MANIFEST_PATH = FIG_DIR / "figure_manifest.json"
ASSIGNMENT_VIEW_PATH = ROOT / "data" / "stc_assignment_view.json"


DISTRICT_META = {
    "industrial": {"label": "工业仓储区", "color": "#d64f4f"},
    "mixed": {"label": "社区混合区", "color": "#f39c12"},
    "cbd": {"label": "商务医疗区", "color": "#2e86de"},
    "park": {"label": "绿地巡检区", "color": "#23a36d"},
    "residential": {"label": "居住末端区", "color": "#8e44ad"},
    "plaza": {"label": "市政保障区", "color": "#16a085"},
}
DISTRICT_ORDER = ["industrial", "mixed", "cbd", "park", "residential", "plaza"]
SCENARIO_CN = {
    "dense_occlusion": "高密度遮挡",
    "building_occlusion": "建筑遮挡",
    "full_comm": "全连通",
    "intermittent_comm": "间歇通信",
    "local_island": "局部孤岛",
    "regular_density": "常规密度",
    "street_canyon_dense": "高密度街谷",
    "dynamic_injection": "动态任务注入",
}

DRONE_STYLE = {
    "heavy": {"label": "重型机链路", "color": "#d94841", "marker": "^"},
    "standard": {"label": "标准机链路", "color": "#2e86de", "marker": "o"},
    "light": {"label": "轻型机链路", "color": "#16a085", "marker": "D"},
}

TASK_TYPE_CN = {
    "emergency_medical": "紧急医疗",
    "medical": "常规医疗",
    "fresh": "生鲜零售",
    "regular": "普通快件",
    "patrol": "巡检回传",
}

TASK_TYPE_COLOR = {
    "emergency_medical": "#d94841",
    "medical": "#4f7dd1",
    "fresh": "#f39c12",
    "regular": "#7f8c8d",
    "patrol": "#27ae60",
}

ALGO_CN = {
    "STC-RCBBA": "STC-RCBBA",
    "原始CBBA": "原始CBBA",
    "ԭʼCBBA": "原始CBBA",
    "Auction": "拍卖算法",
    "Hungarian": "匈牙利",
    "Greedy": "贪心算法",
    "Genetic": "遗传算法",
    "Market": "市场机制",
    "PSO": "粒子群",
    "GWO": "灰狼优化",
    "ACO": "蚁群算法",
    "WOA": "鲸鱼优化",
    "SA": "模拟退火",
    "DE": "差分进化",
    "去掉优先级紧迫项": "去掉优先级项",
    "去掉通信鲁棒共识": "去掉鲁棒共识",
    "去掉走廊冲突代价": "去掉走廊代价",
    "去掉B样条重定形": "去掉B样条",
}

ALGO_ORDER = [
    "STC-RCBBA",
    "原始CBBA",
    "Auction",
    "Hungarian",
    "Greedy",
    "Genetic",
    "ACO",
    "PSO",
    "GWO",
    "DE",
    "SA",
    "WOA",
    "Market",
]

ALGO_FAMILY = {
    "STC-RCBBA": "分布式协同",
    "原始CBBA": "分布式协同",
    "ԭʼCBBA": "分布式协同",
    "Auction": "分布式协同",
    "Hungarian": "集中式精确",
    "Greedy": "快速启发式",
    "Genetic": "群智能优化",
    "Market": "市场机制",
    "PSO": "群智能优化",
    "GWO": "群智能优化",
    "ACO": "群智能优化",
    "WOA": "群智能优化",
    "SA": "群智能优化",
    "DE": "群智能优化",
}

FAMILY_META = {
    "分布式协同": {"color": "#d94841"},
    "集中式精确": {"color": "#2e86de"},
    "快速启发式": {"color": "#6f7f95"},
    "群智能优化": {"color": "#3b82c4"},
    "市场机制": {"color": "#b33b5e"},
}

LAYER_CN = {
    "L1_street_canyon": "街谷层",
    "L2_transition": "过渡层",
    "L3_trunk_corridor": "干线层",
    "L4_emergency": "应急层",
}


def canonical_algo_name(name: str) -> str:
    if name == "STC-RCBBA":
        return name
    if "CBBA" in name:
        return "原始CBBA"
    return name


def setup_font() -> None:
    for font in fm.fontManager.ttflist:
        if any(key in font.name for key in ("Microsoft YaHei", "SimHei", "Noto Sans CJK", "PingFang", "WenQuanYi")):
            plt.rcParams["font.family"] = font.name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 240
    plt.rcParams["savefig.bbox"] = "tight"


def load_data():
    data = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    for section in ("main_results", "communication_results", "complexity_results"):
        for row in data.get(section, []):
            row["algorithm"] = canonical_algo_name(row["algorithm"])
    city_path = ROOT / data["meta"]["city_source"]
    city = json.loads(city_path.read_text(encoding="utf-8"))
    return data, city


def city_bounds(buildings):
    xs = [b["x"] - b["w"] * 0.5 for b in buildings] + [b["x"] + b["w"] * 0.5 for b in buildings]
    zs = [b["z"] - b["d"] * 0.5 for b in buildings] + [b["z"] + b["d"] * 0.5 for b in buildings]
    return min(xs) - 20.0, max(xs) + 20.0, min(zs) - 20.0, max(zs) + 20.0


def district_color(key: str) -> str:
    return DISTRICT_META.get(key, {"color": "#7f8c8d"})["color"]


def lighten(color, factor=0.22):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.clip(rgb + (1.0 - rgb) * factor, 0.0, 1.0))


def darken(color, factor=0.24):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(np.clip(rgb * (1.0 - factor), 0.0, 1.0))


def algo_label(name: str) -> str:
    name = canonical_algo_name(name)
    return ALGO_CN.get(name, name)


def algo_family(name: str) -> str:
    name = canonical_algo_name(name)
    return ALGO_FAMILY.get(name, "其他算法")


def robust_normalize(values, higher_better=True, log_scale=False, lower_q=0.05, upper_q=0.95):
    arr = np.array(values, dtype=float)
    if log_scale:
        arr = np.log1p(np.clip(arr, 0.0, None))
    lo, hi = np.quantile(arr, [lower_q, upper_q])
    if abs(hi - lo) < 1e-9:
        scaled = np.ones_like(arr)
    else:
        clipped = np.clip(arr, lo, hi)
        scaled = (clipped - lo) / (hi - lo)
    return scaled if higher_better else (1.0 - scaled)


def draw_matrix_panel(
    ax,
    matrix,
    row_labels,
    col_labels,
    title,
    cmap,
    annotations=None,
    norm=None,
    vmin=None,
    vmax=None,
    title_size=14,
):
    ax.set_facecolor("#ffffff")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", norm=norm, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(title, fontsize=title_size, weight="bold", pad=12)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        patches.Rectangle(
            (-0.5, -0.5),
            matrix.shape[1],
            matrix.shape[0],
            fill=False,
            edgecolor="#9bb0d8",
            linewidth=1.4,
            linestyle=(0, (4, 3)),
        )
    )
    if annotations is not None:
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                if norm is not None:
                    strength = float(norm(value))
                elif vmin is not None and vmax is not None and abs(vmax - vmin) > 1e-9:
                    strength = float((value - vmin) / (vmax - vmin))
                else:
                    strength = 0.5
                txt_color = "#ffffff" if strength > 0.62 else "#3e4a5d"
                ax.text(c, r, annotations[r][c], ha="center", va="center", fontsize=9.2, color=txt_color)
    return im


def draw_roads(ax, city, alpha=1.0):
    for road in city.get("roads", []):
        if road["category"] not in {"main_road", "secondary_road"}:
            continue
        if abs(road["yaw"]) < 45 or abs(road["yaw"]) > 135:
            w = road["length"]
            d = road["width"]
        else:
            w = road["width"]
            d = road["length"]
        ax.add_patch(
            patches.Rectangle(
                (road["x"] - w * 0.5, road["z"] - d * 0.5),
                w,
                d,
                facecolor="#d9dde5" if road["category"] == "main_road" else "#e7eaf0",
                edgecolor="none",
                alpha=alpha,
                zorder=0,
            )
        )


def draw_buildings_topdown(ax, buildings, mode="height", alpha=1.0, edge="#f9f7f1", linewidth=0.22):
    heights = np.array([b["h"] for b in buildings], dtype=float)
    cmap = LinearSegmentedColormap.from_list("height", ["#f1ddc7", "#c0c7d9", "#5f7399"])
    norm = Normalize(vmin=float(heights.min()), vmax=float(heights.max()))

    for b in buildings:
        if mode == "district":
            face = lighten(district_color(b["district"]), 0.10)
        else:
            base = "#7a8ba8" if b.get("uses_generated_facade") else cmap(norm(b["h"]))
            face = base
        ax.add_patch(
            patches.Rectangle(
                (b["x"] - b["w"] * 0.5, b["z"] - b["d"] * 0.5),
                b["w"],
                b["d"],
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                alpha=alpha,
                zorder=2,
            )
        )


def format_topdown(ax, buildings):
    x_min, x_max, z_min, z_max = city_bounds(buildings)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f7f3ea")


def add_district_markers(ax, hotspots):
    for idx, key in enumerate(DISTRICT_ORDER, start=1):
        pt = hotspots[key]
        ax.scatter(pt[0], pt[2], s=92, color=district_color(key), edgecolor="white", linewidth=1.0, zorder=6)
        ax.text(
            pt[0],
            pt[2],
            str(idx),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
            zorder=7,
        )


def add_short_district_legend(ax):
    ax.axis("off")
    entries = []
    for idx, key in enumerate(DISTRICT_ORDER, start=1):
        entries.append((idx, DISTRICT_META[key]["label"], district_color(key)))

    for i, (idx, label, color) in enumerate(entries):
        col = i % 2
        row = i // 2
        x = 0.04 + col * 0.47
        y = 0.82 - row * 0.28
        badge = patches.FancyBboxPatch(
            (x, y),
            0.07,
            0.13,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color,
            edgecolor="none",
            transform=ax.transAxes,
        )
        ax.add_patch(badge)
        ax.text(x + 0.035, y + 0.065, str(idx), ha="center", va="center", fontsize=10, color="white", weight="bold", transform=ax.transAxes)
        ax.text(x + 0.10, y + 0.065, label, ha="left", va="center", fontsize=11, color="#243447", transform=ax.transAxes)


def cuboid_faces(building):
    x0 = building["x"] - building["w"] * 0.5
    x1 = building["x"] + building["w"] * 0.5
    y0 = building["z"] - building["d"] * 0.5
    y1 = building["z"] + building["d"] * 0.5
    z0 = building["bottom_y"]
    z1 = building["top_y"]
    verts = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    faces = [
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[3], verts[0], verts[4], verts[7]],
    ]
    return faces


def draw_city_oblique(ax, city):
    core = [b for b in city["buildings"] if abs(b["x"]) <= 240 and abs(b["z"]) <= 220]
    buildings = sorted(core or city["buildings"], key=lambda b: (b["x"] + b["z"], b["h"]))
    x_min, x_max, z_min, z_max = city_bounds(buildings)
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
        ax.add_collection3d(
            Poly3DCollection([road_face], facecolors="#d7dce6", edgecolors="none", alpha=1.0)
        )

    height_norm = Normalize(vmin=min(b["h"] for b in buildings), vmax=max(b["h"] for b in buildings))
    height_cmap = LinearSegmentedColormap.from_list("oblique", ["#e9decf", "#bfc8db", "#6379a3"])

    for b in buildings:
        base_color = "#6d89b0" if b.get("uses_generated_facade") else height_cmap(height_norm(b["h"]))
        faces = cuboid_faces(b)
        face_colors = [
            lighten(base_color, 0.24),
            darken(base_color, 0.22),
            darken(base_color, 0.08),
            darken(base_color, 0.18),
            darken(base_color, 0.28),
        ]
        poly = Poly3DCollection(
            faces,
            facecolors=face_colors,
            edgecolors="#f7f4ed",
            linewidths=0.16,
            alpha=0.98,
        )
        ax.add_collection3d(poly)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(0.0, max_h * 1.05)
    ax.view_init(elev=27, azim=-52)
    ax.set_proj_type("persp")
    ax.set_box_aspect((x_max - x_min, z_max - z_min, max_h * 1.10))
    ax.set_facecolor("#f7efe7")
    ax.set_axis_off()


def get_stc_assignment_view(data):
    stc = next(row for row in data["main_results"] if row["algorithm"] == "STC-RCBBA")
    drone_map = {row["id"]: row for row in data["sample_city"]["drones"]}
    task_map = {row["id"]: row for row in data["sample_city"]["tasks"]}

    routes = []
    for sample in stc.get("route_samples", []):
        task = task_map[sample["task_id"]]
        drone = drone_map[sample["drone_id"]]
        routes.append(
            {
                "drone_id": sample["drone_id"],
                "drone_type": drone["drone_type"],
                "task_id": sample["task_id"],
                "task_type": task["task_type"],
                "layer": sample["layer"],
                "points": np.array(sample["points"], dtype=float),
                "pickup_point": np.array(sample.get("pickup_point", task["pickup_pos"]), dtype=float),
                "delivery_point": np.array(sample.get("delivery_point", task["delivery_pos"]), dtype=float),
            }
        )

    balanced_routes = build_balanced_view_routes(max_samples=max(20, len(routes)))
    if len({row["drone_id"] for row in balanced_routes}) > len({row["drone_id"] for row in routes}):
        routes = balanced_routes

    summary = {
        "completed_tasks": int(stc.get("completed_tasks", 0)),
        "total_tasks": int(data["meta"]["task_count"]),
        "assignment_rate": float(stc.get("assignment_rate", 0.0)),
        "weighted_completion_rate": float(stc.get("weighted_completion_rate", 0.0)),
        "time_window_rate": float(stc.get("time_window_rate", 0.0)),
        "corridor_conflicts": int(stc.get("corridor_conflicts", 0)),
        "runtime_ms": float(stc.get("runtime_ms", 0.0)),
        "avg_completion_time_s": float(stc.get("avg_completion_time_s", 0.0)),
        "sample_chain_count": len(routes),
        "sample_active_drone_count": len({row["drone_id"] for row in routes}),
    }
    return stc, routes, drone_map, task_map, summary


def route_focus_bounds(routes, pad=60.0, points_key="points"):
    pts = np.concatenate([row[points_key][:, [0, 2]] for row in routes], axis=0)
    return (
        float(pts[:, 0].min() - pad),
        float(pts[:, 0].max() + pad),
        float(pts[:, 1].min() - pad),
        float(pts[:, 1].max() + pad),
    )


def catmull_rom(points, samples=84):
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 2:
        return np.array(pts, dtype=float)
    if len(pts) < 4:
        return np.array(pts, dtype=float)
    padded = [pts[0], *pts, pts[-1]]
    out = []
    seg_samples = max(14, samples // max(1, len(pts) - 1))
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for t in np.linspace(0.0, 1.0, seg_samples, endpoint=False):
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


def layer_cruise_altitude(layer: str) -> float:
    cfg = FLIGHT_LEVELS.get(layer or "", None)
    if cfg is None:
        return 26.0
    return float((cfg["y_min"] + cfg["y_max"]) * 0.5)


def smooth_leg_display(start, goal, cruise_altitude, bend_sign, bend_scale=0.88, samples=88):
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
    offset = bend_sign * bend_scale * float(np.clip(horiz_dist * 0.10, 10.0, 26.0))

    cp1 = np.array([
        start[0] + vec[0] * 0.24 + perp[0] * offset,
        max(start[1] + 6.0, cruise_altitude * 0.78),
        start[2] + vec[1] * 0.24 + perp[1] * offset,
    ])
    cp2 = np.array([
        start[0] + vec[0] * 0.55 + perp[0] * offset * 0.35,
        cruise_altitude,
        start[2] + vec[1] * 0.55 + perp[1] * offset * 0.35,
    ])
    cp3 = np.array([
        start[0] + vec[0] * 0.82 - perp[0] * offset * 0.55,
        max(goal[1] + 6.0, cruise_altitude * 0.82),
        start[2] + vec[1] * 0.82 - perp[1] * offset * 0.55,
    ])
    return catmull_rom([start, cp1, cp2, cp3, goal], samples=samples)


def smooth_leg_map_display(start, goal, bend_sign, bend_scale=0.30, samples=44):
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
    offset = bend_sign * bend_scale * float(np.clip(horiz_dist * 0.08, 7.0, 16.0))

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
    return catmull_rom([start, cp1, cp2, goal], samples=samples)


def build_display_routes(routes):
    display_routes = []
    for idx, row in enumerate(routes):
        start = np.array(row["points"][0], dtype=float)
        pickup = np.array(row["pickup_point"], dtype=float)
        delivery = np.array(row["delivery_point"], dtype=float)
        bend_sign = 1.0 if idx % 2 == 0 else -1.0
        cruise_alt = max(
            layer_cruise_altitude(row["layer"]),
            float(start[1] + 6.0),
            float(pickup[1] + 8.0),
            float(delivery[1] + 6.0),
        )
        leg1_air = smooth_leg_display(start, pickup, cruise_alt * 0.92, bend_sign, bend_scale=0.82, samples=72)
        leg2_air = smooth_leg_display(pickup, delivery, cruise_alt, -bend_sign, bend_scale=0.92, samples=78)
        leg1_map = smooth_leg_map_display(start, pickup, bend_sign, bend_scale=0.24, samples=34)
        leg2_map = smooth_leg_map_display(pickup, delivery, -bend_sign, bend_scale=0.28, samples=38)
        display = dict(row)
        display["display_points_3d"] = np.vstack([leg1_air, leg2_air[1:]])
        display["display_points_2d"] = np.vstack([leg1_map, leg2_map[1:]])
        display_routes.append(display)
    return display_routes


def build_balanced_view_routes(max_samples=20):
    try:
        import export_path_optimized_assignment_view as assignment_view
    except Exception:
        return []

    try:
        layout, density_meta, drones, tasks, assignments, metrics = assignment_view.build_assignment()
        task_map = {task.id: task for task in tasks}
        samples = assignment_view.select_sample_tasks(drones, task_map, assignments, max_samples)
    except Exception:
        return []

    route_rows = []
    for drone, task_rank, task in samples:
        bundle = assignments.get(drone.id, [])
        start_pos = np.array(drone.position, dtype=float)
        for prev_task_id in bundle[:task_rank]:
            prev_task = task_map[prev_task_id]
            start_pos = np.array(prev_task.delivery_pos, dtype=float)

        layer = getattr(task, "airspace_level", "L2_transition") or "L2_transition"
        route_rows.append(
            {
                "drone_id": drone.id,
                "drone_type": drone.drone_type,
                "task_id": task.id,
                "task_type": task.task_type,
                "layer": layer,
                "points": np.array([start_pos, np.array(task.pickup_pos, dtype=float), np.array(task.delivery_pos, dtype=float)], dtype=float),
                "pickup_point": np.array(task.pickup_pos, dtype=float),
                "delivery_point": np.array(task.delivery_pos, dtype=float),
            }
        )
    return route_rows


def select_buildings_in_bounds(buildings, x_min, x_max, z_min, z_max, pad=24.0):
    selected = []
    for b in buildings:
        bx0 = b["x"] - b["w"] * 0.5
        bx1 = b["x"] + b["w"] * 0.5
        bz0 = b["z"] - b["d"] * 0.5
        bz1 = b["z"] + b["d"] * 0.5
        if bx1 < x_min - pad or bx0 > x_max + pad:
            continue
        if bz1 < z_min - pad or bz0 > z_max + pad:
            continue
        selected.append(b)
    return selected


def style_axis_clean(ax):
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", alpha=0.18, zorder=0)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)


def draw_count_panel(ax, title, labels, values, colors, xmax=None):
    style_axis_clean(ax)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.92, height=0.58, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    if xmax is None:
        xmax = max(values) if values else 1
    ax.set_xlim(0, max(1.0, xmax * 1.18))
    ax.set_title(title, fontsize=13, weight="bold", pad=8)
    for idx, val in enumerate(values):
        ax.text(val + max(0.05, xmax * 0.03), idx, str(int(val)), va="center", fontsize=10, color="#2d3748")


def polish_compact_panel(ax, title):
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.set_title(title, fontsize=13.2, weight="bold", y=1.05, pad=0)


def draw_city_oblique_region(ax, city, buildings, x_min, x_max, z_min, z_max, bg_face="#ffffff"):
    if not buildings:
        buildings = city["buildings"]
        x_min, x_max, z_min, z_max = city_bounds(buildings)

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
        ax.add_collection3d(Poly3DCollection([road_face], facecolors="#e4e9f2", edgecolors="none", alpha=0.95))

    for b in buildings:
        base_color = "#88a0c5" if b.get("uses_generated_facade") else lighten(district_color(b["district"]), 0.34)
        face_colors = [
            lighten(base_color, 0.20),
            darken(base_color, 0.08),
            darken(base_color, 0.18),
            darken(base_color, 0.12),
            darken(base_color, 0.22),
        ]
        poly = Poly3DCollection(
            cuboid_faces(b),
            facecolors=face_colors,
            edgecolors="#fbfdff",
            linewidths=0.14,
            alpha=0.48,
        )
        ax.add_collection3d(poly)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(0.0, max_h * 1.10)
    ax.view_init(elev=28, azim=-58)
    ax.set_proj_type("persp")
    ax.set_box_aspect((x_max - x_min, z_max - z_min, max_h * 1.08))
    ax.set_facecolor(bg_face)
    ax.set_axis_off()


def cumulative_distance(points):
    if len(points) == 0:
        return np.array([], dtype=float)
    diffs = np.diff(points[:, [0, 2]], axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def turning_focus(points):
    if len(points) < 3:
        idx = max(0, len(points) // 2)
        return idx, points[idx]
    diffs = np.diff(points[:, [0, 2]], axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    delta = np.diff(headings)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    idx = int(np.argmax(np.abs(delta))) + 1
    return idx, points[idx]


def write_assignment_snapshot(stc, routes, summary):
    drone_type_counts = Counter(row["drone_type"] for row in routes)
    task_type_counts = Counter(row["task_type"] for row in routes)
    layer_counts = Counter(row["layer"] for row in routes)
    payload = {
        "algorithm": "STC-RCBBA",
        "summary": summary,
        "sample_drone_type_counts": dict(drone_type_counts),
        "sample_task_type_counts": dict(task_type_counts),
        "sample_layer_counts": dict(layer_counts),
        "sample_routes": [
            {
                "drone_id": row["drone_id"],
                "drone_type": row["drone_type"],
                "task_id": row["task_id"],
                "task_type": row["task_type"],
                "layer": row["layer"],
                "points": row["points"].tolist(),
                "pickup_point": row["pickup_point"].tolist(),
                "delivery_point": row["delivery_point"].tolist(),
            }
            for row in routes
        ],
        "benchmark_metrics": {
            "assignment_rate": stc.get("assignment_rate", 0.0),
            "weighted_completion_rate": stc.get("weighted_completion_rate", 0.0),
            "time_window_rate": stc.get("time_window_rate", 0.0),
            "corridor_conflicts": stc.get("corridor_conflicts", 0),
            "runtime_ms": stc.get("runtime_ms", 0.0),
        },
    }
    ASSIGNMENT_VIEW_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_fig_2_1(data, city):
    fig = plt.figure(figsize=(13.4, 8.6), facecolor="#fcfaf5")
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.25], wspace=0.06, hspace=0.05)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_map = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 1])

    draw_city_oblique(ax3d, city)
    draw_roads(ax_map, city, alpha=0.95)
    draw_buildings_topdown(ax_map, city["buildings"], mode="district", alpha=0.92)
    add_district_markers(ax_map, {k: np.array(v, dtype=float) for k, v in data["hotspots"].items()})
    format_topdown(ax_map, city["buildings"])
    ax_map.set_title("业务分区与真实街区对应关系", fontsize=16, weight="bold", pad=10)
    add_short_district_legend(ax_legend)

    fig.suptitle("图2.1 城市密集低空配送场景示意图", fontsize=20, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_1_dense_city_scene.png", dpi=240)
    plt.close(fig)


def generate_fig_2_2(data, city):
    drones = data["sample_city"]["drones"]
    tasks = data["sample_city"]["tasks"]
    pickups = np.array([t["pickup_pos"] for t in tasks], dtype=float)
    deliveries = np.array([t["delivery_pos"] for t in tasks], dtype=float)

    fig = plt.figure(figsize=(13.0, 6.8), facecolor="#fcfaf5")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.08)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    for ax in (ax_left, ax_right):
        draw_roads(ax, city, alpha=0.88)
        draw_buildings_topdown(ax, city["buildings"], mode="height", alpha=0.56, edge="#f7f3ec", linewidth=0.18)
        format_topdown(ax, city["buildings"])

    type_style = {
        "heavy": {"marker": "^", "color": "#d94841", "size": 92, "label": "重型 5 架"},
        "standard": {"marker": "o", "color": "#2e86de", "size": 58, "label": "标准 15 架"},
        "light": {"marker": "D", "color": "#23a36d", "size": 48, "label": "轻型 10 架"},
    }
    for dtype, style in type_style.items():
        pts = np.array([d["position"] for d in drones if d["drone_type"] == dtype], dtype=float)
        ax_left.scatter(pts[:, 0], pts[:, 2], s=style["size"], marker=style["marker"], color=style["color"], edgecolor="white", linewidth=0.8, alpha=0.96, label=style["label"], zorder=5)
    ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=10)
    ax_left.set_title("异构机群部署", fontsize=15, weight="bold", pad=10)

    ax_right.hexbin(pickups[:, 0], pickups[:, 2], gridsize=18, cmap="Oranges", mincnt=1, alpha=0.55, linewidths=0.0, zorder=4)
    ax_right.hexbin(deliveries[:, 0], deliveries[:, 2], gridsize=18, cmap="GnBu", mincnt=1, alpha=0.45, linewidths=0.0, zorder=5)
    ax_right.scatter(pickups[:, 0], pickups[:, 2], s=10, color="#f39c12", alpha=0.28, edgecolor="none", zorder=6)
    ax_right.scatter(deliveries[:, 0], deliveries[:, 2], s=10, color="#1f78b4", alpha=0.24, edgecolor="none", zorder=6)
    ax_right.set_title("任务热点分布", fontsize=15, weight="bold", pad=10)
    ax_right.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#f39c12", markeredgecolor="none", markersize=8, label="取件热点"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f78b4", markeredgecolor="none", markersize=8, label="送达热点"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    fig.suptitle("图2.2 异构无人机与任务热点分布图", fontsize=19, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_2_heterogeneous_hotspots.png", dpi=240)
    plt.close(fig)


def line_intersects_rect_2d(p0: np.ndarray, p1: np.ndarray, rect: dict) -> bool:
    x_min = rect["x"] - rect["w"] * 0.5
    x_max = rect["x"] + rect["w"] * 0.5
    z_min = rect["z"] - rect["d"] * 0.5
    z_max = rect["z"] + rect["d"] * 0.5
    dx = p1[0] - p0[0]
    dz = p1[2] - p0[2]
    t0, t1 = 0.0, 1.0

    for p, q0, q1 in ((dx, x_min - p0[0], x_max - p0[0]), (dz, z_min - p0[2], z_max - p0[2])):
        if abs(p) < 1e-9:
            if q0 > 0 or q1 < 0:
                return False
            continue
        t_enter = q0 / p
        t_exit = q1 / p
        if t_enter > t_exit:
            t_enter, t_exit = t_exit, t_enter
        t0 = max(t0, t_enter)
        t1 = min(t1, t_exit)
        if t0 > t1:
            return False
    return True


def blocked_by_buildings(p0: np.ndarray, p1: np.ndarray, buildings) -> bool:
    altitude = 0.5 * (p0[1] + p1[1])
    for rect in buildings:
        if altitude > rect["h"] + 6.0:
            continue
        if line_intersects_rect_2d(p0, p1, rect):
            return True
    return False


def generate_fig_2_3(data, city):
    drones = data["sample_city"]["drones"]
    graph = np.array(data["sample_city"]["comm_graph_occlusion"], dtype=float)
    positions = np.array([d["position"] for d in drones], dtype=float)
    selected = np.argsort(np.linalg.norm(positions[:, [0, 2]], axis=1))[:14]

    fig = plt.figure(figsize=(13.0, 6.8), facecolor="#fcfaf5")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.08)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    for ax in (ax_left, ax_right):
        draw_roads(ax, city, alpha=0.88)
        draw_buildings_topdown(ax, city["buildings"], mode="height", alpha=0.62, edge="#f9f5ee", linewidth=0.18)
        format_topdown(ax, city["buildings"])

    active_edges = []
    blocked_edges = []
    for i_idx, i in enumerate(selected):
        p_i = positions[i]
        d_i = drones[i]
        for j in selected[i_idx + 1:]:
            p_j = positions[j]
            d_j = drones[j]
            max_range = min(DRONE_TYPES[d_i["drone_type"]]["comm_range"], DRONE_TYPES[d_j["drone_type"]]["comm_range"])
            if np.linalg.norm(p_i[[0, 2]] - p_j[[0, 2]]) > max_range:
                continue
            if graph[i, j] > 0:
                active_edges.append((p_i, p_j))
            elif blocked_by_buildings(p_i, p_j, city["buildings"]):
                blocked_edges.append((p_i, p_j))

    blocked_edges = sorted(blocked_edges, key=lambda item: np.linalg.norm(item[0][[0, 2]] - item[1][[0, 2]]))[:12]

    for p_i, p_j in active_edges:
        ax_left.plot([p_i[0], p_j[0]], [p_i[2], p_j[2]], color="#2e86de", linewidth=1.4, alpha=0.72, zorder=4)
    for p_i, p_j in blocked_edges:
        ax_right.plot([p_i[0], p_j[0]], [p_i[2], p_j[2]], color="#d94841", linewidth=1.4, linestyle=(0, (4, 3)), alpha=0.85, zorder=4)

    for ax in (ax_left, ax_right):
        ax.scatter(positions[selected, 0], positions[selected, 2], s=52, color="#273c75", edgecolor="white", linewidth=0.9, zorder=6)

    ax_left.set_title("可达通信链路", fontsize=15, weight="bold", pad=10)
    ax_right.set_title("建筑遮挡失联链路", fontsize=15, weight="bold", pad=10)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#2e86de", lw=2.0, label="有效通信"),
            Line2D([0], [0], color="#d94841", lw=2.0, linestyle=(0, (4, 3)), label="遮挡失联"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle("图2.3 时变通信图与建筑遮挡示意图", fontsize=19, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_3_comm_occlusion.png", dpi=240)
    plt.close(fig)


def add_flow_box(ax, xy, w, h, title, face, edge="#243447"):
    box = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, title, ha="center", va="center", fontsize=12, color="#1f2d3d", weight="bold", transform=ax.transAxes)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#51606f"))


def generate_fig_2_4():
    fig, ax = plt.subplots(figsize=(12.6, 5.8), facecolor="#fcfaf5")
    ax.axis("off")

    add_flow_box(ax, (0.03, 0.36), 0.15, 0.20, "输入\nU, T, G(t), O", "#e9f0fb")
    add_flow_box(ax, (0.23, 0.60), 0.20, 0.20, "候选筛选\n机型·载荷·电量·邻居数", "#fdebd0")
    add_flow_box(ax, (0.47, 0.60), 0.22, 0.20, "鲁棒束构建\n优先级·时窗·风险·走廊", "#f9e2d2")
    add_flow_box(ax, (0.73, 0.60), 0.21, 0.20, "增量共识\n版本向量·时间戳·事件触发", "#dff0ea")
    add_flow_box(ax, (0.73, 0.22), 0.21, 0.18, "任务束输出\n赢家表 / 时空占用表", "#e9f7ef")
    add_flow_box(ax, (0.47, 0.22), 0.22, 0.18, "time-aware A*\n骨架路径", "#e8eef9")
    add_flow_box(ax, (0.23, 0.22), 0.20, 0.18, "三次B样条\n动力学重定形", "#e8f6f3")

    arrow(ax, 0.18, 0.46, 0.23, 0.70)
    arrow(ax, 0.43, 0.70, 0.47, 0.70)
    arrow(ax, 0.69, 0.70, 0.73, 0.70)
    arrow(ax, 0.84, 0.60, 0.84, 0.40)
    arrow(ax, 0.73, 0.31, 0.69, 0.31)
    arrow(ax, 0.47, 0.31, 0.43, 0.31)

    ax.text(0.50, 0.93, "图2.4 STC-RCBBA 流程图", ha="center", va="center", fontsize=20, weight="bold", transform=ax.transAxes)
    fig.savefig(FIG_DIR / "fig_2_4_stc_rcbba_flow.png", dpi=240)
    plt.close(fig)


def generate_fig_2_5():
    fig = plt.figure(figsize=(12.6, 6.2), facecolor="#fcfaf5")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 0.85], wspace=0.10)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax_left.set_facecolor("#f7f3ea")
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 8)
    ax_left.axis("off")

    corridor_colors = ["#2e86de", "#f39c12", "#16a085", "#8e44ad"]
    for i in range(4):
        rect = patches.FancyBboxPatch((1.0 + i * 1.8, 2.7), 1.45, 2.6, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=lighten(corridor_colors[i], 0.65), edgecolor=corridor_colors[i], linewidth=1.2)
        ax_left.add_patch(rect)
        ax_left.text(1.72 + i * 1.8, 4.0, f"C{i + 1}", ha="center", va="center", fontsize=12, weight="bold", color="#22313f")

    routes = [
        ([(0.8, 3.9), (2.4, 3.9), (4.2, 3.9), (6.0, 3.9)], "#2e86de", "U1"),
        ([(0.8, 4.7), (2.4, 4.7), (4.2, 4.7), (6.0, 4.7)], "#d94841", "U2"),
        ([(0.8, 3.1), (2.4, 3.1), (4.2, 3.1), (6.0, 3.1)], "#23a36d", "U3"),
    ]
    for pts, color, label in routes:
        xs, ys = zip(*pts)
        ax_left.plot(xs, ys, color=color, linewidth=2.6, marker="o", markersize=4.5)
        ax_left.text(xs[0] - 0.25, ys[0], label, ha="right", va="center", fontsize=11, color=color, weight="bold")

    ax_left.add_patch(patches.Rectangle((4.55, 2.75), 1.1, 2.5, facecolor="#f8c3c0", edgecolor="#d94841", linewidth=1.4, alpha=0.48))
    ax_left.text(5.10, 5.62, "冲突走廊", ha="center", va="bottom", fontsize=11, color="#d94841", weight="bold")
    ax_left.set_title("空间走廊占用", fontsize=15, weight="bold", pad=8)

    ax_right.set_facecolor("#fcfaf5")
    ax_right.set_xlim(0, 6)
    ax_right.set_ylim(0, 4)
    ax_right.axis("off")

    for c in range(6):
        for r in range(3):
            ax_right.add_patch(
                patches.Rectangle((0.6 + c * 0.82, 0.7 + r * 0.82), 0.68, 0.58, facecolor="#f4efe6", edgecolor="#ddd6ca", linewidth=0.7)
            )
    row_y = {"U1": 2.34, "U2": 1.52, "U3": 0.70}
    before_after = {
        "U1": [(0, "#2e86de"), (1, "#2e86de"), (2, "#2e86de")],
        "U2": [(1, "#d94841"), (2, "#d94841"), (3, "#d94841")],
        "U3": [(0, "#23a36d"), (1, "#23a36d"), (2, "#23a36d")],
    }
    for label, blocks in before_after.items():
        ax_right.text(0.28, row_y[label] + 0.27, label, ha="right", va="center", fontsize=11, weight="bold", color="#243447")
        for col, color in blocks:
            ax_right.add_patch(patches.Rectangle((0.6 + col * 0.82, row_y[label]), 0.68, 0.58, facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.90))

    ax_right.annotate("", xy=(3.78, 1.98), xytext=(3.05, 1.98), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#d94841"))
    ax_right.text(3.42, 2.22, "时隙后移", ha="center", va="bottom", fontsize=10.5, color="#d94841")
    ax_right.set_title("时间槽冲突消解", fontsize=15, weight="bold", pad=8)

    fig.suptitle("图2.5 走廊占用与冲突消解示意图", fontsize=19, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_5_corridor_resolution.png", dpi=240)
    plt.close(fig)


def generate_fig_2_6(data, city):
    example = data["path_example"]
    debug = example.get("debug", {})
    skeleton = np.array(debug.get("skeleton", []), dtype=float)
    smooth = np.array(debug.get("smooth", []), dtype=float)
    repaired = np.array(debug.get("repaired", []), dtype=float)

    focus_idx, focus_center = turning_focus(skeleton)
    global_anchor = skeleton[max(0, focus_idx - 20): min(len(skeleton), focus_idx + 28)]
    x_min = float(global_anchor[:, 0].min() - 70.0)
    x_max = float(global_anchor[:, 0].max() + 70.0)
    z_min = float(global_anchor[:, 2].min() - 70.0)
    z_max = float(global_anchor[:, 2].max() + 70.0)
    focus_buildings = select_buildings_in_bounds(city["buildings"], x_min, x_max, z_min, z_max, pad=20.0)

    global_skel_mask = (
        (skeleton[:, 0] >= x_min - 10.0)
        & (skeleton[:, 0] <= x_max + 10.0)
        & (skeleton[:, 2] >= z_min - 10.0)
        & (skeleton[:, 2] <= z_max + 10.0)
    )
    global_skeleton = skeleton[global_skel_mask]
    global_smooth_mask = (
        (smooth[:, 0] >= x_min - 30.0)
        & (smooth[:, 0] <= x_max + 30.0)
        & (smooth[:, 2] >= z_min - 30.0)
        & (smooth[:, 2] <= z_max + 30.0)
    )
    global_smooth = smooth[global_smooth_mask]
    if len(global_smooth) < 24:
        global_smooth = smooth

    local_segment = skeleton[max(0, focus_idx - 8): min(len(skeleton), focus_idx + 9)]
    lx_min = float(local_segment[:, 0].min() - 18.0)
    lx_max = float(local_segment[:, 0].max() + 18.0)
    lz_min = float(local_segment[:, 2].min() - 18.0)
    lz_max = float(local_segment[:, 2].max() + 18.0)
    local_buildings = select_buildings_in_bounds(city["buildings"], lx_min, lx_max, lz_min, lz_max, pad=10.0)
    local_repaired_mask = (
        (repaired[:, 0] >= lx_min - 20.0)
        & (repaired[:, 0] <= lx_max + 20.0)
        & (repaired[:, 2] >= lz_min - 20.0)
        & (repaired[:, 2] <= lz_max + 20.0)
    )
    local_repaired = repaired[local_repaired_mask]
    if len(local_repaired) < 12:
        local_repaired = repaired
    local_smooth_mask = (
        (smooth[:, 0] >= lx_min - 25.0)
        & (smooth[:, 0] <= lx_max + 25.0)
        & (smooth[:, 2] >= lz_min - 25.0)
        & (smooth[:, 2] <= lz_max + 25.0)
    )
    local_smooth = smooth[local_smooth_mask]
    if len(local_smooth) < 20:
        local_smooth = smooth

    fig = plt.figure(figsize=(14.2, 8.6), facecolor="#ffffff")
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.18, 0.92],
        height_ratios=[1.0, 0.44],
        wspace=0.10,
        hspace=0.16,
    )
    ax_global = fig.add_subplot(outer[0, 0])
    ax_local = fig.add_subplot(outer[0, 1])
    ax_profile = fig.add_subplot(outer[1, 0])
    ax_process = fig.add_subplot(outer[1, 1])

    for ax, buildings, bounds in (
        (ax_global, focus_buildings, (x_min, x_max, z_min, z_max)),
        (ax_local, local_buildings, (lx_min, lx_max, lz_min, lz_max)),
    ):
        draw_roads(ax, city, alpha=0.90)
        draw_buildings_topdown(ax, buildings, mode="height", alpha=0.72, edge="#f5f7fb", linewidth=0.18)
        format_topdown(ax, buildings if buildings else city["buildings"])
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_facecolor("#ffffff")

    ax_global.plot(
        global_skeleton[:, 0],
        global_skeleton[:, 2],
        color="#ef8f1f",
        linewidth=2.1,
        linestyle=(0, (4, 2)),
        marker="o",
        markevery=max(1, len(global_skeleton) // 12),
        markersize=3.8,
        alpha=0.95,
        zorder=7,
        label="A* 离散骨架",
    )
    ax_global.plot(
        global_smooth[:, 0],
        global_smooth[:, 2],
        color="#2e86de",
        linewidth=2.6,
        alpha=0.96,
        zorder=8,
        label="三次B样条轨迹",
    )
    ax_global.scatter(
        global_skeleton[0, 0],
        global_skeleton[0, 2],
        s=86,
        color="#23a36d",
        edgecolor="white",
        linewidth=1.0,
        zorder=9,
    )
    ax_global.scatter(
        global_skeleton[-1, 0],
        global_skeleton[-1, 2],
        s=86,
        marker="s",
        color="#d94841",
        edgecolor="white",
        linewidth=1.0,
        zorder=9,
    )
    ax_global.set_title("全局骨架路径与连续航迹", fontsize=16, weight="bold", pad=12)

    ax_local.plot(
        local_repaired[:, 0],
        local_repaired[:, 2],
        color="#a8c3ef",
        linewidth=16.0,
        alpha=0.35,
        solid_capstyle="round",
        zorder=4,
        label="安全走廊",
    )
    ax_local.plot(
        local_segment[:, 0],
        local_segment[:, 2],
        color="#ef8f1f",
        linewidth=1.8,
        linestyle=(0, (4, 2)),
        alpha=0.95,
        zorder=7,
        label="骨架折线",
    )
    ax_local.plot(
        local_repaired[:, 0],
        local_repaired[:, 2],
        color="#6c7a92",
        linewidth=1.25,
        linestyle=(0, (2, 2)),
        alpha=0.88,
        zorder=8,
        label="局部重连控制点",
    )
    ax_local.scatter(
        local_repaired[:: max(1, len(local_repaired) // 12), 0],
        local_repaired[:: max(1, len(local_repaired) // 12), 2],
        s=24,
        color="#4b5563",
        edgecolor="white",
        linewidth=0.55,
        zorder=9,
    )
    ax_local.plot(
        local_smooth[:, 0],
        local_smooth[:, 2],
        color="#2e86de",
        linewidth=2.7,
        alpha=0.98,
        zorder=10,
        label="B样条平滑结果",
    )
    ax_local.scatter(
        [focus_center[0]],
        [focus_center[2]],
        s=62,
        color="#d94841",
        edgecolor="white",
        linewidth=0.8,
        zorder=11,
    )
    ax_local.set_title("局部重连与走廊内B样条重定形", fontsize=16, weight="bold", pad=12)

    sk_dist = cumulative_distance(global_skeleton)
    sm_dist = cumulative_distance(global_smooth)
    ax_profile.set_facecolor("#ffffff")
    ax_profile.plot(sk_dist, global_skeleton[:, 1], color="#ef8f1f", linewidth=1.9, linestyle=(0, (4, 2)), label="骨架高度剖面")
    ax_profile.plot(sm_dist, global_smooth[:, 1], color="#2e86de", linewidth=2.3, label="B样条高度剖面")
    ax_profile.fill_between(sm_dist, global_smooth[:, 1], color="#d9e8ff", alpha=0.28)
    ax_profile.set_title("高度剖面与爬升连续性", fontsize=15, weight="bold", pad=10)
    ax_profile.set_xlabel("沿程距离 / m", fontsize=11)
    ax_profile.set_ylabel("高度 / m", fontsize=11)
    ax_profile.grid(alpha=0.18)
    for spine in ax_profile.spines.values():
        spine.set_visible(False)
    ax_profile.legend(loc="upper right", frameon=False, fontsize=10)

    ax_process.axis("off")
    ax_process.set_facecolor("#ffffff")
    cards = [
        ((0.04, 0.58), "#edf4ff", "#5b8def", "1", "A*骨架搜索", "离散可行\n骨架"),
        ((0.36, 0.58), "#f5f8ff", "#7b8da8", "2", "局部走廊修复", "控制点\n局部重连"),
        ((0.68, 0.58), "#eef8f4", "#2e86de", "3", "三次B样条优化", "连续曲线\n可执行航迹"),
    ]
    for (x, y), face, accent, num, title, subtitle in cards:
        card = patches.FancyBboxPatch(
            (x, y),
            0.27,
            0.30,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=face,
            edgecolor="#d7deea",
            linewidth=1.0,
            transform=ax_process.transAxes,
        )
        ax_process.add_patch(card)
        ax_process.add_patch(
            patches.Circle(
                (x + 0.045, y + 0.23),
                0.030,
                transform=ax_process.transAxes,
                facecolor=accent,
                edgecolor="none",
            )
        )
        ax_process.text(x + 0.045, y + 0.23, num, ha="center", va="center", fontsize=11, color="white", weight="bold", transform=ax_process.transAxes)
        ax_process.text(x + 0.085, y + 0.245, title, ha="left", va="center", fontsize=11.0, color="#243447", weight="bold", transform=ax_process.transAxes)
        ax_process.text(x + 0.135, y + 0.105, subtitle, ha="center", va="center", fontsize=9.5, color="#4a5568", transform=ax_process.transAxes)

    ax_process.plot([0.09, 0.16, 0.22], [0.20, 0.30, 0.18], color="#ef8f1f", linewidth=2.0, transform=ax_process.transAxes, clip_on=False)
    ax_process.plot([0.41, 0.48, 0.55], [0.18, 0.31, 0.21], color="#7b8da8", linewidth=1.5, linestyle=(0, (2, 2)), transform=ax_process.transAxes, clip_on=False)
    ax_process.scatter([0.41, 0.48, 0.55], [0.18, 0.31, 0.21], s=18, color="#7b8da8", edgecolor="white", linewidth=0.5, transform=ax_process.transAxes, zorder=3)
    t = np.linspace(0.0, 1.0, 40)
    ax_process.plot(0.72 + 0.18 * t, 0.18 + 0.10 * np.sin(np.pi * t) * np.exp(-0.2 * t), color="#2e86de", linewidth=2.3, transform=ax_process.transAxes, clip_on=False)
    ax_process.text(
        0.02,
        0.06,
        "图中采用同一条真实规划样本，依次展示全局骨架、局部走廊修复与B样条连续化结果。",
        fontsize=10.0,
        color="#5b677a",
        transform=ax_process.transAxes,
    )

    legend_handles = [
        Line2D([0], [0], color="#ef8f1f", lw=2.0, linestyle=(0, (4, 2)), marker="o", markersize=4, label="A*离散骨架"),
        Line2D([0], [0], color="#6c7a92", lw=1.4, linestyle=(0, (2, 2)), marker="o", markersize=4, label="局部重连控制点"),
        Line2D([0], [0], color="#a8c3ef", lw=8.0, alpha=0.45, label="安全走廊"),
        Line2D([0], [0], color="#2e86de", lw=2.4, label="三次B样条航迹"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#23a36d", markeredgecolor="white", markersize=8, label="起点"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#d94841", markeredgecolor="white", markersize=7, label="终点"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.50, 0.01), ncol=6, frameon=False, fontsize=10.3)
    fig.suptitle("图2.6 路径规划与 B 样条轨迹优化示意图", fontsize=20, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_6_astar_bspline.png", dpi=240)
    plt.close(fig)


def generate_fig_2_7(data):
    lookup = {row["algorithm"]: row for row in data["main_results"]}
    ordered_names = [name for name in ALGO_ORDER if name in lookup]
    rows = [lookup[name] for name in ordered_names]

    completion = np.array([float(row["weighted_completion_rate"]) for row in rows], dtype=float)
    ontime = np.array([float(row["time_window_rate"]) for row in rows], dtype=float)
    conflicts = np.array([float(row["corridor_conflicts"]) for row in rows], dtype=float)
    runtime = np.array([float(row["runtime_ms"]) for row in rows], dtype=float)
    energy = np.array([max(0.0, float(row["utility_energy_ratio"])) for row in rows], dtype=float)

    metric_matrix = np.column_stack([
        robust_normalize(completion, higher_better=True),
        robust_normalize(ontime, higher_better=True),
        robust_normalize(conflicts, higher_better=False, log_scale=True),
        robust_normalize(runtime, higher_better=False, log_scale=True),
        robust_normalize(energy, higher_better=True, log_scale=True),
    ])
    metric_labels = ["加权\n完成率", "准时率", "冲突\n控制", "时延\n效率", "能效\n表现"]
    row_labels = [algo_label(name) for name in ordered_names]
    annotations = []
    for row in rows:
        annotations.append([
            f"{row['weighted_completion_rate']:.3f}",
            f"{row['time_window_rate']:.3f}",
            f"{int(row['corridor_conflicts'])}次",
            f"{row['runtime_ms']:.0f}ms",
            f"{max(0.0, row['utility_energy_ratio']):.2f}",
        ])

    fig = plt.figure(figsize=(14.6, 8.5), facecolor="#ffffff")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.16, 0.84], wspace=0.16)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    heat_cmap = LinearSegmentedColormap.from_list(
        "rb_metric",
        ["#dce9ff", "#f6f9ff", "#ffffff", "#f6c9c5", "#d94841"],
    )
    ax_left.set_facecolor("#ffffff")
    ax_left.imshow(metric_matrix, cmap=heat_cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax_left.set_xticks(np.arange(len(metric_labels)))
    ax_left.set_xticklabels(metric_labels, fontsize=10.5)
    ax_left.set_yticks(np.arange(len(row_labels)))
    ax_left.set_yticklabels(row_labels, fontsize=10.4)
    ax_left.set_xlim(-0.96, len(metric_labels) - 0.5)
    ax_left.set_xticks(np.arange(-0.5, metric_matrix.shape[1], 1), minor=True)
    ax_left.set_yticks(np.arange(-0.5, metric_matrix.shape[0], 1), minor=True)
    ax_left.grid(which="minor", color="white", linewidth=1.15)
    ax_left.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.set_title("主场景多指标矩阵", fontsize=16, weight="bold", pad=12)
    for ridx, name in enumerate(ordered_names):
        fam = algo_family(name)
        color = FAMILY_META.get(fam, {"color": "#9aa5b1"})["color"]
        ax_left.add_patch(
            patches.Rectangle(
                (-0.90, ridx - 0.46),
                0.18,
                0.92,
                facecolor=lighten(color, 0.18),
                edgecolor="white",
                linewidth=0.8,
            )
        )
    stc_idx = ordered_names.index("STC-RCBBA")
    ax_left.add_patch(
        patches.Rectangle(
            (-0.96, stc_idx - 0.50),
            len(metric_labels) + 0.46,
            1.00,
            fill=False,
            edgecolor="#d94841",
            linewidth=2.1,
        )
    )
    for col in range(metric_matrix.shape[1]):
        best_val = float(np.max(metric_matrix[:, col]))
        best_rows = np.where(np.isclose(metric_matrix[:, col], best_val, atol=1e-6))[0]
        for ridx in best_rows:
            ax_left.add_patch(
                patches.Rectangle(
                    (col - 0.50, ridx - 0.50),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#6f92d6",
                    linewidth=1.5,
                )
            )
    for ridx in range(metric_matrix.shape[0]):
        for col in range(metric_matrix.shape[1]):
            color = "#ffffff" if metric_matrix[ridx, col] > 0.60 else "#3b4557"
            ax_left.text(col, ridx, annotations[ridx][col], ha="center", va="center", fontsize=9.0, color=color)

    family_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=meta["color"], markeredgecolor="none", label=family)
        for family, meta in FAMILY_META.items()
    ]
    ax_left.legend(handles=family_handles, loc="lower center", bbox_to_anchor=(0.50, -0.16), ncol=5, frameon=False, fontsize=9.6)

    ax_right.set_facecolor("#ffffff")
    max_conf = int(np.max(conflicts))
    ax_right.axvspan(0.94, 1.02, color="#fcecec", alpha=0.92, zorder=0)
    ax_right.axhspan(-0.5, 5.5, color="#edf4ff", alpha=0.92, zorder=0)
    bubble_sizes = 85.0 + 20.0 * np.sqrt(np.maximum(runtime, 1.0))
    for idx, row in enumerate(rows):
        family = algo_family(row["algorithm"])
        color = FAMILY_META.get(family, {"color": "#7f8c8d"})["color"]
        lw = 2.0 if row["algorithm"] == "STC-RCBBA" else 0.9
        edge = "#d94841" if row["algorithm"] == "STC-RCBBA" else "white"
        ax_right.scatter(
            row["weighted_completion_rate"],
            row["corridor_conflicts"],
            s=bubble_sizes[idx],
            color=color,
            edgecolor=edge,
            linewidth=lw,
            alpha=0.92,
            zorder=4,
        )
    label_offsets = {
        "STC-RCBBA": (0.006, 0.9),
        "Greedy": (-0.065, -0.9),
        "Auction": (-0.075, 0.8),
        "Hungarian": (-0.090, -0.8),
        "Genetic": (0.008, 0.9),
        "ACO": (-0.050, -0.8),
        "Market": (0.010, 0.7),
        "原始CBBA": (0.010, 0.6),
    }
    for row in rows:
        if row["algorithm"] not in label_offsets:
            continue
        dx, dy = label_offsets[row["algorithm"]]
        ax_right.text(
            row["weighted_completion_rate"] + dx,
            row["corridor_conflicts"] + dy,
            algo_label(row["algorithm"]),
            fontsize=9.4,
            color="#263445",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#ffffff", edgecolor="none", alpha=0.88),
        )
    ax_right.set_xlim(0.15, 1.03)
    ax_right.set_ylim(max_conf + 3, -1.5)
    ax_right.set_xlabel("加权完成率", fontsize=11.5)
    ax_right.set_ylabel("走廊冲突次数（越少越优）", fontsize=11.5)
    ax_right.grid(alpha=0.16)
    for spine in ax_right.spines.values():
        spine.set_visible(False)
    ax_right.set_title("任务-冲突-时延权衡分布", fontsize=16, weight="bold", pad=12)
    ax_right.text(
        0.955,
        2.0,
        "优选区",
        fontsize=10.2,
        color="#2d5fb3",
        weight="bold",
    )
    family_legend = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8.5, markerfacecolor=meta["color"], markeredgecolor="white", label=family)
        for family, meta in FAMILY_META.items()
    ]
    ax_right.legend(handles=family_legend, loc="lower center", bbox_to_anchor=(0.50, -0.14), ncol=3, frameon=False, fontsize=9.5)

    fig.suptitle("图2.7 多算法综合对比图", fontsize=20, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_7_algorithm_comparison.png", dpi=240)
    plt.close(fig)


def generate_fig_2_8(data):
    lookup = {row["variant"]: row for row in copy.deepcopy(data["ablation_results"])}
    variant_order = [
        "STC-RCBBA",
        "去掉优先级紧迫项",
        "去掉通信鲁棒共识",
        "去掉走廊冲突代价",
        "去掉B样条重定形",
    ]
    rows = [lookup[name] for name in variant_order if name in lookup]
    row_labels = [algo_label(row["variant"]) for row in rows]
    metric_labels = ["加权完成率", "准时率", "能效表现", "冲突抑制", "轨迹友好度"]
    keys = [
        "weighted_completion_rate",
        "time_window_rate",
        "utility_energy_ratio",
        "conflict_suppression",
        "smoothness_index",
    ]

    plot_values = np.column_stack([
        np.array([float(row["weighted_completion_rate"]) for row in rows], dtype=float),
        np.array([float(row["time_window_rate"]) for row in rows], dtype=float),
        np.array([max(0.0, float(row["utility_energy_ratio"])) for row in rows], dtype=float),
        np.array([float(row["conflict_suppression"]) for row in rows], dtype=float),
        np.array([float(row["smoothness_index"]) for row in rows], dtype=float),
    ])
    abs_norm = np.column_stack([
        robust_normalize(plot_values[:, 0], higher_better=True),
        robust_normalize(plot_values[:, 1], higher_better=True),
        robust_normalize(plot_values[:, 2], higher_better=True, log_scale=True),
        robust_normalize(plot_values[:, 3], higher_better=True),
        robust_normalize(plot_values[:, 4], higher_better=True),
    ])
    delta = abs_norm - abs_norm[0]
    delta_max = float(np.max(np.abs(delta[1:]))) if len(rows) > 1 else 1.0
    delta_max = max(delta_max, 0.05)

    abs_annotations = []
    for row in rows:
        abs_annotations.append([
            f"{row['weighted_completion_rate']:.3f}",
            f"{row['time_window_rate']:.3f}",
            f"{max(0.0, row['utility_energy_ratio']):.2f}",
            f"{row['conflict_suppression']:.3f}",
            f"{row['smoothness_index']:.3f}",
        ])
    delta_annotations = [[f"{value:+.2f}" for value in line] for line in delta]

    metric_sensitivity = np.mean(np.abs(delta[1:]), axis=0, keepdims=True)
    metric_annotations = [[f"{value:.2f}" for value in metric_sensitivity[0]]]
    variant_impact = np.column_stack([
        np.mean(np.abs(delta[1:]), axis=1),
        np.max(np.abs(delta[1:]), axis=1),
    ])
    variant_annotations = [[f"{value:.2f}" for value in line] for line in variant_impact]

    warm_cmap = LinearSegmentedColormap.from_list(
        "ablation_rb_abs",
        ["#dce9ff", "#f6f9ff", "#ffffff", "#f6cbc7", "#d94841"],
    )
    delta_cmap = LinearSegmentedColormap.from_list(
        "ablation_delta",
        ["#4567d6", "#eef2ff", "#fff7f2", "#f4b2a9", "#d94545"],
    )

    fig = plt.figure(figsize=(14.1, 8.9), facecolor="#ffffff")
    outer = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.18, 0.92], height_ratios=[1.0, 0.58], wspace=0.16, hspace=0.22)
    ax_abs = fig.add_subplot(outer[0, 0])
    ax_delta = fig.add_subplot(outer[0, 1])
    ax_metric = fig.add_subplot(outer[1, 0])
    ax_variant = fig.add_subplot(outer[1, 1])

    draw_matrix_panel(
        ax_abs,
        abs_norm,
        row_labels,
        metric_labels,
        "完整模型与各消融变体的绝对表现",
        warm_cmap,
        annotations=abs_annotations,
        vmin=0.0,
        vmax=1.0,
        title_size=15,
    )
    draw_matrix_panel(
        ax_delta,
        delta,
        row_labels,
        metric_labels,
        "相对完整模型的指标偏移",
        delta_cmap,
        annotations=delta_annotations,
        norm=TwoSlopeNorm(vmin=-delta_max, vcenter=0.0, vmax=delta_max),
        title_size=15,
    )
    draw_matrix_panel(
        ax_metric,
        metric_sensitivity,
        ["平均敏感度"],
        metric_labels,
        "各指标对模块删减的敏感度",
        warm_cmap,
        annotations=metric_annotations,
        vmin=0.0,
        vmax=max(0.10, float(metric_sensitivity.max())),
        title_size=14,
    )
    draw_matrix_panel(
        ax_variant,
        variant_impact,
        [algo_label(row["variant"]) for row in rows[1:]],
        ["平均偏移", "最大偏移"],
        "各模块删减的整体影响",
        warm_cmap,
        annotations=variant_annotations,
        vmin=0.0,
        vmax=max(0.10, float(variant_impact.max())),
        title_size=14,
    )
    for ax in (ax_abs, ax_delta):
        ax.add_patch(
            patches.Rectangle(
                (-0.5, -0.5),
                len(metric_labels),
                1.0,
                fill=False,
                edgecolor="#d94841",
                linewidth=2.1,
            )
        )
    fig.text(
        0.50,
        0.04,
        "红色表示更强或相对提升，蓝色表示更弱或相对下降；红框行为完整模型基线。",
        fontsize=9.8,
        color="#5b677a",
        ha="center",
    )
    fig.suptitle("图2.8 消融实验热力图", fontsize=20, weight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_2_8_ablation_radar.png", dpi=240)
    plt.close(fig)


def generate_fig_2_9(data, city):
    stc, routes, drone_map, task_map, summary = get_stc_assignment_view(data)
    write_assignment_snapshot(stc, routes, summary)
    display_routes = build_display_routes(routes)

    x_min, x_max, z_min, z_max = route_focus_bounds(display_routes, pad=74.0, points_key="display_points_2d")
    focus_buildings = select_buildings_in_bounds(city["buildings"], x_min, x_max, z_min, z_max)

    fig = plt.figure(figsize=(14.2, 8.4), facecolor="#ffffff")
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.58, 0.82],
        height_ratios=[1.0, 0.12],
        wspace=0.08,
        hspace=0.03,
    )
    ax_map = fig.add_subplot(outer[0, 0])
    side = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 1], height_ratios=[0.45, 0.20, 0.35], hspace=0.62)
    ax_summary = fig.add_subplot(side[0, 0])
    ax_drone = fig.add_subplot(side[1, 0])
    ax_task = fig.add_subplot(side[2, 0])
    ax_leg = fig.add_subplot(outer[1, :])

    draw_roads(ax_map, city, alpha=0.88)
    draw_buildings_topdown(ax_map, focus_buildings, mode="height", alpha=0.42, edge="#ffffff", linewidth=0.22)
    format_topdown(ax_map, focus_buildings)
    ax_map.set_facecolor("#ffffff")
    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(z_min, z_max)
    ax_map.set_title("主实验已分配链路二维总览", fontsize=17.6, weight="bold", pad=16)

    seen_drone = set()
    for row in display_routes:
        pts = row["display_points_2d"]
        pickup = row["pickup_point"]
        delivery = row["delivery_point"]
        style = DRONE_STYLE[row["drone_type"]]
        ax_map.plot(pts[:, 0], pts[:, 2], color="#ffffff", linewidth=4.4, alpha=0.95, zorder=6)
        ax_map.plot(pts[:, 0], pts[:, 2], color=style["color"], linewidth=2.3, alpha=0.84, zorder=7)
        ax_map.scatter(pickup[0], pickup[2], s=38, color="#f39c12", edgecolor="white", linewidth=0.9, zorder=8)
        ax_map.scatter(delivery[0], delivery[2], s=36, marker="s", color="#2e86de", edgecolor="white", linewidth=0.9, zorder=8)
        if row["drone_id"] not in seen_drone:
            ax_map.scatter(
                row["points"][0, 0],
                row["points"][0, 2],
                s=118,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=1.1,
                zorder=9,
            )
            seen_drone.add(row["drone_id"])

    ax_summary.axis("off")
    ax_summary.set_facecolor("#ffffff")
    ax_summary.text(0.02, 0.98, "分配结果摘要", fontsize=18, weight="bold", color="#1f2937", va="top")
    summary_lines = [
        ("已完成任务", f"{summary['completed_tasks']} / {summary['total_tasks']}"),
        ("加权完成率", f"{summary['weighted_completion_rate'] * 100:.1f}%"),
        ("时间窗满足率", f"{summary['time_window_rate'] * 100:.1f}%"),
        ("走廊冲突次数", f"{summary['corridor_conflicts']}"),
        ("展示链路样本", f"{summary['sample_chain_count']} 条"),
        ("样本活跃无人机", f"{summary['sample_active_drone_count']} 架"),
        ("求解时延", f"{summary['runtime_ms']:.1f} ms"),
    ]
    for idx, (label, value) in enumerate(summary_lines):
        y = 0.74 - idx * 0.095
        ax_summary.text(0.03, y, label, fontsize=12.6, color="#4a5568")
        ax_summary.text(0.97, y, value, fontsize=13.4, color="#111827", ha="right", weight="bold")

    drone_type_counts = Counter(row["drone_type"] for row in routes)
    drone_labels = ["重型机", "标准机", "轻型机"]
    drone_values = [drone_type_counts.get(key, 0) for key in ["heavy", "standard", "light"]]
    drone_colors = [DRONE_STYLE[key]["color"] for key in ["heavy", "standard", "light"]]
    draw_count_panel(ax_drone, "样本链路按机型分布", drone_labels, drone_values, drone_colors, xmax=max(drone_values) if drone_values else 1)
    polish_compact_panel(ax_drone, "样本链路按机型分布")

    task_type_counts = Counter(row["task_type"] for row in routes)
    task_keys = [key for key, _ in sorted(task_type_counts.items(), key=lambda item: (-item[1], item[0]))]
    task_labels = [TASK_TYPE_CN.get(key, key) for key in task_keys]
    task_values = [task_type_counts[key] for key in task_keys]
    task_colors = [TASK_TYPE_COLOR.get(key, "#7f8c8d") for key in task_keys]
    draw_count_panel(ax_task, "样本任务按业务分布", task_labels, task_values, task_colors, xmax=max(task_values) if task_values else 1)
    polish_compact_panel(ax_task, "样本任务按业务分布")

    ax_leg.axis("off")
    legend_handles = [
        Line2D([0], [0], color=DRONE_STYLE["heavy"]["color"], lw=3, label="重型机执行链路"),
        Line2D([0], [0], color=DRONE_STYLE["standard"]["color"], lw=3, label="标准机执行链路"),
        Line2D([0], [0], color=DRONE_STYLE["light"]["color"], lw=3, label="轻型机执行链路"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#f39c12", markeredgecolor="white", markersize=8, label="取件点"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#2e86de", markeredgecolor="white", markersize=7, label="送达点"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#6b7280", markeredgecolor="white", markersize=8, label="无人机起点"),
    ]
    ax_leg.legend(handles=legend_handles, loc="center", ncol=6, frameon=False, fontsize=11.3, columnspacing=1.7, handlelength=2.0)

    fig.suptitle("图2.9 STC-RCBBA任务分配结果二维展示图", fontsize=20.5, weight="bold", y=0.992)
    fig.savefig(FIG_DIR / "fig_2_9_assignment_2d.png", dpi=240)
    plt.close(fig)


def generate_fig_2_10(data, city):
    _, routes, _, _, summary = get_stc_assignment_view(data)
    display_routes = build_display_routes(routes)
    x_min, x_max, z_min, z_max = route_focus_bounds(display_routes, pad=64.0, points_key="display_points_3d")
    focus_buildings = select_buildings_in_bounds(city["buildings"], x_min, x_max, z_min, z_max)

    fig = plt.figure(figsize=(14.2, 8.4), facecolor="#ffffff")
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.58, 0.84],
        height_ratios=[1.0, 0.12],
        wspace=0.06,
        hspace=0.03,
    )
    ax3d = fig.add_subplot(outer[0, 0], projection="3d")
    side = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 1], height_ratios=[0.42, 0.24, 0.34], hspace=0.34)
    ax_summary = fig.add_subplot(side[0, 0])
    ax_layer = fig.add_subplot(side[1, 0])
    ax_drone = fig.add_subplot(side[2, 0])
    ax_leg = fig.add_subplot(outer[1, :])

    draw_city_oblique_region(ax3d, city, focus_buildings, x_min, x_max, z_min, z_max, bg_face="#ffffff")

    seen_drone = set()
    for row in display_routes:
        pts = row["display_points_3d"]
        pickup = row["pickup_point"]
        delivery = row["delivery_point"]
        style = DRONE_STYLE[row["drone_type"]]
        ax3d.plot(pts[:, 0], pts[:, 2], pts[:, 1], color="#ffffff", linewidth=4.0, alpha=0.96, zorder=6)
        ax3d.plot(pts[:, 0], pts[:, 2], pts[:, 1], color=style["color"], linewidth=2.5, alpha=0.92, zorder=7)
        ax3d.scatter(pickup[0], pickup[2], pickup[1], s=24, color="#f39c12", edgecolor="white", linewidth=0.55, depthshade=False, zorder=8)
        ax3d.scatter(delivery[0], delivery[2], delivery[1], s=24, marker="s", color="#2e86de", edgecolor="white", linewidth=0.55, depthshade=False, zorder=8)
        if row["drone_id"] not in seen_drone:
            ax3d.scatter(
                row["points"][0, 0],
                row["points"][0, 2],
                row["points"][0, 1],
                s=64,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.8,
                depthshade=False,
                zorder=9,
            )
            seen_drone.add(row["drone_id"])

    ax_summary.axis("off")
    ax_summary.set_facecolor("#ffffff")
    ax_summary.text(0.02, 0.98, "三维执行结果摘要", fontsize=18, weight="bold", color="#1f2937", va="top")
    summary_lines = [
        ("展示链路样本", f"{summary['sample_chain_count']} 条"),
        ("样本活跃无人机", f"{summary['sample_active_drone_count']} 架"),
        ("平均完成时长", f"{summary['avg_completion_time_s']:.1f} s"),
        ("加权完成率", f"{summary['weighted_completion_rate'] * 100:.1f}%"),
        ("准时率", f"{summary['time_window_rate'] * 100:.1f}%"),
        ("走廊冲突", f"{summary['corridor_conflicts']}"),
    ]
    for idx, (label, value) in enumerate(summary_lines):
        y = 0.74 - idx * 0.102
        ax_summary.text(0.03, y, label, fontsize=12.6, color="#4a5568")
        ax_summary.text(0.97, y, value, fontsize=13.2, color="#111827", ha="right", weight="bold")

    layer_counts = Counter(row["layer"] for row in routes)
    layer_keys = [key for key, _ in sorted(layer_counts.items(), key=lambda item: (-item[1], item[0]))]
    layer_labels = [LAYER_CN.get(key, key) for key in layer_keys]
    layer_values = [layer_counts[key] for key in layer_keys]
    layer_colors = ["#4f7dd1", "#16a085", "#d94841"][: len(layer_values)]
    draw_count_panel(ax_layer, "样本链路按空域层分布", layer_labels, layer_values, layer_colors, xmax=max(layer_values) if layer_values else 1)
    polish_compact_panel(ax_layer, "样本链路按空域层分布")

    drone_counts = Counter(row["drone_id"] for row in routes)
    top_drone_items = sorted(drone_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
    drone_labels = [item[0] for item in top_drone_items]
    drone_values = [item[1] for item in top_drone_items]
    drone_colors = ["#5b7cfa"] * len(top_drone_items)
    draw_count_panel(ax_drone, "样本链路按无人机归属", drone_labels, drone_values, drone_colors, xmax=max(drone_values) if drone_values else 1)
    polish_compact_panel(ax_drone, "样本链路按无人机归属")

    ax_leg.axis("off")
    legend_handles = [
        Line2D([0], [0], color=DRONE_STYLE["heavy"]["color"], lw=3, label="重型机执行链路"),
        Line2D([0], [0], color=DRONE_STYLE["standard"]["color"], lw=3, label="标准机执行链路"),
        Line2D([0], [0], color=DRONE_STYLE["light"]["color"], lw=3, label="轻型机执行链路"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#f39c12", markeredgecolor="white", markersize=8, label="取件点"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#2e86de", markeredgecolor="white", markersize=7, label="送达点"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#6b7280", markeredgecolor="white", markersize=8, label="无人机起点"),
    ]
    ax_leg.legend(handles=legend_handles, loc="center", ncol=6, frameon=False, fontsize=11.3, columnspacing=1.7, handlelength=2.0)

    fig.suptitle("图2.10 STC-RCBBA任务分配结果三维展示图", fontsize=20.5, weight="bold", y=0.992)
    fig.savefig(FIG_DIR / "fig_2_10_assignment_3d.png", dpi=240)
    plt.close(fig)


def write_manifest():
    manifest = {
        "图2.1": {"file": str(FIG_DIR / "fig_2_1_dense_city_scene.png"), "insert_after": "2.2.1 节末"},
        "图2.2": {"file": str(FIG_DIR / "fig_2_2_heterogeneous_hotspots.png"), "insert_after": "2.2.2 机群建模后"},
        "图2.3": {"file": str(FIG_DIR / "fig_2_3_comm_occlusion.png"), "insert_after": "2.2.2 通信建模后"},
        "图2.4": {"file": str(FIG_DIR / "fig_2_4_stc_rcbba_flow.png"), "insert_after": "2.2.5 算法流程后"},
        "图2.5": {"file": str(FIG_DIR / "fig_2_5_corridor_resolution.png"), "insert_after": "2.2.5 走廊机制说明后"},
        "图2.6": {"file": str(FIG_DIR / "fig_2_6_astar_bspline.png"), "insert_after": "2.2.6 节末"},
        "图2.7": {"file": str(FIG_DIR / "fig_2_7_algorithm_comparison.png"), "insert_after": "2.2.7 主算法对比后"},
        "图2.8": {"file": str(FIG_DIR / "fig_2_8_ablation_radar.png"), "insert_after": "2.2.7 消融实验后"},
        "图2.9": {"file": str(FIG_DIR / "fig_2_9_assignment_2d.png"), "insert_after": "2.2.7 图2.8后"},
        "图2.10": {"file": str(FIG_DIR / "fig_2_10_assignment_3d.png"), "insert_after": "2.2.7 图2.9后"},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data, city = load_data()

    generate_fig_2_1(data, city)
    generate_fig_2_2(data, city)
    generate_fig_2_3(data, city)
    generate_fig_2_4()
    generate_fig_2_5()
    generate_fig_2_6(data, city)
    generate_fig_2_7(data)
    generate_fig_2_8(data)
    generate_fig_2_9(data, city)
    generate_fig_2_10(data, city)
    write_manifest()

    print("Generated figures:")
    for name in [
        "fig_2_1_dense_city_scene.png",
        "fig_2_2_heterogeneous_hotspots.png",
        "fig_2_3_comm_occlusion.png",
        "fig_2_4_stc_rcbba_flow.png",
        "fig_2_5_corridor_resolution.png",
        "fig_2_6_astar_bspline.png",
        "fig_2_7_algorithm_comparison.png",
        "fig_2_8_ablation_radar.png",
        "fig_2_9_assignment_2d.png",
        "fig_2_10_assignment_3d.png",
    ]:
        print(f"  - {FIG_DIR / name}")


if __name__ == "__main__":
    main()
