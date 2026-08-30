from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "thesis" / "figures"
OUT_PATH = FIG_DIR / "fig_2_6_astar_bspline_intuitive.png"


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def sample_cubic_bspline(points: np.ndarray, samples_per_seg: int = 30) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    if len(pts) < 4:
        return pts.copy()

    padded = np.vstack([pts[0], pts[0], pts, pts[-1], pts[-1]])
    samples = [pts[0].copy()]
    for idx in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[idx - 1], padded[idx], padded[idx + 1], padded[idx + 2]
        for s in range(1, samples_per_seg + 1):
            t = s / samples_per_seg
            t2 = t * t
            t3 = t2 * t
            basis = np.array(
                [
                    (-t3 + 3 * t2 - 3 * t + 1) / 6.0,
                    (3 * t3 - 6 * t2 + 4) / 6.0,
                    (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0,
                    t3 / 6.0,
                ],
                dtype=float,
            )
            point = basis[0] * p0 + basis[1] * p1 + basis[2] * p2 + basis[3] * p3
            samples.append(point)
    samples[-1] = pts[-1].copy()
    return np.asarray(samples, dtype=float)


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    dist = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        dist.append(dist[-1] + float(np.linalg.norm(b - a)))
    return np.asarray(dist, dtype=float)


def draw_buildings(ax, rects, face="#d8d4cf", edge="#f7f5f2", alpha=1.0) -> None:
    for x, y, w, h in rects:
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w,
                h,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
                alpha=alpha,
                zorder=1,
            )
        )


def add_step_badge(ax, number: str, title: str, subtitle: str, color: str) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.86),
            0.50,
            0.12,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor="#f8fbff",
            edgecolor="#d4deea",
            linewidth=1.0,
            transform=ax.transAxes,
            zorder=20,
        )
    )
    ax.add_patch(
        patches.Circle(
            (0.07, 0.92),
            0.035,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            zorder=21,
        )
    )
    ax.text(0.07, 0.92, number, ha="center", va="center", color="white", fontsize=11, weight="bold", transform=ax.transAxes, zorder=22)
    ax.text(0.13, 0.935, title, ha="left", va="center", fontsize=11.5, weight="bold", color="#233548", transform=ax.transAxes, zorder=22)
    ax.text(0.13, 0.895, subtitle, ha="left", va="center", fontsize=9.6, color="#5b677a", transform=ax.transAxes, zorder=22)


def style_planar_axis(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#ffffff")


def main() -> None:
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.0, 7.9), facecolor="#ffffff")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 1.18], width_ratios=[1.28, 1.0], hspace=0.18, wspace=0.12)

    ax_global = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[1, :])

    for ax in (ax_global, ax_zoom):
        style_planar_axis(ax)

    city_rects = [
        (0.8, 1.0, 1.4, 1.5),
        (2.7, 0.8, 1.5, 1.6),
        (5.0, 0.9, 1.5, 1.5),
        (7.1, 1.0, 1.5, 1.4),
        (1.2, 3.7, 1.3, 1.5),
        (3.4, 3.3, 1.8, 1.9),
        (6.0, 3.7, 1.4, 1.4),
        (1.8, 6.3, 1.4, 1.4),
        (4.6, 6.1, 1.8, 1.8),
        (7.5, 6.6, 1.1, 1.1),
    ]
    draw_buildings(ax_global, city_rects, face="#d8d2ca")

    skeleton = np.array(
        [
            [0.9, 1.6],
            [1.6, 3.0],
            [2.2, 5.8],
            [3.8, 5.8],
            [4.8, 7.2],
            [6.8, 7.2],
            [7.8, 8.0],
            [9.0, 8.5],
        ],
        dtype=float,
    )
    smooth_global = sample_cubic_bspline(skeleton, samples_per_seg=32)

    ax_global.plot(smooth_global[:, 0], smooth_global[:, 1], color="#a8c3ef", linewidth=18, alpha=0.20, solid_capstyle="round", zorder=4)
    ax_global.plot(skeleton[:, 0], skeleton[:, 1], color="#ef8f1f", linewidth=2.3, linestyle=(0, (4, 2)), zorder=8)
    ax_global.scatter(skeleton[:, 0], skeleton[:, 1], s=30, color="#ef8f1f", edgecolor="white", linewidth=0.6, zorder=9)
    ax_global.plot(smooth_global[:, 0], smooth_global[:, 1], color="#2e86de", linewidth=3.0, zorder=10)
    ax_global.scatter(skeleton[0, 0], skeleton[0, 1], s=92, color="#22a06b", edgecolor="white", linewidth=1.0, zorder=11)
    ax_global.scatter(skeleton[-1, 0], skeleton[-1, 1], s=92, marker="s", color="#d94841", edgecolor="white", linewidth=1.0, zorder=11)

    zoom_box = patches.Rectangle((1.4, 5.2), 5.8, 2.7, fill=False, edgecolor="#7b8da8", linewidth=1.5, linestyle=(0, (4, 3)), zorder=12)
    ax_global.add_patch(zoom_box)
    ax_global.set_title("A* 骨架路径与三次 B 样条优化结果", fontsize=18, weight="bold", pad=12)

    local_rects = [
        (0.9, 0.9, 0.8, 7.4),
        (2.8, 0.8, 1.8, 3.7),
        (2.9, 7.7, 1.7, 1.0),
        (6.1, 5.1, 1.7, 1.1),
        (7.3, 7.6, 1.2, 1.0),
    ]
    corridor_poly = np.array(
        [
            [2.0, 1.0],
            [2.0, 5.8],
            [3.2, 7.0],
            [6.8, 7.0],
            [8.8, 8.0],
        ],
        dtype=float,
    )
    control_points = np.array(
        [
            [2.0, 1.0],
            [2.0, 3.0],
            [2.0, 5.0],
            [3.2, 6.3],
            [4.5, 7.0],
            [6.8, 7.0],
            [8.6, 8.0],
        ],
        dtype=float,
    )
    smooth_local = sample_cubic_bspline(control_points, samples_per_seg=32)

    draw_buildings(ax_zoom, local_rects, face="#d8d2ca")
    ax_zoom.plot(corridor_poly[:, 0], corridor_poly[:, 1], color="#a8c3ef", linewidth=26, alpha=0.28, solid_capstyle="round", zorder=3)
    ax_zoom.plot(corridor_poly[:, 0], corridor_poly[:, 1], color="#ef8f1f", linewidth=2.0, linestyle=(0, (4, 2)), zorder=7)
    ax_zoom.plot(control_points[:, 0], control_points[:, 1], color="#6c7a92", linewidth=1.35, linestyle=(0, (2, 2)), zorder=8)
    ax_zoom.scatter(control_points[:, 0], control_points[:, 1], s=34, color="#566273", edgecolor="white", linewidth=0.7, zorder=9)
    ax_zoom.plot(smooth_local[:, 0], smooth_local[:, 1], color="#2e86de", linewidth=3.2, zorder=10)
    for idx in (18, len(smooth_local) // 2, len(smooth_local) - 18):
        a = smooth_local[idx]
        b = smooth_local[min(idx + 3, len(smooth_local) - 1)]
        ax_zoom.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.4, color="#2e86de"), zorder=11)
    ax_zoom.scatter(control_points[0, 0], control_points[0, 1], s=92, color="#22a06b", edgecolor="white", linewidth=1.0, zorder=12)
    ax_zoom.scatter(control_points[-1, 0], control_points[-1, 1], s=92, marker="s", color="#d94841", edgecolor="white", linewidth=1.0, zorder=12)
    ax_zoom.set_title("局部转角放大", fontsize=18, weight="bold", pad=12)

    dist_skeleton = np.array([0.0, 60.0, 150.0, 240.0, 330.0, 420.0, 520.0], dtype=float)
    height_skeleton = np.array([18.0, 18.0, 42.0, 42.0, 68.0, 68.0, 82.0], dtype=float)
    profile_ctrl = np.column_stack([dist_skeleton, height_skeleton])
    profile_smooth = sample_cubic_bspline(profile_ctrl, samples_per_seg=36)
    profile_smooth = profile_smooth[np.argsort(profile_smooth[:, 0])]

    ax_profile.set_facecolor("#ffffff")
    ax_profile.step(dist_skeleton, height_skeleton, where="post", color="#ef8f1f", linewidth=2.2, linestyle=(0, (4, 2)))
    ax_profile.scatter(dist_skeleton, height_skeleton, s=30, color="#ef8f1f", edgecolor="white", linewidth=0.6, zorder=3)
    ax_profile.plot(profile_smooth[:, 0], profile_smooth[:, 1], color="#2e86de", linewidth=2.9)
    ax_profile.fill_between(profile_smooth[:, 0], profile_smooth[:, 1], color="#d9e8ff", alpha=0.24)
    ax_profile.set_title("高度剖面", fontsize=16, weight="bold", pad=10)
    ax_profile.set_xlabel("沿程距离 / m", fontsize=11.5)
    ax_profile.set_ylabel("高度 / m", fontsize=11.5)
    ax_profile.grid(alpha=0.18)
    for spine in ax_profile.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D([0], [0], color="#ef8f1f", lw=2.2, linestyle=(0, (4, 2)), marker="o", markersize=5, label="A* 骨架路径"),
        Line2D([0], [0], color="#6c7a92", lw=1.3, linestyle=(0, (2, 2)), marker="o", markersize=5, label="控制点"),
        Line2D([0], [0], color="#a8c3ef", lw=10, alpha=0.35, label="安全走廊"),
        Line2D([0], [0], color="#2e86de", lw=3.0, label="三次 B 样条轨迹"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#22a06b", markeredgecolor="white", markersize=8.5, label="起点"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#d94841", markeredgecolor="white", markersize=8.0, label="终点"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=6, frameon=False, fontsize=10.6)
    fig.suptitle("图2.6 路径规划与三次 B 样条轨迹优化示意图", fontsize=22, weight="bold", y=0.98)
    fig.savefig(OUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
