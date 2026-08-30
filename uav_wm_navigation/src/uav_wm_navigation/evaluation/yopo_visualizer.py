from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def depth_to_flu_points(depth: np.ndarray, stride: int = 5, max_depth_m: float = 20.0) -> np.ndarray:
    """Back-project a 90-degree RDF depth image into body FLU."""
    height, width = depth.shape
    fy = fx = width / 2.0
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0.1) & (z < max_depth_m)
    right = (columns - (width - 1) / 2.0) * z / fx
    down = (rows - (height - 1) / 2.0) * z / fy
    return np.column_stack([z[valid], -right[valid], -down[valid]]).astype(np.float32)


def render_yopo_frame(
    npz_path: str | Path, output_path: str | Path, frame_index: int = -1,
    title: str = "YOPO candidate evaluation",
) -> Path:
    """Render official-YOPO-inspired depth, point cloud and trajectory views."""
    with np.load(Path(npz_path).resolve()) as data:
        frame = frame_index % len(data["depth"])
        depth = data["depth"][frame]
        candidates = data["candidates"][frame]
        selected = int(data["selected"][frame])
        score = data["total_score"][frame]
        risk = data["collision_probability"][frame]
        executed = data["position_nwu"][: frame + 1]
    points = depth_to_flu_points(depth)
    finite = score[np.isfinite(score)]
    replacement = float(np.max(finite)) if finite.size else 1.0
    finite_score = np.nan_to_num(score, nan=replacement, posinf=replacement, neginf=0.0)
    norm = colors.Normalize(vmin=float(np.min(finite_score)), vmax=float(np.max(finite_score) + 1e-6))
    cmap = plt.get_cmap("turbo_r")
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_depth = fig.add_subplot(grid[0, 0])
    depth_image = ax_depth.imshow(np.ma.masked_invalid(depth), cmap="turbo", vmin=0, vmax=20)
    ax_depth.set_title("Metric depth (m)"); ax_depth.set_axis_off()
    fig.colorbar(depth_image, ax=ax_depth, fraction=0.04, pad=0.02)

    ax_3d = fig.add_subplot(grid[0, 1], projection="3d")
    if len(points):
        cloud = points[::2]
        ax_3d.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 0], cmap="cool", s=2, alpha=0.18)
    for index, trajectory in enumerate(candidates):
        color = cmap(norm(finite_score[index]))
        ax_3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color,
                   lw=4.0 if index == selected else 1.4, alpha=1.0 if index == selected else 0.78)
    ax_3d.scatter([0], [0], [0], c="#111827", marker="^", s=90, label="UAV")
    ax_3d.set(xlabel="Forward / m", ylabel="Left / m", zlabel="Up / m",
              title="15 predicted primitives + local depth cloud")
    ax_3d.view_init(elev=24, azim=-58)

    ax_top = fig.add_subplot(grid[1, 0])
    for index, trajectory in enumerate(candidates):
        color = cmap(norm(finite_score[index]))
        ax_top.plot(trajectory[:, 0], trajectory[:, 1], color=color,
                    lw=4.0 if index == selected else 1.4, alpha=1.0 if index == selected else 0.75)
        ax_top.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=15)
    ax_top.scatter(0, 0, c="#111827", marker="^", s=90)
    ax_top.set(title="Top-down candidate fan", xlabel="Forward / m", ylabel="Left / m")
    ax_top.axis("equal"); ax_top.grid(alpha=0.2)

    ax_score = fig.add_subplot(grid[1, 1])
    bars = ax_score.bar(np.arange(len(score)), finite_score, color=[cmap(norm(value)) for value in finite_score])
    if 0 <= selected < len(bars):
        bars[selected].set_edgecolor("black"); bars[selected].set_linewidth(2.5)
    ax_score.set(title=f"Candidate scores (selected #{selected})", xlabel="Lattice / candidate ID",
                 ylabel="Normalized total score")
    ax_score.grid(axis="y", alpha=0.2)
    if np.isfinite(risk).any():
        risk_axis = ax_score.twinx()
        risk_axis.plot(np.arange(len(risk)), risk, "o--", color="#dc2626", lw=1.2)
        risk_axis.set_ylabel("Collision probability"); risk_axis.set_ylim(0, 1)
    fig.suptitle(f"{title} · step {frame} · executed samples {len(executed)}", fontsize=16)
    output = Path(output_path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return output


def render_ablation_trajectories(summaries: list[dict], output_path: str | Path) -> Path:
    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    palette = {"yopo": "#6b7280", "yopo_dreamerv3": "#2563eb", "yopo_jepa": "#16a34a", "yopo_occflow": "#dc2626"}
    for summary in summaries:
        trajectory = np.asarray(summary.get("trajectory_nwu", []), dtype=np.float32)
        if len(trajectory):
            method = summary.get("method", "unknown")
            axis.plot(trajectory[:, 0], trajectory[:, 1], lw=2.5, label=method, color=palette.get(method))
            axis.scatter(trajectory[-1, 0], trajectory[-1, 1], s=35, color=palette.get(method))
    axis.set(title="Paired closed-loop trajectories", xlabel="NWU x / m", ylabel="NWU y / m")
    axis.axis("equal"); axis.grid(alpha=0.25); axis.legend()
    output = Path(output_path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return output
