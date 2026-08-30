from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def read_binary_ply_xyz(path: Path, max_points: int, seed: int) -> np.ndarray:
    with path.open("rb") as f:
        vertex_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header is incomplete.")
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("format") and "binary_little_endian" not in text:
                raise ValueError("Only binary_little_endian PLY is supported.")
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                data_offset = f.tell()
                break

        if vertex_count is None:
            raise ValueError("Could not find vertex count in PLY header.")

        rng = np.random.default_rng(seed)
        sample_count = min(max_points, vertex_count)
        indices = np.sort(rng.choice(vertex_count, size=sample_count, replace=False))

        xyz = np.empty((sample_count, 3), dtype=np.float32)
        record_size = VERTEX_DTYPE.itemsize
        for out_i, vertex_i in enumerate(indices):
            f.seek(data_offset + int(vertex_i) * record_size)
            raw = f.read(record_size)
            xyz[out_i] = struct.unpack_from("<fff", raw, 0)
        return xyz


def robust_project_xy(points: np.ndarray) -> np.ndarray:
    # Use the two widest axes, so the figure remains meaningful if COLMAP axes differ.
    spans = np.ptp(points, axis=0)
    axes = np.argsort(spans)[-2:]
    axes = axes[np.argsort(axes)]
    return points[:, axes]


def crop_percentile(points_2d: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo = np.percentile(points_2d, low, axis=0)
    hi = np.percentile(points_2d, high, axis=0)
    mask = np.all((points_2d >= lo) & (points_2d <= hi), axis=1)
    return points_2d[mask]


def occupied_voxels(points_2d: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    origin = points_2d.min(axis=0)
    ij = np.floor((points_2d - origin) / cell_size).astype(np.int32)
    unique = np.unique(ij, axis=0)
    centers = origin + (unique.astype(np.float32) + 0.5) * cell_size
    return centers, origin


def add_voxel_panel(
    ax,
    points: np.ndarray,
    cell_size: float,
    title: str,
    color: str,
    font: FontProperties,
    max_cells: int = 1800,
) -> None:
    centers, _ = occupied_voxels(points, cell_size)
    if len(centers) > max_cells:
        rng = np.random.default_rng(7)
        centers = centers[rng.choice(len(centers), max_cells, replace=False)]

    ax.scatter(points[:, 0], points[:, 1], s=0.25, c="#9aa4b2", alpha=0.20, linewidths=0)
    patches = [
        Rectangle((cx - cell_size / 2, cy - cell_size / 2), cell_size, cell_size)
        for cx, cy in centers
    ]
    collection = PatchCollection(
        patches,
        facecolor=color,
        edgecolor="#1f2937",
        linewidth=0.25,
        alpha=0.35,
    )
    ax.add_collection(collection)
    ax.set_title(title, fontproperties=font, fontsize=16, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, default=Path(r"D:\colmap\data\output\dense\0\fused.ply"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"D:\AI\UrbanFly\thesis\figures\fig_multiscale_voxel_construction.png"),
    )
    parser.add_argument("--max-points", type=int, default=90_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    font_path = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties()
    plt.rcParams["axes.unicode_minus"] = False

    points = read_binary_ply_xyz(args.ply, args.max_points, args.seed)
    points_2d = crop_percentile(robust_project_xy(points))

    extent = np.ptp(points_2d, axis=0)
    base = float(max(extent) / 18.0)
    cell_sizes = [base, base / 4.0, base / 8.0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.4), dpi=220)
    fig.patch.set_facecolor("white")

    axes[0].scatter(points_2d[:, 0], points_2d[:, 1], s=0.35, c="#334155", alpha=0.45, linewidths=0)
    axes[0].set_title("稀疏点云锚点 P", fontproperties=font, fontsize=16, pad=10)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.2)

    panel_specs = [
        (cell_sizes[0], r"$X_0$: 原始尺度体素  $s_0=\delta$", "#f59e0b"),
        (cell_sizes[1], r"$X_2$: 四分之一尺度体素  $s_2=\delta/4$", "#22c55e"),
        (cell_sizes[2], r"$X_3$: 八分之一尺度体素  $s_3=\delta/8$", "#3b82f6"),
    ]
    for ax, (cell_size, title, color) in zip(axes[1:], panel_specs):
        add_voxel_panel(ax, points_2d, cell_size, title, color, font)

    for ax in axes:
        pad_x = extent[0] * 0.02
        pad_y = extent[1] * 0.02
        lo = points_2d.min(axis=0)
        hi = points_2d.max(axis=0)
        ax.set_xlim(lo[0] - pad_x, hi[0] + pad_x)
        ax.set_ylim(lo[1] - pad_y, hi[1] + pad_y)

    fig.suptitle(
        r"基于点云锚点的多层次体素集合构造：$X_k=\{\lfloor P/(\delta/2^k)\rceil\cdot\delta/2^k\}$",
        fontproperties=font,
        fontsize=20,
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "对比原始尺度、四分之一尺度与八分之一尺度：体素尺度越小，占据集合越贴合点云局部结构。",
        ha="center",
        fontproperties=font,
        fontsize=13,
        color="#475569",
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.84, bottom=0.13, wspace=0.08)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
