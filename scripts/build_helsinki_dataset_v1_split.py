#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_navigation import HelsinkiNavigationStack
from backend.engine.helsinki_spatial_split import HelsinkiSpatialSplit
from backend.engine.helsinki_urban_sampling import HelsinkiUrbanDensity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "helsinki_dataset_v1")
    parser.add_argument("--tasks", type=Path, default=ROOT / "outputs" / "helsinki_urban_core_regression" / "candidate_tasks.json")
    args = parser.parse_args()
    scene = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
    stack = HelsinkiNavigationStack.load(scene)
    density = HelsinkiUrbanDensity(stack.low_altitude_grid)
    partition = HelsinkiSpatialSplit()
    manifest = partition.manifest(density)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "spatial_split_v1.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    masks = partition.masks(stack.low_altitude_grid)
    extent = [
        stack.low_altitude_grid.origin[0],
        stack.low_altitude_grid.origin[0] + density.height.shape[0] * density.resolution,
        stack.low_altitude_grid.origin[2],
        stack.low_altitude_grid.origin[2] + density.height.shape[1] * density.resolution,
    ]
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    axis.imshow(
        density.urban_density_score.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        alpha=0.9,
        vmin=0,
        vmax=1,
    )
    colors = {"train": "#2ca25f", "validation": "#ffb000", "test": "#d7301f"}
    for split, mask in masks.items():
        overlay = np.ma.masked_where(~mask.T, mask.T)
        axis.imshow(
            overlay,
            origin="lower",
            extent=extent,
            cmap=matplotlib.colors.ListedColormap([colors[split]]),
            alpha=0.24,
            interpolation="nearest",
        )
        lower, upper = partition.interior_backend_z_bounds(split)
        axis.text(-485, (lower + upper) / 2, split.upper(), color=colors[split], weight="bold")
    core = np.ma.masked_where(~density.dense_urban_core_mask.T, density.dense_urban_core_mask.T)
    axis.contour(core, levels=[0.5], origin="lower", extent=extent, colors=["#23d5ff"], linewidths=0.8)
    if args.tasks.exists():
        tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
        for task in tasks[:25]:
            start = np.asarray(task["start"])
            goal = np.asarray(task["goal"])
            axis.scatter(start[0], start[2], s=12, c="#ffffff", edgecolors="#111111", linewidths=0.4)
            axis.scatter(goal[0], goal[2], s=18, marker="x", c="#111111", linewidths=0.8)
    axis.set(
        title="Helsinki Dataset v1 spatial split (20 m inter-region guard)",
        xlabel="backend east x (m)",
        ylabel="backend south z (m); canonical north = -z",
        xlim=(-500, 500),
        ylim=(-500, 500),
        aspect="equal",
    )
    figure.savefig(args.output_dir / "spatial_split_v1.png", dpi=180)
    plt.close(figure)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
