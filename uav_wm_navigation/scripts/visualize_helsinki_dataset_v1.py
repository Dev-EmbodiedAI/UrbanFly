#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

ROOT = _bootstrap.PROJECT_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    heightmap_path = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km" / "diagnostics" / "heightmap_0p5m.npz"
    with np.load(heightmap_path) as data:
        height = np.asarray(data["height_m"])
    with h5py.File(args.episode, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        sim = handle["timestamps/sim"][:]
        dt = handle["timestamps/dt"][:]
        time_s = sim - sim[0]
        position = handle["state/position_world"][:]
        next_position = handle["next_state/position_world"][-1]
        executed = np.vstack([position, next_position])
        route = handle["episode/global_route_world"][:]
        global_goal = handle["goal/global_goal_world"][0]
        speed = np.linalg.norm(handle["state/linear_velocity"][:], axis=1)
        clearance = handle["labels/minimum_clearance"][:]
        local_goal = handle["goal/local_goal_world"][:]
        local_goal_body = handle["goal/local_goal_body"][:]
        progress = handle["route/progress"][:]
        remaining = handle["route/remaining_distance"][:]
        commanded = handle["actions/commanded_body_flu"][:]
        factual = handle["actions/executed_body_flu"][:]
        rgb = handle["observations/rgb_front"]
        depth = handle["observations/depth_front"]
        frame_indices = [0, len(sim) // 2, len(sim) - 1]
        rgb_frames = [rgb[index] for index in frame_indices]
        depth_frames = [depth[index] for index in frame_indices]

    figure = plt.figure(figsize=(19, 17), constrained_layout=True)
    grid = figure.add_gridspec(5, 3)
    map_axis = figure.add_subplot(grid[0:2, 0:2])
    map_axis.imshow(
        height,
        origin="lower",
        extent=(-500, 500, -500, 500),
        cmap="terrain",
        vmin=-2,
        vmax=45,
        alpha=0.85,
    )
    map_axis.plot(route[:, 0], route[:, 1], color="#00a6ff", lw=2.0, label="frozen global route")
    map_axis.plot(executed[:, 0], executed[:, 1], color="#ff3b30", lw=1.4, label="executed")
    map_axis.scatter(*executed[0, :2], c="#21b573", s=65, label="start")
    map_axis.scatter(*global_goal[:2], c="#111111", marker="*", s=115, label="goal")
    map_axis.set(
        title=f"Real Helsinki map · {metadata['task_type']}",
        xlabel="canonical east (m)", ylabel="canonical north (m)", aspect="equal",
    )
    map_axis.legend(loc="best")

    timeline = figure.add_subplot(grid[0, 2])
    timeline.plot(time_s, remaining, label="route distance (m)")
    timeline.plot(time_s, position[:, 2], label="altitude (m)")
    timeline.plot(time_s, speed, label="speed (m/s)")
    timeline.plot(time_s, clearance, label="triangle clearance (m)")
    timeline.set(title="State / safety timeline", xlabel="sim time (s)")
    timeline.legend(fontsize=8)

    local_axis = figure.add_subplot(grid[1, 2])
    local_axis.plot(time_s, progress, label="route progress")
    local_axis.plot(time_s, np.linalg.norm(local_goal - position, axis=1), label="local-goal distance")
    local_axis.plot(time_s, local_goal_body[:, 0], label="goal forward")
    local_axis.plot(time_s, local_goal_body[:, 1], label="goal left")
    local_axis.set(title="Local Goal progression", xlabel="sim time (s)")
    local_axis.legend(fontsize=8)

    action_axis = figure.add_subplot(grid[2, :])
    names = ("forward", "left", "up", "yaw-rate")
    for index, name in enumerate(names):
        action_axis.plot(time_s, commanded[:, index], ls="--", lw=1.0, label=f"{name} commanded")
        action_axis.plot(time_s, factual[:, index], lw=1.2, label=f"{name} executed")
    action_axis.set(title=f"Action commanded vs executed · mean dt={dt.mean():.4f}s · P95={np.percentile(dt,95):.4f}s", xlabel="sim time (s)")
    action_axis.legend(ncol=4, fontsize=8)

    labels = ("start", "middle", "end")
    for column, (label, rgb_frame, depth_frame) in enumerate(zip(labels, rgb_frames, depth_frames)):
        rgb_axis = figure.add_subplot(grid[3, column])
        rgb_axis.imshow(rgb_frame)
        rgb_axis.set_title(f"{label} RGB · frame {frame_indices[column]}")
        rgb_axis.axis("off")
        depth_axis = figure.add_subplot(grid[4, column])
        image = depth_axis.imshow(depth_frame, cmap="turbo_r", vmin=0, vmax=min(120, np.percentile(depth_frame, 99)))
        depth_axis.set_title(f"{label} metric depth (m)")
        depth_axis.axis("off")
        figure.colorbar(image, ax=depth_axis, shrink=0.72)
    figure.suptitle(
        f"UrbanFly Helsinki Dataset v1 QA · {args.episode.stem}\n"
        "synchronized RGB-D + factual state/action + triangle safety truth",
        fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=155)
    plt.close(figure)
    print(json.dumps({"status": "PASS", "output": str(args.output), "frames": len(sim)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
