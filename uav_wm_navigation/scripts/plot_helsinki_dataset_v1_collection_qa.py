#!/usr/bin/env python3
"""Render a concise collection QA overview from real Dataset v1 HDF5 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

ROOT = _bootstrap.PROJECT_ROOT.parent


TASK_COLORS = {
    "building_blocked": "#0072B2",
    "street_canyon": "#E69F00",
    "rooftop_to_ground": "#009E73",
    "ground_to_rooftop": "#CC79A7",
    "rooftop_to_rooftop": "#D55E00",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qa_path = args.qa or args.output_dir / "independent_collection_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    heightmap_path = (
        ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
        / "diagnostics" / "heightmap_0p5m.npz"
    )
    with np.load(heightmap_path) as data:
        height = np.asarray(data["height_m"])

    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    map_axis, episode_axis, task_axis, phase_axis = axes.ravel()
    map_axis.imshow(
        height, origin="lower", extent=(-500, 500, -500, 500),
        cmap="terrain", vmin=-2, vmax=45, alpha=0.82,
    )
    seen = set()
    # The independently read-back report may span several continuation runs.
    for path in (Path(episode["path"]) for episode in qa["episodes"]):
        with h5py.File(path, "r") as handle:
            metadata = json.loads(handle.attrs["metadata_json"])
            task = metadata["task_type"]
            route = handle["episode/global_route_world"][:]
            position = handle["state/position_world"][:]
            final = handle["next_state/position_world"][-1]
        color = TASK_COLORS[task]
        label = task if task not in seen else None
        seen.add(task)
        map_axis.plot(route[:, 0], route[:, 1], color=color, lw=0.7, alpha=0.40)
        map_axis.plot(
            np.r_[position[:, 0], final[0]], np.r_[position[:, 1], final[1]],
            color=color, lw=1.2, alpha=0.85, label=label,
        )
        map_axis.scatter(position[0, 0], position[0, 1], color=color, s=8, alpha=0.8)
    map_axis.set(
        title="Real Helsinki spatial coverage · route + executed trajectory",
        xlabel="canonical east (m)", ylabel="canonical north (m)", aspect="equal",
    )
    map_axis.legend(fontsize=8, loc="upper right")

    episodes = qa["episodes"]
    indices = np.arange(len(episodes))
    stale_ratio = np.asarray([item["stale_action_ratio"] for item in episodes]) * 100.0
    clearance = np.asarray([item["minimum_clearance_m"] for item in episodes])
    episode_axis.bar(indices, stale_ratio, color="#56B4E9", alpha=0.8, label="stale action (%)")
    clearance_axis = episode_axis.twinx()
    clearance_axis.plot(indices, clearance, color="#D55E00", lw=1.3, marker=".", label="min clearance")
    clearance_axis.axhline(2.5, color="#D55E00", ls="--", lw=0.9)
    episode_axis.set(title="Per-episode stale-action and clearance", xlabel="episode", ylabel="stale action (%)")
    clearance_axis.set_ylabel("minimum clearance (m)")
    lines, labels = episode_axis.get_legend_handles_labels()
    lines2, labels2 = clearance_axis.get_legend_handles_labels()
    episode_axis.legend(lines + lines2, labels + labels2, fontsize=8)

    tasks = list(TASK_COLORS)
    task_values = [qa["task_summary"][task] for task in tasks]
    x = np.arange(len(tasks))
    task_axis.bar(x - 0.2, [item["episodes"] for item in task_values], width=0.4, label="episodes")
    task_axis.bar(x + 0.2, [item["successes"] for item in task_values], width=0.4, label="successes")
    task_axis.set_xticks(x, [task.replace("_", "\n") for task in tasks], fontsize=8)
    task_axis.set(title="Balanced task outcome", ylabel="episodes")
    task_axis.legend()

    phases = ("start", "middle", "end")
    phase_ratio = [qa["stale_action"]["by_phase"][phase]["stale_action_ratio"] * 100.0 for phase in phases]
    phase_axis.bar(phases, phase_ratio, color=["#009E73", "#F0E442", "#CC79A7"])
    phase_axis.set(title="Stale-action concentration by trajectory phase", ylabel="stale action (%)")
    reset = qa["reset"]
    if "passes" in reset:
        automatic_reset_passes = reset["passes"]
        automatic_reset_expected = reset["expected"]
        restart_count = len(reset.get("process_restart_boundaries", []))
    else:
        automatic_resets = reset.get("within_run_automatic_resets", [])
        automatic_reset_passes = reset.get(
            "automatic_reset_passes",
            sum(item.get("status") == "PASS" for item in automatic_resets),
        )
        automatic_reset_expected = len(automatic_resets)
        restart_count = len(reset.get("process_or_replacement_boundaries", []))
    phase_axis.text(
        0.02, 0.96,
        f"episodes={qa['episode_count']}  transitions={qa['transition_count']}\n"
        f"success={qa['success_rate']:.1%}  collision={qa['collision_rate']:.1%}\n"
        f"clearance min/median={qa['clearance_m']['minimum']:.2f}/{qa['clearance_m']['median']:.2f} m\n"
        f"dt mean/P95/max={qa['dt_s']['mean']:.3f}/{qa['dt_s']['p95']:.3f}/{qa['dt_s']['maximum']:.3f} s\n"
        f"stale={qa['stale_action']['ratio']:.2%}  max burst={qa['stale_action']['maximum_burst']}\n"
        f"auto-reset={automatic_reset_passes}/{automatic_reset_expected}  "
        f"restarts={restart_count}  "
        f"partial={len(qa['partial_files'])}",
        transform=phase_axis.transAxes, va="top", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    figure.suptitle(f"UrbanFly Real Helsinki Dataset v1 · {qa['episode_count']}-Episode QA", fontsize=17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
