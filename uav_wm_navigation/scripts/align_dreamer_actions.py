from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.data.episode_writer import validate_episode


def align(path: Path) -> None:
    temporary = path.with_suffix(".h5.action-align.partial")
    shutil.copy2(path, temporary)
    with h5py.File(temporary, "r+") as handle:
        timestamps = handle["timestamp"][:]
        transition_dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.2
        positions = handle["candidates/positions"]
        velocities = handle["candidates/velocities"]
        accelerations = handle["candidates/accelerations"]
        durations = handle["candidates/duration"]
        selected = handle["selected_index"][:].astype(int)
        current = handle["position"][:]
        actions = np.empty((len(timestamps), 9), dtype=np.float32)
        for step, candidate in enumerate(selected):
            horizon = positions.shape[2]
            dt = float(durations[step, candidate]) / max(horizon - 1, 1)
            action_index = min(max(int(round(transition_dt / max(dt, 1e-6))), 1), horizon - 1)
            actions[step] = np.concatenate([
                positions[step, candidate, action_index] - current[step],
                velocities[step, candidate, action_index], accelerations[step, candidate, action_index],
            ])
        handle["sequence/action"][...] = actions
        handle.attrs["sequence_action_alignment"] = "median_observation_dt"
        handle.attrs["sequence_action_transition_dt_s"] = transition_dt
        handle.flush()
    validate_episode(temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomically align factual Dreamer actions to the observed transition interval.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.dataset_dir.glob("*.h5"))
    for path in paths:
        align(path)
    print(f"aligned {len(paths)} episodes in {args.dataset_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
