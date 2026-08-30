from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from uav_wm_navigation.data import build_episode_occflow_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Build model-specific OccFlow targets outside raw HDF5 episodes.")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    episodes = sorted(set(manifest.get("train", []) + manifest.get("validation", []) + manifest.get("test", [])))
    for episode in episodes:
        destination = args.output_dir / f"{Path(episode).stem}.occflow.npz"
        if not destination.exists():
            build_episode_occflow_cache(episode, destination)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
