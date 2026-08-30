from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from uav_wm_navigation.data import create_grouped_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze corridor-grouped splits and episode content hashes.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    episodes = sorted(args.dataset_dir.glob("*.h5"))
    if not episodes:
        raise RuntimeError("dataset directory has no HDF5 episodes")
    create_grouped_splits(episodes, args.dataset_dir / "splits.json", seed=args.seed)
    print((args.dataset_dir / "splits.json").resolve()); return 0


if __name__ == "__main__":
    raise SystemExit(main())
