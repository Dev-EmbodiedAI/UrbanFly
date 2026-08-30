from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit per-episode candidate danger and label sources.")
    parser.add_argument("dataset_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    result = {}
    for directory in args.dataset_dirs:
        episodes = []
        for path in sorted(directory.glob("*.h5")):
            with h5py.File(path, "r") as handle:
                collision = handle["labels/candidate_collision"][:]
                clearance = handle["labels/candidate_minimum_clearance"][:]
                source = handle["labels/candidate_source"][:]
                values, counts = np.unique(source, return_counts=True)
                actor_count = int(handle["actors/valid_mask"][:].sum()) if "actors/valid_mask" in handle else 0
            episodes.append({
                "episode": path.stem, "steps": len(collision), "danger_fraction": float(collision.mean()),
                "clearance_p05_m": float(np.percentile(clearance, 5)), "actor_records": actor_count,
                "label_sources": {str(int(value)): int(count) for value, count in zip(values, counts)},
            })
        result[str(directory.resolve())] = episodes
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
