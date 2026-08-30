"""Audit consistency between the CityGS detail layer and global collision field."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.engine.collision import DenseSignedDistanceField, SparseStaticCollisionMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path("data/citygs_collision/Residence"),
    )
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_esdf = DenseSignedDistanceField.load(
        args.collision_dir / "global_esdf.npz"
    )
    local_surface = SparseStaticCollisionMap.load(
        args.collision_dir / "local_collision_sparse.npz"
    )
    metadata = json.loads(
        (args.collision_dir / "collision_geometry.json").read_text(
            encoding="utf-8"
        )
    )

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.sample_count, local_surface.voxel_count)
    indices = rng.choice(local_surface.voxel_count, sample_count, replace=False)
    positions = (
        local_surface.origin
        + (local_surface.coords[indices].astype(np.float32) + 0.5)
        * local_surface.resolution
    )
    global_distance = global_esdf.batch_clearance(positions)
    percentiles = np.percentile(
        global_distance,
        [0, 1, 5, 25, 50, 75, 95, 99, 100],
    )

    report = {
        "coordinate_frame": "Y-up ENU-like local metric frame",
        "sample_count": sample_count,
        "source_local_voxels": local_surface.voxel_count,
        "global_resolution_m": global_esdf.resolution,
        "local_resolution_m": local_surface.resolution,
        "mesh_watertight": metadata["buildings"]["watertight"],
        "mesh_winding_consistent": metadata["buildings"]["winding_consistent"],
        "local_detail_contained_by_global_solid_ratio": float(
            np.mean(global_distance <= 0.0)
        ),
        "local_detail_outside_global_solid_ratio": float(
            np.mean(global_distance > 0.0)
        ),
        "global_esdf_at_local_detail_percentiles_m": {
            str(label): float(value)
            for label, value in zip(
                ["p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p100"],
                percentiles,
            )
        },
        "interpretation": (
            "Local samples outside the conservative 1 m solid remain collision-active "
            "through the exact 0.25 m sparse detail query; they are not discarded."
        ),
    }
    output = args.collision_dir / "alignment_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
