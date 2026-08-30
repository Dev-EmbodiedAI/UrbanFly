"""Reproducible spatially isolated Dataset v1 partition for Helsinki."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class HelsinkiSpatialSplit:
    """Backend-frame bands with a 20 m no-sampling buffer between splits.

    The first northern/water-heavy 200 m band is excluded.  The three retained
    regions all contain real building geometry.  A route belongs to a split
    only if every point is inside that split's interior mask; trajectories can
    never be split at frame level.
    """

    guard_m: float = 20.0

    # Bounds are backend/renderer Z (positive south).  Canonical ENU north is
    # its negation.  These values are geometry-derived 200/400/200 m bands.
    raw_backend_z_bounds: Dict[str, tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.raw_backend_z_bounds is None:
            object.__setattr__(
                self,
                "raw_backend_z_bounds",
                {
                    "test": (-300.0, -100.0),
                    "train": (-100.0, 300.0),
                    "validation": (300.0, 500.0),
                },
            )
        if not 0.0 <= self.guard_m < 100.0:
            raise ValueError("guard_m must be within [0, 100)")

    def interior_backend_z_bounds(self, split: str) -> tuple[float, float]:
        lower, upper = self.raw_backend_z_bounds[split]
        return lower + self.guard_m, upper - self.guard_m

    def masks(self, planning_grid) -> dict[str, np.ndarray]:
        shape = planning_grid.heightmap.shape
        result: dict[str, np.ndarray] = {}
        for split in self.raw_backend_z_bounds:
            mask = np.zeros(shape, dtype=bool)
            lower, upper = self.interior_backend_z_bounds(split)
            for iz in range(shape[1]):
                point = planning_grid.grid_to_world_xz(0, iz, 0.0)
                if lower <= float(point[2]) <= upper:
                    mask[:, iz] = True
            result[split] = mask
        return result

    def assign_backend_position(self, position: np.ndarray) -> str | None:
        z = float(np.asarray(position, dtype=float)[2])
        matches = [
            split
            for split in self.raw_backend_z_bounds
            if self.interior_backend_z_bounds(split)[0]
            <= z
            <= self.interior_backend_z_bounds(split)[1]
        ]
        return matches[0] if len(matches) == 1 else None

    def assign_backend_route(self, route: np.ndarray) -> str | None:
        points = np.asarray(route, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("route must have shape [N,3] with finite backend coordinates")
        assignments = {self.assign_backend_position(point) for point in points}
        return assignments.pop() if len(assignments) == 1 and None not in assignments else None

    def manifest(self, density) -> dict[str, object]:
        masks = self.masks(density.grid)
        cell_area = float(density.grid.resolution) ** 2
        statistics = {}
        for split, mask in masks.items():
            urban = mask & density.non_open_mask
            statistics[split] = {
                "backend_z_interior_m": list(self.interior_backend_z_bounds(split)),
                "canonical_north_interior_m": [
                    -self.interior_backend_z_bounds(split)[1],
                    -self.interior_backend_z_bounds(split)[0],
                ],
                "area_m2": float(mask.sum() * cell_area),
                "urban_area_m2": float(urban.sum() * cell_area),
                "urban_density_mean": float(density.local_obstacle_coverage[mask].mean()),
                "dense_core_area_m2": float((mask & density.dense_urban_core_mask).sum() * cell_area),
                "task_endpoint_capacity_cells": int(urban.sum()),
            }
        return {
            "schema": "helsinki-spatial-split-v1",
            "coordinate_frame": "backend [east,up,south]; canonical north=-backend_z",
            "guard_m": self.guard_m,
            "assignment_rule": "episode route must lie wholly inside one buffered split",
            "frame_split_forbidden": True,
            "excluded_region": {
                "backend_z_m": [-500.0, -300.0],
                "reason": "water/open northern band; excluded rather than using it as held-out city",
            },
            "splits": statistics,
        }
