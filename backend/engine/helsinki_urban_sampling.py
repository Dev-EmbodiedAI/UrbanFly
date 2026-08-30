"""Geometry-derived urban density and spatial strata for Helsinki tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class UrbanDensityConfig:
    neighborhood_m: float = 75.0
    ground_height_m: float = 6.0
    building_height_m: float = 8.0
    minimum_obstacle_coverage: float = 0.08
    core_minimum_coverage: float = 0.12
    core_maximum_coverage: float = 0.68
    core_minimum_free_ratio: float = 0.30
    core_score_percentile: float = 55.0


class HelsinkiUrbanDensity:
    """Compute dense-core/peripheral masks without coordinate hard-coding."""

    def __init__(self, planning_grid, config: UrbanDensityConfig | None = None):
        self.grid = planning_grid
        self.config = config or UrbanDensityConfig()
        self.height = np.asarray(planning_grid.heightmap, dtype=np.float32)
        self.resolution = float(planning_grid.resolution)
        window_cells = max(
            3,
            int(round(self.config.neighborhood_m / self.resolution)),
        )
        if window_cells % 2 == 0:
            window_cells += 1
        self.window_cells = window_cells
        self.neighborhood_m = window_cells * self.resolution

        building = self.height >= self.config.building_height_m
        non_ground = self.height >= self.config.ground_height_m
        self.local_building_coverage = ndimage.uniform_filter(
            building.astype(np.float32),
            size=window_cells,
            mode="nearest",
        )
        self.local_obstacle_coverage = ndimage.uniform_filter(
            non_ground.astype(np.float32),
            size=window_cells,
            mode="nearest",
        )
        normalized_height = np.clip(
            (self.height - self.config.ground_height_m) / 28.0,
            0.0,
            1.0,
        )
        self.local_height_score = ndimage.uniform_filter(
            normalized_height.astype(np.float32),
            size=window_cells,
            mode="nearest",
        )
        free_ratio = 1.0 - self.local_obstacle_coverage
        street_mix = np.clip(
            4.0 * self.local_obstacle_coverage * free_ratio,
            0.0,
            1.0,
        )
        raw_score = (
            0.50 * np.clip(self.local_building_coverage / 0.45, 0.0, 1.0)
            + 0.25 * self.local_height_score
            + 0.25 * street_mix
        )
        self.urban_density_score = np.clip(raw_score, 0.0, 1.0).astype(np.float32)

        nx, nz = self.height.shape
        ix = np.arange(nx)[:, None]
        iz = np.arange(nz)[None, :]
        self.distance_to_boundary_m = (
            np.minimum.reduce(
                (
                    np.broadcast_to(ix, (nx, nz)),
                    np.broadcast_to(nx - 1 - ix, (nx, nz)),
                    np.broadcast_to(iz, (nx, nz)),
                    np.broadcast_to(nz - 1 - iz, (nx, nz)),
                )
            ).astype(np.float32)
            * self.resolution
        )
        map_width = min(nx, nz) * self.resolution
        self.edge_exclusion_m = float(np.clip(0.10 * map_width, 80.0, 120.0))
        self.non_open_mask = (
            self.local_obstacle_coverage >= self.config.minimum_obstacle_coverage
        )
        core_eligible = (
            self.non_open_mask
            & (self.distance_to_boundary_m >= self.edge_exclusion_m)
            & (self.local_obstacle_coverage >= self.config.core_minimum_coverage)
            & (self.local_obstacle_coverage <= self.config.core_maximum_coverage)
            & (free_ratio >= self.config.core_minimum_free_ratio)
        )
        eligible_scores = self.urban_density_score[core_eligible]
        self.core_score_threshold = float(
            np.percentile(eligible_scores, self.config.core_score_percentile)
            if eligible_scores.size
            else 1.0
        )
        core = core_eligible & (self.urban_density_score >= self.core_score_threshold)
        # Remove tiny isolated selections; a core must support routes, not only
        # one roof footprint.
        labels, count = ndimage.label(core, structure=np.ones((3, 3), dtype=np.uint8))
        if count:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= max(16, int(round(1200.0 / self.resolution**2)))
            keep[0] = False
            core = keep[labels]
        self.dense_urban_core_mask = core
        inner_distance = max(40.0, self.edge_exclusion_m * 0.5)
        self.peripheral_mixed_mask = (
            self.non_open_mask
            & ~self.dense_urban_core_mask
            & (self.distance_to_boundary_m >= inner_distance)
        )
        self.cross_city_mask = (
            self.distance_to_boundary_m >= inner_distance
        )

    def world_to_cell(self, position: np.ndarray) -> Tuple[int, int]:
        return self.grid.world_to_grid_xz(np.asarray(position, dtype=float))

    def score_at(self, position: np.ndarray) -> float:
        return float(self.urban_density_score[self.world_to_cell(position)])

    def obstacle_density_at(self, position: np.ndarray) -> float:
        return float(self.local_obstacle_coverage[self.world_to_cell(position)])

    def boundary_distance_at(self, position: np.ndarray) -> float:
        return float(self.distance_to_boundary_m[self.world_to_cell(position)])

    def mask_for_stratum(self, stratum: str) -> np.ndarray:
        return {
            "dense_core": self.dense_urban_core_mask,
            "peripheral_mixed": self.peripheral_mixed_mask,
            "cross_city": self.cross_city_mask,
        }[stratum]

    def summary(self) -> Dict[str, object]:
        return {
            "neighborhood_m": self.neighborhood_m,
            "edge_exclusion_m": self.edge_exclusion_m,
            "core_score_threshold": self.core_score_threshold,
            "dense_core_cell_ratio": float(self.dense_urban_core_mask.mean()),
            "peripheral_mixed_cell_ratio": float(self.peripheral_mixed_mask.mean()),
            "non_open_cell_ratio": float(self.non_open_mask.mean()),
            "score_percentiles": {
                str(percentile): float(
                    np.percentile(self.urban_density_score, percentile)
                )
                for percentile in (25, 50, 75, 90, 95)
            },
            "mean_core_obstacle_density": float(
                np.mean(self.local_obstacle_coverage[self.dense_urban_core_mask])
            ) if np.any(self.dense_urban_core_mask) else None,
            "mean_core_boundary_distance_m": float(
                np.mean(self.distance_to_boundary_m[self.dense_urban_core_mask])
            ) if np.any(self.dense_urban_core_mask) else None,
        }

