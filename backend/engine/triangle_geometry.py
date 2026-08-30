"""Triangle-level local collision queries for Helsinki photogrammetry meshes.

The global planner deliberately does not use this module.  It is a local
ground-truth layer backed by trimesh's R-tree triangle acceleration and is
used only after coarse planning/smoothing or for local safety queries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np
import trimesh


@dataclass(frozen=True)
class TriangleQueryResult:
    collision: bool
    minimum_distance_m: float
    sample_count: int
    collision_sample_index: Optional[int] = None
    collision_position: Optional[np.ndarray] = None
    closest_surface_position: Optional[np.ndarray] = None
    triangle_index: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "collision": self.collision,
            "minimum_distance_m": self.minimum_distance_m,
            "sample_count": self.sample_count,
            "collision_sample_index": self.collision_sample_index,
            "collision_position": (
                None if self.collision_position is None else self.collision_position.tolist()
            ),
            "closest_surface_position": (
                None
                if self.closest_surface_position is None
                else self.closest_surface_position.tolist()
            ),
            "triangle_index": self.triangle_index,
        }


class TriangleMeshLocalCollision:
    """R-tree accelerated unsigned surface-distance and swept-sphere queries.

    The Helsinki collision mesh is a non-watertight photogrammetric surface.
    Therefore ``distance_to_surface`` is unsigned and ``is_collision`` means
    that the UAV sphere intersects a triangle surface.  It does not claim a
    reliable inside/outside classification for arbitrary points initialized
    deep inside an open mesh.
    """

    def __init__(self, mesh: trimesh.Trimesh, source_path: Path, build_time_s: float):
        self.mesh = mesh
        self.source_path = Path(source_path)
        self.build_time_s = float(build_time_s)
        self.triangle_count = int(len(mesh.faces))
        self.vertex_count = int(len(mesh.vertices))
        self.bounds = np.asarray(mesh.bounds, dtype=float)
        self.resolution = 0.20
        self.acceleration_structure = "trimesh triangles_tree (libspatialindex R-tree BVH/AABB index)"

    @classmethod
    def load(cls, path: Path | str) -> "TriangleMeshLocalCollision":
        source = Path(path)
        started = time.perf_counter()
        loaded = trimesh.load(source, force="scene", process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = tuple(
                geometry
                for geometry in loaded.geometry.values()
                if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces)
            )
            if not meshes:
                raise ValueError(f"triangle mesh contains no faces: {source}")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise TypeError(f"unsupported triangle asset: {type(loaded).__name__}")
        # Materialize once.  Subsequent proximity queries use this R-tree and
        # never scan all 307k triangles.
        _ = mesh.triangles_tree
        return cls(mesh, source, time.perf_counter() - started)

    def _closest(self, points: np.ndarray):
        samples = np.asarray(points, dtype=float).reshape(-1, 3)
        closest, distances, triangles = trimesh.proximity.closest_point(
            self.mesh,
            samples,
        )
        return (
            np.asarray(closest, dtype=float),
            np.asarray(distances, dtype=float),
            np.asarray(triangles, dtype=np.int64),
        )

    def distance_to_surface(self, position: np.ndarray) -> float:
        _, distances, _ = self._closest(np.asarray(position, dtype=float)[None])
        return float(distances[0])

    def is_collision(self, position: np.ndarray, drone_radius: float) -> bool:
        return self.distance_to_surface(position) <= float(drone_radius) + 1e-9

    def clearance(self, position: np.ndarray, safety_radius: float = 0.0) -> float:
        del safety_radius
        return self.distance_to_surface(position)

    def collides(self, position: np.ndarray, safety_radius: float):
        distance = self.distance_to_surface(position)
        return distance <= float(safety_radius) + 1e-9, distance

    @staticmethod
    def _segment_samples(p0: np.ndarray, p1: np.ndarray, step_m: float) -> np.ndarray:
        start = np.asarray(p0, dtype=float)
        end = np.asarray(p1, dtype=float)
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / max(step_m, 1e-3))))
        return start[None] + np.linspace(0.0, 1.0, count + 1)[:, None] * (end - start)[None]

    def _query_samples(self, samples: np.ndarray, drone_radius: float) -> TriangleQueryResult:
        closest, distances, triangles = self._closest(samples)
        minimum_index = int(np.argmin(distances))
        collision_indices = np.flatnonzero(distances <= float(drone_radius) + 1e-9)
        collision_index = int(collision_indices[0]) if len(collision_indices) else None
        return TriangleQueryResult(
            collision=collision_index is not None,
            minimum_distance_m=float(distances[minimum_index]),
            sample_count=int(len(samples)),
            collision_sample_index=collision_index,
            collision_position=(
                None if collision_index is None else np.asarray(samples[collision_index], dtype=float)
            ),
            closest_surface_position=np.asarray(closest[minimum_index], dtype=float),
            triangle_index=int(triangles[minimum_index]),
        )

    def segment_query(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        drone_radius: float,
        step_m: Optional[float] = None,
    ) -> TriangleQueryResult:
        radius = float(drone_radius)
        # At radius > 0, a step <= radius/3 cannot jump across a diameter-wide
        # collision tube.  Cap at 0.20 m to retain thin-structure sensitivity.
        sample_step = float(step_m or min(0.20, max(0.05, radius / 3.0)))
        return self._query_samples(
            self._segment_samples(p0, p1, sample_step),
            radius,
        )

    def segment_collision(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        drone_radius: float,
    ) -> bool:
        return self.segment_query(p0, p1, drone_radius).collision

    def sweep_collides(
        self,
        start: np.ndarray,
        end: np.ndarray,
        safety_radius: float,
        step: Optional[float] = None,
    ):
        # This is the hot simulator path. Query the R-tree once with the
        # swept-sphere AABB, then do exact point-to-triangle distances only
        # against that small candidate set. Calling trimesh.closest_point for
        # every physics step is correct but needlessly repeats broad-phase
        # work over all sampled points.
        start_point = np.asarray(start, dtype=float)
        end_point = np.asarray(end, dtype=float)
        radius = float(safety_radius)
        lower = np.minimum(start_point, end_point) - radius
        upper = np.maximum(start_point, end_point) + radius
        bounds = tuple(np.concatenate([lower, upper]).tolist())
        triangle_ids = np.fromiter(
            self.mesh.triangles_tree.intersection(bounds), dtype=np.int64
        )
        if len(triangle_ids) == 0:
            return False, float("inf"), None

        sample_step = float(step or min(0.20, max(0.05, radius / 3.0)))
        samples = self._segment_samples(start_point, end_point, sample_step)
        triangles = np.asarray(self.mesh.triangles[triangle_ids], dtype=float)
        triangle_count = len(triangles)
        repeated_triangles = np.tile(triangles, (len(samples), 1, 1))
        repeated_samples = np.repeat(samples, triangle_count, axis=0)
        closest = trimesh.triangles.closest_point(
            repeated_triangles, repeated_samples
        ).reshape(len(samples), triangle_count, 3)
        distances = np.linalg.norm(closest - samples[:, None, :], axis=2)
        flat_minimum = int(np.argmin(distances))
        sample_index, _ = np.unravel_index(flat_minimum, distances.shape)
        minimum_distance = float(distances.ravel()[flat_minimum])
        collision = minimum_distance <= radius + 1e-9
        return (
            collision,
            minimum_distance,
            samples[sample_index].copy() if collision else None,
        )

    def trajectory_query(
        self,
        path: np.ndarray,
        drone_radius: float,
        step_m: Optional[float] = None,
    ) -> TriangleQueryResult:
        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            raise ValueError("trajectory must have shape (N, 3), N >= 2")
        radius = float(drone_radius)
        sample_step = float(step_m or min(0.20, max(0.05, radius / 3.0)))
        sampled = []
        for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            segment = self._segment_samples(start, end, sample_step)
            sampled.extend(segment if segment_index == 0 else segment[1:])
        return self._query_samples(np.asarray(sampled, dtype=float), radius)

    def trajectory_collision(self, path: np.ndarray, drone_radius: float) -> bool:
        return self.trajectory_query(path, drone_radius).collision
