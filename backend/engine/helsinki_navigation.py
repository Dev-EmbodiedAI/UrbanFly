"""Fail-closed navigation stack for the real HelsinkiCentral1km surface.

This module adapts the photogrammetry height surface to the repository's
existing multi-layer :class:`PathPlanner`.  It deliberately keeps the four
navigation layers explicit:

* global planning: existing multi-layer A* in ``PathPlanner``;
* trajectory generation: line-of-sight shortcutting and validated corner arcs;
* tracking: the backend's geometric 6-DOF waypoint controller;
* safety filter: independent swept-surface validation before and during flight.

The stack is not named an expert.  ``PrivilegedMapExpert`` is a qualification
result, not an implementation name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from ..config import HELSINKI_NAVIGATION
from .collision import HeightmapStaticCollisionMap
from .models import DroneState, DroneStateData, Waypoint
from .planner import OccupancyGrid, PathPlanner, PlanningError
from .triangle_geometry import TriangleMeshLocalCollision


class NavigationResult(str, Enum):
    PLANNING_FAILED = "PLANNING_FAILED"
    INVALID_START = "INVALID_START"
    INVALID_GOAL = "INVALID_GOAL"
    PATH_COLLISION = "PATH_COLLISION"
    PATH_GEOMETRY_INVALID = "PATH_GEOMETRY_INVALID"
    TRAJECTORY_GENERATION_FAILED = "TRAJECTORY_GENERATION_FAILED"
    CONTROLLER_TRACKING_FAILED = "CONTROLLER_TRACKING_FAILED"
    ACTION_TIMEOUT = "ACTION_TIMEOUT"
    COLLISION = "COLLISION"
    TIMEOUT = "TIMEOUT"
    SUCCESS = "SUCCESS"


@dataclass
class NavigationPlan:
    start: np.ndarray
    goal: np.ndarray
    global_path: np.ndarray
    simplified_path: np.ndarray
    trajectory: np.ndarray
    validation: Dict[str, object]
    planning_time_ms: float
    flight_level: str
    planner_mode: str
    expert_mode: str = "high_altitude"
    altitude_min_m: Optional[float] = None
    altitude_max_m: Optional[float] = None
    triangle_validation: Optional[Dict[str, object]] = None

    @property
    def path_length_m(self) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1).sum())


class HelsinkiDistanceFieldSlices:
    """Metric 2-D signed-distance slices aligned with the height surface."""

    def __init__(self, root: Path, manifest: dict, collision_map):
        self.root = Path(root)
        self.collision_map = collision_map
        self.slices = []
        for item in manifest["collision"].get("esdf_slices", []):
            with np.load(self.root / item["npz"], allow_pickle=False) as data:
                self.slices.append(
                    (
                        float(data["altitude_m"]),
                        np.asarray(data["signed_distance_m"], dtype=np.float32),
                    )
                )
        self.slices.sort(key=lambda item: item[0])

    def clearance(self, position: np.ndarray) -> float:
        """Return the nearest available altitude slice in metres.

        The result is diagnostic horizontal clearance.  It is not silently
        promoted to a 3-D ESDF; hard validity uses the conservative swept
        height surface.
        """
        if not self.slices:
            return float("nan")
        point = np.asarray(position, dtype=float)
        _, field = min(self.slices, key=lambda item: abs(item[0] - point[1]))
        row, column = self.collision_map._indices(point)
        if row < 0 or column < 0 or row >= field.shape[0] or column >= field.shape[1]:
            return -float("inf")
        return float(field[row, column])


def _max_pool_heightmap(
    collision_map: HeightmapStaticCollisionMap,
    planning_resolution_m: float,
    horizontal_inflation_m: float = 0.0,
) -> OccupancyGrid:
    """Convert row/decreasing-Z Helsinki storage to PathPlanner X/Z cells."""
    ratio = planning_resolution_m / collision_map.resolution
    factor = int(round(ratio))
    if factor < 1 or not math.isclose(ratio, factor, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("planning resolution must be an integer heightmap multiple")

    # Helsinki: [row north-to-south, column west-to-east]. PathPlanner:
    # [x increasing east, z increasing north].  Flip rows, then transpose.
    raw_xz = np.asarray(collision_map.height[::-1, :].T, dtype=np.float32)
    finite = raw_xz[np.isfinite(raw_xz)]
    fill_height = float(np.max(finite)) if finite.size else 1000.0
    raw_xz = np.where(np.isfinite(raw_xz), raw_xz, fill_height)
    if horizontal_inflation_m > 0.0:
        from scipy import ndimage

        radius_cells = int(math.ceil(horizontal_inflation_m / collision_map.resolution))
        raw_xz = ndimage.maximum_filter(
            raw_xz,
            size=radius_cells * 2 + 1,
            mode="nearest",
        )

    nx = (raw_xz.shape[0] - 1) // factor
    nz = (raw_xz.shape[1] - 1) // factor
    trimmed = raw_xz[: nx * factor, : nz * factor]
    height_xz = trimmed.reshape(nx, factor, nz, factor).max(axis=(1, 3))

    min_altitude = float(HELSINKI_NAVIGATION["minimum_altitude_m"])
    max_altitude = float(HELSINKI_NAVIGATION["maximum_altitude_m"])
    ny = int(math.ceil((max_altitude - min_altitude) / planning_resolution_m))
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    origin = np.array(
        [collision_map.origin_x, min_altitude, collision_map.origin_z],
        dtype=float,
    )
    occupancy = OccupancyGrid(
        grid=grid,
        origin=origin,
        resolution=planning_resolution_m,
        heightmap=height_xz,
    )
    # The generic class treats every positive terrain elevation as a building
    # footprint.  For L1 feasibility use the actual low-flight ceiling instead.
    effective_radius = (
        float(HELSINKI_NAVIGATION["drone_radius_m"])
        + float(HELSINKI_NAVIGATION["safety_margin_m"])
    )
    occupancy.footprint_2d = (height_xz > 6.0 - effective_radius).astype(np.uint8)
    occupancy.refresh_clearance_map()
    return occupancy


class HelsinkiNavigationStack:
    """Real-map planning, trajectory generation, validation and execution."""

    def __init__(
        self,
        collision_map: HeightmapStaticCollisionMap,
        distance_slices: HelsinkiDistanceFieldSlices,
        config: Optional[dict] = None,
        local_triangle_geometry: Optional[TriangleMeshLocalCollision] = None,
    ):
        self.config = {**HELSINKI_NAVIGATION, **(config or {})}
        self.collision_map = collision_map
        self.distance_slices = distance_slices
        self.local_triangle_geometry = local_triangle_geometry
        self.drone_radius = float(self.config["drone_radius_m"])
        self.safety_margin = float(self.config["safety_margin_m"])
        self.required_clearance = self.drone_radius + self.safety_margin
        self.planning_clearance = (
            self.required_clearance + float(self.config["tracking_buffer_m"])
        )
        self.low_altitude_planning_clearance = self.planning_clearance + 0.5
        self.grid = _max_pool_heightmap(
            collision_map,
            float(self.config["planning_resolution_m"]),
        )
        self.global_planner = PathPlanner(
            self.grid,
            fast_heightmap_mode=False,
            fail_closed=True,
            safety_margin=(
                self.safety_margin + float(self.config["tracking_buffer_m"])
            ),
        )
        self.low_altitude_grid = _max_pool_heightmap(
            collision_map,
            float(self.config["planning_resolution_m"]),
            horizontal_inflation_m=self.low_altitude_planning_clearance,
        )
        self.low_altitude_planner = PathPlanner(
            self.low_altitude_grid,
            fast_heightmap_mode=False,
            fail_closed=True,
            safety_margin=(
                self.safety_margin + float(self.config["tracking_buffer_m"])
                + 0.5
            ),
        )

    @classmethod
    def load(
        cls,
        scene_root: Path | str,
        config: Optional[dict] = None,
        enable_triangle_geometry: bool = False,
    ) -> "HelsinkiNavigationStack":
        root = Path(scene_root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        collision_map = HeightmapStaticCollisionMap.load(
            root / manifest["collision"]["heightmap"]["uri"]
        )
        slices = HelsinkiDistanceFieldSlices(root, manifest, collision_map)
        triangle_geometry = None
        if enable_triangle_geometry:
            triangle_geometry = TriangleMeshLocalCollision.load(
                root / manifest["collision"]["uri"]
            )
        return cls(
            collision_map,
            slices,
            config=config,
            local_triangle_geometry=triangle_geometry,
        )

    def _point_validity(self, point: np.ndarray, role: str) -> Dict[str, object]:
        point = np.asarray(point, dtype=float)
        reasons: List[str] = []
        if point.shape != (3,) or not np.isfinite(point).all():
            return {"valid": False, "role": role, "reasons": ["non_finite_or_wrong_shape"]}
        row, column = self.collision_map._indices(point)
        if not (0 <= row < self.collision_map.shape[0] and 0 <= column < self.collision_map.shape[1]):
            reasons.append("outside_helsinki_map")
        minimum_altitude = float(self.config["minimum_altitude_m"])
        maximum_altitude = float(self.config["maximum_altitude_m"])
        if point[1] < minimum_altitude or point[1] > maximum_altitude:
            reasons.append("altitude_out_of_bounds")
        surface = self.collision_map.surface_height(point, self.required_clearance)
        clearance = float(point[1] - surface) if np.isfinite(surface) else -float("inf")
        if clearance < self.required_clearance:
            reasons.append("insufficient_surface_clearance")
        return {
            "valid": not reasons,
            "role": role,
            "reasons": reasons,
            "surface_height_m": float(surface),
            "clearance_m": clearance,
            "required_clearance_m": self.required_clearance,
            "distance_slice_clearance_m": self.distance_slices.clearance(point),
        }

    def is_valid_start(self, start: np.ndarray) -> Dict[str, object]:
        return self._point_validity(start, "start")

    def is_valid_goal(self, goal: np.ndarray) -> Dict[str, object]:
        return self._point_validity(goal, "goal")

    def validate_path(
        self,
        path: np.ndarray,
        required_clearance: Optional[float] = None,
        altitude_min_m: Optional[float] = None,
        altitude_max_m: Optional[float] = None,
    ) -> Dict[str, object]:
        clearance_threshold = float(
            self.required_clearance
            if required_clearance is None
            else required_clearance
        )
        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            return {
                "path_valid": False,
                "minimum_clearance_m": -float("inf"),
                "minimum_clearance_margin_m": -float("inf"),
                "number_of_collision_samples": 1,
                "number_of_height_violations": 1,
                "sample_count": 0,
            }
        step = float(self.config["validation_step_m"])
        sampled: List[np.ndarray] = []
        for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            length = float(np.linalg.norm(end - start))
            count = max(1, int(math.ceil(length / step)))
            alpha = np.linspace(0.0, 1.0, count + 1)
            segment = start[None] + alpha[:, None] * (end - start)[None]
            sampled.extend(segment if segment_index == 0 else segment[1:])
        samples = np.asarray(sampled, dtype=float)
        clearances = np.asarray(
            [self.collision_map.clearance(point, clearance_threshold) for point in samples],
            dtype=float,
        )
        collision_mask = clearances < clearance_threshold
        hard_minimum = float(
            self.config["minimum_altitude_m"]
            if altitude_min_m is None
            else altitude_min_m
        )
        hard_maximum = float(
            self.config["maximum_altitude_m"]
            if altitude_max_m is None
            else altitude_max_m
        )
        height_mask = (
            (samples[:, 1] < hard_minimum - 1e-6)
            | (samples[:, 1] > hard_maximum + 1e-6)
            | ~np.isfinite(samples).all(axis=1)
        )
        minimum = float(np.min(clearances)) if len(clearances) else -float("inf")
        return {
            "path_valid": bool(not collision_mask.any() and not height_mask.any()),
            "minimum_clearance_m": minimum,
            "minimum_clearance_margin_m": minimum - clearance_threshold,
            "required_clearance_m": clearance_threshold,
            "number_of_collision_samples": int(collision_mask.sum()),
            "number_of_height_violations": int(height_mask.sum()),
            "sample_count": int(len(samples)),
            "altitude_min_constraint_m": hard_minimum,
            "altitude_max_constraint_m": hard_maximum,
            "minimum_altitude_m": float(np.min(samples[:, 1])) if len(samples) else None,
            "maximum_altitude_m": float(np.max(samples[:, 1])) if len(samples) else None,
        }

    def _segment_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        collides, _, _ = self.collision_map.sweep_collides(
            start,
            end,
            self.planning_clearance,
            step=float(self.config["validation_step_m"]),
        )
        return not collides

    def _shortcut(self, path: np.ndarray) -> np.ndarray:
        points = np.asarray(path, dtype=float)
        result = [points[0]]
        index = 0
        while index < len(points) - 1:
            next_index = index + 1
            # Bound lookahead so a long urban detour does not trigger an
            # O(N^2) sequence of hundreds-of-metre 0.25 m sweep queries.
            furthest = min(len(points) - 1, index + 16)
            for candidate in range(furthest, index, -1):
                if self._segment_valid(points[index], points[candidate]):
                    next_index = candidate
                    break
            result.append(points[next_index])
            index = next_index
        return np.asarray(result, dtype=float)

    def _corner_smooth(
        self,
        path: np.ndarray,
        radius_override: Optional[float] = None,
    ) -> np.ndarray:
        points = np.asarray(path, dtype=float)
        if len(points) < 3:
            return points.copy()
        radius = float(
            self.config["corner_radius_m"]
            if radius_override is None
            else radius_override
        )
        samples_per_corner = int(self.config["corner_samples"])
        result: List[np.ndarray] = [points[0].copy()]
        for index in range(1, len(points) - 1):
            previous, current, following = points[index - 1 : index + 2]
            incoming = current - previous
            outgoing = following - current
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length < 1e-6 or outgoing_length < 1e-6:
                result.append(current.copy())
                continue
            offset = min(radius, incoming_length * 0.25, outgoing_length * 0.25)
            before = current - incoming / incoming_length * offset
            after = current + outgoing / outgoing_length * offset
            corner_points: List[np.ndarray] = [before]
            for alpha in np.linspace(0.0, 1.0, samples_per_corner + 2)[1:-1]:
                curve = (
                    (1.0 - alpha) ** 2 * before
                    + 2.0 * (1.0 - alpha) * alpha * current
                    + alpha**2 * after
                )
                corner_points.append(curve)
            corner_points.append(after)
            candidate = np.asarray([result[-1], *corner_points], dtype=float)
            if self.validate_path(
                candidate,
                required_clearance=self.planning_clearance,
            )["path_valid"]:
                if np.linalg.norm(result[-1] - before) > 1e-6:
                    result.extend(corner_points)
                else:
                    result.extend(corner_points[1:])
            else:
                result.append(current.copy())
        result.append(points[-1].copy())
        return np.asarray(result, dtype=float)

    def _densify(self, path: np.ndarray) -> np.ndarray:
        spacing = float(self.config["trajectory_spacing_m"])
        result = [np.asarray(path[0], dtype=float)]
        for start, end in zip(path[:-1], path[1:]):
            length = float(np.linalg.norm(end - start))
            count = max(1, int(math.ceil(length / spacing)))
            for index in range(1, count + 1):
                result.append(start + (end - start) * (index / count))
        return np.asarray(result, dtype=float)

    def _validated_overflight_candidate(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Build a map-derived climb/cruise/descent corridor when it is valid.

        This is an any-angle optimization, not a failure fallback: the entire
        candidate must pass the same planning-clearance swept audit.  If it
        does not, multi-layer A* remains authoritative.
        """
        planar_distance = float(np.linalg.norm((goal - start)[[0, 2]]))
        count = max(2, int(math.ceil(planar_distance / self.grid.resolution)))
        maximum_surface = -float("inf")
        for alpha in np.linspace(0.0, 1.0, count + 1):
            probe = start + float(alpha) * (goal - start)
            surface = self.collision_map.surface_height(
                probe,
                self.planning_clearance,
            )
            maximum_surface = max(maximum_surface, surface)
        cruise_altitude = max(
            float(start[1]),
            float(goal[1]),
            maximum_surface + self.planning_clearance + 0.5,
        )
        if cruise_altitude > float(self.config["maximum_altitude_m"]):
            return None
        ascent = start.copy()
        ascent[1] = cruise_altitude
        descent = goal.copy()
        descent[1] = cruise_altitude
        candidate = np.asarray((start, ascent, descent, goal), dtype=float)
        keep = np.r_[True, np.linalg.norm(np.diff(candidate, axis=0), axis=1) > 1e-6]
        candidate = candidate[keep]
        validation = self.validate_path(
            candidate,
            required_clearance=self.planning_clearance,
        )
        return candidate if validation["path_valid"] else None

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        flight_level: str = "L2_transition",
        allow_layer_transitions: bool = True,
        expert_mode: str = "high_altitude",
        altitude_min_m: Optional[float] = None,
        altitude_max_m: Optional[float] = None,
    ) -> NavigationPlan:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        start_validity = self.is_valid_start(start)
        if not start_validity["valid"]:
            raise PlanningError(f"{NavigationResult.INVALID_START.value}: {start_validity['reasons']}")
        goal_validity = self.is_valid_goal(goal)
        if not goal_validity["valid"]:
            raise PlanningError(f"{NavigationResult.INVALID_GOAL.value}: {goal_validity['reasons']}")
        if float(np.linalg.norm(goal - start)) <= self.required_clearance * 2.0:
            raise PlanningError("INVALID_GOAL: start and goal are not meaningfully distinct")

        if expert_mode not in {"high_altitude", "low_altitude_3d"}:
            raise PlanningError(f"unsupported expert_mode={expert_mode!r}")
        if expert_mode == "low_altitude_3d":
            if altitude_min_m is None or altitude_max_m is None:
                raise PlanningError(
                    "low_altitude_3d requires hard altitude_min_m and altitude_max_m"
                )
            if float(altitude_min_m) >= float(altitude_max_m):
                raise PlanningError("invalid hard altitude interval")
            if (
                start[1] < float(altitude_min_m) - 1e-6
                or start[1] > float(altitude_max_m) + 1e-6
                or goal[1] < float(altitude_min_m) - 1e-6
                or goal[1] > float(altitude_max_m) + 1e-6
            ):
                raise PlanningError("INVALID_ENDPOINT: endpoint violates task altitude bounds")

        started = time.perf_counter()
        global_path = None
        planner_mode = "bounded_xyz_astar"
        if expert_mode == "high_altitude":
            global_path = (
                self._validated_overflight_candidate(start, goal)
                if allow_layer_transitions
                else None
            )
            planner_mode = "validated_heightmap_overflight"
        if global_path is None:
            planner_mode = (
                "bounded_xyz_astar"
                if expert_mode == "low_altitude_3d"
                else "existing_pathplanner_multilayer_astar"
            )
            selected_planner = (
                self.low_altitude_planner
                if expert_mode == "low_altitude_3d"
                else self.global_planner
            )
            bounded_search_maximum = (
                float(altitude_max_m)
                - float(self.config["low_altitude_vertical_tracking_buffer_m"])
                if expert_mode == "low_altitude_3d"
                else None
            )
            if (
                expert_mode == "low_altitude_3d"
                and max(float(start[1]), float(goal[1])) > bounded_search_maximum + 1e-6
            ):
                raise PlanningError(
                    "endpoint leaves no vertical tracking buffer below the hard ceiling"
                )
            waypoints = selected_planner.plan(
                start,
                goal,
                drone_radius=self.drone_radius,
                flight_level=flight_level,
                cruise_speed=float(self.config["cruise_speed_mps"]),
                allow_layer_transitions=allow_layer_transitions,
                altitude_min_m=(
                    float(altitude_min_m)
                    if expert_mode == "low_altitude_3d"
                    else None
                ),
                altitude_max_m=(
                    bounded_search_maximum
                    if expert_mode == "low_altitude_3d"
                    else None
                ),
                altitude_step_m=(
                    float(self.config["low_altitude_step_m"])
                    if expert_mode == "low_altitude_3d"
                    else None
                ),
            )
            global_path = np.asarray(
                [waypoint.position for waypoint in waypoints],
                dtype=float,
            )
        planning_time_ms = (time.perf_counter() - started) * 1000.0
        global_validation = self.validate_path(
            global_path,
            required_clearance=self.planning_clearance,
            altitude_min_m=altitude_min_m,
            altitude_max_m=altitude_max_m,
        )

        if not global_validation["path_valid"]:
            selected_planner = (
                self.low_altitude_planner
                if expert_mode == "low_altitude_3d"
                else self.global_planner
            )
            debug = selected_planner._last_debug
            recovered = None
            for name in ("repaired", "skeleton"):
                candidate = np.asarray(debug.get(name, []), dtype=float)
                if (
                    len(candidate) >= 2
                    and self.validate_path(
                        candidate,
                        required_clearance=self.planning_clearance,
                        altitude_min_m=altitude_min_m,
                        altitude_max_m=altitude_max_m,
                    )["path_valid"]
                ):
                    recovered = candidate
                    break
            if recovered is None:
                raise PlanningError(
                    f"{NavigationResult.PATH_COLLISION.value}: PathPlanner output failed independent audit"
                )
            global_path = recovered

        simplified = self._shortcut(global_path)
        smoothed = self._corner_smooth(
            simplified,
            radius_override=(
                float(self.config["low_altitude_corner_radius_m"])
                if expert_mode == "low_altitude_3d"
                else None
            ),
        )
        if not self.validate_path(
            smoothed,
            required_clearance=self.planning_clearance,
            altitude_min_m=altitude_min_m,
            altitude_max_m=altitude_max_m,
        )["path_valid"]:
            smoothed = simplified
        trajectory = self._densify(smoothed)
        validation = self.validate_path(
            trajectory,
            altitude_min_m=altitude_min_m,
            altitude_max_m=altitude_max_m,
        )
        if not validation["path_valid"]:
            raise PlanningError(
                f"{NavigationResult.TRAJECTORY_GENERATION_FAILED.value}: final trajectory is invalid"
            )
        triangle_validation = None
        if self.local_triangle_geometry is not None:
            triangle_result = self.local_triangle_geometry.trajectory_query(
                trajectory,
                self.required_clearance,
            )
            triangle_validation = triangle_result.as_dict()
            if triangle_result.collision:
                repaired_trajectory = None
                # Fail closed, but first try less aggressive versions of the
                # same heightmap-approved corridor.  This is local trajectory
                # repair only; the frozen global A* is not rerun or modified.
                for local_candidate in (simplified, global_path):
                    candidate = self._densify(local_candidate)
                    candidate_heightmap = self.validate_path(
                        candidate,
                        altitude_min_m=altitude_min_m,
                        altitude_max_m=altitude_max_m,
                    )
                    if not candidate_heightmap["path_valid"]:
                        continue
                    candidate_triangle = self.local_triangle_geometry.trajectory_query(
                        candidate,
                        self.required_clearance,
                    )
                    if not candidate_triangle.collision:
                        repaired_trajectory = candidate
                        validation = candidate_heightmap
                        triangle_validation = candidate_triangle.as_dict()
                        break
                if repaired_trajectory is None:
                    raise PlanningError(
                        f"{NavigationResult.PATH_GEOMETRY_INVALID.value}: "
                        "triangle-level swept validation rejected the local trajectory"
                    )
                trajectory = repaired_trajectory
        return NavigationPlan(
            start=start,
            goal=goal,
            global_path=global_path,
            simplified_path=simplified,
            trajectory=trajectory,
            validation=validation,
            planning_time_ms=planning_time_ms,
            flight_level=flight_level,
            planner_mode=planner_mode,
            expert_mode=expert_mode,
            altitude_min_m=altitude_min_m,
            altitude_max_m=altitude_max_m,
            triangle_validation=triangle_validation,
        )

    def execute(self, plan: NavigationPlan, timeout_s: Optional[float] = None) -> Dict[str, object]:
        """Execute with UrbanFly's real 6-DOF dynamics and waypoint tracker."""
        from .simulator import Simulator
        from .wind_model import WindModel

        default_timeout = float(self.config["execution_timeout_s"])
        if plan.expert_mode == "low_altitude_3d":
            default_timeout = max(default_timeout, plan.path_length_m / 1.5 + 60.0)
        timeout = float(timeout_s or default_timeout)
        waypoints = [
            Waypoint(
                position=point.copy(),
                metadata={
                    "approved_map_path": True,
                    "low_altitude_3d": plan.expert_mode == "low_altitude_3d",
                    "altitude_min_m": plan.altitude_min_m,
                    "altitude_max_m": plan.altitude_max_m,
                },
            )
            for point in plan.trajectory
        ]
        initial_yaw = 0.0
        for delta in np.diff(plan.trajectory, axis=0):
            if float(np.linalg.norm(delta[[0, 2]])) > 0.25:
                initial_yaw = float(np.degrees(np.arctan2(delta[2], delta[0])))
                break
        drone = DroneStateData(
            id="helsinki-verification-uav",
            drone_type="standard",
            position=plan.start.copy(),
            velocity=np.zeros(3, dtype=float),
            acceleration=np.zeros(3, dtype=float),
            yaw=initial_yaw,
            battery_remaining=350.0,
            payload_current=0.0,
            state=DroneState.TAKEOFF,
            safety_radius=self.required_clearance,
            path=waypoints,
        )
        zero_wind = WindModel(
            global_wind=np.zeros(3),
            turbulence_intensity=0.0,
            gust_amplitude=np.zeros(3),
            random_seed=0,
        )
        simulator = Simulator(
            planner=self.global_planner,
            static_collision_map=(
                self.local_triangle_geometry
                if self.local_triangle_geometry is not None
                else self.collision_map
            ),
            wind_model=zero_wind,
        )
        simulator.drones = [drone]
        simulator._reallocation_interval = float("inf")
        trajectory = [drone.position.copy()]
        speed_saturated_steps = 0
        climb_saturated_steps = 0
        acceleration_saturated_steps = 0
        result = NavigationResult.TIMEOUT
        steps = int(math.ceil(timeout / simulator.dt))
        for _ in range(steps):
            simulator.time += simulator.dt
            simulator._update_drone_dynamics(drone, simulator.dt)
            trajectory.append(drone.position.copy())
            if np.linalg.norm(drone.velocity) >= drone.max_speed * 0.98:
                speed_saturated_steps += 1
            if abs(float(drone.velocity[1])) >= drone.max_climb_rate * 0.98:
                climb_saturated_steps += 1
            if np.linalg.norm(drone.acceleration) >= drone.max_accel * 0.98:
                acceleration_saturated_steps += 1
            if simulator._static_collision_counts.get(drone.id, 0) > 0:
                result = NavigationResult.COLLISION
                break
            if (
                drone.current_path_index >= len(drone.path)
                and np.linalg.norm(drone.position - plan.goal)
                <= float(self.config["goal_tolerance_m"])
            ):
                result = NavigationResult.SUCCESS
                break

        executed = np.asarray(trajectory, dtype=float)
        executed_validation = self.validate_path(
            executed,
            altitude_min_m=plan.altitude_min_m,
            altitude_max_m=plan.altitude_max_m,
        )
        executed_triangle_validation = None
        if self.local_triangle_geometry is not None:
            executed_triangle = self.local_triangle_geometry.trajectory_query(
                executed,
                self.required_clearance,
            )
            executed_triangle_validation = executed_triangle.as_dict()
            if executed_triangle.collision and result == NavigationResult.SUCCESS:
                result = NavigationResult.COLLISION
        if not executed_validation["path_valid"] and result == NavigationResult.SUCCESS:
            if int(executed_validation["number_of_collision_samples"]) > 0:
                result = NavigationResult.COLLISION
            else:
                result = NavigationResult.CONTROLLER_TRACKING_FAILED
        reference_tree = cKDTree(plan.trajectory)
        tracking_error, _ = reference_tree.query(executed, k=1)
        final_error = float(np.linalg.norm(executed[-1] - plan.goal))
        if result == NavigationResult.TIMEOUT and final_error <= float(self.config["goal_tolerance_m"]):
            result = NavigationResult.SUCCESS
        elif result == NavigationResult.TIMEOUT and final_error < 10.0:
            result = NavigationResult.CONTROLLER_TRACKING_FAILED
        return {
            "result": result.value,
            "success": result == NavigationResult.SUCCESS,
            "collision": result == NavigationResult.COLLISION,
            "sim_time_s": float(simulator.time),
            "steps": int(len(executed) - 1),
            "final_error_m": final_error,
            "tracking_error_mean_m": float(np.mean(tracking_error)),
            "tracking_error_rmse_m": float(np.sqrt(np.mean(tracking_error ** 2))),
            "tracking_error_max_m": float(np.max(tracking_error)),
            "speed_saturation_ratio": float(speed_saturated_steps / max(1, len(executed) - 1)),
            "climb_saturation_ratio": float(climb_saturated_steps / max(1, len(executed) - 1)),
            "acceleration_saturation_ratio": float(acceleration_saturated_steps / max(1, len(executed) - 1)),
            "mean_altitude_m": float(np.mean(executed[:, 1])),
            "max_altitude_m": float(np.max(executed[:, 1])),
            "minimum_altitude_m": float(np.min(executed[:, 1])),
            "vertical_travel_m": float(np.abs(np.diff(executed[:, 1])).sum()),
            "actual_collision_count": int(simulator._static_collision_counts.get(drone.id, 0)),
            "final_path_index": int(drone.current_path_index),
            "path_waypoint_count": int(len(drone.path)),
            "final_position": executed[-1].copy(),
            "executed_trajectory": executed,
            "executed_validation": executed_validation,
            "executed_triangle_validation": executed_triangle_validation,
        }


def polyline_length(path: np.ndarray) -> float:
    points = np.asarray(path, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0
