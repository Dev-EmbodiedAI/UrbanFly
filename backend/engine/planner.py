"""
3D 城市低空路径规划器
====================

实现思路：
1. 在多层空域图上执行 time-aware A*，生成离散骨架路径；
2. 在安全走廊内部做局部修复，降低建筑擦碰与局部拥塞风险；
3. 使用三次 B 样条重定形，将离散折线转换为可执行轨迹；
4. 维护时空走廊占用表，为上层任务分配和实验评估提供闭环接口。
"""

from __future__ import annotations

import math
from collections import defaultdict
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    from scipy.interpolate import splprep, splev
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover - fallback path
    splprep = None
    splev = None
    distance_transform_edt = None

from .models import Waypoint, BuildingInfo
from ..config import DEFAULT_FLIGHT_LEVEL, FLIGHT_LEVELS, PATH_PLANNING


class PlanningError(RuntimeError):
    """Raised when a fail-closed planner cannot produce a valid route."""


class OccupancyGrid:
    """高度图 + 建筑占据网格。"""

    def __init__(
        self,
        grid: np.ndarray,
        origin: np.ndarray,
        resolution: float,
        heightmap: np.ndarray = None,
        buildings: List[BuildingInfo] = None,
    ):
        self.grid = grid
        self.origin = np.asarray(origin, dtype=float)
        self.resolution = float(resolution)
        self.heightmap = heightmap
        self.buildings = buildings or []
        self.shape = grid.shape

        if self.heightmap is None and self.buildings:
            self.heightmap = self._build_heightmap_from_buildings()

        if self.heightmap is not None:
            self.footprint_2d = (self.heightmap > 0.1).astype(np.uint8)
        else:
            self.footprint_2d = np.zeros((grid.shape[0], grid.shape[2]), dtype=np.uint8)
        self.refresh_clearance_map()

    def refresh_clearance_map(self) -> None:
        """Precompute metric horizontal clearance for O(1) A* queries."""
        if distance_transform_edt is None:
            self._clearance_map = None
            return
        free = self.footprint_2d == 0
        self._clearance_map = distance_transform_edt(free) * self.resolution

    def _build_heightmap_from_buildings(self) -> np.ndarray:
        nx = self.grid.shape[0]
        nz = self.grid.shape[2]
        hm = np.zeros((nx, nz), dtype=float)
        for building in self.buildings:
            bounds_min = np.asarray(building.bounds_min, dtype=float)
            bounds_max = np.asarray(building.bounds_max, dtype=float)
            x0, z0 = self.world_to_grid_xz(bounds_min)
            x1, z1 = self.world_to_grid_xz(bounds_max)
            gx0, gx1 = sorted((x0, x1))
            gz0, gz1 = sorted((z0, z1))
            hm[gx0:gx1 + 1, gz0:gz1 + 1] = np.maximum(
                hm[gx0:gx1 + 1, gz0:gz1 + 1],
                float(bounds_max[1]),
            )
        return hm

    def in_bounds_xz(self, gx: int, gz: int) -> bool:
        return 0 <= gx < self.footprint_2d.shape[0] and 0 <= gz < self.footprint_2d.shape[1]

    def world_to_grid_xz(self, pos: np.ndarray) -> Tuple[int, int]:
        gx = int((pos[0] - self.origin[0]) / self.resolution)
        gz = int((pos[2] - self.origin[2]) / self.resolution)
        gx = int(np.clip(gx, 0, self.footprint_2d.shape[0] - 1))
        gz = int(np.clip(gz, 0, self.footprint_2d.shape[1] - 1))
        return gx, gz

    def grid_to_world_xz(self, gx: int, gz: int, y: float = 0.0) -> np.ndarray:
        wx = self.origin[0] + (gx + 0.5) * self.resolution
        wz = self.origin[2] + (gz + 0.5) * self.resolution
        return np.array([wx, y, wz], dtype=float)

    def get_height_at(self, gx: int, gz: int) -> float:
        if self.heightmap is None or not self.in_bounds_xz(gx, gz):
            return 0.0
        return float(self.heightmap[gx, gz])

    def get_local_density(self, gx: int, gz: int, radius_cells: int = 2) -> float:
        if not self.in_bounds_xz(gx, gz):
            return 1.0
        x0 = max(0, gx - radius_cells)
        x1 = min(self.footprint_2d.shape[0], gx + radius_cells + 1)
        z0 = max(0, gz - radius_cells)
        z1 = min(self.footprint_2d.shape[1], gz + radius_cells + 1)
        patch = self.footprint_2d[x0:x1, z0:z1]
        return float(np.mean(patch)) if patch.size else 0.0

    def get_clearance_at(self, gx: int, gz: int, max_search_cells: int = 8) -> float:
        if not self.in_bounds_xz(gx, gz):
            return 0.0
        if self._clearance_map is not None:
            return float(
                min(
                    self._clearance_map[gx, gz],
                    (max_search_cells + 1) * self.resolution,
                )
            )
        if self.footprint_2d[gx, gz] == 0:
            best = max_search_cells + 1
            for dx in range(-max_search_cells, max_search_cells + 1):
                for dz in range(-max_search_cells, max_search_cells + 1):
                    nx, nz = gx + dx, gz + dz
                    if not self.in_bounds_xz(nx, nz):
                        continue
                    if self.footprint_2d[nx, nz]:
                        best = min(best, math.hypot(dx, dz))
            return best * self.resolution
        return 0.0

    def get_safe_altitude(self, gx: int, gz: int, margin: float = 8.0) -> float:
        max_h = 0.0
        radius_cells = 2
        for dx in range(-radius_cells, radius_cells + 1):
            for dz in range(-radius_cells, radius_cells + 1):
                nx, nz = gx + dx, gz + dz
                if self.in_bounds_xz(nx, nz):
                    max_h = max(max_h, self.get_height_at(nx, nz))
        return max_h + margin

    def segment_blocked(self, p0: np.ndarray, p1: np.ndarray, clearance_margin: float = 2.0) -> bool:
        seg = p1 - p0
        dist = float(np.linalg.norm(seg))
        if dist < 1e-6:
            gx, gz = self.world_to_grid_xz(p0)
            return p0[1] < self.get_height_at(gx, gz) + clearance_margin

        n_samples = max(3, int(dist / max(self.resolution * 0.8, 1.0)))
        for i in range(n_samples + 1):
            t = i / n_samples
            point = p0 + seg * t
            gx, gz = self.world_to_grid_xz(point)
            if point[1] < self.get_height_at(gx, gz) + clearance_margin:
                return True
        return False


class PathPlanner:
    """多层空域 + 时空走廊 + B 样条轨迹规划器。"""

    def __init__(
        self,
        occupancy_grid: OccupancyGrid = None,
        fast_heightmap_mode: bool = False,
        fail_closed: bool = False,
        safety_margin: float = None,
    ):
        self.grid = occupancy_grid
        self.fast_heightmap_mode = bool(fast_heightmap_mode)
        self.fail_closed = bool(fail_closed)
        self._path_cache: Dict[Tuple, List[Waypoint]] = {}
        self._corridor_occupancy: Dict[Tuple[str, int, int, int], int] = defaultdict(int)
        self._last_debug: Dict = {}
        self.safety_margin = float(
            PATH_PLANNING["safety_margin"]
            if safety_margin is None
            else safety_margin
        )
        self.grid_resolution = float(PATH_PLANNING["grid_resolution"])
        self.time_slot_sec = float(PATH_PLANNING["time_slot_sec"])
        self.corridor_cell_size = float(PATH_PLANNING["corridor_cell_size"])
        self.corridor_capacity = int(PATH_PLANNING["corridor_capacity"])
        # ``L2_mid_level`` and ``L3_high_corridor`` are compatibility aliases,
        # not additional physical layers.  Treating them as adjacent nodes
        # made a rooftop ineligible to climb from L2 to L3 because the alias
        # cell in between inherited a lower ceiling.  Keep one node per real
        # altitude band and resolve aliases in ``_closest_level_index``.
        self.level_order = [
            lvl
            for lvl in (
                "L1_street_canyon",
                "L2_transition",
                "L3_trunk_corridor",
                "L4_emergency",
            )
            if lvl in FLIGHT_LEVELS
        ]

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        drone_radius: float = 2.0,
        flight_level: str = DEFAULT_FLIGHT_LEVEL,
        payload_weight: float = 0.0,
        departure_time: float = 0.0,
        cruise_speed: float = 12.0,
        reserve_corridor: bool = False,
        allow_layer_transitions: bool = True,
        altitude_min_m: float = None,
        altitude_max_m: float = None,
        altitude_step_m: float = None,
    ) -> List[Waypoint]:
        """
        生成一条满足空域层、走廊占用与动力学平滑需求的路径。
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        cache_slot = int(departure_time / max(self.time_slot_sec, 1.0))
        cache_key = (
            tuple(np.round(start, 1)),
            tuple(np.round(goal, 1)),
            flight_level,
            round(drone_radius, 2),
            round(self.safety_margin, 2),
            self.fail_closed,
            bool(allow_layer_transitions),
            None if altitude_min_m is None else round(float(altitude_min_m), 2),
            None if altitude_max_m is None else round(float(altitude_max_m), 2),
            None if altitude_step_m is None else round(float(altitude_step_m), 2),
            cache_slot,
        )
        if cache_key in self._path_cache:
            return [Waypoint(position=wp.position.copy(), arrival_time=wp.arrival_time, action=wp.action, metadata=dict(wp.metadata)) for wp in self._path_cache[cache_key]]

        if self.grid is None:
            if self.fail_closed:
                raise PlanningError("occupancy grid is unavailable")
            direct = self._positions_to_waypoints([start, goal], cruise_speed)
            if reserve_corridor:
                self.reserve_path(direct, departure_time, cruise_speed)
            self._path_cache[cache_key] = direct
            return direct

        if self.fast_heightmap_mode:
            direct_positions = self._heightmap_overflight_route(
                start,
                goal,
                clearance_margin=drone_radius + self.safety_margin + 8.0,
            )
            direct = self._positions_to_waypoints(direct_positions, cruise_speed)
            if reserve_corridor:
                self.reserve_path(direct, departure_time, cruise_speed)
            self._path_cache[cache_key] = direct
            return [
                Waypoint(
                    position=waypoint.position.copy(),
                    arrival_time=waypoint.arrival_time,
                    action=waypoint.action,
                    metadata=dict(waypoint.metadata),
                )
                for waypoint in direct
            ]

        bounded_3d = altitude_min_m is not None or altitude_max_m is not None
        if bounded_3d:
            if altitude_min_m is None or altitude_max_m is None:
                raise PlanningError("bounded 3-D planning requires both altitude_min_m and altitude_max_m")
            skeleton = self._bounded_3d_astar(
                start=start,
                goal=goal,
                drone_radius=drone_radius,
                altitude_min_m=float(altitude_min_m),
                altitude_max_m=float(altitude_max_m),
                altitude_step_m=float(altitude_step_m or self.grid.resolution),
                departure_time=departure_time,
                cruise_speed=cruise_speed,
                payload_weight=payload_weight,
            )
        else:
            skeleton = self._time_aware_astar(
                start=start,
                goal=goal,
                drone_radius=drone_radius,
                preferred_level=flight_level,
                departure_time=departure_time,
                cruise_speed=cruise_speed,
                payload_weight=payload_weight,
                allow_layer_transitions=allow_layer_transitions,
            )
        if bounded_3d:
            # The legacy repair/B-spline stages are layer-aware and may raise
            # points into L2/L3, which would violate a task ceiling.  The
            # Helsinki navigation stack applies its own independently audited
            # shortcut/corner smoothing after this hard-bounded skeleton.
            bounded_debug = dict(self._last_debug)
            waypoints = self._positions_to_waypoints(skeleton, cruise_speed)
            bounded_debug.update(
                start=start.tolist(),
                goal=goal.tolist(),
                flight_level=flight_level,
                skeleton=[point.tolist() for point in skeleton],
                repaired=[point.tolist() for point in skeleton],
                control_points=[point.tolist() for point in skeleton],
                smooth=[point.tolist() for point in skeleton],
            )
            self._last_debug = bounded_debug
            self._path_cache[cache_key] = waypoints
            return [
                Waypoint(
                    position=waypoint.position.copy(),
                    arrival_time=waypoint.arrival_time,
                    action=waypoint.action,
                    metadata=dict(waypoint.metadata),
                )
                for waypoint in waypoints
            ]
        repaired = self._repair_within_corridor(skeleton, drone_radius)
        control_points = self._compress_control_points(repaired)
        smooth = self._bspline_retime(control_points, drone_radius, cruise_speed)
        waypoints = self._positions_to_waypoints(smooth, cruise_speed)

        if reserve_corridor:
            self.reserve_path(waypoints, departure_time, cruise_speed)

        self._last_debug = {
            "start": start.tolist(),
            "goal": goal.tolist(),
            "flight_level": flight_level,
            "search_mode": "bounded_xyz_astar" if bounded_3d else "multilayer_astar",
            "altitude_min_m": altitude_min_m,
            "altitude_max_m": altitude_max_m,
            "altitude_step_m": altitude_step_m,
            "skeleton": [p.tolist() for p in skeleton],
            "repaired": [p.tolist() for p in repaired],
            "control_points": [p.tolist() for p in control_points],
            "smooth": [wp.position.tolist() for wp in waypoints],
            "corridor_signature": self.get_corridor_signature(
                [wp.position for wp in waypoints],
                departure_time=departure_time,
                cruise_speed=cruise_speed,
                preferred_layer=flight_level,
            ),
        }

        if len(self._path_cache) > 768:
            self._path_cache.clear()
        self._path_cache[cache_key] = waypoints
        return [Waypoint(position=wp.position.copy(), arrival_time=wp.arrival_time, action=wp.action, metadata=dict(wp.metadata)) for wp in waypoints]

    def _bounded_3d_astar(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        drone_radius: float,
        altitude_min_m: float,
        altitude_max_m: float,
        altitude_step_m: float,
        departure_time: float,
        cruise_speed: float,
        payload_weight: float,
    ) -> List[np.ndarray]:
        """Search a hard-bounded metric XYZ lattice over the existing map.

        Unlike the legacy four-band planner, the third state component indexes
        explicit metric altitudes.  Horizontal, vertical, and diagonal moves
        are all validated as swept 3-D segments against the same Helsinki
        height surface.  The bounds are hard constraints, never soft costs.
        """
        if not altitude_min_m < altitude_max_m:
            raise PlanningError("altitude_min_m must be lower than altitude_max_m")
        if altitude_step_m <= 0.0:
            raise PlanningError("altitude_step_m must be positive")
        tolerance = 1e-6
        if (
            start[1] < altitude_min_m - tolerance
            or start[1] > altitude_max_m + tolerance
            or goal[1] < altitude_min_m - tolerance
            or goal[1] > altitude_max_m + tolerance
        ):
            raise PlanningError("start or goal violates the hard altitude bounds")

        regular = np.arange(
            altitude_min_m,
            altitude_max_m + altitude_step_m * 0.5,
            altitude_step_m,
            dtype=float,
        )
        regular = regular[regular <= altitude_max_m + tolerance]
        altitudes = np.unique(
            np.round(
                np.r_[regular, altitude_max_m, float(start[1]), float(goal[1])],
                6,
            )
        )
        altitudes = altitudes[
            (altitudes >= altitude_min_m - tolerance)
            & (altitudes <= altitude_max_m + tolerance)
        ]
        if len(altitudes) < 2:
            raise PlanningError("hard altitude interval contains fewer than two levels")

        start_cell = self.grid.world_to_grid_xz(start)
        goal_cell = self.grid.world_to_grid_xz(goal)
        start_alt = int(np.argmin(np.abs(altitudes - start[1])))
        goal_alt = int(np.argmin(np.abs(altitudes - goal[1])))
        start_node = (start_cell[0], start_cell[1], start_alt)
        goal_node = (goal_cell[0], goal_cell[1], goal_alt)
        clearance_margin = drone_radius + self.safety_margin

        # Establish a broad, independently feasible ceiling-slice corridor.
        # This is a search-domain heuristic only: the returned route is still
        # found in the XYZ lattice below and may change altitude at any cell.
        ceiling_free = self.grid.heightmap + clearance_margin <= altitude_max_m + tolerance
        ceiling_start = (start_cell[0], start_cell[1])
        ceiling_goal = (goal_cell[0], goal_cell[1])
        ceiling_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        ceiling_parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        ceiling_cost: Dict[Tuple[int, int], float] = {ceiling_start: 0.0}
        ceiling_closed: set[Tuple[int, int]] = set()
        ceiling_order = 0

        def ceiling_heuristic(cell: Tuple[int, int]) -> float:
            return self.grid.resolution * math.hypot(
                cell[0] - ceiling_goal[0], cell[1] - ceiling_goal[1]
            )

        heappush(
            ceiling_heap,
            (ceiling_heuristic(ceiling_start), ceiling_order, ceiling_start),
        )
        while ceiling_heap:
            _, _, cell = heappop(ceiling_heap)
            if cell in ceiling_closed:
                continue
            ceiling_closed.add(cell)
            if cell == ceiling_goal:
                break
            for dx, dz in (
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ):
                neighbor = (cell[0] + dx, cell[1] + dz)
                if not self.grid.in_bounds_xz(*neighbor) or not ceiling_free[neighbor]:
                    continue
                step_cost = self.grid.resolution * (
                    math.sqrt(2.0) if dx != 0 and dz != 0 else 1.0
                )
                candidate = ceiling_cost[cell] + step_cost
                if candidate + 1e-6 < ceiling_cost.get(neighbor, float("inf")):
                    ceiling_cost[neighbor] = candidate
                    ceiling_parent[neighbor] = cell
                    ceiling_order += 1
                    heappush(
                        ceiling_heap,
                        (candidate + 1.5 * ceiling_heuristic(neighbor), ceiling_order, neighbor),
                    )
        if ceiling_goal not in ceiling_cost:
            raise PlanningError("no connected free-space corridor exists at the hard ceiling")
        ceiling_path = [ceiling_goal]
        ceiling_cell = ceiling_goal
        while ceiling_cell in ceiling_parent:
            ceiling_cell = ceiling_parent[ceiling_cell]
            ceiling_path.append(ceiling_cell)
        ceiling_path.reverse()
        path_mask = np.zeros_like(ceiling_free, dtype=bool)
        for cell in ceiling_path:
            path_mask[cell] = True
        corridor_radius_cells = max(2, int(math.ceil(40.0 / self.grid.resolution)))
        if distance_transform_edt is not None:
            corridor_mask = distance_transform_edt(~path_mask) <= corridor_radius_cells
        else:  # pragma: no cover - SciPy is required by the Helsinki runtime
            corridor_mask = np.ones_like(path_mask, dtype=bool)

        def node_world(node: Tuple[int, int, int]) -> np.ndarray:
            gx, gz, altitude_index = node
            return self.grid.grid_to_world_xz(gx, gz, float(altitudes[altitude_index]))

        def node_feasible(node: Tuple[int, int, int]) -> bool:
            gx, gz, altitude_index = node
            if not self.grid.in_bounds_xz(gx, gz):
                return False
            if not corridor_mask[gx, gz]:
                return False
            altitude = float(altitudes[altitude_index])
            return altitude >= self.grid.get_height_at(gx, gz) + clearance_margin

        if not node_feasible(start_node) or not node_feasible(goal_node):
            raise PlanningError("start or goal lattice cell is occupied after conservative pooling")
        if self.grid.segment_blocked(start, node_world(start_node), clearance_margin):
            raise PlanningError("start connector is blocked")
        if self.grid.segment_blocked(node_world(goal_node), goal, clearance_margin):
            raise PlanningError("goal connector is blocked")

        def heuristic(node: Tuple[int, int, int]) -> float:
            return float(np.linalg.norm(node_world(node) - node_world(goal_node)))

        open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start_node: 0.0}
        closed: set[Tuple[int, int, int]] = set()
        order = 0
        heappush(open_heap, (heuristic(start_node), order, start_node))
        max_iter = max(int(PATH_PLANNING["max_astar_iterations"]), 250000)
        expanded = 0

        while open_heap and expanded < max_iter:
            _, _, node = heappop(open_heap)
            if node in closed:
                continue
            closed.add(node)
            expanded += 1
            if node == goal_node:
                break
            current_cost = g_score[node]
            current_world = node_world(node)
            gx, gz, altitude_index = node
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for da in (-1, 0, 1):
                        if dx == 0 and dz == 0 and da == 0:
                            continue
                        if da != 0 and (dx != 0 or dz != 0):
                            continue
                        neighbor = (gx + dx, gz + dz, altitude_index + da)
                        if neighbor[2] < 0 or neighbor[2] >= len(altitudes):
                            continue
                        if not node_feasible(neighbor):
                            continue
                        neighbor_world = node_world(neighbor)
                        if self.grid.segment_blocked(
                            current_world,
                            neighbor_world,
                            clearance_margin=clearance_margin,
                        ):
                            continue
                        delta = neighbor_world - current_world
                        physical_distance = float(np.linalg.norm(delta))
                        vertical_distance = abs(float(delta[1]))
                        climb_cost = vertical_distance * (
                            float(PATH_PLANNING["climb_penalty_factor"]) - 1.0
                        )
                        tentative = (
                            current_cost
                            + physical_distance
                            + climb_cost
                            + (0.05 * payload_weight * vertical_distance)
                        )
                        if tentative + 1e-6 < g_score.get(neighbor, float("inf")):
                            g_score[neighbor] = tentative
                            came_from[neighbor] = node
                            order += 1
                            # Weighted A* keeps the hard safety and altitude
                            # constraints exact while avoiding near-uniform
                            # expansion across a 1 km city slice.
                            heappush(open_heap, (tentative + 2.0 * heuristic(neighbor), order, neighbor))
        else:
            raise PlanningError(
                f"bounded XYZ A* exhausted {expanded} states without reaching the goal"
            )

        if goal_node not in g_score:
            raise PlanningError("bounded XYZ A* failed to reach the goal")
        reverse_nodes = [goal_node]
        current = goal_node
        while current in came_from:
            current = came_from[current]
            reverse_nodes.append(current)
        reverse_nodes.reverse()
        skeleton = [start.copy()]
        skeleton.extend(node_world(node) for node in reverse_nodes)
        skeleton.append(goal.copy())
        self._last_debug = {
            "search_mode": "bounded_xyz_astar",
            "expanded_states": expanded,
            "altitudes_m": altitudes.tolist(),
            "altitude_min_m": altitude_min_m,
            "altitude_max_m": altitude_max_m,
            "altitude_step_m": altitude_step_m,
            "ceiling_corridor_cells": len(ceiling_path),
            "ceiling_corridor_radius_m": corridor_radius_cells * self.grid.resolution,
            "skeleton": [point.tolist() for point in skeleton],
        }
        return self._dedupe_positions(skeleton)

    def _heightmap_overflight_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        clearance_margin: float,
    ) -> List[np.ndarray]:
        """Fast conservative route for dense CityGS proxies.

        The Gaussian reconstruction already supplies dense geometry. For live
        initialization we sample its aligned height map along the direct
        corridor and build a climb-cruise-descent profile above the highest
        occupied cell, avoiding an expensive multi-layer A* search for every
        initial delivery.
        """
        planar_distance = float(np.linalg.norm((goal - start)[[0, 2]]))
        sample_count = max(8, int(planar_distance / max(self.grid.resolution, 1.0)) + 1)
        max_height = max(float(start[1]), float(goal[1]), 12.0)
        for sample_index in range(sample_count + 1):
            ratio = sample_index / sample_count
            point = start + (goal - start) * ratio
            gx, gz = self.grid.world_to_grid_xz(point)
            max_height = max(max_height, self.grid.get_height_at(gx, gz))

        cruise_altitude = max_height + clearance_margin
        if self.grid.grid.shape[1] > 0:
            grid_ceiling = (
                self.grid.origin[1]
                + self.grid.grid.shape[1] * self.grid.resolution
                - self.safety_margin
            )
            cruise_altitude = min(cruise_altitude, grid_ceiling)

        ascent = start.copy()
        ascent[1] = max(start[1], cruise_altitude)
        descent = goal.copy()
        descent[1] = max(goal[1], cruise_altitude)
        one_third = start + (goal - start) / 3.0
        two_thirds = start + (goal - start) * (2.0 / 3.0)
        one_third[1] = cruise_altitude
        two_thirds[1] = cruise_altitude
        return self._dedupe_positions(
            [start.copy(), ascent, one_third, two_thirds, descent, goal.copy()]
        )

    def _time_aware_astar(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        drone_radius: float,
        preferred_level: str,
        departure_time: float,
        cruise_speed: float,
        payload_weight: float,
        allow_layer_transitions: bool,
    ) -> List[np.ndarray]:
        start_cell = self.grid.world_to_grid_xz(start)
        goal_cell = self.grid.world_to_grid_xz(goal)
        preferred_idx = self._closest_level_index(preferred_level)
        start_level_idx = max(0, min(preferred_idx, len(self.level_order) - 1))
        goal_level_idx = max(0, min(preferred_idx, len(self.level_order) - 1))
        start_node = (start_cell[0], start_cell[1], start_level_idx)

        open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start_node: 0.0}
        order = 0

        heappush(open_heap, (self._heuristic(start_node, goal_cell, goal_level_idx), order, start_node))
        max_iter = int(PATH_PLANNING["max_astar_iterations"])
        best_goal_node = None
        best_goal_cost = float("inf")

        while open_heap and max_iter > 0:
            max_iter -= 1
            _, _, node = heappop(open_heap)
            gx, gz, layer_idx = node
            cur_cost = g_score[node]

            if (gx, gz) == goal_cell:
                best_goal_node = node
                best_goal_cost = cur_cost
                break

            for neighbor, step_cost in self._neighbors(
                node,
                drone_radius,
                allow_layer_transitions=allow_layer_transitions,
            ):
                ngx, ngz, nlayer_idx = neighbor
                if not self._layer_cell_feasible(ngx, ngz, nlayer_idx, drone_radius):
                    continue

                pos_a = self._node_to_world(node, drone_radius)
                pos_b = self._node_to_world(neighbor, drone_radius)
                if self.grid.segment_blocked(pos_a, pos_b, clearance_margin=drone_radius + self.safety_margin):
                    continue

                travel_dist = float(np.linalg.norm(pos_b - pos_a))
                arrival_time = departure_time + (cur_cost + travel_dist) / max(cruise_speed, 0.1)
                congestion_penalty = self._corridor_penalty_for_segment(pos_a, pos_b, self.level_order[nlayer_idx], arrival_time)
                density_penalty = self.grid.get_local_density(ngx, ngz, radius_cells=2) * PATH_PLANNING["density_penalty_factor"]
                clearance = self.grid.get_clearance_at(ngx, ngz)
                clearance_penalty = max(0.0, (drone_radius + self.safety_margin * 1.6 - clearance)) * 0.7
                transition_penalty = 0.0
                if nlayer_idx != layer_idx:
                    transition_penalty = PATH_PLANNING["layer_transition_penalty"] * (1.0 + 0.05 * payload_weight)

                tentative_g = (
                    cur_cost
                    + step_cost
                    + density_penalty
                    + congestion_penalty
                    + clearance_penalty
                    + transition_penalty
                )

                if tentative_g + 1e-6 < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = node
                    order += 1
                    priority = tentative_g + self._heuristic(neighbor, goal_cell, goal_level_idx)
                    heappush(open_heap, (priority, order, neighbor))

        if best_goal_node is None:
            if self.fail_closed:
                raise PlanningError(
                    "multi-layer A* exhausted the search without reaching the goal"
                )
            fallback = [start.copy(), goal.copy()]
            if self.grid.segment_blocked(start, goal, clearance_margin=drone_radius + self.safety_margin):
                midpoint = (start + goal) / 2.0
                mx, mz = self.grid.world_to_grid_xz(midpoint)
                midpoint[1] = self.grid.get_safe_altitude(mx, mz, self.safety_margin + 12.0)
                fallback = [start.copy(), midpoint, goal.copy()]
            return fallback

        rev_nodes = [best_goal_node]
        cur = best_goal_node
        while cur in came_from:
            cur = came_from[cur]
            rev_nodes.append(cur)
        rev_nodes.reverse()

        skeleton = [start.copy()]
        for node in rev_nodes[1:-1]:
            skeleton.append(self._node_to_world(node, drone_radius))
        skeleton.append(goal.copy())

        if len(skeleton) >= 2:
            sx, sz = self.grid.world_to_grid_xz(start)
            gx, gz = self.grid.world_to_grid_xz(goal)
            ascent = start.copy()
            ascent[1] = max(
                start[1],
                self._layer_altitude(sx, sz, start_level_idx, drone_radius),
            )
            descent = goal.copy()
            descent[1] = max(
                goal[1],
                self._layer_altitude(gx, gz, goal_level_idx, drone_radius),
            )
            skeleton.insert(1, ascent)
            skeleton.insert(-1, descent)

        return self._dedupe_positions(skeleton)

    def _neighbors(
        self,
        node: Tuple[int, int, int],
        drone_radius: float,
        allow_layer_transitions: bool = True,
    ) -> Iterable[Tuple[Tuple[int, int, int], float]]:
        gx, gz, layer_idx = node
        step = self.grid.resolution
        for dx, dz in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            ngx, ngz = gx + dx, gz + dz
            if not self.grid.in_bounds_xz(ngx, ngz):
                continue
            dist = step * math.sqrt(2.0) if dx != 0 and dz != 0 else step
            yield (ngx, ngz, layer_idx), dist

        if allow_layer_transitions:
            for delta in (-1, 1):
                nlayer = layer_idx + delta
                if 0 <= nlayer < len(self.level_order):
                    vertical_cost = PATH_PLANNING["layer_transition_penalty"] * 0.35
                    yield (gx, gz, nlayer), vertical_cost

    def _heuristic(self, node: Tuple[int, int, int], goal_cell: Tuple[int, int], goal_level_idx: int) -> float:
        gx, gz, layer_idx = node
        dx = (gx - goal_cell[0]) * self.grid.resolution
        dz = (gz - goal_cell[1]) * self.grid.resolution
        planar = math.hypot(dx, dz)
        layer_gap = abs(layer_idx - goal_level_idx) * PATH_PLANNING["layer_transition_penalty"] * 0.5
        return planar + layer_gap

    def _closest_level_index(self, preferred_level: str) -> int:
        if preferred_level in self.level_order:
            return self.level_order.index(preferred_level)
        if preferred_level == "L3_high_corridor" and "L3_trunk_corridor" in self.level_order:
            return self.level_order.index("L3_trunk_corridor")
        if preferred_level == "L2_mid_level" and "L2_transition" in self.level_order:
            return self.level_order.index("L2_transition")
        return self.level_order.index(DEFAULT_FLIGHT_LEVEL) if DEFAULT_FLIGHT_LEVEL in self.level_order else 0

    def _layer_altitude(
        self,
        gx: int,
        gz: int,
        layer_idx: int,
        drone_radius: float = 0.0,
    ) -> float:
        layer = self.level_order[layer_idx]
        cfg = FLIGHT_LEVELS[layer]
        center = (cfg["y_min"] + cfg["y_max"]) * 0.5
        building_h = self.grid.get_height_at(gx, gz)

        if layer == "L1_street_canyon":
            return max(cfg["y_min"] + 4.0, min(cfg["y_max"] - 1.0, 6.0))
        # The old implementation added an undocumented 8/14/20 m on top of
        # the configured safety margin.  It made a 46 m roof impossible to
        # enter the legal 40--60 m L3 band even though the aircraft plus margin
        # fits.  Use the explicit physical radius + configured margin instead.
        required_altitude = building_h + self.safety_margin + float(drone_radius)
        return max(center, required_altitude)

    def _layer_cell_feasible(self, gx: int, gz: int, layer_idx: int, drone_radius: float) -> bool:
        if not self.grid.in_bounds_xz(gx, gz):
            return False

        layer = self.level_order[layer_idx]
        cfg = FLIGHT_LEVELS[layer]
        altitude = self._layer_altitude(gx, gz, layer_idx, drone_radius)
        if altitude < cfg["y_min"] - 1e-6 or altitude > cfg["y_max"] + 1e-6:
            return False

        if layer == "L1_street_canyon":
            if self.grid.footprint_2d[gx, gz]:
                return False
            if self.grid.get_clearance_at(gx, gz) < drone_radius + self.safety_margin:
                return False
        return True

    def _node_to_world(
        self,
        node: Tuple[int, int, int],
        drone_radius: float = 0.0,
    ) -> np.ndarray:
        gx, gz, layer_idx = node
        y = self._layer_altitude(gx, gz, layer_idx, drone_radius)
        return self.grid.grid_to_world_xz(gx, gz, y)

    def _corridor_penalty_for_segment(self, p0: np.ndarray, p1: np.ndarray, layer: str, arrival_time: float) -> float:
        signature = self.get_corridor_signature([p0, p1], departure_time=arrival_time, cruise_speed=12.0, preferred_layer=layer)
        if not signature:
            return 0.0
        penalty = 0.0
        for key in signature:
            current = self._corridor_occupancy.get(key, 0)
            overflow = max(0, current + 1 - self.corridor_capacity)
            penalty += PATH_PLANNING["corridor_penalty_factor"] * overflow
        return penalty / len(signature)

    def _repair_within_corridor(self, positions: List[np.ndarray], drone_radius: float) -> List[np.ndarray]:
        if self.grid is None or len(positions) <= 2:
            return positions

        repaired = [positions[0].copy()]
        search_radius = int(PATH_PLANNING["local_repair_search_radius"])
        repair_samples = int(PATH_PLANNING["repair_sample_count"])

        for idx in range(1, len(positions)):
            prev = repaired[-1]
            curr = positions[idx].copy()
            blocked = self.grid.segment_blocked(prev, curr, clearance_margin=drone_radius + self.safety_margin)
            clearance_mid = self._segment_min_clearance(prev, curr)

            if blocked or clearance_mid < drone_radius + self.safety_margin * 1.3:
                midpoint = (prev + curr) / 2.0
                mgx, mgz = self.grid.world_to_grid_xz(midpoint)
                best = None
                best_cost = float("inf")
                for dx in range(-search_radius, search_radius + 1):
                    for dz in range(-search_radius, search_radius + 1):
                        ngx, ngz = mgx + dx, mgz + dz
                        if not self.grid.in_bounds_xz(ngx, ngz):
                            continue
                        candidate = self.grid.grid_to_world_xz(ngx, ngz, midpoint[1])
                        candidate[1] = max(candidate[1], self.grid.get_safe_altitude(ngx, ngz, self.safety_margin + 6.0))
                        if self.grid.segment_blocked(prev, candidate, clearance_margin=drone_radius + self.safety_margin):
                            continue
                        if self.grid.segment_blocked(candidate, curr, clearance_margin=drone_radius + self.safety_margin):
                            continue
                        detour = np.linalg.norm(prev - candidate) + np.linalg.norm(candidate - curr)
                        density = self.grid.get_local_density(ngx, ngz)
                        clearance_bonus = self.grid.get_clearance_at(ngx, ngz)
                        cost = detour + density * 25.0 - clearance_bonus * 0.15
                        if cost < best_cost:
                            best_cost = cost
                            best = candidate

                if best is None:
                    segment = curr - prev
                    norm = np.linalg.norm(segment[:2])
                    lateral = np.array([-(curr[2] - prev[2]), 0.0, curr[0] - prev[0]], dtype=float)
                    lateral_norm = np.linalg.norm(lateral)
                    if lateral_norm > 1e-6:
                        lateral = lateral / lateral_norm
                    else:
                        lateral = np.array([1.0, 0.0, 0.0], dtype=float)
                    shift = max(self.grid.resolution * 1.5, drone_radius * 3.0)
                    best = midpoint + lateral * shift
                    best[1] = max(best[1], midpoint[1] + 6.0)

                repaired.append(best)

            if idx < len(positions) - 1 and repair_samples > 0:
                repaired.append(curr)
            else:
                repaired.append(curr)

        return self._dedupe_positions(repaired)

    def _segment_min_clearance(self, p0: np.ndarray, p1: np.ndarray) -> float:
        if self.grid is None:
            return 999.0
        dist = float(np.linalg.norm(p1 - p0))
        n_samples = max(3, int(dist / max(self.grid.resolution, 1.0)))
        min_clearance = float("inf")
        for i in range(n_samples + 1):
            t = i / n_samples
            point = p0 + (p1 - p0) * t
            gx, gz = self.grid.world_to_grid_xz(point)
            min_clearance = min(min_clearance, self.grid.get_clearance_at(gx, gz))
        return min_clearance

    def _bspline_retime(self, positions: List[np.ndarray], drone_radius: float, cruise_speed: float) -> List[np.ndarray]:
        if len(positions) <= 2:
            return positions

        points = [np.asarray(p, dtype=float) for p in positions]
        if len(points) == 3:
            points.insert(1, (points[0] + points[1]) / 2.0)
            points.insert(-1, (points[-2] + points[-1]) / 2.0)

        smooth = self._scipy_bspline(points)
        if not smooth:
            padded = [points[0], points[0], *points, points[-1], points[-1]]
            samples_per_seg = int(PATH_PLANNING["bspline_samples_per_segment"])
            smooth = [points[0].copy()]

            for i in range(1, len(padded) - 2):
                p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
                for s in range(1, samples_per_seg + 1):
                    t = s / samples_per_seg
                    t2 = t * t
                    t3 = t2 * t
                    basis = np.array(
                        [
                            (-t3 + 3 * t2 - 3 * t + 1) / 6.0,
                            (3 * t3 - 6 * t2 + 4) / 6.0,
                            (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0,
                            t3 / 6.0,
                        ],
                        dtype=float,
                    )
                    point = basis[0] * p0 + basis[1] * p1 + basis[2] * p2 + basis[3] * p3
                    smooth.append(point)

            smooth[0] = points[0].copy()
            smooth[-1] = points[-1].copy()

        adjusted = self._enforce_dynamics_and_clearance(smooth, drone_radius, cruise_speed)
        return self._resample_positions(adjusted, int(PATH_PLANNING["smooth_points"]))

    def _scipy_bspline(self, points: List[np.ndarray]) -> List[np.ndarray]:
        if splprep is None or splev is None or len(points) < 4:
            return []
        try:
            pts = np.array(points, dtype=float)
            k = min(3, len(points) - 1)
            smoothing = max(2.0, len(points) * 1.25)
            tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=smoothing, k=k)
            n_samples = max(int(PATH_PLANNING["smooth_points"]), len(points) * 10)
            u_new = np.linspace(0.0, 1.0, n_samples)
            x_new, y_new, z_new = splev(u_new, tck)
            smooth = [np.array([x, y, z], dtype=float) for x, y, z in zip(x_new, y_new, z_new)]
            smooth[0] = pts[0].copy()
            smooth[-1] = pts[-1].copy()
            return smooth
        except Exception:
            return []

    def _enforce_dynamics_and_clearance(
        self,
        positions: List[np.ndarray],
        drone_radius: float,
        cruise_speed: float,
    ) -> List[np.ndarray]:
        if self.grid is None or len(positions) <= 2:
            return positions

        climb_ratio = 0.35
        adjusted = [positions[0].copy()]

        for idx in range(1, len(positions) - 1):
            prev = adjusted[-1]
            cur = positions[idx].copy()
            nxt = positions[idx + 1]

            gx, gz = self.grid.world_to_grid_xz(cur)
            safe_alt = self.grid.get_safe_altitude(
                gx,
                gz,
                PATH_PLANNING["bspline_clearance_margin"],
            )
            cur[1] = max(cur[1], safe_alt)

            horiz = max(np.linalg.norm((cur - prev)[[0, 2]]), 1.0)
            max_climb = max(2.0, horiz * climb_ratio)
            dy = cur[1] - prev[1]
            if abs(dy) > max_climb:
                cur[1] = prev[1] + math.copysign(max_climb, dy)

            if self.grid.segment_blocked(prev, cur, clearance_margin=drone_radius + self.safety_margin):
                cur[1] = max(cur[1], safe_alt + 4.0)

            adjusted.append(cur)

        adjusted.append(positions[-1].copy())
        adjusted[0] = positions[0].copy()
        adjusted[-1] = positions[-1].copy()
        return self._dedupe_positions(adjusted)

    def _positions_to_waypoints(self, positions: List[np.ndarray], cruise_speed: float) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        elapsed = 0.0
        last = None
        for pos in positions:
            pos = np.asarray(pos, dtype=float)
            if last is not None:
                elapsed += float(np.linalg.norm(pos - last)) / max(cruise_speed, 0.1)
            gx, gz = self.grid.world_to_grid_xz(pos) if self.grid is not None else (0, 0)
            wp = Waypoint(
                position=pos,
                arrival_time=elapsed,
                metadata={
                    "corridor_cell": self._corridor_cell_from_world(pos),
                    "grid_x": gx,
                    "grid_z": gz,
                },
            )
            waypoints.append(wp)
            last = pos
        return waypoints

    def get_corridor_signature(
        self,
        positions: List[np.ndarray],
        departure_time: float,
        cruise_speed: float,
        preferred_layer: Optional[str] = None,
    ) -> List[Tuple[str, int, int, int]]:
        if len(positions) < 2:
            return []

        layer = preferred_layer if preferred_layer in FLIGHT_LEVELS else DEFAULT_FLIGHT_LEVEL
        signature: List[Tuple[str, int, int, int]] = []
        cumulative = 0.0
        for idx in range(len(positions) - 1):
            p0 = np.asarray(positions[idx], dtype=float)
            p1 = np.asarray(positions[idx + 1], dtype=float)
            seg = p1 - p0
            dist = float(np.linalg.norm(seg))
            n_samples = max(2, int(dist / max(self.corridor_cell_size * 0.5, 5.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                point = p0 + seg * t
                time_at = departure_time + (cumulative + dist * t) / max(cruise_speed, 0.1)
                slot = int(time_at / max(self.time_slot_sec, 1.0))
                cell_x, cell_z = self._corridor_cell_from_world(point)
                key = (layer, cell_x, cell_z, slot)
                if not signature or signature[-1] != key:
                    signature.append(key)
            cumulative += dist
        return signature

    def reserve_path(self, waypoints: List[Waypoint], departure_time: float, cruise_speed: float, preferred_layer: Optional[str] = None):
        positions = [wp.position for wp in waypoints]
        for key in self.get_corridor_signature(positions, departure_time, cruise_speed, preferred_layer):
            self._corridor_occupancy[key] += 1

    def release_path(self, waypoints: List[Waypoint], departure_time: float, cruise_speed: float, preferred_layer: Optional[str] = None):
        positions = [wp.position for wp in waypoints]
        for key in self.get_corridor_signature(positions, departure_time, cruise_speed, preferred_layer):
            if self._corridor_occupancy.get(key, 0) > 0:
                self._corridor_occupancy[key] -= 1

    def corridor_snapshot(self) -> Dict[Tuple[str, int, int, int], int]:
        return dict(self._corridor_occupancy)

    def estimate_path_metrics(
        self,
        waypoints: List[Waypoint],
        cruise_speed: float,
        preferred_layer: Optional[str] = None,
    ) -> Dict[str, float]:
        if len(waypoints) < 2:
            return {
                "distance_m": 0.0,
                "curvature_cost": 0.0,
                "climb_cost": 0.0,
                "smoothness": 1.0,
                "min_clearance": 999.0,
                "corridor_conflicts": 0.0,
            }

        positions = [wp.position for wp in waypoints]
        distance = 0.0
        curvature = 0.0
        climb_cost = 0.0
        min_clearance = float("inf")

        for i in range(len(positions) - 1):
            step = float(np.linalg.norm(positions[i + 1] - positions[i]))
            distance += step
            climb_cost += abs(float(positions[i + 1][1] - positions[i][1]))
            if self.grid is not None and 0 < i < len(positions) - 2:
                gx, gz = self.grid.world_to_grid_xz(positions[i])
                min_clearance = min(min_clearance, self.grid.get_clearance_at(gx, gz))

        for i in range(1, len(positions) - 1):
            v1 = positions[i] - positions[i - 1]
            v2 = positions[i + 1] - positions[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                curvature += math.acos(cos_theta) ** 2

        signature = self.get_corridor_signature(positions, 0.0, cruise_speed, preferred_layer)
        conflicts = 0.0
        for key in signature:
            load = self._corridor_occupancy.get(key, 0)
            conflicts += max(0, load - self.corridor_capacity)

        smoothness = 1.0 / (1.0 + curvature)
        return {
            "distance_m": distance,
            "curvature_cost": curvature,
            "climb_cost": climb_cost,
            "smoothness": smoothness,
            "min_clearance": min_clearance if min_clearance != float("inf") else 999.0,
            "corridor_conflicts": conflicts,
        }

    def plan_bundle_path(self, drone, bundle, start_time: float = 0.0) -> Tuple[List[Waypoint], float]:
        all_waypoints: List[Waypoint] = []
        total_dist = 0.0
        current_pos = np.asarray(drone.position, dtype=float)
        current_time = float(start_time)

        for task in bundle:
            task_level = getattr(task, "airspace_level", None) or DEFAULT_FLIGHT_LEVEL

            if getattr(task, "is_patrol", False) and task.patrol_waypoints:
                patrol_points = [task.pickup_pos] + list(task.patrol_waypoints) + [task.delivery_pos]
                for idx, waypoint_target in enumerate(patrol_points):
                    route = self.plan(
                        current_pos,
                        waypoint_target,
                        drone_radius=drone.safety_radius,
                        flight_level=task_level,
                        payload_weight=getattr(task, "payload_weight", 0.0),
                        departure_time=current_time,
                        cruise_speed=drone.cruise_speed,
                        reserve_corridor=True,
                    )
                    segment_dist = self._segment_distance(route)
                    current_time += segment_dist / max(drone.cruise_speed, 0.1)
                    total_dist += segment_dist
                    if all_waypoints and route:
                        route = route[1:]
                    all_waypoints.extend(route)
                    if all_waypoints:
                        if idx == 0:
                            all_waypoints[-1].action = "pickup"
                        elif idx == len(patrol_points) - 1:
                            all_waypoints[-1].action = "delivery"
                        else:
                            all_waypoints[-1].action = "hover"
                    current_pos = np.asarray(waypoint_target, dtype=float)
                continue

            route1 = self.plan(
                current_pos,
                task.pickup_pos,
                drone_radius=drone.safety_radius,
                flight_level=task_level,
                payload_weight=0.0,
                departure_time=current_time,
                cruise_speed=drone.cruise_speed,
                reserve_corridor=True,
            )
            dist1 = self._segment_distance(route1)
            current_time += dist1 / max(drone.cruise_speed, 0.1) + getattr(task, "pickup_service_time", 0.0)
            total_dist += dist1
            if all_waypoints and route1:
                route1 = route1[1:]
            all_waypoints.extend(route1)
            if all_waypoints:
                all_waypoints[-1].action = "pickup"

            route2 = self.plan(
                task.pickup_pos,
                task.delivery_pos,
                drone_radius=drone.safety_radius,
                flight_level=task_level,
                payload_weight=getattr(task, "payload_weight", 0.0),
                departure_time=current_time,
                cruise_speed=drone.cruise_speed,
                reserve_corridor=True,
            )
            dist2 = self._segment_distance(route2)
            current_time += dist2 / max(drone.cruise_speed, 0.1) + getattr(task, "delivery_service_time", 0.0)
            total_dist += dist2
            if all_waypoints and route2:
                route2 = route2[1:]
            all_waypoints.extend(route2)
            if all_waypoints:
                all_waypoints[-1].action = "delivery"

            current_pos = np.asarray(task.delivery_pos, dtype=float)

        return all_waypoints, total_dist

    def _segment_distance(self, route: List[Waypoint]) -> float:
        if len(route) < 2:
            return 0.0
        total = 0.0
        for i in range(len(route) - 1):
            total += float(np.linalg.norm(route[i + 1].position - route[i].position))
        return total

    def _corridor_cell_from_world(self, pos: np.ndarray) -> Tuple[int, int]:
        x = int(math.floor(float(pos[0]) / max(self.corridor_cell_size, 1.0)))
        z = int(math.floor(float(pos[2]) / max(self.corridor_cell_size, 1.0)))
        return x, z

    def _dedupe_positions(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        deduped: List[np.ndarray] = []
        for pos in positions:
            pos = np.asarray(pos, dtype=float)
            if not deduped or np.linalg.norm(pos - deduped[-1]) > 1e-3:
                deduped.append(pos)
        return deduped

    def _compress_control_points(self, positions: List[np.ndarray], max_points: int = 16) -> List[np.ndarray]:
        positions = self._dedupe_positions(positions)
        if len(positions) <= max_points:
            return positions

        turn_scores = []
        for i in range(1, len(positions) - 1):
            v1 = positions[i] - positions[i - 1]
            v2 = positions[i + 1] - positions[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                score = 1.0 - float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            else:
                score = 0.0
            turn_scores.append((score, i))

        keep = {0, len(positions) - 1}
        for _, idx in sorted(turn_scores, reverse=True)[: max(0, max_points - 2)]:
            keep.add(idx)

        compressed = [positions[i] for i in sorted(keep)]
        if len(compressed) > max_points:
            indices = np.linspace(0, len(compressed) - 1, max_points).astype(int)
            compressed = [compressed[i] for i in indices]
        return compressed

    def _resample_positions(self, positions: List[np.ndarray], target_count: int) -> List[np.ndarray]:
        if len(positions) <= 2 or len(positions) <= target_count:
            return positions

        pts = [np.asarray(p, dtype=float) for p in positions]
        seg_lengths = [float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])]
        total_length = sum(seg_lengths)
        if total_length < 1e-6:
            return positions

        targets = np.linspace(0.0, total_length, target_count)
        resampled = [pts[0].copy()]
        cum = 0.0
        seg_idx = 0
        for target in targets[1:-1]:
            while seg_idx < len(seg_lengths) - 1 and cum + seg_lengths[seg_idx] < target:
                cum += seg_lengths[seg_idx]
                seg_idx += 1
            seg_len = max(seg_lengths[seg_idx], 1e-6)
            ratio = (target - cum) / seg_len
            point = pts[seg_idx] + ratio * (pts[seg_idx + 1] - pts[seg_idx])
            resampled.append(point)
        resampled.append(pts[-1].copy())
        return resampled

    def clear_cache(self):
        self._path_cache.clear()
