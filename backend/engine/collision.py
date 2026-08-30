"""
碰撞检测与避免
=============
多无人机间的冲突检测和解决。

方法：
1. 时空冲突检测：两机在相同时间窗口内距离 < 安全距离
2. 速度障碍法 (Velocity Obstacle)：动态避碰
3. 高度分离：冲突路径垂直偏移
4. 速度调整：改变到达冲突点的时间
5. 重新规划：以上均无效时重新规划路径
"""

import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations
from pathlib import Path
from scipy.spatial import cKDTree

from .models import DroneStateData, SimulationEvent, EventType, Waypoint
from ..config import COLLISION_AVOIDANCE


class HeightmapStaticCollisionMap:
    """0.5 m 摄影测量表面碰撞场。

    高度图保存每个 X/Z 单元的最高三角面。查询时在安全半径对应的邻域取最高
    表面，因此建筑立面会在机体中心进入建筑轮廓前触发，属于适合飞行安全的
    保守碰撞代理。
    """

    def __init__(self, height, origin_x, origin_z, resolution):
        self.height = np.asarray(height, dtype=np.float32)
        self.origin_x = float(origin_x)
        self.origin_z = float(origin_z)
        self.resolution = float(resolution)
        self.shape = self.height.shape
        self.maximum_z = self.origin_z + (self.shape[0] - 1) * self.resolution
        finite = self.height[np.isfinite(self.height)]
        self.floor_height = float(np.min(finite)) if finite.size else 0.0
        self.voxel_count = int(np.count_nonzero(np.isfinite(self.height)))

    @classmethod
    def load(cls, path) -> "HeightmapStaticCollisionMap":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                data["height_m"],
                float(data["origin_x_m"]),
                float(data["origin_z_m"]),
                float(data["resolution_m"]),
            )

    def _indices(self, position: np.ndarray):
        point = np.asarray(position, dtype=float)
        column = int(np.floor((point[0] - self.origin_x) / self.resolution))
        row = int(np.floor((self.maximum_z - point[2]) / self.resolution))
        return row, column

    def surface_height(self, position: np.ndarray, safety_radius: float = 0.0) -> float:
        row, column = self._indices(position)
        if row < 0 or column < 0 or row >= self.shape[0] or column >= self.shape[1]:
            return float("inf")
        radius_cells = max(0, int(np.ceil(float(safety_radius) / self.resolution)))
        row_min = max(0, row - radius_cells)
        row_max = min(self.shape[0], row + radius_cells + 1)
        col_min = max(0, column - radius_cells)
        col_max = min(self.shape[1], column + radius_cells + 1)
        patch = self.height[row_min:row_max, col_min:col_max]
        finite = patch[np.isfinite(patch)]
        return float(np.max(finite)) if finite.size else self.floor_height

    def clearance(self, position: np.ndarray, safety_radius: float = 0.0) -> float:
        point = np.asarray(position, dtype=float)
        surface = self.surface_height(point, safety_radius)
        if not np.isfinite(surface):
            return -float("inf")
        return float(point[1] - surface)

    def collides(self, position: np.ndarray, safety_radius: float) -> Tuple[bool, float]:
        clearance = self.clearance(position, safety_radius)
        return clearance < float(safety_radius), clearance

    def sweep_collides(
        self,
        start: np.ndarray,
        end: np.ndarray,
        safety_radius: float,
        step: float = None,
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        length = float(np.linalg.norm(end - start))
        sample_step = float(step or min(self.resolution * 0.5, 0.25))
        count = max(1, int(np.ceil(length / max(sample_step, 1e-3))))
        minimum = float("inf")
        hit = None
        for alpha in np.linspace(0.0, 1.0, count + 1):
            point = start + float(alpha) * (end - start)
            clearance = self.clearance(point, safety_radius)
            if clearance < minimum:
                minimum = clearance
            if hit is None and clearance < float(safety_radius):
                hit = point.copy()
        return hit is not None, minimum, hit

    def audit_polyline(
        self,
        positions: np.ndarray,
        safety_radius: float,
    ) -> Dict[str, object]:
        """Validate every segment of a route against the photogrammetry surface."""
        points = np.asarray(positions, dtype=np.float32)
        if len(points) < 2:
            clearance = (
                self.clearance(points[0], safety_radius)
                if len(points)
                else -float("inf")
            )
            return {
                "valid": clearance >= safety_radius,
                "minimum_clearance_m": clearance,
                "collision_segment": None,
                "collision_position": None,
            }

        minimum = float("inf")
        first_collision = None
        for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            collides, clearance, hit = self.sweep_collides(
                start,
                end,
                safety_radius,
            )
            minimum = min(minimum, clearance)
            if collides and first_collision is None:
                first_collision = (
                    segment_index,
                    hit.tolist() if hit is not None else None,
                )
        return {
            "valid": first_collision is None,
            "minimum_clearance_m": minimum,
            "collision_segment": first_collision[0] if first_collision else None,
            "collision_position": first_collision[1] if first_collision else None,
        }


class SparseStaticCollisionMap:
    """Sparse CityGS surface voxels with metric nearest-obstacle queries."""

    def __init__(self, coords, origin, resolution, shape):
        self.coords = np.asarray(coords, dtype=np.uint16)
        self.origin = np.asarray(origin, dtype=np.float32)
        self.resolution = float(resolution)
        self.shape = tuple(int(v) for v in shape)
        self.voxel_half_diagonal = np.sqrt(3.0) * self.resolution * 0.5
        centers = self.origin + (self.coords.astype(np.float32) + 0.5) * self.resolution
        self.tree = cKDTree(centers, compact_nodes=True, balanced_tree=True)
        self.voxel_count = int(len(self.coords))

    @classmethod
    def load(cls, path) -> "SparseStaticCollisionMap":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                data["coords"],
                data["origin"],
                float(data["resolution"]),
                data["shape"],
            )

    def clearance(self, position: np.ndarray) -> float:
        distance, _ = self.tree.query(np.asarray(position, dtype=float), k=1, workers=1)
        return max(0.0, float(distance) - self.voxel_half_diagonal)

    def batch_clearance(self, positions: np.ndarray) -> np.ndarray:
        distances, _ = self.tree.query(
            np.asarray(positions, dtype=float), k=1, workers=1
        )
        return np.maximum(0.0, distances - self.voxel_half_diagonal)

    def collides(self, position: np.ndarray, safety_radius: float) -> Tuple[bool, float]:
        clearance = self.clearance(position)
        return clearance < float(safety_radius), clearance


class DenseSignedDistanceField:
    """Dense global ESDF with metric trilinear queries.

    Samples are stored at voxel centres. Negative values are inside static
    solids; positions outside the reconstructed 500 m volume are deliberately
    treated as occupied so an aircraft cannot leave the mapped world.
    """

    def __init__(self, distance, origin, resolution, truncation):
        self.distance = np.asarray(distance, dtype=np.float16)
        self.origin = np.asarray(origin, dtype=np.float32)
        self.resolution = float(resolution)
        self.truncation = float(truncation)
        self.shape = tuple(int(v) for v in self.distance.shape)
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError("ESDF must be a 3D array with at least two cells per axis")

    @classmethod
    def load(cls, path) -> "DenseSignedDistanceField":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                data["distance"],
                data["origin"],
                float(data["resolution"]),
                float(data["truncation"]),
            )

    def batch_clearance(self, positions: np.ndarray) -> np.ndarray:
        points = np.asarray(positions, dtype=np.float32)
        original_shape = points.shape[:-1]
        points = points.reshape(-1, 3)
        grid = (points - self.origin) / self.resolution - 0.5
        shape = np.asarray(self.shape, dtype=np.int64)
        valid = np.all((grid >= 0.0) & (grid <= shape - 1), axis=1)
        result = np.full(len(points), -self.truncation, dtype=np.float32)
        if not np.any(valid):
            return result.reshape(original_shape)

        coords = grid[valid]
        lower = np.floor(coords).astype(np.int64)
        lower = np.minimum(lower, shape - 2)
        lower = np.maximum(lower, 0)
        fraction = coords - lower
        upper = lower + 1

        x0, y0, z0 = lower.T
        x1, y1, z1 = upper.T
        tx, ty, tz = fraction.T
        field = self.distance
        c000 = field[x0, y0, z0].astype(np.float32)
        c100 = field[x1, y0, z0].astype(np.float32)
        c010 = field[x0, y1, z0].astype(np.float32)
        c110 = field[x1, y1, z0].astype(np.float32)
        c001 = field[x0, y0, z1].astype(np.float32)
        c101 = field[x1, y0, z1].astype(np.float32)
        c011 = field[x0, y1, z1].astype(np.float32)
        c111 = field[x1, y1, z1].astype(np.float32)

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        result[valid] = c0 * (1.0 - tz) + c1 * tz
        return result.reshape(original_shape)

    def clearance(self, position: np.ndarray) -> float:
        return float(self.batch_clearance(np.asarray(position, dtype=np.float32)[None])[0])

    def collides(self, position: np.ndarray, safety_radius: float) -> Tuple[bool, float]:
        clearance = self.clearance(position)
        return clearance < float(safety_radius), clearance


class RuntimeLocalESDFTileCache:
    """Lazy 0.25 m ESDF tiles for detailed planner/debug queries.

    A dense 500 x 170 x 500 m field at 0.25 m would contain 2.72 billion
    samples. This cache materializes only 4 m cubes around queried positions,
    keeping a bounded float16 LRU working set while preserving the real sparse
    CityGS surface as the distance source.
    """

    def __init__(
        self,
        global_esdf: DenseSignedDistanceField,
        local_surface: SparseStaticCollisionMap,
        block_size: int = 16,
        max_blocks: int = 256,
        truncation: float = 16.0,
    ):
        self.global_esdf = global_esdf
        self.local_surface = local_surface
        self.resolution = float(local_surface.resolution)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.truncation = float(truncation)
        self.block_extent = self.resolution * self.block_size
        self._blocks: OrderedDict[Tuple[int, int, int], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _block_key(self, position: np.ndarray) -> Tuple[int, int, int]:
        relative = (np.asarray(position, dtype=np.float32) - self.local_surface.origin)
        return tuple(np.floor(relative / self.block_extent).astype(np.int64))

    def _build_block(self, key: Tuple[int, int, int]) -> np.ndarray:
        block_origin = (
            self.local_surface.origin
            + np.asarray(key, dtype=np.float32) * self.block_extent
        )
        axis = (np.arange(self.block_size, dtype=np.float32) + 0.5) * self.resolution
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.column_stack(
            (
                gx.ravel() + block_origin[0],
                gy.ravel() + block_origin[1],
                gz.ravel() + block_origin[2],
            )
        )
        local_distance = self.local_surface.batch_clearance(points)
        global_distance = self.global_esdf.batch_clearance(points)
        distance = np.minimum(local_distance, global_distance)
        distance = np.clip(distance, -self.truncation, self.truncation)
        return distance.reshape((self.block_size,) * 3).astype(np.float16)

    def get_block(self, position: np.ndarray) -> Tuple[Tuple[int, int, int], np.ndarray]:
        key = self._block_key(position)
        if key in self._blocks:
            self.hits += 1
            self._blocks.move_to_end(key)
            return key, self._blocks[key]
        self.misses += 1
        block = self._build_block(key)
        self._blocks[key] = block
        if len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        return key, block

    def clearance(self, position: np.ndarray) -> float:
        key, block = self.get_block(position)
        block_origin = (
            self.local_surface.origin
            + np.asarray(key, dtype=np.float32) * self.block_extent
        )
        index = np.floor(
            (np.asarray(position, dtype=np.float32) - block_origin) / self.resolution
        ).astype(np.int64)
        index = np.clip(index, 0, self.block_size - 1)
        return float(block[tuple(index)])

    def stats(self) -> Dict[str, float]:
        return {
            "resolution_m": self.resolution,
            "block_size": self.block_size,
            "block_extent_m": self.block_extent,
            "resident_blocks": len(self._blocks),
            "max_blocks": self.max_blocks,
            "resident_megabytes": len(self._blocks) * self.block_size ** 3 * 2 / 1e6,
            "hits": self.hits,
            "misses": self.misses,
        }


class HierarchicalStaticCollisionMap:
    """Global signed ESDF + exact 0.25 m CityGS detail with swept queries."""

    def __init__(
        self,
        global_esdf: DenseSignedDistanceField,
        local_surface: SparseStaticCollisionMap,
    ):
        self.global_esdf = global_esdf
        self.local_surface = local_surface
        self.local_tiles = RuntimeLocalESDFTileCache(global_esdf, local_surface)
        self.resolution = local_surface.resolution
        self.global_resolution = global_esdf.resolution
        self.voxel_count = local_surface.voxel_count

    def batch_clearance(self, positions: np.ndarray) -> np.ndarray:
        points = np.asarray(positions, dtype=np.float32)
        return np.minimum(
            self.global_esdf.batch_clearance(points),
            self.local_surface.batch_clearance(points),
        )

    def clearance(self, position: np.ndarray) -> float:
        return float(self.batch_clearance(np.asarray(position)[None])[0])

    def collides(self, position: np.ndarray, safety_radius: float) -> Tuple[bool, float]:
        clearance = self.clearance(position)
        return clearance < float(safety_radius), clearance

    def sweep_collides(
        self,
        start: np.ndarray,
        end: np.ndarray,
        safety_radius: float,
        step: float = None,
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        length = float(np.linalg.norm(end - start))
        sample_step = float(step or min(self.resolution * 0.5, 0.25))
        count = max(1, int(np.ceil(length / max(sample_step, 1e-3))))
        alpha = np.linspace(0.0, 1.0, count + 1, dtype=np.float32)
        points = start[None] + alpha[:, None] * (end - start)[None]
        clearances = self.batch_clearance(points)
        index = int(np.argmin(clearances))
        minimum = float(clearances[index])
        collides = minimum < float(safety_radius)
        return collides, minimum, points[index].copy() if collides else None

    def audit_polyline(
        self,
        positions: np.ndarray,
        safety_radius: float,
    ) -> Dict[str, object]:
        points = np.asarray(positions, dtype=np.float32)
        if len(points) < 2:
            clearance = self.clearance(points[0]) if len(points) else -self.global_esdf.truncation
            return {
                "valid": clearance >= safety_radius,
                "minimum_clearance_m": clearance,
                "collision_segment": None,
                "collision_position": None,
            }
        minimum = float("inf")
        first_collision = None
        for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            collides, clearance, hit = self.sweep_collides(
                start, end, safety_radius
            )
            minimum = min(minimum, clearance)
            if collides and first_collision is None:
                first_collision = (
                    segment_index,
                    hit.tolist() if hit is not None else None,
                )
        return {
            "valid": first_collision is None,
            "minimum_clearance_m": minimum,
            "collision_segment": first_collision[0] if first_collision else None,
            "collision_position": first_collision[1] if first_collision else None,
        }


class CollisionManager:
    """
    无人机碰撞检测与避免管理器。

    每仿真步检查潜在冲突，必要时执行避碰动作。
    """

    def __init__(self,
                 safety_radius: float = None,
                 warning_time: float = None,
                 time_horizon: float = None):
        self.safety_radius = safety_radius or COLLISION_AVOIDANCE["drone_safety_cylinder_radius"]
        self.warning_time = warning_time or COLLISION_AVOIDANCE["collision_warning_time"]
        self.time_horizon = time_horizon or COLLISION_AVOIDANCE["velocity_obstacle_time_horizon"]
        self.altitude_separation = COLLISION_AVOIDANCE["altitude_separation"]

        # 统计
        self.warnings_issued: int = 0
        self.collisions_avoided: int = 0
        self.path_replans_due_to_collision: int = 0

    def reset(self):
        self.warnings_issued = 0
        self.collisions_avoided = 0
        self.path_replans_due_to_collision = 0

    # ==================================================================
    # 冲突检测
    # ==================================================================

    def detect_conflicts(
        self,
        drones: List[DroneStateData],
        dt: float,
    ) -> List[Tuple[str, str, float, float]]:
        """
        检测所有无人机对之间的潜在冲突。

        Returns:
            List of (drone_id1, drone_id2, time_to_conflict, min_distance)
        """
        conflicts = []

        for d1, d2 in combinations(drones, 2):
            conflict = self._check_pair_conflict(d1, d2, dt)
            if conflict:
                time_to, min_dist = conflict
                conflicts.append((d1.id, d2.id, time_to, min_dist))

        # 按冲突紧急程度排序
        conflicts.sort(key=lambda x: x[2])  # time_to_conflict 升序
        return conflicts

    def _check_pair_conflict(
        self,
        d1: DroneStateData,
        d2: DroneStateData,
        dt: float,
    ) -> Optional[Tuple[float, float]]:
        """
        检查两架无人机是否在未来时间视野内发生冲突。

        方法：向前投影位置，检查最小距离。

        Returns:
            (time_to_conflict, min_distance) 或 None
        """
        pos1 = d1.position.copy()
        vel1 = d1.velocity.copy()
        pos2 = d2.position.copy()
        vel2 = d2.velocity.copy()

        # 当前距离
        current_dist = np.linalg.norm(pos1 - pos2)
        if current_dist < self.safety_radius:
            return (0.0, current_dist)

        # 相对速度
        rel_vel = vel1 - vel2
        rel_speed = np.linalg.norm(rel_vel)

        if rel_speed < 0.01:
            # 几乎静止 → 检查距离
            if current_dist < self.safety_radius * 2:
                return (0.0, current_dist)
            return None

        # 最近接近时间 (CPA: Closest Point of Approach)
        rel_pos = pos1 - pos2
        t_cpa = -np.dot(rel_pos, rel_vel) / (rel_speed ** 2)
        t_cpa = max(0, t_cpa)

        if t_cpa > self.time_horizon:
            return None

        # CPA距离
        cpa_pos1 = pos1 + vel1 * t_cpa
        cpa_pos2 = pos2 + vel2 * t_cpa
        cpa_dist = np.linalg.norm(cpa_pos1 - cpa_pos2)

        if cpa_dist < self.safety_radius:
            return (t_cpa, cpa_dist)

        return None

    # ==================================================================
    # 冲突解决
    # ==================================================================

    def resolve_conflicts(
        self,
        drones: List[DroneStateData],
        conflicts: List[Tuple[str, str, float, float]],
        dt: float,
    ) -> List[SimulationEvent]:
        """
        解决检测到的冲突。

        策略（按优先级）：
        1. 高度分离：较低优先级的下降/上升
        2. 速度调整：加速/减速改变到达时间
        3. 标记需要重规划

        Returns:
            生成的仿真事件列表
        """
        events = []
        modified_drones: Set[str] = set()

        for d1_id, d2_id, time_to, min_dist in conflicts:
            if time_to > self.warning_time:
                continue  # 冲突太远，暂不处理

            d1 = next(d for d in drones if d.id == d1_id)
            d2 = next(d for d in drones if d.id == d2_id)

            # 确定优先级：任务优先级高的优先通行
            d1_priority = self._get_drone_priority(d1)
            d2_priority = self._get_drone_priority(d2)

            low_priority = d1 if d1_priority > d2_priority else d2
            high_priority = d2 if d1_priority > d2_priority else d1

            # 策略1：高度分离
            if d1_id not in modified_drones and d2_id not in modified_drones:
                if self._altitude_separation(low_priority, high_priority):
                    events.append(SimulationEvent(
                        time=0,  # 由simulator填充
                        event_type=EventType.COLLISION_AVOIDED,
                        drone_id=low_priority.id,
                        message=f"高度分离避碰: 与 {high_priority.id}"
                    ))
                    modified_drones.add(low_priority.id)
                    self.collisions_avoided += 1
                    continue

            # 策略2：速度调整
            if d1_id not in modified_drones and d2_id not in modified_drones:
                speed_changed = self._speed_adjustment(low_priority, high_priority)
                if speed_changed:
                    events.append(SimulationEvent(
                        time=0,
                        event_type=EventType.COLLISION_AVOIDED,
                        drone_id=low_priority.id,
                        message=f"速度调整避碰: 与 {high_priority.id}"
                    ))
                    modified_drones.add(low_priority.id)
                    self.collisions_avoided += 1
                    continue

            # 策略3：标记需要重规划
            if time_to < 5.0:  # 紧急
                events.append(SimulationEvent(
                    time=0,
                    event_type=EventType.COLLISION_WARNING,
                    drone_id=d1_id,
                    message=f"需要重新规划路径以避免与 {d2_id} 碰撞"
                ))
                self.warnings_issued += 1

        return events

    def _get_drone_priority(self, drone: DroneStateData) -> int:
        """获取无人机的任务优先级（数字越小优先级越高）"""
        # 有紧急任务的优先
        if drone.current_task_id:
            # 任务priority映射到无人机优先级
            if drone.state.value in ("delivering", "en_route"):
                return 0  # 执行任务中的最高优先
        if drone.is_battery_critical:
            return 0  # 电池紧急
        if drone.state.value == "returning":
            return 1
        return 5  # 空闲

    def _altitude_separation(self, low_priority_drone, high_priority_drone) -> bool:
        """
        高度分离：低优先级无人机爬升或下降以避让。

        Returns:
            是否成功执行高度分离
        """
        alt_diff = low_priority_drone.position[1] - high_priority_drone.position[1]

        if abs(alt_diff) < self.altitude_separation:
            # 需要分离
            target_alt = high_priority_drone.position[1] + np.sign(alt_diff + 0.01) * self.altitude_separation

            # 限制爬升速率
            current_alt = low_priority_drone.position[1]
            max_step = low_priority_drone.max_climb_rate * 2  # 一个时间步的步骤
            new_alt = current_alt + np.clip(target_alt - current_alt, -max_step, max_step)

            # 调整速度向量中的垂直分量
            low_priority_drone.velocity[1] += (new_alt - current_alt) / 0.05  # dt=0.05
            return True

        return False  # 已有足够分离

    def _speed_adjustment(self, low_priority_drone, high_priority_drone) -> bool:
        """调整速度来改变到达冲突点的时间"""
        # 减速10%
        speed = np.linalg.norm(low_priority_drone.velocity)
        factor = COLLISION_AVOIDANCE["speed_adjustment_factor"]
        if speed > 1.0:
            low_priority_drone.velocity *= (1 - factor)
            return True
        return False

    # ==================================================================
    # 速度障碍法 (Velocity Obstacle)
    # ==================================================================

    def compute_velocity_obstacle(
        self,
        drone: DroneStateData,
        other_drones: List[DroneStateData],
    ) -> np.ndarray:
        """
        计算速度障碍，返回调整后的速度向量。

        简化VO实现：
        - 对每架邻近无人机计算VO锥
        - 选择VO锥外最接近当前速度的可行速度
        """
        current_vel = drone.velocity.copy()
        if np.linalg.norm(current_vel) < 0.1:
            return current_vel  # 静止时VO不适用

        best_vel = current_vel
        best_cost = float("inf")
        candidates = self._generate_velocity_candidates(current_vel, drone.max_speed)

        for cand_vel in candidates:
            collision = False
            for other in other_drones:
                if other.id == drone.id:
                    continue
                if self._vo_collision_check(drone.position, cand_vel,
                                            other.position, other.velocity,
                                            self.safety_radius, self.time_horizon):
                    collision = True
                    break

            if not collision:
                cost = np.linalg.norm(cand_vel - current_vel)
                if cost < best_cost:
                    best_cost = cost
                    best_vel = cand_vel

        return best_vel

    def _generate_velocity_candidates(self, current_vel, max_speed, n=8):
        """生成候选速度向量"""
        candidates = [current_vel]  # 保持当前速度

        speed = np.linalg.norm(current_vel)
        dir_vec = current_vel / (speed + 1e-6)

        # 速度变化候选
        for factor in [0.7, 0.85, 1.15, 1.3]:
            cand = current_vel * factor
            if np.linalg.norm(cand) <= max_speed:
                candidates.append(cand)

        # 方向变化候选
        for angle in [-30, -15, 15, 30]:
            rad = np.radians(angle)
            rot = np.array([
                [np.cos(rad), 0, -np.sin(rad)],
                [0, 1, 0],
                [np.sin(rad), 0, np.cos(rad)],
            ])
            cand = rot @ current_vel
            if np.linalg.norm(cand) <= max_speed:
                candidates.append(cand)

        return candidates

    def _vo_collision_check(self, pos_a, vel_a, pos_b, vel_b, radius, time_horizon) -> bool:
        """检查速度a是否会导致与b碰撞（VO核心逻辑）"""
        rel_pos = pos_a - pos_b
        rel_vel = vel_a - vel_b

        # 解二次方程: |rel_pos + rel_vel * t|^2 = radius^2
        a = np.dot(rel_vel, rel_vel)
        b = 2 * np.dot(rel_pos, rel_vel)
        c = np.dot(rel_pos, rel_pos) - radius ** 2

        if a < 1e-10:
            return c <= 0

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return False

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        return (0 <= t1 <= time_horizon) or (0 <= t2 <= time_horizon)
