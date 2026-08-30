"""
任务分配器基类
=============
定义所有分配算法必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np
import time as _time

from ..models import Task, DroneStateData, SimulationStats


class BaseAllocator(ABC):
    """
    任务分配器抽象基类。

    所有分配算法（CBBA、匈牙利、贪心、拍卖、遗传、市场）都继承此类。
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self.last_runtime_ms: float = 0.0
        self.iteration_count: int = 0

    @abstractmethod
    def allocate(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        """
        执行任务分配。

        Args:
            drones: 无人机状态列表
            tasks: 待分配任务列表（含已分配和待分配）
            comm_graph: N×N 通信邻接矩阵 (None = 全连通)
            current_time: 当前仿真时间

        Returns:
            Dict[str, List[str]]: drone_id → [task_id, ...] 分配映射
        """
        pass

    @abstractmethod
    def run_iteration(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        """
        运行一次算法迭代（用于去中心化算法的增量更新）。

        对于集中式算法（匈牙利、遗传），此方法等同于 allocate()。
        对于去中心化算法（CBBA、拍卖），每次调用运行一轮共识。

        Returns:
            当前分配映射
        """
        pass

    def get_stats(self) -> Dict:
        """返回算法运行统计"""
        return {
            "algorithm": self.name,
            "iterations": self.iteration_count,
            "last_runtime_ms": self.last_runtime_ms,
        }

    def reset(self):
        """重置算法状态"""
        self.last_runtime_ms = 0.0
        self.iteration_count = 0

    # ------------------------------------------------------------------
    # 工具方法（子类共用）
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_path_distance(
        drone: DroneStateData,
        bundle: List[Task],
        path_cache: Dict = None,
    ) -> float:
        """
        估算执行bundle中任务的总路径距离。

        使用简化的2-opt序列：
        drone.pos → bundle[0].pickup → bundle[0].delivery → bundle[1].pickup → ...

        注：这是估算（直线距离），实际路径规划会给出精确距离。

        Args:
            drone: 无人机状态
            bundle: 任务列表
            path_cache: 距离缓存 {(from_pos, to_pos): distance}
        """
        if not bundle:
            return 0.0

        cache = path_cache or {}
        total_dist = 0.0
        current_pos = drone.position

        for task in bundle:
            # 到取件点
            key1 = (tuple(current_pos), tuple(task.pickup_pos))
            if key1 in cache:
                dist1 = cache[key1]
            else:
                dist1 = np.linalg.norm(current_pos - task.pickup_pos)
                cache[key1] = dist1

            total_dist += dist1

            # 取件点到递送点
            if task.is_patrol and task.patrol_waypoints:
                # 巡检任务：遍历所有航点
                patrol_pos = task.pickup_pos
                for wp in task.patrol_waypoints:
                    key_p = (tuple(patrol_pos), tuple(wp))
                    if key_p in cache:
                        dp = cache[key_p]
                    else:
                        dp = np.linalg.norm(patrol_pos - wp)
                        cache[key_p] = dp
                    total_dist += dp
                    patrol_pos = wp
                current_pos = task.delivery_pos
            else:
                key2 = (tuple(task.pickup_pos), tuple(task.delivery_pos))
                if key2 in cache:
                    dist2 = cache[key2]
                else:
                    dist2 = np.linalg.norm(task.pickup_pos - task.delivery_pos)
                    cache[key2] = dist2
                total_dist += dist2
                current_pos = task.delivery_pos

        return total_dist

    @staticmethod
    def _estimate_arrival_times(
        path_distance: float,
        speed: float,
        start_time: float = 0.0,
    ) -> float:
        """估算到达时间"""
        if speed <= 0:
            return float("inf")
        return start_time + path_distance / speed

    @staticmethod
    def _check_battery_feasibility(
        drone: DroneStateData,
        bundle: List[Task],
        path_distance: float,
    ) -> bool:
        """
        检查电池是否足够完成bundle中所有任务并返回最近充电站。

        保留15%安全余量。
        """
        # 总能耗估算
        avg_payload = sum(t.payload_weight for t in bundle) / max(len(bundle), 1)
        total_energy = (
            path_distance * drone.energy_per_meter
            + path_distance * drone.energy_per_kg_meter * avg_payload
        )

        # 返回充电站的距离（简化：取场景中心最近充电站）
        # 实际需要知道充电站位置，这里用粗略估计
        return_home_dist = 200.0  # 粗略估计
        return_energy = return_home_dist * drone.energy_per_meter

        total_needed = total_energy + return_energy
        available = drone.battery_remaining * 0.85  # 15% 安全余量

        return total_needed <= available
