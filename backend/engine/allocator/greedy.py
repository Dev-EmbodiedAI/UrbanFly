"""
贪心最近邻 — 去中心化任务分配 baseline
=====================================
每架无人机独立选择距离最近的任务，无协调机制。

特点：计算极快，但分配质量最低，容易多个无人机争抢同一任务。
"""

import numpy as np
from typing import List, Dict, Optional
import time as _time

from .base import BaseAllocator
from ..models import Task, DroneStateData
from ...config import PRIORITY_WEIGHTS


class GreedyAllocator(BaseAllocator):
    """贪心最近邻分配器"""

    def __init__(self, max_tasks_per_drone: int = 5):
        super().__init__(name="greedy")
        self.max_tasks_per_drone = max_tasks_per_drone

    def allocate(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        """每架无人机独立抢最近的任务（有冲突时先到先得）"""
        t_start = _time.perf_counter()

        pending = [t for t in tasks if t.status.value not in ("completed", "failed", "cancelled")]
        assigned_set = set()
        assignments = {d.id: [] for d in drones}

        for drone in sorted(drones, key=lambda d: d.id):  # 按ID排序模拟先到先得
            if not pending:
                break

            # 计算此无人机到每个待分配任务的距离
            scores = []
            for task in pending:
                if task.id in assigned_set:
                    continue
                dist = np.linalg.norm(drone.position - task.pickup_pos)
                priority_weight = PRIORITY_WEIGHTS.get(task.priority, 1.0)
                score = dist / priority_weight  # 越小越好
                scores.append((score, task))

            scores.sort(key=lambda x: x[0])

            # 取最近的 max_tasks_per_drone 个
            for i in range(min(self.max_tasks_per_drone, len(scores))):
                task = scores[i][1]
                assignments[drone.id].append(task.id)
                assigned_set.add(task.id)

            # 更新 pending
            pending = [t for t in pending if t.id not in assigned_set]

        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        return self.allocate(drones, tasks, comm_graph, current_time)
