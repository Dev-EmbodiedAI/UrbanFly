"""
匈牙利算法 — 集中式任务分配 baseline
====================================
使用 scipy.optimize.linear_sum_assignment 求解最小成本分配问题。

特点：全局最优解，但需要全局信息（中心节点），无法处理通信受阻。
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.optimize import linear_sum_assignment
import time as _time

from .base import BaseAllocator
from ..models import Task, DroneStateData
from ...config import PRIORITY_WEIGHTS


class HungarianAllocator(BaseAllocator):
    """基于匈牙利算法的集中式任务分配"""

    def __init__(self):
        super().__init__(name="hungarian")

    def allocate(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        """执行匈牙利分配（支持多任务分配，通过多轮分配实现）"""
        t_start = _time.perf_counter()

        M = len(drones)
        pending_tasks = [t for t in tasks if t.status.value not in ("completed", "failed", "cancelled")]

        if not pending_tasks:
            return {d.id: [] for d in drones}

        # 每架无人机最多分配 max_tasks_per_drone 个任务
        max_per_drone = 5
        assignments = {d.id: [] for d in drones}

        # 多轮分配：每轮每架无人机最多一个任务
        for round_idx in range(max_per_drone):
            if not pending_tasks:
                break

            N = min(len(pending_tasks), M)
            if N == 0:
                break

            # 构建成本矩阵
            cost = np.zeros((M, M))
            for i, drone in enumerate(drones):
                for j in range(M):
                    if j < len(pending_tasks):
                        task = pending_tasks[j]
                        dist = np.linalg.norm(drone.position - task.pickup_pos) + \
                               np.linalg.norm(task.pickup_pos - task.delivery_pos)
                        priority_weight = PRIORITY_WEIGHTS.get(task.priority, 1.0)
                        cost[i, j] = dist / priority_weight
                    else:
                        cost[i, j] = 1e9  # 虚拟任务

            # 求解
            row_ind, col_ind = linear_sum_assignment(cost)

            # 更新分配
            assigned_this_round = set()
            for r, c in zip(row_ind, col_ind):
                if c < len(pending_tasks) and cost[r, c] < 1e8:
                    drone = drones[r]
                    task = pending_tasks[c]
                    assignments[drone.id].append(task.id)
                    assigned_this_round.add(c)

            # 移除已分配的任务
            pending_tasks = [t for i, t in enumerate(pending_tasks) if i not in assigned_this_round]

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
