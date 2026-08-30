"""
基于市场机制 — 去中心化任务分配 baseline
=========================================
类似CBBA的bundle构建但没有共识阶段，每架无人机独立"购买"任务。

特点：作为CBBA的直接对比 — 有bundle优化但无协商协调。
"""

import numpy as np
from typing import List, Dict, Optional
import time as _time

from .base import BaseAllocator
from ..models import Task, DroneStateData
from ...config import PRIORITY_WEIGHTS, CBBA


class MarketAllocator(BaseAllocator):
    """基于市场机制的无共识分配器"""

    def __init__(self, max_tasks_per_drone: int = 8):
        super().__init__(name="market")
        self.max_tasks_per_drone = max_tasks_per_drone

    def allocate(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        """
        市场机制分配：
        1. 所有待分配任务进入"市场"
        2. 每架无人机对市场中的任务"出价"（基于边际收益）
        3. 每轮最高出价者获得任务
        4. 无共识阶段 — 无人机只知道自己赢得的任务
        """
        t_start = _time.perf_counter()

        pending = [t for t in tasks if t.status.value not in ("completed", "failed", "cancelled")]
        if not pending:
            return {d.id: [] for d in drones}

        # 每架无人机的当前bundle
        bundles: Dict[str, List[Task]] = {d.id: [] for d in drones}
        unassigned = list(pending)
        market_tasks = {t.id: t for t in pending}

        # 多轮市场
        max_rounds = 20
        for round_idx in range(max_rounds):
            if not unassigned:
                break

            bids = []  # (bid, task_id, drone_id, best_position)

            for drone in drones:
                bundle = bundles[drone.id]
                if len(bundle) >= self.max_tasks_per_drone:
                    continue

                for task in unassigned:
                    if task.payload_weight > drone.max_payload:
                        continue

                    score, position = self._marginal_market_gain(drone, task, bundle)
                    if score > 0:
                        bids.append((score, task.id, drone.id, position))

            if not bids:
                break

            # 最高出价者获胜
            bids.sort(key=lambda x: x[0], reverse=True)
            best_bid, best_task_id, best_drone_id, best_pos = bids[0]

            task = market_tasks[best_task_id]
            bundles[best_drone_id].insert(best_pos, task)
            unassigned = [t for t in unassigned if t.id != best_task_id]

        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return {d.id: [t.id for t in bundles[d.id]] for d in drones}

    def _marginal_market_gain(self, drone, task, bundle):
        """计算市场边际收益（简化版，无共识同步）"""
        best_score = -float("inf")
        best_position = -1

        for insert_pos in range(len(bundle) + 1):
            temp_bundle = bundle[:insert_pos] + [task] + bundle[insert_pos:]

            old_dist = self._estimate_path_distance(drone, bundle, None)
            new_dist = self._estimate_path_distance(drone, temp_bundle, None)

            dist_inc = new_dist - old_dist
            priority_weight = PRIORITY_WEIGHTS.get(task.priority, 1.0)

            score = task.reward * priority_weight - dist_inc * drone.energy_per_meter * 0.01

            if score > best_score:
                best_score = score
                best_position = insert_pos

        return best_score, best_position

    def run_iteration(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        return self.allocate(drones, tasks, comm_graph, current_time)
