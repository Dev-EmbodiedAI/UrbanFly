"""
拍卖算法 — 去中心化任务分配 baseline
===================================
多轮密封拍卖，每轮无人机对未分配任务出价，最高价者获胜。

特点：比CBBA简单（无bundle构建），但缺乏任务排序优化。
"""

import numpy as np
from typing import List, Dict, Optional
import time as _time

from .base import BaseAllocator
from ..models import Task, DroneStateData
from ...config import PRIORITY_WEIGHTS


class AuctionAllocator(BaseAllocator):
    """多轮拍卖分配器"""

    def __init__(self, max_rounds: int = 20, epsilon: float = 0.1):
        super().__init__(name="auction")
        self.max_rounds = max_rounds
        self.epsilon = epsilon  # 最小提价幅度

    def allocate(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        """执行多轮拍卖"""
        t_start = _time.perf_counter()

        pending = [t for t in tasks if t.status.value not in ("completed", "failed", "cancelled")]
        if not pending:
            return {d.id: [] for d in drones}

        # 任务价格（被当前最高出价者的价格取代）
        task_prices: Dict[str, float] = {t.id: t.reward * 0.1 for t in pending}
        task_winners: Dict[str, str] = {}  # task_id → drone_id
        assignments: Dict[str, List[str]] = {d.id: [] for d in drones}
        max_per_drone = 5

        for round_idx in range(self.max_rounds):
            bids_changed = False

            for drone in drones:
                # 当前已分配的任务数
                if len(assignments[drone.id]) >= max_per_drone:
                    continue

                best_task_id = None
                best_profit = -float("inf")

                for task in pending:
                    if task.id in task_winners:
                        continue

                    # 计算此任务对无人机的价值
                    dist = np.linalg.norm(drone.position - task.pickup_pos) + \
                           np.linalg.norm(task.pickup_pos - task.delivery_pos)
                    priority_weight = PRIORITY_WEIGHTS.get(task.priority, 1.0)
                    value = task.reward * priority_weight - dist * drone.energy_per_meter

                    # 利润 = 价值 - 当前价格
                    price = task_prices[task.id]
                    profit = value - price

                    if profit > best_profit:
                        best_profit = profit
                        best_task_id = task.id

                if best_task_id is not None and best_profit > 0:
                    # 出价：当前价格 + 利润 - epsilon（保证正利润）
                    bid = task_prices[best_task_id] + best_profit - self.epsilon

                    # 更新最高出价
                    if bid > task_prices[best_task_id]:
                        task_prices[best_task_id] = bid
                        task_winners[best_task_id] = drone.id
                        bids_changed = True

            if not bids_changed:
                break

        # 最终分配
        for task_id, drone_id in task_winners.items():
            if len(assignments[drone_id]) < max_per_drone:
                assignments[drone_id].append(task_id)

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
