"""
灰狼优化算法 (GWO) — 多无人机任务分配
=====================================
模拟灰狼群体的社会等级制度和狩猎机制。

每只"狼"编码为 n_tasks 维整数向量。
α/β/δ 三只领头狼引导搜索。
"""
import numpy as np
from typing import List, Dict, Optional
import time as _time
from .base import BaseAllocator
from ..models import Task, DroneStateData


class GWOAllocator(BaseAllocator):
    """基于灰狼优化的任务分配器"""

    def __init__(self, n_wolves=50, n_iterations=80):
        super().__init__(name="GWO")
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations

    def allocate(self, drones, tasks, comm_graph=None, current_time=0.0):
        t_start = _time.perf_counter()
        n_drones = len(drones)
        n_tasks = len(tasks)

        if n_drones == 0 or n_tasks == 0:
            return {d.id: [] for d in drones}

        drone_pos = np.array([d.position for d in drones])
        task_pickup = np.array([t.pickup_pos for t in tasks])
        task_delivery = np.array([t.delivery_pos for t in tasks])
        drone_payload = np.array([d.max_payload for d in drones])
        task_payload = np.array([t.payload_weight for t in tasks])
        feasible = task_payload[np.newaxis, :] <= drone_payload[:, np.newaxis]

        pw = {0: 10, 1: 5, 2: 2.5, 3: 1, 4: 0.3}
        task_weights = np.array([pw.get(t.priority, 1.0) for t in tasks])
        task_rewards = np.array([t.reward for t in tasks])

        def fitness(assignment):
            score = 0.0
            for t_idx, d_idx in enumerate(assignment):
                if 0 <= d_idx < n_drones and feasible[d_idx, t_idx]:
                    dist = (np.linalg.norm(drone_pos[d_idx] - task_pickup[t_idx]) +
                            np.linalg.norm(task_pickup[t_idx] - task_delivery[t_idx]))
                    score += task_weights[t_idx] * task_rewards[t_idx] - dist * 0.05
            # 负载均衡惩罚
            loads = np.bincount(np.clip(assignment, 0, n_drones-1), minlength=n_drones)
            score -= np.std(loads) * 50
            return score

        # 初始化狼群
        wolves = np.random.randint(0, n_drones, (self.n_wolves, n_tasks))
        scores = np.array([fitness(w) for w in wolves])
        top_idx = np.argsort(scores)[-3:]
        alpha = wolves[top_idx[2]].copy()
        beta = wolves[top_idx[1]].copy()
        delta = wolves[top_idx[0]].copy()

        for it in range(self.n_iterations):
            a = 2.0 - 2.0 * it / self.n_iterations  # 线性衰减

            for i in range(self.n_wolves):
                r1, r2 = np.random.rand(2, n_tasks)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha - wolves[i])
                X1 = alpha - A1 * D_alpha

                r1, r2 = np.random.rand(2, n_tasks)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta - wolves[i])
                X2 = beta - A2 * D_beta

                r1, r2 = np.random.rand(2, n_tasks)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta - wolves[i])
                X3 = delta - A3 * D_delta

                wolves[i] = np.clip(np.round((X1 + X2 + X3) / 3).astype(int), 0, n_drones - 1)

            scores = np.array([fitness(w) for w in wolves])
            top_idx = np.argsort(scores)[-3:]
            if scores[top_idx[2]] > fitness(alpha):
                alpha = wolves[top_idx[2]].copy()
            if scores[top_idx[1]] > fitness(beta):
                beta = wolves[top_idx[1]].copy()
            if scores[top_idx[0]] > fitness(delta):
                delta = wolves[top_idx[0]].copy()

        assignments = {d.id: [] for d in drones}
        for t_idx, d_idx in enumerate(alpha):
            if feasible[d_idx, t_idx]:
                assignments[drones[d_idx].id].append(tasks[t_idx].id)

        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(self, drones, tasks, comm_graph=None, current_time=0.0):
        return self.allocate(drones, tasks, comm_graph, current_time)
