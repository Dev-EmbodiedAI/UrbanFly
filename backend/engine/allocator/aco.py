"""
蚁群优化 (ACO) — 多无人机任务分配
=================================
每只蚂蚁依次为每个任务选择一架无人机，信息素矩阵为 (n_tasks, n_drones)。
"""
import numpy as np
from typing import List, Dict, Optional
import time as _time
from .base import BaseAllocator
from ..models import Task, DroneStateData


class ACOAllocator(BaseAllocator):
    """基于蚁群优化的任务分配器"""

    def __init__(self, n_ants=30, n_iterations=80, alpha=1.0, beta=2.0, rho=0.1):
        super().__init__(name="ACO")
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha_param = alpha  # 信息素重要性
        self.beta_param = beta    # 启发式信息重要性
        self.rho = rho            # 信息素挥发率

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

        # 启发式信息: 距离越近、收益越高 → 越优先选择
        dist_dp = np.linalg.norm(
            drone_pos[:, np.newaxis, :] - task_pickup[np.newaxis, :, :], axis=2)
        dist_pd = np.linalg.norm(
            task_pickup - task_delivery, axis=1)
        total_dist = dist_dp + dist_pd[np.newaxis, :]  # (n_drones, n_tasks)

        heuristic = (task_weights[np.newaxis, :] * task_rewards[np.newaxis, :]) / (total_dist + 1.0)
        heuristic[:, ~feasible.any(axis=0)] = 1e-6
        heuristic = np.clip(heuristic, 1e-6, None)

        # 信息素矩阵
        pheromone = np.ones((n_tasks, n_drones))

        def fitness(assignment):
            score = 0.0
            for t_idx, d_idx in enumerate(assignment):
                if feasible[d_idx, t_idx]:
                    score += task_weights[t_idx] * task_rewards[t_idx] - total_dist[d_idx, t_idx] * 0.05
            return score

        best_ant = None
        best_score = -float("inf")

        for _ in range(self.n_iterations):
            all_ants = np.zeros((self.n_ants, n_tasks), dtype=int)
            for ant_idx in range(self.n_ants):
                for t_idx in range(n_tasks):
                    probs = (pheromone[t_idx] ** self.alpha_param *
                             heuristic[:, t_idx] ** self.beta_param)
                    probs /= probs.sum()
                    all_ants[ant_idx, t_idx] = np.random.choice(n_drones, p=probs)

            scores = np.array([fitness(ant) for ant in all_ants])
            best_ant_idx = np.argmax(scores)

            if scores[best_ant_idx] > best_score:
                best_score = scores[best_ant_idx]
                best_ant = all_ants[best_ant_idx].copy()

            # 更新信息素
            pheromone *= (1 - self.rho)
            for ant_idx in range(self.n_ants):
                delta = scores[ant_idx] / (abs(scores[best_ant_idx]) + 1)
                for t_idx in range(n_tasks):
                    pheromone[t_idx, all_ants[ant_idx, t_idx]] += delta * 0.1

        assignments = {d.id: [] for d in drones}
        if best_ant is not None:
            for t_idx, d_idx in enumerate(best_ant):
                if feasible[d_idx, t_idx]:
                    assignments[drones[d_idx].id].append(tasks[t_idx].id)

        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(self, drones, tasks, comm_graph=None, current_time=0.0):
        return self.allocate(drones, tasks, comm_graph, current_time)
