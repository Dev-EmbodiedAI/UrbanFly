"""
粒子群优化 (PSO) — 多无人机任务分配
===================================
每个粒子编码为 n_tasks 维整数向量，每个元素 ∈ [0, n_drones-1]，
表示对应任务分配给哪架无人机。
"""
import numpy as np
from typing import List, Dict, Optional
import time as _time
from .base import BaseAllocator
from ..models import Task, DroneStateData


class PSOAllocator(BaseAllocator):
    """基于粒子群优化的任务分配器"""

    def __init__(self, n_particles=50, n_iterations=100, w=0.7, c1=1.5, c2=1.5):
        super().__init__(name="PSO")
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w      # 惯性权重
        self.c1 = c1    # 个体认知系数
        self.c2 = c2    # 社会认知系数

    def allocate(self, drones, tasks, comm_graph=None, current_time=0.0):
        t_start = _time.perf_counter()
        n_drones = len(drones)
        n_tasks = len(tasks)

        if n_drones == 0 or n_tasks == 0:
            return {d.id: [] for d in drones}

        # 预计算距离
        drone_pos = np.array([d.position for d in drones])
        task_pickup = np.array([t.pickup_pos for t in tasks])
        task_delivery = np.array([t.delivery_pos for t in tasks])
        drone_payload = np.array([d.max_payload for d in drones])
        task_payload = np.array([t.payload_weight for t in tasks])

        # 可行分配掩码
        feasible = task_payload[np.newaxis, :] <= drone_payload[:, np.newaxis]

        # 权重
        pw = {0: 10, 1: 5, 2: 2.5, 3: 1, 4: 0.3}
        task_weights = np.array([pw.get(t.priority, 1.0) for t in tasks])
        task_rewards = np.array([t.reward for t in tasks])

        def fitness(assignment):
            """计算分配方案的适应度 (越高越好)"""
            score = 0.0
            assigned = set()
            for t_idx, d_idx in enumerate(assignment):
                if 0 <= d_idx < n_drones and feasible[d_idx, t_idx]:
                    assigned.add(t_idx)
                    # 距离成本
                    dist = (np.linalg.norm(drone_pos[d_idx] - task_pickup[t_idx]) +
                            np.linalg.norm(task_pickup[t_idx] - task_delivery[t_idx]))
                    # 收益 = 权重×奖励 - 距离成本
                    score += task_weights[t_idx] * task_rewards[t_idx] - dist * 0.05
            # 未分配惩罚
            for t_idx in range(n_tasks):
                if t_idx not in assigned:
                    score -= task_weights[t_idx] * task_rewards[t_idx] * 0.01
            return score

        # 初始化粒子群
        particles = np.random.randint(0, n_drones, (self.n_particles, n_tasks))
        velocities = np.random.randn(self.n_particles, n_tasks) * 2
        p_best = particles.copy()
        p_best_score = np.array([fitness(p) for p in particles])
        g_best_idx = np.argmax(p_best_score)
        g_best = p_best[g_best_idx].copy()
        g_best_score = p_best_score[g_best_idx]

        for _ in range(self.n_iterations):
            r1, r2 = np.random.rand(self.n_particles, n_tasks), np.random.rand(self.n_particles, n_tasks)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (p_best - particles) +
                          self.c2 * r2 * (g_best - particles))
            particles = np.clip(np.round(particles + velocities).astype(int), 0, n_drones - 1)

            scores = np.array([fitness(p) for p in particles])
            improved = scores > p_best_score
            p_best[improved] = particles[improved]
            p_best_score[improved] = scores[improved]

            if np.max(scores) > g_best_score:
                g_best_idx = np.argmax(scores)
                g_best = particles[g_best_idx].copy()
                g_best_score = scores[g_best_idx]

        # 构建分配结果
        assignments = {d.id: [] for d in drones}
        for t_idx, d_idx in enumerate(g_best):
            if feasible[d_idx, t_idx]:
                assignments[drones[d_idx].id].append(tasks[t_idx].id)

        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(self, drones, tasks, comm_graph=None, current_time=0.0):
        return self.allocate(drones, tasks, comm_graph, current_time)
