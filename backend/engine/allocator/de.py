"""
差分进化 (DE) — 多无人机任务分配
================================
通过变异、交叉、选择操作优化任务分配方案。
"""
import numpy as np
from typing import List, Dict, Optional
import time as _time
from .base import BaseAllocator
from ..models import Task, DroneStateData


class DEAllocator(BaseAllocator):
    """基于差分进化的任务分配器"""

    def __init__(self, pop_size=50, n_iterations=80, F=0.8, CR=0.7):
        super().__init__(name="DE")
        self.pop_size = pop_size
        self.n_iterations = n_iterations
        self.F = F      # 变异因子
        self.CR = CR    # 交叉概率

    def allocate(self, drones, tasks, comm_graph=None, current_time=0.0):
        t_start = _time.perf_counter()
        n_drones = len(drones)
        n_tasks = len(tasks)
        if n_drones == 0 or n_tasks == 0:
            return {d.id: [] for d in drones}

        drone_pos = np.array([d.position for d in drones])
        tp = np.array([t.pickup_pos for t in tasks])
        td = np.array([t.delivery_pos for t in tasks])
        drone_pl = np.array([d.max_payload for d in drones])
        task_pl = np.array([t.payload_weight for t in tasks])
        feasible = task_pl[np.newaxis, :] <= drone_pl[:, np.newaxis]
        pw = {0: 10, 1: 5, 2: 2.5, 3: 1, 4: 0.3}
        tw = np.array([pw.get(t.priority, 1.0) for t in tasks])
        tr = np.array([t.reward for t in tasks])

        def fitness(a):
            s = 0.0
            for ti, di in enumerate(a):
                if 0 <= di < n_drones and feasible[di, ti]:
                    d = np.linalg.norm(drone_pos[di]-tp[ti]) + np.linalg.norm(tp[ti]-td[ti])
                    s += tw[ti]*tr[ti] - d*0.05
            loads = np.bincount(np.clip(a, 0, n_drones-1), minlength=n_drones)
            s -= np.std(loads)*30
            return s

        pop = np.random.randint(0, n_drones, (self.pop_size, n_tasks))
        scores = np.array([fitness(p) for p in pop])

        for _ in range(self.n_iterations):
            for i in range(self.pop_size):
                # 选择3个不同的个体
                candidates = [j for j in range(self.pop_size) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # 变异
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(np.round(mutant).astype(int), 0, n_drones - 1)

                # 交叉
                cross_mask = np.random.random(n_tasks) < self.CR
                if not cross_mask.any():
                    cross_mask[np.random.randint(0, n_tasks)] = True
                trial = np.where(cross_mask, mutant, pop[i])

                # 选择
                trial_score = fitness(trial)
                if trial_score > scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score

        best_idx = np.argmax(scores)
        best = pop[best_idx]
        assignments = {d.id: [] for d in drones}
        for ti, di in enumerate(best):
            if feasible[di, ti]:
                assignments[drones[di].id].append(tasks[ti].id)
        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(self, drones, tasks, comm_graph=None, current_time=0.0):
        return self.allocate(drones, tasks, comm_graph, current_time)
