"""
模拟退火 (SA) — 多无人机任务分配
================================
从随机分配开始，通过温度递减的Metropolis准则逐步优化。
"""
import numpy as np
from typing import List, Dict, Optional
import time as _time
from .base import BaseAllocator
from ..models import Task, DroneStateData


class SAAllocator(BaseAllocator):
    """基于模拟退火的任务分配器"""

    def __init__(self, T_init=1000.0, T_min=0.01, alpha=0.95, steps_per_T=50):
        super().__init__(name="SA")
        self.T_init = T_init
        self.T_min = T_min
        self.alpha = alpha
        self.steps_per_T = steps_per_T

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
            s -= np.std(loads)*25
            return s

        current = np.random.randint(0, n_drones, n_tasks)
        current_score = fitness(current)
        best = current.copy()
        best_score = current_score
        T = self.T_init

        while T > self.T_min:
            for _ in range(self.steps_per_T):
                neighbor = current.copy()
                idx = np.random.randint(0, n_tasks)
                neighbor[idx] = np.random.randint(0, n_drones)
                neighbor_score = fitness(neighbor)
                delta = neighbor_score - current_score

                if delta > 0 or np.random.random() < np.exp(delta / T):
                    current = neighbor
                    current_score = neighbor_score
                    if current_score > best_score:
                        best = current.copy()
                        best_score = current_score

            T *= self.alpha

        assignments = {d.id: [] for d in drones}
        for ti, di in enumerate(best):
            if feasible[di, ti]:
                assignments[drones[di].id].append(tasks[ti].id)
        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(self, drones, tasks, comm_graph=None, current_time=0.0):
        return self.allocate(drones, tasks, comm_graph, current_time)
