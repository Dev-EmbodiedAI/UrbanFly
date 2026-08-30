"""
鲸鱼优化算法 (WOA) — 多无人机任务分配
=======================================
模拟座头鲸气泡网捕食行为。
"""
import numpy as np
from typing import List, Dict, Optional
import time as _time
from .base import BaseAllocator
from ..models import Task, DroneStateData


class WOAAllocator(BaseAllocator):
    """基于鲸鱼优化的任务分配器"""

    def __init__(self, n_whales=50, n_iterations=80):
        super().__init__(name="WOA")
        self.n_whales = n_whales
        self.n_iterations = n_iterations

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

        whales = np.random.randint(0, n_drones, (self.n_whales, n_tasks))
        scores = np.array([fitness(w) for w in whales])
        best_idx = np.argmax(scores)
        best = whales[best_idx].copy()
        best_score = scores[best_idx]

        for it in range(self.n_iterations):
            a_lin = 2.0 - 2.0 * it / self.n_iterations
            for i in range(self.n_whales):
                r = np.random.random()
                A = 2*a_lin*np.random.random(n_tasks) - a_lin
                C = 2*np.random.random(n_tasks)
                if r < 0.5:
                    if np.linalg.norm(A) < 1:
                        D = np.abs(C*best - whales[i])
                        whales[i] = np.clip(np.round(best - A*D).astype(int), 0, n_drones-1)
                    else:
                        rand_idx = np.random.randint(0, self.n_whales)
                        D = np.abs(C*whales[rand_idx] - whales[i])
                        whales[i] = np.clip(np.round(whales[rand_idx] - A*D).astype(int), 0, n_drones-1)
                else:
                    D_best = np.abs(best - whales[i])
                    l = np.random.uniform(-1, 1, n_tasks)
                    whales[i] = np.clip(np.round(D_best*np.exp(l)*np.cos(2*np.pi*l) + best).astype(int), 0, n_drones-1)

            scores = np.array([fitness(w) for w in whales])
            if np.max(scores) > best_score:
                best_idx = np.argmax(scores)
                best = whales[best_idx].copy()
                best_score = scores[best_idx]

        assignments = {d.id: [] for d in drones}
        for ti, di in enumerate(best):
            if feasible[di, ti]:
                assignments[drones[di].id].append(tasks[ti].id)
        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(self, drones, tasks, comm_graph=None, current_time=0.0):
        return self.allocate(drones, tasks, comm_graph, current_time)
