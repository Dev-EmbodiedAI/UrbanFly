"""
遗传算法 — 集中式任务分配 baseline
==================================
使用遗传算法进行全局优化任务分配。

特点：适合作为离线最优参考，但计算量大不适合实时分配。
"""

import numpy as np
from typing import List, Dict, Optional
import time as _time
import random

from .base import BaseAllocator
from ..models import Task, DroneStateData
from ...config import PRIORITY_WEIGHTS


class GeneticAllocator(BaseAllocator):
    """基于遗传算法的全局优化分配器"""

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_ratio: float = 0.1,
                 random_seed: int = 42):
        super().__init__(name="genetic")

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.rng = random.Random(random_seed)

    def allocate(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        """使用遗传算法优化任务分配"""
        t_start = _time.perf_counter()

        pending = [t for t in tasks if t.status.value not in ("completed", "failed", "cancelled")]
        if not pending:
            return {d.id: [] for d in drones}

        M = len(drones)
        N = len(pending)

        # 染色体编码：长度为N的数组，每个基因值为0..M-1（分配到哪架无人机）
        def create_chromosome() -> np.ndarray:
            # 确保任务分布合理
            chrom = np.zeros(N, dtype=int)
            for j in range(N):
                # 倾向于分配到较近的无人机
                task = pending[j]
                distances = [np.linalg.norm(d.position - task.pickup_pos) for d in drones]
                # 软最大化：距离近的更高概率
                probs = 1.0 / (np.array(distances) + 1e-6)
                probs /= probs.sum()
                chrom[j] = np.random.choice(M, p=probs)
            return chrom

        def fitness(chromosome: np.ndarray) -> float:
            """适应度 = 总收益 - 总成本（越大越好）"""
            drone_loads = {d.id: [] for d in drones}

            for j, drone_idx in enumerate(chromosome):
                if drone_idx < M:
                    drone_loads[drones[drone_idx].id].append(pending[j])

            total_fitness = 0.0
            for drone in drones:
                bundle = drone_loads[drone.id]
                if len(bundle) > 8:  # 惩罚过度负载
                    total_fitness -= 1000 * (len(bundle) - 8)
                    continue

                # 路径成本
                current = drone.position
                for task in sorted(bundle, key=lambda t: np.linalg.norm(current - t.pickup_pos)):
                    dist = np.linalg.norm(current - task.pickup_pos) + \
                           np.linalg.norm(task.pickup_pos - task.delivery_pos)
                    priority_weight = PRIORITY_WEIGHTS.get(task.priority, 1.0)
                    total_fitness += task.reward * priority_weight - dist * 0.1
                    current = task.delivery_pos

            return total_fitness

        # 初始化种群
        population = [create_chromosome() for _ in range(self.population_size)]
        best_chromosome = None
        best_fitness_val = -float("inf")

        for gen in range(self.generations):
            # 计算适应度
            fitnesses = [fitness(chrom) for chrom in population]

            for i, f in enumerate(fitnesses):
                if f > best_fitness_val:
                    best_fitness_val = f
                    best_chromosome = population[i].copy()

            # 精英保留
            n_elites = max(1, int(self.population_size * self.elite_ratio))
            elite_indices = np.argsort(fitnesses)[-n_elites:]
            new_population = [population[i].copy() for i in elite_indices]

            # 选择 + 交叉 + 变异
            while len(new_population) < self.population_size:
                # 锦标赛选择
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)

                if self.rng.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = p1.copy()

                if self.rng.random() < self.mutation_rate:
                    child = self._mutate(child, M)

                new_population.append(child)

            population = new_population

        # 解码最优染色体
        assignments = {d.id: [] for d in drones}
        if best_chromosome is not None:
            for j, drone_idx in enumerate(best_chromosome):
                if drone_idx < M and len(assignments[drones[drone_idx].id]) < 8:
                    assignments[drones[drone_idx].id].append(pending[j].id)

        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def _tournament_select(self, population, fitnesses, tournament_size=3):
        idx = [self.rng.randint(0, len(population) - 1) for _ in range(tournament_size)]
        best_idx = max(idx, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def _crossover(self, p1, p2):
        """单点交叉"""
        if len(p1) < 2:
            return p1.copy()
        point = self.rng.randint(1, len(p1) - 1)
        child = np.concatenate([p1[:point], p2[point:]])
        return child

    def _mutate(self, chromosome, M):
        """随机变异"""
        idx = self.rng.randint(0, len(chromosome) - 1)
        chromosome[idx] = self.rng.randint(0, M - 1)
        return chromosome

    def run_iteration(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        return self.allocate(drones, tasks, comm_graph, current_time)
