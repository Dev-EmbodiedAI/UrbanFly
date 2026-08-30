"""
算法对比评估框架
===============
在同一场景下运行多种分配算法并收集对比指标。

指标：
- 总完成任务数
- 按优先级加权的完成率
- 时间窗满足率
- 总飞行距离
- 总能耗
- 通信受阻下的性能退化率
- 算法运行时间
- 任务响应延迟
"""

import json
import time as _time
import numpy as np
from typing import List, Dict, Optional, Type
from collections import defaultdict

from .base import BaseAllocator
from .cbba import CBBAAllocator
from .hungarian import HungarianAllocator
from .greedy import GreedyAllocator
from .auction import AuctionAllocator
from .genetic import GeneticAllocator
from .market import MarketAllocator
from ..models import Task, DroneStateData, TaskStatus


class AlgorithmBenchmark:
    """
    算法对比评估框架。

    用法:
        bench = AlgorithmBenchmark(drones, tasks)
        results = bench.run_all()
        bench.print_report(results)
    """

    def __init__(self, drones: List[DroneStateData], tasks: List[Task]):
        self.drones = drones
        self.tasks = tasks
        self.algorithms: Dict[str, BaseAllocator] = {}

        # 注册所有算法
        self._register_algorithms()

    def _register_algorithms(self):
        """注册所有可用的分配算法"""
        self.algorithms = {
            "cbba": CBBAAllocator(),
            "hungarian": HungarianAllocator(),
            "greedy": GreedyAllocator(),
            "auction": AuctionAllocator(),
            "genetic": GeneticAllocator(population_size=50, generations=30),
            "market": MarketAllocator(),
        }

    def run_single(self, algorithm_name: str, comm_graph: np.ndarray = None,
                   runs: int = 3) -> Dict:
        """运行单一算法多次并取平均值"""
        if algorithm_name not in self.algorithms:
            return {"error": f"Unknown algorithm: {algorithm_name}"}

        allocator = self.algorithms[algorithm_name]
        results = []

        for run in range(runs):
            # 重置任务状态
            for task in self.tasks:
                task.status = TaskStatus.PENDING
                task.assigned_to = None

            # 重置无人机
            for drone in self.drones:
                drone.assigned_tasks = []
                drone.current_task_id = None

            # 运行分配
            t_start = _time.perf_counter()
            assignments = allocator.allocate(
                self.drones, self.tasks, comm_graph, current_time=0.0
            )
            runtime_ms = (_time.perf_counter() - t_start) * 1000.0

            # 评估指标
            metrics = self._evaluate(assignments, runtime_ms)
            results.append(metrics)

        # 聚合
        avg_metrics = {}
        for key in results[0]:
            values = [r[key] for r in results]
            avg_metrics[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        return {
            "algorithm": algorithm_name,
            "runs": runs,
            "metrics": avg_metrics,
        }

    def run_all(self, comm_scenarios: Dict[str, np.ndarray] = None,
                runs: int = 3) -> List[Dict]:
        """
        在所有通信场景下运行所有算法。

        Args:
            comm_scenarios: {"scenario_name": adjacency_matrix}
            runs: 每个场景-算法组合的运行次数
        """
        if comm_scenarios is None:
            comm_scenarios = {
                "ideal": None,  # 全连通
            }

        all_results = []

        for scenario_name, comm_graph in comm_scenarios.items():
            for algo_name in self.algorithms:
                result = self.run_single(algo_name, comm_graph, runs)
                result["comm_scenario"] = scenario_name
                all_results.append(result)

        return all_results

    def _evaluate(self, assignments: Dict[str, List[str]],
                  runtime_ms: float) -> Dict:
        """计算评估指标"""
        # 收集分配统计
        all_assigned = set()
        for task_ids in assignments.values():
            all_assigned.update(task_ids)

        total_assigned = len(all_assigned)
        total_pending = len([t for t in self.tasks
                            if t.status.value not in ("completed", "failed", "cancelled")])

        # 按优先级统计
        priority_assigned = defaultdict(int)
        priority_total = defaultdict(int)
        for task in self.tasks:
            priority_total[task.priority] += 1
            if task.id in all_assigned:
                priority_assigned[task.priority] += 1

        # 加权完成率
        weighted_sum = 0.0
        weight_sum = 0.0
        priority_weights = {0: 10, 1: 5, 2: 2.5, 3: 1, 4: 0.3}
        for p, total in priority_total.items():
            assigned = priority_assigned.get(p, 0)
            w = priority_weights.get(p, 1.0)
            weighted_sum += (assigned / total if total > 0 else 0) * w
            weight_sum += w
        weighted_rate = weighted_sum / weight_sum if weight_sum > 0 else 0

        # 时间窗满足率（粗略估算）
        on_time = 0
        for task in self.tasks:
            if task.id in all_assigned:
                # 假设到达时间 = 路径距离 / 速度
                drone = self._get_assignee(task.id, assignments)
                if drone:
                    dist = np.linalg.norm(drone.position - task.pickup_pos) + \
                           np.linalg.norm(task.pickup_pos - task.delivery_pos)
                    arrival = dist / drone.max_speed
                    if arrival <= task.time_window[1]:
                        on_time += 1

        on_time_rate = on_time / total_assigned if total_assigned > 0 else 0

        # 总飞行距离估算
        total_distance = 0.0
        for drone in self.drones:
            drone_tasks = assignments.get(drone.id, [])
            current = drone.position
            for tid in drone_tasks:
                task = self._find_task(tid)
                if task:
                    total_distance += np.linalg.norm(current - task.pickup_pos)
                    total_distance += np.linalg.norm(task.pickup_pos - task.delivery_pos)
                    current = task.delivery_pos

        # 负载均衡
        bundle_sizes = [len(v) for v in assignments.values()]
        load_balance_std = float(np.std(bundle_sizes)) if bundle_sizes else 0

        return {
            "total_assigned": total_assigned,
            "total_pending": total_pending,
            "assignment_rate": total_assigned / total_pending if total_pending > 0 else 0,
            "weighted_priority_rate": weighted_rate,
            "on_time_rate": on_time_rate,
            "total_distance": total_distance,
            "avg_distance_per_task": total_distance / total_assigned if total_assigned > 0 else 0,
            "load_balance_std": load_balance_std,
            "runtime_ms": runtime_ms,
            "priority_coverage": dict(priority_assigned),
        }

    def _find_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def _get_assignee(self, task_id: str,
                      assignments: Dict[str, List[str]]) -> Optional[DroneStateData]:
        for drone in self.drones:
            if task_id in assignments.get(drone.id, []):
                return drone
        return None

    @staticmethod
    def print_report(results: List[Dict]):
        """打印格式化的对比报告"""
        print("\n" + "=" * 80)
        print("算法对比报告")
        print("=" * 80)

        # 表头
        header = f"{'算法':<15} {'分配率':>8} {'加权率':>8} {'准时率':>8} {'耗时(ms)':>10} {'负载STD':>8}"
        print(header)
        print("-" * 80)

        for result in results:
            m = result["metrics"]
            algo = result["algorithm"]
            comm = result.get("comm_scenario", "ideal")

            ar = m.get("assignment_rate", {})
            wr = m.get("weighted_priority_rate", {})
            otr = m.get("on_time_rate", {})
            rt = m.get("runtime_ms", {})
            lbs = m.get("load_balance_std", {})

            print(f"{algo:<15} "
                  f"{ar.get('mean', 0):>7.1%} "
                  f"{wr.get('mean', 0):>7.1%} "
                  f"{otr.get('mean', 0):>7.1%} "
                  f"{rt.get('mean', 0):>9.1f} "
                  f"{lbs.get('mean', 0):>7.1f} "
                  f"[{comm}]")

        print("-" * 80)

    @staticmethod
    def export_to_json(results: List[Dict], filepath: str):
        """导出结果到JSON文件"""
        # 清理numpy类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
