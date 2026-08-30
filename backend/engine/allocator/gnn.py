"""
GNN-inspired 任务分配器 (概念验证)
==================================
简化版消息传递网络: 将无人机-任务建模为二部图,
通过迭代消息传递学习分配偏好。
"""
import numpy as np
from typing import List, Dict, Optional
import time as _time
from .base import BaseAllocator
from ..models import Task, DroneStateData


class GNNAllocator(BaseAllocator):
    """GNN启发的任务分配器 (概念验证版)

    思路: 构建无人机-任务二部图, 通过多轮消息传递
          更新节点嵌入, 最终通过相似度得分完成分配。
    """

    def __init__(self, n_layers=5, embedding_dim=32):
        super().__init__(name="GNN")
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

    def allocate(self, drones, tasks, comm_graph=None, current_time=0.0):
        t_start = _time.perf_counter()
        n_drones = len(drones)
        n_tasks = len(tasks)
        if n_drones == 0 or n_tasks == 0:
            return {d.id: [] for d in drones}

        # 无人机特征: [pos_x, pos_y, pos_z, max_payload, speed, battery]
        drone_feat = np.array([
            [d.position[0]/400, d.position[1]/400, d.position[2]/100,
             d.max_payload/15, d.cruise_speed/20, d.battery_remaining/500]
            for d in drones
        ])

        # 任务特征: [pickup_x, pickup_y, pickup_z, delivery_x, delivery_y, delivery_z,
        #            payload, priority_weight, reward/500]
        pw = {0: 10, 1: 5, 2: 2.5, 3: 1, 4: 0.3}
        task_feat = np.array([
            [t.pickup_pos[0]/400, t.pickup_pos[1]/400, t.pickup_pos[2]/100,
             t.delivery_pos[0]/400, t.delivery_pos[1]/400, t.delivery_pos[2]/100,
             t.payload_weight/10, pw.get(t.priority, 1)/10, t.reward/500]
            for t in tasks
        ])

        # 简化消息传递: 多次迭代更新嵌入
        d_embed = drone_feat.copy()
        t_embed = task_feat.copy()

        for _ in range(self.n_layers):
            # 任务←无人机消息: 每个任务聚合最近K个无人机的特征
            dist_dp = np.linalg.norm(
                np.array([d.position for d in drones])[:, np.newaxis, :] -
                np.array([t.pickup_pos for t in tasks])[np.newaxis, :, :],
                axis=2
            )  # (n_drones, n_tasks)

            # 任务更新: 加权聚合无人机嵌入 (距离越近权重越大)
            weights = np.exp(-dist_dp / 200.0)  # (n_drones, n_tasks)
            weights /= weights.sum(axis=0, keepdims=True) + 1e-8
            t_embed_new = weights.T @ d_embed  # (n_tasks, embed_dim)
            t_embed = 0.5 * t_embed + 0.5 * t_embed_new

            # 无人机更新: 聚合可执行任务的嵌入
            payload_feasible = (np.array([t.payload_weight for t in tasks])[np.newaxis, :] <=
                               np.array([d.max_payload for d in drones])[:, np.newaxis])
            d_weights = np.exp(-dist_dp / 200.0) * payload_feasible
            d_weights /= d_weights.sum(axis=1, keepdims=True) + 1e-8
            d_embed_new = d_weights @ t_embed  # (n_drones, embed_dim)
            d_embed = 0.5 * d_embed + 0.5 * d_embed_new

        # 计算分配得分矩阵: (n_drones, n_tasks)
        score_matrix = d_embed @ t_embed.T

        # 贪婪分配 (按得分排序, 每架无人机最多8个任务)
        assignments = {d.id: [] for d in drones}
        max_tasks_per_drone = 8
        drone_loads = np.zeros(n_drones)
        drone_pl = np.array([d.max_payload for d in drones])
        task_pl = np.array([t.payload_weight for t in tasks])
        task_assigned = np.zeros(n_tasks, dtype=bool)

        # 按得分降序分配
        flat_indices = np.argsort(score_matrix.flatten())[::-1]
        for flat_idx in flat_indices:
            d_idx, t_idx = np.unravel_index(flat_idx, score_matrix.shape)
            if (not task_assigned[t_idx] and
                drone_loads[d_idx] + 1 <= max_tasks_per_drone and
                task_pl[t_idx] <= drone_pl[d_idx]):
                assignments[drones[d_idx].id].append(tasks[t_idx].id)
                task_assigned[t_idx] = True
                drone_loads[d_idx] += 1

        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(self, drones, tasks, comm_graph=None, current_time=0.0):
        return self.allocate(drones, tasks, comm_graph, current_time)
