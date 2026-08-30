"""
通信模型
=======
模拟城市环境中无人机间通信的受限情况。

受阻类型：
1. 距离限制：超出最大通信范围无法直接通信
2. 建筑遮挡：信号被高层建筑阻挡（射线检测）
3. 带宽限制：信道容量有限，拥塞时丢包
4. 间歇性中断：模拟城市峡谷中的信号波动
"""

import numpy as np
from typing import List, Dict, Optional, Set
from ..config import COMMUNICATION, COMM_SCENARIOS


class CommunicationModel:
    """
    无人机间通信模型。

    负责：
    - 计算通信拓扑（邻接矩阵）
    - 判断两架无人机是否能通信
    - 模拟消息广播（含丢包/延迟）
    - 跟踪通信断开/恢复事件
    """

    def __init__(self, buildings=None, comm_scenario="building_blocked", random_seed=None):
        """
        Args:
            buildings: BuildingInfo 对象列表
            comm_scenario: 通信场景名 ("ideal"|"distance_limited"|"building_blocked"|"intermittent"|"harsh")
            random_seed: 随机种子
        """
        scenario_cfg = COMM_SCENARIOS.get(comm_scenario, COMM_SCENARIOS["building_blocked"])

        self.buildings = buildings or []
        self.max_range = scenario_cfg["max_range"]
        self.bandwidth = scenario_cfg["bandwidth"]
        self.packet_loss_base = scenario_cfg["packet_loss_base"]
        self.use_building_block = scenario_cfg["use_building_block"]
        self.ray_check_samples = COMMUNICATION["ray_check_samples"]
        self.update_interval = COMMUNICATION["comm_update_interval"]

        self.rng = np.random.RandomState(random_seed or 42)
        self._adj_matrix: Optional[np.ndarray] = None
        self._last_update_time: float = -1.0
        self._disconnection_events: List[tuple] = []  # (drone_id1, drone_id2, time)

    def reset(self):
        """重置通信状态"""
        self._adj_matrix = None
        self._last_update_time = -1.0
        self._disconnection_events = []

    # ------------------------------------------------------------------
    # 核心通信判断
    # ------------------------------------------------------------------

    def can_communicate(self, drone_a, drone_b) -> bool:
        """
        判断两架无人机是否能直接通信。

        检查顺序：
        1. 距离是否在通信范围内
        2. 信号是否被建筑物遮挡
        3. 概率性丢包（信号衰减）

        Args:
            drone_a, drone_b: DroneStateData 对象
        Returns:
            bool: 是否能通信
        """
        pos_a = drone_a.position
        pos_b = drone_b.position

        # 1. 距离检查
        dist = np.linalg.norm(pos_a - pos_b)
        if dist > self.max_range:
            return False

        # 2. 建筑物遮挡检查
        if self.use_building_block and self._is_blocked_by_building(pos_a, pos_b):
            return False

        # 3. 信号强度衰减 → 概率性丢包
        signal_strength = max(0.0, 1.0 - (dist / self.max_range))
        effective_strength = signal_strength * (1.0 - self.packet_loss_base)

        if effective_strength < 1.0:
            if self.rng.random() > effective_strength:
                return False

        return True

    def _is_blocked_by_building(self, pos_a: np.ndarray, pos_b: np.ndarray) -> bool:
        """
        射线检测：两点之间的线段是否穿过任何建筑物包围盒。

        对线段进行采样，检查每个采样点是否在建筑包围盒内。
        """
        if not self.buildings:
            return False

        for t in np.linspace(0, 1, self.ray_check_samples):
            sample = pos_a + t * (pos_b - pos_a)
            for building in self.buildings:
                if building.contains(sample):
                    return True
        return False

    # ------------------------------------------------------------------
    # 通信拓扑
    # ------------------------------------------------------------------

    def get_communication_graph(self, drones: List) -> np.ndarray:
        """
        计算当前时刻的通信拓扑（邻接矩阵）。

        Returns:
            np.ndarray: N×N 布尔邻接矩阵, adj[i][j] = True 表示 i 能发消息给 j
                       注意：信号衰减可能导致非对称（概率性丢包）
        """
        n = len(drones)
        adj = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i][j] = True
                elif i < j:
                    can = self.can_communicate(drones[i], drones[j])
                    adj[i][j] = can
                    adj[j][i] = can  # 对称假设（除非概率丢包导致不对称）

        # 对于概率丢包，需要独立计算每个方向
        if self.packet_loss_base > 0:
            for i in range(n):
                for j in range(n):
                    if i != j and adj[i][j]:
                        # 独立判断反方向
                        if not self.can_communicate(drones[j], drones[i]):
                            adj[j][i] = False

        self._adj_matrix = adj
        return adj

    def get_neighbors(self, drone_id: str, drones: List) -> List[str]:
        """获取指定无人机当前可通信的邻居列表"""
        drone_map = {d.id: i for i, d in enumerate(drones)}
        if drone_id not in drone_map:
            return []

        idx = drone_map[drone_id]
        if self._adj_matrix is None:
            self.get_communication_graph(drones)

        neighbors = []
        for j, d in enumerate(drones):
            if idx != j and self._adj_matrix[idx][j]:
                neighbors.append(d.id)

        return neighbors

    def get_connected_components(self, drones: List) -> List[Set[str]]:
        """计算通信图的连通分量"""
        if self._adj_matrix is None:
            self.get_communication_graph(drones)

        n = len(drones)
        visited = [False] * n
        components = []

        for i in range(n):
            if not visited[i]:
                component = set()
                self._dfs_component(i, visited, component, drones)
                components.append(component)

        return components

    def _dfs_component(self, node: int, visited: List[bool], component: Set[str], drones: List):
        """DFS 遍历连通分量"""
        visited[node] = True
        component.add(drones[node].id)
        for j in range(len(drones)):
            if not visited[j] and self._adj_matrix[node][j]:
                self._dfs_component(j, visited, component, drones)

    # ------------------------------------------------------------------
    # 消息模拟
    # ------------------------------------------------------------------

    def broadcast(self, sender, message, drones: List) -> List:
        """
        模拟广播：消息只到达能通信的无人机。

        Args:
            sender: 发送者 DroneStateData
            message: CBBA消息或其它消息
            drones: 所有无人机列表

        Returns:
            成功接收消息的无人机列表
        """
        recipients = []
        msg_count = 0

        for drone in drones:
            if drone.id == sender.id:
                continue

            if self.can_communicate(sender, drone):
                # 带宽限制：超出信道容量的消息随机丢弃
                msg_count += 1
                accept_prob = min(1.0, self.bandwidth / len(drones))
                if self.rng.random() < accept_prob:
                    recipients.append(drone)

        return recipients

    def unicast(self, sender, target, message) -> bool:
        """单播：发送消息给特定无人机"""
        return self.can_communicate(sender, target)

    # ------------------------------------------------------------------
    # 通信事件追踪
    # ------------------------------------------------------------------

    def update_disconnection_tracking(self, drones: List, current_time: float):
        """追踪通信断开/恢复事件"""
        if self._adj_matrix is None:
            self.get_communication_graph(drones)
            return

        prev_adj = self._adj_matrix.copy()
        new_adj = self.get_communication_graph(drones)

        n = len(drones)
        for i in range(n):
            for j in range(i + 1, n):
                if prev_adj[i][j] and not new_adj[i][j]:
                    self._disconnection_events.append(
                        (drones[i].id, drones[j].id, current_time)
                    )

        self._adj_matrix = new_adj

    def get_disconnection_count(self) -> int:
        """获取通信断开事件总数"""
        return len(self._disconnection_events)

    def get_topology_stats(self, drones: List) -> Dict:
        """获取通信拓扑统计信息"""
        if self._adj_matrix is None:
            self.get_communication_graph(drones)

        n = len(drones)
        total_edges = np.sum(self._adj_matrix) // 2 - n // 2  # 减去自环
        max_edges = n * (n - 1) // 2
        components = self.get_connected_components(drones)

        return {
            "num_drones": n,
            "num_edges": int(total_edges),
            "max_edges": max_edges,
            "connectivity_ratio": round(total_edges / max_edges, 4) if max_edges > 0 else 1.0,
            "num_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "isolated_drones": sum(1 for c in components if len(c) == 1),
        }
