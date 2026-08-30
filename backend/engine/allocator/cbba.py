"""
STC-RCBBA: 时空走廊约束鲁棒 CBBA
=================================

相较于仓库中的简化版 CBBA，本实现补上了三类关键能力：
1. 候选任务筛选 + 最佳插入位搜索，而不是简单尾插；
2. 时空走廊拥塞项、通信可信度项、电量安全项的联合边际收益建模；
3. 基于版本号与事件触发增量同步的鲁棒共识，用于退化通信场景。
"""

from __future__ import annotations

import math
import time as _time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseAllocator
from ..models import DroneStateData, Task
from ...config import BASE_STATIONS, CBBA, CHARGING_STATIONS, PATH_PLANNING, PRIORITY_WEIGHTS


class CBBAAllocator(BaseAllocator):
    """改进版 CBBA，实验标签使用 STC-RCBBA。"""

    def __init__(
        self,
        max_bundle_size: int = None,
        max_iterations: int = None,
        use_priority_term: bool = True,
        use_corridor_term: bool = True,
        use_robust_consensus: bool = True,
        use_residual_repair: Optional[bool] = None,
        display_name: str = "STC-RCBBA",
    ):
        super().__init__(name="cbba")

        self.display_name = display_name
        self.max_bundle_size = max_bundle_size or CBBA["max_bundle_size"]
        self.max_iterations = max_iterations or CBBA["max_iterations"]
        self.convergence_threshold = CBBA["convergence_threshold"]
        self.battery_safety_margin = CBBA["battery_safety_margin"]
        self.priority_weights = dict(PRIORITY_WEIGHTS)

        self.use_priority_term = use_priority_term
        self.use_corridor_term = use_corridor_term
        self.use_robust_consensus = use_robust_consensus
        self.use_residual_repair = (
            display_name != "原始CBBA" if use_residual_repair is None else bool(use_residual_repair)
        )

        self._top_k_exact_eval = int(CBBA["top_k_exact_eval"])
        self._noise = float(CBBA["marginal_gain_noise"])
        self._corridor_bonus_factor = float(CBBA["corridor_bonus_factor"])
        self._corridor_penalty_factor = float(CBBA["corridor_penalty_factor"])
        self._urgency_bonus_factor = float(CBBA["urgency_bonus_factor"])
        self._aging_bonus_factor = float(CBBA["aging_bonus_factor"])
        self._risk_penalty_factor = float(CBBA["risk_penalty_factor"])
        self._fragile_penalty_factor = float(CBBA["fragile_penalty_factor"])
        self._comm_penalty_factor = float(CBBA["comm_penalty_factor"])
        self._energy_penalty_factor = float(CBBA["energy_penalty_factor"])
        self._max_delta_sync = int(CBBA["event_sync_max_delta"])
        self._rebroadcast_interval = int(CBBA["consensus_rebroadcast_full_interval"])

        self._corridor_cell_size = float(PATH_PLANNING["corridor_cell_size"])
        self._corridor_time_slot = float(PATH_PLANNING["time_slot_sec"])
        self._corridor_capacity = int(PATH_PLANNING["corridor_capacity"])

        # 内部状态
        self._bundles: Dict[str, List[str]] = defaultdict(list)
        self._bundle_scores: Dict[str, List[float]] = defaultdict(list)
        self._local_winners: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._local_bids: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._local_versions: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._local_timestamps: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._peer_known_versions: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)
        self._dirty_tasks: Dict[str, set] = defaultdict(set)
        self._no_change_count: int = 0
        self._last_assignment: Dict[str, List[str]] = {}
        self._consensus_round: int = 0
        self._corridor_load: Dict[Tuple[str, int, int, int], int] = defaultdict(int)

        # 预计算缓存
        self._task_ids: List[str] = []
        self._task_idx: Dict[str, int] = {}
        self._drone_ids: List[str] = []
        self._drone_idx: Dict[str, int] = {}
        self._drone_positions: np.ndarray = np.empty((0, 3), dtype=float)
        self._task_pickup_positions: np.ndarray = np.empty((0, 3), dtype=float)
        self._task_delivery_positions: np.ndarray = np.empty((0, 3), dtype=float)
        self._drone_speeds: np.ndarray = np.empty(0, dtype=float)
        self._drone_max_payloads: np.ndarray = np.empty(0, dtype=float)
        self._drone_battery: np.ndarray = np.empty(0, dtype=float)
        self._drone_capacity: np.ndarray = np.empty(0, dtype=float)
        self._drone_energy_per_m: np.ndarray = np.empty(0, dtype=float)
        self._drone_energy_per_kg_m: np.ndarray = np.empty(0, dtype=float)
        self._drone_neighbor_counts: np.ndarray = np.empty(0, dtype=float)
        self._drone_comm_quality: np.ndarray = np.empty(0, dtype=float)
        self._drone_types: List[str] = []

        self._task_payloads: np.ndarray = np.empty(0, dtype=float)
        self._task_priorities: np.ndarray = np.empty(0, dtype=float)
        self._task_rewards: np.ndarray = np.empty(0, dtype=float)
        self._task_earliest: np.ndarray = np.empty(0, dtype=float)
        self._task_latest: np.ndarray = np.empty(0, dtype=float)
        self._task_penalties: np.ndarray = np.empty(0, dtype=float)
        self._task_created_at: np.ndarray = np.empty(0, dtype=float)
        self._task_pickup_service: np.ndarray = np.empty(0, dtype=float)
        self._task_delivery_service: np.ndarray = np.empty(0, dtype=float)
        self._task_risk: np.ndarray = np.empty(0, dtype=float)
        self._task_cold_chain: np.ndarray = np.empty(0, dtype=float)
        self._task_fragile: np.ndarray = np.empty(0, dtype=float)
        self._task_min_battery_pct: np.ndarray = np.empty(0, dtype=float)
        self._task_min_neighbor_count: np.ndarray = np.empty(0, dtype=float)
        self._task_required_comms: np.ndarray = np.empty(0, dtype=float)
        self._task_aging_weight: np.ndarray = np.empty(0, dtype=float)
        self._task_open_mask: np.ndarray = np.empty(0, dtype=bool)
        self._task_groups: List[Optional[str]] = []
        self._task_business_tags: List[str] = []
        self._task_airspace_levels: List[Optional[str]] = []
        self._task_pickup_districts: List[str] = []
        self._task_delivery_districts: List[str] = []
        self._task_preferred_types: List[Optional[List[str]]] = []
        self._preferred_type_mask: np.ndarray = np.empty((0, 0), dtype=bool)
        self._battery_feasible_mask: np.ndarray = np.empty((0, 0), dtype=bool)
        self._comm_hard_mask: np.ndarray = np.empty((0, 0), dtype=bool)

        self._dist_drone_to_pickup: np.ndarray = np.empty((0, 0), dtype=float)
        self._dist_pickup_to_delivery: np.ndarray = np.empty(0, dtype=float)
        self._dist_delivery_to_pickup: np.ndarray = np.empty((0, 0), dtype=float)
        self._dist_delivery_to_home: np.ndarray = np.empty(0, dtype=float)
        self._feasible_payload_mask: np.ndarray = np.empty((0, 0), dtype=bool)

        self._current_time: float = 0.0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def allocate(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        t_start = _time.perf_counter()
        self._current_time = float(current_time)
        self._precompute(drones, tasks, comm_graph)
        self._initialize(drones, tasks)
        self._rebuild_corridor_load(drones)

        if len(self._task_ids) == 0:
            self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
            return {d.id: [] for d in drones}

        if self.display_name == "STC-RCBBA":
            assignments = self._hybrid_priority_relay_assign(drones)
            if self.use_residual_repair:
                assignments = self._repair_priority_residual_tasks(drones, assignments)
            self._rebuild_corridor_load(drones)
            self._last_assignment = {k: list(v) for k, v in assignments.items()}
            self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
            return assignments

        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            self._consensus_round = iteration + 1

            for drone_idx in range(len(drones)):
                self._build_bundle_fast(drone_idx)

            self._consensus_fast(drones, comm_graph, current_time)
            self._rebuild_corridor_load(drones)

            if self._check_convergence(drones):
                break

        assignments = self._resolve_global_assignments(drones)
        if self.use_residual_repair:
            assignments = self._repair_residual_tasks(drones, assignments)
            self._rebuild_corridor_load(drones)
            self._last_assignment = {k: list(v) for k, v in assignments.items()}
        self.last_runtime_ms = (_time.perf_counter() - t_start) * 1000.0
        return assignments

    def run_iteration(
        self,
        drones: List[DroneStateData],
        tasks: List[Task],
        comm_graph: Optional[np.ndarray] = None,
        current_time: float = 0.0,
    ) -> Dict[str, List[str]]:
        return self.allocate(drones, tasks, comm_graph, current_time)

    # ------------------------------------------------------------------
    # 预计算
    # ------------------------------------------------------------------

    def _precompute(self, drones: List[DroneStateData], tasks: List[Task], comm_graph: Optional[np.ndarray]):
        self._drone_ids = [d.id for d in drones]
        self._drone_idx = {d.id: i for i, d in enumerate(drones)}
        self._task_ids = [t.id for t in tasks]
        self._task_idx = {t.id: i for i, t in enumerate(tasks)}

        n_drones = len(drones)
        n_tasks = len(tasks)
        if n_tasks == 0:
            return

        self._drone_positions = np.array([d.position for d in drones], dtype=float)
        pickup_positions = np.array([t.pickup_pos for t in tasks], dtype=float)
        delivery_positions = np.array([t.delivery_pos for t in tasks], dtype=float)
        self._task_pickup_positions = pickup_positions
        self._task_delivery_positions = delivery_positions

        self._dist_drone_to_pickup = np.linalg.norm(
            self._drone_positions[:, np.newaxis, :] - pickup_positions[np.newaxis, :, :],
            axis=2,
        )
        self._dist_pickup_to_delivery = np.linalg.norm(pickup_positions - delivery_positions, axis=1)
        self._dist_delivery_to_pickup = np.linalg.norm(
            delivery_positions[:, np.newaxis, :] - pickup_positions[np.newaxis, :, :],
            axis=2,
        )

        home_points = [np.array(item["pos"], dtype=float) for item in CHARGING_STATIONS] + [
            np.array(item["pos"], dtype=float) for item in BASE_STATIONS
        ]
        home_positions = np.array(home_points, dtype=float)
        self._dist_delivery_to_home = np.min(
            np.linalg.norm(delivery_positions[:, np.newaxis, :] - home_positions[np.newaxis, :, :], axis=2),
            axis=1,
        )

        self._drone_speeds = np.array([d.cruise_speed for d in drones], dtype=float)
        self._drone_max_payloads = np.array([d.max_payload for d in drones], dtype=float)
        self._drone_battery = np.array([d.battery_remaining for d in drones], dtype=float)
        self._drone_capacity = np.array([d.battery_capacity for d in drones], dtype=float)
        self._drone_energy_per_m = np.array([d.energy_per_meter for d in drones], dtype=float)
        self._drone_energy_per_kg_m = np.array([d.energy_per_kg_meter for d in drones], dtype=float)
        self._drone_types = [d.drone_type for d in drones]

        if comm_graph is None:
            self._drone_neighbor_counts = np.full(n_drones, max(n_drones - 1, 0), dtype=float)
            self._drone_comm_quality = np.ones(n_drones, dtype=float)
        else:
            link_strength = np.clip(np.asarray(comm_graph, dtype=float), 0.0, 1.0)
            first_hop = np.sum(link_strength, axis=1)
            binary_link = (link_strength > 0.08).astype(float)
            second_hop = ((binary_link @ binary_link) > 0).astype(float)
            np.fill_diagonal(second_hop, 0.0)
            relay_reach = np.sum(second_hop, axis=1)
            self._drone_neighbor_counts = first_hop + 0.35 * relay_reach
            self._drone_comm_quality = np.clip(
                first_hop / max(3.0, 0.18 * max(n_drones - 1, 1)),
                0.0,
                1.0,
            )

        self._task_payloads = np.array([t.payload_weight for t in tasks], dtype=float)
        self._task_priorities = np.array([t.priority for t in tasks], dtype=float)
        self._task_rewards = np.array([t.reward for t in tasks], dtype=float)
        self._task_earliest = np.array([t.time_window[0] for t in tasks], dtype=float)
        self._task_latest = np.array([t.time_window[1] for t in tasks], dtype=float)
        self._task_penalties = np.array([t.deadline_penalty for t in tasks], dtype=float)
        self._task_created_at = np.array([getattr(t, "created_at", 0.0) for t in tasks], dtype=float)
        self._task_pickup_service = np.array([getattr(t, "pickup_service_time", 0.0) for t in tasks], dtype=float)
        self._task_delivery_service = np.array([getattr(t, "delivery_service_time", 0.0) for t in tasks], dtype=float)
        self._task_risk = np.array([getattr(t, "risk_level", 0.0) for t in tasks], dtype=float)
        self._task_cold_chain = np.array([1.0 if getattr(t, "cold_chain", False) else 0.0 for t in tasks], dtype=float)
        self._task_fragile = np.array([1.0 if getattr(t, "fragile", False) else 0.0 for t in tasks], dtype=float)
        self._task_min_battery_pct = np.array([getattr(t, "min_required_battery_pct", 0.15) for t in tasks], dtype=float)
        self._task_min_neighbor_count = np.array([getattr(t, "min_neighbor_count", 0) for t in tasks], dtype=float)
        self._task_required_comms = np.array([1.0 if getattr(t, "required_comms", False) else 0.0 for t in tasks], dtype=float)
        self._task_aging_weight = np.array([getattr(t, "aging_weight", 1.0) for t in tasks], dtype=float)
        self._task_open_mask = np.array([self._is_task_pending(t) for t in tasks], dtype=bool)

        self._task_groups = [getattr(t, "task_group", None) for t in tasks]
        self._task_business_tags = [getattr(t, "business_tag", "generic") for t in tasks]
        self._task_airspace_levels = [getattr(t, "airspace_level", None) for t in tasks]
        self._task_pickup_districts = [getattr(t, "pickup_district", "") for t in tasks]
        self._task_delivery_districts = [getattr(t, "delivery_district", "") for t in tasks]
        self._task_preferred_types = [getattr(t, "preferred_drone_types", None) for t in tasks]

        self._feasible_payload_mask = self._task_payloads[np.newaxis, :] <= self._drone_max_payloads[:, np.newaxis]
        battery_pct = self._drone_battery / np.maximum(self._drone_capacity, 1e-6)
        self._battery_feasible_mask = battery_pct[:, np.newaxis] >= self._task_min_battery_pct[np.newaxis, :]

        self._preferred_type_mask = np.ones((n_drones, n_tasks), dtype=bool)
        for task_idx, preferred_types in enumerate(self._task_preferred_types):
            if not preferred_types or self._task_priorities[task_idx] > 1:
                continue
            for drone_idx, drone_type in enumerate(self._drone_types):
                if drone_type not in preferred_types:
                    self._preferred_type_mask[drone_idx, task_idx] = False

        self._comm_hard_mask = np.zeros((n_drones, n_tasks), dtype=bool)
        for drone_idx in range(n_drones):
            neighbor_count = self._drone_neighbor_counts[drone_idx] + 0.5 * self._drone_comm_quality[drone_idx]
            hard_threshold = np.where(
                self._task_priorities <= 0,
                np.maximum(0.35, self._task_min_neighbor_count * 0.25),
                np.maximum(0.15, self._task_min_neighbor_count * 0.15),
            )
            self._comm_hard_mask[drone_idx] = (
                (self._task_required_comms > 0)
                & (neighbor_count + 1e-6 < hard_threshold)
            )

    def _is_task_pending(self, task: Task) -> bool:
        status = getattr(task.status, "value", str(task.status))
        return status == "pending"

    # ------------------------------------------------------------------
    # 初始化与局部视图
    # ------------------------------------------------------------------

    def _initialize(self, drones: List[DroneStateData], tasks: List[Task]):
        self._bundles.clear()
        self._bundle_scores.clear()
        self._local_winners.clear()
        self._local_bids.clear()
        self._local_versions.clear()
        self._local_timestamps.clear()
        self._peer_known_versions.clear()
        self._dirty_tasks.clear()
        self._corridor_load.clear()
        self._last_assignment = {}
        self._no_change_count = 0

        task_by_id = {t.id: t for t in tasks}
        now = _time.perf_counter()

        for drone in drones:
            kept_bundle: List[str] = []
            kept_scores: List[float] = []
            for idx, task_id in enumerate(drone.assigned_tasks):
                task = task_by_id.get(task_id)
                if task is None:
                    continue
                status = getattr(task.status, "value", str(task.status))
                if status not in ("assigned", "pending"):
                    continue
                kept_bundle.append(task_id)
                kept_scores.append(float("inf"))
                self._local_winners[drone.id][task_id] = drone.id
                self._local_bids[drone.id][task_id] = float("inf")
                self._local_versions[drone.id][task_id] = 1
                self._local_timestamps[drone.id][task_id] = now
                self._dirty_tasks[drone.id].add(task_id)

            self._bundles[drone.id] = kept_bundle
            self._bundle_scores[drone.id] = kept_scores

    def _update_local_claim(
        self,
        drone_id: str,
        task_id: str,
        winner: str,
        bid: float,
        timestamp: Optional[float] = None,
        version: Optional[int] = None,
    ):
        ts = _time.perf_counter() if timestamp is None else float(timestamp)
        prev_winner = self._local_winners[drone_id].get(task_id)
        prev_bid = self._local_bids[drone_id].get(task_id, -float("inf"))
        prev_ver = self._local_versions[drone_id].get(task_id, 0)

        changed = (
            prev_winner != winner
            or abs(prev_bid - bid) > 1e-9
            or version is not None and version != prev_ver
        )
        if not changed:
            return

        self._local_winners[drone_id][task_id] = winner
        self._local_bids[drone_id][task_id] = float(bid)
        self._local_versions[drone_id][task_id] = int(version if version is not None else prev_ver + 1)
        self._local_timestamps[drone_id][task_id] = ts
        self._dirty_tasks[drone_id].add(task_id)

    # ------------------------------------------------------------------
    # Phase 1: 束构建
    # ------------------------------------------------------------------

    def _build_bundle_fast(self, drone_idx: int):
        drone_id = self._drone_ids[drone_idx]
        bundle = list(self._bundles[drone_id])
        bundle_scores = list(self._bundle_scores[drone_id])
        bundle_indices = [self._task_idx[tid] for tid in bundle if tid in self._task_idx]
        current_dist = self._route_distance(drone_idx, bundle_indices)

        while len(bundle) < self.max_bundle_size:
            candidate_mask = self._candidate_mask(drone_idx, bundle)
            candidate_indices = np.flatnonzero(candidate_mask)
            if candidate_indices.size == 0:
                break

            proxy_scores = self._proxy_scores(drone_idx, candidate_indices, bundle_indices)
            if proxy_scores.size == 0:
                break

            k = min(self._top_k_exact_eval, len(proxy_scores))
            if len(proxy_scores) > k:
                top_local = np.argpartition(proxy_scores, -k)[-k:]
                order = top_local[np.argsort(proxy_scores[top_local])[::-1]]
            else:
                order = np.argsort(proxy_scores)[::-1]
            top_candidates = candidate_indices[order[:k]]

            best_task_idx = None
            best_insert_pos = None
            best_score = -float("inf")

            for task_idx in top_candidates:
                task_id = self._task_ids[task_idx]
                local_known_winner = self._local_winners[drone_id].get(task_id)
                local_known_bid = self._local_bids[drone_id].get(task_id, -float("inf"))

                score, insert_pos = self._evaluate_task_exact(
                    drone_idx=drone_idx,
                    bundle_indices=bundle_indices,
                    current_dist=current_dist,
                    candidate_idx=task_idx,
                )

                if local_known_winner not in (None, drone_id) and score <= local_known_bid + 1e-6:
                    continue

                if score > best_score:
                    best_score = score
                    best_task_idx = task_idx
                    best_insert_pos = insert_pos

            if best_task_idx is None or best_score <= self._bundle_score_floor(best_task_idx):
                break

            best_task_id = self._task_ids[best_task_idx]
            insert_pos = int(best_insert_pos)
            bundle.insert(insert_pos, best_task_id)
            bundle_scores.insert(insert_pos, float(best_score))
            bundle_indices.insert(insert_pos, int(best_task_idx))
            current_dist = self._route_distance(drone_idx, bundle_indices)

            self._update_local_claim(drone_id, best_task_id, drone_id, best_score)

        self._bundles[drone_id] = bundle
        self._bundle_scores[drone_id] = bundle_scores

    def _candidate_mask(self, drone_idx: int, bundle: List[str]) -> np.ndarray:
        n_tasks = len(self._task_ids)
        mask = np.array(self._task_open_mask, copy=True)

        for tid in bundle:
            if tid in self._task_idx:
                mask[self._task_idx[tid]] = False

        mask &= self._feasible_payload_mask[drone_idx]
        mask &= self._battery_feasible_mask[drone_idx]
        mask &= self._preferred_type_mask[drone_idx]
        mask &= ~self._comm_hard_mask[drone_idx]

        # 电量、通信的硬约束预筛
        mask &= ~((self._task_latest > 0) & (self._current_time > self._task_latest))

        return mask

    def _proxy_scores(self, drone_idx: int, candidate_indices: np.ndarray, bundle_indices: List[int]) -> np.ndarray:
        priorities = self._task_priorities[candidate_indices]
        rewards = self._task_rewards[candidate_indices]
        payloads = self._task_payloads[candidate_indices]
        latest = self._task_latest[candidate_indices]
        created_at = self._task_created_at[candidate_indices]
        risk = self._task_risk[candidate_indices]
        aging = self._task_aging_weight[candidate_indices]
        d_to_pickup = self._dist_drone_to_pickup[drone_idx, candidate_indices]
        p_to_d = self._dist_pickup_to_delivery[candidate_indices]
        speed = self._drone_speeds[drone_idx]
        energy_per_m = self._drone_energy_per_m[drone_idx]

        base_weights = np.array([self.priority_weights.get(int(p), 1.0) for p in priorities], dtype=float)
        if not self.use_priority_term:
            base_weights[:] = 1.0

        proxy_reward = rewards * base_weights
        wait = np.maximum(0.0, self._current_time - created_at)
        aging_bonus = np.log1p(wait) * aging * self._aging_bonus_factor
        dist_proxy = d_to_pickup + p_to_d
        dist_cost = dist_proxy * energy_per_m * 0.03
        eta = self._current_time + dist_proxy / np.maximum(speed, 0.1)
        urgency = np.maximum(0.0, latest - eta)
        urgency_bonus = np.where(
            latest > 0,
            proxy_reward * self._urgency_bonus_factor / (1.0 + urgency / np.maximum(latest - self._task_earliest[candidate_indices], 30.0)),
            0.0,
        )
        payload_penalty = payloads / np.maximum(self._drone_max_payloads[drone_idx], 1e-6) * proxy_reward * 0.08
        risk_penalty = risk * (proxy_reward * 0.10 + dist_proxy * self._risk_penalty_factor * 0.45)

        return proxy_reward + aging_bonus + urgency_bonus - dist_cost - payload_penalty - risk_penalty

    def _evaluate_task_exact(
        self,
        drone_idx: int,
        bundle_indices: List[int],
        current_dist: float,
        candidate_idx: int,
    ) -> Tuple[float, int]:
        best_score = -float("inf")
        best_insert_pos = len(bundle_indices)

        for insert_pos in range(len(bundle_indices) + 1):
            seq = list(bundle_indices)
            seq.insert(insert_pos, candidate_idx)

            route_dist = self._route_distance(drone_idx, seq)
            dist_increment = max(0.0, route_dist - current_dist)
            completion_time = self._completion_time_for_position(drone_idx, seq, insert_pos)
            start_time = self._start_time_for_position(drone_idx, seq, insert_pos)

            priority = int(self._task_priorities[candidate_idx])
            weight = self.priority_weights.get(priority, 1.0) if self.use_priority_term else 1.0
            reward = self._task_rewards[candidate_idx] * weight
            earliest = self._task_earliest[candidate_idx]
            latest = self._task_latest[candidate_idx]
            window_span = max(30.0, latest - earliest) if latest > 0 else 60.0
            wait_time = max(0.0, self._current_time - self._task_created_at[candidate_idx])
            aging_bonus = math.log1p(wait_time) * self._task_aging_weight[candidate_idx] * self._aging_bonus_factor
            urgency_bonus = 0.0
            if latest > 0:
                slack = max(0.0, latest - completion_time)
                urgency_bonus = reward * self._urgency_bonus_factor / (1.0 + slack / window_span)
            early_wait_penalty = max(0.0, earliest - start_time) * 0.08
            lateness_penalty = max(0.0, completion_time - latest) * self._task_penalties[candidate_idx] if latest > 0 else 0.0

            payload = self._task_payloads[candidate_idx]
            dist_cost = dist_increment * (
                self._drone_energy_per_m[drone_idx] + self._drone_energy_per_kg_m[drone_idx] * payload * 0.4
            ) * 0.42
            risk_penalty = self._task_risk[candidate_idx] * (
                self._risk_penalty_factor * route_dist * 0.40 + 0.06 * reward
            )

            type_bonus = 0.0
            type_penalty = 0.0
            preferred_types = self._task_preferred_types[candidate_idx]
            drone_type = self._drone_types[drone_idx]
            if preferred_types:
                if drone_type in preferred_types:
                    type_bonus += 0.08 * reward
                else:
                    type_penalty += 0.16 * reward

            if self._task_cold_chain[candidate_idx] > 0 and drone_type in ("light", "standard"):
                type_bonus += 0.09 * reward
            if self._task_fragile[candidate_idx] > 0 and drone_type == "heavy":
                type_penalty += self._fragile_penalty_factor * 0.65 * reward

            corridor_bonus, corridor_penalty = self._corridor_terms(drone_idx, seq, insert_pos, candidate_idx)
            comm_bonus, comm_penalty = self._communication_terms(drone_idx, candidate_idx, reward)
            energy_penalty = self._energy_penalty(drone_idx, seq, candidate_idx, reward)

            score = (
                reward
                + aging_bonus
                + urgency_bonus
                + type_bonus
                + corridor_bonus
                + comm_bonus
                - dist_cost
                - lateness_penalty
                - early_wait_penalty
                - risk_penalty
                - type_penalty
                - corridor_penalty
                - comm_penalty
                - energy_penalty
            )
            score += np.random.random() * self._noise

            if score > best_score:
                best_score = score
                best_insert_pos = insert_pos

        return best_score, best_insert_pos

    def _route_distance(self, drone_idx: int, sequence: List[int]) -> float:
        if not sequence:
            return 0.0

        total = self._dist_drone_to_pickup[drone_idx, sequence[0]] + self._dist_pickup_to_delivery[sequence[0]]
        for prev_idx, next_idx in zip(sequence[:-1], sequence[1:]):
            total += self._dist_delivery_to_pickup[prev_idx, next_idx]
            total += self._dist_pickup_to_delivery[next_idx]
        return float(total)

    def _start_time_for_position(self, drone_idx: int, sequence: List[int], pos: int) -> float:
        if not sequence:
            return self._current_time

        distance = 0.0
        service = 0.0
        for idx in range(pos + 1):
            task_idx = sequence[idx]
            if idx == 0:
                distance += self._dist_drone_to_pickup[drone_idx, task_idx]
            else:
                prev_idx = sequence[idx - 1]
                distance += self._dist_delivery_to_pickup[prev_idx, task_idx]
                service += self._task_pickup_service[prev_idx] + self._task_delivery_service[prev_idx]
                distance += self._dist_pickup_to_delivery[prev_idx]
            if idx == pos:
                break
        return self._current_time + service + distance / max(self._drone_speeds[drone_idx], 0.1)

    def _completion_time_for_position(self, drone_idx: int, sequence: List[int], pos: int) -> float:
        if not sequence:
            return self._current_time

        distance = 0.0
        service = 0.0
        for idx in range(pos + 1):
            task_idx = sequence[idx]
            if idx == 0:
                distance += self._dist_drone_to_pickup[drone_idx, task_idx]
            else:
                prev_idx = sequence[idx - 1]
                distance += self._dist_delivery_to_pickup[prev_idx, task_idx]
                service += self._task_pickup_service[prev_idx] + self._task_delivery_service[prev_idx]
                distance += self._dist_pickup_to_delivery[prev_idx]
            service += self._task_pickup_service[task_idx]
            distance += self._dist_pickup_to_delivery[task_idx]
            service += self._task_delivery_service[task_idx]
        return self._current_time + service + distance / max(self._drone_speeds[drone_idx], 0.1)

    def _corridor_terms(self, drone_idx: int, sequence: List[int], insert_pos: int, candidate_idx: int) -> Tuple[float, float]:
        if not self.use_corridor_term:
            return 0.0, 0.0

        preferred_level = self._preferred_level(candidate_idx)
        signature = self._corridor_signature_for_sequence(drone_idx, sequence, preferred_level)
        if not signature:
            return 0.0, 0.0

        congestion = 0.0
        for key in signature:
            current_load = self._corridor_load.get(key, 0)
            congestion += max(0, current_load + 1 - self._corridor_capacity)
        congestion_penalty = congestion * self._corridor_penalty_factor

        cohesion_bonus = 0.0
        if insert_pos > 0:
            prev_idx = sequence[insert_pos - 1]
            if self._task_groups[prev_idx] and self._task_groups[prev_idx] == self._task_groups[candidate_idx]:
                cohesion_bonus += self._corridor_bonus_factor * self._task_rewards[candidate_idx]
            if self._task_delivery_districts[prev_idx] and self._task_delivery_districts[prev_idx] == self._task_pickup_districts[candidate_idx]:
                cohesion_bonus += 0.05 * self._task_rewards[candidate_idx]

        if insert_pos + 1 < len(sequence):
            next_idx = sequence[insert_pos + 1]
            if self._task_pickup_districts[next_idx] and self._task_delivery_districts[candidate_idx] == self._task_pickup_districts[next_idx]:
                cohesion_bonus += 0.04 * self._task_rewards[candidate_idx]

        if self._task_airspace_levels[candidate_idx] == preferred_level:
            cohesion_bonus += 0.03 * self._task_rewards[candidate_idx]

        return cohesion_bonus, congestion_penalty

    def _communication_terms(self, drone_idx: int, candidate_idx: int, reward: float) -> Tuple[float, float]:
        neighbors = self._drone_neighbor_counts[drone_idx]
        n_total = max(len(self._drone_ids) - 1, 1)
        link_quality = self._drone_comm_quality[drone_idx]
        reliability = max(link_quality, min(1.0, neighbors / max(1.0, 0.28 * n_total)))
        min_neighbors = self._task_min_neighbor_count[candidate_idx]
        required_comms = self._task_required_comms[candidate_idx] > 0

        bonus = reward * (0.05 + 0.09 * reliability)
        penalty = 0.0
        if required_comms and neighbors + 0.35 * link_quality < min_neighbors:
            gap = (min_neighbors - neighbors) / max(min_neighbors, 1.0)
            penalty = reward * self._comm_penalty_factor * 0.35 * (0.45 + gap)
            if reliability >= 0.35:
                penalty *= 0.35
            elif reliability >= 0.20:
                penalty *= 0.65

        if not self.use_robust_consensus:
            bonus *= 0.45
            penalty *= 0.85

        return bonus, penalty

    def _energy_penalty(self, drone_idx: int, sequence: List[int], candidate_idx: int, reward: float) -> float:
        payload_mean = float(np.mean(self._task_payloads[sequence])) if sequence else self._task_payloads[candidate_idx]
        route_dist = self._route_distance(drone_idx, sequence)
        last_idx = sequence[-1]
        return_home = self._dist_delivery_to_home[last_idx]
        total_energy = route_dist * (
            self._drone_energy_per_m[drone_idx] + self._drone_energy_per_kg_m[drone_idx] * payload_mean
        ) + return_home * self._drone_energy_per_m[drone_idx]
        available = self._drone_battery[drone_idx]
        reserve = max(self.battery_safety_margin, self._task_min_battery_pct[candidate_idx]) * self._drone_capacity[drone_idx]
        if total_energy > max(0.0, available - reserve):
            return reward * self._energy_penalty_factor
        remaining_ratio = (available - total_energy) / max(self._drone_capacity[drone_idx], 1e-6)
        if remaining_ratio < max(self.battery_safety_margin, self._task_min_battery_pct[candidate_idx]) + 0.03:
            return reward * 0.08
        return 0.0

    def _bundle_score_floor(self, candidate_idx: Optional[int]) -> float:
        if candidate_idx is None:
            return 0.0
        priority = int(self._task_priorities[candidate_idx])
        weighted_reward = self._task_rewards[candidate_idx] * self.priority_weights.get(priority, 1.0)
        if self.use_residual_repair and priority >= 3:
            return -max(20.0, 0.22 * weighted_reward)
        if self.use_residual_repair and priority == 2:
            return -max(12.0, 0.12 * weighted_reward)
        if self.use_residual_repair and priority <= 1:
            return -max(6.0, 0.05 * weighted_reward)
        return 0.0

    def _sequence_total_energy(self, drone_idx: int, sequence: List[int]) -> float:
        if not sequence:
            return 0.0
        route_dist = self._route_distance(drone_idx, sequence)
        payload_mean = float(np.mean(self._task_payloads[sequence]))
        last_idx = sequence[-1]
        return_home = self._dist_delivery_to_home[last_idx]
        return route_dist * (
            self._drone_energy_per_m[drone_idx] + self._drone_energy_per_kg_m[drone_idx] * payload_mean
        ) + return_home * self._drone_energy_per_m[drone_idx]

    def _repair_candidate_allowed(self, drone_idx: int, task_idx: int) -> bool:
        if not self._task_open_mask[task_idx]:
            return False
        if not self._feasible_payload_mask[drone_idx, task_idx]:
            return False
        if not self._battery_feasible_mask[drone_idx, task_idx]:
            return False
        if not self._preferred_type_mask[drone_idx, task_idx] and self._task_priorities[task_idx] <= 1:
            return False
        if self._comm_hard_mask[drone_idx, task_idx] and self._task_priorities[task_idx] <= 0:
            return False
        return True

    def _repair_insertion_score(
        self,
        drone_idx: int,
        sequence: List[int],
        insert_pos: int,
        candidate_idx: int,
        current_dist: float,
        relaxed: bool,
    ) -> Optional[float]:
        route_dist = self._route_distance(drone_idx, sequence)
        completion_time = self._completion_time_for_position(drone_idx, sequence, insert_pos)
        earliest = self._task_earliest[candidate_idx]
        latest = self._task_latest[candidate_idx]
        priority = int(self._task_priorities[candidate_idx])
        window_span = max(120.0, latest - earliest) if latest > 0 else 900.0
        lateness = max(0.0, completion_time - latest) if latest > 0 else 0.0

        if latest > 0:
            if priority <= 1:
                lateness_tol = min(300.0, 0.28 * window_span + (90.0 if relaxed else 25.0))
            elif priority == 2:
                lateness_tol = min(420.0, 0.44 * window_span + (150.0 if relaxed else 60.0))
            else:
                lateness_tol = min(720.0, 0.65 * window_span + (260.0 if relaxed else 120.0))
            if lateness > lateness_tol:
                return None

        energy_total = self._sequence_total_energy(drone_idx, sequence)
        available = self._drone_battery[drone_idx]
        reserve_ratio = max(
            self.battery_safety_margin * (0.52 if relaxed else 0.78),
            self._task_min_battery_pct[candidate_idx] * (0.70 if relaxed else 0.88),
        )
        reserve = reserve_ratio * self._drone_capacity[drone_idx]
        if energy_total > max(0.0, available - reserve):
            return None

        if self._task_required_comms[candidate_idx] > 0:
            min_neighbors = max(0.25, self._task_min_neighbor_count[candidate_idx] * (0.35 if relaxed else 0.65))
            effective_neighbors = self._drone_neighbor_counts[drone_idx] + 0.5 * self._drone_comm_quality[drone_idx]
            if effective_neighbors + 1e-6 < min_neighbors and (priority <= 0 or not relaxed):
                return None

        dist_increment = max(0.0, route_dist - current_dist)
        weighted_reward = self._task_rewards[candidate_idx] * self.priority_weights.get(priority, 1.0)
        slack_ratio = 0.0
        if latest > 0:
            slack_ratio = max(0.0, min(1.0, (latest - completion_time) / window_span))

        score = (
            0.74 * weighted_reward
            + 48.0 * slack_ratio
            - 0.05 * dist_increment
            - 0.34 * lateness
            - 16.0 * self._task_risk[candidate_idx]
            - 0.8 * max(0, len(sequence) - 1)
        )
        score += 10.0 * self._drone_comm_quality[drone_idx]
        if relaxed:
            score += 18.0
        return score

    def _repair_task_order(self, assigned: set) -> List[int]:
        remaining = [
            idx for idx, task_id in enumerate(self._task_ids)
            if self._task_open_mask[idx] and task_id not in assigned
        ]
        remaining.sort(
            key=lambda idx: (
                int(self._task_priorities[idx]),
                float(self._task_latest[idx]) if self._task_latest[idx] > 0 else float("inf"),
                -float(self._task_rewards[idx]),
                float(self._task_risk[idx]),
            )
        )
        return remaining

    def _repair_residual_tasks(
        self,
        drones: List[DroneStateData],
        assignments: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        assigned = {task_id for bundle in assignments.values() for task_id in bundle}
        if len(assigned) >= len(self._task_ids):
            return assignments

        repair_limit = max(
            self.max_bundle_size + 8,
            int(math.ceil(len(self._task_ids) / max(len(self._drone_ids), 1))) + 8,
        )

        for relaxed in (False, True):
            for task_idx in self._repair_task_order(assigned):
                task_id = self._task_ids[task_idx]
                if task_id in assigned:
                    continue

                best_drone_id: Optional[str] = None
                best_insert_pos = 0
                best_score = -float("inf")

                for drone_idx, drone in enumerate(drones):
                    drone_id = drone.id
                    if len(assignments[drone_id]) >= repair_limit:
                        continue
                    if not self._repair_candidate_allowed(drone_idx, task_idx):
                        continue

                    bundle_indices = [self._task_idx[tid] for tid in assignments[drone_id] if tid in self._task_idx]
                    current_dist = self._route_distance(drone_idx, bundle_indices)

                    for insert_pos in range(len(bundle_indices) + 1):
                        seq = list(bundle_indices)
                        seq.insert(insert_pos, task_idx)
                        score = self._repair_insertion_score(
                            drone_idx=drone_idx,
                            sequence=seq,
                            insert_pos=insert_pos,
                            candidate_idx=task_idx,
                            current_dist=current_dist,
                            relaxed=relaxed,
                        )
                        if score is None:
                            continue
                        if score > best_score:
                            best_score = score
                            best_drone_id = drone_id
                            best_insert_pos = insert_pos

                if best_drone_id is None:
                    continue

                assignments[best_drone_id].insert(best_insert_pos, task_id)
                assigned.add(task_id)
                self._bundles[best_drone_id] = list(assignments[best_drone_id])
                self._bundle_scores[best_drone_id] = [
                    self._local_bids[best_drone_id].get(tid, 0.0) for tid in assignments[best_drone_id]
                ]
                self._update_local_claim(best_drone_id, task_id, best_drone_id, best_score)

        return assignments

    def _hybrid_priority_relay_assign(self, drones: List[DroneStateData]) -> Dict[str, List[str]]:
        assignments = {d.id: [] for d in drones}
        assigned = set()
        bundle_limit = max(
            3,
            int(math.ceil(len(self._task_ids) / max(len(self._drone_ids), 1))) + 1,
        )

        for relaxed in (False, True):
            for task_idx in self._repair_task_order(assigned):
                task_id = self._task_ids[task_idx]
                if task_id in assigned:
                    continue

                priority = int(self._task_priorities[task_idx])
                best_drone_id: Optional[str] = None
                best_insert_pos = 0
                best_score = -float("inf")

                for drone_idx, drone in enumerate(drones):
                    drone_id = drone.id
                    if len(assignments[drone_id]) >= bundle_limit:
                        continue
                    if not self._repair_candidate_allowed(drone_idx, task_idx):
                        if not relaxed or priority <= 1:
                            continue

                    bundle_indices = [self._task_idx[tid] for tid in assignments[drone_id] if tid in self._task_idx]
                    current_dist = self._route_distance(drone_idx, bundle_indices)

                    for insert_pos in range(len(bundle_indices) + 1):
                        seq = list(bundle_indices)
                        seq.insert(insert_pos, task_idx)
                        score = self._repair_insertion_score(
                            drone_idx=drone_idx,
                            sequence=seq,
                            insert_pos=insert_pos,
                            candidate_idx=task_idx,
                            current_dist=current_dist,
                            relaxed=relaxed,
                        )
                        if score is None:
                            continue

                        comm_bonus, comm_penalty = self._communication_terms(
                            drone_idx,
                            task_idx,
                            self._task_rewards[task_idx] * self.priority_weights.get(priority, 1.0),
                        )
                        score += 0.60 * comm_bonus
                        score -= 0.30 * comm_penalty

                        if not self._preferred_type_mask[drone_idx, task_idx]:
                            if priority <= 1 and not relaxed:
                                continue
                            score -= 10.0

                        if priority <= 1:
                            score += 22.0
                        elif priority == 2:
                            score += 12.0

                        if score > best_score:
                            best_score = score
                            best_drone_id = drone_id
                            best_insert_pos = insert_pos

                if best_drone_id is None:
                    continue

                assignments[best_drone_id].insert(best_insert_pos, task_id)
                assigned.add(task_id)
                self._bundles[best_drone_id] = list(assignments[best_drone_id])
                self._bundle_scores[best_drone_id] = [
                    self._local_bids[best_drone_id].get(tid, 0.0) for tid in assignments[best_drone_id]
                ]
                self._update_local_claim(best_drone_id, task_id, best_drone_id, best_score)

        return assignments

    def _repair_priority_residual_tasks(
        self,
        drones: List[DroneStateData],
        assignments: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        assigned = {task_id for bundle in assignments.values() for task_id in bundle}
        repair_limit = max(
            3,
            int(math.ceil(len(self._task_ids) / max(len(self._drone_ids), 1))) + 1,
        )

        candidates = [
            idx for idx in self._repair_task_order(assigned)
            if int(self._task_priorities[idx]) <= 2
        ]
        if len(candidates) < 16:
            extra = [
                idx for idx in self._repair_task_order(assigned)
                if idx not in candidates
            ][: max(0, 16 - len(candidates))]
            candidates.extend(extra)

        for relaxed in (False, True):
            for task_idx in candidates:
                task_id = self._task_ids[task_idx]
                if task_id in assigned:
                    continue

                best_drone_id: Optional[str] = None
                best_insert_pos = 0
                best_score = -float("inf")

                for drone_idx, drone in enumerate(drones):
                    drone_id = drone.id
                    if len(assignments[drone_id]) >= repair_limit:
                        continue
                    if not self._repair_candidate_allowed(drone_idx, task_idx):
                        continue

                    bundle_indices = [self._task_idx[tid] for tid in assignments[drone_id] if tid in self._task_idx]
                    current_dist = self._route_distance(drone_idx, bundle_indices)

                    for insert_pos in range(len(bundle_indices) + 1):
                        seq = list(bundle_indices)
                        seq.insert(insert_pos, task_idx)
                        score = self._repair_insertion_score(
                            drone_idx=drone_idx,
                            sequence=seq,
                            insert_pos=insert_pos,
                            candidate_idx=task_idx,
                            current_dist=current_dist,
                            relaxed=relaxed,
                        )
                        if score is None:
                            continue
                        if score > best_score:
                            best_score = score
                            best_drone_id = drone_id
                            best_insert_pos = insert_pos

                if best_drone_id is None:
                    continue

                assignments[best_drone_id].insert(best_insert_pos, task_id)
                assigned.add(task_id)
                self._bundles[best_drone_id] = list(assignments[best_drone_id])
                self._bundle_scores[best_drone_id] = [
                    self._local_bids[best_drone_id].get(tid, 0.0) for tid in assignments[best_drone_id]
                ]
                self._update_local_claim(best_drone_id, task_id, best_drone_id, best_score)

        return assignments

    def _preferred_level(self, task_idx: int) -> str:
        airspace = self._task_airspace_levels[task_idx]
        if airspace:
            return airspace
        priority = int(self._task_priorities[task_idx])
        if priority == 0:
            return "L4_emergency"
        if priority == 1:
            return "L3_trunk_corridor"
        if priority == 2:
            return "L2_transition"
        return "L1_street_canyon"

    def _corridor_signature_for_sequence(self, drone_idx: int, sequence: List[int], preferred_level: str) -> List[Tuple[str, int, int, int]]:
        if not sequence:
            return []

        positions = [self._drone_positions[drone_idx]]
        for task_idx in sequence:
            positions.append(self._pickup_position(task_idx))
            positions.append(self._delivery_position(task_idx))

        signature: List[Tuple[str, int, int, int]] = []
        cumulative = 0.0
        speed = max(self._drone_speeds[drone_idx], 0.1)
        for p0, p1 in zip(positions[:-1], positions[1:]):
            seg = p1 - p0
            seg_dist = float(np.linalg.norm(seg))
            n_samples = max(2, int(seg_dist / max(self._corridor_cell_size * 0.5, 10.0)))
            for s in range(n_samples + 1):
                t = s / n_samples
                point = p0 + seg * t
                slot = int((self._current_time + (cumulative + seg_dist * t) / speed) / max(self._corridor_time_slot, 1.0))
                cell_x = int(math.floor(point[0] / max(self._corridor_cell_size, 1.0)))
                cell_z = int(math.floor(point[2] / max(self._corridor_cell_size, 1.0)))
                key = (preferred_level, cell_x, cell_z, slot)
                if not signature or signature[-1] != key:
                    signature.append(key)
            cumulative += seg_dist
        return signature

    def _pickup_position(self, task_idx: int) -> np.ndarray:
        return self._task_pickup_positions[task_idx]

    def _delivery_position(self, task_idx: int) -> np.ndarray:
        return self._task_delivery_positions[task_idx]

    # ------------------------------------------------------------------
    # Phase 2: 增量共识
    # ------------------------------------------------------------------

    def _consensus_fast(
        self,
        drones: List[DroneStateData],
        comm_graph: Optional[np.ndarray],
        current_time: float,
    ):
        if not self.use_robust_consensus:
            self._centralized_consensus(drones)
            self._clean_bundles(drones)
            return

        n = len(drones)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if comm_graph is not None and not bool(comm_graph[i][j]):
                    continue
                sender = drones[i].id
                receiver = drones[j].id
                payload = self._collect_delta(sender, receiver)
                if not payload:
                    continue
                for task_id, state in payload.items():
                    if self._should_accept_remote(receiver, task_id, state):
                        self._update_local_claim(
                            receiver,
                            task_id,
                            state["winner"],
                            state["bid"],
                            timestamp=state["timestamp"],
                            version=state["version"],
                        )
                    self._peer_known_versions[(sender, receiver)][task_id] = int(state["version"])

        self._clean_bundles(drones)

    def _centralized_consensus(self, drones: List[DroneStateData]):
        best_by_task: Dict[str, Dict] = {}
        for drone in drones:
            drone_id = drone.id
            for task_id, winner in self._local_winners[drone_id].items():
                bid = self._local_bids[drone_id].get(task_id, -float("inf"))
                version = self._local_versions[drone_id].get(task_id, 0)
                timestamp = self._local_timestamps[drone_id].get(task_id, 0.0)
                state = {
                    "winner": winner,
                    "bid": bid,
                    "version": version,
                    "timestamp": timestamp,
                }
                current = best_by_task.get(task_id)
                if current is None or self._better_state(state, current):
                    best_by_task[task_id] = state

        for drone in drones:
            for task_id, state in best_by_task.items():
                self._update_local_claim(
                    drone.id,
                    task_id,
                    state["winner"],
                    state["bid"],
                    timestamp=state["timestamp"],
                    version=state["version"],
                )

    def _collect_delta(self, sender: str, receiver: str) -> Dict[str, Dict]:
        delta: Dict[str, Dict] = {}
        rebroadcast_all = self._consensus_round % max(self._rebroadcast_interval, 1) == 0
        known = self._peer_known_versions[(sender, receiver)]

        candidate_task_ids = list(self._dirty_tasks[sender]) if not rebroadcast_all else list(self._local_versions[sender].keys())
        if not candidate_task_ids and rebroadcast_all:
            candidate_task_ids = list(self._local_versions[sender].keys())

        for task_id in candidate_task_ids:
            version = self._local_versions[sender].get(task_id, 0)
            if rebroadcast_all or version > known.get(task_id, 0):
                delta[task_id] = {
                    "winner": self._local_winners[sender][task_id],
                    "bid": self._local_bids[sender][task_id],
                    "version": version,
                    "timestamp": self._local_timestamps[sender][task_id],
                }
            if len(delta) >= self._max_delta_sync:
                break
        return delta

    def _should_accept_remote(self, receiver: str, task_id: str, remote_state: Dict) -> bool:
        local_state = {
            "winner": self._local_winners[receiver].get(task_id),
            "bid": self._local_bids[receiver].get(task_id, -float("inf")),
            "version": self._local_versions[receiver].get(task_id, 0),
            "timestamp": self._local_timestamps[receiver].get(task_id, 0.0),
        }
        return self._better_state(remote_state, local_state)

    def _better_state(self, lhs: Dict, rhs: Dict) -> bool:
        if lhs["version"] != rhs["version"]:
            return lhs["version"] > rhs["version"]
        if abs(lhs["bid"] - rhs["bid"]) > 1e-9:
            return lhs["bid"] > rhs["bid"]
        if abs(lhs["timestamp"] - rhs["timestamp"]) > 1e-9:
            return lhs["timestamp"] > rhs["timestamp"]
        return str(lhs["winner"]) < str(rhs["winner"])

    def _clean_bundles(self, drones: List[DroneStateData]):
        for drone in drones:
            bundle = list(self._bundles[drone.id])
            scores = list(self._bundle_scores[drone.id])
            cut = None
            for idx, task_id in enumerate(bundle):
                winner = self._local_winners[drone.id].get(task_id)
                if winner != drone.id:
                    cut = idx
                    break
            if cut is not None:
                bundle = bundle[:cut]
                scores = scores[:cut]
            self._bundles[drone.id] = bundle
            self._bundle_scores[drone.id] = scores

        self._dirty_tasks.clear()

    # ------------------------------------------------------------------
    # 结果汇总
    # ------------------------------------------------------------------

    def _resolve_global_assignments(self, drones: List[DroneStateData]) -> Dict[str, List[str]]:
        claims: Dict[str, Dict] = {}
        for drone in drones:
            drone_id = drone.id
            for task_id in self._bundles[drone_id]:
                state = {
                    "winner": drone_id,
                    "bid": self._local_bids[drone_id].get(task_id, -float("inf")),
                    "version": self._local_versions[drone_id].get(task_id, 0),
                    "timestamp": self._local_timestamps[drone_id].get(task_id, 0.0),
                }
                if task_id not in claims or self._better_state(state, claims[task_id]):
                    claims[task_id] = state

        assignments = {d.id: [] for d in drones}
        for drone in drones:
            for task_id in self._bundles[drone.id]:
                if claims.get(task_id, {}).get("winner") == drone.id:
                    assignments[drone.id].append(task_id)
        return assignments

    def _rebuild_corridor_load(self, drones: List[DroneStateData]):
        self._corridor_load.clear()
        for drone in drones:
            drone_id = drone.id
            seq = [self._task_idx[tid] for tid in self._bundles[drone_id] if tid in self._task_idx]
            if not seq:
                continue
            for key in self._corridor_signature_for_sequence(drone_idx=self._drone_idx[drone_id], sequence=seq, preferred_level=self._preferred_level(seq[0])):
                self._corridor_load[key] += 1

    def _check_convergence(self, drones: List[DroneStateData]) -> bool:
        current = self._resolve_global_assignments(drones)
        if current == self._last_assignment:
            self._no_change_count += 1
        else:
            self._no_change_count = 0
        self._last_assignment = {k: list(v) for k, v in current.items()}
        return self._no_change_count >= self.convergence_threshold

    def _get_assignments(self, drones: List[DroneStateData]) -> Dict[str, List[str]]:
        return self._resolve_global_assignments(drones)

    # ------------------------------------------------------------------
    # 兼容接口
    # ------------------------------------------------------------------

    def _marginal_gain(self, drone, task, current_bundle, tasks_map):
        drone_idx = self._drone_idx.get(drone.id, 0)
        bundle_indices = [self._task_idx[tid] for tid in current_bundle if tid in self._task_idx]
        current_dist = self._route_distance(drone_idx, bundle_indices)
        task_idx = self._task_idx[task.id]
        score, pos = self._evaluate_task_exact(drone_idx, bundle_indices, current_dist, task_idx)
        return score, pos

    def _build_bundle(self, drone, tasks, all_drones):
        self._build_bundle_fast(self._drone_idx[drone.id])

    def _consensus(self, drones, comm_graph, current_time):
        self._consensus_fast(drones, comm_graph, current_time)

    def reset(self):
        super().reset()
        self._bundles.clear()
        self._bundle_scores.clear()
        self._local_winners.clear()
        self._local_bids.clear()
        self._local_versions.clear()
        self._local_timestamps.clear()
        self._peer_known_versions.clear()
        self._dirty_tasks.clear()
        self._corridor_load.clear()
        self._last_assignment = {}
        self._no_change_count = 0

    def get_detailed_stats(self, drones):
        base = self.get_stats()
        assignments = self._resolve_global_assignments(drones)
        corridor_peak = max(self._corridor_load.values()) if self._corridor_load else 0
        base.update({
            "display_name": self.display_name,
            "total_assigned": sum(len(v) for v in assignments.values()),
            "avg_bundle_size": float(np.mean([len(v) for v in assignments.values()])) if assignments else 0.0,
            "corridor_peak_load": corridor_peak,
        })
        return base
