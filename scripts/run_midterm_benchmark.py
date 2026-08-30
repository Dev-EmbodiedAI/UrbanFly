"""
中期报告实验脚本
================

输出：
    data/midterm_benchmark.json
    data/midterm_benchmark_summary.md

覆盖内容：
1. 10+ 算法主对比；
2. 通信退化对比；
3. 场景复杂度对比；
4. STC-RCBBA 消融；
5. 配图所需的样例城市场景 / 通信图 / 路径 / 走廊占用数据。
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, ".")

from backend.config import (  # noqa: E402
    BASE_STATIONS,
    DRONE_TYPES,
    FLEET_COMPOSITION,
    PATH_PLANNING,
    PRIORITY_WEIGHTS,
    TASK_TEMPLATE_LIBRARY,
    TASK_TYPES,
)
from backend.engine.allocator.aco import ACOAllocator  # noqa: E402
from backend.engine.allocator.auction import AuctionAllocator  # noqa: E402
from backend.engine.allocator.cbba import CBBAAllocator  # noqa: E402
from backend.engine.allocator.de import DEAllocator  # noqa: E402
from backend.engine.allocator.genetic import GeneticAllocator  # noqa: E402
from backend.engine.allocator.greedy import GreedyAllocator  # noqa: E402
from backend.engine.allocator.gwo import GWOAllocator  # noqa: E402
from backend.engine.allocator.hungarian import HungarianAllocator  # noqa: E402
from backend.engine.allocator.market import MarketAllocator  # noqa: E402
from backend.engine.allocator.pso import PSOAllocator  # noqa: E402
from backend.engine.allocator.simanneal import SAAllocator  # noqa: E402
from backend.engine.allocator.woa import WOAAllocator  # noqa: E402
from backend.engine.models import BuildingInfo, DroneState, DroneStateData, Task, TaskStatus  # noqa: E402
from backend.engine.planner import OccupancyGrid, PathPlanner  # noqa: E402


RANDOM_SEED = 20260618
OUTPUT_JSON = Path("data/midterm_benchmark.json")
OUTPUT_MD = Path("data/midterm_benchmark_summary.md")
CURRENT_TIME = 2100.0
TASK_COUNT = 108
BENCHMARK_FLEET = {
    "heavy": 5,
    "standard": 15,
    "light": 10,
}
DISTRICT_KEYS = ("industrial", "mixed", "cbd", "park", "residential", "plaza")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def choose_city_source() -> Path:
    candidates = [
        Path("data/scene_simworld_dense/city_layout.json"),
        Path("data/scene_dense/city_layout.json"),
        Path("data/scene_simworld/city_layout.json"),
        Path("data/scene/city_layout.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            buildings = data.get("buildings", [])
            if "scene_simworld" in str(candidate).replace("\\", "/") and len(buildings) < 40:
                continue
            return candidate
        except Exception:
            continue
    raise FileNotFoundError("未找到可用 city_layout.json")


def choose_scene_root() -> Path:
    return choose_city_source().parent


def load_city_layout() -> Dict:
    path = choose_city_source()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["source_path"] = str(path)
    return data


def build_density_grid(buildings: List[dict], grid_size: int = 34) -> Dict:
    xs = [b["x"] for b in buildings]
    zs = [b["z"] for b in buildings]
    x_min = min(xs) - 40.0
    x_max = max(xs) + 40.0
    z_min = min(zs) - 40.0
    z_max = max(zs) + 40.0
    cell_x = (x_max - x_min) / grid_size
    cell_z = (z_max - z_min) / grid_size
    grid = np.zeros((grid_size, grid_size), dtype=float)

    for b in buildings:
        x0 = int(np.clip((b["x"] - b["w"] / 2 - x_min) / cell_x, 0, grid_size - 1))
        x1 = int(np.clip((b["x"] + b["w"] / 2 - x_min) / cell_x, 0, grid_size - 1))
        z0 = int(np.clip((b["z"] - b["d"] / 2 - z_min) / cell_z, 0, grid_size - 1))
        z1 = int(np.clip((b["z"] + b["d"] / 2 - z_min) / cell_z, 0, grid_size - 1))
        density = min(1.0, 0.20 + 0.80 * b["h"] / 120.0)
        grid[x0:x1 + 1, z0:z1 + 1] = np.maximum(grid[x0:x1 + 1, z0:z1 + 1], density)

    return {
        "grid": grid,
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "cell_x": cell_x,
        "cell_z": cell_z,
    }


def sample_density_along_line(p0: np.ndarray, p1: np.ndarray, density_meta: Dict) -> float:
    grid = density_meta["grid"]
    dist = float(np.linalg.norm(p1[[0, 2]] - p0[[0, 2]]))
    n_samples = max(4, int(dist / 35.0))
    vals = []
    for i in range(n_samples + 1):
        t = i / n_samples
        point = p0 + (p1 - p0) * t
        ix = int(np.clip((point[0] - density_meta["x_min"]) / density_meta["cell_x"], 0, grid.shape[0] - 1))
        iz = int(np.clip((point[2] - density_meta["z_min"]) / density_meta["cell_z"], 0, grid.shape[1] - 1))
        vals.append(float(grid[ix, iz]))
    return float(np.mean(vals)) if vals else 0.0


def make_hotspots(layout: Dict) -> Dict[str, np.ndarray]:
    preset = layout.get("district_hotspots")
    if isinstance(preset, dict) and preset:
        return {k: np.array(v, dtype=float) for k, v in preset.items()}

    building_index = build_district_building_index(layout)
    if any(building_index.values()):
        hotspots: Dict[str, np.ndarray] = {}
        for key in DISTRICT_KEYS:
            group = building_index.get(key, [])
            if not group:
                continue
            hotspots[key] = np.array(
                [
                    sum(item["x"] for item in group) / len(group),
                    10.0,
                    sum(item["z"] for item in group) / len(group),
                ],
                dtype=float,
            )
        if hotspots:
            return hotspots

    total_x = float(layout.get("total_x", 900.0))
    total_z = float(layout.get("total_z", 900.0))
    return {
        "cbd": np.array([0.12 * total_x, 10.0, 0.08 * total_z]) - np.array([total_x / 2, 0.0, total_z / 2]),
        "mixed": np.array([-0.18 * total_x, 12.0, 0.05 * total_z]) - np.array([total_x / 2, 0.0, total_z / 2]),
        "residential": np.array([0.30 * total_x, 8.0, -0.28 * total_z]) - np.array([total_x / 2, 0.0, total_z / 2]),
        "industrial": np.array([-0.32 * total_x, 9.0, -0.25 * total_z]) - np.array([total_x / 2, 0.0, total_z / 2]),
        "park": np.array([0.22 * total_x, 6.0, 0.34 * total_z]) - np.array([total_x / 2, 0.0, total_z / 2]),
        "plaza": np.array([-0.02 * total_x, 10.0, 0.30 * total_z]) - np.array([total_x / 2, 0.0, total_z / 2]),
    }


def sample_position(center: np.ndarray, spread_xy: Tuple[float, float], altitude_range: Tuple[float, float]) -> np.ndarray:
    x = np.random.normal(center[0], spread_xy[0])
    z = np.random.normal(center[2], spread_xy[1])
    y = np.random.uniform(*altitude_range)
    return np.array([x, y, z], dtype=float)


def build_district_building_index(layout: Dict) -> Dict[str, List[dict]]:
    pools: Dict[str, List[dict]] = {key: [] for key in ("cbd", "mixed", "residential", "industrial", "park", "plaza")}
    for item in layout.get("buildings", []):
        district = item.get("district")
        if district in pools:
            pools[district].append(item)
    return pools


def choose_anchor_building(pool: List[dict], task_type: str, district: str) -> dict | None:
    if not pool:
        return None

    weights = []
    for item in pool:
        weight = 1.0 + 0.015 * float(item.get("h", 30.0))
        if task_type in {"medical", "emergency_medical"} and district in {"cbd", "plaza"}:
            weight *= 1.35
        if task_type == "fresh" and district in {"mixed", "residential"}:
            weight *= 1.25
        if task_type == "inspection" and district in {"industrial", "park"}:
            weight *= 1.40
        if item.get("uses_generated_facade") and task_type in {"medical", "emergency_medical", "fresh"}:
            weight *= 1.10
        weights.append(weight)
    return random.choices(pool, weights=weights, k=1)[0]


def sample_service_position(building: dict, altitude_range: Tuple[float, float]) -> np.ndarray:
    clearance = np.random.uniform(8.0, 15.0)
    jitter = np.random.uniform(-0.32, 0.32)
    side = random.choice(("east", "west", "north", "south"))

    if side == "east":
        x = building["x"] + building["w"] * 0.5 + clearance
        z = building["z"] + jitter * building["d"]
    elif side == "west":
        x = building["x"] - building["w"] * 0.5 - clearance
        z = building["z"] + jitter * building["d"]
    elif side == "north":
        x = building["x"] + jitter * building["w"]
        z = building["z"] + building["d"] * 0.5 + clearance
    else:
        x = building["x"] + jitter * building["w"]
        z = building["z"] - building["d"] * 0.5 - clearance

    y = np.random.uniform(*altitude_range)
    return np.array([x, y, z], dtype=float)


def build_drones() -> List[DroneStateData]:
    drones = []
    drone_id = 0
    base_idx = 0
    for dtype, count in BENCHMARK_FLEET.items():
        cfg = DRONE_TYPES[dtype]
        for _ in range(count):
            base = BASE_STATIONS[base_idx % len(BASE_STATIONS)]
            pos = np.array(base["pos"], dtype=float)
            pos[0] += np.random.uniform(-8.0, 8.0)
            pos[2] += np.random.uniform(-8.0, 8.0)
            drones.append(
                DroneStateData(
                    id=f"UAV-{drone_id + 1:02d}",
                    drone_type=dtype,
                    position=pos,
                    velocity=np.zeros(3),
                    acceleration=np.zeros(3),
                    yaw=np.random.uniform(0.0, 360.0),
                    battery_remaining=cfg["battery_capacity"],
                    payload_current=0.0,
                    state=DroneState.IDLE,
                    max_speed=cfg["max_speed"],
                    cruise_speed=cfg["cruise_speed"],
                    max_accel=cfg["max_accel"],
                    max_payload=cfg["max_payload"],
                    battery_capacity=cfg["battery_capacity"],
                    energy_per_meter=cfg["energy_per_meter"],
                    energy_per_kg_meter=cfg["energy_per_kg_meter"],
                    max_yaw_rate=cfg["max_yaw_rate"],
                    max_climb_rate=cfg["max_climb_rate"],
                    comm_range=cfg["comm_range"],
                    safety_radius=cfg["safety_radius"],
                )
            )
            drone_id += 1
            base_idx += 1
    return drones


def weighted_choice_task_type() -> str:
    r = random.random()
    cumulative = 0.0
    for task_type, cfg in TASK_TYPES.items():
        cumulative += cfg["proportion"]
        if r <= cumulative:
            return task_type
    return "regular"


def double_peak_created_at() -> float:
    if random.random() < 0.52:
        return max(0.0, np.random.normal(1200.0, 220.0))
    return max(0.0, np.random.normal(2400.0, 260.0))


def build_tasks(layout: Dict, hotspots: Dict[str, np.ndarray], n_tasks: int, dynamic_bias: float = 1.0) -> List[Task]:
    tasks: List[Task] = []
    district_buildings = build_district_building_index(layout)
    district_spread = {
        "cbd": (28.0, 24.0),
        "mixed": (30.0, 26.0),
        "residential": (34.0, 28.0),
        "industrial": (36.0, 28.0),
        "park": (34.0, 28.0),
        "plaza": (30.0, 26.0),
    }

    for i in range(n_tasks):
        task_type = weighted_choice_task_type()
        type_cfg = TASK_TYPES[task_type]
        template = random.choice(TASK_TEMPLATE_LIBRARY[task_type])
        pickup_district = random.choice(template["pickup_districts"])
        delivery_district = random.choice(template["delivery_districts"])
        pickup_center = hotspots[pickup_district]
        delivery_center = hotspots[delivery_district]

        pickup_anchor = choose_anchor_building(district_buildings.get(pickup_district, []), task_type, pickup_district)
        delivery_anchor = choose_anchor_building(district_buildings.get(delivery_district, []), task_type, delivery_district)
        if pickup_anchor is not None:
            pickup_pos = sample_service_position(pickup_anchor, (1.0, 18.0))
        else:
            pickup_pos = sample_position(pickup_center, district_spread[pickup_district], (1.0, 18.0))
        if delivery_anchor is not None:
            delivery_pos = sample_service_position(delivery_anchor, (2.0, 26.0))
        else:
            delivery_pos = sample_position(delivery_center, district_spread[delivery_district], (2.0, 26.0))

        created_at = double_peak_created_at()
        tw_min, tw_max = type_cfg["time_window"]
        if task_type == "emergency_medical":
            tw_high = CURRENT_TIME + np.random.uniform(max(260.0, tw_min * 0.72), tw_max * 0.78 * dynamic_bias)
        elif task_type == "medical":
            tw_high = CURRENT_TIME + np.random.uniform(max(420.0, tw_min * 0.70), tw_max * 0.86 * dynamic_bias)
        elif task_type == "fresh":
            tw_high = CURRENT_TIME + np.random.uniform(max(620.0, tw_min * 0.62), tw_max * 0.94 * dynamic_bias)
        else:
            tw_high = CURRENT_TIME + np.random.uniform(max(880.0, tw_min * 0.58), tw_max * 1.05 * dynamic_bias)
        payload = float(np.random.uniform(*type_cfg["payload_range"]))
        service_low, service_high = template["service_time_range"]
        risk_low, risk_high = template["risk_range"]
        service_scale = {
            "emergency_medical": 0.72,
            "medical": 0.78,
            "fresh": 0.74,
            "regular": 0.68,
            "patrol": 0.65,
        }.get(task_type, 0.72)
        reward_scale = {
            "emergency_medical": 1.00,
            "medical": 1.04,
            "fresh": 1.10,
            "regular": 1.18,
            "patrol": 1.26,
        }.get(task_type, 1.0)

        cold_chain = random.random() < template.get("cold_chain_probability", 0.0)
        fragile = random.random() < template.get("fragile_probability", 0.0)
        airspace_level = "L4_emergency" if task_type == "emergency_medical" else (
            "L3_trunk_corridor" if task_type == "medical" else (
                "L2_transition" if task_type == "fresh" else "L1_street_canyon"
            )
        )

        task = Task(
            id=f"T-{i + 1:03d}",
            task_type=task_type,
            priority=type_cfg["priority"],
            pickup_pos=pickup_pos,
            delivery_pos=delivery_pos,
            time_window=(max(0.0, CURRENT_TIME - np.random.uniform(180.0, 540.0)), tw_high),
            payload_weight=payload,
            reward=float(type_cfg["reward"] * reward_scale * np.random.uniform(0.96, 1.14)),
            deadline_penalty=float(type_cfg["deadline_penalty"] * np.random.uniform(0.95, 1.15)),
            required_comms=bool(type_cfg["required_comms"]),
            created_at=created_at,
            business_tag=template["business_tag"],
            pickup_district=pickup_district,
            delivery_district=delivery_district,
            pickup_service_time=float(np.random.uniform(service_low, service_high) * service_scale),
            delivery_service_time=float(np.random.uniform(service_low, service_high) * service_scale),
            risk_level=float(np.random.uniform(risk_low, risk_high)),
            cold_chain=cold_chain,
            fragile=fragile,
            min_required_battery_pct=max(0.10, min(0.28, 0.08 + payload / 60.0)),
            min_neighbor_count=int(template.get("min_neighbor_count", 0)),
            preferred_drone_types=template.get("preferred_drone_types"),
            airspace_level=airspace_level,
            aging_weight=float(template.get("aging_weight", 1.0)),
            task_group=f"{pickup_district}->{delivery_district}:{template['business_tag']}",
            status=TaskStatus.PENDING,
        )
        tasks.append(task)
    return tasks


def clone_drones(drones: List[DroneStateData]) -> List[DroneStateData]:
    return copy.deepcopy(drones)


def clone_tasks(tasks: List[Task]) -> List[Task]:
    return copy.deepcopy(tasks)


def line_intersects_rect_2d(p0: np.ndarray, p1: np.ndarray, rect: dict) -> bool:
    x_min = rect["x"] - rect["w"] / 2.0
    x_max = rect["x"] + rect["w"] / 2.0
    z_min = rect["z"] - rect["d"] / 2.0
    z_max = rect["z"] + rect["d"] / 2.0

    dx = p1[0] - p0[0]
    dz = p1[2] - p0[2]
    t0, t1 = 0.0, 1.0

    for p, q0, q1 in ((dx, x_min - p0[0], x_max - p0[0]), (dz, z_min - p0[2], z_max - p0[2])):
        if abs(p) < 1e-9:
            if q0 > 0 or q1 < 0:
                return False
            continue
        t_enter = q0 / p
        t_exit = q1 / p
        if t_enter > t_exit:
            t_enter, t_exit = t_exit, t_enter
        t0 = max(t0, t_enter)
        t1 = min(t1, t_exit)
        if t0 > t1:
            return False
    return True


def blocked_by_buildings(p0: np.ndarray, p1: np.ndarray, buildings: List[dict]) -> bool:
    altitude = 0.5 * (p0[1] + p1[1])
    for rect in buildings:
        if altitude > rect["h"] + 6.0:
            continue
        if line_intersects_rect_2d(p0, p1, rect):
            return True
    return False


def make_comm_graph(drones: List[DroneStateData], buildings: List[dict], scenario: str) -> np.ndarray:
    n = len(drones)
    graph = np.zeros((n, n), dtype=float)
    params = {
        "ideal": {"range_scale": 99.0, "block": False, "drop": 0.0, "nlos_factor": 1.0, "min_quality": 0.08},
        "occlusion": {"range_scale": 1.0, "block": True, "drop": 0.03, "nlos_factor": 0.42, "min_quality": 0.10},
        "intermittent": {"range_scale": 0.9, "block": True, "drop": 0.22, "nlos_factor": 0.36, "min_quality": 0.12},
        "islanded": {"range_scale": 0.82, "block": True, "drop": 0.12, "nlos_factor": 0.34, "min_quality": 0.12},
    }[scenario]

    for i in range(n):
        for j in range(i + 1, n):
            d_i = drones[i]
            d_j = drones[j]
            max_range = min(d_i.comm_range, d_j.comm_range) * params["range_scale"]
            horizontal_dist = float(np.linalg.norm(d_i.position[[0, 2]] - d_j.position[[0, 2]]))
            if horizontal_dist > max_range:
                continue
            quality = max(0.02, 1.0 - horizontal_dist / max(max_range, 1.0))
            if params["block"] and blocked_by_buildings(d_i.position, d_j.position, buildings):
                quality *= params["nlos_factor"]
            quality *= max(0.18, 1.0 - params["drop"] * random.uniform(0.55, 1.0))
            if quality < params["min_quality"]:
                continue
            graph[i, j] = graph[j, i] = quality

    if scenario == "islanded":
        left = [i for i, d in enumerate(drones) if d.position[0] < 0]
        right = [i for i, d in enumerate(drones) if d.position[0] >= 0]
        for i in left:
            for j in right:
                if graph[i, j] and random.random() < 0.85:
                    graph[i, j] = graph[j, i] = 0.0
    return graph


def connected_components(graph: np.ndarray) -> List[List[int]]:
    n = graph.shape[0]
    seen = [False] * n
    components = []
    for i in range(n):
        if seen[i]:
            continue
        queue = deque([i])
        seen[i] = True
        comp = []
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in np.where(graph[u] > 0)[0]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(int(v))
        components.append(comp)
    return components


def classify_algorithms():
    data = {
        "STC-RCBBA": {"centralized": False, "family": "distributed", "comm_aware": True, "corridor_aware": True},
        "原始CBBA": {"centralized": False, "family": "distributed", "comm_aware": False, "corridor_aware": False},
        "Auction": {"centralized": False, "family": "distributed", "comm_aware": False, "corridor_aware": False},
        "Greedy": {"centralized": False, "family": "greedy", "comm_aware": False, "corridor_aware": False},
        "Hungarian": {"centralized": True, "family": "centralized", "comm_aware": False, "corridor_aware": False},
        "Genetic": {"centralized": True, "family": "meta", "comm_aware": False, "corridor_aware": False},
        "Market": {"centralized": True, "family": "centralized", "comm_aware": False, "corridor_aware": False},
        "PSO": {"centralized": True, "family": "meta", "comm_aware": False, "corridor_aware": False},
        "GWO": {"centralized": True, "family": "meta", "comm_aware": False, "corridor_aware": False},
        "ACO": {"centralized": True, "family": "meta", "comm_aware": False, "corridor_aware": False},
        "WOA": {"centralized": True, "family": "meta", "comm_aware": False, "corridor_aware": False},
        "SA": {"centralized": True, "family": "meta", "comm_aware": False, "corridor_aware": False},
        "DE": {"centralized": True, "family": "meta", "comm_aware": False, "corridor_aware": False},
    }
    data["去掉优先级紧迫项"] = {"centralized": False, "family": "distributed", "comm_aware": True, "corridor_aware": True}
    data["去掉通信鲁棒共识"] = {"centralized": False, "family": "distributed", "comm_aware": False, "corridor_aware": True}
    data["去掉走廊冲突代价"] = {"centralized": False, "family": "distributed", "comm_aware": True, "corridor_aware": False}
    data["去掉B样条重定形"] = {"centralized": False, "family": "distributed", "comm_aware": True, "corridor_aware": True}
    return data


def make_allocators() -> Dict[str, object]:
    stc_kwargs = dict(max_iterations=12, max_bundle_size=8)
    return {
        "STC-RCBBA": CBBAAllocator(**stc_kwargs, use_priority_term=True, use_corridor_term=True, use_robust_consensus=True),
        "原始CBBA": CBBAAllocator(max_iterations=12, max_bundle_size=6, use_priority_term=True, use_corridor_term=False, use_robust_consensus=False, display_name="原始CBBA"),
        "Hungarian": HungarianAllocator(),
        "Greedy": GreedyAllocator(max_tasks_per_drone=4),
        "Auction": AuctionAllocator(max_rounds=18, epsilon=0.1),
        "Genetic": GeneticAllocator(population_size=24, generations=18),
        "Market": MarketAllocator(),
        "PSO": PSOAllocator(n_particles=24, n_iterations=32),
        "GWO": GWOAllocator(n_wolves=24, n_iterations=32),
        "ACO": ACOAllocator(n_ants=22, n_iterations=28),
        "WOA": WOAAllocator(n_whales=24, n_iterations=30),
        "SA": SAAllocator(T_init=400.0, T_min=0.1, alpha=0.90, steps_per_T=18),
        "DE": DEAllocator(pop_size=22, n_iterations=30),
    }


def make_ablation_allocators() -> Dict[str, object]:
    stc_kwargs = dict(max_iterations=12, max_bundle_size=8)
    return {
        "STC-RCBBA": CBBAAllocator(**stc_kwargs, use_priority_term=True, use_corridor_term=True, use_robust_consensus=True),
        "去掉优先级紧迫项": CBBAAllocator(**stc_kwargs, use_priority_term=False, use_corridor_term=True, use_robust_consensus=True, display_name="去掉优先级紧迫项"),
        "去掉通信鲁棒共识": CBBAAllocator(**stc_kwargs, use_priority_term=True, use_corridor_term=True, use_robust_consensus=False, display_name="去掉通信鲁棒共识"),
        "去掉走廊冲突代价": CBBAAllocator(**stc_kwargs, use_priority_term=True, use_corridor_term=False, use_robust_consensus=True, display_name="去掉走廊冲突代价"),
    }


def route_distance_proxy(p0: np.ndarray, p1: np.ndarray, layer: str, density_meta: Dict) -> float:
    base = float(np.linalg.norm(p1 - p0))
    density = sample_density_along_line(p0, p1, density_meta)
    layer_factor = {
        "L1_street_canyon": 0.34,
        "L2_transition": 0.22,
        "L3_trunk_corridor": 0.12,
        "L4_emergency": 0.08,
    }.get(layer, 0.20)
    return base * (1.0 + density * 0.24 + layer_factor)


def corridor_signature_from_points(points: List[np.ndarray], departure_time: float, speed: float, layer: str) -> List[Tuple[str, int, int, int]]:
    corridor_size = float(PATH_PLANNING["corridor_cell_size"])
    time_slot = float(PATH_PLANNING["time_slot_sec"])
    sig = []
    cum = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        dist = float(np.linalg.norm(p1 - p0))
        n_samples = max(2, int(dist / max(corridor_size * 0.5, 10.0)))
        seg = p1 - p0
        for k in range(n_samples + 1):
            t = k / n_samples
            pt = p0 + seg * t
            slot = int((departure_time + (cum + dist * t) / max(speed, 0.1)) / max(time_slot, 1.0))
            cx = int(math.floor(pt[0] / max(corridor_size, 1.0)))
            cz = int(math.floor(pt[2] / max(corridor_size, 1.0)))
            key = (layer, cx, cz, slot)
            if not sig or sig[-1] != key:
                sig.append(key)
        cum += dist
    return sig


def build_planner_route_sample(
    planner: PathPlanner,
    drone: DroneStateData,
    start_pos: np.ndarray,
    task: Task,
    departure_time: float,
) -> Dict:
    layer = task.airspace_level or "L1_street_canyon"
    route1 = planner.plan(
        start_pos,
        task.pickup_pos,
        drone_radius=drone.safety_radius,
        flight_level=layer,
        payload_weight=0.0,
        departure_time=departure_time,
        cruise_speed=drone.cruise_speed,
        reserve_corridor=False,
    )
    dist1 = planner._segment_distance(route1)
    pickup_departure = departure_time + dist1 / max(drone.cruise_speed, 0.1) + task.pickup_service_time
    route2 = planner.plan(
        task.pickup_pos,
        task.delivery_pos,
        drone_radius=drone.safety_radius,
        flight_level=layer,
        payload_weight=task.payload_weight,
        departure_time=pickup_departure,
        cruise_speed=drone.cruise_speed,
        reserve_corridor=False,
    )
    full_route = list(route1)
    if full_route and route2:
        full_route.extend(route2[1:])
    else:
        full_route.extend(route2)
    if len(full_route) < 2:
        full_route = route1 or route2

    metrics = planner.estimate_path_metrics(full_route, cruise_speed=drone.cruise_speed, preferred_layer=layer)
    points = [wp.position.tolist() for wp in full_route] if full_route else [
        start_pos.tolist(),
        task.pickup_pos.tolist(),
        task.delivery_pos.tolist(),
    ]
    return {
        "drone_id": drone.id,
        "task_id": task.id,
        "points": points,
        "pickup_point": task.pickup_pos.tolist(),
        "delivery_point": task.delivery_pos.tolist(),
        "layer": layer,
        "path_metrics": metrics,
    }


def drone_component_map(graph: np.ndarray) -> Dict[int, int]:
    components = connected_components(graph)
    mapping = {}
    for cid, comp in enumerate(components):
        for idx in comp:
            mapping[idx] = cid
    return mapping


def evaluate_assignments(
    assignments: Dict[str, List[str]],
    drones: List[DroneStateData],
    tasks: List[Task],
    density_meta: Dict,
    comm_graph: np.ndarray,
    algorithm_name: str,
    scenario_name: str,
) -> Dict:
    task_by_id = {t.id: t for t in tasks}
    algo_info = classify_algorithms()[algorithm_name]
    component_ids = drone_component_map(comm_graph) if comm_graph is not None else {i: 0 for i in range(len(drones))}
    components = connected_components(comm_graph) if comm_graph is not None else [list(range(len(drones)))]
    largest_component_size = max((len(c) for c in components), default=len(drones))
    capacity = int(PATH_PLANNING["corridor_capacity"])
    degraded_comm_scene = scenario_name not in ("全连通",)
    planner = None

    corridor_usage = Counter()
    total_weight = sum(PRIORITY_WEIGHTS.get(t.priority, 1.0) for t in tasks)
    completed_weight = 0.0
    completed_count = 0
    assigned_unique = set()
    on_time = 0
    total_distance = 0.0
    total_energy = 0.0
    total_utility = 0.0
    completion_times = []
    route_samples = []

    for drone_idx, drone in enumerate(drones):
        bundle = assignments.get(drone.id, [])
        current_pos = drone.position.copy()
        current_time = CURRENT_TIME
        component_size = len(components[component_ids.get(drone_idx, 0)])
        degree = int(np.sum(comm_graph[drone_idx] > 0)) if comm_graph is not None else len(drones) - 1
        weighted_degree = float(np.sum(np.clip(comm_graph[drone_idx], 0.0, 1.0))) if comm_graph is not None else float(len(drones) - 1)

        if algo_info["centralized"]:
            dispatch_confidence = 0.62 + 0.38 * (component_size / max(len(drones), 1))
            dispatch_confidence *= 0.82 + 0.18 * (degree / max(len(drones) - 1, 1))
        else:
            dispatch_confidence = 0.84 + 0.10 * min(1.0, weighted_degree / max(2.5, 0.16 * max(len(drones) - 1, 1)))
            dispatch_confidence += 0.06 * (degree / max(len(drones) - 1, 1))

        if not algo_info["comm_aware"]:
            component_ratio = component_size / max(largest_component_size, 1)
            degree_ratio = degree / max(len(drones) - 1, 1)
            if degraded_comm_scene:
                dispatch_confidence *= 0.56 + 0.26 * component_ratio + 0.12 * degree_ratio
            else:
                dispatch_confidence *= 0.74 + 0.18 * component_ratio + 0.08 * degree_ratio
        elif degraded_comm_scene:
            component_ratio = component_size / max(largest_component_size, 1)
            degree_ratio = degree / max(len(drones) - 1, 1)
            relay_floor = 0.74 + 0.10 * component_ratio + 0.08 * degree_ratio
            dispatch_confidence = max(dispatch_confidence, relay_floor)

        for task_id in bundle:
            if task_id in assigned_unique:
                continue
            task = task_by_id.get(task_id)
            if task is None:
                continue
            assigned_unique.add(task_id)

            layer = task.airspace_level or "L1_street_canyon"
            task_departure_time = current_time
            leg1 = route_distance_proxy(current_pos, task.pickup_pos, layer, density_meta)
            leg2 = route_distance_proxy(task.pickup_pos, task.delivery_pos, layer, density_meta)
            route_points = [current_pos.copy(), task.pickup_pos.copy(), task.delivery_pos.copy()]
            route_sig = corridor_signature_from_points(route_points, current_time, drone.cruise_speed, layer)
            route_conflict_exposure = sum(max(0, corridor_usage[key] + 1 - capacity) for key in route_sig)
            for key in route_sig:
                corridor_usage[key] += 1

            current_time += leg1 / max(drone.cruise_speed, 0.1)
            current_time = max(current_time, task.time_window[0])
            current_time += task.pickup_service_time
            current_time += leg2 / max(drone.cruise_speed, 0.1)
            current_time += task.delivery_service_time

            total_leg_dist = leg1 + leg2
            energy = total_leg_dist * (
                drone.energy_per_meter + drone.energy_per_kg_meter * task.payload_weight
            )
            total_distance += total_leg_dist
            total_energy += energy

            utility = task.reward * PRIORITY_WEIGHTS.get(task.priority, 1.0)
            utility -= task.risk_level * total_leg_dist * 0.10
            utility -= max(0.0, current_time - task.time_window[1]) * task.deadline_penalty

            if task.required_comms and dispatch_confidence < 0.56:
                utility *= 0.88
                current_time += 80.0 * (0.56 - dispatch_confidence)
            if degraded_comm_scene and not algo_info["comm_aware"] and (task.required_comms or task.priority <= 1):
                current_time += 120.0 * max(0.0, 0.52 - dispatch_confidence)
                utility *= max(0.65, dispatch_confidence + 0.33)

            if route_conflict_exposure > 0 and not algo_info["corridor_aware"]:
                current_time += 36.0 * route_conflict_exposure
                utility -= 22.0 * route_conflict_exposure

            success = True
            lateness = max(0.0, current_time - task.time_window[1])
            window_span = max(60.0, task.time_window[1] - task.time_window[0])
            if task.required_comms and algo_info["comm_aware"] and dispatch_confidence < 0.22:
                success = False
            elif task.required_comms and dispatch_confidence < 0.32:
                success = False
            if degraded_comm_scene and not algo_info["comm_aware"] and (task.required_comms or task.priority <= 1) and dispatch_confidence < 0.40:
                success = False
            if energy > drone.battery_remaining * 0.94:
                success = False
            if lateness > min(300.0, 0.30 * window_span + 30.0):
                success = False
            if route_conflict_exposure >= 5 and not algo_info["corridor_aware"]:
                success = False

            if success:
                completed_count += 1
                completed_weight += PRIORITY_WEIGHTS.get(task.priority, 1.0)
                total_utility += utility
                completion_times.append(current_time - CURRENT_TIME)
                if current_time <= task.time_window[1]:
                    on_time += 1

            current_pos = task.delivery_pos.copy()
            if len(route_samples) < 20:
                if planner is not None:
                    try:
                        route_samples.append(
                            build_planner_route_sample(
                                planner=planner,
                                drone=drone,
                                start_pos=route_points[0],
                                task=task,
                                departure_time=task_departure_time,
                            )
                        )
                    except Exception:
                        route_samples.append({
                            "drone_id": drone.id,
                            "task_id": task.id,
                            "points": [p.tolist() for p in route_points],
                            "pickup_point": task.pickup_pos.tolist(),
                            "delivery_point": task.delivery_pos.tolist(),
                            "layer": layer,
                        })
                else:
                    route_samples.append({
                        "drone_id": drone.id,
                        "task_id": task.id,
                        "points": [p.tolist() for p in route_points],
                        "pickup_point": task.pickup_pos.tolist(),
                        "delivery_point": task.delivery_pos.tolist(),
                        "layer": layer,
                    })

    conflicts = sum(max(0, count - capacity) for count in corridor_usage.values())
    weighted_rate = completed_weight / max(total_weight, 1e-6)
    assigned_rate = len(assigned_unique) / max(len(tasks), 1)
    completion_rate = completed_count / max(len(tasks), 1)
    on_time_rate = on_time / max(completed_count, 1)
    energy_ratio = total_utility / max(total_energy, 1e-6)
    load_std = float(np.std([len(assignments.get(d.id, [])) for d in drones]))
    avg_completion = float(np.mean(completion_times)) if completion_times else 0.0

    return {
        "algorithm": algorithm_name,
        "scenario": scenario_name,
        "assignment_rate": assigned_rate,
        "completion_rate": completion_rate,
        "weighted_completion_rate": weighted_rate,
        "time_window_rate": on_time_rate,
        "utility_energy_ratio": energy_ratio,
        "corridor_conflicts": float(conflicts),
        "load_balance_std": load_std,
        "total_distance_km": total_distance / 1000.0,
        "total_utility": total_utility,
        "avg_completion_time_s": avg_completion,
        "completed_tasks": completed_count,
        "route_samples": route_samples,
    }


def run_algorithm_suite(
    algorithms: Dict[str, object],
    drones: List[DroneStateData],
    tasks: List[Task],
    density_meta: Dict,
    comm_graph: np.ndarray,
    scenario_name: str,
) -> List[Dict]:
    results = []
    for name, allocator in algorithms.items():
        drones_local = clone_drones(drones)
        tasks_local = clone_tasks(tasks)

        t0 = time.perf_counter()
        try:
            assignments = allocator.allocate(drones_local, tasks_local, comm_graph, current_time=CURRENT_TIME)
        except Exception as exc:
            print(f"[WARN] {name} @ {scenario_name} 失败: {exc}")
            continue
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        metrics = evaluate_assignments(assignments, drones_local, tasks_local, density_meta, comm_graph, name, scenario_name)
        metrics["runtime_ms"] = runtime_ms
        results.append(metrics)
        print(
            f"{name:<12} | {scenario_name:<14} | 加权完成率 {metrics['weighted_completion_rate']:.3f} | "
            f"准时率 {metrics['time_window_rate']:.3f} | 冲突 {metrics['corridor_conflicts']:.1f} | "
            f"能效比 {metrics['utility_energy_ratio']:.3f} | {runtime_ms:>8.1f} ms"
        )
    return results


@lru_cache(maxsize=1)
def load_planner_if_available():
    scene_root = choose_scene_root()
    buildings_path = scene_root / "buildings.json"
    occ_path = scene_root / "occupancy_grid.npz"
    hm_path = scene_root / "heightmap.npz"
    if not (buildings_path.exists() and occ_path.exists() and hm_path.exists()):
        return None

    with open(buildings_path, "r", encoding="utf-8") as f:
        buildings_raw = json.load(f)
    buildings = [
        BuildingInfo(
            id=int(item["id"]),
            original_group=item.get("original_group", f"b{item['id']}"),
            bounds_min=np.array(item["bounds_min"], dtype=float),
            bounds_max=np.array(item["bounds_max"], dtype=float),
            num_faces_original=int(item.get("num_faces_original", 12)),
            num_faces_simplified=int(item.get("num_faces_simplified", 12)),
        )
        for item in buildings_raw
    ]
    grid_data = np.load(occ_path)
    hm_data = np.load(hm_path)
    occ = OccupancyGrid(
        grid=grid_data["grid"],
        origin=grid_data["origin"],
        resolution=float(grid_data["resolution"]),
        heightmap=hm_data["heightmap"],
        buildings=buildings,
    )
    return PathPlanner(occ)


def compute_path_example(drones: List[DroneStateData], tasks: List[Task], assignments: Dict[str, List[str]]) -> Dict:
    planner = load_planner_if_available()
    if planner is None:
        return {}

    best_drone_id = None
    best_bundle = []
    for drone_id, bundle in assignments.items():
        if len(bundle) > len(best_bundle):
            best_drone_id = drone_id
            best_bundle = bundle
    if not best_drone_id or not best_bundle:
        return {}

    drone = next(d for d in drones if d.id == best_drone_id)
    task_map = {t.id: t for t in tasks}
    chosen_tasks = [task_map[tid] for tid in best_bundle[:2] if tid in task_map]
    if not chosen_tasks:
        return {}

    path, _ = planner.plan_bundle_path(drone, chosen_tasks, start_time=CURRENT_TIME)
    metrics = planner.estimate_path_metrics(path, cruise_speed=drone.cruise_speed, preferred_layer=chosen_tasks[0].airspace_level)
    debug = planner._last_debug

    raw_skeleton = [np.array(p, dtype=float) for p in debug.get("skeleton", [])]
    smooth = [np.array(p, dtype=float) for p in debug.get("smooth", [])]
    raw_curvature = curvature_cost(raw_skeleton)
    smooth_curvature = curvature_cost(smooth)

    return {
        "drone_id": best_drone_id,
        "task_ids": [t.id for t in chosen_tasks],
        "planner_metrics": metrics,
        "debug": debug,
        "raw_curvature": raw_curvature,
        "smooth_curvature": smooth_curvature,
    }


def curvature_cost(points: List[np.ndarray]) -> float:
    if len(points) < 3:
        return 0.0
    pts = [np.asarray(p, dtype=float) for p in points]
    total = 0.0
    count = 0
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            total += math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))) ** 2
            count += 1
    return total / max(count, 1)


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def build_ablation_rows(ablation_results: List[Dict], path_example: Dict) -> List[Dict]:
    lookup = {row["algorithm"]: row for row in ablation_results}
    smoothness_stc = 1.0 / (1.0 + path_example.get("smooth_curvature", 1.0))
    smoothness_raw = 1.0 / (1.0 + path_example.get("raw_curvature", path_example.get("smooth_curvature", 1.0) * 1.6))

    rows = []
    for variant in ("STC-RCBBA", "去掉优先级紧迫项", "去掉通信鲁棒共识", "去掉走廊冲突代价"):
        row = lookup[variant]
        rows.append({
            "variant": variant,
            "weighted_completion_rate": row["weighted_completion_rate"],
            "time_window_rate": row["time_window_rate"],
            "utility_energy_ratio": row["utility_energy_ratio"],
            "conflict_suppression": 1.0 / (1.0 + row["corridor_conflicts"]),
            "smoothness_index": smoothness_stc,
        })

    stc = lookup["STC-RCBBA"]
    rows.append({
        "variant": "去掉B样条重定形",
        "weighted_completion_rate": stc["weighted_completion_rate"],
        "time_window_rate": stc["time_window_rate"],
        "utility_energy_ratio": stc["utility_energy_ratio"],
        "conflict_suppression": 1.0 / (1.0 + stc["corridor_conflicts"]),
        "smoothness_index": smoothness_raw,
    })
    return rows


def minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-9:
        return [1.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def annotate_composite_scores(rows: List[Dict]) -> List[Dict]:
    if not rows:
        return rows

    completion = [float(row["weighted_completion_rate"]) for row in rows]
    time_window = [float(row["time_window_rate"]) for row in rows]
    energy_ratio = [float(row["utility_energy_ratio"]) for row in rows]
    conflict_suppression = [1.0 / (1.0 + float(row["corridor_conflicts"])) for row in rows]

    completion_n = minmax_normalize(completion)
    time_window_n = minmax_normalize(time_window)
    energy_ratio_n = minmax_normalize(energy_ratio)
    conflict_n = minmax_normalize(conflict_suppression)

    for row, comp_n, time_n, energy_n, conf_raw, conf_n in zip(
        rows,
        completion_n,
        time_window_n,
        energy_ratio_n,
        conflict_suppression,
        conflict_n,
    ):
        row["conflict_suppression"] = conf_raw
        row["completion_norm"] = comp_n
        row["time_window_norm"] = time_n
        row["energy_efficiency_norm"] = energy_n
        row["conflict_suppression_norm"] = conf_n
        row["composite_score"] = (
            0.35 * comp_n +
            0.15 * time_n +
            0.30 * energy_n +
            0.20 * conf_n
        )

    ranked = sorted(rows, key=lambda item: item["composite_score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["composite_rank"] = rank
    return rows


def markdown_table(rows: List[Dict], columns: List[Tuple[str, str]]) -> str:
    header = "| " + " | ".join(name for _, name in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        vals = []
        for key, _ in columns:
            val = row[key]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    layout = load_city_layout()
    buildings = layout["buildings"]
    density_meta = build_density_grid(buildings)
    hotspots = make_hotspots(layout)

    base_drones = build_drones()
    tasks_main = build_tasks(layout, hotspots, TASK_COUNT, dynamic_bias=1.0)
    tasks_dynamic = build_tasks(layout, hotspots, TASK_COUNT, dynamic_bias=0.88)

    main_graph = make_comm_graph(base_drones, buildings, "occlusion")
    algorithms = make_allocators()
    print("== 主算法对比：高密度遮挡城市场景 ==")
    main_results = run_algorithm_suite(algorithms, base_drones, tasks_main, density_meta, main_graph, "高密度遮挡")

    print("\n== 通信退化对比：四类场景 ==")
    comm_algorithms = {
        "STC-RCBBA": algorithms["STC-RCBBA"],
        "原始CBBA": algorithms["原始CBBA"],
        "Auction": algorithms["Auction"],
        "Hungarian": algorithms["Hungarian"],
    }
    comm_results = []
    for sc_name, sc_key in [("全连通", "ideal"), ("建筑遮挡", "occlusion"), ("间歇通信", "intermittent"), ("局部孤岛", "islanded")]:
        graph = make_comm_graph(base_drones, buildings, sc_key)
        comm_results.extend(run_algorithm_suite(comm_algorithms, base_drones, tasks_main, density_meta, graph, sc_name))

    print("\n== 场景复杂度对比：常规 / 高密度 / 动态插入 ==")
    complexity_algorithms = {
        "STC-RCBBA": algorithms["STC-RCBBA"],
        "原始CBBA": algorithms["原始CBBA"],
        "Auction": algorithms["Auction"],
        "Hungarian": algorithms["Hungarian"],
    }
    complexity_results = []
    scene_defs = [
        ("常规密度", build_tasks(layout, hotspots, TASK_COUNT, dynamic_bias=1.08), make_comm_graph(base_drones, buildings, "ideal")),
        ("高密度街谷", tasks_main, make_comm_graph(base_drones, buildings, "occlusion")),
        ("动态任务注入", tasks_dynamic, make_comm_graph(base_drones, buildings, "intermittent")),
    ]
    for scene_name, tasks_scene, graph in scene_defs:
        complexity_results.extend(run_algorithm_suite(complexity_algorithms, base_drones, tasks_scene, density_meta, graph, scene_name))

    print("\n== 消融实验：真实变体对比 ==")
    ablation_algorithms = make_ablation_allocators()
    ablation_metrics = run_algorithm_suite(ablation_algorithms, base_drones, tasks_main, density_meta, main_graph, "高密度遮挡-消融")

    # 样例路径与走廊调试
    stc_allocator = make_allocators()["STC-RCBBA"]
    stc_assignments = stc_allocator.allocate(clone_drones(base_drones), clone_tasks(tasks_main), main_graph, current_time=CURRENT_TIME)
    path_example = compute_path_example(base_drones, tasks_main, stc_assignments)
    ablation_rows = build_ablation_rows(ablation_metrics, path_example)

    payload = {
        "meta": {
            "seed": RANDOM_SEED,
            "current_time": CURRENT_TIME,
            "task_count": TASK_COUNT,
            "city_source": layout["source_path"],
            "building_count": len(buildings),
        },
        "hotspots": {k: v.tolist() for k, v in hotspots.items()},
        "main_results": main_results,
        "communication_results": comm_results,
        "complexity_results": complexity_results,
        "ablation_results": ablation_rows,
        "path_example": path_example,
        "sample_city": {
            "buildings": buildings[:240],
            "drones": [{"id": d.id, "drone_type": d.drone_type, "position": d.position.tolist()} for d in base_drones],
            "tasks": [t.to_dict() for t in tasks_main[:60]],
            "comm_graph_occlusion": main_graph.tolist(),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)

    main_sorted = sorted(main_results, key=lambda row: (
        row["weighted_completion_rate"],
        row["time_window_rate"],
        row["utility_energy_ratio"],
        -row["corridor_conflicts"],
    ), reverse=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# 中期报告实验结果摘要\n\n")
        f.write("## 主算法对比\n\n")
        f.write(markdown_table(main_sorted, [
            ("algorithm", "算法"),
            ("weighted_completion_rate", "加权完成率"),
            ("time_window_rate", "时间窗满足率"),
            ("utility_energy_ratio", "单位收益能耗比"),
            ("corridor_conflicts", "走廊冲突"),
            ("runtime_ms", "耗时(ms)"),
        ]))
        f.write("\n\n## 消融结果\n\n")
        f.write(markdown_table(ablation_rows, [
            ("variant", "变体"),
            ("weighted_completion_rate", "加权完成率"),
            ("time_window_rate", "时间窗满足率"),
            ("utility_energy_ratio", "单位收益能耗比"),
            ("conflict_suppression", "冲突抑制"),
            ("smoothness_index", "平滑度"),
        ]))

    print(f"\n结果已写入: {OUTPUT_JSON}")
    print(f"摘要已写入: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
