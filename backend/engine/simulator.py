"""
仿真主引擎
=========
多无人机城市配送仿真的核心循环。

功能：
- 时间步进仿真循环
- 无人机运动学更新（含风扰）
- 周期性任务分配
- 碰撞检测与解决
- 事件生成与追踪
- 动态任务插入
- 统计信息收集
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple
import time as _time
from collections import defaultdict, deque

from .models import (
    DroneStateData, Task, Waypoint, SimulationEvent, EventType,
    TaskStatus, DroneState, SimulationStats, TaskPriority,
)
from .communication import CommunicationModel
from .wind_model import WindModel
from .collision import CollisionManager
from .multirotor_dynamics import MultirotorDynamics, MultirotorParameters
from .urban_world_model import UrbanWorldModelConfig, UrbanWorldModelMPC
from .dynamic_actors import ScriptedActorField
from ..agents.simulator_bridge import SemanticFleetSimulatorBridge
from .helsinki_frames import body_flu_yaw_rate_to_backend_degrees
from .planner import PathPlanner
from .allocator.base import BaseAllocator
from .allocator.cbba import CBBAAllocator
from ..config import (
    SIMULATION, DRONE_TYPES, FLEET_COMPOSITION, MULTIROTOR_DYNAMICS,
    CAMERA_SENSORS,
    TASK_GENERATION, TASK_TYPES, PRIORITY_WEIGHTS,
    BASE_STATIONS, CHARGING_STATIONS,
    DISTRICT_TASK_RULES, TASK_TEMPLATE_LIBRARY,
)


class Simulator:
    """
    多无人机城市配送仿真引擎。

    典型用法:
        sim = Simulator(scene_config, allocator, comm_model, wind_model)
        sim.initialize_scenario(scenario_def)
        while not sim.is_finished:
            sim.step()
            state = sim.get_state_snapshot()
            # 发送state到前端...
    """

    def __init__(self,
                 scene_config=None,
                 allocator: BaseAllocator = None,
                 comm_model: CommunicationModel = None,
                 wind_model: WindModel = None,
                 planner: PathPlanner = None,
                 static_collision_map=None):
        """
        Args:
            scene_config: SceneConfig 场景配置
            allocator: 任务分配器
            comm_model: 通信模型
            wind_model: 风场模型
            planner: 路径规划器
        """
        self.scene_config = scene_config
        self.allocator = allocator or CBBAAllocator()
        self.comm_model = comm_model or CommunicationModel()
        self.wind_model = wind_model or WindModel()
        self.planner = planner
        self.static_collision_map = static_collision_map

        # 实体
        self.drones: List[DroneStateData] = []
        self.tasks: List[Task] = []
        self.pending_tasks: List[Task] = []

        # 充电站
        self.charging_stations = CHARGING_STATIONS

        # 仿真状态
        self.time: float = 0.0
        self.dt: float = SIMULATION["dt"]
        self.speed_multiplier: float = 1.0
        self.state: str = "stopped"  # "stopped" | "running" | "paused" | "completed"
        self.duration: float = SIMULATION["default_duration"]

        # 事件与统计
        self.events: List[SimulationEvent] = []
        self.stats = SimulationStats()
        self._step_count: int = 0
        self._last_reallocation_time: float = -float("inf")
        self._reallocation_interval: float = SIMULATION["reallocation_interval"]
        self._state_push_counter: int = 0
        self._task_id_counter: int = 0

        # 动态任务生成
        self._last_emergency_gen: float = 0.0
        self._last_regular_batch: float = 0.0
        self.dynamic_tasks_enabled: bool = False

        # 场景分区索引
        self._blocks_by_district: Dict[str, List] = defaultdict(list)
        self._scene_blocks: List = []

        # 碰撞管理
        self.collision_manager = CollisionManager()
        self._last_static_collision_event: Dict[str, float] = {}
        self._multirotor_models: Dict[str, MultirotorDynamics] = {}
        self._world_model_controllers: Dict[str, UrbanWorldModelMPC] = {}
        self._external_policy_commands: Dict[str, dict] = {}
        self._external_policy_visualizations: Dict[str, dict] = {}
        self._external_policy_interventions: Dict[str, int] = {}
        self._static_collision_counts: Dict[str, int] = {}
        self.dynamic_actor_field = ScriptedActorField()
        self._dynamic_collision_active: Dict[str, set[int]] = {}
        self._episode_appearance_perturbation: dict = {}
        self._episode_dynamics_perturbation: dict = {}
        self._episode_wind_offset = np.zeros(3, dtype=float)
        self.semantic_fleet_bridge = SemanticFleetSimulatorBridge()

        # 算法性能追踪
        self._alloc_runtimes: List[float] = []
        self._plan_runtimes: List[float] = []

    # ==================================================================
    # 初始化
    # ==================================================================

    def initialize_scenario(self, scenario_def):
        """
        从场景定义初始化仿真。

        Args:
            scenario_def: ScenarioDefinition 对象
        """
        # A scenario switch starts a new experiment. Do not leak the previous
        # time axis, events, or collision statistics into the new run.
        self.time = 0.0
        self._step_count = 0
        self._state_push_counter = 0
        self._last_reallocation_time = -float("inf")
        self._last_static_collision_event = {}
        self.events = []
        self.stats = SimulationStats()
        self.collision_manager.reset()
        self._external_policy_visualizations = {}

        self.duration = scenario_def.duration
        self.dynamic_tasks_enabled = scenario_def.dynamic_tasks_enabled

        # 初始化无人机编队
        self._init_drones(scenario_def)

        # 确保无人机不在建筑内
        self._nudge_drones_out_of_buildings()

        # 建立场景分区索引
        self._index_scene_blocks()

        # 初始化任务
        self._init_tasks(scenario_def)

        # 配置通信模型
        comm_cfg = scenario_def.comm_scenario if hasattr(scenario_def, 'comm_scenario') else "building_blocked"
        self.comm_model = CommunicationModel(
            buildings=self.scene_config.buildings if self.scene_config else [],
            comm_scenario=comm_cfg
        )
        self.comm_model.get_communication_graph(self.drones)
        for drone in self.drones:
            drone.comm_neighbors = self.comm_model.get_neighbors(
                drone.id, self.drones
            )

        semantic_agent_enabled = self.semantic_fleet_bridge.configure(
            scenario_def, self
        )

        # 初始任务分配
        if semantic_agent_enabled:
            self.semantic_fleet_bridge.reallocate(self)
        else:
            self._reallocate_tasks()

        # A fully scripted scenario already owns every aircraft route.  Do not
        # run the legacy task-bundle planner first: on the Helsinki map that
        # unrelated route can fail before the explicit script or external
        # policy episode is installed.  Non-scripted scenarios retain the
        # existing planning path unchanged.
        has_scripted_route = any(
            bool(config.get("scripted_path"))
            for config in getattr(scenario_def, "drones", [])
        )
        if not has_scripted_route and not semantic_agent_enabled:
            self._plan_new_paths()

        # 实景单机验收等场景可提供确定性航线。它在任务规划之后应用，
        # 从而不会被任务分配器覆盖；所有航段仍需通过静态碰撞场复核。
        self._apply_scripted_paths(scenario_def)
        self._configure_world_models(scenario_def)
        actor_density = (
            1.0
            if scenario_def.name == "single_uav_world_model"
            else (0.0 if semantic_agent_enabled else 0.35)
        )
        self.dynamic_actor_field.reset(
            self._get_scene_bounds(), seed=20260731, density=actor_density
        )
        self._dynamic_collision_active = {}

        self.state = "running"
        self._emit_event(EventType.TAKEOFF, message=f"场景 '{scenario_def.name}' 启动")

    def _init_drones(self, scenario_def):
        """初始化无人机编队"""
        self.drones = []
        self._multirotor_models = {}
        self._world_model_controllers = {}
        self._external_policy_commands = {}
        self._external_policy_interventions = {}
        self._static_collision_counts = {}

        # 如果有自定义配置则使用，否则使用全局FLEET_COMPOSITION
        if scenario_def.drones:
            for i, drone_cfg in enumerate(scenario_def.drones):
                drone = self._create_drone(drone_cfg)
                self.drones.append(drone)
        else:
            # 使用默认编队
            drone_id = 0
            base_idx = 0
            base_deploy_counts = defaultdict(int)
            for drone_type, count in FLEET_COMPOSITION.items():
                type_cfg = DRONE_TYPES[drone_type]
                for _ in range(count):
                    base_slot = base_idx % len(BASE_STATIONS)
                    base = BASE_STATIONS[base_slot]
                    deploy_index = base_deploy_counts[base["id"]]
                    base_deploy_counts[base["id"]] += 1
                    base_idx += 1

                    drone = DroneStateData(
                        id=f"UAV-{drone_id + 1:02d}",
                        drone_type=drone_type,
                        position=self._get_staging_position(base_slot, deploy_index, drone_type),
                        velocity=np.zeros(3),
                        acceleration=np.zeros(3),
                        yaw=np.random.uniform(0, 360),
                        battery_remaining=type_cfg["battery_capacity"],
                        payload_current=0.0,
                        state=DroneState.LANDED,
                        **{k: v for k, v in type_cfg.items() if k != "label" and k != "color"}
                    )
                    self.drones.append(drone)
                    drone_id += 1

    def _get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """优先从建筑包围盒推导局部场景边界，规避旧 scene_config 坐标残留"""
        if (
            self.scene_config
            and str(self.scene_config.metadata.get("layout", "")).startswith("citygs")
        ):
            half_size = self.scene_config.bounds_size / 2
            return self.scene_config.bounds_center - half_size, self.scene_config.bounds_center + half_size

        if self.scene_config and self.scene_config.buildings:
            mins = np.min([b.bounds_min for b in self.scene_config.buildings], axis=0)
            maxs = np.max([b.bounds_max for b in self.scene_config.buildings], axis=0)
            return mins, maxs

        if self.scene_config:
            half_size = self.scene_config.bounds_size / 2
            return self.scene_config.bounds_center - half_size, self.scene_config.bounds_center + half_size

        return np.array([-400.0, 0.0, -450.0]), np.array([400.0, 35.0, 450.0])

    def _get_staging_position(self, base_slot: int, deploy_index: int, drone_type: str) -> np.ndarray:
        """让同一基地的无人机按环形机位散开，避免全部挤在一起起飞"""
        base = np.array(BASE_STATIONS[base_slot]["pos"], dtype=float)
        ring = deploy_index // 4
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        angle = base_slot * 0.75 + deploy_index * golden_angle
        radius = 12.0 + ring * 10.0 + {"heavy": 4.0, "standard": 1.5, "light": -1.5}.get(drone_type, 0.0)

        radial = np.array([np.cos(angle), 0.0, np.sin(angle)])
        tangent = np.array([-radial[2], 0.0, radial[0]])
        lateral = ((deploy_index % 3) - 1) * 4.0

        pos = base + radial * radius + tangent * lateral
        scene_min, scene_max = self._get_scene_bounds()
        margin = np.array([12.0, 0.0, 12.0])
        pos[0] = np.clip(pos[0], scene_min[0] + margin[0], scene_max[0] - margin[0])
        pos[2] = np.clip(pos[2], scene_min[2] + margin[2], scene_max[2] - margin[2])
        pos[1] = max(scene_min[1] + 0.8, 0.8 + ring * 0.15)
        return pos

    def _sample_task_position(self) -> np.ndarray:
        """在场景中采样一个合理的任务位置（避开建筑内部）"""
        import random

        scene_min, scene_max = self._get_scene_bounds()
        max_y = min(35.0, scene_max[1] + 18.0)

        if self.planner and getattr(self.planner, "grid", None) is not None:
            grid = self.planner.grid
            for _ in range(80):
                pos = np.array([
                    random.uniform(scene_min[0], scene_max[0]),
                    random.uniform(2.0, max_y),
                    random.uniform(scene_min[2], scene_max[2]),
                ])
                gx, gz = grid.world_to_grid_xz(pos)
                h = grid.get_height_at(gx, gz)
                fp = grid.footprint_2d[gx, gz] if 0 <= gx < grid.footprint_2d.shape[0] and 0 <= gz < grid.footprint_2d.shape[1] else 1
                if fp == 0:  # 不在建筑内
                    return pos

        if self.static_collision_map is not None and hasattr(
            self.static_collision_map,
            "surface_height",
        ):
            for _ in range(80):
                x = random.uniform(scene_min[0] + 5.0, scene_max[0] - 5.0)
                z = random.uniform(scene_min[2] + 5.0, scene_max[2] - 5.0)
                surface = self.static_collision_map.surface_height(
                    np.array([x, 0.0, z]),
                    1.5,
                )
                if np.isfinite(surface):
                    return np.array(
                        [x, surface + random.uniform(3.0, 16.0), z],
                        dtype=float,
                    )

        return np.array([
            random.uniform(scene_min[0], scene_max[0]),
            random.uniform(2.0, max_y),
            random.uniform(scene_min[2], scene_max[2]),
        ])

    @staticmethod
    def _point_in_polygon_2d(x: float, z: float, polygon: np.ndarray) -> bool:
        """射线法判断点是否在多边形内"""
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, z1 = polygon[i]
            x2, z2 = polygon[(i + 1) % n]
            if ((z1 > z) != (z2 > z)) and (x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-9) + x1):
                inside = not inside
        return inside

    def _sample_block_by_district(self, districts: List[str]):
        """按分区候选集采样街区；场景缺失时返回None"""
        import random
        candidates = []
        for district in districts:
            candidates.extend(self._blocks_by_district.get(district, []))
        if not candidates:
            candidates = self._scene_blocks
        if not candidates:
            return None
        return random.choice(candidates)

    def _sample_task_position_for_block(self, block, purpose: str = "pickup", patrol_mode: bool = False) -> np.ndarray:
        """在指定街区内采样任务位置，优先落在街区内部的可飞行区域上方"""
        import random

        if block is None or getattr(block, "polygon", None) is None or len(block.polygon) < 3:
            return self._sample_task_position()

        polygon = np.asarray(block.polygon, dtype=float)
        min_x, min_z = np.min(polygon[:, 0]), np.min(polygon[:, 1])
        max_x, max_z = np.max(polygon[:, 0]), np.max(polygon[:, 1])

        district = getattr(block, "district", "mixed")
        altitude_ranges = {
            "cbd": (12.0, 32.0),
            "mixed": (8.0, 26.0),
            "residential": (4.0, 18.0),
            "industrial": (6.0, 22.0),
            "park": (6.0, 16.0),
            "plaza": (8.0, 24.0),
        }
        low, high = altitude_ranges.get(district, (5.0, 20.0))
        if patrol_mode:
            low, high = max(8.0, low), high + 6.0
        elif purpose == "delivery" and district in ("cbd", "mixed"):
            low, high = low + 4.0, high + 3.0

        if self.planner and getattr(self.planner, "grid", None) is not None:
            grid = self.planner.grid
            for _ in range(80):
                x = random.uniform(min_x, max_x)
                z = random.uniform(min_z, max_z)
                if not self._point_in_polygon_2d(x, z, polygon):
                    continue
                y = random.uniform(low, high)
                pos = np.array([x, y, z], dtype=float)
                gx, gz = grid.world_to_grid_xz(pos)
                if 0 <= gx < grid.footprint_2d.shape[0] and 0 <= gz < grid.footprint_2d.shape[1]:
                    safe_alt = grid.get_safe_altitude(gx, gz, 4.0)
                    if purpose == "delivery" and district in ("cbd", "mixed"):
                        pos[1] = max(pos[1], min(safe_alt + random.uniform(3.0, 8.0), high + 10.0))
                    elif grid.footprint_2d[gx, gz] == 1:
                        pos[1] = max(pos[1], safe_alt + random.uniform(2.0, 6.0))
                    return pos

        for _ in range(40):
            x = random.uniform(min_x, max_x)
            z = random.uniform(min_z, max_z)
            if self._point_in_polygon_2d(x, z, polygon):
                return np.array([x, random.uniform(low, high), z], dtype=float)

        return self._sample_task_position()

    def _make_patrol_waypoints_for_block(self, block, num_points: int = 5) -> List[np.ndarray]:
        """围绕分区街区生成巡检航点"""
        waypoints = []
        if block is None or len(getattr(block, "polygon", [])) < 3:
            return [self._sample_task_position() for _ in range(num_points)]

        centroid = np.asarray(block.centroid, dtype=float)
        polygon = np.asarray(block.polygon, dtype=float)
        offsets = polygon - centroid
        order = np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))
        ordered = polygon[order]

        for i in range(num_points):
            p = ordered[i % len(ordered)]
            wp = np.array([p[0], np.random.uniform(12.0, 28.0), p[1]], dtype=float)
            waypoints.append(wp)
        return waypoints

    def _nudge_drones_out_of_buildings(self):
        """把卡在建筑内的无人机挪到最近的空地"""
        if self.static_collision_map is not None and hasattr(
            self.static_collision_map,
            "surface_height",
        ):
            for drone in self.drones:
                surface = self.static_collision_map.surface_height(
                    drone.position,
                    drone.safety_radius,
                )
                if np.isfinite(surface):
                    drone.position[1] = max(
                        drone.position[1],
                        surface + drone.safety_radius + 0.5,
                    )
        if self.planner is None or self.planner.grid is None:
            return
        grid = self.planner.grid
        fp = grid.footprint_2d
        for drone in self.drones:
            gx, gz = grid.world_to_grid_xz(drone.position)
            if 0 <= gx < fp.shape[0] and 0 <= gz < fp.shape[1] and fp[gx, gz] == 0:
                continue  # 已在空地
            # BFS 找最近空地
            from collections import deque
            visited = set()
            q = deque([(gx, gz, 0)])
            while q:
                cx, cz, d = q.popleft()
                if (cx, cz) in visited or d > 12:
                    continue
                visited.add((cx, cz))
                if 0 <= cx < fp.shape[0] and 0 <= cz < fp.shape[1] and fp[cx, cz] == 0:
                    world = grid.grid_to_world_xz(cx, cz)
                    drone.position[0] = world[0]
                    drone.position[2] = world[2]
                    break
                for dx, dz in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
                    q.append((cx + dx, cz + dz, d + 1))

    def _index_scene_blocks(self):
        """建立分区街区索引，供任务生成与平台展示复用"""
        self._blocks_by_district = defaultdict(list)
        self._scene_blocks = list(getattr(self.scene_config, "blocks", []) or [])
        for block in self._scene_blocks:
            self._blocks_by_district[getattr(block, "district", "mixed")].append(block)

    def _create_drone(self, cfg: dict) -> DroneStateData:
        """从配置字典创建无人机"""
        drone_type = cfg.get("drone_type", "standard")
        type_cfg = DRONE_TYPES.get(drone_type, DRONE_TYPES["standard"])

        return DroneStateData(
            id=cfg["id"],
            drone_type=drone_type,
            position=np.array(cfg.get("start", [0, 0, 0]), dtype=float),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=0.0,
            battery_remaining=cfg.get("battery_capacity", type_cfg["battery_capacity"]),
            payload_current=0.0,
            state=DroneState.LANDED,
            **{k: v for k, v in type_cfg.items() if k != "label" and k != "color"}
        )

    def _init_tasks(self, scenario_def):
        """初始化任务列表"""
        self.tasks = []

        if scenario_def.tasks:
            for t_cfg in scenario_def.tasks:
                task = self._create_task(t_cfg)
                self.tasks.append(task)
        else:
            task_count = TASK_GENERATION["total_tasks"]
            if self.scene_config:
                task_count = int(
                    self.scene_config.metadata.get("default_task_count") or task_count
                )
            self._generate_random_tasks(task_count)

        self._task_id_counter = len(self.tasks)
        self.pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def _create_task(self, cfg: dict) -> Task:
        """从配置字典创建任务"""
        task_type = cfg.get("task_type", "regular")
        type_cfg = TASK_TYPES.get(task_type, TASK_TYPES["regular"])

        return Task(
            id=cfg["id"],
            task_type=task_type,
            priority=cfg.get("priority", type_cfg["priority"]),
            pickup_pos=np.array(cfg["pickup"], dtype=float),
            delivery_pos=np.array(cfg.get("delivery", cfg["pickup"]), dtype=float),
            time_window=tuple(cfg.get("time_window", type_cfg["time_window"])),
            payload_weight=cfg.get("weight", 1.0),
            reward=cfg.get("reward", type_cfg["reward"]),
            deadline_penalty=cfg.get("deadline_penalty", type_cfg["deadline_penalty"]),
            required_comms=cfg.get("required_comms", type_cfg["required_comms"]),
            created_at=cfg.get("created_at", 0.0),
            business_tag=cfg.get("business_tag", "generic"),
            pickup_block_id=cfg.get("pickup_block_id"),
            delivery_block_id=cfg.get("delivery_block_id"),
            pickup_district=cfg.get("pickup_district", ""),
            delivery_district=cfg.get("delivery_district", ""),
            pickup_service_time=cfg.get("pickup_service_time", 0.0),
            delivery_service_time=cfg.get("delivery_service_time", 0.0),
            risk_level=cfg.get("risk_level", 0.0),
            cold_chain=cfg.get("cold_chain", False),
            fragile=cfg.get("fragile", False),
            min_required_battery_pct=cfg.get("min_required_battery_pct", TASK_GENERATION["default_min_battery_pct"]),
            min_neighbor_count=cfg.get("min_neighbor_count", TASK_GENERATION["default_min_neighbor_count"]),
            preferred_drone_types=cfg.get("preferred_drone_types"),
            airspace_level=cfg.get("airspace_level"),
            aging_weight=cfg.get("aging_weight", 1.0),
            task_group=cfg.get("task_group"),
            patrol_waypoints=[np.array(w, dtype=float) for w in cfg.get("patrol_waypoints", [])] or None,
        )

    # ==================================================================
    # 主仿真循环
    # ==================================================================

    def step(self) -> Optional[dict]:
        """
        推进一个物理时间步长。

        Returns:
            如果到了推送时刻，返回状态快照；否则返回 None
        """
        if self.state != "running":
            return None

        sim_dt = self.dt * self.speed_multiplier
        self.time += sim_dt
        self._step_count += 1
        self.dynamic_actor_field.update(self.time)

        # 1. 更新通信拓扑
        comm_graph = self.comm_model.get_communication_graph(self.drones)
        for drone in self.drones:
            drone.comm_neighbors = self.comm_model.get_neighbors(drone.id, self.drones)

        self.semantic_fleet_bridge.update(self)

        # 2. 周期性任务分配
        if (
            not self.semantic_fleet_bridge.enabled
            and self.time - self._last_reallocation_time >= self._reallocation_interval
        ):
            self._reallocate_tasks()
            self._last_reallocation_time = self.time

        # 3. 路径规划（新分配的任务需要路径）
        if not self.semantic_fleet_bridge.enabled:
            self._plan_new_paths()

        # 4. 无人机物理更新
        for drone in self.drones:
            if drone.state != DroneState.CHARGING:
                self._update_drone_dynamics(drone, sim_dt)

        # 5. 碰撞检测与解决
        conflicts = self.collision_manager.detect_conflicts(self.drones, sim_dt)
        if conflicts:
            collision_events = self.collision_manager.resolve_conflicts(
                self.drones, conflicts, sim_dt
            )
            self.events.extend(collision_events)
        self._check_dynamic_actor_collisions()

        # 6. 事件检测
        self._check_events()

        # 7. 动态任务生成
        if self.dynamic_tasks_enabled:
            self._generate_dynamic_tasks()

        # 8. 检查仿真结束
        if self.time >= self.duration:
            self._finish()

        # 9. 状态推送
        self._state_push_counter += 1
        push_every_n_steps = int(SIMULATION["state_push_interval"] / self.dt)
        if self._state_push_counter >= push_every_n_steps:
            self._state_push_counter = 0
            return self.get_state_snapshot()

        return None

    def run_until(self, end_time: float) -> List[dict]:
        """运行仿真直到指定时间，返回所有状态快照"""
        snapshots = []
        while self.time < end_time and self.state == "running":
            snapshot = self.step()
            if snapshot:
                snapshots.append(snapshot)
        return snapshots

    def _check_dynamic_actor_collisions(self) -> None:
        """Register each actor contact once while bodies overlap."""
        for drone in self.drones:
            previous = self._dynamic_collision_active.setdefault(drone.id, set())
            current: set[int] = set()
            for actor in self.dynamic_actor_field.actors:
                margin = actor.half_extent + np.asarray([0.5, 0.35, 0.5])
                if np.all(np.abs(drone.position - actor.position) <= margin):
                    current.add(actor.actor_id)
                    if actor.actor_id not in previous:
                        self._static_collision_counts[drone.id] = self._static_collision_counts.get(drone.id, 0) + 1
            if current:
                drone.world_model_state["collision"] = True
                drone.world_model_state["collision_source"] = "scripted_dynamic_actor"
                drone.world_model_state["actual_collision_count"] = self._static_collision_counts.get(drone.id, 0)
            self._dynamic_collision_active[drone.id] = current

    # ==================================================================
    # 无人机动力学更新
    # ==================================================================

    def _update_drone_dynamics(self, drone: DroneStateData, dt: float):
        """用四电机 6-DOF 刚体模型推进无人机状态。"""
        waypoints = drone.path
        idx = drone.current_path_index
        target_wp = waypoints[idx] if idx < len(waypoints) else None
        low_altitude_tracking = bool(
            target_wp is not None
            and target_wp.metadata.get("low_altitude_3d", False)
        )
        low_altitude_ceiling = (
            target_wp.metadata.get("altitude_max_m")
            if low_altitude_tracking
            else None
        )
        low_altitude_floor = (
            target_wp.metadata.get("altitude_min_m")
            if low_altitude_tracking
            else None
        )

        if target_wp is None:
            target = drone.position.copy()
            desired_velocity = np.zeros(3)
            desired_yaw = drone.yaw
            if drone.state not in (DroneState.LANDED, DroneState.CHARGING):
                drone.state = DroneState.HOVERING
        else:
            target = target_wp.position
            direction = target - drone.position
            distance = float(np.linalg.norm(direction))

            if distance < 1.0:
                drone.current_path_index = idx + 1
                if target_wp.action == "pickup":
                    drone.state = DroneState.PICKING_UP
                    if drone.current_task_id:
                        task = self._find_task(drone.current_task_id)
                        if task:
                            drone.payload_current = task.payload_weight
                            self._emit_event(
                                EventType.PICKUP_COMPLETE,
                                drone.id,
                                drone.current_task_id,
                                f"取件完成: {task.id}",
                            )
                elif target_wp.action == "delivery":
                    drone.state = DroneState.DELIVERING
                    if drone.current_task_id:
                        task = self._find_task(drone.current_task_id)
                        if task:
                            task.status = TaskStatus.COMPLETED
                            drone.tasks_completed += 1
                            drone.payload_current = 0.0
                            self._emit_event(
                                EventType.DELIVERY_COMPLETE,
                                drone.id,
                                drone.current_task_id,
                                f"递送完成: {task.id}",
                            )
                            if drone.current_task_id in drone.assigned_tasks:
                                drone.assigned_tasks.remove(drone.current_task_id)
                            drone.current_task_id = None
                elif target_wp.action == "charge":
                    drone.state = DroneState.CHARGING
                    self._emit_event(
                        EventType.CHARGING_STARTED,
                        drone.id,
                        message="开始充电",
                    )
                return

            load_ratio = (
                drone.payload_current / drone.max_payload
                if drone.max_payload > 0
                else 0.0
            )
            effective_max_speed = drone.max_speed * (1.0 - 0.3 * load_ratio)
            if low_altitude_tracking:
                effective_max_speed = min(effective_max_speed, 7.0)
            position_gain = 0.65 if low_altitude_tracking else 1.2
            desired_velocity = (
                direction / max(distance, 1e-6)
                * min(effective_max_speed, distance * position_gain)
            )
            vertical_limit = (
                min(drone.max_climb_rate, 1.5)
                if low_altitude_tracking
                else drone.max_climb_rate
            )
            desired_velocity[1] = np.clip(
                desired_velocity[1],
                -vertical_limit,
                vertical_limit,
            )
            if low_altitude_tracking:
                ceiling = target_wp.metadata.get("altitude_max_m")
                floor = target_wp.metadata.get("altitude_min_m")
                vertical_velocity = float(drone.velocity[1])
                braking_acceleration = max(0.8, min(float(drone.max_accel), 1.8))
                guard_m = 0.75
                if ceiling is not None:
                    headroom = float(ceiling) - guard_m - float(drone.position[1])
                    allowed_up = math.sqrt(
                        max(0.0, 2.0 * braking_acceleration * max(headroom, 0.0))
                    )
                    desired_velocity[1] = min(desired_velocity[1], allowed_up)
                    if headroom < 2.5 or vertical_velocity > allowed_up:
                        ceiling_brake = 0.55 * headroom - 0.9 * max(vertical_velocity, 0.0)
                        desired_velocity[1] = min(desired_velocity[1], ceiling_brake)
                if floor is not None:
                    footroom = float(drone.position[1]) - (float(floor) + guard_m)
                    allowed_down = math.sqrt(
                        max(0.0, 2.0 * braking_acceleration * max(footroom, 0.0))
                    )
                    desired_velocity[1] = max(desired_velocity[1], -allowed_down)
                    if footroom < 2.5 or -vertical_velocity > allowed_down:
                        floor_brake = -0.55 * footroom - 0.9 * min(vertical_velocity, 0.0)
                        desired_velocity[1] = max(desired_velocity[1], floor_brake)
                desired_velocity[1] = float(
                    np.clip(desired_velocity[1], -vertical_limit, vertical_limit)
                )
            horizontal = np.hypot(direction[0], direction[2])
            desired_yaw = (
                np.degrees(np.arctan2(direction[2], direction[0]))
                if horizontal > 0.25
                else drone.yaw
            )

        model = self._multirotor_models.get(drone.id)
        if model is None:
            profile = MULTIROTOR_DYNAMICS.get(
                drone.drone_type,
                MULTIROTOR_DYNAMICS["standard"],
            )
            model = MultirotorDynamics(MultirotorParameters.from_dict(profile))
            model.initialize(drone.yaw)
            self._multirotor_models[drone.id] = model

        previous_position = drone.position.copy()
        wind_velocity = self.wind_model.get_wind(drone.position, self.time) + self._episode_wind_offset
        world_model = self._world_model_controllers.get(drone.id)
        if world_model is not None and target_wp is not None:
            try:
                decision = world_model.plan(
                    simulation_time_s=self.time,
                    position_world_m=drone.position,
                    velocity_world_mps=drone.velocity,
                    yaw_degrees=drone.yaw,
                    goal_world_m=target_wp.position,
                    dynamics_model=model,
                    wind_velocity=wind_velocity,
                    payload_mass=drone.payload_current,
                    max_acceleration=drone.max_accel,
                )
                command = np.asarray(
                    decision["command_world_mps"],
                    dtype=float,
                )
                desired_velocity = command
                target = np.asarray(
                    decision["local_target_world_m"],
                    dtype=float,
                )
                horizontal_command = np.hypot(command[0], command[2])
                if horizontal_command > 0.2:
                    desired_yaw = np.degrees(
                        np.arctan2(command[2], command[0])
                    )
                drone.world_model_state = decision
            except Exception as error:
                drone.world_model_state = {
                    "enabled": True,
                    "backend": world_model.backend_name,
                    "status": "degraded_to_geometric_path_tracker",
                    "error_type": type(error).__name__,
                }

        external_command = self._external_policy_commands.get(drone.id)
        if external_command is not None:
            if self.time <= external_command["valid_until_sim_time"]:
                raw_command = np.asarray(
                    external_command["command_world_mps"],
                    dtype=float,
                )
                executed_command = raw_command.copy()
                intervention_reasons = []
                lookahead_s = float(external_command.get("shield_lookahead_s", 0.8))
                if (
                    external_command.get("shield_enabled", True)
                    and self.static_collision_map is not None
                    and hasattr(self.static_collision_map, "sweep_collides")
                ):
                    predicted_position = (
                        drone.position + executed_command * lookahead_s
                    )
                    collides, clearance, _ = self.static_collision_map.sweep_collides(
                        drone.position,
                        predicted_position,
                        max(0.75, min(2.0, float(drone.safety_radius))),
                    )
                    if collides:
                        executed_command[0] = 0.0
                        executed_command[2] = 0.0
                        executed_command[1] = max(executed_command[1], 1.0)
                        intervention_reasons.append("static_collision_sweep")
                speed = float(np.linalg.norm(executed_command[[0, 2]]))
                if speed > 6.0:
                    executed_command[[0, 2]] *= 6.0 / speed
                    intervention_reasons.append("horizontal_speed_limit")
                clipped_vertical = float(np.clip(executed_command[1], -3.0, 3.0))
                if clipped_vertical != executed_command[1]:
                    intervention_reasons.append("vertical_speed_limit")
                    executed_command[1] = clipped_vertical
                desired_velocity = executed_command
                # External policies command velocity, not a hidden attraction
                # to the mission goal.  Keep the position loop local so the
                # 6-DOF controller tracks the commanded velocity without the
                # built-in path follower biasing the result.
                target = drone.position + executed_command * min(
                    lookahead_s, 0.35
                )
                yaw_rate_body_flu_degrees = float(
                    np.clip(external_command["yaw_rate_degrees_s"], -60.0, 60.0)
                )
                yaw_rate_backend_degrees = body_flu_yaw_rate_to_backend_degrees(
                    np.deg2rad(yaw_rate_body_flu_degrees)
                )
                desired_yaw = float(
                    external_command.get("desired_yaw_backend_degrees", drone.yaw)
                    + yaw_rate_backend_degrees * dt
                )
                external_command["desired_yaw_backend_degrees"] = desired_yaw
                if intervention_reasons:
                    self._external_policy_interventions[drone.id] = (
                        self._external_policy_interventions.get(drone.id, 0) + 1
                    )
                drone.world_model_state = {
                    "enabled": True,
                    "backend": external_command["policy_family"],
                    "status": "external_learned_policy",
                    "policy_step_id": external_command["step_id"],
                    "raw_action_normalized": external_command[
                        "raw_action_normalized"
                    ],
                    "raw_action_physical_body_flu": external_command[
                        "raw_action_physical_body_flu"
                    ],
                    "raw_command_world_mps": raw_command.tolist(),
                    "command_world_mps": executed_command.tolist(),
                    "yaw_rate_degrees_s": yaw_rate_body_flu_degrees,
                    "yaw_rate_backend_degrees_s": yaw_rate_backend_degrees,
                    "desired_yaw_backend_degrees": desired_yaw,
                    "executed_action_physical_body_flu": [
                        float(
                            np.cos(np.deg2rad(drone.yaw)) * executed_command[0]
                            + np.sin(np.deg2rad(drone.yaw)) * executed_command[2]
                        ),
                        float(
                            np.sin(np.deg2rad(drone.yaw)) * executed_command[0]
                            - np.cos(np.deg2rad(drone.yaw)) * executed_command[2]
                        ),
                        float(executed_command[1]),
                        float(np.deg2rad(yaw_rate_body_flu_degrees)),
                    ],
                    "safety_enabled": bool(
                        external_command.get("shield_enabled", True)
                    ),
                    "safety_intervened": bool(intervention_reasons),
                    "safety_intervention_reasons": intervention_reasons,
                    "safety_intervention_count": self._external_policy_interventions.get(
                        drone.id, 0
                    ),
                    "action_delta_l2": float(
                        np.linalg.norm(executed_command - raw_command)
                    ),
                    "inference_latency_ms": external_command[
                        "inference_latency_ms"
                    ],
                    "predicted_risk": external_command["predicted_risk"],
                    "stale_action": False,
                    "actual_collision_count": self._static_collision_counts.get(
                        drone.id, 0
                    ),
                }
            else:
                # A missing learned-policy command never falls back invisibly to
                # geometric MPC: the aircraft hovers and the telemetry says why.
                desired_velocity = np.zeros(3, dtype=float)
                target = drone.position.copy()
                desired_yaw = drone.yaw
                drone.world_model_state = {
                    "enabled": True,
                    "backend": external_command["policy_family"],
                    "status": "policy_timeout_hover",
                    "policy_step_id": external_command["step_id"],
                    "raw_action_normalized": external_command[
                        "raw_action_normalized"
                    ],
                    "raw_action_physical_body_flu": external_command[
                        "raw_action_physical_body_flu"
                    ],
                    "command_world_mps": [0.0, 0.0, 0.0],
                    "executed_action_physical_body_flu": [0.0, 0.0, 0.0, 0.0],
                    "safety_enabled": True,
                    "safety_intervened": True,
                    "safety_intervention_reasons": ["policy_command_timeout"],
                    "safety_intervention_count": self._external_policy_interventions.get(
                        drone.id, 0
                    ),
                    "inference_latency_ms": external_command[
                        "inference_latency_ms"
                    ],
                    "predicted_risk": external_command["predicted_risk"],
                    "stale_action": True,
                    "actual_collision_count": self._static_collision_counts.get(
                        drone.id, 0
                    ),
                }

        result = model.step(
            position=drone.position,
            velocity=drone.velocity,
            target_position=target,
            target_velocity=desired_velocity,
            desired_yaw_degrees=desired_yaw,
            wind_velocity=wind_velocity,
            payload_mass=drone.payload_current,
            max_acceleration=drone.max_accel,
            dt=dt,
        )

        if low_altitude_tracking:
            # The photogrammetry benchmark exposes a vertical instability in
            # the generic rigid-body tracker at low ceilings.  Apply the same
            # acceleration-limited velocity envelope to every low-altitude
            # task.  Horizontal motion and attitude remain the original 6-DOF
            # model; only the high-level vertical velocity channel is filtered.
            previous_vertical_velocity = float(drone.velocity[1])
            target_vertical_velocity = float(desired_velocity[1])
            vertical_acceleration_limit = min(float(drone.max_accel), 1.5)
            velocity_delta = float(
                np.clip(
                    target_vertical_velocity - previous_vertical_velocity,
                    -vertical_acceleration_limit * dt,
                    vertical_acceleration_limit * dt,
                )
            )
            filtered_vertical_velocity = float(
                np.clip(
                    previous_vertical_velocity + velocity_delta,
                    -1.5,
                    1.5,
                )
            )
            filtered_altitude = float(
                previous_position[1] + filtered_vertical_velocity * dt
            )
            altitude_intervened = False
            if low_altitude_ceiling is not None:
                maximum_altitude = float(low_altitude_ceiling) - 0.05
                if filtered_altitude > maximum_altitude:
                    filtered_altitude = maximum_altitude
                    filtered_vertical_velocity = min(0.0, filtered_vertical_velocity)
                    altitude_intervened = True
            if low_altitude_floor is not None:
                minimum_altitude = float(low_altitude_floor) + 0.05
                if filtered_altitude < minimum_altitude:
                    filtered_altitude = minimum_altitude
                    filtered_vertical_velocity = max(0.0, filtered_vertical_velocity)
                    altitude_intervened = True
            result["position"][1] = filtered_altitude
            result["velocity"][1] = filtered_vertical_velocity
            result["acceleration"][1] = (
                filtered_vertical_velocity - previous_vertical_velocity
            ) / max(dt, 1e-6)
            if altitude_intervened:
                drone.world_model_state = {
                    **drone.world_model_state,
                    "altitude_safety_intervened": True,
                }

        drone.position = result["position"]
        drone.velocity = result["velocity"]
        drone.acceleration = result["acceleration"]
        drone.orientation_quaternion = result["orientation"]
        drone.angular_velocity = result["angular_velocity"]
        drone.motor_omega = result["motor_omega"]
        drone.motor_thrusts = result["motor_thrusts"]
        drone.total_thrust = float(result["total_thrust"])
        drone.power_w = float(result["power_w"])
        drone.roll = float(result["roll"])
        drone.pitch = float(result["pitch"])
        drone.yaw = float(result["yaw"])

        # 最低地面约束。高精城市表面的最终碰撞仍由下方静态场扫掠负责。
        if drone.position[1] < 0.45:
            drone.position[1] = 0.45
            drone.velocity[1] = max(0.0, drone.velocity[1])
            model.handle_collision(np.array([0.0, 1.0, 0.0]))

        is_static_escape = (
            target_wp is not None
            and target_wp.metadata.get("avoidance")
            in {"citygs_local_collision", "citygs_hierarchical_collision"}
        )
        if (
            self.static_collision_map is not None
            and drone.position[1] > 4.0
            and not is_static_escape
        ):
            static_radius = max(0.75, min(2.0, float(drone.safety_radius)))
            if hasattr(self.static_collision_map, "sweep_collides"):
                collides, clearance, collision_position = (
                    self.static_collision_map.sweep_collides(
                        previous_position,
                        drone.position,
                        static_radius,
                    )
                )
            else:
                collides, clearance = self.static_collision_map.collides(
                    drone.position,
                    static_radius,
                )
                collision_position = drone.position.copy() if collides else None

            if collides:
                self._static_collision_counts[drone.id] = (
                    self._static_collision_counts.get(drone.id, 0) + 1
                )
                drone.world_model_state["actual_collision_count"] = (
                    self._static_collision_counts[drone.id]
                )
                drone.world_model_state["collision"] = True
                drone.position = previous_position
                drone.velocity *= 0.12
                drone.velocity[1] = max(drone.velocity[1], 1.0)
                model.handle_collision()

                safe_altitude = previous_position[1] + max(
                    4.0,
                    static_radius * 2.0,
                )
                if self.planner is not None and self.planner.grid is not None:
                    gx, gz = self.planner.grid.world_to_grid_xz(previous_position)
                    safe_altitude = max(
                        safe_altitude,
                        self.planner.grid.get_safe_altitude(
                            gx,
                            gz,
                            static_radius + 3.0,
                        ),
                    )
                escape = previous_position.copy()
                escape[1] = safe_altitude
                drone.path.insert(
                    drone.current_path_index,
                    Waypoint(
                        position=escape,
                        action="hover",
                        metadata={
                            "avoidance": "citygs_hierarchical_collision",
                            "resolution_m": self.static_collision_map.resolution,
                            "collision_position": (
                                collision_position.tolist()
                                if collision_position is not None
                                else None
                            ),
                        },
                    ),
                )
                self.collision_manager.collisions_avoided += 1
                last_event = self._last_static_collision_event.get(
                    drone.id,
                    -float("inf"),
                )
                if self.time - last_event >= 1.0:
                    self._emit_event(
                        EventType.COLLISION_AVOIDED,
                        drone.id,
                        message=(
                            f"刚体扫掠碰撞规避；最近净空 {clearance:.2f} m"
                        ),
                    )
                    self._last_static_collision_event[drone.id] = self.time

        drone.total_distance_traveled += float(
            np.linalg.norm(drone.position - previous_position)
        )
        energy_used = drone.power_w * dt / 3600.0
        drone.battery_remaining -= energy_used
        drone.total_energy_consumed += energy_used

        if drone.state == DroneState.LANDED and target_wp is not None:
            drone.state = DroneState.TAKEOFF
        elif (
            target_wp is not None
            and drone.state
            not in (
                DroneState.PICKING_UP,
                DroneState.DELIVERING,
                DroneState.CHARGING,
                DroneState.RETURNING,
            )
        ):
            drone.state = DroneState.EN_ROUTE

        if drone.is_battery_critical and drone.state != DroneState.RETURNING:
            drone.state = DroneState.RETURNING
            self._emit_event(
                EventType.BATTERY_CRITICAL,
                drone.id,
                message=f"电量极低 ({drone.battery_pct:.1%})，强制返回",
            )
        elif drone.is_battery_low and drone.state != DroneState.RETURNING:
            self._emit_event(
                EventType.BATTERY_LOW,
                drone.id,
                message=f"电量低 ({drone.battery_pct:.1%})",
            )

    def _update_drone_legacy_kinematics(self, drone: DroneStateData, dt: float):
        """
        更新一架无人机的运动学。

        运动学模型：
        - 3DOF平移 + 偏航
        - 加速度限制
        - 爬升速率限制
        - 风扰动
        - 负载对速度的影响
        - 电池消耗
        """
        # 获取下一个路径点
        waypoints = drone.path
        idx = drone.current_path_index

        if idx >= len(waypoints):
            # 无更多路径点 → 悬停
            drone.state = DroneState.HOVERING
            drone.velocity *= 0.9  # 减速
            return

        target_wp = waypoints[idx]
        target = target_wp.position

        # 方向向量
        direction = target - drone.position
        dist = np.linalg.norm(direction)

        # 到达路径点？
        if dist < 1.0:
            drone.current_path_index = idx + 1

            # 处理路径点动作
            if target_wp.action == "pickup":
                drone.state = DroneState.PICKING_UP
                if drone.current_task_id:
                    task = self._find_task(drone.current_task_id)
                    if task:
                        drone.payload_current = task.payload_weight
                        self._emit_event(
                            EventType.PICKUP_COMPLETE, drone.id, drone.current_task_id,
                            f"取件完成: {task.id}"
                        )
            elif target_wp.action == "delivery":
                drone.state = DroneState.DELIVERING
                if drone.current_task_id:
                    task = self._find_task(drone.current_task_id)
                    if task:
                        task.status = TaskStatus.COMPLETED
                        drone.tasks_completed += 1
                        drone.payload_current = 0.0
                        self._emit_event(
                            EventType.DELIVERY_COMPLETE, drone.id, drone.current_task_id,
                            f"递送完成: {task.id}"
                        )
                        # 从assigned_tasks移除
                        if drone.current_task_id in drone.assigned_tasks:
                            drone.assigned_tasks.remove(drone.current_task_id)
                        drone.current_task_id = None
            elif target_wp.action == "charge":
                drone.state = DroneState.CHARGING
                self._emit_event(
                    EventType.CHARGING_STARTED, drone.id,
                    message=f"开始充电"
                )

            return

        # 有效最大速度（受负载影响）
        load_ratio = drone.payload_current / drone.max_payload if drone.max_payload > 0 else 0
        effective_max_speed = drone.max_speed * (1 - 0.3 * load_ratio)

        # 期望速度
        desired_vel = direction / dist * min(effective_max_speed, dist * 2.0)

        # 加速度限制
        accel_desired = (desired_vel - drone.velocity) / dt
        accel_mag = np.linalg.norm(accel_desired)
        if accel_mag > drone.max_accel:
            accel_desired = accel_desired / accel_mag * drone.max_accel

        # 风扰动
        wind = self.wind_model.get_wind(drone.position, self.time)
        accel_desired += wind * 0.1

        # 爬升速率限制
        climb_accel = accel_desired[1]
        max_climb_accel = drone.max_climb_rate / dt
        if abs(climb_accel) > max_climb_accel:
            accel_desired[1] = np.sign(climb_accel) * max_climb_accel

        # 积分更新
        drone.acceleration = accel_desired
        drone.velocity += accel_desired * dt
        prev_pos = drone.position.copy()
        drone.position += drone.velocity * dt

        # The 1 m signed field plans globally; the 0.25 m CityGS surface layer
        # is the final gate for facade detail, roof edges and vegetation.
        # Sweep the complete integration segment so a fast step cannot tunnel
        # through a thin obstacle between its two endpoint samples.
        is_static_escape = (
            target_wp.metadata.get("avoidance")
            in {"citygs_local_collision", "citygs_hierarchical_collision"}
        )
        if (
            self.static_collision_map is not None
            and drone.position[1] > 4.0
            and not is_static_escape
        ):
            static_radius = max(0.75, min(2.0, float(drone.safety_radius)))
            if hasattr(self.static_collision_map, "sweep_collides"):
                collides, clearance, collision_position = (
                    self.static_collision_map.sweep_collides(
                        prev_pos,
                        drone.position,
                        static_radius,
                    )
                )
            else:
                collides, clearance = self.static_collision_map.collides(
                    drone.position, static_radius
                )
                collision_position = drone.position.copy() if collides else None
            if collides:
                drone.position = prev_pos
                drone.velocity *= 0.15
                drone.velocity[1] = max(
                    drone.velocity[1], min(drone.max_climb_rate, 2.0)
                )

                safe_altitude = prev_pos[1] + max(4.0, static_radius * 2.0)
                if self.planner is not None and self.planner.grid is not None:
                    gx, gz = self.planner.grid.world_to_grid_xz(prev_pos)
                    safe_altitude = max(
                        safe_altitude,
                        self.planner.grid.get_safe_altitude(
                            gx, gz, static_radius + 3.0
                        ),
                    )
                escape = prev_pos.copy()
                escape[1] = safe_altitude
                drone.path.insert(
                    drone.current_path_index,
                    Waypoint(
                        position=escape,
                        action="hover",
                        metadata={
                            "avoidance": "citygs_hierarchical_collision",
                            "resolution_m": self.static_collision_map.resolution,
                            "collision_position": (
                                collision_position.tolist()
                                if collision_position is not None
                                else None
                            ),
                        },
                    ),
                )
                self.collision_manager.collisions_avoided += 1

                last_event = self._last_static_collision_event.get(
                    drone.id, -float("inf")
                )
                if self.time - last_event >= 1.0:
                    self._emit_event(
                        EventType.COLLISION_AVOIDED,
                        drone.id,
                        message=(
                            f"1 m ESDF + 0.25 m 细节层扫掠规避；"
                            f"最近净空 {clearance:.2f} m"
                        ),
                    )
                    self._last_static_collision_event[drone.id] = self.time

        # 累计距离
        drone.total_distance_traveled += np.linalg.norm(drone.position - prev_pos)

        # 偏航朝向运动方向
        horiz_speed = np.linalg.norm([drone.velocity[0], drone.velocity[2]])
        if horiz_speed > 0.5:
            target_yaw = np.degrees(np.arctan2(drone.velocity[2], drone.velocity[0]))
            yaw_diff = (target_yaw - drone.yaw + 180) % 360 - 180
            max_step = drone.max_yaw_rate * dt
            drone.yaw += np.clip(yaw_diff, -max_step, max_step)

        # 电池消耗
        speed = np.linalg.norm(drone.velocity)
        energy_used = (
            drone.energy_per_meter * speed
            + drone.energy_per_kg_meter * drone.payload_current * speed
        ) * dt
        drone.battery_remaining -= energy_used
        drone.total_energy_consumed += energy_used

        # 更新状态
        if drone.state == DroneState.LANDED:
            drone.state = DroneState.TAKEOFF
        elif drone.state not in (DroneState.PICKING_UP, DroneState.DELIVERING,
                                  DroneState.CHARGING, DroneState.RETURNING):
            drone.state = DroneState.EN_ROUTE

        # 电池检查
        if drone.is_battery_critical and drone.state != DroneState.RETURNING:
            drone.state = DroneState.RETURNING
            self._emit_event(
                EventType.BATTERY_CRITICAL, drone.id,
                message=f"电量极低 ({drone.battery_pct:.1%})，强制返回"
            )
        elif drone.is_battery_low and drone.state != DroneState.RETURNING:
            self._emit_event(
                EventType.BATTERY_LOW, drone.id,
                message=f"电量低 ({drone.battery_pct:.1%})"
            )

    # ==================================================================
    # 任务分配 & 路径规划
    # ==================================================================

    def _reallocate_tasks(self):
        """运行任务分配算法"""
        t_start = _time.perf_counter()
        comm_graph = self.comm_model._adj_matrix

        pending = [t for t in self.tasks if t.status in (TaskStatus.PENDING,)]

        if not pending:
            return

        assignments = self.allocator.run_iteration(
            self.drones, self.tasks, comm_graph, self.time
        )

        self._alloc_runtimes.append((_time.perf_counter() - t_start) * 1000)

        # 更新无人机分配
        for drone in self.drones:
            if drone.id in assignments:
                drone.assigned_tasks = assignments[drone.id]
                if drone.assigned_tasks and not drone.current_task_id:
                    drone.current_task_id = drone.assigned_tasks[0]

        # 更新任务状态
        for drone in self.drones:
            for task_id in drone.assigned_tasks:
                task = self._find_task(task_id)
                if task and task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_to = drone.id
                    self._emit_event(
                        EventType.TASK_ASSIGNED, drone.id, task_id,
                        f"任务分配: {task_id} → {drone.id}"
                    )

    def _plan_new_paths(self):
        """为需要新路径的无人机规划路径"""
        if self.planner is None:
            return

        for drone in self.drones:
            if drone.assigned_tasks and not drone.path:
                t_start = _time.perf_counter()

                bundle = [self._find_task(tid) for tid in drone.assigned_tasks]
                bundle = [t for t in bundle if t is not None]

                if bundle:
                    path, dist = self.planner.plan_bundle_path(drone, bundle)
                    drone.path = self._validate_and_repair_static_path(drone, path)
                    drone.current_path_index = 0

                self._plan_runtimes.append((_time.perf_counter() - t_start) * 1000)

    def _apply_scripted_paths(self, scenario_def):
        """装载场景定义中的确定性试飞航线，并用实景碰撞场逐段复核。"""
        configs_by_id = {
            cfg.get("id"): cfg
            for cfg in getattr(scenario_def, "drones", [])
            if cfg.get("id")
        }
        for drone in self.drones:
            cfg = configs_by_id.get(drone.id, {})
            scripted_path = cfg.get("scripted_path")
            if not scripted_path:
                continue

            origin = np.asarray(drone.position, dtype=float).copy()
            relative = bool(cfg.get("scripted_path_relative", False))
            waypoints = []
            for index, item in enumerate(scripted_path):
                if isinstance(item, dict):
                    raw_position = item.get("offset" if relative else "position")
                    if raw_position is None:
                        raw_position = item.get("position", item.get("offset"))
                    action = item.get("action")
                    phase = item.get("phase", f"航段 {index + 1}")
                else:
                    raw_position = item
                    action = None
                    phase = f"航段 {index + 1}"

                if raw_position is None:
                    continue
                position = np.asarray(raw_position, dtype=float)
                if relative:
                    position = origin + position
                if self.static_collision_map is not None and hasattr(
                    self.static_collision_map,
                    "surface_height",
                ):
                    surface = self.static_collision_map.surface_height(
                        position,
                        drone.safety_radius,
                    )
                    if np.isfinite(surface):
                        position[1] = max(
                            position[1],
                            float(surface) + drone.safety_radius + 3.0,
                        )
                waypoints.append(
                    Waypoint(
                        position=position,
                        action=action,
                        metadata={
                            "scripted": True,
                            "phase": phase,
                            "sequence": index,
                        },
                    )
                )

            world_model_enabled = bool(
                cfg.get("world_model", {}).get("enabled", False)
            )
            if world_model_enabled:
                for waypoint in waypoints:
                    waypoint.metadata["local_planner"] = (
                        "city_belief_multirotor_mpc_v1"
                    )
                # These are mission-level goals, not a pre-cleared trajectory.
                # Local obstacle avoidance belongs to the receding-horizon
                # world model; the independent collision shield remains active.
                drone.path = waypoints
            else:
                drone.path = self._validate_and_repair_static_path(
                    drone,
                    waypoints,
                )
            drone.current_path_index = 0

    def _configure_world_models(self, scenario_def):
        """Instantiate per-aircraft world-model controllers requested by a scenario."""
        self._world_model_controllers = {}
        configs_by_id = {
            cfg.get("id"): cfg
            for cfg in getattr(scenario_def, "drones", [])
            if cfg.get("id")
        }
        for drone in self.drones:
            cfg = configs_by_id.get(drone.id, {})
            world_model_cfg = cfg.get("world_model", {})
            if not world_model_cfg.get("enabled", False):
                drone.world_model_state = {}
                continue
            controller = UrbanWorldModelMPC(
                self.static_collision_map,
                UrbanWorldModelConfig.from_dict(world_model_cfg),
            )
            self._world_model_controllers[drone.id] = controller
            drone.world_model_state = {
                "enabled": True,
                "backend": controller.backend_name,
                "status": "initializing_belief",
                "learned_backend": (
                    "disabled_unvalidated_source_to_city_domain"
                ),
            }

    def _validate_and_repair_static_path(
        self,
        drone: DroneStateData,
        path: List[Waypoint],
    ) -> List[Waypoint]:
        """Audit complete route segments and lift any penetrating segment.

        This is intentionally a geometric/kinematic repair layer. It does not
        pretend to be a flight controller: inserted climb and traverse points
        will later be followed by the existing bounded-acceleration kinematics.
        """
        collision_map = self.static_collision_map
        if not path or collision_map is None or not hasattr(collision_map, "sweep_collides"):
            return path

        radius = max(0.75, min(2.0, float(drone.safety_radius)))
        positions = np.vstack(
            [np.asarray(drone.position, dtype=float)]
            + [np.asarray(waypoint.position, dtype=float) for waypoint in path]
        )
        audit = collision_map.audit_polyline(positions, radius)
        if audit["valid"]:
            for waypoint in path:
                waypoint.metadata["static_clearance_validated"] = True
            return path

        repaired: List[Waypoint] = []
        cursor = np.asarray(drone.position, dtype=float).copy()
        repaired_segments = 0
        for waypoint in path:
            target = np.asarray(waypoint.position, dtype=float).copy()
            collides, _, _ = collision_map.sweep_collides(cursor, target, radius)
            if collides:
                safe_altitude = self._safe_overflight_altitude(
                    cursor,
                    target,
                    radius,
                )
                climb = cursor.copy()
                climb[1] = safe_altitude
                traverse = target.copy()
                traverse[1] = safe_altitude
                avoidance_metadata = {
                    "avoidance": "citygs_path_audit",
                    "static_clearance_validated": True,
                    "global_resolution_m": getattr(
                        collision_map, "global_resolution", None
                    ),
                    "local_resolution_m": collision_map.resolution,
                }
                if np.linalg.norm(climb - cursor) > 0.5:
                    repaired.append(
                        Waypoint(
                            position=climb,
                            action="hover",
                            metadata=dict(avoidance_metadata),
                        )
                    )
                if np.linalg.norm(traverse - climb) > 0.5:
                    repaired.append(
                        Waypoint(
                            position=traverse,
                            metadata=dict(avoidance_metadata),
                        )
                    )
                repaired_segments += 1

            waypoint.metadata["static_clearance_validated"] = True
            repaired.append(waypoint)
            cursor = target

        repaired_positions = np.vstack(
            [np.asarray(drone.position, dtype=float)]
            + [np.asarray(waypoint.position, dtype=float) for waypoint in repaired]
        )
        final_audit = collision_map.audit_polyline(repaired_positions, radius)
        for waypoint in repaired:
            waypoint.metadata["route_minimum_clearance_m"] = float(
                final_audit["minimum_clearance_m"]
            )
            waypoint.metadata["static_clearance_validated"] = bool(
                final_audit["valid"]
            )

        self.collision_manager.path_replans_due_to_collision += repaired_segments
        self._emit_event(
            EventType.COLLISION_AVOIDED,
            drone.id,
            message=(
                f"路径穿透校验修复 {repaired_segments} 段；"
                f"复核最小净空 {final_audit['minimum_clearance_m']:.2f} m"
            ),
        )
        return repaired

    def _safe_overflight_altitude(
        self,
        start: np.ndarray,
        end: np.ndarray,
        safety_radius: float,
    ) -> float:
        safe_altitude = max(float(start[1]), float(end[1])) + safety_radius + 4.0
        if self.planner is not None and self.planner.grid is not None:
            horizontal_distance = float(
                np.linalg.norm((np.asarray(end) - np.asarray(start))[[0, 2]])
            )
            sample_count = max(
                2,
                int(
                    np.ceil(
                        horizontal_distance
                        / max(self.planner.grid.resolution, 1.0)
                    )
                ),
            )
            for alpha in np.linspace(0.0, 1.0, sample_count + 1):
                point = np.asarray(start) + alpha * (
                    np.asarray(end) - np.asarray(start)
                )
                gx, gz = self.planner.grid.world_to_grid_xz(point)
                safe_altitude = max(
                    safe_altitude,
                    self.planner.grid.get_safe_altitude(
                        gx,
                        gz,
                        safety_radius + 5.0,
                    ),
                )

        collision_map = self.static_collision_map
        if collision_map is not None and hasattr(collision_map, "surface_height"):
            horizontal_distance = float(
                np.linalg.norm((np.asarray(end) - np.asarray(start))[[0, 2]])
            )
            sample_count = max(2, int(np.ceil(horizontal_distance / 2.0)))
            for alpha in np.linspace(0.0, 1.0, sample_count + 1):
                point = np.asarray(start) + alpha * (
                    np.asarray(end) - np.asarray(start)
                )
                surface = collision_map.surface_height(point, safety_radius)
                if np.isfinite(surface):
                    safe_altitude = max(
                        safe_altitude,
                        float(surface) + safety_radius + 3.0,
                    )

        global_esdf = getattr(self.static_collision_map, "global_esdf", None)
        if global_esdf is not None:
            mapped_ceiling = (
                global_esdf.origin[1]
                + (global_esdf.shape[1] - 1.5) * global_esdf.resolution
            )
            safe_altitude = min(safe_altitude, float(mapped_ceiling))
        return safe_altitude

    # ==================================================================
    # 事件系统
    # ==================================================================

    def _check_events(self):
        """检查各类事件触发条件"""
        for drone in self.drones:
            # 任务超时
            if drone.current_task_id:
                task = self._find_task(drone.current_task_id)
                if task and task.is_expired(self.time) and task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.FAILED
                    self._emit_event(
                        EventType.TASK_FAILED, drone.id, task.id,
                        f"任务超时: {task.id}"
                    )

    def _generate_dynamic_tasks(self):
        """动态生成新任务"""
        # 紧急任务
        interval = TASK_GENERATION["dynamic_emergency_interval"]
        if self.time - self._last_emergency_gen >= interval:
            self._last_emergency_gen = self.time
            task = self._generate_single_task("emergency_medical")
            self.tasks.append(task)
            self.pending_tasks.append(task)

            self._emit_event(
                EventType.NEW_TASK_GENERATED, task_id=task.id,
                message=f"动态紧急任务: {task.id}"
            )

        # 常规任务批次
        batch_interval = TASK_GENERATION["dynamic_regular_batch_interval"]
        if self.time - self._last_regular_batch >= batch_interval:
            self._last_regular_batch = self.time
            for _ in range(TASK_GENERATION["dynamic_regular_batch_size"]):
                task = self._generate_single_task("regular")
                self.tasks.append(task)
                self.pending_tasks.append(task)

    def _next_task_id(self) -> str:
        self._task_id_counter += 1
        return f"T-{self._task_id_counter:03d}"

    def _sample_task_template(self, task_type: str) -> dict:
        import random
        templates = TASK_TEMPLATE_LIBRARY.get(task_type, [])
        if not templates:
            return {}
        return random.choice(templates)

    def _generate_single_task(self, task_type: str) -> Task:
        """生成单个复杂任务：分区感知 + 业务属性 + 兼容性约束"""
        import random
        type_cfg = TASK_TYPES[task_type]
        template = self._sample_task_template(task_type)

        pickup_block = self._sample_block_by_district(template.get("pickup_districts", []))
        delivery_block = self._sample_block_by_district(template.get("delivery_districts", []))

        if pickup_block is None:
            pickup = self._sample_task_position()
            pickup_district = "mixed"
            pickup_block_id = None
        else:
            pickup = self._sample_task_position_for_block(pickup_block, purpose="pickup")
            pickup_district = pickup_block.district
            pickup_block_id = pickup_block.id

        if delivery_block is None:
            delivery = self._sample_task_position()
            delivery_district = "mixed"
            delivery_block_id = None
        else:
            patrol_mode = task_type == "patrol"
            delivery = self._sample_task_position_for_block(delivery_block, purpose="delivery", patrol_mode=patrol_mode)
            delivery_district = delivery_block.district
            delivery_block_id = delivery_block.id

        service_range = template.get("service_time_range", TASK_GENERATION["service_time_range"])
        pickup_service = random.uniform(*service_range)
        delivery_service = random.uniform(*service_range)
        risk_low, risk_high = template.get("risk_range", TASK_GENERATION["risk_range"])
        risk_level = random.uniform(risk_low, risk_high)
        cold_chain = random.random() < template.get("cold_chain_probability", 0.0)
        fragile = random.random() < template.get("fragile_probability", 0.0)
        preferred_drone_types = template.get("preferred_drone_types")
        min_neighbor_count = template.get("min_neighbor_count", 0)
        aging_weight = template.get("aging_weight", 1.0)

        # 业务标签：优先使用模板，否则根据分区规则推断
        if template.get("business_tag"):
            business_tag = template["business_tag"]
        else:
            district_rule = DISTRICT_TASK_RULES.get(delivery_district) or DISTRICT_TASK_RULES.get(pickup_district, {})
            business_tag = random.choice(district_rule.get("business_tags", ["generic"]))

        payload_weight = random.uniform(*type_cfg["payload_range"])
        if preferred_drone_types == ["light", "standard"]:
            payload_weight = min(payload_weight, 5.0)
        if task_type == "regular" and business_tag == "warehouse_linehaul":
            payload_weight = min(max(payload_weight, 4.0), type_cfg["payload_range"][1])

        task_group = f"{pickup_district}->{delivery_district}:{business_tag}"
        reward = type_cfg["reward"]
        reward *= 1.0 + 0.20 * risk_level
        reward *= 1.0 + 0.15 * float(cold_chain)
        reward *= 1.0 + 0.10 * float(fragile)

        patrol_waypoints = None
        if task_type == "patrol":
            patrol_waypoints = self._make_patrol_waypoints_for_block(delivery_block or pickup_block, type_cfg.get("patrol_waypoints", 5))
            if patrol_waypoints:
                pickup = patrol_waypoints[0].copy()
                delivery = patrol_waypoints[-1].copy()

        # 更复杂的电量下界：高风险/冷链任务要求更高返航余量
        min_required_battery_pct = TASK_GENERATION["default_min_battery_pct"] + 0.05 * float(cold_chain) + 0.08 * risk_level

        return Task(
            id=self._next_task_id(),
            task_type=task_type,
            priority=type_cfg["priority"],
            pickup_pos=pickup,
            delivery_pos=delivery,
            time_window=(self.time + type_cfg["time_window"][0], self.time + type_cfg["time_window"][1]),
            payload_weight=payload_weight,
            reward=reward,
            deadline_penalty=type_cfg["deadline_penalty"],
            required_comms=type_cfg["required_comms"],
            created_at=self.time,
            business_tag=business_tag,
            pickup_block_id=pickup_block_id,
            delivery_block_id=delivery_block_id,
            pickup_district=pickup_district,
            delivery_district=delivery_district,
            pickup_service_time=pickup_service,
            delivery_service_time=delivery_service,
            risk_level=risk_level,
            cold_chain=cold_chain,
            fragile=fragile,
            min_required_battery_pct=min_required_battery_pct,
            min_neighbor_count=min_neighbor_count,
            preferred_drone_types=preferred_drone_types,
            airspace_level="L2_mid_level" if task_type in ("emergency_medical", "medical") else "L1_street_canyon",
            aging_weight=aging_weight,
            task_group=task_group,
            patrol_waypoints=patrol_waypoints,
        )

    def _emit_event(self, event_type: EventType, drone_id: str = None,
                    task_id: str = None, message: str = ""):
        """发射仿真事件"""
        event = SimulationEvent(
            time=self.time,
            event_type=event_type,
            drone_id=drone_id,
            task_id=task_id,
            message=message,
        )
        self.events.append(event)

    # ==================================================================
    # 状态获取
    # ==================================================================

    def get_state_snapshot(self) -> dict:
        """获取当前仿真状态快照（用于前端渲染）"""
        drone_snapshots = []
        for drone in self.drones:
            snapshot = drone.to_dict()
            world_model = dict(snapshot.get("world_model") or {})
            visualization = self._external_policy_visualizations.get(drone.id)
            if visualization:
                world_model.update(visualization)
            snapshot["world_model"] = world_model
            drone_snapshots.append(snapshot)
        return {
            "t": round(self.time, 2),
            "state": self.state,
            "speed_multiplier": self.speed_multiplier,
            "scene_meta": self.scene_config.to_dict() if self.scene_config else {},
            "drones": drone_snapshots,
            "actors": self.dynamic_actor_field.snapshot(),
            "tasks": [t.to_dict() for t in self.tasks],
            "comm_graph": self.comm_model._adj_matrix.tolist() if self.comm_model._adj_matrix is not None else [],
            "events": [e.to_dict() for e in self.events[-50:]],
            "stats": self.stats.to_dict(),
            "alloc_stats": self.allocator.get_stats() if self.allocator else {},
            "semantic_agent": self.semantic_fleet_bridge.snapshot(),
            "topology_stats": self.comm_model.get_topology_stats(self.drones),
            "sensor_config": CAMERA_SENSORS,
            "appearance_perturbation": dict(self._episode_appearance_perturbation),
            "dynamics_perturbation": dict(self._episode_dynamics_perturbation),
            "physics": {
                "engine": "urbanfly_fast_multirotor_6dof",
                "rate_hz": round(1.0 / self.dt, 1),
                "motor_model": "first_order_thrust_mixer",
                "collision": "swept_static_field",
            },
        }

    def _compute_stats(self) -> SimulationStats:
        """计算当前统计信息"""
        stats = SimulationStats()

        stats.total_tasks = len(self.tasks)
        stats.tasks_completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        stats.tasks_in_progress = sum(1 for t in self.tasks if t.status in (TaskStatus.ASSIGNED, TaskStatus.EN_ROUTE_PICKUP, TaskStatus.EN_ROUTE_DELIVERY))
        stats.tasks_pending = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)
        stats.tasks_failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)

        # 按时完成率
        completed_on_time = sum(
            1 for t in self.tasks
            if t.status == TaskStatus.COMPLETED and t.time_window[1] > 0
        )
        total_with_deadline = sum(1 for t in self.tasks if t.time_window[1] > 0)
        stats.on_time_completion_rate = completed_on_time / total_with_deadline if total_with_deadline > 0 else 1.0

        # 平均响应时间
        assigned_tasks = [t for t in self.tasks if t.assigned_to]
        if assigned_tasks:
            stats.avg_response_time = np.mean([t.created_at for t in assigned_tasks])

        # 总距离和能耗
        stats.total_distance = sum(d.total_distance_traveled for d in self.drones)
        stats.total_energy = sum(d.total_energy_consumed for d in self.drones)

        # 电池
        stats.avg_battery_at_end = np.mean([d.battery_pct for d in self.drones])

        # 碰撞
        stats.collision_warnings = self.collision_manager.warnings_issued
        stats.path_replans = self.collision_manager.path_replans_due_to_collision
        stats.comm_disconnections = self.comm_model.get_disconnection_count()

        # 算法耗时
        stats.algorithm_runtime_ms = self.allocator.last_runtime_ms if self.allocator else 0

        # 按类型/优先级统计
        for t in self.tasks:
            stats.tasks_by_type[t.task_type] = stats.tasks_by_type.get(t.task_type, 0) + 1
            stats.tasks_by_priority[t.priority] = stats.tasks_by_priority.get(t.priority, 0) + 1
            stats.tasks_by_business[t.business_tag] = stats.tasks_by_business.get(t.business_tag, 0) + 1

        if self.tasks:
            stats.avg_task_risk = float(np.mean([t.risk_level for t in self.tasks]))
            stats.cold_chain_tasks = sum(1 for t in self.tasks if t.cold_chain)
            stats.fragile_tasks = sum(1 for t in self.tasks if t.fragile)

        return stats

    # ==================================================================
    # 控制接口
    # ==================================================================

    def play(self):
        self.state = "running"

    def pause(self):
        self.state = "paused"

    def stop(self):
        self.state = "stopped"

    def set_speed(self, multiplier: float):
        self.speed_multiplier = max(0.1, min(20.0, multiplier))

    def set_external_policy_action(
        self,
        drone_id: str,
        action_normalized,
        *,
        step_id: int,
        policy_family: str,
        inference_latency_ms: float = 0.0,
        predicted_risk: float = 0.0,
        shield_enabled: bool = True,
        timeout_s: float = 0.45,
    ) -> dict:
        """Publish one 5 Hz body-FLU command for the 50 Hz controller."""

        drone = next((item for item in self.drones if item.id == drone_id), None)
        if drone is None:
            raise ValueError(f"unknown drone: {drone_id}")
        action = np.asarray(action_normalized, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action_normalized must be a finite vector of length 4")
        action = np.clip(action, -1.0, 1.0)
        previous = self._external_policy_commands.get(drone_id)
        if previous is not None and int(step_id) <= int(previous["step_id"]):
            raise ValueError("policy step_id must increase strictly")
        physical = action * np.asarray(
            [6.0, 6.0, 3.0, 60.0],
            dtype=float,
        )
        yaw_rad = np.deg2rad(drone.yaw)
        forward = np.asarray([np.cos(yaw_rad), 0.0, np.sin(yaw_rad)])
        # Backend world is [east, up, south]. Geographic/body left is
        # negative renderer Z at yaw zero; the previous sign was FRU while
        # the API and stored action metadata called it FLU.
        left = np.asarray([np.sin(yaw_rad), 0.0, -np.cos(yaw_rad)])
        command_world = (
            forward * physical[0]
            + left * physical[1]
            + np.asarray([0.0, physical[2], 0.0])
        )
        command = {
            "drone_id": drone_id,
            "step_id": int(step_id),
            "accepted_sim_time": float(self.time),
            "policy_family": str(policy_family),
            "raw_action_normalized": action.tolist(),
            "raw_action_physical_body_flu": physical.tolist(),
            "command_world_mps": command_world.tolist(),
            "yaw_rate_degrees_s": float(physical[3]),
            "desired_yaw_backend_degrees": float(
                previous.get("desired_yaw_backend_degrees", drone.yaw)
                if previous is not None
                and self.time <= float(previous["valid_until_sim_time"])
                else drone.yaw
            ),
            "inference_latency_ms": max(0.0, float(inference_latency_ms)),
            "predicted_risk": float(np.clip(predicted_risk, 0.0, 1.0)),
            "shield_enabled": bool(shield_enabled),
            "valid_until_sim_time": self.time + float(
                np.clip(timeout_s, 0.2, 1.0)
            ),
            "shield_lookahead_s": 0.8,
        }
        self._external_policy_commands[drone_id] = command
        return dict(command)

    def set_external_policy_visualization(
        self,
        drone_id: str,
        visualization: dict,
    ) -> dict:
        """Store display-only planner telemetry without changing control."""

        drone = next((item for item in self.drones if item.id == drone_id), None)
        if drone is None:
            raise ValueError(f"unknown drone: {drone_id}")
        candidate_count = int(visualization.get("candidate_count", -1))
        selected_index = int(visualization.get("selected_index", -1))
        decision_sequence = int(visualization.get("decision_sequence", -1))
        if candidate_count != 15 or not 0 <= selected_index < 15:
            raise ValueError("planner visualization must contain 15 candidates and a valid selection")
        previous = self._external_policy_visualizations.get(drone_id)
        if previous and decision_sequence <= int(previous.get("decision_sequence", -1)):
            raise ValueError("visualization decision_sequence must increase strictly")
        top_candidates = list(visualization.get("top_candidates") or [])
        selected = list(visualization.get("selected_trajectory_world_m") or [])
        if len(top_candidates) != 15 or not selected:
            raise ValueError("planner visualization must expose all candidates and the selected trajectory")

        def validate_trajectory(points, name):
            array = np.asarray(points, dtype=float)
            if array.ndim != 2 or array.shape[1] != 3 or not 2 <= len(array) <= 128:
                raise ValueError(f"{name} must contain 2..128 3-D points")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite coordinates")
            return array.tolist()

        normalized_candidates = []
        seen_indices = set()
        for item in top_candidates:
            index = int(item.get("candidate_index", -1))
            if not 0 <= index < 15 or index in seen_indices:
                raise ValueError("candidate indices must cover 0..14 exactly once")
            seen_indices.add(index)
            normalized_candidates.append({
                "candidate_index": index,
                "score": float(item.get("score", 0.0)),
                "collision_probability": float(np.clip(item.get("collision_probability", 0.0), 0.0, 1.0)),
                "uncertainty": float(max(0.0, item.get("uncertainty", 0.0))),
                "predicted_collision": bool(item.get("predicted_collision", False)),
                "trajectory_world_m": validate_trajectory(item.get("trajectory_world_m"), f"candidate {index}"),
            })
        if seen_indices != set(range(15)):
            raise ValueError("candidate indices must cover 0..14 exactly once")
        normalized = {
            "decision_sequence": decision_sequence,
            "candidate_count": 15,
            "selected_index": selected_index,
            "raw_selected_index": int(visualization.get("raw_selected_index", selected_index)),
            "selection_method": str(visualization.get("selection_method", "unknown")),
            "selected_trajectory_world_m": validate_trajectory(selected, "selected trajectory"),
            "top_candidates": normalized_candidates,
            "planner_latency_ms": float(max(0.0, visualization.get("planner_latency_ms", 0.0))),
            "predicted_risk": float(np.clip(visualization.get("predicted_risk", 0.0), 0.0, 1.0)),
            "control_authority": str(visualization.get("control_authority", "candidate_reranker_only")),
            "visualization_only": True,
        }
        latent = np.asarray(visualization.get("latent_state", []), dtype=float)
        predicted_latent = np.asarray(
            visualization.get("predicted_next_latent", []), dtype=float
        )
        if latent.size or predicted_latent.size:
            if latent.ndim != 1 or not 16 <= len(latent) <= 256 or not np.isfinite(latent).all():
                raise ValueError("latent_state must contain 16..256 finite values")
            if predicted_latent.shape != latent.shape or not np.isfinite(predicted_latent).all():
                raise ValueError("predicted_next_latent must align with latent_state")
            normalized["latent_state"] = np.clip(latent, -100.0, 100.0).tolist()
            normalized["predicted_next_latent"] = np.clip(
                predicted_latent, -100.0, 100.0
            ).tolist()
            normalized["latent_norm"] = float(np.linalg.norm(latent))
            normalized["latent_delta_norm"] = float(np.linalg.norm(predicted_latent - latent))
            normalized["ensemble_uncertainty"] = float(
                max(0.0, visualization.get("ensemble_uncertainty", 0.0))
            )
        if not all(np.isfinite(value) for value in (
            normalized["planner_latency_ms"], normalized["predicted_risk"],
            normalized.get("latent_norm", 0.0), normalized.get("latent_delta_norm", 0.0),
            normalized.get("ensemble_uncertainty", 0.0),
        )):
            raise ValueError("planner visualization metrics must be finite")
        self._external_policy_visualizations[drone_id] = normalized
        return dict(normalized)

    def configure_external_policy_episode(
        self,
        drone_id: str,
        *,
        start_world_m=None,
        goal_world_m=None,
        yaw_degrees: float = 0.0,
        policy_family: str = "external_policy",
        shield_enabled: bool = True,
        episode_seed: int = 20260731,
        dynamic_actor_density: float = 1.0,
        appearance_perturbation: dict | None = None,
        dynamics_perturbation: dict | None = None,
        episode_duration_s: float | None = None,
    ) -> dict:
        """Create an auditable external-policy episode on the current city scene.

        UrbanFly stores world vectors as ``[x, up, z]``.  The WebSocket
        adapter performs the explicit conversion from its ``[north, west, up]``
        navigation frame.  Once configured, a neutral external command owns
        the aircraft until the first policy action arrives, so the built-in
        geometric/world-model controller cannot move it invisibly.
        """

        drone = next((item for item in self.drones if item.id == drone_id), None)
        if drone is None:
            raise ValueError(f"unknown drone: {drone_id}")

        def vector3(value, name, default):
            if value is None:
                return np.asarray(default, dtype=float).copy()
            vector = np.asarray(value, dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} must be a finite vector of length 3")
            return vector.copy()

        start = vector3(start_world_m, "start_world_m", drone.position)
        current_goal = (
            drone.path[-1].position
            if drone.path
            else drone.position
        )
        goal = vector3(goal_world_m, "goal_world_m", current_goal)
        if not np.isfinite(yaw_degrees):
            raise ValueError("yaw_degrees must be finite")

        scene_min, scene_max = self._get_scene_bounds()
        for name, point in (("start_world_m", start), ("goal_world_m", goal)):
            if np.any(point < scene_min) or np.any(point > scene_max):
                raise ValueError(
                    f"{name} is outside scene bounds "
                    f"{scene_min.tolist()}..{scene_max.tolist()}"
                )
        if np.linalg.norm(goal - start) < 1.0:
            raise ValueError("policy episode start and goal must be at least 1 m apart")
        if episode_duration_s is not None:
            requested_duration = float(episode_duration_s)
            if not np.isfinite(requested_duration) or not 30.0 <= requested_duration <= 1800.0:
                raise ValueError("episode_duration_s must be finite and within [30, 1800]")
            self.duration = max(float(self.duration), float(self.time) + requested_duration)

        collision_radius = max(0.75, min(2.0, float(drone.safety_radius)))
        if self.static_collision_map is not None and hasattr(
            self.static_collision_map, "collides"
        ):
            for name, point in (("start_world_m", start), ("goal_world_m", goal)):
                collides, clearance = self.static_collision_map.collides(
                    point, collision_radius
                )
                if collides:
                    raise ValueError(
                        f"{name} intersects the city collision field "
                        f"(clearance={float(clearance):.3f} m)"
                    )

        drone.position = start
        drone.velocity = np.zeros(3, dtype=float)
        drone.acceleration = np.zeros(3, dtype=float)
        drone.angular_velocity = np.zeros(3, dtype=float)
        drone.yaw = float(yaw_degrees)
        drone.roll = 0.0
        drone.pitch = 0.0
        drone.path = [
            Waypoint(
                position=goal,
                action="hover",
                metadata={
                    "external_policy_goal": True,
                    "local_planner": str(policy_family),
                },
            )
        ]
        drone.current_path_index = 0
        drone.state = DroneState.HOVERING

        appearance = dict(appearance_perturbation or {})
        dynamics = dict(dynamics_perturbation or {})
        allowed_appearance = {"exposure_ev", "fog_density", "color_temperature_k", "camera_noise_std", "frame_drop_probability"}
        allowed_dynamics = {"wind_world_mps", "mass_scale", "drag_scale", "motor_delay_ms", "control_jitter_ms"}
        if set(appearance) - allowed_appearance or set(dynamics) - allowed_dynamics:
            raise ValueError("unsupported appearance or dynamics perturbation field")
        appearance_defaults = {"exposure_ev": 0.0, "fog_density": 0.0, "color_temperature_k": 6500.0, "camera_noise_std": 0.0, "frame_drop_probability": 0.0}
        dynamics_defaults = {"wind_world_mps": [0.0, 0.0, 0.0], "mass_scale": 1.0, "drag_scale": 1.0, "motor_delay_ms": 0.0, "control_jitter_ms": 0.0}
        appearance = {**appearance_defaults, **appearance}; dynamics = {**dynamics_defaults, **dynamics}
        for name, bounds in {"exposure_ev": (-3, 3), "fog_density": (0, 0.2), "color_temperature_k": (2500, 12000), "camera_noise_std": (0, 0.1), "frame_drop_probability": (0, 0.5)}.items():
            value = float(appearance[name])
            if not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"appearance perturbation {name} is outside {bounds}")
            appearance[name] = value
        for name, bounds in {"mass_scale": (0.5, 2), "drag_scale": (0.3, 3), "motor_delay_ms": (0, 500), "control_jitter_ms": (0, 200)}.items():
            value = float(dynamics[name])
            if not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"dynamics perturbation {name} is outside {bounds}")
            dynamics[name] = value
        wind = np.asarray(dynamics["wind_world_mps"], dtype=float)
        if wind.shape != (3,) or not np.isfinite(wind).all() or np.linalg.norm(wind) > 15.0:
            raise ValueError("wind_world_mps must be a finite vector with norm <= 15 m/s")
        dynamics["wind_world_mps"] = wind.tolist()
        self._episode_appearance_perturbation = appearance
        self._episode_dynamics_perturbation = dynamics
        self._episode_wind_offset = wind

        profile = dict(MULTIROTOR_DYNAMICS.get(
            drone.drone_type,
            MULTIROTOR_DYNAMICS["standard"],
        ))
        profile["mass"] = float(profile["mass"]) * dynamics["mass_scale"]
        profile["inertia"] = (np.asarray(profile["inertia"], dtype=float) * dynamics["mass_scale"]).tolist()
        profile["linear_drag"] = (np.asarray(profile["linear_drag"], dtype=float) * dynamics["drag_scale"]).tolist()
        profile["motor_time_constant"] = float(profile["motor_time_constant"]) + dynamics["motor_delay_ms"] / 1000.0
        model = MultirotorDynamics(MultirotorParameters.from_dict(profile))
        model.initialize(drone.yaw)
        self._multirotor_models[drone.id] = model
        drone.orientation_quaternion = model.orientation.copy()
        drone.motor_omega = model.motor_omega.copy()

        self._world_model_controllers.pop(drone.id, None)
        self._external_policy_commands.pop(drone.id, None)
        self._external_policy_visualizations.pop(drone.id, None)
        self._external_policy_interventions[drone.id] = 0
        self._static_collision_counts[drone.id] = 0
        self.dynamic_actor_field.reset(
            self._get_scene_bounds(), seed=int(episode_seed), density=float(dynamic_actor_density)
        )
        self._dynamic_collision_active[drone.id] = set()
        self.set_external_policy_action(
            drone.id,
            np.zeros(4, dtype=float),
            step_id=-1,
            policy_family=str(policy_family),
            shield_enabled=bool(shield_enabled),
            timeout_s=1.0,
        )
        drone.world_model_state = {
            "enabled": True,
            "backend": str(policy_family),
            "status": "external_policy_episode_ready",
            "policy_step_id": -1,
            "safety_enabled": bool(shield_enabled),
            "safety_intervention_count": 0,
            "actual_collision_count": 0,
        }
        return {
            "drone_id": drone.id,
            "start_world_m": start.tolist(),
            "goal_world_m": goal.tolist(),
            "yaw_degrees": drone.yaw,
            "sim_time": float(self.time),
            "policy_family": str(policy_family),
            "shield_enabled": bool(shield_enabled),
            "episode_seed": int(episode_seed),
            "dynamic_actor_density": float(dynamic_actor_density),
            "episode_duration_s": float(self.duration - self.time),
            "appearance_perturbation": appearance,
            "dynamics_perturbation": dynamics,
        }

    def clear_external_policy(self, drone_id: str) -> None:
        self._external_policy_commands.pop(drone_id, None)

    def select_algorithm(self, algorithm_name: str):
        """切换分配算法"""
        from .allocator.hungarian import HungarianAllocator
        from .allocator.greedy import GreedyAllocator
        from .allocator.auction import AuctionAllocator
        from .allocator.genetic import GeneticAllocator
        from .allocator.market import MarketAllocator

        algorithms = {
            "cbba": CBBAAllocator,
            "hungarian": HungarianAllocator,
            "greedy": GreedyAllocator,
            "auction": AuctionAllocator,
            "genetic": GeneticAllocator,
            "market": MarketAllocator,
        }

        if algorithm_name in algorithms:
            self.allocator = algorithms[algorithm_name]()
            return True
        return False

    def _finish(self):
        """仿真结束"""
        self.state = "completed"
        self.stats = self._compute_stats()
        self._emit_event(
            EventType.SCENARIO_END,
            message=f"仿真完成: {self.stats.tasks_completed}/{self.stats.total_tasks} 任务完成"
        )

    # ==================================================================
    # 内部辅助
    # ==================================================================

    def _find_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def _generate_random_tasks(self, count: int):
        """生成复杂任务集（分区感知业务流 + 多属性约束）"""
        import random

        for i in range(count):
            # 按比例选择任务类型
            rand = random.random()
            cumulative = 0
            task_type = "regular"
            for ttype, cfg in TASK_TYPES.items():
                cumulative += cfg["proportion"]
                if rand <= cumulative:
                    task_type = ttype
                    break

            self.tasks.append(self._generate_single_task(task_type))

    @property
    def is_finished(self) -> bool:
        return self.state in ("completed", "stopped")
