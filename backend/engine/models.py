"""
UrbanFly 核心数据类型
=====================
定义仿真中使用的所有数据结构：无人机、任务、路径、CBBA消息等。
"""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import uuid
import time as _time


# ============================================================================
# 枚举类型
# ============================================================================

class TaskPriority(IntEnum):
    """任务优先级 (数字越小优先级越高)"""
    EMERGENCY_MEDICAL = 0
    MEDICAL = 1
    FRESH = 2
    REGULAR = 3
    PATROL = 4


class TaskType(Enum):
    """任务类型"""
    EMERGENCY_MEDICAL = ("emergency_medical", TaskPriority.EMERGENCY_MEDICAL)
    MEDICAL = ("medical", TaskPriority.MEDICAL)
    FRESH = ("fresh", TaskPriority.FRESH)
    REGULAR = ("regular", TaskPriority.REGULAR)
    PATROL = ("patrol", TaskPriority.PATROL)

    def __init__(self, label, priority):
        self.label = label
        self.priority = priority


class DroneType(Enum):
    """无人机类型"""
    HEAVY = "heavy"
    STANDARD = "standard"
    LIGHT = "light"


class DroneState(Enum):
    """无人机运行状态"""
    IDLE = "idle"               # 空闲，等待任务
    TAKEOFF = "takeoff"         # 起飞中
    EN_ROUTE = "en_route"       # 前往目标途中
    PICKING_UP = "picking_up"   # 取件中
    DELIVERING = "delivering"   # 递送中
    HOVERING = "hovering"       # 悬停等待
    RETURNING = "returning"     # 返回基地
    CHARGING = "charging"       # 充电中
    EMERGENCY = "emergency"     # 紧急状态 (低电量/故障)
    LANDED = "landed"           # 已降落


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"            # 待分配
    ASSIGNED = "assigned"          # 已分配
    EN_ROUTE_PICKUP = "en_route_pickup"  # 前往取件
    PICKING_UP = "picking_up"      # 取件中
    EN_ROUTE_DELIVERY = "en_route_delivery"  # 前往递送
    DELIVERING = "delivering"      # 递送中
    COMPLETED = "completed"        # 已完成
    FAILED = "failed"              # 失败 (超时/取消)
    CANCELLED = "cancelled"        # 已取消


class EventType(Enum):
    """仿真事件类型"""
    TAKEOFF = "takeoff"
    LANDED = "landed"
    PICKUP_COMPLETE = "pickup_complete"
    DELIVERY_COMPLETE = "delivery_complete"
    TASK_ASSIGNED = "task_assigned"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    CHARGING_STARTED = "charging_started"
    CHARGING_COMPLETE = "charging_complete"
    COLLISION_WARNING = "collision_warning"
    COLLISION_AVOIDED = "collision_avoided"
    PATH_REPLANNED = "path_replanned"
    COMMUNICATION_LOST = "communication_lost"
    COMMUNICATION_RESTORED = "communication_restored"
    NEW_TASK_GENERATED = "new_task_generated"
    DRONE_RETURNING = "drone_returning"
    SCENARIO_END = "scenario_end"


# ============================================================================
# 核心数据结构
# ============================================================================

@dataclass
class Waypoint:
    """路径点"""
    position: np.ndarray        # (x, y, z) 米
    arrival_time: float = 0.0   # 预计到达时间 (仿真秒)
    action: Optional[str] = None  # 在此路径点的动作: "pickup" / "delivery" / "charge" / "hover"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "pos": self.position.tolist(),
            "arrival_time": self.arrival_time,
            "action": self.action,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d):
        return Waypoint(
            position=np.array(d["pos"]),
            arrival_time=d.get("arrival_time", 0),
            action=d.get("action"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Task:
    """配送/巡检任务"""
    id: str
    task_type: str                    # "emergency_medical" | "medical" | "fresh" | "regular" | "patrol"
    priority: int                     # 0-4
    pickup_pos: np.ndarray            # (x, y, z) 取件位置
    delivery_pos: np.ndarray          # (x, y, z) 递送位置 (巡检任务为终点)
    time_window: Tuple[float, float]  # (earliest_start, latest_completion) 秒
    payload_weight: float             # 货物重量 kg
    reward: float                     # 完成奖励
    deadline_penalty: float           # 超时惩罚系数 (每秒)
    required_comms: bool              # 是否需要持续通信
    created_at: float                 # 任务生成时间 (仿真秒)
    business_tag: str = "generic"     # 业务标签: hospital_transfer / cold_chain / patrol 等
    pickup_block_id: Optional[int] = None
    delivery_block_id: Optional[int] = None
    pickup_district: str = ""
    delivery_district: str = ""
    pickup_service_time: float = 0.0
    delivery_service_time: float = 0.0
    risk_level: float = 0.0           # 风险等级 [0, 1]
    cold_chain: bool = False          # 是否冷链
    fragile: bool = False             # 是否易碎
    min_required_battery_pct: float = 0.15
    min_neighbor_count: int = 0       # 通信敏感任务的最低邻居数
    preferred_drone_types: Optional[List[str]] = None
    airspace_level: Optional[str] = None
    aging_weight: float = 1.0         # 随等待时间增长的优先系数
    task_group: Optional[str] = None  # 用于同批次/同走廊任务聚合
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None  # 分配的无人机ID
    patrol_waypoints: Optional[List[np.ndarray]] = None  # 巡检任务的航点列表

    def to_dict(self):
        return {
            "id": self.id,
            "task_type": self.task_type,
            "priority": self.priority,
            "pickup_pos": self.pickup_pos.tolist(),
            "delivery_pos": self.delivery_pos.tolist(),
            "time_window": list(self.time_window),
            "payload_weight": self.payload_weight,
            "reward": self.reward,
            "deadline_penalty": self.deadline_penalty,
            "required_comms": self.required_comms,
            "created_at": self.created_at,
            "business_tag": self.business_tag,
            "pickup_block_id": self.pickup_block_id,
            "delivery_block_id": self.delivery_block_id,
            "pickup_district": self.pickup_district,
            "delivery_district": self.delivery_district,
            "pickup_service_time": self.pickup_service_time,
            "delivery_service_time": self.delivery_service_time,
            "risk_level": self.risk_level,
            "cold_chain": self.cold_chain,
            "fragile": self.fragile,
            "min_required_battery_pct": self.min_required_battery_pct,
            "min_neighbor_count": self.min_neighbor_count,
            "preferred_drone_types": self.preferred_drone_types,
            "airspace_level": self.airspace_level,
            "aging_weight": self.aging_weight,
            "task_group": self.task_group,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "patrol_waypoints": [w.tolist() for w in self.patrol_waypoints] if self.patrol_waypoints else None,
        }

    @property
    def is_patrol(self):
        return self.task_type == "patrol"

    @property
    def time_window_open(self):
        """任务是否已到最早开始时间"""
        return _time.time() >= self.time_window[0]

    def is_expired(self, current_time: float):
        """任务是否已过期（需要传入当前仿真时间）"""
        if self.time_window[1] == 0:  # 无硬性截止
            return False
        return current_time > self.time_window[1]


@dataclass
class DroneStateData:
    """无人机运行状态数据"""
    id: str
    drone_type: str                   # "heavy" | "standard" | "light"
    position: np.ndarray              # (x, y, z) 米
    velocity: np.ndarray              # (vx, vy, vz) m/s
    acceleration: np.ndarray          # (ax, ay, az) m/s²
    yaw: float                        # 偏航角 degrees
    battery_remaining: float          # 剩余电量 Wh
    payload_current: float            # 当前负载 kg
    state: DroneState                 # 运行状态

    # 配置参数 (从DroneConfig复制)
    max_speed: float = 15.0
    cruise_speed: float = 12.0
    max_accel: float = 3.0
    max_payload: float = 8.0
    battery_capacity: float = 350.0
    energy_per_meter: float = 0.08
    energy_per_kg_meter: float = 0.005
    max_yaw_rate: float = 45.0
    max_climb_rate: float = 4.0
    comm_range: float = 300.0
    safety_radius: float = 2.5

    # 6-DOF 多旋翼动力学状态
    roll: float = 0.0
    pitch: float = 0.0
    orientation_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motor_omega: np.ndarray = field(default_factory=lambda: np.zeros(4))
    motor_thrusts: np.ndarray = field(default_factory=lambda: np.zeros(4))
    total_thrust: float = 0.0
    power_w: float = 0.0
    dynamics_model: str = "urbanfly_fast_multirotor_6dof"
    world_model_state: Dict = field(default_factory=dict)

    # 任务相关
    assigned_tasks: List[str] = field(default_factory=list)  # 分配的任务ID列表
    current_task_id: Optional[str] = None
    current_path_index: int = 0
    path: List[Waypoint] = field(default_factory=list)

    # 通信相关
    comm_neighbors: List[str] = field(default_factory=list)  # 当前可通信的邻居无人机ID

    # 统计
    total_distance_traveled: float = 0.0
    total_energy_consumed: float = 0.0
    tasks_completed: int = 0

    def to_dict(self):
        return {
            "id": self.id,
            "drone_type": self.drone_type,
            "pos": self.position.tolist(),
            "vel": self.velocity.tolist(),
            "accel": self.acceleration.tolist(),
            "yaw": self.yaw,
            "roll": round(float(self.roll), 3),
            "pitch": round(float(self.pitch), 3),
            "orientation": self.orientation_quaternion.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "motor_omega": self.motor_omega.tolist(),
            "motor_thrusts": self.motor_thrusts.tolist(),
            "total_thrust": round(float(self.total_thrust), 3),
            "power_w": round(float(self.power_w), 2),
            "dynamics_model": self.dynamics_model,
            "world_model": self.world_model_state,
            "battery": round(self.battery_remaining / self.battery_capacity, 4) if self.battery_capacity > 0 else 0,
            "battery_wh": round(self.battery_remaining, 1),
            "payload": round(self.payload_current, 2),
            "max_payload": self.max_payload,
            "state": self.state.value,
            "assigned_tasks": self.assigned_tasks,
            "current_task": self.current_task_id,
            "comm_neighbors": self.comm_neighbors,
            "path_remaining": [w.to_dict()["pos"] for w in self.path[self.current_path_index:self.current_path_index+10]],
            "speed": round(float(np.linalg.norm(self.velocity)), 2),
            "tasks_completed": self.tasks_completed,
            "total_distance": round(self.total_distance_traveled, 1),
        }

    @property
    def battery_pct(self) -> float:
        return self.battery_remaining / self.battery_capacity if self.battery_capacity > 0 else 0

    @property
    def is_battery_low(self) -> bool:
        return self.battery_pct < 0.20

    @property
    def is_battery_critical(self) -> bool:
        return self.battery_pct < 0.05


@dataclass
class CBBAMessage:
    """CBBA 共识阶段消息"""
    sender_id: str
    timestamp: float
    bundle: List[str]                    # 当前分配的任务ID列表
    bundle_scores: List[float]           # 每个任务的边际收益
    winners: Dict[str, str]              # task_id → winning drone_id
    winner_bids: Dict[str, float]        # task_id → winning bid
    timestamp_vector: Dict[str, float]   # task_id → 最后更新时间

    def to_dict(self):
        return {
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "bundle": self.bundle,
            "bundle_scores": self.bundle_scores,
            "winners": self.winners,
            "winner_bids": self.winner_bids,
            "timestamp_vector": self.timestamp_vector,
        }

    @staticmethod
    def from_dict(d):
        return CBBAMessage(
            sender_id=d["sender_id"],
            timestamp=d["timestamp"],
            bundle=d["bundle"],
            bundle_scores=d["bundle_scores"],
            winners=d["winners"],
            winner_bids=d["winner_bids"],
            timestamp_vector=d["timestamp_vector"],
        )


@dataclass
class SimulationEvent:
    """仿真事件"""
    time: float
    event_type: EventType
    drone_id: Optional[str] = None
    task_id: Optional[str] = None
    message: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "time": self.time,
            "type": self.event_type.value,
            "drone_id": self.drone_id,
            "task_id": self.task_id,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class BuildingInfo:
    """建筑包围盒信息 (从OBJ预处理提取)"""
    id: int
    original_group: str
    bounds_min: np.ndarray       # (x, y, z) 包围盒最小点
    bounds_max: np.ndarray       # (x, y, z) 包围盒最大点
    num_faces_original: int
    num_faces_simplified: int = 0

    @property
    def center(self) -> np.ndarray:
        return (self.bounds_min + self.bounds_max) / 2

    @property
    def height(self) -> float:
        return self.bounds_max[1] - self.bounds_min[1]

    @property
    def roof_level(self) -> float:
        return self.bounds_max[1]

    def contains(self, point: np.ndarray) -> bool:
        """检查点是否在建筑包围盒内"""
        return np.all(point >= self.bounds_min) and np.all(point <= self.bounds_max)

    def line_intersects(self, p1: np.ndarray, p2: np.ndarray, num_samples: int = 20) -> bool:
        """检查线段是否穿过建筑包围盒（采样法）"""
        for t in np.linspace(0, 1, num_samples):
            if self.contains(p1 + t * (p2 - p1)):
                return True
        return False

    def to_dict(self):
        return {
            "id": self.id,
            "original_group": self.original_group,
            "bounds_min": self.bounds_min.tolist(),
            "bounds_max": self.bounds_max.tolist(),
            "center": self.center.tolist(),
            "height": self.height,
            "roof_level": self.roof_level,
            "num_faces_original": self.num_faces_original,
            "num_faces_simplified": self.num_faces_simplified,
            "contains": None,  # 不序列化函数
            "line_intersects": None,
        }


@dataclass
class BlockInfo:
    """城市街区/分区信息"""
    id: int
    name: str
    district: str
    polygon: np.ndarray
    area: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.polygon, axis=0)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "district": self.district,
            "polygon": self.polygon.tolist(),
            "area": self.area,
            "metadata": self.metadata,
        }


@dataclass
class SceneConfig:
    """场景配置"""
    name: str
    bounds_center: np.ndarray     # 场景中心 (x, y, z)
    bounds_size: np.ndarray       # 场景尺寸 (sx, sy, sz)
    buildings: List[BuildingInfo] = field(default_factory=list)
    blocks: List[BlockInfo] = field(default_factory=list)
    grid_resolution: float = 5.0
    occupancy_grid_shape: Tuple[int, int, int] = (0, 0, 0)
    occupancy_grid_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": self.name,
            "bounds_center": self.bounds_center.tolist(),
            "bounds_size": self.bounds_size.tolist(),
            "num_buildings": len(self.buildings),
            "num_blocks": len(self.blocks),
            "grid_resolution": self.grid_resolution,
            "occupancy_grid_shape": list(self.occupancy_grid_shape),
            "occupancy_grid_origin": self.occupancy_grid_origin.tolist(),
            "metadata": self.metadata,
        }


@dataclass
class ScenarioDefinition:
    """预定义场景"""
    name: str
    description: str
    duration: float                         # 仿真时长 (秒)
    comm_scenario: str                      # 通信场景: "ideal" | "distance_limited" | "building_blocked" | "intermittent" | "harsh"
    algorithm: str                          # 使用的分配算法: "cbba" | "hungarian" | "greedy" | "auction" | "genetic" | "market"
    drones: List[dict]                      # 无人机配置列表
    tasks: List[dict]                       # 任务配置列表
    dynamic_tasks_enabled: bool = False     # 是否启用动态任务生成
    events: List[dict] = field(default_factory=list)  # 脚本化事件
    semantic_agent: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "duration": self.duration,
            "comm_scenario": self.comm_scenario,
            "algorithm": self.algorithm,
            "num_drones": len(self.drones),
            "num_tasks": len(self.tasks),
            "dynamic_tasks_enabled": self.dynamic_tasks_enabled,
            "semantic_agent": self.semantic_agent,
        }


@dataclass
class SimulationStats:
    """仿真统计信息"""
    total_tasks: int = 0
    tasks_completed: int = 0
    tasks_in_progress: int = 0
    tasks_pending: int = 0
    tasks_failed: int = 0
    tasks_by_type: Dict[str, int] = field(default_factory=dict)
    tasks_by_priority: Dict[int, int] = field(default_factory=dict)
    tasks_by_business: Dict[str, int] = field(default_factory=dict)
    on_time_completion_rate: float = 0.0    # 按时完成率
    avg_response_time: float = 0.0          # 平均响应时间 (秒)
    total_distance: float = 0.0             # 总飞行距离 (m)
    total_energy: float = 0.0               # 总能耗 (Wh)
    avg_battery_at_end: float = 0.0         # 仿真结束平均电量
    avg_task_risk: float = 0.0
    cold_chain_tasks: int = 0
    fragile_tasks: int = 0
    collision_warnings: int = 0
    path_replans: int = 0
    comm_disconnections: int = 0
    algorithm_runtime_ms: float = 0.0       # 算法运行时间 (毫秒)

    def to_dict(self):
        return {
            "total_tasks": self.total_tasks,
            "completed": self.tasks_completed,
            "in_progress": self.tasks_in_progress,
            "pending": self.tasks_pending,
            "failed": self.tasks_failed,
            "tasks_by_type": self.tasks_by_type,
            "tasks_by_priority": self.tasks_by_priority,
            "tasks_by_business": self.tasks_by_business,
            "on_time_rate": round(self.on_time_completion_rate, 4),
            "avg_response_time": round(self.avg_response_time, 2),
            "total_distance": round(self.total_distance, 1),
            "total_energy": round(self.total_energy, 1),
            "avg_battery": round(self.avg_battery_at_end, 4),
            "avg_task_risk": round(self.avg_task_risk, 3),
            "cold_chain_tasks": self.cold_chain_tasks,
            "fragile_tasks": self.fragile_tasks,
            "collision_warnings": self.collision_warnings,
            "path_replans": self.path_replans,
            "comm_disconnections": self.comm_disconnections,
            "algorithm_runtime_ms": round(self.algorithm_runtime_ms, 1),
        }
