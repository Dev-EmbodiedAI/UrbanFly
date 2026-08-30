"""
UrbanFly 全局配置
===============
统一管理所有仿真参数、无人机编队配置、任务参数、通信参数等。
"""

import numpy as np

# ============================================================================
# 仿真参数
# ============================================================================
SIMULATION = {
    "dt": 0.02,                # 50 Hz 物理；5 Hz 策略动作保持 10 个物理步
    "state_push_interval": 0.10,  # 10 Hz 遥测与同步 RGB-D
    "reallocation_interval": 5.0,  # 任务重分配间隔 (秒)
    "default_duration": 900,    # 默认仿真时长 (秒), 15分钟
    "speed_multipliers": [1.0, 2.0, 5.0, 10.0],  # 可用的仿真加速倍率
}

# ============================================================================
# 无人机编队配置 (30架)
# ============================================================================
# 无人机类型定义
DRONE_TYPES = {
    "heavy": {
        "label": "重型运输无人机",
        "color": "#e74c3c",        # 红色
        "max_speed": 12.0,         # m/s
        "cruise_speed": 10.0,      # 巡航速度 m/s
        "max_accel": 2.5,          # m/s²
        "max_payload": 15.0,       # kg
        "battery_capacity": 500.0, # Wh
        "energy_per_meter": 0.12,  # Wh/m (空载)
        "energy_per_kg_meter": 0.008,  # Wh/kg/m (载重能耗增量)
        "max_yaw_rate": 30.0,      # deg/s
        "max_climb_rate": 3.0,     # m/s
        "comm_range": 350.0,       # 通信范围 m
        "safety_radius": 3.0,      # 安全半径 m
    },
    "standard": {
        "label": "标准配送无人机",
        "color": "#3498db",        # 蓝色
        "max_speed": 15.0,
        "cruise_speed": 12.0,
        "max_accel": 3.0,
        "max_payload": 8.0,
        "battery_capacity": 350.0,
        "energy_per_meter": 0.08,
        "energy_per_kg_meter": 0.005,
        "max_yaw_rate": 45.0,
        "max_climb_rate": 4.0,
        "comm_range": 300.0,
        "safety_radius": 2.5,
    },
    "light": {
        "label": "轻型快速无人机",
        "color": "#2ecc71",        # 绿色
        "max_speed": 20.0,
        "cruise_speed": 16.0,
        "max_accel": 4.0,
        "max_payload": 3.0,
        "battery_capacity": 200.0,
        "energy_per_meter": 0.04,
        "energy_per_kg_meter": 0.003,
        "max_yaw_rate": 60.0,
        "max_climb_rate": 5.0,
        "comm_range": 250.0,
        "safety_radius": 2.0,
    },
}

# ============================================================================
# 多旋翼刚体动力学
# ============================================================================
# 四旋翼 X 构型。参数均为 SI 单位；推力模型为 T = k_f * omega^2，
# 反扭矩模型为 Q = k_m * T。控制链路为位置/速度外环 + SO(3) 姿态内环。
MULTIROTOR_DYNAMICS = {
    "heavy": {
        "mass": 8.5,
        "inertia": [0.42, 0.68, 0.42],
        "arm_length": 0.72,
        "max_thrust_per_motor": 58.0,
        "motor_time_constant": 0.075,
        "yaw_moment_ratio": 0.018,
        "linear_drag": [0.38, 0.52, 0.38],
        "angular_drag": [0.10, 0.14, 0.10],
        "position_gain": 0.38,
        "velocity_gain": 1.45,
        "attitude_gain": [18.0, 12.0, 18.0],
        "angular_rate_gain": [4.8, 3.4, 4.8],
        "hover_power_w": 1180.0,
        "avionics_power_w": 55.0,
    },
    "standard": {
        "mass": 3.2,
        "inertia": [0.095, 0.16, 0.095],
        "arm_length": 0.52,
        "max_thrust_per_motor": 26.0,
        "motor_time_constant": 0.055,
        "yaw_moment_ratio": 0.016,
        "linear_drag": [0.22, 0.30, 0.22],
        "angular_drag": [0.045, 0.065, 0.045],
        "position_gain": 0.46,
        "velocity_gain": 1.65,
        "attitude_gain": [11.0, 7.5, 11.0],
        "angular_rate_gain": [2.5, 1.8, 2.5],
        "hover_power_w": 470.0,
        "avionics_power_w": 34.0,
    },
    "light": {
        "mass": 1.35,
        "inertia": [0.026, 0.043, 0.026],
        "arm_length": 0.34,
        "max_thrust_per_motor": 12.0,
        "motor_time_constant": 0.040,
        "yaw_moment_ratio": 0.014,
        "linear_drag": [0.12, 0.16, 0.12],
        "angular_drag": [0.020, 0.030, 0.020],
        "position_gain": 0.58,
        "velocity_gain": 1.85,
        "attitude_gain": [6.2, 4.2, 6.2],
        "angular_rate_gain": [1.25, 0.9, 1.25],
        "hover_power_w": 205.0,
        "avionics_power_w": 22.0,
    },
}

# UrbanFly 机载视觉传感器。RGB 与透视深度共用光心、曝光时刻和内参，
# 避免多帧拼接时的位姿漂移。body_pose 使用 [x, y, z] 米与 RPY 角度。
CAMERA_SENSORS = {
    "front_center": {
        "body_pose": {
            "position": [1.05, 0.02, 0.0],
            "roll_pitch_yaw_degrees": [0.0, -8.0, 0.0],
        },
        "capture_settings": {
            # Dataset v1 capture resolution.  The browser bridge must finish
            # RGB-D encoding before the next 10 Hz control observation; the
            # former 640x360 targets took roughly two simulator seconds per
            # packet on the acceptance-test WebGL runtime.
            "width": 160,
            "height": 90,
            "fov_degrees": 90.0,
            "near_clip_m": 0.3,
            "far_clip_m": 120.0,
            "frame_rate_hz": 10.0,
            "motion_blur": 0.0,
        },
        "image_types": ["Scene", "DepthPerspective"],
        "depth_unit": "meter",
        "gimbal_stabilization": {
            "roll": False,
            "pitch": False,
            "yaw": False,
        },
    },
}

# 编队组成
FLEET_COMPOSITION = {
    "heavy": 5,
    "standard": 15,
    "light": 10,
}

# 充电站配置
CHARGING_STATIONS = [
    {"id": "charge_01", "pos": [-350, 1, -400], "capacity": 5, "charge_rate": 50},   # Wh/s
    {"id": "charge_02", "pos": [350, 1, -400], "capacity": 5, "charge_rate": 50},
    {"id": "charge_03", "pos": [-350, 1, 400], "capacity": 5, "charge_rate": 50},
    {"id": "charge_04", "pos": [350, 1, 400], "capacity": 5, "charge_rate": 50},
    {"id": "charge_05", "pos": [0, 1, 0], "capacity": 8, "charge_rate": 80},  # 中央大型充电站
]

# 无人机起降点 (散布在街道交叉口)
# 城市: 16×16 blocks, block=44m, street=10m, total=874m
# 街道中心: -432 + i*54, i=0..16
import random as _random
_random.seed(42)
_BASE_STREETS = [-432 + i * 54 for i in range(17)]
BASE_STATIONS = [
    {"id": f"base_{i+1:02d}", "pos": [
        _random.choice(_BASE_STREETS) + _random.uniform(-4, 4),
        0.5,
        _random.choice(_BASE_STREETS) + _random.uniform(-4, 4),
    ]}
    for i in range(30)
]

# ============================================================================
# 任务参数
# ============================================================================
TASK_TYPES = {
    "emergency_medical": {
        "priority": 0,
        "label": "紧急医疗物资",
        "color": "#ff0000",
        "time_window": (300, 600),      # 5-10分钟 (秒)
        "payload_range": (0.5, 3.0),    # kg
        "reward": 500,
        "deadline_penalty": 200,         # 超时惩罚系数
        "required_comms": True,
        "proportion": 0.05,              # 占总任务的5%
    },
    "medical": {
        "priority": 1,
        "label": "医疗物资配送",
        "color": "#ff6600",
        "time_window": (600, 1200),     # 10-20分钟
        "payload_range": (1.0, 8.0),
        "reward": 300,
        "deadline_penalty": 100,
        "required_comms": True,
        "proportion": 0.10,
    },
    "fresh": {
        "priority": 2,
        "label": "生鲜配送",
        "color": "#ffcc00",
        "time_window": (900, 1800),     # 15-30分钟
        "payload_range": (2.0, 10.0),
        "reward": 150,
        "deadline_penalty": 50,
        "required_comms": False,
        "proportion": 0.20,
    },
    "regular": {
        "priority": 3,
        "label": "常规物资配送",
        "color": "#0066ff",
        "time_window": (1800, 3600),    # 30-60分钟
        "payload_range": (1.0, 15.0),
        "reward": 80,
        "deadline_penalty": 20,
        "required_comms": False,
        "proportion": 0.50,
    },
    "patrol": {
        "priority": 4,
        "label": "普通巡检任务",
        "color": "#999999",
        "time_window": (0, 7200),       # 无硬性约束
        "payload_range": (0.0, 0.5),
        "reward": 40,
        "deadline_penalty": 5,
        "required_comms": False,
        "proportion": 0.15,
        "patrol_waypoints": 5,          # 巡检航点数
    },
}

# 任务优先级权重 (用于边际收益计算)
PRIORITY_WEIGHTS = {
    0: 10.0,   # 紧急医疗 — 最高权重
    1: 5.0,    # 医疗物资
    2: 2.5,    # 生鲜配送
    3: 1.0,    # 常规配送
    4: 0.3,    # 巡检任务
}

# 任务生成参数
TASK_GENERATION = {
    "total_tasks": 100,          # 初始总任务数
    "pickup_height_range": (0.5, 35.0),   # 取件高度范围 (地面 ~ 建筑屋顶)
    "delivery_height_range": (0.5, 35.0), # 递送高度范围
    "dynamic_emergency_interval": 60.0,   # 动态紧急任务生成间隔 (秒)
    "dynamic_regular_batch_interval": 120.0,  # 动态常规任务批次间隔
    "dynamic_regular_batch_size": 5,      # 每批常规任务数
    "service_time_range": (15.0, 120.0),
    "risk_range": (0.05, 0.95),
    "default_min_battery_pct": 0.15,
    "default_min_neighbor_count": 0,
}

# 分区到业务热点的映射，用于更复杂的任务建模
DISTRICT_TASK_RULES = {
    "cbd": {
        "pickup_bias": ["medical", "regular", "patrol"],
        "delivery_bias": ["medical", "regular", "fresh"],
        "business_tags": ["hospital_transfer", "high_value_delivery", "financial_district_patrol"],
    },
    "mixed": {
        "pickup_bias": ["medical", "fresh", "regular"],
        "delivery_bias": ["medical", "fresh", "regular"],
        "business_tags": ["retail_replenishment", "community_supply", "express_dispatch"],
    },
    "residential": {
        "pickup_bias": ["regular", "patrol"],
        "delivery_bias": ["medical", "fresh", "regular"],
        "business_tags": ["last_mile_delivery", "community_patrol", "home_medical_support"],
    },
    "industrial": {
        "pickup_bias": ["fresh", "regular", "patrol"],
        "delivery_bias": ["fresh", "regular"],
        "business_tags": ["warehouse_linehaul", "industrial_supply", "perimeter_inspection"],
    },
    "park": {
        "pickup_bias": ["patrol"],
        "delivery_bias": ["patrol"],
        "business_tags": ["greenbelt_patrol", "public_safety_watch"],
    },
    "plaza": {
        "pickup_bias": ["medical", "patrol", "regular"],
        "delivery_bias": ["medical", "regular"],
        "business_tags": ["civic_center_dispatch", "event_security_patrol"],
    },
}

# 复杂任务模板库：业务语义 + 兼容性约束 + 服务时间
TASK_TEMPLATE_LIBRARY = {
    "emergency_medical": [
        {
            "business_tag": "hospital_transfer",
            "pickup_districts": ["mixed", "industrial", "plaza"],
            "delivery_districts": ["cbd", "mixed", "residential"],
            "service_time_range": (20.0, 40.0),
            "risk_range": (0.55, 0.95),
            "cold_chain_probability": 0.35,
            "fragile_probability": 0.55,
            "preferred_drone_types": ["light", "standard"],
            "min_neighbor_count": 1,
            "aging_weight": 1.8,
        },
        {
            "business_tag": "blood_supply",
            "pickup_districts": ["industrial", "mixed"],
            "delivery_districts": ["cbd", "residential"],
            "service_time_range": (25.0, 50.0),
            "risk_range": (0.65, 0.98),
            "cold_chain_probability": 0.80,
            "fragile_probability": 0.25,
            "preferred_drone_types": ["light", "standard"],
            "min_neighbor_count": 2,
            "aging_weight": 2.0,
        },
    ],
    "medical": [
        {
            "business_tag": "clinic_supply",
            "pickup_districts": ["mixed", "industrial"],
            "delivery_districts": ["residential", "mixed", "cbd"],
            "service_time_range": (35.0, 80.0),
            "risk_range": (0.30, 0.75),
            "cold_chain_probability": 0.45,
            "fragile_probability": 0.30,
            "preferred_drone_types": ["standard", "light"],
            "min_neighbor_count": 1,
            "aging_weight": 1.45,
        },
    ],
    "fresh": [
        {
            "business_tag": "cold_chain_retail",
            "pickup_districts": ["industrial", "mixed"],
            "delivery_districts": ["mixed", "residential", "cbd"],
            "service_time_range": (30.0, 90.0),
            "risk_range": (0.20, 0.65),
            "cold_chain_probability": 0.70,
            "fragile_probability": 0.20,
            "preferred_drone_types": ["standard", "heavy"],
            "min_neighbor_count": 0,
            "aging_weight": 1.2,
        },
    ],
    "regular": [
        {
            "business_tag": "warehouse_linehaul",
            "pickup_districts": ["industrial", "mixed"],
            "delivery_districts": ["residential", "mixed", "cbd", "plaza"],
            "service_time_range": (25.0, 110.0),
            "risk_range": (0.10, 0.55),
            "cold_chain_probability": 0.05,
            "fragile_probability": 0.35,
            "preferred_drone_types": ["heavy", "standard"],
            "min_neighbor_count": 0,
            "aging_weight": 1.0,
        },
        {
            "business_tag": "reverse_logistics",
            "pickup_districts": ["residential", "mixed", "cbd"],
            "delivery_districts": ["industrial", "mixed"],
            "service_time_range": (30.0, 120.0),
            "risk_range": (0.10, 0.45),
            "cold_chain_probability": 0.0,
            "fragile_probability": 0.15,
            "preferred_drone_types": ["standard", "heavy"],
            "min_neighbor_count": 0,
            "aging_weight": 1.1,
        },
    ],
    "patrol": [
        {
            "business_tag": "infrastructure_patrol",
            "pickup_districts": ["industrial", "park", "cbd", "plaza"],
            "delivery_districts": ["industrial", "park", "cbd", "plaza"],
            "service_time_range": (45.0, 140.0),
            "risk_range": (0.15, 0.85),
            "cold_chain_probability": 0.0,
            "fragile_probability": 0.0,
            "preferred_drone_types": ["light", "standard"],
            "min_neighbor_count": 0,
            "aging_weight": 0.9,
        },
    ],
}

# ============================================================================
# 通信模型参数
# ============================================================================
COMMUNICATION = {
    "max_range": 300.0,          # 默认最大通信距离 (m)
    "bandwidth": 100,            # 信道容量 (消息数/秒)
    "packet_loss_base": 0.0,     # 基础丢包率
    "building_signal_attenuation": 0.6,  # 建筑穿透信号衰减系数
    "ray_check_samples": 20,     # 视线检查采样点数
    "comm_update_interval": 1.0, # 通信拓扑更新间隔 (秒)
}

# 通信场景预设
COMM_SCENARIOS = {
    "ideal": {
        "max_range": 9999.0,
        "bandwidth": 9999,
        "packet_loss_base": 0.0,
        "use_building_block": False,
    },
    "distance_limited": {
        "max_range": 300.0,
        "bandwidth": 200,
        "packet_loss_base": 0.0,
        "use_building_block": False,
    },
    "building_blocked": {
        "max_range": 300.0,
        "bandwidth": 200,
        "packet_loss_base": 0.05,
        "use_building_block": True,
    },
    "intermittent": {
        "max_range": 250.0,
        "bandwidth": 150,
        "packet_loss_base": 0.20,
        "use_building_block": True,
    },
    "harsh": {
        "max_range": 150.0,
        "bandwidth": 50,
        "packet_loss_base": 0.50,
        "use_building_block": True,
    },
}

# ============================================================================
# 路径规划参数
# ============================================================================
PATH_PLANNING = {
    "grid_resolution": 5.0,      # 体素网格分辨率 (m)
    "safety_margin": 2.0,        # 安全边界 (m)
    "max_astar_iterations": 100000,
    "rrt_star_max_iter": 800,
    "rrt_star_step_size": 15.0,  # 米
    "rrt_star_goal_sample_rate": 0.2,
    "smooth_points": 150,        # 路径平滑插值点数
    "climb_penalty_factor": 2.0, # 爬升惩罚因子
    "time_slot_sec": 30.0,       # 时空走廊时间槽长度
    "corridor_cell_size": 70.0,  # 走廊聚合网格尺寸
    "corridor_capacity": 3,      # 单走廊单时间槽容量
    "density_penalty_factor": 4.0,
    "corridor_penalty_factor": 18.0,
    "layer_transition_penalty": 20.0,
    "detour_penalty_factor": 1.2,
    "local_repair_search_radius": 3,
    "repair_sample_count": 7,
    "bspline_samples_per_segment": 18,
    "bspline_clearance_margin": 4.0,
    "bspline_curvature_weight": 0.35,
    "bspline_climb_weight": 0.25,
    "bspline_clearance_weight": 0.30,
    "bspline_time_consistency_weight": 0.10,
}

# HelsinkiCentral1km privileged-map navigation.  These values are kept
# separate from the legacy multi-UAV scenario configuration so the physical
# airframe radius and the requested operational margin remain auditable.
HELSINKI_NAVIGATION = {
    "planning_resolution_m": 5.0,
    "drone_radius_m": 0.75,
    "safety_margin_m": 1.75,
    "tracking_buffer_m": 1.5,
    "minimum_altitude_m": -5.0,
    "maximum_altitude_m": 88.0,
    "validation_step_m": 0.25,
    "trajectory_spacing_m": 3.0,
    "corner_radius_m": 4.0,
    "corner_samples": 5,
    "cruise_speed_mps": 12.0,
    "goal_tolerance_m": 2.0,
    "execution_timeout_s": 180.0,
    "low_altitude_step_m": 2.5,
    "low_altitude_min_ceiling_m": 15.0,
    "low_altitude_max_ceiling_m": 40.0,
    "low_altitude_vertical_tracking_buffer_m": 1.0,
    "low_altitude_corner_radius_m": 12.0,
}

# 高度层定义 (相对于场景Y轴)
FLIGHT_LEVELS = {
    "L4_emergency":      {"y_min": 65.0, "y_max": 90.0, "label": "应急优先层"},
    "L3_trunk_corridor": {"y_min": 40.0, "y_max": 60.0, "label": "干线巡航层"},
    "L3_high_corridor": {"y_min": 40.0, "y_max": 60.0, "label": "高层自由走廊"},
    "L2_transition":    {"y_min": 18.0, "y_max": 38.0, "label": "社区园区过渡层"},
    "L2_mid_level":     {"y_min": 15.0, "y_max": 35.0, "label": "中层"},
    "L1_street_canyon": {"y_min": -20.0, "y_max": 10.0, "label": "低层街道峡谷"},
    "L0_ground":        {"y_min": -25.0, "y_max": -18.0, "label": "地面起降层"},
}

# 默认路径规划飞行层 (L1 = 街道峡谷，需要绕建筑)
DEFAULT_FLIGHT_LEVEL = "L1_street_canyon"

# ============================================================================
# 碰撞避免参数
# ============================================================================
COLLISION_AVOIDANCE = {
    "drone_safety_cylinder_radius": 5.0,   # 无人机安全圆柱半径 (m)
    "drone_safety_cylinder_height": 8.0,   # 无人机安全圆柱高度 (m)
    "collision_warning_time": 10.0,         # 碰撞预警时间 (秒)
    "velocity_obstacle_time_horizon": 5.0,  # VO时间视野 (秒)
    "altitude_separation": 5.0,            # 高度分离 (m)
    "speed_adjustment_factor": 0.3,        # 速度调整因子
}

# ============================================================================
# CBBA算法参数
# ============================================================================
CBBA = {
    "max_bundle_size": 10,       # 每架无人机最大bundle容量
    "max_iterations": 100,       # 最大共识迭代次数
    "convergence_threshold": 3,  # 连续无变化迭代数视为收敛
    "battery_safety_margin": 0.15,  # 电池安全余量 (15%)
    "marginal_gain_noise": 0.01,    # 边际收益噪声 (打破平局)
    "top_k_exact_eval": 10,         # 精确插入评估的候选数
    "corridor_bonus_factor": 0.10,
    "corridor_penalty_factor": 0.24,
    "urgency_bonus_factor": 0.45,
    "aging_bonus_factor": 4.5,
    "risk_penalty_factor": 0.08,
    "fragile_penalty_factor": 0.06,
    "comm_penalty_factor": 0.22,
    "energy_penalty_factor": 0.55,
    "event_sync_max_delta": 48,
    "consensus_rebroadcast_full_interval": 4,
}

# 高密度城市生成配置
DENSE_CITY = {
    "blocks_x": 22,
    "blocks_z": 20,
    "block_size": 36,
    "street_width": 8,
    "height_min": 8,
    "height_max": 120,
    "height_center_bias": 2.3,
    "vacancy_rate": 0.03,
    "park_rate": 0.02,
    "plaza_center": True,
    "plaza_radius": 1,
    "hotspot_count": 6,
}

# ============================================================================
# WebSocket 服务器参数
# ============================================================================
SERVER = {
    "host": "localhost",
    "port": 8765,
    "static_dir": "../../frontend/dist",   # 相对backend/server目录
    "cors_origins": ["*"],
}

# ============================================================================
# OBJ文件路径
# ============================================================================
PATHS = {
    "obj_file": "C:/Users/caste/Desktop/paris/Paris_city_only.obj",
    "mtl_file": "C:/Users/caste/Desktop/paris/Paris.mtl",
    "output_scene": "data/scene/",
    "output_scenarios": "data/scenarios/",
}
