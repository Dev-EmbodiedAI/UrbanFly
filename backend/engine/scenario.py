"""
场景引擎
=======
加载和管理预定义仿真场景。
"""

import json
import os
from typing import List, Dict, Optional

from .models import ScenarioDefinition


class ScenarioEngine:
    """
    场景加载与编排引擎。

    负责：
    - 从JSON文件加载场景定义
    - 验证场景配置
    - 提供场景列表
    """

    def __init__(self, scenarios_dir: str = None):
        self.scenarios_dir = scenarios_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "scenarios"
        )
        self.scenarios: Dict[str, ScenarioDefinition] = {}
        self._load_all()

    def _load_all(self):
        """加载场景目录中的所有JSON场景"""
        if not os.path.isdir(self.scenarios_dir):
            return

        for filename in sorted(os.listdir(self.scenarios_dir)):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(self.scenarios_dir, filename)
                    scenario = self._load_from_file(filepath)
                    if scenario:
                        self.scenarios[scenario.name] = scenario
                except Exception as e:
                    print(f"Warning: Failed to load scenario {filename}: {e}")

    def _load_from_file(self, filepath: str) -> Optional[ScenarioDefinition]:
        """从JSON文件加载场景"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ScenarioDefinition(
            name=data["name"],
            description=data.get("description", ""),
            duration=data.get("duration", 900),
            comm_scenario=data.get("comm_scenario", "building_blocked"),
            algorithm=data.get("algorithm", "cbba"),
            drones=data.get("drones", []),
            tasks=data.get("tasks", []),
            dynamic_tasks_enabled=data.get("dynamic_tasks_enabled", False),
            events=data.get("events", []),
            semantic_agent=data.get("semantic_agent", {}),
        )

    def get_scenario(self, name: str) -> Optional[ScenarioDefinition]:
        """获取指定名称的场景"""
        return self.scenarios.get(name)

    def list_scenarios(self) -> List[dict]:
        """列出所有可用场景"""
        return [
            {"name": s.name, "description": s.description, "duration": s.duration,
             "num_drones": len(s.drones) or 30, "num_tasks": len(s.tasks) or 100,
             "comm_scenario": s.comm_scenario, "algorithm": s.algorithm}
            for s in self.scenarios.values()
        ]

    @staticmethod
    def create_default() -> 'ScenarioEngine':
        """创建包含默认场景的引擎"""
        engine = ScenarioEngine()

        # 如果无场景文件，创建内存中的默认场景
        if not engine.scenarios:
            engine.scenarios = _create_default_scenarios()

        return engine


def _create_default_scenarios() -> Dict[str, ScenarioDefinition]:
    """创建内置场景（无需 JSON 文件）。"""
    scenarios = {}

    scenarios["single_uav_world_model"] = ScenarioDefinition(
        name="single_uav_world_model",
        description=(
            "单架四旋翼在城市实景中使用局部高度信念、"
            "6-DOF 想象 rollout 与滚动优化自主绕飞"
        ),
        duration=240,
        comm_scenario="ideal",
        algorithm="cbba",
        drones=[
            {
                "id": "WM-UAV-01",
                "drone_type": "standard",
                "start": [0.0, 2.0, 0.0],
                "scripted_path_relative": True,
                "world_model": {
                    "enabled": True,
                    "planning_interval_s": 0.5,
                    "horizon_s": 3.0,
                    "rollout_dt_s": 0.25,
                    "sensor_range_m": 78.0,
                    "sensor_horizontal_fov_deg": 110.0,
                    "collision_clearance_m": 3.0,
                    "comfort_clearance_m": 6.0,
                },
                "scripted_path": [
                    {
                        "offset": [0.0, 15.0, 0.0],
                        "action": "takeoff",
                        "phase": "世界模型垂直起飞",
                    },
                    {
                        "offset": [140.0, 15.0, 0.0],
                        "phase": "跨建筑群自主绕飞",
                    },
                    {
                        "offset": [170.0, 17.0, 45.0],
                        "phase": "滚动预测转弯",
                    },
                    {
                        "offset": [140.0, 15.0, 80.0],
                        "action": "hover",
                        "phase": "目标区悬停",
                    },
                ],
            }
        ],
        tasks=[
            {
                "id": "WORLD-MODEL-FLIGHT-01",
                "task_type": "patrol",
                "pickup": [0.0, 20.0, 0.0],
                "delivery": [140.0, 25.0, 80.0],
                "time_window": [0.0, 240.0],
                "weight": 0.0,
                "reward": 150.0,
                "business_tag": "world_model_closed_loop_acceptance",
            }
        ],
        dynamic_tasks_enabled=False,
    )

    # 单机飞行动力学验收：不依赖任务规划器，使用一条可复现的脚本航线。
    scenarios["single_uav_dynamics"] = ScenarioDefinition(
        name="single_uav_dynamics",
        description="单架标准四旋翼在城市实景中完成起飞、前飞、连续转弯并悬停；用于 RGB-D、6-DOF 动力学和碰撞验收",
        duration=180,
        comm_scenario="ideal",
        algorithm="cbba",
        drones=[
            {
                "id": "UAV-01",
                "drone_type": "standard",
                "start": [0.0, 2.0, 0.0],
                "scripted_path_relative": True,
                "scripted_path": [
                    {"offset": [0.0, 10.0, 0.0], "action": "takeoff", "phase": "垂直起飞"},
                    {"offset": [55.0, 25.0, 0.0], "phase": "向东前飞"},
                    {"offset": [95.0, 30.0, 40.0], "phase": "第一次转弯"},
                    {"offset": [85.0, 28.0, 100.0], "phase": "沿街区前飞"},
                    {"offset": [25.0, 24.0, 115.0], "phase": "第二次转弯"},
                    {"offset": [-15.0, 20.0, 55.0], "phase": "返航段"},
                    {"offset": [0.0, 18.0, 20.0], "action": "hover", "phase": "定点悬停"},
                ],
            }
        ],
        tasks=[
            {
                "id": "FLIGHT-ACCEPTANCE-01",
                "task_type": "patrol",
                "pickup": [0.0, 12.0, 0.0],
                "delivery": [0.0, 20.0, 20.0],
                "time_window": [0.0, 180.0],
                "weight": 0.0,
                "reward": 100.0,
                "business_tag": "flight_dynamics_acceptance",
            }
        ],
        dynamic_tasks_enabled=False,
    )

    # 场景1：理想通信
    scenarios["normal"] = ScenarioDefinition(
        name="normal",
        description="理想通信条件下的标准配送场景：30架无人机执行100个任务",
        duration=900,
        comm_scenario="ideal",
        algorithm="cbba",
        drones=[],
        tasks=[],
        dynamic_tasks_enabled=False,
    )

    # 场景2：通信受阻
    scenarios["comm_constrained"] = ScenarioDefinition(
        name="comm_constrained",
        description="城市建筑遮挡下的受限通信：测试CBBA在通信图稀疏时的表现",
        duration=900,
        comm_scenario="building_blocked",
        algorithm="cbba",
        drones=[],
        tasks=[],
        dynamic_tasks_enabled=False,
    )

    # 场景3：动态任务
    scenarios["dynamic"] = ScenarioDefinition(
        name="dynamic",
        description="动态任务插入场景：运行中随机生成紧急+常规新任务",
        duration=1200,
        comm_scenario="intermittent",
        algorithm="cbba",
        drones=[],
        tasks=[],
        dynamic_tasks_enabled=True,
    )

    # 场景4：紧急医疗
    scenarios["emergency"] = ScenarioDefinition(
        name="emergency",
        description="大量紧急医疗物资配送：50%任务为P0/P1级",
        duration=600,
        comm_scenario="building_blocked",
        algorithm="cbba",
        drones=[],
        tasks=[],
        dynamic_tasks_enabled=True,
    )

    scenarios["qwen_semantic_fleet"] = ScenarioDefinition(
        name="qwen_semantic_fleet",
        description=(
            "4架无人机在 Helsinki 数字孪生中处理临时障碍、动态禁飞区、"
            "阵风和单机故障；语义慢环只产生受门控的任务/路径约束"
        ),
        duration=300,
        comm_scenario="building_blocked",
        algorithm="hungarian",
        drones=[
            {"id": "UAV-SCOUT", "drone_type": "light", "start": [-260, 75, -140]},
            {"id": "UAV-ALPHA", "drone_type": "standard", "start": [-220, 80, 130]},
            {"id": "UAV-BRAVO", "drone_type": "standard", "start": [180, 85, -120]},
            {"id": "UAV-RESERVE", "drone_type": "standard", "start": [240, 75, 150]},
        ],
        tasks=[
            {"id": "INSPECT-WEST", "task_type": "patrol", "pickup": [-170, 80, -80], "delivery": [40, 85, -40], "time_window": [0, 300], "weight": 1.0, "reward": 180},
            {"id": "MEDICAL-NORTH", "task_type": "medical", "pickup": [-80, 75, 160], "delivery": [180, 85, 190], "time_window": [0, 300], "weight": 2.0, "reward": 500, "required_comms": True},
            {"id": "INSPECT-EAST", "task_type": "patrol", "pickup": [130, 85, -150], "delivery": [310, 90, -190], "time_window": [0, 300], "weight": 1.0, "reward": 160},
            {"id": "DELIVERY-SOUTH", "task_type": "regular", "pickup": [70, 75, 50], "delivery": [-190, 80, 170], "time_window": [0, 300], "weight": 4.0, "reward": 220},
        ],
        dynamic_tasks_enabled=False,
        semantic_agent={
            "enabled": True,
            "provider": "deterministic_simulator_cues",
            "max_tasks_per_drone": 1,
            "reallocation_interval_s": 5.0,
            "accept_external_proposals": True,
        },
        events=[
            {"time_s": 20, "observer_drone_id": "UAV-SCOUT", "semantic_event": {"event_id": "evt-crane", "event_type": "temporary_obstacle", "source_drone_id": "UAV-SCOUT", "position": [-20, 85, -55], "radius_m": 26, "confidence": 0.94, "severity": 0.72, "ttl_s": 120, "evidence": "RGB-D crane boom occupies the corridor", "affected_task_ids": ["INSPECT-WEST"]}},
            {"time_s": 45, "observer_drone_id": "UAV-SCOUT", "semantic_event": {"event_id": "evt-nofly", "event_type": "no_fly_zone", "source_drone_id": "UAV-SCOUT", "position": [80, 82, 175], "radius_m": 42, "confidence": 0.99, "severity": 1.0, "ttl_s": 120, "evidence": "authoritative temporary airspace notice", "affected_task_ids": ["MEDICAL-NORTH"]}},
            {"time_s": 75, "observer_drone_id": "UAV-BRAVO", "semantic_event": {"event_id": "evt-gust", "event_type": "weather_hazard", "source_drone_id": "UAV-BRAVO", "position": [170, 85, -160], "radius_m": 70, "confidence": 0.89, "severity": 0.65, "ttl_s": 70, "evidence": "wind estimator corroborated visual motion", "affected_task_ids": ["INSPECT-EAST"]}},
            {"time_s": 55, "observer_drone_id": "UAV-ALPHA", "semantic_event": {"event_id": "evt-failure", "event_type": "drone_failure", "source_drone_id": "UAV-ALPHA", "position": [-40, 80, 120], "radius_m": 1, "confidence": 1.0, "severity": 1.0, "ttl_s": 245, "evidence": "actuator health monitor fault latch"}},
        ],
    )

    return scenarios
