"""
WebSocket 消息协议
=================
定义前后端通信的消息格式和序列化方法。
"""

import json
from typing import Any, Dict, Optional
from enum import Enum


class MessageType(str, Enum):
    """消息类型枚举"""

    # 后端 → 前端
    SCENARIO_START = "scenario_start"
    SCENARIO_LIST = "scenario_list"
    SIM_STATE = "sim_state"
    EVENT = "event"
    SCENARIO_END = "scenario_end"
    ERROR = "error"
    ALGORITHM_LIST = "algorithm_list"

    # 前端 → 后端
    CONTROL = "control"
    SELECT_SCENARIO = "select_scenario"
    SELECT_ALGORITHM = "select_algorithm"
    PING = "ping"
    PONG = "pong"
    POLICY_ACTION = "policy_action"
    POLICY_ACTION_ACK = "policy_action_ack"
    POLICY_SUBSCRIBE = "policy_subscribe"
    POLICY_EPISODE_CONFIG = "policy_episode_config"
    POLICY_EPISODE_ACK = "policy_episode_ack"
    SENSOR_BRIDGE_CONTROL = "sensor_bridge_control"
    SEMANTIC_EVENT_PROPOSAL = "semantic_event_proposal"
    SEMANTIC_EVENT_ACK = "semantic_event_ack"


class ControlAction(str, Enum):
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    SET_SPEED = "set_speed"
    RESET = "reset"
    GET_STATUS = "get_status"


def create_message(msg_type: MessageType | str, payload: Any) -> str:
    """创建JSON消息字符串"""
    return json.dumps({
        "type": msg_type.value if isinstance(msg_type, MessageType) else str(msg_type),
        "payload": payload,
    }, ensure_ascii=False)


def parse_message(message: str) -> Dict:
    """解析JSON消息"""
    try:
        data = json.loads(message)
        if "type" not in data or "payload" not in data:
            return {"type": "error", "payload": {"message": "Invalid message format"}}
        return data
    except json.JSONDecodeError as e:
        return {"type": "error", "payload": {"message": f"JSON parse error: {e}"}}


def create_scenario_start(scenario_name: str, num_drones: int, num_tasks: int,
                          bounds: Dict, algorithm: str) -> str:
    """创建场景启动消息"""
    return create_message(MessageType.SCENARIO_START, {
        "name": scenario_name,
        "drones": num_drones,
        "tasks": num_tasks,
        "bounds": bounds,
        "algorithm": algorithm,
    })


def create_sim_state(state: Dict) -> str:
    """创建仿真状态消息"""
    return create_message(MessageType.SIM_STATE, state)


def create_event(event: Dict) -> str:
    """创建事件消息"""
    return create_message(MessageType.EVENT, event)


def create_scenario_end(summary: Dict) -> str:
    """创建场景结束消息"""
    return create_message(MessageType.SCENARIO_END, summary)


def create_scenario_list(scenarios: list) -> str:
    """创建场景列表消息"""
    return create_message(MessageType.SCENARIO_LIST, scenarios)


def create_algorithm_list(algorithms: list) -> str:
    """创建算法列表消息"""
    return create_message(MessageType.ALGORITHM_LIST, algorithms)


def create_error(message: str) -> str:
    """创建错误消息"""
    return create_message(MessageType.ERROR, {"message": message})
