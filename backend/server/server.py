"""
WebSocket + HTTP 服务器
======================
基于 aiohttp 的异步服务器，负责：
- 提供前端静态文件服务
- WebSocket 连接管理
- 仿真命令转发
- 状态广播
"""

import asyncio
import json
import sys
import os
import hashlib
import re
import time
import math
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aiohttp import web
import aiohttp

from .protocol import (
    parse_message, MessageType, ControlAction,
    create_scenario_start, create_sim_state, create_event,
    create_scenario_end, create_scenario_list, create_algorithm_list,
    create_error, create_message,
)
from .runtime_metrics import RuntimeMetrics
from .semantic_nodes import SemanticNodeStore
from ..config import SERVER


def _runtime_root() -> Path:
    """Return the source checkout or packaged release root.

    Packaged backends are frozen into ``bin/UrbanFly.Backend`` and therefore
    cannot resolve runtime assets relative to ``__file__``.  The desktop and
    Linux launchers set ``URBANFLY_ROOT`` explicitly; source checkouts retain
    the historical relative-path fallback.
    """

    configured = os.environ.get("URBANFLY_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


class SimulationServer:
    """
    UrbanFly 仿真服务器。

    管理 HTTP 路由和 WebSocket 连接池。
    """

    def __init__(self, simulator=None, scenario_engine=None,
                 host: str = None, port: int = None):
        self.host = host or SERVER["host"]
        self.port = port or SERVER["port"]
        self.simulator = simulator
        self.scenario_engine = scenario_engine

        # WebSocket 客户端池
        self._clients: set = set()
        self._client_send_locks: dict = {}
        self._client_latest_state: dict = {}
        self._client_state_tasks: dict = {}
        self._client_runtime: dict = {}
        self._policy_clients: set = set()
        self._lockstep_policy_clients: set = set()
        self._lockstep_capture_after_sim_time: float | None = None
        self._latest_sensor_packet: bytes | None = None
        self._capture_started_wall: dict[tuple[str, float], float] = {}
        self._sim_loop_task = None
        self.metrics = RuntimeMetrics()

        self.runtime_root = _runtime_root()
        configured_static = os.environ.get("URBANFLY_STATIC_DIR")
        self.static_dir = str(
            Path(configured_static).expanduser().resolve()
            if configured_static
            else self.runtime_root / "frontend" / "dist"
        )

        self.recording_dir = self.runtime_root / "outputs" / "runtime_recordings"
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_node_store = SemanticNodeStore(
            self.runtime_root / "data" / "semantic_annotations" / "helsinki_business_nodes.json"
        )
        self.app = web.Application(client_max_size=2 * 1024 ** 3)
        self._setup_routes()

    def _setup_routes(self):
        """设置HTTP路由"""
        @web.middleware
        async def browser_isolation_middleware(request, handler):
            response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Range"
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
            response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
            if request.method == "OPTIONS":
                return web.Response(status=204)
            return response

        self.app.middlewares.append(browser_isolation_middleware)

        # WebSocket 端点
        self.app.router.add_get("/ws", self._ws_handler)
        self.app.router.add_get("/api/health", self._health_handler)
        self.app.router.add_get("/api/semantic-nodes", self._semantic_nodes_get_handler)
        self.app.router.add_put("/api/semantic-nodes", self._semantic_nodes_put_handler)
        self.app.router.add_post("/api/runtime-recordings/{recording_id}", self._recording_upload_handler)
        self.app.router.add_get("/", self._index_handler)

        # 静态文件服务 — 场景数据（glTF, npz等）
        data_dir = str(self.runtime_root / "data" / "scene")
        if os.path.isdir(data_dir):
            self.app.router.add_static("/data/scene/", data_dir)
            print(f"[Server] Serving scene data from: {data_dir}")

        # CityGS assets are intentionally served separately from collision proxies.
        citygs_dir = str(self.runtime_root / "data" / "citygs_visualization")
        if os.path.isdir(citygs_dir):
            self.app.router.add_static("/data/citygs_visualization/", citygs_dir)
            print(f"[Server] Serving CityGS data from: {citygs_dir}")

        collision_dir = str(self.runtime_root / "data" / "citygs_collision")
        if os.path.isdir(collision_dir):
            self.app.router.add_static("/data/citygs_collision/", collision_dir)
            print(f"[Server] Serving CityGS collision proxies from: {collision_dir}")

        helsinki_mesh_dir = str(self.runtime_root / "data" / "helsinki_mesh")
        if os.path.isdir(helsinki_mesh_dir):
            self.app.router.add_static("/data/helsinki_mesh/", helsinki_mesh_dir)
            print(f"[Server] Serving city mesh data from: {helsinki_mesh_dir}")

        # 静态文件服务 — 前端（通配前缀必须最后注册）
        frontend_dir = os.path.abspath(self.static_dir)
        if os.path.isdir(frontend_dir):
            self.app.router.add_static("/", frontend_dir, show_index=True)

    async def _index_handler(self, request):
        """返回构建后的前端入口页。"""
        index_path = os.path.abspath(os.path.join(self.static_dir, "index.html"))
        if not os.path.isfile(index_path):
            raise web.HTTPNotFound(text="Frontend build not found")
        return web.FileResponse(index_path)

    async def _health_handler(self, request):
        """Return a bounded local observability snapshot for the operator UI."""
        snapshot = self.metrics.snapshot()
        simulator_state = getattr(self.simulator, "state", "unavailable")
        simulator_time = getattr(self.simulator, "time", None)
        return web.json_response(
            {
                "schema": "urbanfly-runtime-health-v1",
                "status": "ok",
                "server_time_unix_s": time.time(),
                "simulator": {
                    "state": simulator_state,
                    "sim_time_s": float(simulator_time) if simulator_time is not None else None,
                    "loop_running": bool(
                        self._sim_loop_task is not None
                        and not self._sim_loop_task.done()
                    ),
                },
                "clients": {
                    "total": len(self._clients),
                    "policy": len(self._policy_clients),
                    "lockstep_policy": len(self._lockstep_policy_clients),
                    "state_drains": sum(
                        1 for task in self._client_state_tasks.values()
                        if task is not None and not task.done()
                    ),
                },
                "metrics": snapshot,
                "surfaces": [
                    {**entry[1], "age_s": round(time.perf_counter() - entry[0], 3)}
                    for entry in self._client_runtime.values()
                ],
            }
        )

    async def _semantic_nodes_get_handler(self, request):
        try:
            return web.json_response(self.semantic_node_store.load())
        except ValueError as error:
            raise web.HTTPInternalServerError(text=str(error)) from error

    async def _semantic_nodes_put_handler(self, request):
        try:
            document = await request.json()
            saved = self.semantic_node_store.save(document)
        except (json.JSONDecodeError, ValueError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(saved)

    async def _recording_upload_handler(self, request):
        recording_id = str(request.match_info["recording_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", recording_id):
            raise web.HTTPBadRequest(text="invalid recording id")
        final_path = (self.recording_dir / f"{recording_id}.webm").resolve()
        if final_path.parent != self.recording_dir.resolve() or final_path.exists():
            raise web.HTTPConflict(text="recording already exists")
        partial = final_path.with_suffix(".webm.partial")
        digest = hashlib.sha256(); size = 0
        try:
            with partial.open("wb") as handle:
                async for chunk in request.content.iter_chunked(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024 ** 3:
                        raise web.HTTPRequestEntityTooLarge(max_size=2 * 1024 ** 3, actual_size=size)
                    digest.update(chunk); handle.write(chunk)
                handle.flush(); os.fsync(handle.fileno())
            if size < 1024:
                raise web.HTTPBadRequest(text="recording is empty")
            os.replace(partial, final_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        metadata = {
            "schema": "urbanfly-runtime-recording-v1",
            "recording_id": recording_id, "video_path": str(final_path),
            "bytes": size, "sha256": digest.hexdigest(),
            "mime_type": request.content_type,
            "target_fps": float(request.headers.get("X-Target-Fps", 30)),
            "sim_start_s": float(request.headers.get("X-Sim-Start-S", 0)),
            "sim_end_s": float(request.headers.get("X-Sim-End-S", 0)),
            "wall_duration_s": float(request.headers.get("X-Wall-Duration-S", 0)),
            "source": request.headers.get("X-Source", "unknown"),
            "screenshot_stitching": False,
        }
        manifest = final_path.with_suffix(".manifest.json")
        temporary = manifest.with_suffix(".json.partial")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, manifest)
        return web.json_response({**metadata, "manifest_path": str(manifest)})

    # ==================================================================
    # WebSocket 处理
    # ==================================================================

    async def _ws_handler(self, request):
        """WebSocket 连接处理器"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._clients.add(ws)
        self._client_send_locks[ws] = asyncio.Lock()
        self.metrics.increment("websocket_connections_total")
        print(f"[Server] Client connected. Total clients: {len(self._clients)}")

        try:
            # 发送初始信息
            if self.scenario_engine:
                scenarios = self.scenario_engine.list_scenarios()
                await self._send_to_client(ws, create_scenario_list(scenarios))

                algorithms = [
                    {"id": "cbba", "name": "CBBA 改进版", "type": "去中心化"},
                    {"id": "hungarian", "name": "匈牙利算法", "type": "集中式"},
                    {"id": "greedy", "name": "贪心最近邻", "type": "去中心化"},
                    {"id": "auction", "name": "拍卖算法", "type": "去中心化"},
                    {"id": "genetic", "name": "遗传算法", "type": "集中式"},
                    {"id": "market", "name": "市场机制", "type": "去中心化"},
                ]
                await self._send_to_client(ws, create_algorithm_list(algorithms))

            # 消息循环
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_sensor_packet(ws, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[Server] WS error: {ws.exception()}")

        finally:
            was_policy_client = ws in self._policy_clients
            self._drop_client(ws)
            print(f"[Server] Client disconnected. Total clients: {len(self._clients)}")
            if was_policy_client and not self._policy_clients:
                self._lockstep_capture_after_sim_time = None
                if self.simulator is not None and self.simulator.state == "paused":
                    self.simulator.play()
                await self._broadcast(
                    create_message(
                        MessageType.SENSOR_BRIDGE_CONTROL,
                        {"enabled": False},
                    )
                )

        return ws

    async def _handle_message(self, ws, raw_msg: str):
        """处理客户端消息"""
        data = parse_message(raw_msg)
        msg_type = data.get("type", "")
        payload = data.get("payload", {})

        if msg_type == "control":
            await self._handle_control(ws, payload)
        elif msg_type == "select_scenario":
            await self._handle_select_scenario(ws, payload)
        elif msg_type == "select_algorithm":
            await self._handle_select_algorithm(ws, payload)
        elif msg_type == "policy_action":
            await self._handle_policy_action(ws, payload)
        elif msg_type == "policy_visualization":
            await self._handle_policy_visualization(ws, payload)
        elif msg_type == "policy_episode_config":
            await self._handle_policy_episode_config(ws, payload)
        elif msg_type == "policy_subscribe":
            self._policy_clients.add(ws)
            if bool(payload.get("lockstep", False)):
                self._lockstep_policy_clients.add(ws)
            await self._broadcast(
                create_message(
                    MessageType.SENSOR_BRIDGE_CONTROL,
                    {"enabled": True},
                )
            )
        elif msg_type == "semantic_event_proposal":
            await self._handle_semantic_event_proposal(ws, payload)
        elif msg_type == "sensor_capture_started":
            capture_time = float(payload.get("sim_time", -1.0))
            vehicle_name = str(payload.get("vehicle_name", ""))
            self._capture_started_wall[(vehicle_name, round(capture_time, 6))] = time.perf_counter()
            # Bound at insertion even if a failed renderer never sends a packet.
            while len(self._capture_started_wall) > 256:
                self._capture_started_wall.pop(next(iter(self._capture_started_wall)))
            self.metrics.increment("sensor_capture_started_total")
            if self._lockstep_policy_clients and self.simulator is not None:
                threshold = self._lockstep_capture_after_sim_time
                is_current = abs(capture_time - float(self.simulator.time)) <= 0.11
                if is_current and (threshold is None or capture_time + 1e-9 >= threshold):
                    self.simulator.pause()
                    # Ignore duplicate notifications from the captured state
                    # until the next accepted policy action rearms lockstep.
                    self._lockstep_capture_after_sim_time = float("inf")
        elif msg_type == "runtime_recording_control":
            await self._broadcast(create_message("runtime_recording_control", payload))
        elif msg_type == "runtime_recording_started":
            await self._broadcast(create_message("runtime_recording_started", payload))
        elif msg_type == "runtime_recording_complete":
            await self._broadcast(create_message("runtime_recording_ack", payload))
        elif msg_type == "runtime_recording_failed":
            await self._broadcast(create_message("runtime_recording_failed", payload))
        elif msg_type == "runtime_client_status":
            self._update_client_runtime(ws, payload)
        elif msg_type == "ping":
            await self._send_to_client(
                ws,
                json.dumps(
                    {
                        "type": "pong",
                        "payload": {
                            "client_time_ms": payload.get("client_time_ms"),
                            "server_time_unix_ms": time.time() * 1000.0,
                        },
                    }
                ),
            )
        else:
            await self._send_to_client(
                ws,
                create_error(f"Unknown message type: {msg_type}"),
            )

    def _update_client_runtime(self, ws, payload):
        """Accept bounded diagnostics, never arbitrary client state or control."""
        if ws not in self._clients or not isinstance(payload, dict):
            return
        snapshot = {
            "surface": "desktop" if payload.get("surface") == "desktop" else "browser",
            "scene_ready": payload.get("scene_ready") is True,
            "hidden": payload.get("hidden") is True,
        }
        for section, keys in {
            "presentation": (
                "target_fps", "fps", "render_submission_p95_ms", "frames_in_window",
                "skipped_frames", "pixel_ratio", "geometries", "textures", "draw_calls",
            ),
            "sensors": (
                "captures", "latest_capture_ms", "bridge_frames", "bridge_dropped_frames",
                "bridge_skipped_busy", "latest_bridge_encode_ms", "last_bridge_sim_time",
            ),
        }.items():
            values = payload.get(section)
            values = values if isinstance(values, dict) else {}
            snapshot[section] = {
                key: value for key in keys
                if isinstance((value := values.get(key)), (int, float))
                and not isinstance(value, bool) and 0 <= value <= 1e15
                and math.isfinite(value)
            }
        sensors = payload.get("sensors")
        sensors = sensors if isinstance(sensors, dict) else {}
        snapshot["sensors"]["bridge_enabled"] = sensors.get("bridge_enabled") is True
        snapshot["sensors"]["streaming"] = sensors.get("streaming") is True
        presentation = payload.get("presentation")
        snapshot["presentation"]["idle"] = isinstance(presentation, dict) and presentation.get("idle") is True
        self._client_runtime[ws] = (time.perf_counter(), snapshot)

    async def _handle_sensor_packet(self, ws, packet: bytes):
        """Validate and relay one compressed, synchronized RGB-D packet."""
        if len(packet) < 8 or packet[:4] != b"UFWM":
            await self._send_to_client(ws, create_error("Invalid sensor packet magic"))
            return
        header_length = int.from_bytes(packet[4:8], "little")
        if header_length <= 0 or 8 + header_length > len(packet):
            await self._send_to_client(ws, create_error("Invalid sensor packet header"))
            return
        if len(packet) > 8 * 1024 * 1024:
            await self._send_to_client(ws, create_error("Sensor packet exceeds 8 MiB"))
            return
        try:
            header = json.loads(packet[8:8 + header_length].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            await self._send_to_client(ws, create_error("Malformed sensor packet JSON"))
            return
        if header.get("schema") != "urbanfly-sensor-packet-v2":
            await self._send_to_client(ws, create_error("Unsupported sensor schema"))
            return
        expected = (
            8
            + header_length
            + int(header.get("rgb_length", -1))
            + int(header.get("depth_length", -1))
        )
        if expected != len(packet):
            await self._send_to_client(ws, create_error("Sensor packet length mismatch"))
            return
        self._latest_sensor_packet = bytes(packet)
        self.metrics.increment("sensor_packets_total")
        self.metrics.increment("sensor_packet_bytes_total", len(packet))
        capture_key = (
            str(header.get("vehicle_name", "")),
            round(float(header.get("sim_time", -1.0)), 6),
        )
        capture_started = self._capture_started_wall.pop(capture_key, None)
        if capture_started is not None:
            self.metrics.observe(
                "sensor_capture_to_packet_ms",
                (time.perf_counter() - capture_started) * 1000.0,
            )
        if len(self._capture_started_wall) > 256:
            oldest = sorted(self._capture_started_wall.items(), key=lambda item: item[1])[:64]
            for key, _ in oldest:
                self._capture_started_wall.pop(key, None)
        if self._policy_clients:
            await asyncio.gather(
                *(
                    self._send_bytes_to_client(client, packet)
                    for client in list(self._policy_clients)
                    if client is not ws
                )
            )

    async def _handle_semantic_event_proposal(self, ws, payload: dict):
        """Accept VLM semantic proposals only through the deterministic gate."""

        if self.simulator is None:
            await self._send_to_client(ws, create_error("No simulator instance"))
            return
        try:
            proposals = payload.get("events", [])
            if not isinstance(proposals, list):
                raise ValueError("semantic events must be a list")
            decisions = self.simulator.semantic_fleet_bridge.ingest_external_proposals(
                self.simulator,
                observer_drone_id=str(payload.get("observer_drone_id", "")),
                observation_timestamp_s=float(payload["observation_timestamp_s"]),
                proposals=proposals,
            )
            self.metrics.increment("semantic_proposals_total", len(proposals))
            self.metrics.increment(
                "semantic_proposals_accepted_total",
                sum(bool(item["accepted"]) for item in decisions),
            )
            await self._send_to_client(
                ws,
                create_message(
                    MessageType.SEMANTIC_EVENT_ACK,
                    {
                        "observation_timestamp_s": float(
                            payload["observation_timestamp_s"]
                        ),
                        "decisions": decisions,
                    },
                ),
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            await self._send_to_client(
                ws, create_error(f"Semantic proposal rejected: {error}")
            )

    async def _handle_policy_action(self, ws, payload: dict):
        """Accept an explicit learned-policy action without hidden fallback."""
        if self.simulator is None:
            await self._send_to_client(ws, create_error("No simulator instance"))
            return
        try:
            accepted = self.simulator.set_external_policy_action(
                str(payload.get("drone_id", "UAV-01")),
                payload.get("action_normalized"),
                step_id=int(payload.get("step_id", -1)),
                policy_family=str(payload.get("policy_family", "unknown_policy")),
                inference_latency_ms=float(payload.get("inference_latency_ms", 0.0)),
                predicted_risk=float(payload.get("predicted_risk", 0.0)),
                shield_enabled=bool(payload.get("shield_enabled", True)),
                timeout_s=float(payload.get("timeout_s", 0.45)),
            )
        except (TypeError, ValueError) as error:
            await self._send_to_client(ws, create_error(str(error)))
            return
        await self._send_to_client(
            ws,
            create_message(
                MessageType.POLICY_ACTION_ACK,
                {
                    "drone_id": accepted["drone_id"],
                    "step_id": accepted["step_id"],
                    "accepted_sim_time": accepted["accepted_sim_time"],
                    "valid_until_sim_time": accepted["valid_until_sim_time"],
                },
            ),
        )
        if ws in self._lockstep_policy_clients:
            duration_s = max(0.01, float(payload.get("duration_s", 0.1)))
            self._lockstep_capture_after_sim_time = (
                float(accepted["accepted_sim_time"]) + 0.8 * duration_s
            )
            self.simulator.play()

    async def _handle_policy_visualization(self, ws, payload: dict):
        """Accept display-only candidate telemetry from the real planner."""
        if self.simulator is None:
            await self._send_to_client(ws, create_error("No simulator instance"))
            return
        try:
            accepted = self.simulator.set_external_policy_visualization(
                str(payload.get("drone_id", "WM-UAV-01")), payload
            )
        except (TypeError, ValueError) as error:
            await self._send_to_client(ws, create_error(str(error)))
            return
        await self._send_to_client(
            ws,
            create_message(
                "policy_visualization_ack",
                {
                    "drone_id": str(payload.get("drone_id", "WM-UAV-01")),
                    "decision_sequence": accepted["decision_sequence"],
                },
            ),
        )

    async def _handle_policy_episode_config(self, ws, payload: dict):
        """Reset the single-aircraft policy episode without a hidden planner."""
        if self.simulator is None:
            await self._send_to_client(ws, create_error("No simulator instance"))
            return
        try:
            configured = self.simulator.configure_external_policy_episode(
                str(payload.get("drone_id", "WM-UAV-01")),
                start_world_m=payload.get("start_world_m"),
                goal_world_m=payload.get("goal_world_m"),
                yaw_degrees=float(payload.get("yaw_degrees", 0.0)),
                policy_family=str(payload.get("policy_family", "external_policy")),
                shield_enabled=bool(payload.get("shield_enabled", True)),
                episode_seed=int(payload.get("episode_seed", 20260731)),
                dynamic_actor_density=float(payload.get("dynamic_actor_density", 1.0)),
                appearance_perturbation=payload.get("appearance_perturbation"),
                dynamics_perturbation=payload.get("dynamics_perturbation"),
                episode_duration_s=payload.get("episode_duration_s"),
            )
        except (TypeError, ValueError) as error:
            await self._send_to_client(ws, create_error(str(error)))
            return
        await self._send_to_client(
            ws,
            create_message(MessageType.POLICY_EPISODE_ACK, configured),
        )
        if ws in self._lockstep_policy_clients:
            self._lockstep_capture_after_sim_time = float(configured["sim_time"]) + 0.05
            self.simulator.play()

    async def _handle_control(self, ws, payload: dict):
        """处理仿真控制命令"""
        if self.simulator is None:
            await self._send_to_client(ws, create_error("No simulator instance"))
            return

        action = payload.get("action", "")

        if action == ControlAction.PLAY:
            self.simulator.play()
            self._start_sim_loop()
        elif action == ControlAction.PAUSE:
            self.simulator.pause()
            self._stop_sim_loop()
        elif action == ControlAction.STOP:
            self.simulator.stop()
            self._stop_sim_loop()
        elif action == ControlAction.SET_SPEED:
            speed = float(payload.get("value", 1.0))
            self.simulator.set_speed(speed)
        elif action == ControlAction.GET_STATUS:
            if self.simulator:
                state = self.simulator.get_state_snapshot()
                await self._send_to_client(ws, create_sim_state(state))

    async def _handle_select_scenario(self, ws, payload: dict):
        """处理场景选择"""
        if self.scenario_engine is None or self.simulator is None:
            await self._send_to_client(
                ws,
                create_error("Server not fully initialized"),
            )
            return

        scenario_name = payload.get("name", "")
        scenario = self.scenario_engine.get_scenario(scenario_name)

        if scenario is None:
            await self._send_to_client(
                ws,
                create_error(f"Unknown scenario: {scenario_name}"),
            )
            return

        # 初始化仿真
        self._stop_sim_loop()
        self.simulator.initialize_scenario(scenario)

        # 通知前端
        bounds = {
            "center": self.simulator.scene_config.bounds_center.tolist() if self.simulator.scene_config else [0, 0, 0],
            "size": self.simulator.scene_config.bounds_size.tolist() if self.simulator.scene_config else [857, 54, 944],
        }
        await self._send_to_client(
            ws,
            create_scenario_start(
                scenario.name,
                len(self.simulator.drones),
                len(self.simulator.tasks),
                bounds,
                scenario.algorithm,
            ),
        )
        await self._send_to_client(
            ws,
            create_sim_state(self.simulator.get_state_snapshot()),
        )
        self._start_sim_loop()

    async def _handle_select_algorithm(self, ws, payload: dict):
        """处理算法切换"""
        if self.simulator is None:
            await self._send_to_client(ws, create_error("No simulator instance"))
            return

        algorithm = payload.get("algorithm", "cbba")
        success = self.simulator.select_algorithm(algorithm)

        if success:
            await self._send_to_client(
                ws,
                json.dumps({
                    "type": "algorithm_changed",
                    "payload": {"algorithm": algorithm}
                }),
            )
        else:
            await self._send_to_client(
                ws,
                create_error(f"Unknown algorithm: {algorithm}"),
            )

    # ==================================================================
    # 仿真循环
    # ==================================================================

    def _start_sim_loop(self):
        """启动仿真主循环（异步任务）"""
        if self._sim_loop_task is None or self._sim_loop_task.done():
            self._sim_loop_task = asyncio.create_task(self._sim_loop())

    def _stop_sim_loop(self):
        """停止仿真主循环"""
        task = self._sim_loop_task
        self._sim_loop_task = None
        if task and not task.done():
            task.cancel()

    async def _sim_loop(self):
        """
        仿真主循环：
        1. 步进仿真
        2. 收集状态快照
        3. 广播给所有连接的客户端
        4. 检测仿真结束
        """
        try:
            while True:
                if self.simulator is None or self.simulator.state != "running":
                    await asyncio.sleep(0.1)
                    continue

                wall_step_started = time.perf_counter()
                # 步进仿真
                snapshot = self.simulator.step()
                self.metrics.increment("sim_steps_total")
                self.metrics.observe(
                    "sim_step_ms",
                    (time.perf_counter() - wall_step_started) * 1000.0,
                )

                if snapshot:
                    # 广播状态
                    msg = create_sim_state(snapshot)
                    await self._broadcast(msg, coalesce_state=True)

                # 检查事件
                while self.simulator.events:
                    event = self.simulator.events.pop(0)
                    await self._broadcast(create_event(event.to_dict()))

                # 检查仿真结束
                if self.simulator.state == "completed":
                    summary = self.simulator.stats.to_dict()
                    await self._broadcast(create_scenario_end(summary))
                    break

                # 固定墙钟 20 Hz 调度；speed_multiplier 只改变每步仿真时间，
                # 不再无上限灌入 WebSocket，避免 RGB-D 页面积压陈旧状态帧。
                elapsed = time.perf_counter() - wall_step_started
                await asyncio.sleep(max(0.0, self.simulator.dt - elapsed))

        except asyncio.CancelledError:
            pass

    async def _broadcast(self, message: str, *, coalesce_state: bool = False):
        """Broadcast reliable messages; coalesce replaceable UI state frames."""
        if not self._clients:
            return
        reliable = []
        for ws in list(self._clients):
            if coalesce_state and ws not in self._policy_clients:
                self._queue_latest_state(ws, message)
            else:
                reliable.append(self._send_to_client(ws, message))
        if reliable:
            await asyncio.gather(*reliable)

    def _queue_latest_state(self, ws, message: str):
        """Keep at most one pending state per UI client (latest-value semantics)."""
        if ws not in self._clients:
            return
        if self._client_latest_state.get(ws) is not None:
            self.metrics.increment("sim_state_coalesced_total")
        self._client_latest_state[ws] = message
        task = self._client_state_tasks.get(ws)
        if task is None or task.done():
            self._client_state_tasks[ws] = asyncio.create_task(
                self._drain_latest_state(ws)
            )

    async def _drain_latest_state(self, ws):
        try:
            while ws in self._clients:
                message = self._client_latest_state.pop(ws, None)
                if message is None:
                    break
                await self._send_to_client(ws, message)
                self.metrics.increment("sim_state_sent_total")
        finally:
            current = asyncio.current_task()
            if self._client_state_tasks.get(ws) is current:
                self._client_state_tasks.pop(ws, None)

    def _drop_client(self, ws):
        was_connected = ws in self._clients
        self._policy_clients.discard(ws)
        self._lockstep_policy_clients.discard(ws)
        self._clients.discard(ws)
        self._client_send_locks.pop(ws, None)
        self._client_latest_state.pop(ws, None)
        self._client_runtime.pop(ws, None)
        task = self._client_state_tasks.pop(ws, None)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        if was_connected:
            self.metrics.increment("websocket_disconnects_total")

    async def _send_to_client(self, ws, message: str):
        """Serialize writes per socket while allowing clients to send in parallel."""
        lock = self._client_send_locks.get(ws)
        if lock is None:
            return
        try:
            started = time.perf_counter()
            async with lock:
                await ws.send_str(message)
            self.metrics.increment("websocket_text_messages_sent_total")
            self.metrics.increment("websocket_text_bytes_sent_total", len(message.encode("utf-8")))
            self.metrics.observe("websocket_text_send_ms", (time.perf_counter() - started) * 1000.0)
        except Exception:
            self.metrics.increment("websocket_send_errors_total")
            self._drop_client(ws)

    async def _send_bytes_to_client(self, ws, payload: bytes):
        lock = self._client_send_locks.get(ws)
        if lock is None:
            return
        try:
            started = time.perf_counter()
            async with lock:
                await ws.send_bytes(payload)
            self.metrics.increment("websocket_binary_messages_sent_total")
            self.metrics.increment("websocket_binary_bytes_sent_total", len(payload))
            self.metrics.observe("websocket_binary_send_ms", (time.perf_counter() - started) * 1000.0)
        except Exception:
            self.metrics.increment("websocket_send_errors_total")
            self._drop_client(ws)

    # ==================================================================
    # 启动/停止
    # ==================================================================

    def run(self):
        """启动服务器（阻塞）"""
        print(f"[Server] Starting UrbanFly server on http://{self.host}:{self.port}")
        print(f"[Server] WebSocket endpoint: ws://{self.host}:{self.port}/ws")
        print(f"[Server] Static files: {self.static_dir}")

        web.run_app(self.app, host=self.host, port=self.port)

    def get_app(self):
        """获取 aiohttp Application 实例（用于测试或嵌入）"""
        return self.app


# ==================================================================
# 独立运行入口
# ==================================================================

def main():
    """独立运行服务器（用于开发/测试）"""
    import json
    import numpy as np

    from ..engine.simulator import Simulator
    from ..engine.scenario import ScenarioEngine
    from ..engine.models import SceneConfig, BuildingInfo, BlockInfo
    from ..engine.planner import OccupancyGrid, PathPlanner
    from ..engine.helsinki_navigation import HelsinkiNavigationStack
    from ..engine.collision import (
        DenseSignedDistanceField,
        HeightmapStaticCollisionMap,
        HierarchicalStaticCollisionMap,
        SparseStaticCollisionMap,
    )

    # 加载场景数据
    repo_data_dir = str(_runtime_root() / "data")
    citygs_proxy_dir = os.path.join(repo_data_dir, "citygs_collision", "Residence")
    fallback_scene_dir = os.path.join(repo_data_dir, "scene")
    data_dir = os.environ.get(
        "URBANFLY_SCENE_DIR",
        citygs_proxy_dir if os.path.isfile(os.path.join(citygs_proxy_dir, "scene_config.json"))
        else fallback_scene_dir,
    )

    scene_config = None
    planner = None
    static_collision_map = None

    try:
        with open(os.path.join(data_dir, "scene_config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(data_dir, "buildings.json")) as f:
            buildings_data = json.load(f)
        road_network_path = os.path.join(data_dir, "road_network.json")
        road_network = {}
        if os.path.exists(road_network_path):
            with open(road_network_path, encoding="utf-8") as f:
                road_network = json.load(f)

        buildings = [
            BuildingInfo(
                id=b["id"],
                original_group=b["original_group"],
                bounds_min=np.array(b["bounds_min"]),
                bounds_max=np.array(b["bounds_max"]),
                num_faces_original=b["num_faces_original"],
            )
            for b in buildings_data
        ]
        blocks = [
            BlockInfo(
                id=b["id"],
                name=b["name"],
                district=b.get("district", "mixed"),
                polygon=np.array(b.get("polygon", []), dtype=float),
                area=float(b.get("area", 0.0)),
                metadata={
                    "num_buildings": b.get("num_buildings", 0),
                    "buildings": b.get("buildings", []),
                    **b.get("metadata", {}),
                },
            )
            for b in road_network.get("blocks", [])
        ]

        if "bounds_center" in cfg and "bounds_size" in cfg:
            bounds_center = np.asarray(cfg["bounds_center"], dtype=float)
            bounds_size = np.asarray(cfg["bounds_size"], dtype=float)
        else:
            bounds_min = np.min([b.bounds_min for b in buildings], axis=0)
            bounds_max = np.max([b.bounds_max for b in buildings], axis=0)
            bounds_center = (bounds_min + bounds_max) / 2
            bounds_size = bounds_max - bounds_min

        scene_config = SceneConfig(
            name=cfg.get("name", "UrbanFly City"),
            bounds_center=bounds_center,
            bounds_size=bounds_size,
            buildings=buildings,
            blocks=blocks,
            grid_resolution=float(cfg.get("grid_resolution", 5.0)),
            metadata={
                "district_distribution": cfg.get("district_distribution", {}),
                "layout": "citygs_metric_proxy" if cfg.get("visual_asset") else cfg.get("layout", "hybrid"),
                "roads": road_network.get("roads", []),
                "visual_asset": cfg.get("visual_asset"),
                "coordinate_frame": cfg.get("coordinate_frame"),
                "proxy_stats": cfg.get("proxy_stats", {}),
                "default_task_count": cfg.get("default_task_count"),
            },
        )

        # 加载占据网格
        grid_data = np.load(os.path.join(data_dir, "occupancy_grid.npz"))
        hm_data = np.load(os.path.join(data_dir, "heightmap.npz"))

        occ_grid = OccupancyGrid(
            grid=grid_data["grid"],
            origin=grid_data["origin"],
            resolution=float(grid_data["resolution"]),
            heightmap=hm_data["heightmap"],
            buildings=buildings,
        )

        planner = PathPlanner(
            occ_grid,
            fast_heightmap_mode=bool(cfg.get("visual_asset")),
        )
        local_collision_path = os.path.join(data_dir, "local_collision_sparse.npz")
        if os.path.isfile(local_collision_path):
            local_collision_map = SparseStaticCollisionMap.load(local_collision_path)
            print(
                f"[Server] Loaded local collision layer: "
                f"{local_collision_map.voxel_count:,} voxels @ "
                f"{local_collision_map.resolution:.2f} m"
            )
            global_esdf_path = os.path.join(data_dir, "global_esdf.npz")
            if os.path.isfile(global_esdf_path):
                global_esdf = DenseSignedDistanceField.load(global_esdf_path)
                static_collision_map = HierarchicalStaticCollisionMap(
                    global_esdf,
                    local_collision_map,
                )
                scene_config.metadata["collision_field"] = {
                    "type": "hierarchical_esdf",
                    "global_resolution_m": global_esdf.resolution,
                    "global_shape": list(global_esdf.shape),
                    "local_resolution_m": local_collision_map.resolution,
                    "local_voxels": local_collision_map.voxel_count,
                    "sweep_step_m": min(local_collision_map.resolution * 0.5, 0.25),
                }
                print(
                    f"[Server] Loaded hierarchical collision field: "
                    f"{global_esdf.shape} @ {global_esdf.resolution:.2f} m global + "
                    f"{local_collision_map.resolution:.2f} m local"
                )
            else:
                static_collision_map = local_collision_map
        print(f"[Server] Loaded scene: {len(buildings)} buildings, "
              f"grid {occ_grid.shape}")

    except Exception as e:
        print(f"[Server] Warning: Could not load scene data: {e}")
        print("[Server] Running without scene — drones will use straight-line paths")

    # 默认将物理世界切换到当前前端使用的城市 1 km 实景坐标系。
    # 高度图来自同一份 L18 摄影测量碰撞网格，不再让可视城市与旧 CityGS
    # 代理碰撞场错位。设置 URBANFLY_USE_HELSINKI=0 可回退到旧实验场。
    helsinki_dir = os.path.join(
        repo_data_dir,
        "helsinki_mesh",
        "HelsinkiCentral1km",
    )
    helsinki_manifest_path = os.path.join(helsinki_dir, "manifest.json")
    use_helsinki = os.environ.get("URBANFLY_USE_HELSINKI", "1") != "0"
    if use_helsinki and os.path.isfile(helsinki_manifest_path):
        try:
            with open(helsinki_manifest_path, encoding="utf-8") as file:
                helsinki_manifest = json.load(file)
            heightmap_path = os.path.join(
                helsinki_dir,
                helsinki_manifest["collision"]["heightmap"]["uri"],
            )
            helsinki_navigation = HelsinkiNavigationStack.load(helsinki_dir)
            static_collision_map = helsinki_navigation.collision_map
            bounds = helsinki_manifest["collision"]["bounds"]
            minimum = np.asarray(bounds["minimum"], dtype=float)
            maximum = np.asarray(bounds["maximum"], dtype=float)
            scene_config = SceneConfig(
                name="CityCentral1km",
                bounds_center=(minimum + maximum) / 2.0,
                bounds_size=maximum - minimum,
                buildings=[],
                blocks=[],
                grid_resolution=static_collision_map.resolution,
                metadata={
                    "layout": "helsinki_photogrammetry_mesh",
                    "visual_asset": "CityCentral1km",
                    "licensed_source_attribution": helsinki_manifest["source"],
                    "coordinate_frame": helsinki_manifest["local_frame"],
                    "collision_field": {
                        "type": "conservative_heightmap_surface",
                        "resolution_m": static_collision_map.resolution,
                        "shape": list(static_collision_map.shape),
                        "source_triangles": helsinki_manifest["collision"]["triangles"],
                        "sweep_step_m": min(
                            static_collision_map.resolution * 0.5,
                            0.25,
                        ),
                    },
                    "global_planner": {
                        "type": "multi_layer_2p5d_astar",
                        "planning_resolution_m": helsinki_navigation.grid.resolution,
                        "drone_radius_m": helsinki_navigation.drone_radius,
                        "safety_margin_m": helsinki_navigation.safety_margin,
                        "fail_closed": True,
                    },
                    "default_task_count": 30,
                },
            )
            planner = helsinki_navigation.global_planner
            print(
                f"[Server] Physical world aligned to {scene_config.name}: "
                f"{static_collision_map.shape} @ "
                f"{static_collision_map.resolution:.2f} m height surface"
            )
            print(
                f"[Server] Helsinki global planner connected: "
                f"{helsinki_navigation.grid.shape} @ "
                f"{helsinki_navigation.grid.resolution:.2f} m, fail-closed"
            )
        except Exception as error:
            print(f"[Server] Warning: city physical world not loaded: {error}")

    # 创建仿真引擎
    simulator = Simulator(
        scene_config=scene_config,
        planner=planner,
        static_collision_map=static_collision_map,
    )
    scenario_engine = ScenarioEngine.create_default()

    server = SimulationServer(
        simulator=simulator,
        scenario_engine=scenario_engine,
    )
    server.run()


if __name__ == "__main__":
    main()
