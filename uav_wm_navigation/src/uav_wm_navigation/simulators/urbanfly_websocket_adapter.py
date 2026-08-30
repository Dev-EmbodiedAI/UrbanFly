from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from typing import Any, Callable

import aiohttp
import numpy as np
from scipy.spatial.transform import Rotation

from uav_wm_navigation.simulators.base import SimulatorAdapter
from uav_wm_navigation.simulators.urbanfly_sensor_packet import (
    DecodedUrbanFlyPacket,
    decode_urbanfly_sensor_packet,
)
from uav_wm_navigation.types import ActorState, SensorFrame, VehicleState


def urbanfly_world_to_nwu(vector: np.ndarray | list[float]) -> np.ndarray:
    """Convert UrbanFly browser world ``[x, up, z]`` to navigation NWU."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("UrbanFly world vector must have shape [3]")
    return value[[0, 2, 1]]


def nwu_to_urbanfly_world(vector: np.ndarray | list[float]) -> np.ndarray:
    """Convert navigation NWU to UrbanFly browser world ``[x, up, z]``."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("NWU vector must have shape [3]")
    return value[[0, 2, 1]]


class UrbanFlyWebSocketAdapter(SimulatorAdapter):
    """Synchronous planner adapter for UrbanFly's browser-rendered RGB-D loop.

    The aiohttp WebSocket client lives on a dedicated event-loop thread.  The
    navigation loop therefore keeps a synchronous adapter contract while
    receiving actual browser RGB-D frames and backend
    6-DOF state.  No controller fallback is implemented in this class.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.url = str(config.get("websocket_url", "ws://127.0.0.1:8765/ws"))
        self.drone_id = str(config.get("vehicle_name", "WM-UAV-01"))
        self.urbanfly_scenario = str(
            config.get("urbanfly_scenario", "single_uav_world_model")
        )
        self.policy_family = str(config.get("policy_family", "yopo_direct"))
        self.shield_enabled = bool(config.get("backend_safety_shield", True))
        self.connection_timeout_s = float(config.get("connection_timeout_s", 10.0))
        self.sensor_timeout_s = float(config.get("sensor_timeout_s", 10.0))
        self.command_timeout_s = float(config.get("command_timeout_s", 3.0))
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._connected = False
        self._closed = False
        self._receiver_exception: BaseException | None = None
        self._last_error: str | None = None
        self._latest_state: dict[str, Any] | None = None
        self._latest_packet: DecodedUrbanFlyPacket | None = None
        self._state_revision = 0
        self._packet_revision = 0
        self._scenario_revision = 0
        self._episode_revision = 0
        self._episode_ack: dict[str, Any] | None = None
        self._action_ack_step = -2
        self._action_ack: dict[str, Any] | None = None
        self._policy_step = 0
        self._episode_id = f"urbanfly-{uuid.uuid4().hex[:12]}"
        self._requested_start_nwu: np.ndarray | None = None
        self._goal_nwu: np.ndarray | None = None
        self._episode_applied = False
        self._last_velocity_nwu = np.zeros(3, dtype=np.float64)
        self._last_velocity_time: float | None = None
        self._last_acceleration_nwu = np.zeros(3, dtype=np.float64)
        self._last_collision_count = 0
        self._last_depth_revision = -1
        self._packet_state_pending = False
        self._jitter_rng = np.random.default_rng(int(config.get("seed", 20260731)))
        self._recording_id: str | None = None
        self._recording_ack: dict[str, Any] | None = None
        self._recording_started = False
        self._latest_planner_latency_ms = 0.0
        self._latest_predicted_risk = 0.0

    def connect(self) -> None:
        with self._condition:
            if self._connected:
                return
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._thread_main,
                    name="urbanfly-websocket-adapter",
                    daemon=True,
                )
                self._thread.start()
            self._wait_locked(lambda: self._connected, self.connection_timeout_s)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._client_main())
        except BaseException as error:  # surfaced to synchronous callers
            with self._condition:
                self._receiver_exception = error
                self._condition.notify_all()
        finally:
            with self._condition:
                self._connected = False
                self._condition.notify_all()
            loop.close()

    async def _client_main(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None, connect=self.connection_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._session = session
            async with session.ws_connect(self.url, heartbeat=20.0) as ws:
                self._ws = ws
                with self._condition:
                    self._connected = True
                    self._condition.notify_all()
                await ws.send_json(
                    {
                        "type": "policy_subscribe",
                        "payload": {
                            "lockstep": bool(self.config.get("policy_lockstep", False))
                        },
                    }
                )
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        self._handle_text(message.data)
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        self._handle_binary(bytes(message.data))
                    elif message.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break

    def _handle_text(self, raw: str) -> None:
        message = json.loads(raw)
        message_type = str(message.get("type", ""))
        payload = message.get("payload", {})
        with self._condition:
            if message_type == "sim_state":
                self._latest_state = payload
                self._state_revision += 1
            elif message_type == "scenario_start":
                self._scenario_revision += 1
            elif message_type == "policy_episode_ack":
                self._episode_ack = dict(payload)
                self._episode_revision += 1
            elif message_type == "policy_action_ack":
                self._action_ack_step = int(payload.get("step_id", -2))
                self._action_ack = dict(payload)
            elif message_type == "error":
                self._last_error = str(payload.get("message", "UrbanFly error"))
            elif message_type == "runtime_recording_ack":
                if payload.get("recording_id") == self._recording_id:
                    self._recording_ack = dict(payload)
            elif message_type == "runtime_recording_started":
                if payload.get("recording_id") == self._recording_id:
                    self._recording_started = True
            elif message_type == "runtime_recording_failed":
                if payload.get("recording_id") == self._recording_id:
                    self._last_error = str(payload.get("error", "browser recording failed"))
            self._condition.notify_all()

    def _handle_binary(self, raw: bytes) -> None:
        decoded = decode_urbanfly_sensor_packet(
            raw,
            episode_id=self._episode_id,
        )
        with self._condition:
            self._latest_packet = decoded
            self._packet_revision += 1
            self._condition.notify_all()

    def _wait_locked(
        self,
        predicate: Callable[[], bool],
        timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if self._last_error:
                error = self._last_error
                self._last_error = None
                raise RuntimeError(error)
            if self._receiver_exception is not None:
                raise RuntimeError("UrbanFly WebSocket receiver stopped") from self._receiver_exception
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"UrbanFly response timed out after {timeout_s:.1f} s")
            self._condition.wait(remaining)

    def _send(self, message_type: str, payload: dict[str, Any]) -> None:
        self.connect()
        if self._loop is None or self._ws is None:
            raise RuntimeError("UrbanFly WebSocket is not connected")
        with self._condition:
            self._last_error = None
        future = asyncio.run_coroutine_threadsafe(
            self._ws.send_json({"type": message_type, "payload": payload}),
            self._loop,
        )
        future.result(timeout=self.command_timeout_s)

    def _select_scenario(self) -> None:
        with self._condition:
            revision = self._scenario_revision
        self._send("select_scenario", {"name": self.urbanfly_scenario})
        with self._condition:
            self._wait_locked(
                lambda: self._scenario_revision > revision and self._latest_state is not None,
                self.sensor_timeout_s,
            )
        self._episode_applied = False
        self._policy_step = 0

    def reset(self) -> None:
        self._episode_id = f"urbanfly-{uuid.uuid4().hex[:12]}"
        self._requested_start_nwu = None
        self._goal_nwu = None
        self._last_collision_count = 0
        # Policy step ids restart from zero for every episode.  Reset both
        # halves of the ACK state so step 0 cannot be satisfied by the final
        # ACK from the preceding episode.
        self._action_ack_step = -2
        self._action_ack = None
        self._last_depth_revision = -1
        self._packet_state_pending = False
        self._last_velocity_time = None
        self._last_velocity_nwu.fill(0.0)
        self._last_acceleration_nwu.fill(0.0)
        if hasattr(self, "_last_canonical_state"):
            del self._last_canonical_state
        self._select_scenario()

    def configure_scenario(self, scenario: str, difficulty: str, seed: int) -> None:
        # Scenario/difficulty/seed remain experiment metadata.  The actual
        # photogrammetry scene is selected explicitly by urbanfly_scenario.
        self.config["benchmark_scenario"] = str(scenario)
        self.config["difficulty"] = str(difficulty)
        self.config["seed"] = int(seed)

    def _apply_episode_config(self) -> None:
        if self._goal_nwu is None:
            return
        payload: dict[str, Any] = {
            "drone_id": self.drone_id,
            "goal_world_m": nwu_to_urbanfly_world(self._goal_nwu).tolist(),
            "yaw_degrees": float(self.config.get("initial_yaw_degrees", 0.0)),
            "policy_family": self.policy_family,
            "shield_enabled": self.shield_enabled,
            "episode_seed": int(self.config.get("seed", 20260731)),
            "dynamic_actor_density": float(self.config.get("dynamic_actor_density", 1.0)),
            "appearance_perturbation": dict(self.config.get("appearance_perturbation", {})),
        }
        dynamics = dict(self.config.get("dynamics_perturbation", {}))
        if "wind_nwu_mps" in dynamics:
            dynamics["wind_world_mps"] = nwu_to_urbanfly_world(dynamics.pop("wind_nwu_mps")).tolist()
        payload["dynamics_perturbation"] = dynamics
        if self._requested_start_nwu is not None:
            payload["start_world_m"] = nwu_to_urbanfly_world(
                self._requested_start_nwu
            ).tolist()
        with self._condition:
            revision = self._episode_revision
            packet_revision = self._packet_revision
            self._episode_ack = None
        self._send("policy_episode_config", payload)
        with self._condition:
            self._wait_locked(
                lambda: self._episode_revision > revision,
                self.command_timeout_s,
            )
            acknowledged_sim_time = float((self._episode_ack or {}).get("sim_time", 0.0))
            # A browser encode started before the scenario reset can finish
            # after the ACK.  Wait for a packet in the acknowledged episode
            # time window instead of accepting that stale previous-epoch frame.
            self._wait_locked(
                lambda: (
                    self._packet_revision > packet_revision
                    and self._latest_packet is not None
                    and acknowledged_sim_time - 0.5
                    <= float(self._latest_packet.observation.sim_time)
                    <= acknowledged_sim_time + 2.0
                ),
                self.sensor_timeout_s,
            )
        self._episode_applied = True
        self._policy_step = 0

    def set_initial_pose(self, position_nwu: np.ndarray) -> None:
        position = np.asarray(position_nwu, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("initial position must be a finite NWU vector of length 3")
        self._requested_start_nwu = position.copy()
        if self._goal_nwu is not None:
            self._apply_episode_config()

    def set_goal(self, goal_nwu: np.ndarray) -> None:
        goal = np.asarray(goal_nwu, dtype=np.float64)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("goal must be a finite NWU vector of length 3")
        self._goal_nwu = goal.copy()
        self._apply_episode_config()

    def takeoff(self) -> None:
        if not self._episode_applied:
            self._apply_episode_config()

    def land(self) -> None:
        try:
            self.execute_velocity_command(
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
                0.0,
                0.5,
            )
        except (RuntimeError, TimeoutError):
            pass

    def start_synchronized_recording(
        self,
        output_dir,
        fps: float = 30.0,
        *,
        layout: str = "rgbd_world_model",
    ) -> None:
        """Start a continuous browser runtime recording (never screenshot stitching)."""

        if self._recording_id is not None:
            raise RuntimeError("runtime recording is already active")
        safe_episode = re.sub(r"[^A-Za-z0-9_.-]", "-", self._episode_id)
        self._recording_id = f"{safe_episode}-{int(time.time() * 1000)}"
        self._recording_ack = None
        self._recording_started = False
        self._send("runtime_recording_control", {
            "action": "start", "recording_id": self._recording_id,
            "fps": float(fps), "camera_mode": "follow", "video_bitrate": 12_000_000,
            "follow_distance_m": 22.0, "follow_height_m": 9.0,
            "layout": str(layout),
            "requested_output_dir": str(output_dir),
        })
        with self._condition:
            self._wait_locked(lambda: self._recording_started, 10.0)

    def stop_synchronized_recording(self) -> dict[str, Any]:
        if self._recording_id is None:
            raise RuntimeError("runtime recording is not active")
        recording_id = self._recording_id
        self._send("runtime_recording_control", {"action": "stop", "recording_id": recording_id})
        with self._condition:
            self._wait_locked(lambda: self._recording_ack is not None, 90.0)
            result = dict(self._recording_ack or {})
        self._recording_id = None
        self._recording_ack = None
        self._recording_started = False
        return {"runtime": result}

    def _latest_decoded(self) -> DecodedUrbanFlyPacket:
        with self._condition:
            self._wait_locked(
                lambda: self._latest_packet is not None,
                self.sensor_timeout_s,
            )
            assert self._latest_packet is not None
            return self._latest_packet

    def get_depth(self) -> SensorFrame:
        with self._condition:
            self._wait_locked(
                lambda: (
                    self._latest_packet is not None
                    and self._packet_revision > self._last_depth_revision
                ),
                self.sensor_timeout_s,
            )
            assert self._latest_packet is not None
            decoded = self._latest_packet
            self._last_depth_revision = self._packet_revision
            self._packet_state_pending = True
        observation = decoded.observation
        return SensorFrame(
            timestamp=observation.sim_time,
            depth_m=observation.depth_m.copy(),
            valid_mask=observation.depth_valid_mask.copy(),
            rgb=observation.rgb.copy(),
            camera_intrinsics=observation.camera_intrinsics.copy(),
        )

    def get_rgb(self) -> np.ndarray:
        return self._latest_decoded().observation.rgb.copy()

    def _kinematics_from_packet(
        self,
        decoded: DecodedUrbanFlyPacket,
    ) -> VehicleState:
        header = decoded.header
        pose = header["vehicle_pose"]
        position = urbanfly_world_to_nwu(pose["position"])
        velocity = urbanfly_world_to_nwu(
            header.get("linear_velocity_world_mps", [0.0, 0.0, 0.0])
        )
        dynamics = header.get("dynamics") or {}
        roll = float(dynamics.get("roll_degrees", 0.0))
        pitch = float(dynamics.get("pitch_degrees", 0.0))
        yaw = float(header.get("yaw_degrees", dynamics.get("yaw_degrees", 0.0)))
        orientation = Rotation.from_euler(
            "xyz", [roll, pitch, yaw], degrees=True
        ).as_quat()
        body_angular = np.asarray(
            decoded.observation.angular_velocity_body_flu_rps,
            dtype=np.float64,
        )
        angular_velocity = Rotation.from_quat(orientation).apply(body_angular)
        timestamp = float(header["sim_time"])
        if self._last_velocity_time is not None and timestamp > self._last_velocity_time:
            self._last_acceleration_nwu = (
                velocity - self._last_velocity_nwu
            ) / (timestamp - self._last_velocity_time)
        self._last_velocity_nwu = velocity.copy()
        self._last_velocity_time = timestamp
        return VehicleState(
            timestamp=timestamp,
            position=position,
            orientation_xyzw=orientation,
            linear_velocity=velocity,
            angular_velocity=angular_velocity,
            linear_acceleration=self._last_acceleration_nwu.copy(),
            vehicle_name=str(header.get("vehicle_name", self.drone_id)),
            frame="nwu",
        )

    def _kinematics_from_state(self, state: dict[str, Any]) -> VehicleState:
        drones = state.get("drones") or []
        drone = next(
            (item for item in drones if item.get("id") == self.drone_id),
            drones[0] if drones else None,
        )
        if drone is None:
            raise RuntimeError("UrbanFly state does not contain a drone")
        position = urbanfly_world_to_nwu(drone["pos"])
        velocity = urbanfly_world_to_nwu(drone.get("vel", [0.0, 0.0, 0.0]))
        acceleration = urbanfly_world_to_nwu(
            drone.get("accel", [0.0, 0.0, 0.0])
        )
        roll = float(drone.get("roll", 0.0))
        pitch = float(drone.get("pitch", 0.0))
        yaw = float(drone.get("yaw", 0.0))
        orientation = Rotation.from_euler(
            "xyz", [roll, pitch, yaw], degrees=True
        ).as_quat()
        body_angular_urbanfly = np.asarray(
            drone.get("angular_velocity", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        body_angular_flu = body_angular_urbanfly[[0, 2, 1]]
        angular_velocity = Rotation.from_quat(orientation).apply(
            body_angular_flu
        )
        return VehicleState(
            timestamp=float(state.get("t", 0.0)),
            position=position,
            orientation_xyzw=orientation,
            linear_velocity=velocity,
            angular_velocity=angular_velocity,
            linear_acceleration=acceleration,
            vehicle_name=str(drone.get("id", self.drone_id)),
            frame="nwu",
        )

    def get_kinematics(self) -> VehicleState:
        with self._condition:
            if self._packet_state_pending and self._latest_packet is not None:
                decoded = self._latest_packet
                self._packet_state_pending = False
                return self._kinematics_from_packet(decoded)
            self._wait_locked(
                lambda: self._latest_state is not None,
                self.sensor_timeout_s,
            )
            assert self._latest_state is not None
            state = self._latest_state
        return self._kinematics_from_state(state)

    def get_collision_info(self) -> dict[str, object]:
        header = self._latest_decoded().header
        world_model = header.get("world_model") or {}
        count = int(world_model.get("actual_collision_count", 0))
        collided = bool(world_model.get("collision", False) or count > self._last_collision_count)
        self._last_collision_count = max(self._last_collision_count, count)
        return {
            "has_collided": collided,
            "object_name": "urbanfly_static_collision_field" if collided else "",
            "collision_count": count,
            "safety_intervened": bool(world_model.get("safety_intervened", False)),
            "safety_intervention_reasons": list(
                world_model.get("safety_intervention_reasons", [])
            ),
        }

    def get_actor_states(self) -> list[ActorState]:
        """Expose deterministic actor truth only through the privileged stream."""
        with self._condition:
            state = dict(self._latest_state or {})
        result: list[ActorState] = []
        for actor in state.get("actors", []):
            result.append(
                ActorState(
                    actor_id=int(actor["id"]),
                    actor_type=str(actor.get("actor_type", "unknown")),
                    position=urbanfly_world_to_nwu(actor["pos"]),
                    velocity=urbanfly_world_to_nwu(actor.get("vel", [0.0, 0.0, 0.0])),
                    bbox_extent=urbanfly_world_to_nwu(actor.get("bbox_extent", [0.5, 0.5, 0.5])),
                    timestamp=float(state.get("t", 0.0)),
                    scripted=bool(actor.get("scripted", True)),
                    frame="nwu",
                )
            )
        return result

    def publish_planner_visualization(
        self,
        candidates,
        *,
        selected_index: int,
        decision_sequence: int,
        metadata: dict[str, object],
    ) -> None:
        """Send all 15 real YOPO trajectories to the browser overlay."""

        if len(candidates) != 15:
            raise ValueError("UrbanFly planner visualization requires exactly 15 candidates")
        if not 0 <= int(selected_index) < len(candidates):
            raise ValueError("selected_index is outside the candidate set")
        scores_value = metadata.get("total_scores")
        if scores_value is None:
            scores = np.asarray([candidate.yopo_cost for candidate in candidates], dtype=np.float64)
        else:
            scores = np.asarray(scores_value, dtype=np.float64)
        if scores.shape != (15,) or not np.isfinite(scores).all():
            raise ValueError("planner visualization scores must be 15 finite values")
        collision = np.asarray(
            metadata.get("collision_probability", np.zeros(15)), dtype=np.float64
        )
        uncertainty = np.asarray(
            metadata.get("uncertainty", np.zeros(15)), dtype=np.float64
        )
        if collision.shape != (15,) or uncertainty.shape != (15,):
            raise ValueError("planner visualization model arrays must align with 15 candidates")
        order = [int(selected_index)] + [
            int(index) for index in np.argsort(scores) if int(index) != int(selected_index)
        ]

        def trajectory_world(index: int) -> list[list[float]]:
            trajectory = candidates[index]
            valid = np.asarray(trajectory.valid_mask, dtype=bool)
            points = np.asarray(trajectory.positions, dtype=np.float64)[valid]
            return [nwu_to_urbanfly_world(point).tolist() for point in points]

        top_candidates = [
            {
                "candidate_index": index,
                "score": float(scores[index]),
                "collision_probability": float(np.clip(collision[index], 0.0, 1.0)),
                "uncertainty": float(max(0.0, uncertainty[index])),
                "predicted_collision": bool(collision[index] >= 0.5),
                "trajectory_world_m": trajectory_world(index),
            }
            for index in order
        ]
        self._latest_planner_latency_ms = float(
            max(0.0, metadata.get("planner_latency_ms", 0.0))
        )
        self._latest_predicted_risk = float(
            np.clip(collision[int(selected_index)], 0.0, 1.0)
        )
        self._send("policy_visualization", {
            "drone_id": self.drone_id,
            "decision_sequence": int(decision_sequence),
            "candidate_count": 15,
            "selected_index": int(selected_index),
            "raw_selected_index": int(metadata.get("raw_selected_index", selected_index)),
            "selection_method": str(metadata.get("selection_method", self.policy_family)),
            "selected_trajectory_world_m": trajectory_world(int(selected_index)),
            "top_candidates": top_candidates,
            "planner_latency_ms": self._latest_planner_latency_ms,
            "predicted_risk": self._latest_predicted_risk,
        })

    def publish_policy_visualization(self, payload: dict[str, object]) -> None:
        """Publish bounded display telemetry; this channel has no control authority."""

        self._send(
            "policy_visualization",
            {"drone_id": self.drone_id, **dict(payload)},
        )

    def get_timestamp(self) -> float:
        with self._condition:
            if self._latest_state is not None:
                return float(self._latest_state.get("t", 0.0))
        return float(self._latest_decoded().header["sim_time"])

    def execute_velocity_command(
        self,
        velocity_nwu: np.ndarray,
        yaw_rate: float,
        duration: float,
    ) -> None:
        velocity = np.asarray(velocity_nwu, dtype=np.float64)
        duration = float(duration)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("velocity must be a finite NWU vector of length 3")
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        jitter_ms = float(self.config.get("dynamics_perturbation", {}).get("control_jitter_ms", 0.0))
        if jitter_ms > 0.0:
            time.sleep(float(self._jitter_rng.uniform(0.0, jitter_ms)) / 1000.0)
        state = self.get_kinematics()
        body_velocity = Rotation.from_quat(state.orientation_xyzw).inv().apply(velocity)
        action = np.array(
            [
                body_velocity[0] / 6.0,
                body_velocity[1] / 6.0,
                body_velocity[2] / 3.0,
                np.degrees(float(yaw_rate)) / 60.0,
            ],
            dtype=np.float64,
        )
        action = np.clip(action, -1.0, 1.0)
        step_id = self._policy_step
        self._policy_step += 1
        with self._condition:
            state_revision = self._state_revision
        self._send(
            "policy_action",
            {
                "drone_id": self.drone_id,
                "step_id": step_id,
                "policy_family": self.policy_family,
                "action_normalized": action.tolist(),
                "inference_latency_ms": self._latest_planner_latency_ms,
                "predicted_risk": self._latest_predicted_risk,
                "shield_enabled": self.shield_enabled,
                "timeout_s": max(0.45, duration + 0.25),
            },
        )
        with self._condition:
            self._wait_locked(
                lambda: self._action_ack_step >= step_id,
                self.command_timeout_s,
            )
            required_updates = max(1, int(np.ceil(duration / 0.1 * 0.8)))
            self._wait_locked(
                lambda: self._state_revision >= state_revision + required_updates,
                max(self.command_timeout_s, duration + 1.0),
            )

    def pause(self) -> None:
        self._send("control", {"action": "pause"})

    def continue_simulation(self) -> None:
        self._send("control", {"action": "play"})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop, ws = self._loop, self._ws
        if loop is not None and ws is not None and not ws.closed:
            future = asyncio.run_coroutine_threadsafe(ws.close(), loop)
            try:
                future.result(timeout=2.0)
            except (TimeoutError, RuntimeError):
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
