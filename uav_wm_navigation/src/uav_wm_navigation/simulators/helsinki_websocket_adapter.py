"""Dataset-v1 adapter for the real Helsinki browser/backend runtime."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from backend.engine.helsinki_frames import (
    BACKEND_TO_ENU,
    ENU_TO_BACKEND,
    backend_world_to_enu,
    backend_yaw_to_enu_radians,
    enu_to_backend_world,
    enu_yaw_to_backend_degrees,
)
from uav_wm_navigation.types import ActorState, SensorFrame, VehicleState

from .urbanfly_websocket_adapter import UrbanFlyWebSocketAdapter


CAMERA_RDF_TO_RENDERER_CAMERA = np.diag([1.0, -1.0, -1.0])


def _wxyz_backend_to_enu_flu(quaternion_wxyz) -> np.ndarray:
    value = np.asarray(quaternion_wxyz, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("backend orientation must be finite wxyz")
    backend_rotation = Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()
    canonical_rotation = BACKEND_TO_ENU @ backend_rotation @ ENU_TO_BACKEND
    return Rotation.from_matrix(canonical_rotation).as_quat()


def _camera_pose_enu(header: dict) -> np.ndarray:
    quaternion = np.asarray(header["camera_orientation"], dtype=np.float64)
    backend_rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    rotation_enu_rdf = (
        BACKEND_TO_ENU @ backend_rotation @ CAMERA_RDF_TO_RENDERER_CAMERA
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation_enu_rdf
    pose[:3, 3] = backend_world_to_enu(header["camera_position"])
    return pose


class HelsinkiWebSocketAdapter(UrbanFlyWebSocketAdapter):
    """Expose canonical ENU/FLU state while retaining the real WebGL RGB-D loop.

    The generic historical adapter exposes an incorrectly labelled NWU
    permutation.  This class is intentionally selected explicitly by the
    Helsinki Dataset v1 collector and has no mock or procedural fallback.
    """

    canonical_world_frame = "enu"
    canonical_body_frame = "flu"

    def _apply_episode_config(self) -> None:
        if self._goal_nwu is None:
            return
        payload = {
            "drone_id": self.drone_id,
            "goal_world_m": enu_to_backend_world(self._goal_nwu).tolist(),
            "yaw_degrees": enu_yaw_to_backend_degrees(
                float(self.config.get("initial_yaw_enu_radians", 0.0))
            ),
            "policy_family": self.policy_family,
            "shield_enabled": self.shield_enabled,
            "episode_seed": int(self.config.get("seed", 20260731)),
            "dynamic_actor_density": float(self.config.get("dynamic_actor_density", 0.0)),
            "episode_duration_s": self.config.get("episode_duration_s"),
            "appearance_perturbation": dict(self.config.get("appearance_perturbation", {})),
        }
        dynamics = dict(self.config.get("dynamics_perturbation", {}))
        if "wind_enu_mps" in dynamics:
            dynamics["wind_world_mps"] = enu_to_backend_world(
                dynamics.pop("wind_enu_mps")
            ).tolist()
        payload["dynamics_perturbation"] = dynamics
        if self._requested_start_nwu is not None:
            payload["start_world_m"] = enu_to_backend_world(
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
            acknowledged = float((self._episode_ack or {}).get("sim_time", 0.0))
            self._wait_locked(
                lambda: (
                    self._packet_revision > packet_revision
                    and self._latest_packet is not None
                    and acknowledged - 0.5
                    <= float(self._latest_packet.observation.sim_time)
                    <= acknowledged + 2.0
                ),
                self.sensor_timeout_s,
            )
        self._episode_applied = True
        self._policy_step = 0

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
            camera_pose_nwu=_camera_pose_enu(decoded.header),
        )

    def _canonical_state(self, decoded) -> VehicleState:
        header = decoded.header
        pose = header["vehicle_pose"]
        orientation = _wxyz_backend_to_enu_flu(pose["orientation"])
        velocity = backend_world_to_enu(
            header.get("linear_velocity_world_mps", [0.0, 0.0, 0.0])
        )
        body_angular = np.asarray(
            decoded.observation.angular_velocity_body_flu_rps,
            dtype=np.float64,
        )
        angular_world = Rotation.from_quat(orientation).apply(body_angular)
        timestamp = float(header["sim_time"])
        if self._last_velocity_time is not None and timestamp > self._last_velocity_time:
            self._last_acceleration_nwu = (
                velocity - self._last_velocity_nwu
            ) / (timestamp - self._last_velocity_time)
        self._last_velocity_nwu = velocity.copy()
        self._last_velocity_time = timestamp
        state = VehicleState(
            timestamp=timestamp,
            position=backend_world_to_enu(pose["position"]),
            orientation_xyzw=orientation,
            linear_velocity=velocity,
            angular_velocity=angular_world,
            linear_acceleration=self._last_acceleration_nwu.copy(),
            vehicle_name=str(header.get("vehicle_name", self.drone_id)),
            frame="enu",
        )
        self._last_canonical_state = state
        return state

    def _kinematics_from_packet(self, decoded) -> VehicleState:
        return self._canonical_state(decoded)

    def _kinematics_from_state(self, state: dict) -> VehicleState:
        drones = state.get("drones") or []
        drone = next(
            (item for item in drones if item.get("id") == self.drone_id),
            drones[0] if drones else None,
        )
        if drone is None:
            raise RuntimeError("UrbanFly state does not contain a drone")
        orientation = _wxyz_backend_to_enu_flu(
            drone.get("orientation", [1.0, 0.0, 0.0, 0.0])
        )
        body_backend = np.asarray(
            drone.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        body_flu = np.asarray([body_backend[0], -body_backend[2], body_backend[1]])
        return VehicleState(
            timestamp=float(state.get("t", 0.0)),
            position=backend_world_to_enu(drone["pos"]),
            orientation_xyzw=orientation,
            linear_velocity=backend_world_to_enu(drone.get("vel", [0.0, 0.0, 0.0])),
            angular_velocity=Rotation.from_quat(orientation).apply(body_flu),
            linear_acceleration=backend_world_to_enu(drone.get("accel", [0.0, 0.0, 0.0])),
            vehicle_name=str(drone.get("id", self.drone_id)),
            frame="enu",
        )

    def get_actor_states(self) -> list[ActorState]:
        with self._condition:
            state = dict(self._latest_state or {})
        return [
            ActorState(
                actor_id=int(actor["id"]),
                actor_type=str(actor.get("actor_type", "unknown")),
                position=backend_world_to_enu(actor["pos"]),
                velocity=backend_world_to_enu(actor.get("vel", [0.0, 0.0, 0.0])),
                bbox_extent=np.abs(
                    backend_world_to_enu(actor.get("bbox_extent", [0.5, 0.5, 0.5]))
                ),
                timestamp=float(state.get("t", 0.0)),
                scripted=bool(actor.get("scripted", True)),
                frame="enu",
            )
            for actor in state.get("actors", [])
        ]

    def execute_velocity_command(
        self,
        velocity_nwu: np.ndarray,
        yaw_rate: float,
        duration: float,
        *,
        inference_latency_ms: float = 0.0,
        predicted_risk: float = 0.0,
    ) -> dict[str, object]:
        """Apply ENU velocity and return the backend's factual executed action."""

        velocity = np.asarray(velocity_nwu, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("velocity must be a finite ENU vector of length 3")
        state = getattr(self, "_last_canonical_state", None)
        if state is None:
            state = self.get_kinematics()
        body_velocity = Rotation.from_quat(state.orientation_xyzw).inv().apply(velocity)
        normalized = np.clip(
            np.asarray(
                [
                    body_velocity[0] / 6.0,
                    body_velocity[1] / 6.0,
                    body_velocity[2] / 3.0,
                    float(yaw_rate) / np.deg2rad(60.0),
                ]
            ),
            -1.0,
            1.0,
        )
        step_id = self._policy_step
        self._policy_step += 1
        with self._condition:
            state_revision = self._state_revision
            packet_revision = self._packet_revision
        self._send(
            "policy_action",
            {
                "drone_id": self.drone_id,
                "step_id": step_id,
                "policy_family": self.policy_family,
                "action_normalized": normalized.tolist(),
                "inference_latency_ms": max(0.0, float(inference_latency_ms)),
                "predicted_risk": float(np.clip(predicted_risk, 0.0, 1.0)),
                "shield_enabled": self.shield_enabled,
                "timeout_s": max(0.45, float(duration) + 0.25),
                "duration_s": float(duration),
            },
        )
        with self._condition:
            self._wait_locked(lambda: self._action_ack_step >= step_id, self.command_timeout_s)
            action_ack = dict(self._action_ack or {})
            if not bool(self.config.get("policy_lockstep", False)):
                required = max(1, int(np.ceil(float(duration) / 0.1 * 0.8)))
                self._wait_locked(
                    lambda: self._state_revision >= state_revision + required,
                    max(self.command_timeout_s, float(duration) + 1.0),
                )
            self._wait_locked(
                lambda: (
                    self._packet_revision > packet_revision
                    and self._latest_packet is not None
                    and int(
                        (self._latest_packet.header.get("world_model") or {}).get(
                            "policy_step_id", -99
                        )
                    )
                    == step_id
                ),
                self.sensor_timeout_s,
            )
            world_model = dict(
                (self._latest_packet.header.get("world_model") if self._latest_packet else {})
                or {}
            )
        commanded = normalized * np.asarray(
            [6.0, 6.0, 3.0, np.deg2rad(60.0)], dtype=np.float64
        )
        executed = np.asarray(
            world_model.get("executed_action_physical_body_flu", commanded),
            dtype=np.float64,
        )
        if executed.shape != (4,) or not np.isfinite(executed).all():
            raise RuntimeError("backend did not expose a finite executed FLU action")
        return {
            "step_id": step_id,
            "action_timestamp": float(action_ack["accepted_sim_time"]),
            "action_commanded_body_flu": commanded.astype(np.float32),
            "action_executed_body_flu": executed.astype(np.float32),
            "safety_intervened": bool(world_model.get("safety_intervened", False)),
            "safety_intervention_reasons": list(
                world_model.get("safety_intervention_reasons", [])
            ),
            "stale_action": bool(world_model.get("stale_action", False)),
        }
