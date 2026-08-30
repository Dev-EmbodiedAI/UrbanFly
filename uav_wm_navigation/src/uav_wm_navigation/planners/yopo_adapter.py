from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from uav_wm_navigation.planners.base import CandidatePlanner, PlanningContext
from uav_wm_navigation.planners.polynomial import sample_quintic
from uav_wm_navigation.types import CandidateTrajectory


def build_yopo_observation(
    state,
    local_goal_nwu: np.ndarray,
    desired_position_nwu: np.ndarray,
    reference_acceleration_nwu: np.ndarray,
) -> np.ndarray:
    """Build official YOPO's 9-D body-FLU observation.

    Layout is ``[body velocity, body reference acceleration, body goal]``.
    Keeping this transformation explicit makes route-goal wiring testable
    without loading the CUDA network.
    """
    rotation_nwu_flu = Rotation.from_quat(state.orientation_xyzw).as_matrix()
    velocity_body = rotation_nwu_flu.T @ state.linear_velocity
    acceleration_body = rotation_nwu_flu.T @ np.asarray(reference_acceleration_nwu, dtype=np.float64)
    goal_body = rotation_nwu_flu.T @ (
        np.asarray(local_goal_nwu, dtype=np.float64) - np.asarray(desired_position_nwu, dtype=np.float64)
    )
    return np.concatenate([velocity_body, acceleration_body, goal_body]).astype(np.float32)


def preprocess_depth(depth_m: np.ndarray, width: int, height: int, depth_max_m: float, min_depth_m: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [height, width]")
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    depth = np.clip(np.nan_to_num(depth, nan=0.0, posinf=depth_max_m, neginf=0.0), 0.0, depth_max_m)
    normalized = depth / depth_max_m
    invalid |= normalized < min_depth_m / depth_max_m
    if invalid.all():
        raise ValueError("depth frame has no valid pixels")
    if invalid.any():
        normalized = cv2.inpaint(
            np.round(normalized * 255).astype(np.uint8), invalid.astype(np.uint8), 1, cv2.INPAINT_NS
        ).astype(np.float32) / 255.0
    return normalized[None, None]


class YOPOAdapter(CandidatePlanner):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.yopo_root = Path(config["yopo_root"]).expanduser().resolve()
        self.checkpoint = Path(config["checkpoint"]).expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)
        if str(self.yopo_root) not in sys.path:
            sys.path.insert(0, str(self.yopo_root))
        from config.config import cfg
        cfg["train"] = False
        cfg["velocity"] = float(config.get("velocity", 4.5))
        from policy.primitive import LatticePrimitive
        from policy.state_transform import StateTransform
        from policy.yopo_network import YopoNetwork
        self.cfg = cfg
        requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.lattice = LatticePrimitive.get_instance()
        self.state_transform = StateTransform()
        self.policy = YopoNetwork().to(self.device)
        state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        self.horizon_steps = int(config.get("horizon_steps", 21))
        self.depth_max_m = float(config.get("depth_max_m", 20.0))
        self.min_depth_m = float(config.get("min_depth_m", 0.8))
        self.plan_from_reference = bool(config.get("plan_from_reference", True))
        self.reference_reset_distance_m = float(config.get("reference_reset_distance_m", 1.0))
        self.reference_position: np.ndarray | None = None
        self.reference_velocity: np.ndarray | None = None
        self.reference_acceleration: np.ndarray | None = None
        self._reference_lock = threading.Lock()

    @property
    def candidate_count(self) -> int:
        return int(self.lattice.traj_num)

    def plan(self, context: PlanningContext) -> list[CandidateTrajectory]:
        state = context.state
        with self._reference_lock:
            if self.reference_position is None:
                self.reference_position = state.position.copy()
                self.reference_velocity = state.linear_velocity.copy()
                self.reference_acceleration = np.zeros(3, dtype=np.float64)
            if self.plan_from_reference and np.linalg.norm(self.reference_position - state.position) > self.reference_reset_distance_m:
                self.reference_position = state.position.copy()
                self.reference_velocity = state.linear_velocity.copy()
                self.reference_acceleration = np.zeros(3, dtype=np.float64)
            desired_position = self.reference_position.copy()
            desired_velocity = self.reference_velocity.copy()
            desired_acceleration = self.reference_acceleration.copy()
        start_position = desired_position if self.plan_from_reference else state.position.copy()
        start_velocity = desired_velocity if self.plan_from_reference else state.linear_velocity.copy()
        rotation_nwu_flu = Rotation.from_quat(state.orientation_xyzw).as_matrix()
        # Instantaneous acceleration contains controller/gravity transients
        # after takeoff. Official YOPO plans from a smooth reference
        # acceleration; feeding the raw transient makes every short receding
        # horizon dive. Zero is the stable reference default.
        # Official test_yopo_ros.py always feeds the continuously updated
        # desired acceleration, even when position/velocity start from odometry.
        reference_acceleration = desired_acceleration
        # The official node also forms goal direction from desire_pos in both
        # plan modes; it is not recomputed from odometry in direct-control mode.
        observation = build_yopo_observation(
            state, context.local_goal_nwu, desired_position, reference_acceleration
        )
        depth = preprocess_depth(
            context.sensor.depth_m, int(self.cfg["image_width"]), int(self.cfg["image_height"]),
            self.depth_max_m, self.min_depth_m,
        )
        depth_tensor = torch.from_numpy(depth).to(self.device)
        observation_tensor = torch.from_numpy(observation[None]).to(self.device)
        prepared = self.state_transform.prepare_input(self.state_transform.normalize_obs(observation_tensor.clone()))
        with torch.inference_mode():
            raw_endstates, raw_scores = self.policy(depth_tensor, prepared)
        endstates = raw_endstates.detach().cpu().numpy().reshape(9, self.candidate_count).T
        scores = raw_scores.detach().cpu().numpy().reshape(self.candidate_count)
        lattice_ids = torch.arange(self.candidate_count - 1, -1, -1)
        body_endstates = self.state_transform.pred_to_endstate_cpu(endstates, lattice_ids)
        candidates: list[CandidateTrajectory] = []
        for index, (endstate, score) in enumerate(zip(body_endstates, scores)):
            matrix_body = endstate.reshape(3, 3).T
            matrix_world = rotation_nwu_flu @ matrix_body
            end_position = start_position + matrix_world[:, 0]
            positions, velocities, accelerations = sample_quintic(
                start_position, start_velocity, reference_acceleration,
                end_position, matrix_world[:, 1], matrix_world[:, 2],
                float(self.lattice.segment_time), self.horizon_steps,
            )
            candidates.append(CandidateTrajectory(
                trajectory_id=f"yopo-{index:02d}", positions=positions, velocities=velocities,
                accelerations=accelerations, duration=float(self.lattice.segment_time),
                yopo_cost=float(score), valid_mask=np.ones(self.horizon_steps, dtype=bool),
                metadata={
                    "source": "yopo", "network_index": index,
                    "lattice_id": int(self.candidate_count - 1 - index),
                    "checkpoint": str(self.checkpoint),
                },
            ))
        return candidates

    def commit_selected(self, trajectory: CandidateTrajectory, executed_duration_s: float) -> None:
        """Advance the desired state exactly as official YOPO plan-from-reference does."""
        if not self.plan_from_reference:
            return
        sample_times = np.linspace(0.0, float(trajectory.duration), len(trajectory.positions))
        target_time = float(np.clip(executed_duration_s, 0.0, trajectory.duration))
        position = np.array([
            np.interp(target_time, sample_times, trajectory.positions[:, axis]) for axis in range(3)
        ])
        velocity = np.array([
            np.interp(target_time, sample_times, trajectory.velocities[:, axis]) for axis in range(3)
        ])
        acceleration = np.array([
            np.interp(target_time, sample_times, trajectory.accelerations[:, axis]) for axis in range(3)
        ])
        self.update_control_reference(position, velocity, acceleration)

    def update_control_reference(
        self, position_nwu: np.ndarray, velocity_nwu: np.ndarray, acceleration_nwu: np.ndarray,
    ) -> None:
        """Mirror desire_pos/vel/acc updated by the official 50 Hz timer."""
        with self._reference_lock:
            self.reference_position = np.asarray(position_nwu, dtype=np.float64).copy()
            self.reference_velocity = np.asarray(velocity_nwu, dtype=np.float64).copy()
            self.reference_acceleration = np.asarray(acceleration_nwu, dtype=np.float64).copy()

    def reset_reference(self) -> None:
        with self._reference_lock:
            self.reference_position = None
            self.reference_velocity = None
            self.reference_acceleration = None
