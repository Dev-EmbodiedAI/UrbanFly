from __future__ import annotations

from collections import deque
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch

from uav_wm_navigation.control.reranker import RiskReranker
from uav_wm_navigation.types import CandidateTrajectory, RiskPrediction, RerankDecision, TimestampedSensorFrame

from .factory import build_world_model
from .uncertainty import mc_dropout_predict


def _rotation_world_from_body_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float32)


def _resample(values: np.ndarray, steps: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(target, source, values[:, column]) for column in range(values.shape[1])], axis=-1)


class CandidateWorldModelRuntime:
    """Checkpoint-backed candidate scorer for the asynchronous YOPO loop.

    It deliberately cannot generate a trajectory or a control command.  Its
    only authority is to reorder the same 15 primitives emitted by YOPO.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        weights: dict[str, float],
        timeout_ms: float = 80.0,
        max_risk: float = 0.75,
        mc_dropout_samples: int | None = None,
        device: str | None = None,
        local_goal_lookahead_m: float = 10.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = dict(checkpoint["config"])
        self.family = str(self.config["model"]).lower()
        if self.family not in {"dreamerv3", "jepa"}:
            raise ValueError(f"realtime experiment only supports dreamerv3/jepa, got {self.family}")
        self.model = build_world_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.calibration = dict(checkpoint.get("calibration", {}))
        self.reranker = RiskReranker(weights, checkpoint.get("normalization"))
        self.timeout_ms = float(timeout_ms)
        self.max_risk = float(max_risk)
        self.samples = int(mc_dropout_samples or self.config.get("mc_dropout_samples", 5))
        self.history = int(self.config.get("history", 4))
        self.depth_max_m = float(self.config.get("depth_max_m", 20.0))
        self.trajectory_steps = int(self.config.get("trajectory_steps", 16))
        self.local_goal_lookahead_m = float(local_goal_lookahead_m)
        self._depth: deque[np.ndarray] = deque(maxlen=self.history)
        self._state: deque[np.ndarray] = deque(maxlen=self.history)
        # begin_flight_interval() resets histories while the planner thread may
        # already be preparing the next inference.  Guard reset/append/snapshot
        # as one atomic operation so neither model can observe an empty or
        # partially populated temporal window.
        self._history_lock = threading.Lock()

    def reset(self) -> None:
        with self._history_lock:
            self._depth.clear()
            self._state.clear()

    def _inputs(
        self, frame: TimestampedSensorFrame, candidates: tuple[CandidateTrajectory, ...], goal_nwu: np.ndarray
    ) -> tuple[torch.Tensor, ...]:
        sensor, state = frame.sensor, frame.state
        depth = np.asarray(sensor.depth_m, dtype=np.float32)
        valid = np.asarray(sensor.valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
        depth = np.where(valid, depth, self.depth_max_m)
        depth = np.clip(depth, 0.0, self.depth_max_m)
        if depth.shape != (96, 160):
            depth = cv2.resize(depth, (160, 96), interpolation=cv2.INTER_NEAREST)
        state_vector = np.concatenate([
            state.position, state.orientation_xyzw, state.linear_velocity, state.angular_velocity,
        ]).astype(np.float32)
        with self._history_lock:
            self._depth.append(depth / self.depth_max_m)
            self._state.append(state_vector)
            while len(self._depth) < self.history:
                self._depth.appendleft(self._depth[0].copy())
                self._state.appendleft(self._state[0].copy())
            depth_history = np.stack(tuple(self._depth)).astype(np.float32)
            state_history = np.stack(tuple(self._state)).astype(np.float32)
        rotation = _rotation_world_from_body_xyzw(state.orientation_xyzw)
        trajectories = []
        for candidate in candidates:
            acceleration = candidate.accelerations
            if acceleration is None:
                acceleration = np.zeros_like(candidate.positions)
            combined = np.concatenate([
                (candidate.positions - state.position[None]) @ rotation,
                candidate.velocities @ rotation,
                acceleration @ rotation,
            ], axis=-1)
            trajectories.append(_resample(combined, self.trajectory_steps))
        goal_body = (np.asarray(goal_nwu, dtype=np.float32) - state.position) @ rotation
        goal_distance = float(np.linalg.norm(goal_body))
        if goal_distance > self.local_goal_lookahead_m > 0.0:
            goal_body *= self.local_goal_lookahead_m / goal_distance
        return (
            torch.from_numpy(depth_history[None, :, None]).to(self.device),
            torch.from_numpy(state_history[None]).to(self.device),
            torch.from_numpy(goal_body[None].astype(np.float32)).to(self.device),
            torch.from_numpy(np.stack(trajectories)[None].astype(np.float32)).to(self.device),
        )

    def rank(
        self, frame: TimestampedSensorFrame, candidates: tuple[CandidateTrajectory, ...], goal_nwu: np.ndarray
    ) -> tuple[RerankDecision, list[RiskPrediction], float]:
        started = time.perf_counter()
        output = mc_dropout_predict(self.model, self._inputs(frame, candidates, goal_nwu), self.samples, self.calibration)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        predictions = [RiskPrediction(
            collision_probability=float(output["collision_probability"][0, index].cpu()),
            minimum_clearance=float(output["minimum_clearance"][0, index].cpu()),
            goal_progress=float(output["goal_progress"][0, index].cpu()),
            failure_probability=float(output["failure_probability"][0, index].cpu()),
            uncertainty=float(output["uncertainty"][0, index].cpu()),
        ) for index in range(len(candidates))]
        decision = self.reranker.rank(
            list(candidates), predictions, latency_ms=latency_ms,
            timeout_ms=self.timeout_ms, max_risk=self.max_risk,
        )
        return decision, predictions, latency_ms
