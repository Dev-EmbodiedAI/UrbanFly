from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from uav_wm_navigation.planners.mppi import MPPIPlan, MPPIPlanner
from uav_wm_navigation.types import SensorFrame, VehicleState, WorldModelObservation
from uav_wm_navigation.world_models.base import PLANNING_STATE_DIM
from uav_wm_navigation.world_models.jepa_adapter import JEPAWorldModelAdapter


@dataclass(slots=True)
class WAMMPCDecision:
    action_normalized: np.ndarray
    predicted_risk: float
    plan: MPPIPlan | None
    used_fallback: bool
    error: str | None
    diagnostics: dict[str, Any]


class WAMMPCController:
    """Single-UAV receding-horizon controller; executes no action itself."""

    def __init__(
        self,
        world_model: JEPAWorldModelAdapter,
        planner: MPPIPlanner,
        *,
        history: int = 4,
        depth_max_m: float = 20.0,
        depth_shape: tuple[int, int] = (96, 160),
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.planner = planner
        self.history = int(history)
        self.depth_max_m = float(depth_max_m)
        self.depth_shape = tuple(int(value) for value in depth_shape)
        self._depth_history: deque[np.ndarray] = deque(maxlen=self.history)
        self._state_history: deque[np.ndarray] = deque(maxlen=self.history)
        self._previous_prediction: dict[str, torch.Tensor] | None = None
        self.prediction_errors: list[dict[str, float]] = []
        self.failure_count = 0

    def reset(self) -> None:
        self._depth_history.clear()
        self._state_history.clear()
        self._previous_prediction = None
        self.prediction_errors.clear()
        self.failure_count = 0
        self.planner.reset()

    @staticmethod
    def _yaw(state: VehicleState) -> float:
        x, y, z, w = state.orientation_xyzw.astype(np.float64)
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _append_history(self, observation: WorldModelObservation, state: VehicleState) -> None:
        depth = np.asarray(observation.depth_m, dtype=np.float32)
        valid = np.asarray(observation.depth_valid_mask, dtype=bool)
        valid &= np.isfinite(depth) & (depth > 0.0)
        if not valid.any():
            raise ValueError("invalid depth: no finite positive pixels")
        clean = np.where(valid, depth, self.depth_max_m)
        clean = np.clip(clean, 0.0, self.depth_max_m)
        if clean.shape != self.depth_shape:
            clean = cv2.resize(clean, self.depth_shape[::-1], interpolation=cv2.INTER_NEAREST)
        self._depth_history.append((clean / self.depth_max_m).astype(np.float32))
        self._state_history.append(np.concatenate([
            state.position,
            state.orientation_xyzw,
            state.linear_velocity,
            state.angular_velocity,
        ]).astype(np.float32))
        while len(self._depth_history) < self.history:
            self._depth_history.appendleft(self._depth_history[0].copy())
            self._state_history.appendleft(self._state_history[0].copy())

    def _planning_state(self, observation: WorldModelObservation, state: VehicleState) -> torch.Tensor:
        depth = np.asarray(observation.depth_m, dtype=np.float32)
        valid = np.asarray(observation.depth_valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
        height, width = depth.shape
        roi = np.zeros_like(valid)
        roi[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = True
        forward = depth[valid & roi]
        clearance = float(np.percentile(forward, 10.0)) if forward.size else 0.0
        value = np.zeros(PLANNING_STATE_DIM, dtype=np.float32)
        value[0:3] = state.position
        value[3:6] = state.linear_velocity
        value[6] = self._yaw(state)
        value[7] = state.angular_velocity[2]
        value[8] = min(clearance, self.depth_max_m)
        return torch.from_numpy(value[None]).to(self.device)

    def _encode(self, observation: WorldModelObservation) -> torch.Tensor:
        model_observation = {
            "depth": torch.from_numpy(np.stack(tuple(self._depth_history))[None, :, None]).to(self.device),
            "state_history": torch.from_numpy(np.stack(tuple(self._state_history))[None]).to(self.device),
            "goal_body": torch.from_numpy(observation.goal_body_flu_m[None].astype(np.float32)).to(self.device),
        }
        with torch.inference_mode():
            return self.world_model.encode(model_observation)

    def _record_prediction_error(self, latent: torch.Tensor, state: VehicleState) -> dict[str, float]:
        if self._previous_prediction is None:
            return {}
        predicted = self._previous_prediction
        errors = {
            "latent_prediction_error": float(torch.mean((latent[0] - predicted["latent"]) ** 2).item()),
            "position_prediction_error_m": float(
                np.linalg.norm(state.position - predicted["position"].cpu().numpy())
            ),
            "velocity_prediction_error_mps": float(
                np.linalg.norm(state.linear_velocity - predicted["velocity"].cpu().numpy())
            ),
        }
        self.prediction_errors.append(errors)
        return errors

    def plan(
        self,
        observation: WorldModelObservation,
        state: VehicleState,
        goal_nwu: np.ndarray,
    ) -> WAMMPCDecision:
        total_started = time.perf_counter()
        try:
            self._append_history(observation, state)
            encode_started = time.perf_counter()
            latent = self._encode(observation)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            encoder_latency_ms = (time.perf_counter() - encode_started) * 1000.0
            errors = self._record_prediction_error(latent, state)
            planning_state = self._planning_state(observation, state)
            goal = torch.as_tensor(goal_nwu, dtype=torch.float32, device=self.device)
            plan = self.planner.plan(self.world_model, latent, planning_state, goal)
            first_risk = float(plan.predicted_collision_probability[0].item())
            self._previous_prediction = {
                "latent": plan.predicted_latents[0].detach(),
                "position": plan.predicted_positions[0].detach(),
                "velocity": plan.predicted_velocities[0].detach(),
            }
            diagnostics = {
                **plan.diagnostics,
                **errors,
                "encoder_latency_ms": encoder_latency_ms,
                "total_planning_latency_ms": (time.perf_counter() - total_started) * 1000.0,
                "cost": plan.total_cost,
                "cost_components": plan.cost_components,
                "fallback": False,
            }
            return WAMMPCDecision(
                action_normalized=self.world_model.planner_action_to_normalized(
                    plan.first_action
                ).cpu().numpy().astype(np.float32),
                predicted_risk=first_risk,
                plan=plan,
                used_fallback=False,
                error=None,
                diagnostics=diagnostics,
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            self.failure_count += 1
            self._previous_prediction = None
            return WAMMPCDecision(
                action_normalized=np.zeros(4, dtype=np.float32),
                predicted_risk=1.0,
                plan=None,
                used_fallback=True,
                error=f"{type(error).__name__}: {error}",
                diagnostics={
                    "total_planning_latency_ms": (time.perf_counter() - total_started) * 1000.0,
                    "fallback": True,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
