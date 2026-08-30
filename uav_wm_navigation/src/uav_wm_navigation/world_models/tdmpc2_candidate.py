from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from uav_wm_navigation.types import (
    CandidatePrediction,
    CandidateTrajectory,
    RiskPrediction,
    SensorFrame,
    VehicleState,
    WorldModelObservation,
    EpisodeSpec,
)

from .tdmpc2_continuous import TDMPC2ContinuousPolicy
from .tdmpc2_visual import TDMPC2VisualPolicy


def _resample(values: np.ndarray, steps: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    target = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    return np.stack(
        [np.interp(target, source, values[:, column]) for column in range(values.shape[1])],
        axis=-1,
    ).astype(np.float32)


def candidate_actions_body_flu(
    candidate: CandidateTrajectory,
    state: VehicleState,
    *,
    horizon_steps: int,
    forward_limit_mps: float = 6.0,
    lateral_limit_mps: float = 6.0,
    vertical_limit_mps: float = 3.0,
    yaw_rate_limit_rps: float = float(np.deg2rad(60.0)),
) -> np.ndarray:
    """Convert one world-frame trajectory to normalized FLU velocity actions."""

    rotation_world_from_body = Rotation.from_quat(state.orientation_xyzw).as_matrix()
    velocity_body = np.asarray(candidate.velocities, dtype=np.float32) @ rotation_world_from_body
    velocity_body = _resample(velocity_body, horizon_steps)
    dt = float(candidate.duration) / max(horizon_steps - 1, 1)
    heading = np.unwrap(np.arctan2(velocity_body[:, 1], velocity_body[:, 0]))
    yaw_rate = np.gradient(heading, dt).astype(np.float32)
    actions = np.column_stack([velocity_body, yaw_rate])
    limits = np.asarray(
        [
            forward_limit_mps,
            lateral_limit_mps,
            vertical_limit_mps,
            yaw_rate_limit_rps,
        ],
        dtype=np.float32,
    )
    return np.clip(actions / limits, -1.0, 1.0).astype(np.float32)


class TDMPC2CandidateAssistant:
    """Use a trained TD-MPC2 model only to score planner candidates.

    The assistant never emits a new control action in the month-end comparison.
    It receives exactly the same candidate set as the direct planner, predicts
    return/risk for each sequence, and exposes candidate-level risk heads to the
    shared reranker.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        horizon_steps: int = 15,
        discount: float = 0.97,
        risk_weight: float = 8.0,
        device: str | None = None,
    ) -> None:
        self.policy = TDMPC2ContinuousPolicy(
            checkpoint=checkpoint,
            device=device,
            horizon=horizon_steps,
            candidates=1,
            elites=1,
            iterations=1,
            discount=discount,
            risk_weight=risk_weight,
        )
        self.horizon_steps = int(horizon_steps)
        self.discount = float(discount)
        self.risk_weight = float(risk_weight)
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.episode_id = ""
        self.step_id = 0

    def reset(self, episode_id: str | EpisodeSpec) -> None:
        self.episode_id = episode_id.episode_id if isinstance(episode_id, EpisodeSpec) else str(episode_id)
        self.step_id = 0
        self.previous_action.fill(0.0)
        self.policy.reset(self.episode_id)

    def _observation(
        self,
        sensor: SensorFrame,
        state: VehicleState,
        goal_nwu: np.ndarray,
    ) -> WorldModelObservation:
        rotation_world_from_body = Rotation.from_quat(state.orientation_xyzw).as_matrix()
        rgb = (
            np.asarray(sensor.rgb, dtype=np.uint8)
            if sensor.rgb is not None
            else np.zeros((*sensor.depth_m.shape, 3), dtype=np.uint8)
        )
        intrinsics = (
            np.asarray(sensor.camera_intrinsics, dtype=np.float32)
            if sensor.camera_intrinsics is not None
            else np.eye(3, dtype=np.float32)
        )
        return WorldModelObservation(
            episode_id=self.episode_id,
            step_id=self.step_id,
            sim_time=float(state.timestamp),
            rgb=rgb,
            depth_m=np.asarray(sensor.depth_m, dtype=np.float32),
            depth_valid_mask=np.asarray(sensor.valid_mask, dtype=bool),
            goal_body_flu_m=(
                np.asarray(goal_nwu, dtype=np.float32) - state.position
            )
            @ rotation_world_from_body,
            linear_velocity_body_flu_mps=state.linear_velocity
            @ rotation_world_from_body,
            angular_velocity_body_flu_rps=state.angular_velocity
            @ rotation_world_from_body,
            gravity_body_flu=np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
            @ rotation_world_from_body,
            previous_action=self.previous_action,
            sensor_timestamp=float(sensor.timestamp),
            state_timestamp=float(state.timestamp),
            camera_intrinsics=intrinsics,
            camera_extrinsics_body=np.eye(4, dtype=np.float32),
        )

    def predict(
        self,
        candidates: list[CandidateTrajectory] | tuple[CandidateTrajectory, ...],
        sensor: SensorFrame,
        state: VehicleState,
        goal_nwu: np.ndarray,
    ) -> tuple[list[RiskPrediction], np.ndarray, float]:
        if not candidates:
            raise ValueError("at least one candidate is required")
        predictions_v3, predicted_return, latency_ms = self.predict_candidates(
            candidates, sensor, state, goal_nwu
        )
        predictions = [
            RiskPrediction(
                collision_probability=item.collision_probability,
                minimum_clearance=item.minimum_clearance,
                goal_progress=item.goal_progress,
                failure_probability=item.failure_probability,
                uncertainty=item.epistemic_uncertainty,
                latent_states=item.predicted_state_1s_2s_3s,
            )
            for item in predictions_v3
        ]
        return predictions, predicted_return, latency_ms

    def predict_candidates(
        self,
        candidates: list[CandidateTrajectory] | tuple[CandidateTrajectory, ...],
        sensor: SensorFrame,
        state: VehicleState,
        goal_nwu: np.ndarray,
    ) -> tuple[list[CandidatePrediction], np.ndarray, float]:
        if not candidates:
            raise ValueError("at least one candidate is required")
        started = time.perf_counter()
        observation = self._observation(sensor, state, goal_nwu)
        self.policy.observe(observation)
        action_sequences = np.stack(
            [
                candidate_actions_body_flu(
                    candidate,
                    state,
                    horizon_steps=self.horizon_steps,
                )
                for candidate in candidates
            ]
        )
        output = self.policy.predict(action_sequences)
        discounts = self.discount ** np.arange(self.horizon_steps, dtype=np.float32)
        predicted_return = (
            ((output["reward"] - self.risk_weight * output["risk"]) * discounts).sum(
                axis=1
            )
            + discounts[-1] * output["q"][:, -1]
        )
        predictions: list[CandidatePrediction] = []
        for index, candidate in enumerate(candidates):
            risk = float(np.clip(output["risk"][index].max(), 0.0, 1.0))
            geometric_progress = float(
                np.linalg.norm(np.asarray(goal_nwu) - state.position)
                - np.linalg.norm(np.asarray(goal_nwu) - candidate.positions[-1])
            )
            progress = (
                float(output["progress"][index].sum())
                if "progress" in output
                else geometric_progress
            )
            clearance = float(output.get("clearance", 120.0 * (1.0 - output["risk"]))[index].min())
            uncertainty = float(output.get("uncertainty", np.zeros_like(output["risk"]))[index].max())
            continuation = output.get("continuation", 1.0 - output["risk"])[index]
            predicted_states = output.get(
                "predicted_state_1s_2s_3s",
                np.zeros((len(candidates), 3, 3), dtype=np.float32),
            )[index]
            predictions.append(
                CandidatePrediction(
                    goal_progress=progress,
                    collision_probability=risk,
                    minimum_clearance=clearance,
                    cpa_risk=risk,
                    terminal_value=float(output["q"][index, -1]),
                    epistemic_uncertainty=uncertainty,
                    failure_probability=float(np.clip(1.0 - np.min(continuation), 0.0, 1.0)),
                    predicted_state_1s_2s_3s=predicted_states,
                )
            )
        self.previous_action = action_sequences[int(np.argmax(predicted_return)), 0]
        self.step_id += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        return predictions, predicted_return.astype(np.float32), latency_ms


class VisualTDMPC2CandidateAssistant(TDMPC2CandidateAssistant):
    """Formal v3 assistant: real RGB-D encoder, same YOPO candidate set."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        horizon_steps: int = 15,
        discount: float = 0.97,
        risk_weight: float = 8.0,
        device: str | None = None,
    ) -> None:
        self.policy = TDMPC2VisualPolicy(
            checkpoint=checkpoint,
            device=device,
            horizon=horizon_steps,
            candidates=1,
            elites=1,
            iterations=1,
            discount=discount,
            risk_weight=risk_weight,
        )
        self.horizon_steps = int(horizon_steps)
        self.discount = float(discount)
        self.risk_weight = float(risk_weight)
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.episode_id = ""
        self.step_id = 0
