from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping

import numpy as np


class LabelSource(IntEnum):
    """Provenance for a candidate-level target stored in HDF5 v2."""

    INVALID = 0
    STATIC_ESDF = 1
    SCRIPTED_ACTOR = 2
    ACTOR_CONSTANT_VELOCITY = 3
    FACTUAL_EXECUTION = 4
    STATIC_DEPTH = 5


@dataclass(slots=True)
class ActorState:
    actor_id: int
    actor_type: str
    position: np.ndarray
    velocity: np.ndarray
    bbox_extent: np.ndarray
    timestamp: float
    scripted: bool = False
    frame: str = "nwu"

    def __post_init__(self) -> None:
        self.position = _vector(self.position, 3, "actor.position")
        self.velocity = _vector(self.velocity, 3, "actor.velocity")
        self.bbox_extent = _vector(self.bbox_extent, 3, "actor.bbox_extent")
        if (self.bbox_extent < 0).any():
            raise ValueError("actor bbox extents must be non-negative")


@dataclass(frozen=True, slots=True)
class VoxelGridSpec:
    minimum_flu: tuple[float, float, float] = (-4.0, -8.0, -3.0)
    maximum_flu: tuple[float, float, float] = (20.0, 8.0, 5.0)
    resolution_m: float = 0.5

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return tuple(
            int(round((upper - lower) / self.resolution_m))
            for lower, upper in zip(self.minimum_flu, self.maximum_flu)
        )

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        x, y, z = self.shape_xyz
        return z, y, x


@dataclass(slots=True)
class WorldModelPrediction:
    collision_logits: Any
    failure_logits: Any
    minimum_clearance: Any
    goal_progress: Any
    uncertainty: Any
    auxiliary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelCalibration:
    collision_temperature: float = 1.0
    failure_temperature: float = 1.0
    normalization: dict[str, tuple[float, float]] = field(default_factory=dict)


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with shape ({size},), got {array.shape}")
    return array


@dataclass(slots=True)
class VehicleState:
    timestamp: float
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    vehicle_name: str = "SimpleFlight"
    frame: str = "nwu"

    def __post_init__(self) -> None:
        self.position = _vector(self.position, 3, "position")
        self.orientation_xyzw = _vector(self.orientation_xyzw, 4, "orientation_xyzw")
        norm = float(np.linalg.norm(self.orientation_xyzw))
        if norm < 1e-8:
            raise ValueError("orientation quaternion cannot be zero")
        self.orientation_xyzw /= norm
        self.linear_velocity = _vector(self.linear_velocity, 3, "linear_velocity")
        self.angular_velocity = _vector(self.angular_velocity, 3, "angular_velocity")
        self.linear_acceleration = _vector(self.linear_acceleration, 3, "linear_acceleration")


@dataclass(slots=True)
class SensorFrame:
    timestamp: float
    depth_m: np.ndarray
    valid_mask: np.ndarray
    rgb: np.ndarray | None = None
    camera_intrinsics: np.ndarray | None = None
    camera_pose_nwu: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        if self.depth_m.ndim != 2 or self.valid_mask.shape != self.depth_m.shape:
            raise ValueError("depth_m and valid_mask must have the same [height, width] shape")
        if self.rgb is not None:
            self.rgb = np.asarray(self.rgb, dtype=np.uint8)
            if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
                raise ValueError("rgb must have shape [rgb_height, rgb_width, 3]")


@dataclass(slots=True)
class CandidateTrajectory:
    trajectory_id: str
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray | None
    duration: float
    yopo_cost: float
    valid_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    frame: str = "nwu"

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.velocities = np.asarray(self.velocities, dtype=np.float32)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape [H, 3]")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must match positions")
        if self.accelerations is not None:
            self.accelerations = np.asarray(self.accelerations, dtype=np.float32)
            if self.accelerations.shape != self.positions.shape:
                raise ValueError("accelerations must match positions")
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        if self.valid_mask.shape != (self.positions.shape[0],):
            raise ValueError("valid_mask must have shape [H]")
        if self.duration <= 0 or not np.isfinite(self.duration):
            raise ValueError("duration must be finite and positive")
        arrays = [self.positions, self.velocities]
        if self.accelerations is not None:
            arrays.append(self.accelerations)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("trajectory arrays must be finite")


@dataclass(slots=True)
class RiskPrediction:
    collision_probability: float
    minimum_clearance: float
    goal_progress: float
    failure_probability: float
    uncertainty: float
    latent_states: np.ndarray | None = None


@dataclass(slots=True)
class RerankDecision:
    selected_index: int
    original_ranking: list[int]
    reranked: list[int]
    total_scores: list[float]
    components: list[dict[str, float]]
    reason: str
    used_fallback: bool
    latency_ms: float


@dataclass(slots=True)
class TimestampedSensorFrame:
    """A synchronized observation handed from the sensor worker to YOPO."""

    sequence_id: int
    received_monotonic_s: float
    sensor: SensorFrame
    state: VehicleState

    @property
    def synchronization_error_s(self) -> float:
        return abs(float(self.sensor.timestamp) - float(self.state.timestamp))


@dataclass(slots=True)
class TrajectoryPlan:
    """Immutable planner publication consumed by the 50 Hz controller."""

    sequence_id: int
    sensor_sequence_id: int
    created_monotonic_s: float
    valid_until_monotonic_s: float
    candidates: tuple[CandidateTrajectory, ...]
    selected_index: int
    planner_latency_ms: float
    local_goal_nwu: np.ndarray
    selection_method: str = "yopo_raw_argmin"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("trajectory plan must contain candidates")
        if not 0 <= int(self.selected_index) < len(self.candidates):
            raise ValueError("selected_index is outside candidates")
        self.local_goal_nwu = _vector(self.local_goal_nwu, 3, "local_goal_nwu")
        expected = int(np.argmin([candidate.yopo_cost for candidate in self.candidates]))
        if self.selection_method == "yopo_raw_argmin" and int(self.selected_index) != expected:
            raise ValueError("pure YOPO plan must select the raw cost argmin")
        if self.valid_until_monotonic_s <= self.created_monotonic_s:
            raise ValueError("trajectory plan validity must be positive")

    @property
    def selected(self) -> CandidateTrajectory:
        return self.candidates[int(self.selected_index)]


@dataclass(slots=True)
class ControlSample:
    sequence_id: int
    monotonic_s: float
    plan_sequence_id: int
    reference_position_nwu: np.ndarray
    reference_velocity_nwu: np.ndarray
    reference_acceleration_nwu: np.ndarray
    actual_position_nwu: np.ndarray
    actual_velocity_nwu: np.ndarray
    command_velocity_nwu: np.ndarray
    desired_yaw_rad: float
    command_yaw_rate_rps: float
    mode: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "reference_position_nwu", "reference_velocity_nwu", "reference_acceleration_nwu", "actual_position_nwu",
            "actual_velocity_nwu", "command_velocity_nwu",
        ):
            setattr(self, name, _vector(getattr(self, name), 3, name))


@dataclass(slots=True)
class RecordingManifest:
    stream: str
    video_path: str
    metadata_path: str
    target_fps: float
    frame_count: int
    source_duration_s: float
    measured_source_fps: float
    dropped_frames: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ActionLimits:
    """Physical limits for the normalized UrbanFly policy action.

    Body axes use FLU: forward, left and up.  Yaw rate is positive
    counter-clockwise when viewed from above.
    """

    forward_mps: float = 6.0
    lateral_mps: float = 6.0
    vertical_mps: float = 3.0
    yaw_rate_rps: float = float(np.deg2rad(60.0))

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(
            [self.forward_mps, self.lateral_mps, self.vertical_mps, self.yaw_rate_rps],
            dtype=np.float32,
        )


@dataclass(slots=True)
class BodyVelocityAction:
    """Normalized four-dimensional action and its physical FLU command."""

    normalized: np.ndarray
    limits: ActionLimits = field(default_factory=ActionLimits)

    def __post_init__(self) -> None:
        value = np.asarray(self.normalized, dtype=np.float32)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError("normalized action must be a finite vector with shape (4,)")
        self.normalized = np.clip(value, -1.0, 1.0)

    @property
    def physical(self) -> np.ndarray:
        return self.normalized * self.limits.vector

    @classmethod
    def from_physical(
        cls,
        physical: np.ndarray,
        limits: ActionLimits | None = None,
    ) -> "BodyVelocityAction":
        action_limits = limits or ActionLimits()
        value = np.asarray(physical, dtype=np.float32)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError("physical action must be a finite vector with shape (4,)")
        return cls(value / action_limits.vector, action_limits)


@dataclass(slots=True)
class WorldModelObservation:
    """Non-privileged, step-synchronized policy observation.

    Dynamic actor states, collision geometry and zone labels deliberately do
    not live in this structure.  They belong to the privileged label stream.
    """

    episode_id: str
    step_id: int
    sim_time: float
    rgb: np.ndarray
    depth_m: np.ndarray
    depth_valid_mask: np.ndarray
    goal_body_flu_m: np.ndarray
    linear_velocity_body_flu_mps: np.ndarray
    angular_velocity_body_flu_rps: np.ndarray
    gravity_body_flu: np.ndarray
    previous_action: np.ndarray
    sensor_timestamp: float
    state_timestamp: float
    camera_intrinsics: np.ndarray
    camera_extrinsics_body: np.ndarray

    def __post_init__(self) -> None:
        self.rgb = np.asarray(self.rgb, dtype=np.uint8)
        self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
        self.depth_valid_mask = np.asarray(self.depth_valid_mask, dtype=bool)
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape [height, width, 3]")
        if self.depth_m.ndim != 2 or self.depth_valid_mask.shape != self.depth_m.shape:
            raise ValueError("depth and depth_valid_mask must share shape [height, width]")
        for name in (
            "goal_body_flu_m",
            "linear_velocity_body_flu_mps",
            "angular_velocity_body_flu_rps",
            "gravity_body_flu",
        ):
            setattr(self, name, _vector(getattr(self, name), 3, name))
        self.previous_action = _vector(self.previous_action, 4, "previous_action")
        self.camera_intrinsics = np.asarray(self.camera_intrinsics, dtype=np.float32)
        self.camera_extrinsics_body = np.asarray(
            self.camera_extrinsics_body, dtype=np.float32
        )
        if self.camera_intrinsics.shape != (3, 3):
            raise ValueError("camera_intrinsics must have shape (3, 3)")
        if self.camera_extrinsics_body.shape != (4, 4):
            raise ValueError("camera_extrinsics_body must have shape (4, 4)")
        if abs(float(self.sensor_timestamp) - float(self.state_timestamp)) > 0.11:
            raise ValueError("sensor/state synchronization error exceeds 110 ms")


@dataclass(slots=True)
class SafetyAudit:
    episode_id: str
    step_id: int
    sim_time: float
    raw_action_normalized: np.ndarray
    raw_action_physical: np.ndarray
    executed_action_physical: np.ndarray
    intervened: bool
    reasons: tuple[str, ...]
    action_delta_l2: float
    minimum_depth_m: float
    predicted_risk: float

    def __post_init__(self) -> None:
        self.raw_action_normalized = _vector(
            self.raw_action_normalized, 4, "raw_action_normalized"
        )
        self.raw_action_physical = _vector(
            self.raw_action_physical, 4, "raw_action_physical"
        )
        self.executed_action_physical = _vector(
            self.executed_action_physical, 4, "executed_action_physical"
        )


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Immutable identity and randomization contract for one v3 rollout."""

    episode_id: str
    route_id: str
    split: str
    tile_ids: tuple[str, ...]
    scenario: str
    seed: int
    start_nwu_m: tuple[float, float, float]
    goal_nwu_m: tuple[float, float, float]
    appearance_parameters: Mapping[str, float] = field(default_factory=dict)
    dynamics_parameters: Mapping[str, float] = field(default_factory=dict)
    actor_script_id: str = ""
    counterfactual_parent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.route_id:
            raise ValueError("episode_id and route_id cannot be empty")
        if self.split not in {"train", "validation", "test", "calibration"}:
            raise ValueError("split must be train, validation, test, or calibration")
        if not self.tile_ids:
            raise ValueError("tile_ids cannot be empty")
        for name, value in (("start_nwu_m", self.start_nwu_m), ("goal_nwu_m", self.goal_nwu_m)):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite three-vector")


@dataclass(slots=True)
class CandidatePrediction:
    """Shared candidate-level output used by every world-model assistant."""

    goal_progress: float
    collision_probability: float
    minimum_clearance: float
    cpa_risk: float
    terminal_value: float
    epistemic_uncertainty: float
    failure_probability: float
    predicted_state_1s_2s_3s: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "goal_progress", "collision_probability", "minimum_clearance",
            "cpa_risk", "terminal_value", "epistemic_uncertainty",
            "failure_probability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in ("collision_probability", "cpa_risk", "epistemic_uncertainty", "failure_probability"):
            setattr(self, name, float(np.clip(getattr(self, name), 0.0, 1.0)))
        states = np.asarray(self.predicted_state_1s_2s_3s, dtype=np.float32)
        if states.shape != (3, 3) or not np.isfinite(states).all():
            raise ValueError("predicted_state_1s_2s_3s must have shape (3, 3)")
        self.predicted_state_1s_2s_3s = states
