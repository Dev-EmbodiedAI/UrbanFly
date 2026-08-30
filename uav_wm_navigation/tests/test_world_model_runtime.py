from collections import deque
import threading

import numpy as np
import torch

from uav_wm_navigation.types import CandidateTrajectory, SensorFrame, TimestampedSensorFrame, VehicleState
from uav_wm_navigation.world_models.runtime import CandidateWorldModelRuntime


def _runtime_without_checkpoint() -> CandidateWorldModelRuntime:
    runtime = CandidateWorldModelRuntime.__new__(CandidateWorldModelRuntime)
    runtime.device = torch.device("cpu")
    runtime.history = 4
    runtime.depth_max_m = 20.0
    runtime.trajectory_steps = 16
    runtime.local_goal_lookahead_m = 10.0
    runtime._depth = deque(maxlen=runtime.history)
    runtime._state = deque(maxlen=runtime.history)
    runtime._history_lock = threading.Lock()
    return runtime


def _frame_and_candidates() -> tuple[TimestampedSensorFrame, tuple[CandidateTrajectory, ...]]:
    state = VehicleState(
        timestamp=1.0,
        position=np.zeros(3),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    sensor = SensorFrame(
        timestamp=1.0,
        depth_m=np.full((96, 160), 8.0, dtype=np.float32),
        valid_mask=np.ones((96, 160), dtype=bool),
    )
    frame = TimestampedSensorFrame(0, 1.0, sensor, state)
    positions = np.stack([np.linspace(0.0, 5.0, 21), np.zeros(21), np.zeros(21)], axis=-1)
    velocities = np.gradient(positions, axis=0)
    candidates = tuple(
        CandidateTrajectory(
            trajectory_id=str(index),
            positions=positions,
            velocities=velocities,
            accelerations=np.zeros_like(positions),
            duration=2.0,
            yopo_cost=float(index),
            valid_mask=np.ones(21, dtype=bool),
        )
        for index in range(15)
    )
    return frame, candidates


def test_history_reset_is_atomic_with_realtime_input_snapshot() -> None:
    runtime = _runtime_without_checkpoint()
    frame, candidates = _frame_and_candidates()
    errors: list[BaseException] = []

    def prepare_inputs() -> None:
        try:
            for _ in range(100):
                depth, state, goal, trajectories = runtime._inputs(frame, candidates, np.array([10.0, 0.0, 0.0]))
                assert depth.shape == (1, 4, 1, 96, 160)
                assert state.shape == (1, 4, 13)
                assert goal.shape == (1, 3)
                assert trajectories.shape == (1, 15, 16, 9)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=prepare_inputs)
    worker.start()
    for _ in range(100):
        runtime.reset()
    worker.join()
    assert errors == []

