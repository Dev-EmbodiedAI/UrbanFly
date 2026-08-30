from __future__ import annotations

import numpy as np

from uav_wm_navigation.evaluation import (
    aggregate_navigation_metrics,
    navigation_error,
    polyline_length,
    success_weighted_path_length,
)
from uav_wm_navigation.types import CandidateTrajectory, VehicleState
from uav_wm_navigation.world_models import candidate_actions_body_flu


def test_sr_ne_spl_definitions() -> None:
    path = np.asarray(
        [[0.0, 0.0, 2.0], [3.0, 0.0, 2.0], [3.0, 4.0, 2.0]],
        dtype=np.float32,
    )
    assert polyline_length(path) == 7.0
    assert navigation_error(path[-1], np.asarray([3.0, 5.0, 2.0])) == 1.0
    assert success_weighted_path_length(True, 7.0, 5.0) == 5.0 / 7.0
    assert success_weighted_path_length(False, 5.0, 5.0) == 0.0

    aggregate = aggregate_navigation_metrics(
        np.asarray([True, False, True]),
        np.asarray([0.5, 4.0, 1.0]),
        np.asarray([0.8, 0.0, 1.0]),
    )
    assert aggregate["episodes"] == 3
    assert aggregate["successes"] == 2
    assert np.isclose(aggregate["sr"], 2.0 / 3.0)
    assert np.isclose(aggregate["ne_m"], 5.5 / 3.0)
    assert np.isclose(aggregate["spl"], 0.6)


def test_tdmpc2_candidate_conversion_is_normalized() -> None:
    steps = 9
    candidate = CandidateTrajectory(
        trajectory_id="straight",
        positions=np.column_stack(
            [
                np.linspace(0.0, 12.0, steps),
                np.zeros(steps),
                np.full(steps, 2.0),
            ]
        ),
        velocities=np.tile(np.asarray([9.0, 0.0, 0.0]), (steps, 1)),
        accelerations=np.zeros((steps, 3)),
        duration=2.0,
        yopo_cost=1.0,
        valid_mask=np.ones(steps, dtype=bool),
    )
    state = VehicleState(
        timestamp=0.0,
        position=np.asarray([0.0, 0.0, 2.0]),
        orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    actions = candidate_actions_body_flu(
        candidate,
        state,
        horizon_steps=15,
    )
    assert actions.shape == (15, 4)
    assert np.isfinite(actions).all()
    assert np.max(np.abs(actions)) <= 1.0
    assert np.allclose(actions[:, 0], 1.0)
    assert np.allclose(actions[:, 1:], 0.0)
