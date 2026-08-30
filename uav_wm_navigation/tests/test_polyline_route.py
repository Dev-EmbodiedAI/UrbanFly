from __future__ import annotations

import numpy as np

from uav_wm_navigation.control import PolylineRoute
from uav_wm_navigation.planners.yopo_adapter import build_yopo_observation
from uav_wm_navigation.types import VehicleState


def _state(position=(0.0, 0.0, 60.0), velocity=(4.0, 0.0, 0.0)) -> VehicleState:
    return VehicleState(
        timestamp=0.0,
        position=np.asarray(position, dtype=np.float64),
        orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=np.asarray(velocity, dtype=np.float64),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
    )


def test_polyline_projection_local_goal_and_monotonic_progress() -> None:
    route = PolylineRoute(np.asarray([
        [0.0, 0.0, 60.0],
        [40.0, 0.0, 60.0],
        [70.0, 30.0, 60.0],
        [110.0, 30.0, 60.0],
    ]))
    first = route.observe(np.asarray([10.0, 2.0, 60.0]), speed_mps=4.0)
    assert np.isclose(first.progress_m, 10.0)
    assert np.isclose(first.cross_track_error_m, 2.0)
    assert first.segment_index == 0
    assert np.isclose(first.lookahead_m, 18.0)
    assert np.allclose(first.local_goal_nwu, [28.0, 0.0, 60.0])
    regressed = route.observe(np.asarray([8.0, 1.0, 60.0]), speed_mps=4.0)
    assert regressed.progress_m == first.progress_m
    assert regressed.nearest_progress_m < regressed.progress_m


def test_polyline_turn_awareness_shortens_lookahead() -> None:
    route = PolylineRoute(np.asarray([
        [0.0, 0.0, 60.0],
        [40.0, 0.0, 60.0],
        [60.0, 25.0, 60.0],
    ]))
    observation = route.observe(np.asarray([25.0, 0.0, 60.0]), speed_mps=6.0)
    assert observation.distance_to_turn_m == 15.0
    assert observation.lookahead_m == 12.0
    assert np.allclose(observation.local_goal_nwu, [37.0, 0.0, 60.0])


def test_dynamic_route_goal_enters_last_three_yopo_observation_values() -> None:
    state = _state()
    desired = state.position.copy()
    first_goal = np.asarray([18.0, 0.0, 60.0])
    turn_goal = np.asarray([12.0, 12.0, 60.0])
    first = build_yopo_observation(state, first_goal, desired, np.zeros(3))
    turn = build_yopo_observation(state, turn_goal, desired, np.zeros(3))
    assert first.shape == (9,)
    assert np.allclose(first[-3:], [18.0, 0.0, 0.0])
    assert np.allclose(turn[-3:], [12.0, 12.0, 0.0])
    assert not np.allclose(first[-3:], turn[-3:])
