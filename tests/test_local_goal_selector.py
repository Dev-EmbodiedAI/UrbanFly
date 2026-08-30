import numpy as np
import pytest

from backend.engine.local_goal import LocalGoalSelector


def test_local_goal_uses_arc_length_and_returns_remaining_route():
    selector = LocalGoalSelector(lookahead_distance_m=20.0)
    path = np.array([[0.0, 5.0, 0.0], [30.0, 5.0, 0.0], [30.0, 10.0, 30.0]])
    result = selector.select(
        current_position_world=np.array([5.0, 5.0, 2.0]),
        current_velocity_world=np.zeros(3),
        yaw_degrees=0.0,
        global_path=path,
    )
    np.testing.assert_allclose(result.local_goal_world, [25.0, 5.0, 0.0], atol=1e-6)
    # Backend +Z is geographic south, so a target 2 m toward -Z is left/north.
    np.testing.assert_allclose(result.local_goal_body_flu, [20.0, 2.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.remaining_global_path[0], [5.0, 5.0, 0.0], atol=1e-6)
    assert result.route_progress_m == pytest.approx(5.0)


def test_local_goal_body_flu_rotates_with_yaw():
    selector = LocalGoalSelector(lookahead_distance_m=10.0)
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0]])
    result = selector.select(np.zeros(3), np.zeros(3), 90.0, path)
    np.testing.assert_allclose(result.local_goal_world, [0.0, 0.0, 10.0], atol=1e-6)
    np.testing.assert_allclose(result.local_goal_body_flu, [10.0, 0.0, 0.0], atol=1e-6)


def test_local_goal_rejects_out_of_range_lookahead():
    with pytest.raises(ValueError):
        LocalGoalSelector(lookahead_distance_m=5.0)
