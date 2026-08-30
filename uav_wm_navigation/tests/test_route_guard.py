import numpy as np

from uav_wm_navigation.control import RouteManager, rank_route_consistent_candidates
from uav_wm_navigation.types import CandidateTrajectory


def candidate(name: str, endpoint: tuple[float, float, float]) -> CandidateTrajectory:
    points = np.linspace(np.zeros(3), np.asarray(endpoint, dtype=np.float32), 16)
    velocity = np.gradient(points, axis=0).astype(np.float32)
    return CandidateTrajectory(name, points, velocity, np.zeros_like(points), 2.0, 0.0, np.ones(16, bool))


def test_route_guard_rejects_low_cost_backtracking_and_large_lateral_motion() -> None:
    route = RouteManager(np.asarray([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))
    route.update(np.zeros(3))
    candidates = [candidate("backward", (-4.0, 0.0, 0.0)), candidate("forward", (6.0, 0.2, 0.0)),
                  candidate("lateral", (5.0, 5.0, 0.0))]
    selected, ranking, scores, metrics, reason = rank_route_consistent_candidates(
        candidates, route, np.asarray([0.0, 0.4, 0.1]),
        {"maximum_lateral_m": 2.0, "minimum_endpoint_progress_m": 0.5, "maximum_regression_m": 0.2},
    )
    assert selected == 1
    assert ranking == [1]
    assert np.isinf(scores[0]) and np.isinf(scores[2])
    assert metrics[0]["endpoint_progress_m"] <= 0.0
    assert reason == "route_consistent"


def test_route_projection_can_measure_regression_without_changing_monotonic_progress() -> None:
    route = RouteManager(np.asarray([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))
    route.update(np.asarray([10.0, 0.0, 0.0]))
    projected, lateral = route.project_nearest(np.asarray([8.0, 1.0, 0.0]))
    assert projected == 8.0
    assert lateral == 1.0
    assert route.progress_m == 10.0


def test_route_guard_hysteresis_retains_an_eligible_previous_primitive() -> None:
    route = RouteManager(np.asarray([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))
    route.update(np.zeros(3))
    candidates = [candidate("previous", (6.0, 0.2, 0.0)), candidate("noisy_winner", (6.1, -0.2, 0.0))]
    selected, *_ = rank_route_consistent_candidates(
        candidates, route, np.asarray([0.25, 0.20]),
        {"maximum_lateral_m": 2.0, "hysteresis_bonus": 1.05}, previous_trajectory_id="previous",
    )
    assert selected == 0
