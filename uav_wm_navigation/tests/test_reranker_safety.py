import numpy as np

from uav_wm_navigation.control import RiskReranker, SafetyFilter
from uav_wm_navigation.types import CandidateTrajectory, RiskPrediction, SensorFrame, VehicleState


def candidates():
    return [CandidateTrajectory(str(i), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), 1.0, float(cost), np.ones(3, bool)) for i, cost in enumerate([0.1, 0.5, 0.9])]


def predictions(high=False):
    risk = [0.95, 0.95, 0.95] if high else [0.8, 0.05, 0.2]
    return [RiskPrediction(risk[i], 1.0 + i, 0.2 + i, risk[i], 0.01 * i) for i in range(3)]


def test_reranker_selects_safe_candidate_and_fallbacks() -> None:
    reranker = RiskReranker({"yopo": 1, "collision": 5, "progress": 1, "clearance": 1, "failure": 2, "uncertainty": 1})
    decision = reranker.rank(candidates(), predictions())
    assert decision.selected_index == 1
    assert not decision.used_fallback
    high = reranker.rank(candidates(), predictions(high=True))
    assert high.selected_index == -1 and high.used_fallback
    invalid = predictions()
    invalid[0].uncertainty = float("nan")
    assert reranker.rank(candidates(), invalid).reason == "model_invalid_or_timeout"


def test_safety_filter_depth_stop_and_limits() -> None:
    state = VehicleState(0, np.array([0, 0, 2]), np.array([0, 0, 0, 1]), np.zeros(3), np.zeros(3))
    safe = SafetyFilter({"max_speed_mps": 2, "max_acceleration_mps2": 100, "emergency_depth_m": 1, "slow_depth_m": 3})
    clear = SensorFrame(0, np.full((4, 4), 10.0), np.ones((4, 4), bool))
    result = safe.apply(np.array([5, 0, 0]), 5.0, state, clear, 0.1)
    assert np.linalg.norm(result.velocity) <= 2.0 + 1e-6
    assert abs(result.yaw_rate) <= safe.max_yaw_rate
    blocked = SensorFrame(0, np.full((4, 4), 0.5), np.ones((4, 4), bool))
    assert safe.apply(np.ones(3), 0, state, blocked, 0.1).mode == "hover"


def test_safety_filter_uses_configured_forward_depth_roi() -> None:
    state = VehicleState(0, np.array([0, 0, 2]), np.array([0, 0, 0, 1]), np.zeros(3), np.zeros(3))
    safe = SafetyFilter({
        "max_acceleration_mps2": 100, "emergency_depth_m": 1.0, "slow_depth_m": 2.0,
        "depth_roi_fraction": [0.25, 0.75, 0.25, 0.75],
    })
    depth = np.full((8, 8), 0.4)
    depth[2:6, 2:6] = 8.0
    margin_obstacle = SensorFrame(0, depth, np.ones((8, 8), bool))
    assert safe.apply(np.ones(3), 0, state, margin_obstacle, 0.1).mode != "hover"
    depth[3:5, 3:5] = 0.3
    central_obstacle = SensorFrame(0, depth, np.ones((8, 8), bool))
    assert safe.apply(np.ones(3), 0, state, central_obstacle, 0.1).mode == "hover"
