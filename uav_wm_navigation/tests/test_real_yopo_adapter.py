from pathlib import Path

import numpy as np
import pytest

from uav_wm_navigation.planners import PlanningContext, YOPOAdapter
from uav_wm_navigation.simulators import MockSimulator


ROOT = Path(__file__).resolve().parents[2]
YOPO_ROOT = ROOT / "YOPO-YOPO-Simple" / "YOPO"
CHECKPOINT = YOPO_ROOT / "saved/urbanfly/YOPO_0/epoch24.pth"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="local real YOPO checkpoint is unavailable")
def test_real_yopo_weight_exposes_fifteen_continuous_candidates() -> None:
    simulator = MockSimulator()
    simulator.connect()
    simulator.takeoff()
    planner = YOPOAdapter({
        "yopo_root": str(YOPO_ROOT), "checkpoint": str(CHECKPOINT), "device": "cuda",
        "velocity": 4.5, "horizon_steps": 11, "depth_max_m": 20.0, "min_depth_m": 0.8,
    })
    state, sensor = simulator.get_kinematics(), simulator.get_depth()
    candidates = planner.plan(PlanningContext(sensor, state, np.array([12.0, 0.0, 2.0])))
    assert len(candidates) == planner.candidate_count == 15
    assert all(item.positions.shape == (11, 3) for item in candidates)
    assert all(np.isfinite(item.yopo_cost) for item in candidates)
    assert all(np.allclose(item.positions[0], state.position, atol=1e-4) for item in candidates)
    assert {item.metadata["source"] for item in candidates} == {"yopo"}
