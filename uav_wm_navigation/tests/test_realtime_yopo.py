from __future__ import annotations

import time

import h5py
import numpy as np

from uav_wm_navigation.control import LatestValue, RealtimeYOPORunner, SafetyFilter
from uav_wm_navigation.planners.polynomial import sample_quintic
from uav_wm_navigation.simulators import MockSimulator
from uav_wm_navigation.types import CandidateTrajectory, TrajectoryPlan


class DeterministicFifteenPlanner:
    candidate_count = 15

    def plan(self, context):
        result = []
        for index in range(15):
            lateral = (index - 7) * 0.08
            end = context.state.position + np.array([4.0, lateral, 0.0])
            positions, velocities, accelerations = sample_quintic(
                context.state.position, context.state.linear_velocity, np.zeros(3),
                end, np.array([2.5, 0.0, 0.0]), np.zeros(3), 2.0, 21,
            )
            result.append(CandidateTrajectory(
                f"fake-{index}", positions, velocities, accelerations, 2.0,
                float(abs(index - 7)), np.ones(21, dtype=bool), metadata={"source": "test"},
            ))
        return result


def test_latest_value_drops_queue_history() -> None:
    mailbox: LatestValue[int] = LatestValue()
    mailbox.publish(1)
    mailbox.publish(2)
    value, version = mailbox.get()
    assert value == 2
    assert version == 2


def test_trajectory_plan_rejects_non_argmin_selection() -> None:
    candidates = tuple(DeterministicFifteenPlanner().plan(type("C", (), {
        "state": MockSimulator().get_kinematics(),
    })()))
    with np.testing.assert_raises(ValueError):
        TrajectoryPlan(0, 0, 1.0, 2.0, candidates, 0, 1.0, np.array([4.0, 0.0, 2.0]))


def test_world_model_plan_may_only_explicitly_rerank_yopo_candidates() -> None:
    candidates = tuple(DeterministicFifteenPlanner().plan(type("C", (), {
        "state": MockSimulator().get_kinematics(),
    })()))
    plan = TrajectoryPlan(
        0, 0, 1.0, 2.0, candidates, 6, 4.0, np.array([4.0, 0.0, 2.0]),
        selection_method="dreamerv3_rerank", metadata={"raw_selected_index": 7},
    )
    assert plan.selected_index == 6
    assert len(plan.candidates) == 15
    assert plan.metadata["raw_selected_index"] == 7


def test_realtime_runner_decouples_workers_and_saves_compact_telemetry(tmp_path) -> None:
    simulator = MockSimulator(seed=2, scenario="OpenSpace", control_dt=0.02)
    simulator.connect()
    simulator.takeoff()
    safety = SafetyFilter({
        "max_speed_mps": 3.0, "max_acceleration_mps2": 20.0, "max_yaw_rate_rps": 1.5,
        "target_altitude_m": 2.0, "emergency_depth_m": 0.4, "slow_depth_m": 1.2,
        "collision_debounce_steps": 1,
    })
    runner = RealtimeYOPORunner(
        simulator, DeterministicFifteenPlanner(), safety, np.array([20.0, 0.0, 2.0]),
        {"sensor_hz": 40.0, "planner_hz": 20.0, "control_hz": 50.0,
         "sensor_stale_s": 0.3, "plan_stale_s": 0.5, "handoff_s": 0.05},
    )
    runner.start()
    time.sleep(0.55)
    runner.stop("test_complete")
    assert not runner.errors
    assert len(runner.sensor_records) >= 10
    assert len(runner.plan_records) >= 5
    assert len(runner.control_records) >= 15
    assert all(len(item["costs"]) == 15 for item in runner.plan_records)
    assert all(item["selected_index"] == 7 for item in runner.plan_records)
    telemetry = tmp_path / "telemetry.h5"
    runner.save_telemetry(telemetry)
    with h5py.File(telemetry, "r") as handle:
        assert handle.attrs["schema"] == "uav-wm-nav-yopo-realtime-v1"
        assert handle["plans/candidate_positions_nwu"].shape[1:] == (15, 21, 3)
        assert handle["plans/depth_mm"].dtype == np.uint16
        assert handle["control/actual_position_nwu"].shape[1] == 3
