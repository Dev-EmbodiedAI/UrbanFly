import numpy as np

from uav_wm_navigation.control import SafetyFilter, TrajectoryExecutor
from uav_wm_navigation.simulators import MockSimulator
from uav_wm_navigation.types import CandidateTrajectory


def make_lateral_candidate() -> CandidateTrajectory:
    times = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    positions = np.column_stack([2.0 * times, times, np.full_like(times, 2.0)])
    velocities = np.tile(np.array([2.0, 1.0, 0.0], dtype=np.float32), (6, 1))
    return CandidateTrajectory(
        "lateral", positions, velocities, np.zeros_like(positions), 1.0, 0.0, np.ones(6, dtype=bool)
    )


def test_route_aligned_controller_damps_lateral_command_and_yaw_deadband() -> None:
    simulator = MockSimulator(seed=4)
    simulator.connect()
    simulator.takeoff()
    executor = TrajectoryExecutor(
        simulator, SafetyFilter({"max_acceleration_mps2": 100.0, "max_yaw_rate_rps": 10.0}),
        control_dt=0.1, position_kp=0.0, yaw_kp=1.4, yaw_deadband_degrees=2.0,
        route_lateral_velocity_scale=0.25,
    )
    records = executor.execute_prefix(
        make_lateral_candidate(), 0.1, heading_target_nwu=np.array([10.0, 0.01, 2.0])
    )
    assert len(records) == 1
    desired = np.asarray(records[0]["desired_velocity_nwu"])
    assert desired[1] < 0.3
    assert records[0]["raw_yaw_rate_rps"] == 0.0


def test_velocity_low_pass_reduces_command_reversal() -> None:
    simulator = MockSimulator(seed=5)
    simulator.connect()
    simulator.takeoff()
    executor = TrajectoryExecutor(
        simulator, SafetyFilter({"max_acceleration_mps2": 100.0}), control_dt=0.1,
        position_kp=0.0, velocity_smoothing_alpha=0.25,
    )
    positive = make_lateral_candidate()
    negative = make_lateral_candidate()
    negative.velocities[:, 1] *= -1.0
    first = executor.execute_prefix(positive, 0.1)[0]
    second = executor.execute_prefix(negative, 0.1)[0]
    assert first["desired_velocity_nwu"][1] > 0.9
    assert second["desired_velocity_nwu"][1] > 0.0
