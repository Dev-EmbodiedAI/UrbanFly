import numpy as np

from backend.config import MULTIROTOR_DYNAMICS
from backend.engine.collision import HeightmapStaticCollisionMap
from backend.engine.multirotor_dynamics import (
    MultirotorDynamics,
    MultirotorParameters,
)
from backend.engine.scenario import ScenarioEngine
from backend.engine.simulator import Simulator
from backend.engine.urban_world_model import (
    LocalHeightBelief,
    UrbanWorldModelConfig,
    UrbanWorldModelMPC,
)


def _wall_map() -> HeightmapStaticCollisionMap:
    height = np.zeros((101, 101), dtype=np.float32)
    # X=8..18 m, Z=-12..12 m: a roof higher than the nominal flight level.
    height[38:63, 58:69] = 30.0
    return HeightmapStaticCollisionMap(
        height=height,
        origin_x=-50.0,
        origin_z=-50.0,
        resolution=1.0,
    )


def _standard_dynamics() -> MultirotorDynamics:
    model = MultirotorDynamics(
        MultirotorParameters.from_dict(MULTIROTOR_DYNAMICS["standard"])
    )
    model.initialize(0.0)
    return model


def test_sensor_limited_belief_stops_at_blocking_roof():
    config = UrbanWorldModelConfig(
        sensor_range_m=60.0,
        sensor_ray_count=31,
        belief_resolution_m=1.0,
    )
    belief = LocalHeightBelief(_wall_map(), config)
    belief.observe(
        np.array([0.0, 20.0, 0.0]),
        yaw_degrees=0.0,
        simulation_time_s=0.0,
    )

    assert belief.surface_height(np.array([8.0, 20.0, 0.0])) == 30.0
    assert belief.surface_height(np.array([45.0, 20.0, 0.0])) is None


def test_vectorized_rollouts_match_independent_6dof_clones():
    config = UrbanWorldModelConfig(horizon_s=1.0, rollout_dt_s=0.25)
    controller = UrbanWorldModelMPC(_wall_map(), config)
    model = _standard_dynamics()
    position = np.array([1.0, 18.0, -2.0])
    velocity = np.array([1.2, -0.1, 0.4])
    wind = np.array([0.8, 0.0, -0.3])
    commands = np.array(
        [
            [4.0, 0.0, 0.0],
            [6.0, 1.5, 2.0],
            [0.0, -1.0, 0.0],
        ]
    )

    scalar = np.asarray([
        controller._rollout(
            command=command,
            position=position,
            velocity=velocity,
            yaw_degrees=12.0,
            dynamics_model=model,
            wind_velocity=wind,
            payload_mass=0.7,
            max_acceleration=4.0,
        )
        for command in commands
    ])
    vectorized = controller._rollout_batch(
        commands=commands,
        position=position,
        velocity=velocity,
        yaw_degrees=12.0,
        dynamics_model=model,
        wind_velocity=wind,
        payload_mass=0.7,
        max_acceleration=4.0,
    )

    assert np.allclose(vectorized, scalar, rtol=1e-10, atol=1e-10)


def test_world_model_imagines_safe_command_around_wall():
    config = UrbanWorldModelConfig(
        planning_interval_s=0.1,
        horizon_s=3.0,
        rollout_dt_s=0.25,
        sensor_range_m=60.0,
        sensor_ray_count=41,
        belief_resolution_m=1.0,
    )
    controller = UrbanWorldModelMPC(_wall_map(), config)
    decision = controller.plan(
        simulation_time_s=0.0,
        position_world_m=np.array([0.0, 20.0, 0.0]),
        velocity_world_mps=np.zeros(3),
        yaw_degrees=0.0,
        goal_world_m=np.array([60.0, 20.0, 0.0]),
        dynamics_model=_standard_dynamics(),
        wind_velocity=np.zeros(3),
        payload_mass=0.0,
        max_acceleration=3.0,
    )

    command = np.asarray(decision["command_world_mps"])
    assert decision["candidate_count"] > 20
    assert decision["safe_candidate_count"] > 0
    assert decision["predicted_collision"] is False
    # It either diverts/climbs or slows enough to remain outside the imagined
    # collision horizon.  Here the optimal first action is a deliberate brake.
    assert (
        abs(command[1]) > 0.1
        or abs(command[2]) > 0.1
        or command[0] < 9.0
    )
    assert len(decision["selected_trajectory_world_m"]) == 12


def test_world_model_scenario_integrates_with_single_uav_simulator():
    scenario = ScenarioEngine.create_default().get_scenario(
        "single_uav_world_model"
    )
    assert scenario is not None
    assert len(scenario.drones) == 1

    simulator = Simulator(static_collision_map=_wall_map())
    simulator.initialize_scenario(scenario)
    drone = simulator.drones[0]
    assert drone.id == "WM-UAV-01"
    assert drone.id in simulator._world_model_controllers

    for _ in range(20):
        simulator.step()

    state = drone.world_model_state
    assert state["enabled"] is True
    assert state["backend"] == "city_belief_multirotor_mpc_v1"
    assert state["decision_sequence"] >= 1
    assert state["belief_cell_count"] > 0
    assert len(state["top_candidates"]) > 0
    assert np.all(np.isfinite(drone.orientation_quaternion))
