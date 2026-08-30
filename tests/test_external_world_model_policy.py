import numpy as np

from backend.engine.scenario import ScenarioEngine
from backend.engine.simulator import Simulator
from backend.config import MULTIROTOR_DYNAMICS


def test_external_policy_action_is_auditable_and_times_out_to_hover():
    scenario = ScenarioEngine.create_default().get_scenario(
        "single_uav_world_model"
    )
    simulator = Simulator()
    simulator.initialize_scenario(scenario)
    drone = simulator.drones[0]

    accepted = simulator.set_external_policy_action(
        drone.id,
        [0.5, -0.25, 0.2, 0.1],
        step_id=0,
        policy_family="tdmpc2_continuous",
        inference_latency_ms=42.0,
        predicted_risk=0.2,
        shield_enabled=True,
        timeout_s=0.2,
    )
    assert accepted["raw_action_normalized"] == [0.5, -0.25, 0.2, 0.1]
    simulator.step()
    telemetry = drone.world_model_state
    assert telemetry["status"] == "external_learned_policy"
    assert telemetry["backend"] == "tdmpc2_continuous"
    assert telemetry["raw_action_normalized"] == [0.5, -0.25, 0.2, 0.1]
    assert len(telemetry["command_world_mps"]) == 3
    assert telemetry["inference_latency_ms"] == 42.0
    assert np.isfinite(drone.velocity).all()

    for _ in range(12):
        simulator.step()
    telemetry = drone.world_model_state
    assert telemetry["status"] == "policy_timeout_hover"
    assert telemetry["stale_action"] is True
    assert telemetry["command_world_mps"] == [0.0, 0.0, 0.0]
    assert telemetry["safety_intervention_reasons"] == [
        "policy_command_timeout"
    ]


def test_positive_flu_yaw_rate_increases_canonical_enu_yaw():
    scenario = ScenarioEngine.create_default().get_scenario("single_uav_world_model")
    simulator = Simulator()
    simulator.initialize_scenario(scenario)
    drone = simulator.drones[0]
    initial_yaw_enu = -np.deg2rad(drone.yaw)
    simulator.set_external_policy_action(
        drone.id,
        [0.0, 0.0, 0.0, 0.75],
        step_id=0,
        policy_family="yaw_direction_test",
        shield_enabled=False,
        timeout_s=1.0,
    )
    for _ in range(20):
        simulator.step()
    final_yaw_enu = -np.deg2rad(drone.yaw)
    assert final_yaw_enu > initial_yaw_enu + np.deg2rad(2.0)
    assert drone.world_model_state["executed_action_physical_body_flu"][3] > 0.0
    assert drone.world_model_state["yaw_rate_backend_degrees_s"] < 0.0


def test_external_episode_applies_replayable_appearance_and_dynamics_perturbations():
    scenario = ScenarioEngine.create_default().get_scenario("single_uav_world_model")
    simulator = Simulator()
    simulator.initialize_scenario(scenario)
    drone = simulator.drones[0]
    start = drone.position.copy()
    goal = start + np.asarray([20.0, 0.0, 0.0])
    configured = simulator.configure_external_policy_episode(
        drone.id, start_world_m=start, goal_world_m=goal,
        episode_seed=303, dynamic_actor_density=1.5,
        appearance_perturbation={
            "exposure_ev": -0.5, "fog_density": 0.05,
            "color_temperature_k": 4500, "camera_noise_std": 0.01,
            "frame_drop_probability": 0.05,
        },
        dynamics_perturbation={
            "wind_world_mps": [2.0, 0.0, -1.0], "mass_scale": 1.2,
            "drag_scale": 1.3, "motor_delay_ms": 60, "control_jitter_ms": 10,
        },
    )
    assert configured["episode_seed"] == 303
    assert configured["appearance_perturbation"]["fog_density"] == 0.05
    assert np.allclose(simulator._episode_wind_offset, [2.0, 0.0, -1.0])
    parameters = simulator._multirotor_models[drone.id].parameters
    baseline = MULTIROTOR_DYNAMICS[drone.drone_type]
    assert np.isclose(parameters.mass, baseline["mass"] * 1.2)
    assert np.allclose(parameters.linear_drag, np.asarray(baseline["linear_drag"]) * 1.3)
    assert np.isclose(parameters.motor_time_constant, baseline["motor_time_constant"] + 0.06)
    snapshot = simulator.get_state_snapshot()
    assert snapshot["appearance_perturbation"]["camera_noise_std"] == 0.01
    assert snapshot["dynamics_perturbation"]["control_jitter_ms"] == 10.0


def test_external_planner_visualization_exposes_all_candidates_without_changing_control():
    scenario = ScenarioEngine.create_default().get_scenario("single_uav_world_model")
    simulator = Simulator()
    simulator.initialize_scenario(scenario)
    drone = simulator.drones[0]
    command = simulator.set_external_policy_action(
        drone.id, [0.2, 0.0, 0.0, 0.0], step_id=0,
        policy_family="yopo_tdmpc2", shield_enabled=False,
    )
    origin = drone.position.copy()
    candidates = []
    for index in range(15):
        points = [
            origin.tolist(),
            (origin + np.asarray([5.0, 0.0, float(index - 7)])).tolist(),
        ]
        candidates.append({
            "candidate_index": index,
            "score": float(index),
            "collision_probability": index / 20.0,
            "uncertainty": 0.01 * index,
            "predicted_collision": index >= 10,
            "trajectory_world_m": points,
        })
    accepted = simulator.set_external_policy_visualization(drone.id, {
        "decision_sequence": 0,
        "candidate_count": 15,
        "selected_index": 3,
        "raw_selected_index": 1,
        "selection_method": "tdmpc2_visual_rerank",
        "selected_trajectory_world_m": candidates[3]["trajectory_world_m"],
        "top_candidates": candidates,
        "planner_latency_ms": 24.0,
        "predicted_risk": 0.15,
    })
    assert accepted["visualization_only"] is True
    simulator.step()
    snapshot = simulator.get_state_snapshot()["drones"][0]["world_model"]
    assert snapshot["candidate_count"] == 15
    assert len(snapshot["top_candidates"]) == 15
    assert snapshot["selected_index"] == 3
    assert snapshot["raw_action_normalized"] == command["raw_action_normalized"]
    assert simulator._external_policy_commands[drone.id]["step_id"] == 0
