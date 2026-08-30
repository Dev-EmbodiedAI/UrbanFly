from __future__ import annotations

import numpy as np
import torch

from uav_wm_navigation.control import SafetyFilter, TrajectoryExecutor, WAMMPCController
from uav_wm_navigation.data import WAMMPCTransitionDataset, collect_episode
from uav_wm_navigation.envs.urbanfly_world_model_env import UrbanFlyEnvConfig, UrbanFlyWorldModelEnv
from uav_wm_navigation.planners import MPPICostWeights, MPPIPlanner
from uav_wm_navigation.planners import MockCandidatePlanner
from uav_wm_navigation.simulators import MockSimulator
from uav_wm_navigation.world_models import JEPAWorldModelAdapter


def small_model() -> JEPAWorldModelAdapter:
    config = {
        "latent_dim": 32,
        "jepa": {
            "model": "jepa", "state_dim": 13, "trajectory_dim": 9,
            "latent_dim": 32, "layers": 1, "heads": 4,
            "max_horizon": 32, "dropout": 0.0,
        },
        "adapter": {"hidden_dim": 48, "dynamics_layers": 1},
    }
    model, provenance = JEPAWorldModelAdapter.from_config(config)
    assert not provenance["trained"]
    return model.eval()


def test_jepa_adapter_vectorized_rollout_shapes_and_finite_values() -> None:
    model = small_model()
    observation = {
        "depth": torch.ones(1, 4, 1, 32, 48),
        "state_history": torch.zeros(1, 4, 13),
        "goal_body": torch.tensor([[5.0, 0.0, 0.0]]),
    }
    latent = model.encode(observation)
    state = torch.zeros(1, 9)
    state[:, 8] = 20.0
    actions = torch.zeros(23, 7, 4)
    actions[..., 0] = 0.25
    output = model.rollout(latent, state, actions, dt=0.1)
    assert output["latent"].shape == (23, 7, 32)
    assert output["position"].shape == (23, 7, 3)
    assert output["collision_probability"].shape == (23, 7)
    assert torch.isfinite(output["latent"]).all()
    assert torch.all(output["position"][:, -1, 0] > 0.0)


def test_mppi_is_batched_bounded_and_warm_started() -> None:
    model = small_model()
    latent = model.encode({
        "depth": torch.ones(1, 4, 1, 32, 48),
        "state_history": torch.zeros(1, 4, 13),
        "goal_body": torch.tensor([[4.0, 0.0, 0.0]]),
    })
    state = torch.zeros(1, 9)
    state[:, 8] = 20.0
    planner = MPPIPlanner(
        horizon=6, num_samples=32, num_iterations=2, dt=0.1,
        cost_weights=MPPICostWeights(collision=5.0), save_debug=True, seed=9,
    )
    first = planner.plan(model, latent, state, torch.tensor([4.0, 0.0, 0.0]))
    assert first.candidate_positions.shape == (32, 6, 3)
    assert first.candidate_action_sequences.shape == (32, 6, 4)
    assert torch.all(first.action_sequence <= 1.0) and torch.all(first.action_sequence >= -1.0)
    assert first.diagnostics["finite_candidate_fraction"] == 1.0
    previous_tail = first.action_sequence[1:].clone()
    assert torch.allclose(planner._mean[:-1], previous_tail)


def test_single_uav_receding_horizon_mock_loop_moves_toward_goal() -> None:
    simulator = MockSimulator(seed=2, scenario="OpenSpace", control_dt=0.02)
    env = UrbanFlyWorldModelEnv(
        simulator,
        config=UrbanFlyEnvConfig(
            physics_hz=50, sensor_hz=10, policy_hz=10,
            success_radius_m=0.5, success_dwell_s=0.1, max_episode_s=5.0,
        ),
        seed=2,
    )
    model = small_model()
    planner = MPPIPlanner(horizon=6, num_samples=48, num_iterations=2, dt=0.1, seed=2)
    controller = WAMMPCController(model, planner, history=4, depth_max_m=20.0, depth_shape=(96, 160))
    goal = np.asarray([4.0, 0.0, 2.0], dtype=np.float32)
    observation, info = env.reset(goal_nwu=goal, scenario="OpenSpace")
    initial_distance = float(np.linalg.norm(goal - info["state"].position))
    controller.reset()
    try:
        for _ in range(6):
            decision = controller.plan(observation, info["state"], goal)
            assert not decision.used_fallback
            observation, _, terminated, truncated, info = env.step(
                decision.action_normalized, predicted_risk=decision.predicted_risk, shield_enabled=False
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    assert float(np.linalg.norm(goal - info["state"].position)) < initial_distance
    assert controller.failure_count == 0


def test_existing_hdf5_pipeline_feeds_wam_transition_training(tmp_path) -> None:
    simulator = MockSimulator(seed=3, scenario="OpenSpace")
    candidates = MockCandidatePlanner(candidate_count=3, horizon_steps=8)
    executor = TrajectoryExecutor(simulator, SafetyFilter({"max_acceleration_mps2": 20.0}), 0.1)
    episode = collect_episode(
        simulator, candidates, executor, tmp_path, "wam_episode",
        np.asarray([4.0, 0.0, 2.0]), steps=8, future_horizon=3,
    )
    dataset = WAMMPCTransitionDataset([episode], history=2)
    sample = dataset[0]
    assert sample["depth"].shape == (2, 1, 96, 160)
    assert sample["next_depth"].shape == (2, 1, 96, 160)
    assert sample["planning_state"].shape == (9,)
    assert sample["action"].shape == (4,)
    assert torch.isfinite(sample["next_planning_state"]).all()


def test_invalid_depth_uses_hover_fallback_instead_of_crashing() -> None:
    simulator = MockSimulator(seed=4, scenario="OpenSpace")
    env = UrbanFlyWorldModelEnv(simulator)
    goal = np.asarray([4.0, 0.0, 2.0], dtype=np.float32)
    observation, info = env.reset(goal_nwu=goal, scenario="OpenSpace")
    observation.depth_m.fill(np.nan)
    observation.depth_valid_mask.fill(False)
    controller = WAMMPCController(
        small_model(), MPPIPlanner(horizon=3, num_samples=8, num_iterations=1),
        history=2, depth_shape=(96, 160),
    )
    try:
        decision = controller.plan(observation, info["state"], goal)
    finally:
        env.close()
    assert decision.used_fallback
    assert np.array_equal(decision.action_normalized, np.zeros(4, dtype=np.float32))
    assert "invalid depth" in decision.error
