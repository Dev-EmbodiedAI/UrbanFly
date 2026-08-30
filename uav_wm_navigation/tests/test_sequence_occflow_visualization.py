from pathlib import Path

import h5py
import numpy as np
import torch

from uav_wm_navigation.control import SafetyFilter, TrajectoryExecutor
from uav_wm_navigation.data import DreamerSequenceDataset, collect_episode, validate_episode
from uav_wm_navigation.evaluation import depth_to_flu_points, render_yopo_frame
from uav_wm_navigation.planners import MockCandidatePlanner
from uav_wm_navigation.simulators import MockSimulator
from uav_wm_navigation.world_models import DreamerV3WorldModel, OccFlowWorldModel, WorldModelLoss


def make_episode(tmp_path: Path, steps: int = 10) -> Path:
    simulator = MockSimulator(seed=7)
    planner = MockCandidatePlanner(candidate_count=15, horizon_steps=16)
    executor = TrajectoryExecutor(simulator, SafetyFilter({"max_acceleration_mps2": 20}), 0.1)
    return collect_episode(
        simulator, planner, executor, tmp_path, "sequence_episode", np.array([8.0, 0.0, 2.0]),
        steps=steps, future_horizon=3,
    )


def test_hdf5_v2_dreamer_sequence_is_bounded_and_aligned(tmp_path: Path) -> None:
    path = make_episode(tmp_path)
    assert validate_episode(path)["schema_version"] == 2
    with h5py.File(path, "r") as handle:
        assert handle["sequence/action"].shape == (10, 9)
        assert handle["sequence/is_first"][:].tolist() == [1] + [0] * 9
        assert handle["sequence/is_terminal"][-1] == 1
        assert np.allclose(handle["sequence/continuation"][-1], 0)
    dataset = DreamerSequenceDataset([path], sequence_length=6, stride=2)
    assert len(dataset) == 3
    sample = dataset[1]
    assert sample["depth"].shape == (6, 1, 96, 160)
    assert sample["action"].shape == (6, 9)
    assert sample["is_first"][0] == 1


def test_dreamer_sequence_objective_is_finite() -> None:
    model = DreamerV3WorldModel(latent_dim=24, deterministic_dim=32, stochastic_groups=4, stochastic_classes=4)
    total, parts = model.sequence_training_loss(
        torch.rand(2, 7, 1, 48, 80), torch.rand(2, 7, 13), torch.rand(2, 7, 3),
        torch.rand(2, 7, 9), torch.randn(2, 7), torch.ones(2, 7),
        torch.tensor([[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
        burn_in=2,
    )
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in parts.values())
    total.backward()
    assert model.reward_head.weight.grad is not None


def test_occflow_protocol_forward_backward() -> None:
    model = OccFlowWorldModel(latent_dim=16, history=4, future_steps=2)
    depth = torch.rand(1, 4, 1, 48, 80)
    state = torch.rand(1, 4, 13)
    goal = torch.rand(1, 3)
    trajectories = torch.rand(1, 3, 4, 9)
    output = model(depth, state, goal, trajectories)
    assert output["occupancy_logits"].shape == (1, 2, 3, 16, 32, 48)
    target = {
        "collision": torch.zeros(1, 3), "failure": torch.zeros(1, 3),
        "minimum_clearance": torch.ones(1, 3), "goal_progress": torch.ones(1, 3),
        "label_valid_mask": torch.ones(1, 3),
        "occupancy": torch.zeros_like(output["occupancy_logits"]),
        "flow": torch.zeros_like(output["flow"]),
    }
    loss, _ = WorldModelLoss()(output, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_yopo_style_visualization_renders(tmp_path: Path) -> None:
    depth = np.full((2, 96, 160), 8.0, dtype=np.float32)
    t = np.linspace(0, 1, 16, dtype=np.float32)
    candidates = np.stack([
        np.column_stack([8 * t, (index - 7) * 0.2 * t, 0.3 * np.sin(np.pi * t)])
        for index in range(15)
    ])
    archive = tmp_path / "visual.npz"
    np.savez_compressed(
        archive, depth=depth, candidates=np.stack([candidates, candidates]), selected=np.array([7, 8]),
        yopo_cost=np.ones((2, 15)), total_score=np.tile(np.linspace(1, 0, 15), (2, 1)),
        collision_probability=np.tile(np.linspace(0, 1, 15), (2, 1)),
        position_nwu=np.array([[0, 0, 2], [0.2, 0, 2]], dtype=np.float32),
    )
    assert depth_to_flu_points(depth[0]).shape[1] == 3
    output = render_yopo_frame(archive, tmp_path / "yopo.png")
    assert output.exists() and output.stat().st_size > 10_000
