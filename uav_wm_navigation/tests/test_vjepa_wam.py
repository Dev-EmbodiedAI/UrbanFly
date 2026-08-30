from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from uav_wm_navigation.control import SafetyFilter, TrajectoryExecutor
from uav_wm_navigation.data import (
    VJEPAHDF5SequenceDataset,
    audit_hdf5_action_semantics,
    collect_episode,
)
from uav_wm_navigation.planners import MockCandidatePlanner
from uav_wm_navigation.simulators import MockSimulator
from uav_wm_navigation.world_models import (
    VJEPAWorldModelAdapter,
    load_vjepa_wam_checkpoint,
    save_vjepa_wam_checkpoint,
    vjepa_wam_multistep_loss,
)


class TestOnlyTemporalEncoder(nn.Module):
    """Small injected encoder for interface tests; never a production fallback."""

    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.output = nn.Linear(8, output_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.output(self.network(video.permute(0, 2, 1, 3, 4)).flatten(1))


def _model(safety_mode: str = "combined") -> VJEPAWorldModelAdapter:
    return VJEPAWorldModelAdapter(
        TestOnlyTemporalEncoder(), encoder_dim=32, latent_dim=32,
        proprio_hidden_dim=24, depth_dim=16, predictor_hidden_dim=48,
        safety_mode=safety_mode,
    )


def test_vjepa_recursive_rollout_loss_and_checkpoint_provenance(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = _model()
    batch_size, history, rollout = 2, 3, 2
    sequence = history + rollout
    batch = {
        "rgb": torch.rand(batch_size, sequence, 3, 24, 32),
        "depth": torch.rand(batch_size, sequence, 1, 24, 32),
        "proprio": torch.rand(batch_size, sequence, 16),
        "goal_body": torch.rand(batch_size, sequence, 3),
        "planning_state": torch.zeros(batch_size, rollout + 1, 9),
        "action_physical": torch.rand(batch_size, rollout, 4),
        "dt": torch.full((batch_size, rollout), 0.2),
        "target_position": torch.rand(batch_size, rollout, 3),
        "target_velocity": torch.rand(batch_size, rollout, 3),
        "collision": torch.zeros(batch_size, rollout),
    }
    batch["planning_state"][..., 8] = 5.0
    loss, components = vjepa_wam_multistep_loss(
        model, batch, history_frames=history, rollout_steps=rollout
    )
    assert torch.isfinite(loss)
    assert set(components) == {"latent", "position", "velocity", "collision"}
    loss.backward()
    assert model.physics_probe[-1].weight.grad is not None

    official_fixture = tmp_path / "official-test-fixture.pt"
    torch.save({"fixture": True}, official_fixture)
    with pytest.raises(ValueError, match="untrained"):
        save_vjepa_wam_checkpoint(
            tmp_path / "invalid.pt", model, config={}, official_checkpoint=official_fixture,
            training_steps=0, encoder_model_name="test-only-injected-encoder",
        )
    checkpoint = save_vjepa_wam_checkpoint(
        tmp_path / "wam.pt", model, config={"history_frames": history},
        official_checkpoint=official_fixture, training_steps=1,
        encoder_model_name="test-only-injected-encoder",
    )
    payload = load_vjepa_wam_checkpoint(_model(), checkpoint, official_checkpoint=official_fixture)
    assert payload["training_steps"] == 1
    assert payload["pixel_reconstruction"] is False


def test_vjepa_hdf5_has_explicit_aligned_actions(tmp_path: Path) -> None:
    simulator = MockSimulator(seed=11)
    planner = MockCandidatePlanner(candidate_count=3, horizon_steps=8)
    executor = TrajectoryExecutor(
        simulator, SafetyFilter({"max_acceleration_mps2": 20}), 0.1,
        action_noise_std=(0.1, 0.1, 0.05, 0.02),
        action_noise_bound=(0.2, 0.2, 0.1, 0.04), seed=11,
    )
    path = collect_episode(
        simulator, planner, executor, tmp_path, "episode", np.array([8.0, 0.0, 2.0]),
        steps=8, future_horizon=3, collection_mode="perturbed_expert",
    )
    dataset = VJEPAHDF5SequenceDataset(
        [path], history_frames=2, rollout_steps=2, image_size=(24, 32)
    )
    sample = dataset[0]
    assert sample["rgb"].shape == (4, 3, 24, 32)
    assert sample["action_physical"].shape == (2, 4)
    assert torch.all(sample["dt"] > 0)
    report = audit_hdf5_action_semantics([path])
    assert report["all_explicit"]
    assert report["episodes"][0]["maximum_denormalization_error"] < 1e-5


@pytest.mark.parametrize("mode", ["geometry_only", "learned_only", "combined"])
def test_vjepa_safety_modes(mode: str) -> None:
    model = _model(mode)
    state = torch.zeros(2, 9)
    state[:, 8] = torch.tensor([0.2, 5.0])
    output = model.predict_step(torch.zeros(2, 32), state, torch.zeros(2, 4), dt=0.2)
    assert output["collision_probability"].shape == (2,)
    assert torch.all((output["collision_probability"] >= 0) & (output["collision_probability"] <= 1))
