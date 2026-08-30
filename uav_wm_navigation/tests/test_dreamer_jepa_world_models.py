import torch

from uav_wm_navigation.world_models import (
    ActionConditionedJEPAWorldModel,
    DreamerV3WorldModel,
    WorldModelLoss,
    build_world_model,
)


def model_inputs(batch: int = 2):
    return (
        torch.rand(batch, 4, 1, 48, 80),
        torch.rand(batch, 4, 13),
        torch.rand(batch, 3),
        torch.rand(batch, 5, 8, 9),
    )


def targets(batch: int = 2):
    return {
        "collision": torch.randint(0, 2, (batch, 5)).float(),
        "failure": torch.randint(0, 2, (batch, 5)).float(),
        "minimum_clearance": torch.rand(batch, 5) * 5,
        "goal_progress": torch.rand(batch, 5),
        "label_valid_mask": torch.ones(batch, 5),
    }


def test_dreamerv3_rssm_forward_backward() -> None:
    model = DreamerV3WorldModel(latent_dim=32, deterministic_dim=48, stochastic_groups=4, stochastic_classes=4)
    output = model(*model_inputs())
    assert output["latent_states"].shape == (2, 5, 8, 32)
    assert torch.isfinite(output["rssm_kl"])
    loss, parts = WorldModelLoss({"collision": 1, "clearance": 1, "progress": 1, "failure": 1, "dynamics": 0.1})(output, targets())
    assert torch.isfinite(loss)
    assert torch.isfinite(parts["dynamics"])
    loss.backward()
    assert model.prior[-1].weight.grad is not None


def test_jepa_future_latent_prediction_and_ema() -> None:
    model = ActionConditionedJEPAWorldModel(latent_dim=32, layers=1, heads=4, mask_ratio=0.25)
    inputs = model_inputs()
    output = model(
        *inputs,
        future_depth=torch.rand(2, 4, 1, 48, 80),
        selected_index=torch.tensor([1, 3]),
        future_valid_mask=torch.ones(2, 4),
    )
    assert output["latent_states"].shape == (2, 5, 8, 32)
    assert torch.isfinite(output["jepa_loss"])
    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    loss, _ = WorldModelLoss({"collision": 1, "clearance": 1, "progress": 1, "failure": 1, "dynamics": 0.5})(output, targets())
    loss.backward()
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    with torch.no_grad():
        next(model.context.depth.network.parameters()).add_(0.1)
    model.update_target_encoder()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.target_encoder.parameters()))


def test_factory_builds_both_research_variants() -> None:
    common = {"history": 4, "state_dim": 13, "trajectory_dim": 9, "latent_dim": 32, "dropout": 0.1}
    assert isinstance(build_world_model({**common, "model": "dreamerv3"}), DreamerV3WorldModel)
    assert isinstance(build_world_model({**common, "model": "jepa"}), ActionConditionedJEPAWorldModel)
