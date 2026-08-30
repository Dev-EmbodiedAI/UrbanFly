import torch

from uav_wm_navigation.world_models import GRUWorldModel, TransformerWorldModel, WorldModelLoss, mc_dropout_predict


def inputs():
    return (
        torch.rand(2, 4, 1, 48, 80),
        torch.rand(2, 4, 13),
        torch.rand(2, 3),
        torch.rand(2, 5, 8, 9),
    )


def targets():
    return {
        "collision": torch.randint(0, 2, (2, 5)).float(),
        "failure": torch.randint(0, 2, (2, 5)).float(),
        "minimum_clearance": torch.rand(2, 5) * 5,
        "goal_progress": torch.rand(2, 5),
    }


def assert_forward_backward(model) -> None:
    output = model(*inputs())
    assert output["collision_logits"].shape == (2, 5)
    assert output["latent_states"].shape[:3] == (2, 5, 8)
    loss, parts = WorldModelLoss()(output, targets())
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_gru_forward_backward_and_mc_dropout() -> None:
    model = GRUWorldModel()
    assert_forward_backward(model)
    prediction = mc_dropout_predict(model, inputs(), samples=3)
    assert prediction["uncertainty"].shape == (2, 5)
    assert torch.all(prediction["uncertainty"] >= 0)


def test_transformer_forward_backward() -> None:
    assert_forward_backward(TransformerWorldModel())

