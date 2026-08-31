import torch

from backend.integrations.swarm_imitation import (
    SharedSwarmImitationPolicy,
    SwarmImitationConfig,
)


def _batch(batch, drones):
    depth = torch.rand(batch, drones, 64, 64, 1)
    state = torch.randn(batch, drones, 190)
    teacher = torch.empty(batch, drones, 5).uniform_(-1.0, 1.0)
    teacher[..., 3] = torch.rand(batch, drones)
    return depth, state, teacher


def test_forward_supports_two_and_eight_drones_with_action_bounds():
    model = SharedSwarmImitationPolicy()
    for drones in (2, 8):
        depth, state, _ = _batch(2, drones)
        output = model(depth, state)
        assert output["action"].shape == (2, drones, 5)
        assert output["collision_probability"].shape == (2, drones)
        assert torch.all(output["action"][..., 0:3].abs() <= 1.0)
        assert torch.all((output["action"][..., 3] >= 0.0) & (output["action"][..., 3] <= 1.0))
        assert torch.all(output["action"][..., 4].abs() <= 1.0)


def test_imitation_and_collision_losses_backpropagate():
    torch.manual_seed(7)
    model = SharedSwarmImitationPolicy(
        SwarmImitationConfig(feature_width=64, attention_heads=4, attention_layers=1)
    )
    depth, state, teacher = _batch(2, 4)
    output = model(depth, state)
    loss = model.imitation_loss(
        output,
        teacher,
        collision_label=torch.randint(0, 2, (2, 4), dtype=torch.float32),
    )
    loss["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert torch.isfinite(loss["loss"])
    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_padding_mask_excludes_missing_agents_from_loss():
    model = SharedSwarmImitationPolicy(
        SwarmImitationConfig(feature_width=64, attention_heads=4, attention_layers=1)
    )
    depth, state, teacher = _batch(1, 8)
    padding = torch.tensor([[False, False, False, True, True, True, True, True]])
    output = model(depth, state, padding_mask=padding)
    losses = model.imitation_loss(
        output,
        teacher,
        valid_mask=~padding,
    )
    assert torch.isfinite(losses["loss"])


def test_checkpoint_roundtrip(tmp_path):
    config = SwarmImitationConfig(feature_width=64, attention_heads=4, attention_layers=1)
    model = SharedSwarmImitationPolicy(config)
    path = tmp_path / "policy.pt"
    model.save_checkpoint(path, source="unit-test")
    loaded = SharedSwarmImitationPolicy.load_checkpoint(path)
    assert loaded.config == config
    assert all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), loaded.state_dict().values()))
