import numpy as np
import torch

from urbanfly_vln.navigation_world_model import (
    HelsinkiLatentWorldModel,
    NavigationWorldModelConfig,
    load_navigation_world_model_checkpoint,
    save_navigation_world_model_checkpoint,
)


def test_latent_world_model_prediction_and_checkpoint(tmp_path):
    config = NavigationWorldModelConfig(latent_dim=12, hidden_dim=24, ensemble_size=3)
    mean = np.linspace(-0.2, 0.2, 17, dtype=np.float32)
    std = np.linspace(0.5, 1.5, 17, dtype=np.float32)
    model = HelsinkiLatentWorldModel(config, target_mean=mean, target_std=std).eval()
    latent = torch.zeros((2, 12))
    actions = torch.zeros((2, 15, 4))
    result = model.predict(latent, actions)
    assert result["physical_mean"].shape == (2, 15, 5)
    assert result["next_latent_mean"].shape == (2, 15, 12)
    assert torch.isfinite(result["physical_std"]).all()
    checkpoint = tmp_path / "world_model.pt"
    save_navigation_world_model_checkpoint(checkpoint, model, metadata={"status": "TEST"})
    loaded, metadata = load_navigation_world_model_checkpoint(checkpoint)
    assert metadata["status"] == "TEST"
    with torch.inference_mode():
        loaded_result = loaded.predict(latent, actions)
    torch.testing.assert_close(loaded_result["physical_mean"], result["physical_mean"])


def test_latent_world_model_rejects_bad_shapes():
    model = HelsinkiLatentWorldModel(
        NavigationWorldModelConfig(latent_dim=8, hidden_dim=16, ensemble_size=2)
    )
    with np.testing.assert_raises(ValueError):
        model.predict(torch.zeros((1, 7)), torch.zeros((1, 15, 4)))
    with np.testing.assert_raises(ValueError):
        model.predict(torch.zeros((1, 8)), torch.zeros((15, 4)))
