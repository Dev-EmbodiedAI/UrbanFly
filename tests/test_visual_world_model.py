from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from urbanfly_vln.direct_visual_world_model import (
    DirectVisualWorldModel,
    DirectWorldModelConfig,
)
from urbanfly_vln.visual_world_model import (
    RSSMState,
    VisualRSSM,
    VisualWorldModelConfig,
    VisualWorldModelPlanner,
    save_visual_checkpoint,
)


class VisualWorldModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VisualWorldModelConfig(
            base_channels=8,
            embed_dim=32,
            deter_dim=32,
            stoch_dim=8,
            hidden_dim=32,
        )
        self.model = VisualRSSM(self.config)

    def batch(self) -> dict[str, torch.Tensor]:
        return {
            "observations": torch.rand(2, 3, 4, 64, 64),
            "actions": torch.rand(2, 3, 4),
            "states": torch.rand(2, 3, 12),
            "rewards": torch.rand(2, 3),
            "risks": torch.zeros(2, 3),
            "continues": torch.ones(2, 3),
        }

    def test_sequence_loss_and_backward(self) -> None:
        loss, metrics = self.model.loss(self.batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("reconstruction", metrics)
        loss.backward()
        self.assertIsNotNone(self.model.encoder.convs[0].weight.grad)

    def test_imagination_shapes(self) -> None:
        initial = self.model.initial(5)
        result = self.model.imagine(initial, torch.rand(5, 7, 4))
        self.assertEqual(result["reward"].shape, (5, 7))
        self.assertEqual(result["state"].shape, (5, 7, 12))

    def test_planner_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "visual.pt"
            save_visual_checkpoint(checkpoint, self.model)
            planner = VisualWorldModelPlanner.load(checkpoint, device=torch.device("cpu"), horizon=2)
            planner.belief = RSSMState(torch.zeros(1, 32), torch.zeros(1, 8))
            selected, records = planner.select_delta(
                [np.asarray([5.0, 0.0, 0.0]), np.asarray([4.0, 2.0, 0.0])]
            )
            self.assertIn(selected, (0, 1))
            self.assertEqual(len(records), 2)
            self.assertIn("risk_probability", records[0])

    def test_direct_world_model_loss(self) -> None:
        config = DirectWorldModelConfig(base_channels=8, embed_dim=32, hidden_dim=32)
        model = DirectVisualWorldModel(config)
        batch = {
            "observations": torch.rand(2, 4, 64, 64),
            "states": torch.rand(2, 12),
            "actions": torch.rand(2, 4),
            "next_states": torch.rand(2, 12),
            "rewards": torch.rand(2),
        }
        loss, metrics = model.loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("state", metrics)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
