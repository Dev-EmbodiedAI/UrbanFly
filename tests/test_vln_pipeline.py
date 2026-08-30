from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_vln_world_model_demo import synthetic_episode
from urbanfly_vln.episode_builder import build_episode
from urbanfly_vln.risk_world_model import LinearRiskWorldModel
from urbanfly_vln.risk_world_model import instruction_embedding
from urbanfly_vln.schema import Episode
from urbanfly_vln.latent_world_model import DynamicsMLP, LatentWorldModelEnsemble
from urbanfly_vln.world_model_data import WorldModelSample, grouped_split
from urbanfly_vln.world_model_metrics import (
    binary_auroc,
    expected_calibration_error,
    fit_ensemble_temperature,
    json_ready,
    risk_report,
)
import numpy as np
import torch


class UrbanFlyVlnTest(unittest.TestCase):
    def test_schema_round_trip(self) -> None:
        episode = synthetic_episode(0, steps=12)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episode.json"
            episode.write_json(path)
            restored = Episode.read_json(path)
        self.assertEqual(restored.episode_id, episode.episode_id)
        self.assertEqual(len(restored.steps), 12)

    def test_world_model_contract(self) -> None:
        episodes = [synthetic_episode(index, steps=30) for index in range(4)]
        model = LinearRiskWorldModel(language_dimensions=8)
        metrics = model.fit(episodes[:3])
        validation = model.evaluate(episodes[3:])
        self.assertGreater(metrics.examples, 20)
        self.assertGreaterEqual(validation.risk_accuracy, 0.5)
        self.assertTrue(validation.transition_rmse >= 0.0)

    def test_urbanfly_run_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            summary = {"success": True, "collision_steps": 0, "carla_map": "Town10HD"}
            (run_dir / "long_range_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with (run_dir / "global_route.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["waypoint_idx", "x", "y", "z"])
                writer.writerows([[0, 0, 0, -15], [1, 10, 0, -15], [2, 10, 10, -15]])
            fields = [
                "time_s", "replan_step", "x", "y", "z", "vx", "vy", "vz",
                "target_waypoint_idx", "final_goal_distance_m", "min_depth_m",
                "p05_depth_m", "collision_events",
            ]
            with (run_dir / "long_range_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, [0, 0, 0, 0, -15, 1, 0, 0, 1, 14, 2, 3, 0])))
                writer.writerow(dict(zip(fields, [1, 1, 10, 0, -15, 0, 1, 0, 2, 10, 2, 3, 0])))
                writer.writerow(dict(zip(fields, [2, 2, 10, 10, -15, 0, 0, 0, 2, 0, 2, 3, 0])))
            episode = build_episode(run_dir)
        self.assertEqual(episode.scene_id, "Town10HD")
        self.assertIn("turn left", episode.instruction)
        self.assertEqual(len(episode.steps), 3)

    def test_latent_world_model_candidate_ranking(self) -> None:
        models = [DynamicsMLP(hidden_dim=16) for _ in range(2)]
        ensemble = LatentWorldModelEnsemble(
            models=models,
            x_mean=np.zeros(10, dtype=np.float32),
            x_scale=np.ones(10, dtype=np.float32),
            y_mean=np.zeros(6, dtype=np.float32),
            y_scale=np.ones(6, dtype=np.float32),
            device=torch.device("cpu"),
        )
        predictions = ensemble.rank_candidates(
            state=np.array([3.0, 0.0, 10.0, 12.0, 100.0, 0.2], dtype=np.float32),
            action_vectors=[np.array([10.0, 0.0, 0.0]), np.array([15.0, 3.0, 0.0])],
            horizon=2,
        )
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(np.isfinite(item.score) for item in predictions))
        self.assertTrue(all(0.0 <= item.risk_probability <= 1.0 for item in predictions))

    def test_grouped_split_keeps_runs_disjoint(self) -> None:
        samples = [
            WorldModelSample(
                features=np.zeros(12, dtype=np.float32),
                target=np.zeros(6, dtype=np.float32),
                risk=float(index % 2),
                source=source,
                scene_id="Town03",
                instruction="fly east",
                replan_step=index,
            )
            for source in ("run-a", "run-b", "run-c")
            for index in range(4)
        ]
        split = grouped_split(samples, validation_sources={"run-c"})
        train_sources = {samples[index].source for index in split.train_indices}
        validation_sources = {samples[index].source for index in split.validation_indices}
        self.assertEqual(validation_sources, {"run-c"})
        self.assertFalse(train_sources & validation_sources)

    def test_risk_metrics_include_calibration(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        scores = np.array([0.05, 0.2, 0.7, 0.95], dtype=np.float32)
        self.assertEqual(binary_auroc(labels, scores), 1.0)
        self.assertLess(expected_calibration_error(labels, scores), 0.2)
        report = risk_report(labels, scores)
        self.assertEqual(report["recall"], 1.0)
        self.assertIn("ece", report)
        self.assertIsNone(json_ready({"auc": float("nan")})["auc"])
        member_logits = np.array([[-3.0, -1.0, 1.0, 3.0], [-2.5, -0.5, 0.5, 2.5]])
        self.assertGreater(fit_ensemble_temperature(labels, member_logits), 0.0)

    def test_instruction_embedding_supports_chinese_and_zero_dimensions(self) -> None:
        self.assertEqual(instruction_embedding("向东飞行", 0).shape, (0,))
        east = instruction_embedding("向东飞行，避开建筑", 16)
        west = instruction_embedding("向西飞行，避开树木", 16)
        self.assertAlmostEqual(float(np.linalg.norm(east)), 1.0)
        self.assertFalse(np.allclose(east, west))

    def test_v1_checkpoint_compatibility_and_safety_fallback(self) -> None:
        model = DynamicsMLP(hidden_dim=16)
        payload = {
            "format": "urbanfly-latent-world-model-v1",
            "hidden_dim": 16,
            "model_state_dicts": [model.state_dict()],
            "x_mean": np.zeros(10, dtype=np.float32),
            "x_scale": np.ones(10, dtype=np.float32),
            "y_mean": np.zeros(6, dtype=np.float32),
            "y_scale": np.ones(6, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "v1.pt"
            torch.save(payload, checkpoint)
            ensemble = LatentWorldModelEnsemble.load(checkpoint, device=torch.device("cpu"))
        self.assertEqual(ensemble.language_dimensions, 0)
        selection = ensemble.select_candidate(
            state=np.zeros(6, dtype=np.float32),
            action_vectors=[np.array([2.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])],
            max_risk_probability=0.0,
        )
        self.assertTrue(selection.used_fallback)
        self.assertEqual(selection.reason, "no_candidate_passed_safety_gate")

    def test_v2_language_condition_and_candidate_specific_states(self) -> None:
        language_dim = 8
        models = [DynamicsMLP(hidden_dim=16, input_dim=10 + language_dim) for _ in range(2)]
        ensemble = LatentWorldModelEnsemble(
            models=models,
            x_mean=np.zeros(10 + language_dim, dtype=np.float32),
            x_scale=np.ones(10 + language_dim, dtype=np.float32),
            y_mean=np.zeros(6, dtype=np.float32),
            y_scale=np.ones(6, dtype=np.float32),
            device=torch.device("cpu"),
            language_dimensions=language_dim,
        )
        states = np.stack([np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)])
        predictions = ensemble.rank_candidates(
            state=states,
            action_vectors=[np.array([2.0, 0.0, 0.0]), np.array([4.0, 1.0, 0.0])],
            instruction="turn left after the tower",
        )
        self.assertEqual(len(predictions), 2)
        with self.assertRaises(ValueError):
            ensemble.rank_candidates(states[:1], [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])])


if __name__ == "__main__":
    unittest.main()
