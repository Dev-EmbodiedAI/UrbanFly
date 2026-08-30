from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from uav_wm_navigation.control.transparent_safety import TransparentSafetyLayer
from uav_wm_navigation.data.splits import load_split_manifest, validate_route_split
from uav_wm_navigation.data.world_model_dataset_v3 import UrbanFlyRGBDSequenceDataset
from uav_wm_navigation.data.world_model_v3 import (
    PrivilegedStepLabels,
    StreamingWebDatasetWriter,
    WorldModelV3StepRecord,
)
from uav_wm_navigation.envs.urbanfly_world_model_env import UrbanFlyWorldModelEnv
from uav_wm_navigation.evaluation.paired_benchmark import expected_jobs, load_evaluation_manifest, summarize_results, validate_results
from uav_wm_navigation.simulators.mock_simulator import MockSimulator
from uav_wm_navigation.types import EpisodeSpec, SafetyAudit
from uav_wm_navigation.world_models.tdmpc2_visual import TDMPC2VisualNetwork, TDMPC2VisualPolicy
from uav_wm_navigation.world_models.tdmpc2_visual_training import VisualTDMPC2Trainer


def _episode() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="v3-test", route_id="route-001", split="train",
        tile_ids=("673498a1",), scenario="OpenSpace", seed=101,
        start_nwu_m=(-400.0, -400.0, 2.0), goal_nwu_m=(-380.0, -400.0, 2.0),
        actor_script_id="actors-101",
    )


def _audit(observation, step: int) -> SafetyAudit:
    zero = np.zeros(4, dtype=np.float32)
    return SafetyAudit(
        episode_id=observation.episode_id, step_id=step, sim_time=observation.sim_time,
        raw_action_normalized=zero, raw_action_physical=zero,
        executed_action_physical=zero, intervened=False, reasons=(),
        action_delta_l2=0.0, minimum_depth_m=20.0, predicted_risk=0.0,
    )


def test_split_manifest_is_hashed_and_enforces_buffer() -> None:
    manifest = load_split_manifest(Path(__file__).resolve().parents[1] / "configs/urbanfly_spatial_split_v1.json")
    touched = validate_route_split(
        np.asarray([[-450.0, -450.0, 10.0], [-300.0, -300.0, 10.0]]),
        "train", manifest,
    )
    assert touched == ("673498a1",)


def test_v3_writer_streams_rgbd_and_separates_privileged(tmp_path) -> None:
    episode = _episode()
    simulator = MockSimulator(scenario="OpenSpace", depth_shape=(24, 40), control_dt=0.02)
    env = UrbanFlyWorldModelEnv(
        simulator,
        safety_layer=TransparentSafetyLayer(enabled=False, filter_config={"max_acceleration_mps2": 100.0}),
    )
    observation, info = env.reset(goal_nwu=np.asarray(episode.goal_nwu_m), episode_spec=episode)
    assert info["schema"] == "urbanfly-world-model-v3"
    writer = StreamingWebDatasetWriter(tmp_path, shard_prefix="v3", max_samples_per_shard=2, state_row_group_size=1)
    for step in range(4):
        writer.append(WorldModelV3StepRecord(
            observation=observation, episode_spec=episode, action_time=observation.sim_time,
            raw_action_normalized=np.zeros(4), executed_action_physical=np.zeros(4),
            reward=0.1, collision=False, success=False, done=step == 3,
            minimum_clearance_m=20.0, safety_audit=_audit(observation, step),
            privileged=PrivilegedStepLabels(
                tile_id="673498a1", zone_type="transport",
                dynamic_actor_states=[{"position": [5.0, 0.0, 0.0]}],
                cpa_risk_map=np.linspace(0.0, 1.0, 34, dtype=np.float32),
            ),
        ))
        if step < 3:
            observation, *_ = env.step(np.zeros(4), shield_enabled=False)
    manifest_path = writer.close()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "urbanfly-world-model-v3"
    assert len(manifest["shards"]) == 2
    for shard in manifest["shards"]:
        path = tmp_path / shard["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"]
        assert (tmp_path / shard["state_table"]).exists()
    policy = next(iter(UrbanFlyRGBDSequenceDataset([manifest_path], sequence_length=2, image_size=(32, 48), view="policy")))
    assert policy["rgb"].shape == (2, 3, 32, 48)
    assert "cpa_risk" not in policy
    supervised = next(iter(UrbanFlyRGBDSequenceDataset([manifest_path], sequence_length=2, image_size=(32, 48), view="world_model_supervision")))
    assert supervised["cpa_risk"].shape == (2,)
    assert supervised["cpa_risk"].max() == 1.0


def test_visual_tdmpc2_trains_and_predicts_real_rgbd(tmp_path) -> None:
    model = TDMPC2VisualNetwork()
    trainer = VisualTDMPC2Trainer(model)
    batch = {
        "rgb": torch.rand(2, 2, 3, 64, 96), "depth": torch.rand(2, 2, 1, 64, 96),
        "depth_valid": torch.ones(2, 2, 1, 64, 96), "proprio": torch.rand(2, 2, 16),
        "action": torch.rand(2, 2, 4) * 2 - 1, "reward": torch.rand(2, 2),
        "continuation": torch.ones(2, 2), "collision": torch.zeros(2, 2),
        "cpa_risk": torch.rand(2, 2), "minimum_clearance": torch.rand(2, 2) * 20,
    }
    loss = trainer.train_step(batch)
    assert np.isfinite(loss.total)
    checkpoint = tmp_path / "visual.pt"
    torch.save({
        "schema": "urbanfly-world-model-v3", "family": "tdmpc2_visual",
        "training_steps": trainer.steps, "model": model.state_dict(),
    }, checkpoint)
    simulator = MockSimulator(scenario="OpenSpace", depth_shape=(36, 64))
    env = UrbanFlyWorldModelEnv(simulator)
    observation, _ = env.reset(goal_nwu=np.asarray([20.0, 0.0, 2.0]))
    policy = TDMPC2VisualPolicy(checkpoint=checkpoint, device="cpu", horizon=3, candidates=4, elites=2, iterations=1, image_size=(64, 96))
    policy.reset(observation.episode_id); policy.observe(observation)
    prediction = policy.predict(np.zeros((5, 3, 4), dtype=np.float32))
    assert prediction["risk"].shape == (5, 3)
    assert prediction["predicted_state_1s_2s_3s"].shape == (5, 3, 3)


def test_quick_eval_manifest_is_hashed_complete_and_rejects_missing_formal_results() -> None:
    manifest = load_evaluation_manifest(
        Path(__file__).resolve().parents[1] / "outputs/manifests/urbanfly_v3_quick_eval_120.json"
    )
    jobs = expected_jobs(manifest)
    assert len(jobs) == 3360
    assert len({item["route_id"] for item in jobs}) == 120
    with np.testing.assert_raises_regex(RuntimeError, "3360 of 3360"):
        validate_results([], manifest)
    first = jobs[0]
    record = {
        **first, "success": False, "collision": False,
        "navigation_error_m": 12.0, "spl": 0.0, "path_length_m": 5.0,
        "decision_steps": 10, "intervention_steps": 0, "latency_ms": 4.0,
    }
    normalized, missing = validate_results([record], manifest, allow_incomplete=True)
    assert len(missing) == 3359
    summary = summarize_results(normalized)
    assert summary[0]["sr"] == 0.0
    assert summary[0]["ne_m"] == 12.0
