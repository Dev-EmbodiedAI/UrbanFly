import json

import h5py
import numpy as np
import torch

from scripts.run_helsinki_observation_policy import terminal_capture_action

from urbanfly_vln.observation_policy import (
    STATE_FEATURE_NAMES,
    HelsinkiObservationPolicy,
    ObservationPolicyConfig,
    load_observation_policy_checkpoint,
    save_observation_policy_checkpoint,
)
from urbanfly_vln.observation_policy_data import (
    HelsinkiObservationPolicyDataset,
    action_statistics,
    load_qa_episode_records,
    tail_episode_split,
)


def _episode(path, index, task, steps=4):
    episode_id = f"HelsinkiCentral1km_real_smoke_{index:03d}_{task}"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations/rgb_front", data=np.zeros((steps, 12, 20, 3), np.uint8))
        handle.create_dataset("observations/depth_front", data=np.full((steps, 12, 20), 5.0, np.float32))
        handle.create_dataset("observations/depth_valid", data=np.ones((steps, 12, 20), np.uint8))
        handle.create_dataset("goal/local_goal_body", data=np.tile([10.0, 1.0, 0.5], (steps, 1)))
        handle.create_dataset("state/linear_velocity", data=np.tile([2.0, 0.0, 0.0], (steps, 1)))
        handle.create_dataset("state/angular_velocity", data=np.zeros((steps, 3), np.float32))
        handle.create_dataset("state/orientation_xyzw", data=np.tile([0.0, 0.0, 0.0, 1.0], (steps, 1)))
        action = np.tile([3.0, 0.1, 0.2, 0.05], (steps, 1)).astype(np.float32)
        handle.create_dataset("actions/commanded_body_flu", data=action)
    return {
        "path": str(path),
        "episode_id": episode_id,
        "task_type": task,
        "steps": steps,
        "integrity_status": "PASS",
    }


def _qa(tmp_path):
    tasks = ["building_blocked", "street_canyon", "rooftop_to_ground", "ground_to_rooftop"]
    episodes = [_episode(tmp_path / f"ep_{index}.h5", index, tasks[index]) for index in range(4)]
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "gate_checks": {"stale_action_zero": True},
                "corrupted_hdf5": [],
                "partial_files": [],
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dataset_uses_only_public_policy_features_and_episode_split(tmp_path):
    records = load_qa_episode_records(_qa(tmp_path))
    split = tail_episode_split(records, validation_episodes=1)
    assert [record.episode_index for record in split.train] == [0, 1, 2]
    assert [record.episode_index for record in split.validation] == [3]
    mean, std = action_statistics(split.train)
    np.testing.assert_allclose(mean, [3.0, 0.1, 0.2, 0.05], atol=1e-6)
    assert np.all(std > 0)
    dataset = HelsinkiObservationPolicyDataset(split.train, history_frames=2)
    sample = dataset[0]
    assert sample["rgb"].shape == (2, 12, 20, 3)
    assert sample["public_state"].shape == (len(STATE_FEATURE_NAMES),)
    np.testing.assert_allclose(sample["public_state"][-4:], 0.0)
    second = dataset[1]
    np.testing.assert_allclose(second["public_state"][-4:], [0.5, 1 / 60, 1 / 15, 0.05 / np.deg2rad(60)], rtol=1e-5)


def test_policy_forward_and_checkpoint_roundtrip(tmp_path):
    config = ObservationPolicyConfig(history_frames=2, image_height=32, image_width=48)
    model = HelsinkiObservationPolicy(
        config,
        action_mean=np.asarray([3.0, 0.0, 0.0, 0.0], np.float32),
        action_std=np.asarray([0.5, 0.2, 0.4, 0.1], np.float32),
    )
    model.eval()
    rgb = torch.zeros((3, 2, 12, 20, 3), dtype=torch.uint8)
    depth = torch.full((3, 2, 12, 20), 5.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    state = torch.zeros((3, len(STATE_FEATURE_NAMES)))
    output = model(rgb, depth, valid, state)
    assert output.shape == (3, 4)
    assert torch.isfinite(output).all()
    latent = model.encode(rgb, depth, valid, state)
    assert latent.shape == (3, 192)
    assert torch.isfinite(latent).all()
    path = tmp_path / "policy.pt"
    save_observation_policy_checkpoint(path, model, metadata={"status": "TEST"})
    loaded, metadata = load_observation_policy_checkpoint(path)
    assert metadata["status"] == "TEST"
    with torch.inference_mode():
        torch.testing.assert_close(loaded(rgb, depth, valid, state), output)


def test_terminal_capture_is_general_and_bounded():
    learned = np.asarray([0.2, 0.0, -0.1, 0.0])
    far, far_blend = terminal_capture_action(learned, np.asarray([12.0, 0.0, 0.0]), 12.0)
    np.testing.assert_allclose(far, learned)
    assert far_blend == 0.0
    capture, blend = terminal_capture_action(learned, np.asarray([-4.0, 3.0, 1.0]), 2.0)
    assert blend == 1.0
    assert capture[0] < 0.0 and capture[1] > 0.0 and capture[2] > 0.0
    assert np.all(np.abs(capture) <= np.asarray([6.0, 6.0, 3.0, np.deg2rad(60.0)]))


def test_canonical_qa_and_manifest_are_training_readable(tmp_path):
    legacy = json.loads(_qa(tmp_path).read_text(encoding="utf-8"))
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "urbanfly-helsinki-canonical-dataset-v1",
                "status": "PASS",
                "episodes": [
                    {
                        **item,
                        "hdf5_readback": "PASS",
                    }
                    for item in legacy["episodes"]
                ],
            }
        ),
        encoding="utf-8",
    )
    canonical_qa = tmp_path / "dataset_qa.json"
    canonical_qa.write_text(
        json.dumps(
            {
                "schema": "urbanfly-helsinki-canonical-dataset-v1-qa",
                "status": "PASS",
                "episode_count": 100,
                "stale_action_count": 0,
                "collision_count": 0,
                "partial_count": 0,
                "all_hdf5_readback_pass": True,
                "manifest": str(manifest),
            }
        ),
        encoding="utf-8",
    )
    records = load_qa_episode_records(canonical_qa)
    assert [record.episode_index for record in records] == [0, 1, 2, 3]
