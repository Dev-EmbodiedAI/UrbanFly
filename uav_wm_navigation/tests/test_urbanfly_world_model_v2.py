from __future__ import annotations

import json
import tarfile
import zlib

import cv2
import numpy as np
import pytest
import torch

from uav_wm_navigation.control.transparent_safety import TransparentSafetyLayer
from uav_wm_navigation.data.world_model_v2 import (
    SCHEMA_NAME,
    WebDatasetShardWriter,
    WorldModelStepRecord,
)
from uav_wm_navigation.data.world_model_dataset_v2 import (
    UrbanFlyTransitionDataset,
)
from uav_wm_navigation.envs.urbanfly_world_model_env import (
    UrbanFlyEnvConfig,
    UrbanFlyWorldModelEnv,
)
from uav_wm_navigation.evaluation.metrics import ndcg, wilson_interval
from uav_wm_navigation.evaluation.world_model_report import (
    write_world_model_report,
)
from uav_wm_navigation.risk.cpa import (
    DepthHistory,
    cpa_risk,
    cpa_risk_map,
)
from uav_wm_navigation.simulators.mock_simulator import MockSimulator
from uav_wm_navigation.simulators.urbanfly_sensor_packet import (
    decode_urbanfly_sensor_packet,
)
from uav_wm_navigation.types import (
    ActionLimits,
    BodyVelocityAction,
    SafetyAudit,
)
from uav_wm_navigation.world_models.tdmpc2_continuous import (
    TDMPC2ContinuousPolicy,
    TDMPC2Network,
)
from uav_wm_navigation.world_models.tdmpc2_training import TDMPC2Trainer


def test_action_contract_and_cpa_are_bounded() -> None:
    action = BodyVelocityAction(np.asarray([2.0, -2.0, 0.5, 1.0]))
    assert np.allclose(action.normalized, [1.0, -1.0, 0.5, 1.0])
    assert np.allclose(
        action.physical,
        [6.0, -6.0, 1.5, np.deg2rad(60.0)],
    )
    approaching = cpa_risk([6.0, 0.0, 0.0], [-2.0, 0.0, 0.0])
    departing = cpa_risk([6.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    assert approaching.risk > departing.risk
    assert approaching.time_to_cpa_s == pytest.approx(3.0)
    risk_map = cpa_risk_map(
        np.asarray([[6.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.asarray([[-2.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    )
    assert risk_map.shape == (34,)
    assert np.all((0.0 <= risk_map) & (risk_map <= 1.0))


def test_arr_fly_depth_history_has_15_frames() -> None:
    history = DepthHistory()
    history.append(np.full((60, 340), 12.0, dtype=np.float32))
    value = history.array()
    assert value.shape == (15, 6, 34)
    assert np.allclose(value, 12.0)


def test_env_runs_50_10_5_hz_and_audits_raw_vs_executed() -> None:
    simulator = MockSimulator(
        scenario="OpenSpace", depth_shape=(36, 64), control_dt=0.02
    )
    shield = TransparentSafetyLayer(
        enabled=True,
        filter_config={"max_acceleration_mps2": 100.0},
    )
    env = UrbanFlyWorldModelEnv(
        simulator,
        config=UrbanFlyEnvConfig(max_episode_s=5.0),
        safety_layer=shield,
    )
    observation, reset_info = env.reset(
        goal_nwu=np.asarray([20.0, 0.0, 2.0]), scenario="OpenSpace"
    )
    assert observation.step_id == 0
    next_observation, _, terminated, truncated, info = env.step(
        np.asarray([0.5, 0.0, 0.0, 0.0])
    )
    assert next_observation.step_id == 1
    assert simulator.sim_time == pytest.approx(0.2)
    assert len(env.sensor_frames) == 3  # reset plus two synchronized 10 Hz frames
    assert not terminated and not truncated
    assert info["safety_audit"].raw_action_physical.shape == (4,)
    assert info["executed_action"].shape == (4,)
    assert reset_info["schema"] == SCHEMA_NAME


def test_webdataset_writer_separates_privileged_labels(tmp_path) -> None:
    simulator = MockSimulator(
        scenario="OpenSpace", depth_shape=(18, 32), control_dt=0.02
    )
    env = UrbanFlyWorldModelEnv(
        simulator,
        safety_layer=TransparentSafetyLayer(
            enabled=False, filter_config={"max_acceleration_mps2": 100.0}
        ),
    )
    observation, _ = env.reset(
        goal_nwu=np.asarray([20.0, 0.0, 2.0]), scenario="OpenSpace"
    )
    zero = np.zeros(4, dtype=np.float32)
    audit = SafetyAudit(
        episode_id=observation.episode_id,
        step_id=0,
        sim_time=observation.sim_time,
        raw_action_normalized=zero,
        raw_action_physical=zero,
        executed_action_physical=zero,
        intervened=False,
        reasons=(),
        action_delta_l2=0.0,
        minimum_depth_m=20.0,
        predicted_risk=0.0,
    )
    writer = WebDatasetShardWriter(tmp_path, max_samples_per_shard=1)
    writer.append(
        WorldModelStepRecord(
            observation=observation,
            raw_action=zero,
            executed_action=zero,
            reward=0.0,
            collision=False,
            minimum_clearance_m=20.0,
            zone_type="residential",
            dynamics_parameters={"mass_kg": 1.5},
            safety_audit=audit,
            privileged_labels={"actor_states": [{"id": 7}]},
        )
    )
    manifest_path = writer.close()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == SCHEMA_NAME
    assert manifest["policy_inputs_exclude_privileged"] is True
    shard = tmp_path / manifest["shards"][0]
    with tarfile.open(shard) as handle:
        names = handle.getnames()
        public_name = next(name for name in names if name.endswith(".json") and ".privileged." not in name)
        privileged_name = next(name for name in names if name.endswith(".privileged.json"))
        public = json.load(handle.extractfile(public_name))
        privileged = json.load(handle.extractfile(privileged_name))
    assert "actor_states" not in public
    assert privileged["actor_states"][0]["id"] == 7


def test_tdmpc2_refuses_random_policy_and_plans_in_explicit_test_mode() -> None:
    simulator = MockSimulator(scenario="OpenSpace", depth_shape=(18, 32))
    env = UrbanFlyWorldModelEnv(simulator)
    observation, _ = env.reset(
        goal_nwu=np.asarray([20.0, 0.0, 2.0]), scenario="OpenSpace"
    )
    guarded = TDMPC2ContinuousPolicy(
        horizon=2, candidates=4, elites=2, iterations=1
    )
    guarded.reset(observation.episode_id)
    guarded.observe(observation)
    with pytest.raises(RuntimeError, match="checkpoint required"):
        guarded.act()
    policy = TDMPC2ContinuousPolicy(
        horizon=2,
        candidates=8,
        elites=2,
        iterations=1,
        allow_untrained_for_tests=True,
        device="cpu",
    )
    policy.reset(observation.episode_id)
    policy.observe(observation)
    action = policy.act()
    prediction = policy.predict(np.zeros((3, 2, 4), dtype=np.float32))
    assert action.normalized.shape == (4,)
    assert np.all(np.abs(action.normalized) <= 1.0)
    assert prediction["risk"].shape == (3, 2)
    assert policy.diagnostics()["status"] == "untrained_test_mode"


def test_registered_statistics() -> None:
    proportion, lower, upper = wilson_interval(80, 100)
    assert proportion == 0.8
    assert lower < proportion < upper
    assert ndcg([3, 2, 0], [0.9, 0.8, 0.1]) == pytest.approx(1.0)


def test_binary_sensor_bridge_round_trip() -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    depth_u16 = np.arange(24, dtype="<u2").reshape(4, 6) * 100
    depth_payload = zlib.compress(depth_u16.tobytes())
    header = {
        "schema": "urbanfly-sensor-packet-v2",
        "sequence": 12,
        "timestamp_ns": 2_000_000_000,
        "sim_time": 2.0,
        "vehicle_name": "WM-UAV-01",
        "width": 6,
        "height": 4,
        "rgb_codec": "jpeg_q95",
        "rgb_length": len(jpeg),
        "depth_codec": "u16",
        "depth_compression": "deflate",
        "depth_length": len(depth_payload),
        "depth_scale_m": 120.0 / 65535.0,
        "intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 2.5, "cy": 1.5},
        "goal_body_flu_m": [12.0, 1.0, 2.0],
        "linear_velocity_body_flu_mps": [1.0, 0.0, 0.0],
        "angular_velocity_body_flu_rps": [0.0, 0.0, 0.1],
    }
    header_bytes = json.dumps(header).encode()
    packet = (
        b"UFWM"
        + len(header_bytes).to_bytes(4, "little")
        + header_bytes
        + jpeg.tobytes()
        + depth_payload
    )
    decoded = decode_urbanfly_sensor_packet(
        packet,
        episode_id="bridge-test",
    )
    assert decoded.observation.rgb.shape == (4, 6, 3)
    assert decoded.observation.depth_m.shape == (4, 6)
    assert decoded.observation.step_id == 12
    assert decoded.observation.goal_body_flu_m.tolist() == [12.0, 1.0, 2.0]


def test_raw_rgb_metric_depth_sensor_bridge_round_trip() -> None:
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    depth_u16 = (np.arange(24, dtype="<u2") + 1).reshape(4, 6) * 100
    header = {
        "schema": "urbanfly-sensor-packet-v2",
        "sequence": 13,
        "timestamp_ns": 2_100_000_000,
        "sim_time": 2.1,
        "vehicle_name": "WM-UAV-01",
        "width": 6,
        "height": 4,
        "rgb_codec": "raw_rgb8",
        "rgb_length": rgb.nbytes,
        "depth_codec": "u16",
        "depth_compression": "none",
        "depth_length": depth_u16.nbytes,
        "depth_scale_m": 120.0 / 65535.0,
        "intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 2.5, "cy": 1.5},
        "goal_body_flu_m": [12.0, 1.0, 2.0],
        "linear_velocity_body_flu_mps": [1.0, 0.0, 0.0],
        "angular_velocity_body_flu_rps": [0.0, 0.0, 0.1],
    }
    header_bytes = json.dumps(header).encode()
    packet = (
        b"UFWM"
        + len(header_bytes).to_bytes(4, "little")
        + header_bytes
        + rgb.tobytes()
        + depth_u16.tobytes()
    )
    decoded = decode_urbanfly_sensor_packet(packet, episode_id="raw-bridge-test")
    assert np.array_equal(decoded.observation.rgb, rgb)
    assert decoded.observation.depth_m.shape == (4, 6)
    assert decoded.observation.depth_m[0, 0] == pytest.approx(
        100 * 120.0 / 65535.0
    )


def test_public_shards_train_and_load_provenanced_tdmpc2(tmp_path) -> None:
    simulator = MockSimulator(
        scenario="OpenSpace", depth_shape=(12, 20), control_dt=0.02
    )
    env = UrbanFlyWorldModelEnv(
        simulator,
        safety_layer=TransparentSafetyLayer(
            enabled=False, filter_config={"max_acceleration_mps2": 100.0}
        ),
    )
    observation, _ = env.reset(
        goal_nwu=np.asarray([20.0, 0.0, 2.0]), scenario="OpenSpace"
    )
    writer = WebDatasetShardWriter(tmp_path, shard_prefix="train")
    for step in range(3):
        zero = np.zeros(4, dtype=np.float32)
        audit = SafetyAudit(
            episode_id=observation.episode_id,
            step_id=step,
            sim_time=observation.sim_time,
            raw_action_normalized=zero,
            raw_action_physical=zero,
            executed_action_physical=zero,
            intervened=False,
            reasons=(),
            action_delta_l2=0.0,
            minimum_depth_m=20.0,
            predicted_risk=0.1,
        )
        writer.append(
            WorldModelStepRecord(
                observation=observation,
                raw_action=zero,
                executed_action=zero,
                reward=0.1,
                collision=False,
                minimum_clearance_m=20.0,
                zone_type="open",
                dynamics_parameters={},
                safety_audit=audit,
            )
        )
        if step < 2:
            observation, *_ = env.step(zero, shield_enabled=False)
    manifest = writer.close()
    dataset = UrbanFlyTransitionDataset([manifest])
    assert len(dataset) == 2
    batch = {
        name: torch.stack([dataset[0][name], dataset[1][name]])
        for name in dataset[0]
    }
    model = TDMPC2Network()
    trainer = TDMPC2Trainer(model)
    loss = trainer.train_step(batch)
    assert np.isfinite(loss.total)
    checkpoint = tmp_path / "trained.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "schema": SCHEMA_NAME,
            "family": "tdmpc2_continuous",
            "training_steps": 1,
        },
        checkpoint,
    )
    loaded = TDMPC2ContinuousPolicy(
        checkpoint=checkpoint,
        horizon=2,
        candidates=4,
        elites=2,
        iterations=1,
        device="cpu",
    )
    assert loaded.diagnostics()["trained"] is True


def test_offline_report_keeps_failures_and_writes_svg(tmp_path) -> None:
    records = [
        {
            "method": "tdmpc2",
            "group": "unseen_tiles",
            "route_id": "r1",
            "success": True,
            "path_length_m": 100,
            "shortest_path_m": 90,
            "minimum_clearance_m": 4,
            "latency_ms": 120,
            "decision_steps": 30,
            "intervention_steps": 1,
        },
        {
            "method": "tdmpc2",
            "group": "unseen_tiles",
            "route_id": "r2",
            "success": False,
            "collision": True,
            "collision_count": 1,
            "path_length_m": 80,
            "shortest_path_m": 90,
            "minimum_clearance_m": 0,
            "latency_ms": 160,
            "decision_steps": 20,
            "intervention_steps": 2,
            "failure_reason": "collision",
            "video_path": "r2.mp4",
        },
    ]
    paths = write_world_model_report(records, tmp_path)
    assert all(path.exists() for path in paths.values())
    report = paths["html"].read_text(encoding="utf-8")
    assert "r2.mp4" in report
    assert "50.0%" in report
