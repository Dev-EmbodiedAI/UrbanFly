from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backend.digital_twin import HelsinkiDigitalTwinAdapter


def test_formal_helsinki_video_uses_validated_command_horizon_by_default():
    from scripts.run_helsinki_world_model_video import parser

    args = parser().parse_args(
        [
            "--policy",
            "policy.pt",
            "--world-model",
            "world_model.pt",
            "--output-dir",
            "output",
        ]
    )
    assert args.action_duration_s == 0.5


class FakeHelsinkiRawAdapter:
    def __init__(self) -> None:
        self.timestamp = 0.1
        self.closed = False
        self.calls: list[str] = []

    def connect(self): self.calls.append("connect")
    def reset(self): self.calls.append("reset")
    def configure_scenario(self, *_): self.calls.append("configure")
    def set_initial_pose(self, *_): self.calls.append("start")
    def set_goal(self, *_): self.calls.append("goal")
    def takeoff(self): self.calls.append("takeoff")

    def get_depth(self):
        return SimpleNamespace(
            timestamp=self.timestamp,
            rgb=np.zeros((8, 12, 3), dtype=np.uint8),
            depth_m=np.ones((8, 12), dtype=np.float32),
            valid_mask=np.ones((8, 12), dtype=bool),
        )

    def get_kinematics(self):
        return SimpleNamespace(
            position=np.asarray([1.0, 2.0, 3.0]),
            orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

    def execute_velocity_command(self, *_args, **_kwargs):
        self.timestamp += 0.1
        return {
            "stale_action": False,
            "safety_intervened": False,
            "action_executed_body_flu": [0.0, 0.0, 0.0, 0.0],
        }

    def get_collision_info(self): return {"has_collided": False}
    def publish_policy_visualization(self, *_): pass
    def start_synchronized_recording(self, *_args, **_kwargs): pass
    def stop_synchronized_recording(self): return {}
    def close(self): self.closed = True


def test_helsinki_adapter_enforces_reset_action_fresh_feedback_lifecycle() -> None:
    raw = FakeHelsinkiRawAdapter()
    adapter = HelsinkiDigitalTwinAdapter(raw_adapter=raw)
    initial = adapter.connect_and_reset(
        task_type="rooftop_to_ground",
        split="test",
        seed=1,
        start_enu_m=np.zeros(3),
        goal_enu_m=np.ones(3),
    )
    assert initial.sequence == 0
    assert raw.calls == ["connect", "reset", "configure", "start", "goal", "takeoff"]
    feedback = adapter.step_velocity(
        np.zeros(3), 0.0, 0.1, inference_latency_ms=1.0, predicted_risk=0.0
    )
    assert feedback.observation.sequence == 1
    assert feedback.observation.timestamp_s > initial.timestamp_s
    assert not feedback.stale_action
    assert not feedback.has_collided
    adapter.close()
    assert raw.closed and not adapter.active


def test_helsinki_adapter_rejects_stale_sensor_feedback() -> None:
    raw = FakeHelsinkiRawAdapter()
    adapter = HelsinkiDigitalTwinAdapter(raw_adapter=raw)
    adapter.connect_and_reset(
        task_type="test",
        split="test",
        seed=1,
        start_enu_m=np.zeros(3),
        goal_enu_m=np.ones(3),
    )
    with pytest.raises(RuntimeError, match="timestamp"):
        adapter._observe()
