from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from uav_wm_navigation.data.helsinki_dataset_v1 import (
    HelsinkiDatasetV1Writer,
    HelsinkiTransition,
    validate_helsinki_dataset_v1,
)
from uav_wm_navigation.data.helsinki_dataset_v1_qa import (
    audit_helsinki_collection,
    maximum_true_burst,
)
from uav_wm_navigation.types import SensorFrame, VehicleState


def _state(timestamp: float, east: float) -> VehicleState:
    return VehicleState(
        timestamp=timestamp,
        position=np.asarray([east, 5.0, 12.0]),
        orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=np.asarray([1.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        frame="enu",
    )


def _transition(index: int, final: bool = False) -> HelsinkiTransition:
    timestamp = 1.0 + index * 0.1
    sensor = SensorFrame(
        timestamp=timestamp,
        rgb=np.full((6, 8, 3), index, dtype=np.uint8),
        depth_m=np.full((6, 8), 10.0, dtype=np.float32),
        valid_mask=np.ones((6, 8), dtype=bool),
        camera_intrinsics=np.asarray([[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]]),
        camera_pose_nwu=np.eye(4),
    )
    return HelsinkiTransition(
        sensor=sensor,
        state=_state(timestamp, index * 0.1),
        next_state=_state(timestamp + 0.1, (index + 1) * 0.1),
        wall_timestamp=1000.0 + timestamp,
        action_timestamp=timestamp + 0.001,
        global_goal_world=np.asarray([10.0, 5.0, 12.0]),
        local_goal_world=np.asarray([5.0 + index * 0.1, 5.0, 12.0]),
        local_goal_body=np.asarray([5.0, 0.0, 0.0]),
        global_route_progress=index * 0.1,
        remaining_route_distance=10.0 - index * 0.1,
        action_commanded=np.asarray([1.0, 0.0, 0.0, 0.0]),
        action_executed=np.asarray([1.0, 0.0, 0.0, 0.0]),
        reward=0.1,
        collision=False,
        minimum_clearance=4.0,
        success=final,
        terminated=final,
        truncated=False,
    )


def test_dataset_v1_atomic_write_and_integrity_readback(tmp_path) -> None:
    writer = HelsinkiDatasetV1Writer(
        tmp_path,
        "episode-000",
        {
            "scene_id": "HelsinkiCentral1km",
            "scene_seed": 7,
            "task_type": "building_blocked",
            "collection_mode": "expert",
            "start_world": [0.0, 5.0, 12.0],
            "global_goal_world": [10.0, 5.0, 12.0],
            "spatial_split": "train",
            "urban_region_type": "dense_core",
        },
    )
    writer.append(_transition(0))
    writer.append(_transition(1, final=True))
    result = validate_helsinki_dataset_v1(writer.close())
    assert result["status"] == "PASS"
    assert result["steps"] == 2
    assert all(result["checks"].values())
    assert not list(tmp_path.glob("*.partial"))


def test_dataset_v1_rejects_off_by_one_action_timestamp() -> None:
    transition = _transition(0)
    with pytest.raises(ValueError, match="action timestamp"):
        HelsinkiTransition(
            **{
                name: getattr(transition, name)
                for name in transition.__dataclass_fields__
                if name != "action_timestamp"
            },
            action_timestamp=transition.next_state.timestamp,
        )


def test_collection_qa_audits_stale_bursts_phases_and_resets(tmp_path) -> None:
    stale_patterns = ([True, True, False], [False, True, True])
    task_types = ("building_blocked", "street_canyon")
    for episode_index, (pattern, task_type) in enumerate(zip(stale_patterns, task_types)):
        writer = HelsinkiDatasetV1Writer(
            tmp_path,
            f"episode-{episode_index:03d}",
            {
                "scene_id": "HelsinkiCentral1km",
                "scene_seed": episode_index,
                "task_type": task_type,
                "collection_mode": "expert",
                "start_world": [0.0, 5.0, 12.0],
                "global_goal_world": [10.0, 5.0, 12.0],
                "spatial_split": "train",
                "urban_region_type": "dense_core",
            },
        )
        for index, stale in enumerate(pattern):
            transition = _transition(index, final=index == len(pattern) - 1)
            if stale:
                transition = replace(
                    transition,
                    stale_action=True,
                    action_executed=np.zeros(4, dtype=np.float32),
                )
            writer.append(transition)
        writer.close()

    report = audit_helsinki_collection(
        tmp_path,
        expected_episodes=2,
        reset_transitions=[
            {
                "automatic_reset": "PASS",
                "checks": {"stale_action_inheritance_absent": True},
            }
        ],
        output_path=tmp_path / "checkpoint_002_qa.json",
    )

    assert report["status"] == "PASS"
    assert report["stale_action"]["count"] == 4
    assert report["stale_action"]["ratio"] == pytest.approx(4 / 6)
    assert report["stale_action"]["maximum_burst"] == 2
    assert report["stale_action"]["by_phase"]["middle"]["stale_action_count"] == 2
    assert report["stale_action"]["executed_timeout_hover_correct"]
    assert report["reset"]["passes"] == 1
    assert report["gate_checks"]["cross_episode_stale_action_zero"]
    assert not list(tmp_path.glob("*.partial"))


def test_maximum_true_burst_handles_edges() -> None:
    assert maximum_true_burst(np.asarray([], dtype=bool)) == 0
    assert maximum_true_burst(np.asarray([True, True, False, True])) == 2
