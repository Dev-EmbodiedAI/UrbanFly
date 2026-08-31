from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.digital_twin import (
    DigitalTwinMission,
    DigitalTwinObservation,
    EXPECTED_SWARM_ENVIRONMENTS,
    GoalConditionedWorldModelPolicy,
    audit_cross_environment_reports,
)


def mission() -> DigitalTwinMission:
    return DigitalTwinMission(
        environment_id="test:city",
        episode_id="episode-1",
        starts_enu_m=np.asarray([[0, 0, 2], [0, 5, 2]], dtype=np.float32),
        goals_enu_m=np.asarray([[10, 5, 2], [10, 0, 2]], dtype=np.float32),
        agent_provider="test_agent",
        privileged_goal_mode=True,
    )


def observation(sequence: int, timestamp_s: float) -> DigitalTwinObservation:
    state = np.zeros((2, 190), dtype=np.float32)
    state[0, :3] = [0, 0, 2]
    state[1, :3] = [0, 5, 2]
    state[:, 5] = 0.0
    depth = np.ones((2, 128, 128, 1), dtype=np.float32)
    return DigitalTwinObservation(
        environment_id="test:city",
        episode_id="episode-1",
        sequence=sequence,
        timestamp_s=timestamp_s,
        depth=depth,
        state=state,
    )


def test_agent_assignment_and_world_model_action_contract() -> None:
    model = GoalConditionedWorldModelPolicy()
    assigned = model.reset(mission())
    np.testing.assert_allclose(assigned, [[10, 0, 2], [10, 5, 2]])
    result = model.act(observation(0, 0.0))
    assert result.action.shape == (2, 5)
    assert result.candidate_scores.shape == (2, 5)
    assert np.isfinite(result.action).all()
    assert np.all((-1 <= result.action[:, :3]) & (result.action[:, :3] <= 1))
    assert np.all((0 <= result.action[:, 3]) & (result.action[:, 3] <= 1))
    assert np.all((-1 <= result.action[:, 4]) & (result.action[:, 4] <= 1))
    assert np.all(result.predicted_clearance_m > 19.0)


def test_world_model_rejects_reused_observation() -> None:
    model = GoalConditionedWorldModelPolicy()
    model.reset(mission())
    current = observation(0, 0.0)
    model.act(current)
    with pytest.raises(RuntimeError, match="stale observation"):
        model.act(current)


def test_observation_contract_rejects_bad_depth_range() -> None:
    current = observation(0, 0.0)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        DigitalTwinObservation(
            environment_id=current.environment_id,
            episode_id=current.episode_id,
            sequence=0,
            timestamp_s=0.0,
            depth=np.full((2, 128, 128, 1), 2.0, dtype=np.float32),
            state=current.state,
        )


def _qa_report(environment: str) -> dict:
    return {
        "schema": "urbanfly-cross-environment-digital-twin-navigation-v1",
        "status": "PASS",
        "environment": environment,
        "seed": 7,
        "requested_drones": 2,
        "control_contract": "submission_zip.v1",
        "benchmark_eligible": False,
        "steps": 10,
        "agent_observations": 11,
        "world_model_decisions": 10,
        "executions": 10,
        "fresh_feedbacks": 10,
        "causal_chain_complete": True,
        "success": True,
        "per_drone_success": [True, True],
        "per_drone_collision": [False, False],
        "per_drone_failure_reason": ["NONE", "NONE"],
        "minimum_predicted_clearance_m": [2.5, 3.0],
        "minimum_predicted_separation_m": [5.0, 5.0],
        "native_score": {"final_score": 0.8},
    }


def test_cross_environment_qa_requires_complete_causal_safe_matrix(tmp_path: Path) -> None:
    paths = []
    for index, environment in enumerate(EXPECTED_SWARM_ENVIRONMENTS):
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(_qa_report(environment)), encoding="utf-8")
        paths.append(path)
    result = audit_cross_environment_reports(paths)
    assert result["status"] == "PASS"
    assert result["episodes"] == 5
    assert result["successful_drones"] == result["total_drones"] == 10
    assert result["collisions"] == 0
    assert result["total_steps"] == result["world_model_decisions"] == 50


def test_cross_environment_qa_fails_closed_on_collision(tmp_path: Path) -> None:
    paths = []
    for index, environment in enumerate(EXPECTED_SWARM_ENVIRONMENTS):
        report = _qa_report(environment)
        if index == 0:
            report["per_drone_collision"][0] = True
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="zero_collision"):
        audit_cross_environment_reports(paths)
