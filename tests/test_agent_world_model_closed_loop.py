from __future__ import annotations

import pytest

from backend.agents.helsinki_closed_loop import (
    AgentStatus,
    ClosedLoopViolation,
    HelsinkiAgentWorldModelRuntime,
    SemanticMissionPlan,
    WorldModelActionDecision,
)


def decision(step: int) -> WorldModelActionDecision:
    return WorldModelActionDecision.create(
        step=step,
        selected_index=1,
        candidate_count=3,
        predicted_risk=0.12,
        uncertainty=0.08,
        action_body_flu=[1.0, 0.0, 0.0, 0.0],
    )


def test_semantic_mission_requires_passed_bounded_authority() -> None:
    report = {
        "status": "PASS",
        "deterministic_gate": "PASS",
        "provider": "qwen_openai_compatible_api",
        "model": "qwen-plus",
        "api_called": True,
        "control_authority": "semantic waypoint ordering only; no flight actions",
        "waypoint_order": ["A", "B"],
        "ordered_waypoints_backend": [[1, 2, 3], [4, 5, 6]],
    }
    mission = SemanticMissionPlan.from_report(report)
    assert mission.api_called is True
    assert mission.waypoint_order == ("A", "B")

    report["control_authority"] = "unbounded flight commands"
    with pytest.raises(ClosedLoopViolation, match="safely bounded"):
        SemanticMissionPlan.from_report(report)


def test_agent_world_model_execution_feedback_chain_completes() -> None:
    mission = SemanticMissionPlan.deterministic([[5, 0, 0], [10, 0, 0]])
    runtime = HelsinkiAgentWorldModelRuntime(
        mission,
        [[5, 0, 0], [10, 0, 0]],
        semantic_waypoint_tolerance_m=0.2,
        final_goal_tolerance_m=0.2,
    )
    initial = runtime.begin(observation_timestamp_s=0.1, position_enu=[0, 0, 0])
    assert initial.status is AgentStatus.RUNNING

    runtime.authorize_world_model(decision(0))
    first = runtime.accept_execution_feedback(
        step=0,
        feedback_timestamp_s=0.2,
        position_enu=[5, 0, 0],
        executed_action_body_flu=[1, 0, 0, 0],
        stale_action=False,
        collision=False,
        safety_intervened=False,
    )
    assert first.status is AgentStatus.RUNNING
    assert first.active_waypoint_index == 1

    runtime.authorize_world_model(decision(1))
    final = runtime.accept_execution_feedback(
        step=1,
        feedback_timestamp_s=0.3,
        position_enu=[10, 0, 0],
        executed_action_body_flu=[1, 0, 0, 0],
        stale_action=False,
        collision=False,
        safety_intervened=True,
    )
    assert final.status is AgentStatus.COMPLETE
    snapshot = runtime.snapshot()
    assert snapshot["causal_chain_complete"] is True
    assert snapshot["agent_observations"] == 3
    assert snapshot["world_model_decisions"] == 2
    assert snapshot["actions_authorized"] == 2
    assert snapshot["executions"] == 2
    assert snapshot["fresh_feedbacks"] == 2
    assert snapshot["feedback_to_next_policy"] == 1
    assert snapshot["waypoints_reached"] == 2
    assert snapshot["safety_interventions"] == 1


@pytest.mark.parametrize(
    ("stale", "collision", "message"),
    [
        (True, False, "stale action"),
        (False, True, "collision"),
    ],
)
def test_agent_aborts_on_unsafe_execution_feedback(
    stale: bool, collision: bool, message: str
) -> None:
    runtime = HelsinkiAgentWorldModelRuntime(
        SemanticMissionPlan.deterministic([[10, 0, 0]]),
        [[10, 0, 0]],
    )
    runtime.begin(observation_timestamp_s=1.0, position_enu=[0, 0, 0])
    runtime.authorize_world_model(decision(0))
    with pytest.raises(ClosedLoopViolation, match=message):
        runtime.accept_execution_feedback(
            step=0,
            feedback_timestamp_s=1.1,
            position_enu=[0.1, 0, 0],
            executed_action_body_flu=[0, 0, 0, 0],
            stale_action=stale,
            collision=collision,
            safety_intervened=False,
        )
    snapshot = runtime.snapshot()
    assert snapshot["status"] == "ABORTED"
    assert snapshot["causal_chain_complete"] is False


def test_feedback_must_be_newer_than_action_observation() -> None:
    runtime = HelsinkiAgentWorldModelRuntime(
        SemanticMissionPlan.deterministic([[10, 0, 0]]),
        [[10, 0, 0]],
    )
    runtime.begin(observation_timestamp_s=2.0, position_enu=[0, 0, 0])
    runtime.authorize_world_model(decision(0))
    with pytest.raises(ClosedLoopViolation, match="not newer"):
        runtime.accept_execution_feedback(
            step=0,
            feedback_timestamp_s=2.0,
            position_enu=[0, 0, 0],
            executed_action_body_flu=[0, 0, 0, 0],
            stale_action=False,
            collision=False,
            safety_intervened=False,
        )
