import json
from types import SimpleNamespace
from urllib import request

import numpy as np

from backend.agents.semantic_fleet import (
    DeterministicSemanticInterpreter,
    FleetCoordinator,
    FleetDrone,
    FleetTask,
    ObservationPacket,
    OpenAICompatibleQwenVLClient,
    SemanticEvent,
    SemanticEventGate,
    SemanticEventType,
    SemanticFleetRuntime,
)
from backend.agents.simulator_bridge import SemanticFleetSimulatorBridge
from backend.engine.models import DroneState, TaskStatus


def _event(**overrides):
    payload = {
        "event_id": "evt-1",
        "event_type": "temporary_obstacle",
        "timestamp_s": 10.0,
        "source_drone_id": "UAV-1",
        "position": [0, 20, 0],
        "radius_m": 10.0,
        "confidence": 0.9,
        "severity": 0.7,
        "ttl_s": 30.0,
        "evidence": "RGB-D detection",
        "source": "simulator",
    }
    payload.update(overrides)
    return SemanticEvent.from_mapping(
        payload,
        default_timestamp_s=10.0,
        default_source_drone_id="UAV-1",
    )


def test_event_gate_accepts_supported_fresh_event_and_rejects_stale_one():
    gate = SemanticEventGate()
    accepted = gate.validate(_event(), now_s=10.0, known_drone_ids={"UAV-1"})
    stale = gate.validate(
        _event(timestamp_s=1.0), now_s=10.0, known_drone_ids={"UAV-1"}
    )
    assert accepted.accepted
    assert stale.reason == "stale_event"


def test_event_gate_fails_closed_on_low_confidence_or_missing_evidence():
    gate = SemanticEventGate()
    low = gate.validate(
        _event(confidence=0.4), now_s=10.0, known_drone_ids={"UAV-1"}
    )
    missing = gate.validate(
        _event(evidence=""), now_s=10.0, known_drone_ids={"UAV-1"}
    )
    assert not low.accepted and low.reason == "low_confidence"
    assert not missing.accepted and missing.reason == "missing_evidence"


def test_vlm_hazard_requires_independent_type_specific_corroboration():
    gate = SemanticEventGate()
    unsupported = gate.validate(
        _event(source="vlm"), now_s=10.0, known_drone_ids={"UAV-1"}
    )
    supported = gate.validate(
        _event(source="vlm", temporal_support_count=2, depth_support=True),
        now_s=10.0,
        known_drone_ids={"UAV-1"},
    )
    assert not unsupported.accepted
    assert unsupported.reason == "insufficient_rgbd_temporal_support"
    assert supported.accepted


def test_qwen_json_parser_accepts_fenced_object():
    raw = "```json\n" + json.dumps({"events": []}) + "\n```"
    assert OpenAICompatibleQwenVLClient._parse_json_object(raw) == {"events": []}


def test_qwen_client_sends_bounded_multiframe_request_and_parses_event(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            result = {
                "choices": [{
                    "message": {
                        "content": json.dumps({"events": [{
                            "event_id": "qwen-1",
                            "event_type": "weather_hazard",
                            "position": [10, 30, -5],
                            "radius_m": 25,
                            "confidence": 0.91,
                            "severity": 0.6,
                            "ttl_s": 20,
                            "evidence": "consistent debris motion across frames",
                        }]})
                    }
                }]
            }
            return json.dumps(result).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    client = OpenAICompatibleQwenVLClient(
        "http://127.0.0.1:9000", timeout_s=2.5
    )
    packet = ObservationPacket(
        timestamp_s=12.0,
        drone_id="UAV-1",
        telemetry={"battery": 0.8},
        frame_data_urls=tuple(f"data:image/jpeg;base64,{index}" for index in range(6)),
    )
    events = client.analyze(packet)
    content = captured["body"]["messages"][0]["content"]
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["timeout"] == 2.5
    assert sum(item["type"] == "image_url" for item in content) == 4
    assert captured["body"]["temperature"] == 0.0
    assert events[0].event_id == "qwen-1"
    assert events[0].source_drone_id == "UAV-1"


def test_qwen_base_url_does_not_duplicate_v1(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return json.dumps({
                "choices": [{"message": {"content": "{\"events\":[]}"}}]
            }).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return Response()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    client = OpenAICompatibleQwenVLClient(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="test-only",
    )
    client.analyze(ObservationPacket(0.0, "UAV-1", {}, ()))
    assert captured["url"] == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


def test_no_fly_zone_generates_auditable_detour_for_intersecting_task():
    drones = [
        FleetDrone("UAV-A", (-30, 20, 0), 0.9, 5.0),
        FleetDrone("UAV-B", (-30, 20, 100), 0.9, 5.0),
    ]
    tasks = [
        FleetTask("CROSS", (-20, 20, 0), (30, 20, 0), 1.0, 0),
        FleetTask("CLEAR", (-20, 20, 100), (30, 20, 100), 1.0, 1),
    ]
    no_fly = _event(
        event_type="no_fly_zone", position=[0, 20, 0], radius_m=12.0
    )
    plan = FleetCoordinator(max_tasks_per_drone=1).allocate(
        drones, tasks, [no_fly], now_s=10.0
    )
    assert plan.blocked_task_ids == ()
    assert "CROSS" in plan.replan_task_ids
    assert len(plan.task_routes["CROSS"]) > 3
    assert len(plan.task_routes["CLEAR"]) == 3


def test_no_fly_zone_blocks_task_with_endpoint_inside_zone():
    drones = [FleetDrone("UAV-A", (-30, 20, 0), 0.9, 5.0)]
    tasks = [FleetTask("INSIDE", (-5, 20, 0), (30, 20, 0), 1.0, 0)]
    plan = FleetCoordinator().allocate(
        drones,
        tasks,
        [_event(event_type="no_fly_zone", position=[0, 20, 0], radius_m=12.0)],
        now_s=10.0,
    )
    assert plan.blocked_task_ids == ("INSIDE",)


def test_failure_event_reallocates_away_from_source_drone():
    runtime = SemanticFleetRuntime(DeterministicSemanticInterpreter())
    drones = [
        FleetDrone("UAV-A", (0, 20, 0), 0.9, 5.0),
        FleetDrone("UAV-B", (100, 20, 0), 0.9, 5.0),
    ]
    tasks = [FleetTask("T-1", (2, 20, 0), (10, 20, 0), 1.0, 0)]
    initial = runtime.reallocate(drones, tasks, now_s=0.0)
    assert initial.assignments["UAV-A"] == ("T-1",)

    packet = ObservationPacket(
        timestamp_s=5.0,
        drone_id="UAV-A",
        semantic_cues=({
            "event_id": "failure-a",
            "event_type": SemanticEventType.DRONE_FAILURE.value,
            "position": [0, 20, 0],
            "radius_m": 1.0,
            "confidence": 1.0,
            "severity": 1.0,
            "ttl_s": 30.0,
            "evidence": "health monitor",
        },),
    )
    decisions = runtime.ingest(packet, known_drone_ids={"UAV-A", "UAV-B"})
    updated = runtime.reallocate(drones, tasks, now_s=5.0)
    assert decisions[0].accepted
    assert updated.assignments["UAV-A"] == ()
    assert updated.assignments["UAV-B"] == ("T-1",)


def test_expired_event_is_removed_from_runtime():
    runtime = SemanticFleetRuntime(DeterministicSemanticInterpreter())
    runtime.ingest(
        ObservationPacket(
            timestamp_s=1.0,
            drone_id="UAV-1",
            semantic_cues=({
                "event_id": "short",
                "event_type": "goal_landmark",
                "position": [1, 2, 3],
                "radius_m": 0,
                "confidence": 0.9,
                "severity": 0.0,
                "ttl_s": 1.0,
                "evidence": "landmark sign",
            },),
        ),
        known_drone_ids={"UAV-1"},
    )
    runtime.expire(2.1)
    assert runtime.active_events == {}


def test_mid_mission_drone_failure_creates_explicit_payload_recovery_pickup():
    bridge = SemanticFleetSimulatorBridge()
    task = SimpleNamespace(
        id="T-RECOVER",
        status=TaskStatus.EN_ROUTE_DELIVERY,
        assigned_to="UAV-A",
        pickup_pos=np.array([0.0, 10.0, 0.0]),
    )
    drone = SimpleNamespace(
        id="UAV-A",
        current_task_id="T-RECOVER",
        position=np.array([12.0, 40.0, -7.0]),
        payload_current=2.5,
        state=DroneState.DELIVERING,
        path=[object()],
        assigned_tasks=["T-RECOVER"],
    )
    simulator = SimpleNamespace(drones=[drone], tasks=[task], time=33.0)
    bridge._apply_event_effect(
        simulator,
        _event(
            event_type="drone_failure",
            source_drone_id="UAV-A",
            position=[12.0, 40.0, -7.0],
        ),
    )
    assert drone.state == DroneState.EMERGENCY
    assert drone.current_task_id is None
    assert drone.payload_current == 0.0
    assert task.status == TaskStatus.PENDING
    assert task.assigned_to is None
    np.testing.assert_allclose(task.pickup_pos, [12.0, 40.0, -7.0])
    assert bridge.failure_recoveries[0]["payload_was_onboard"] is True
    assert bridge.failed_drone_ids == {"UAV-A"}


def test_external_qwen_proposal_is_checked_against_current_simulator_clock():
    bridge = SemanticFleetSimulatorBridge()
    bridge.enabled = True
    bridge.accept_external_proposals = True
    simulator = SimpleNamespace(
        time=20.0,
        drones=[SimpleNamespace(id="UAV-A")],
        events=[],
    )
    decisions = bridge.ingest_external_proposals(
        simulator,
        observer_drone_id="UAV-A",
        observation_timestamp_s=10.0,
        proposals=[{
            "event_id": "late-qwen",
            "event_type": "temporary_obstacle",
            "position": [0, 20, 0],
            "radius_m": 10,
            "confidence": 0.95,
            "severity": 0.8,
            "ttl_s": 20,
            "evidence": "old frame",
        }],
    )
    assert decisions == [{
        "event_id": "late-qwen", "accepted": False, "reason": "stale_event"
    }]
    assert bridge.runtime.active_events == {}


def test_external_qwen_cannot_self_assert_trusted_corroboration():
    bridge = SemanticFleetSimulatorBridge()
    bridge.enabled = True
    bridge.accept_external_proposals = True
    simulator = SimpleNamespace(
        time=10.0,
        drones=[SimpleNamespace(id="UAV-A")],
        events=[],
    )
    decisions = bridge.ingest_external_proposals(
        simulator,
        observer_drone_id="UAV-A",
        observation_timestamp_s=10.0,
        proposals=[{
            "event_id": "self-asserted",
            "event_type": "temporary_obstacle",
            "position": [0, 20, 0],
            "radius_m": 10,
            "confidence": 0.99,
            "severity": 0.9,
            "ttl_s": 20,
            "evidence": "model says it saw motion",
            "temporal_support_count": 4,
            "depth_support": True,
        }],
    )
    assert decisions[0]["accepted"] is False
    assert decisions[0]["reason"] == "insufficient_rgbd_temporal_support"
