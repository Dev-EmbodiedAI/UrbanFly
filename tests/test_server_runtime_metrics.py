import asyncio
import json

import pytest

from backend.server.runtime_metrics import RuntimeMetrics
from backend.server.server import SimulationServer


def test_runtime_metrics_snapshot_is_bounded_and_reports_percentiles():
    metrics = RuntimeMetrics(window_size=4)
    metrics.increment("frames", 2)
    for value in (1, 2, 3, 4, 100):
        metrics.observe("latency_ms", value)

    snapshot = metrics.snapshot()

    assert snapshot["counters"]["frames"] == 2
    assert snapshot["windows"]["latency_ms"]["samples"] == 4
    assert snapshot["windows"]["latency_ms"]["mean"] == pytest.approx(27.25)
    assert snapshot["windows"]["latency_ms"]["p50"] == pytest.approx(3.5)
    assert snapshot["windows"]["latency_ms"]["maximum"] == 100


def test_latest_state_queue_coalesces_slow_ui_client():
    async def exercise():
        class SlowSocket:
            def __init__(self):
                self.release = asyncio.Event()
                self.messages = []

            async def send_str(self, message):
                self.messages.append(message)
                await self.release.wait()

        server = SimulationServer()
        socket = SlowSocket()
        server._clients.add(socket)
        server._client_send_locks[socket] = asyncio.Lock()

        server._queue_latest_state(socket, "state-1")
        await asyncio.sleep(0)
        server._queue_latest_state(socket, "state-2")
        server._queue_latest_state(socket, "state-3")
        socket.release.set()
        await server._client_state_tasks[socket]

        assert socket.messages == ["state-1", "state-3"]
        assert server.metrics.snapshot()["counters"]["sim_state_coalesced_total"] == 1

    asyncio.run(exercise())


def test_client_runtime_is_sanitized_bounded_and_removed_on_disconnect():
    async def exercise():
        server = SimulationServer()
        socket = object()
        server._clients.add(socket)
        server._update_client_runtime(socket, {
            "surface": "desktop", "scene_ready": True, "hidden": True,
            "presentation": {"fps": 60, "textures": float("nan"), "extra": "x" * 10000},
            "sensors": {"bridge_frames": -1, "bridge_enabled": True},
        })
        response = await server._health_handler(None)
        surfaces = json.loads(response.text)["surfaces"]
        assert surfaces[0]["presentation"] == {"fps": 60, "idle": False}
        assert surfaces[0]["sensors"] == {"bridge_enabled": True, "streaming": False}
        assert surfaces[0]["scene_ready"] is True
        server._drop_client(socket)
        assert not server._client_runtime
    asyncio.run(exercise())


def test_capture_start_bookkeeping_is_bounded_without_packets():
    async def exercise():
        server = SimulationServer()
        for index in range(500):
            await server._handle_message(None, json.dumps({
                "type": "sensor_capture_started",
                "payload": {"sim_time": index, "vehicle_name": "probe"},
            }))
        assert len(server._capture_started_wall) == 256
    asyncio.run(exercise())


def test_policy_client_state_is_not_coalesced():
    async def exercise():
        class Socket:
            def __init__(self):
                self.messages = []

            async def send_str(self, message):
                self.messages.append(message)

        server = SimulationServer()
        socket = Socket()
        server._clients.add(socket)
        server._policy_clients.add(socket)
        server._client_send_locks[socket] = asyncio.Lock()

        await server._broadcast("reliable-state", coalesce_state=True)

        assert socket.messages == ["reliable-state"]
        assert socket not in server._client_state_tasks

    asyncio.run(exercise())


def test_health_snapshot_exposes_runtime_schema_and_client_counts():
    async def exercise():
        server = SimulationServer()
        response = await server._health_handler(None)
        payload = response.text

        assert response.status == 200
        assert '"schema": "urbanfly-runtime-health-v1"' in payload
        assert '"total": 0' in payload
        assert '"metrics"' in payload

    asyncio.run(exercise())
