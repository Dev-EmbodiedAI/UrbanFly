"""Real Helsinki runtime diagnostic; never writes or resumes dataset episodes."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import urllib.request

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "uav_wm_navigation" / "src"))
from uav_wm_navigation.simulators.helsinki_websocket_adapter import HelsinkiWebSocketAdapter


def health():
    with urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=3) as response:
        return json.load(response)


async def stop_simulator():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect("http://127.0.0.1:8765/ws") as socket:
            await socket.send_json({"type": "control", "payload": {"action": "stop"}})
            await asyncio.sleep(0.2)


def distribution(values):
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))} if values else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resets", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--first-task", type=int, default=53)
    parser.add_argument("--sensor-snapshot", type=Path,
                        help="Optional initial RGB-D/state NPZ for read-only image equivalence QA")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.sensor_snapshot and args.sensor_snapshot.exists():
        raise FileExistsError(args.sensor_snapshot)
    before = health()
    if before["clients"]["policy"]:
        raise RuntimeError("An existing policy client is active; do not interfere")
    surfaces = [item for item in before["surfaces"] if item["age_s"] < 5]
    if len(surfaces) != 1 or not surfaces[0]["scene_ready"]:
        raise RuntimeError("Exactly one fresh, fully loaded real sensor surface is required")
    manifest = ROOT / "outputs/helsinki_dataset_v1/real_500_dataset_v1_20260825/smoke_tasks.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))
    result = {"schema": "urbanfly-runtime-probe-v1", "dataset_episodes_written": 0,
              "kind": "reset + hover lockstep diagnostics, NOT full navigation episodes",
              "health_before": before, "resets": [], "status": "RUNNING"}
    adapter = HelsinkiWebSocketAdapter({
        "websocket_url": "ws://127.0.0.1:8765/ws", "urbanfly_scenario": "single_uav_world_model",
        "vehicle_name": "WM-UAV-01", "policy_family": "runtime_pipeline_probe",
        "backend_safety_shield": True, "policy_lockstep": True,
        "sensor_timeout_s": 20.0, "command_timeout_s": 5.0, "dynamic_actor_density": 0.0,
    })
    wall_latencies, sim_dts = [], []
    try:
        adapter.connect()
        for index in range(args.resets):
            record = records[args.first_task + index]
            route = np.asarray(record["route_enu"], dtype=np.float64)
            entry = {"task_index": args.first_task + index, "task_type": record["task"]["task_type"],
                     "actions": [], "status": "RUNNING"}
            result["resets"].append(entry)
            started = time.perf_counter()
            adapter.reset()
            adapter.configure_scenario(record["task"]["task_type"], "runtime_probe", 20260828 + index)
            adapter.set_initial_pose(route[0])
            adapter.set_goal(route[-1])
            adapter.takeoff()
            frame = adapter.get_depth()
            state = adapter.get_kinematics()
            if index == 0 and args.sensor_snapshot:
                args.sensor_snapshot.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.sensor_snapshot, rgb=frame.rgb, depth=frame.depth_m,
                                    position=state.position, orientation=state.orientation_xyzw,
                                    timestamp=frame.timestamp, intrinsics=frame.camera_intrinsics)
            entry["first_frame_s"] = time.perf_counter() - started
            previous = frame.timestamp
            for step in range(args.steps):
                started = time.perf_counter()
                action = adapter.execute_velocity_command(np.zeros(3), 0.0, 0.1)
                frame = adapter.get_depth()
                state = adapter.get_kinematics()
                elapsed = time.perf_counter() - started
                dt = frame.timestamp - previous
                assert dt > 0, f"Timestamp regression: {dt}"
                assert abs(state.timestamp - frame.timestamp) < 1e-6
                assert frame.rgb.shape == (90, 160, 3) and frame.rgb.dtype == np.uint8
                assert frame.depth_m.shape == (90, 160) and np.isfinite(frame.depth_m).all()
                assert abs(np.linalg.norm(state.orientation_xyzw) - 1) < 1e-4
                assert np.isfinite(action["action_executed_body_flu"]).all()
                assert action["step_id"] == step, "Cross-reset policy sequence inheritance"
                wall_latencies.append(elapsed)
                sim_dts.append(dt)
                entry["actions"].append({"step_id": step, "timestamp": frame.timestamp,
                    "dt_s": dt, "wall_s": elapsed, "stale_action": action["stale_action"],
                    "executed_body_flu": action["action_executed_body_flu"].tolist()})
                previous = frame.timestamp
            entry["status"] = "PASS"
            print(f"reset {index + 1}/{args.resets}: {len(entry['actions'])} synchronized actions PASS", flush=True)
        result["status"] = "PASS"
    except BaseException as error:
        result["status"] = "FAIL"
        result["error"] = repr(error)
        raise
    finally:
        try:
            adapter.close()
            asyncio.run(stop_simulator())
        except Exception as error:
            result["cleanup_error"] = repr(error)
            result["status"] = "FAIL"
        result["actions_completed"] = len(wall_latencies)
        result["wall_action_s"] = distribution(wall_latencies)
        result["sim_dt_s"] = distribution(sim_dts)
        result["stale_actions"] = sum(action["stale_action"] for item in result["resets"] for action in item["actions"])
        try:
            result["health_after"] = health()
        except Exception as error:
            result["health_after_error"] = repr(error)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({key: result[key] for key in (
            "status", "actions_completed", "wall_action_s", "sim_dt_s", "stale_actions")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
