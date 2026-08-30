#!/usr/bin/env python
"""Run one auditable Helsinki learned-policy episode without writing a dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import aiohttp
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "uav_wm_navigation" / "src"))

from uav_wm_navigation.control.route_manager import PolylineRoute  # noqa: E402
from uav_wm_navigation.simulators.helsinki_websocket_adapter import (  # noqa: E402
    HelsinkiWebSocketAdapter,
)
from urbanfly_vln.observation_policy import (  # noqa: E402
    ACTION_LIMITS,
    load_observation_policy_checkpoint,
)
from urbanfly_vln.observation_policy_data import (  # noqa: E402
    load_qa_episode_records,
    public_state_features,
)


def health() -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=3) as response:
        return json.load(response)


async def stop_simulator() -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect("http://127.0.0.1:8765/ws") as socket:
            await socket.send_json({"type": "control", "payload": {"action": "stop"}})
            await asyncio.sleep(0.2)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--qa", type=Path, required=True)
    result.add_argument("--episode", type=int, default=80)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-steps", type=int, default=1400)
    result.add_argument("--action-duration-s", type=float, default=0.1)
    result.add_argument("--goal-tolerance-m", type=float, default=3.0)
    result.add_argument("--maximum-cross-track-m", type=float, default=15.0)
    result.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return result


def read_episode_contract(record) -> dict[str, object]:
    with h5py.File(record.path, "r") as handle:
        route = np.asarray(handle["episode/global_route_world"][:], dtype=np.float64)
        start = np.asarray(handle["episode/start_world"][:], dtype=np.float64)
        goal = np.asarray(handle["episode/global_goal_world"][:], dtype=np.float64)
        initial_orientation = np.asarray(handle["state/orientation_xyzw"][0], dtype=np.float64)
    initial_yaw = float(Rotation.from_quat(initial_orientation).as_euler("xyz")[2])
    return {"route": route, "start": start, "goal": goal, "initial_yaw": initial_yaw}


def inference(
    model,
    rgb_history: deque[np.ndarray],
    depth_history: deque[np.ndarray],
    valid_history: deque[np.ndarray],
    state_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    while len(rgb_history) < model.config.history_frames:
        rgb_history.appendleft(rgb_history[0].copy())
        depth_history.appendleft(depth_history[0].copy())
        valid_history.appendleft(valid_history[0].copy())
    rgb = torch.from_numpy(np.stack(rgb_history)).unsqueeze(0).to(device)
    depth = torch.from_numpy(np.stack(depth_history)).unsqueeze(0).to(device)
    valid = torch.from_numpy(np.stack(valid_history)).unsqueeze(0).to(device)
    state = torch.from_numpy(state_features).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        action = model(rgb, depth, valid, state)[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return action.cpu().numpy(), (time.perf_counter() - started) * 1000.0


def terminal_capture_action(
    learned_action: np.ndarray,
    local_goal_body: np.ndarray,
    remaining_route_m: float,
    *,
    blend_start_m: float = 8.0,
    full_capture_m: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Blend a transparent Local-Goal terminal capture near route completion.

    This is a generic terminal controller, not a route planner and not an
    episode-specific override.  It closes the out-of-distribution case where
    the learned policy passes the final point and has never seen the goal
    behind the aircraft in expert demonstrations.
    """

    learned = np.asarray(learned_action, dtype=np.float64)
    goal = np.asarray(local_goal_body, dtype=np.float64)
    if learned.shape != (4,) or goal.shape != (3,) or not np.isfinite(learned).all() or not np.isfinite(goal).all():
        raise ValueError("terminal capture requires finite action/local-goal vectors")
    if not 0.0 <= full_capture_m < blend_start_m:
        raise ValueError("terminal capture blend distances are invalid")
    blend = float(
        np.clip(
            (blend_start_m - float(remaining_route_m))
            / (blend_start_m - full_capture_m),
            0.0,
            1.0,
        )
    )
    distance = float(np.linalg.norm(goal))
    if distance <= 1e-6 or blend <= 0.0:
        return np.clip(learned, -ACTION_LIMITS, ACTION_LIMITS), blend
    speed = float(np.clip(0.6 * distance, 0.5, 2.5))
    capture = np.zeros(4, dtype=np.float64)
    capture[:3] = goal / distance * speed
    capture[3] = float(np.clip(np.arctan2(goal[1], max(goal[0], 0.2)), -0.6, 0.6))
    action = (1.0 - blend) * learned + blend * capture
    return np.clip(action, -ACTION_LIMITS, ACTION_LIMITS), blend


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.action_duration_s <= 0 or args.max_steps <= 0:
        raise ValueError("action duration and max steps must be positive")
    before = health()
    if int((before.get("clients") or {}).get("policy", 0)) != 0:
        raise RuntimeError("an existing policy client is active; refusing to interfere")
    surfaces = [item for item in before.get("surfaces", []) if float(item.get("age_s", 99)) < 5]
    if len(surfaces) != 1 or not bool(surfaces[0].get("scene_ready")):
        raise RuntimeError("exactly one fresh, fully loaded real sensor surface is required")

    records = load_qa_episode_records(args.qa)
    by_index = {record.episode_index: record for record in records}
    if args.episode not in by_index:
        raise ValueError(f"episode {args.episode:03d} is not in the selected QA set")
    record = by_index[args.episode]
    contract = read_episode_contract(record)
    route = PolylineRoute(
        contract["route"],
        normal_lookahead_m=20.0,
        turn_lookahead_m=20.0,
        lookahead_speed_gain_s=0.0,
        maximum_lookahead_m=20.0,
    )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    model, model_metadata = load_observation_policy_checkpoint(args.checkpoint, device=device)
    history_size = model.config.history_frames
    rgb_history: deque[np.ndarray] = deque(maxlen=history_size)
    depth_history: deque[np.ndarray] = deque(maxlen=history_size)
    valid_history: deque[np.ndarray] = deque(maxlen=history_size)
    previous_action = np.zeros(4, dtype=np.float32)
    result = {
        "schema": "urbanfly-helsinki-observation-policy-online-qa-v1",
        "status": "RUNNING",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_offline_status": model_metadata.get("status"),
        "source_qa": str(args.qa.resolve()),
        "episode_index": args.episode,
        "episode_id": record.episode_id,
        "task_type": record.task_type,
        "dataset_episodes_written": 0,
        "steps": [],
        "health_before": before,
    }
    adapter = HelsinkiWebSocketAdapter(
        {
            "websocket_url": "ws://127.0.0.1:8765/ws",
            "urbanfly_scenario": "single_uav_world_model",
            "vehicle_name": "WM-UAV-01",
            "policy_family": "helsinki_observation_policy_v1",
            "backend_safety_shield": True,
            "policy_lockstep": True,
            "sensor_timeout_s": 20.0,
            "command_timeout_s": 5.0,
            "dynamic_actor_density": 0.0,
            "initial_yaw_enu_radians": contract["initial_yaw"],
        }
    )
    try:
        adapter.connect()
        adapter.reset()
        adapter.configure_scenario(record.task_type, "learned_policy_online_qa", 20260829 + args.episode)
        adapter.set_initial_pose(contract["start"])
        adapter.set_goal(contract["goal"])
        adapter.takeoff()
        frame = adapter.get_depth()
        state = adapter.get_kinematics()
        previous_timestamp = float(frame.timestamp)
        rgb_history.append(frame.rgb)
        depth_history.append(frame.depth_m)
        valid_history.append(frame.valid_mask)

        for step in range(args.max_steps):
            speed = float(np.linalg.norm(state.linear_velocity))
            route_state = route.observe(state.position, speed_mps=speed)
            local_goal_body = Rotation.from_quat(state.orientation_xyzw).inv().apply(
                route_state.local_goal_nwu - state.position
            )
            features = public_state_features(
                local_goal_body=local_goal_body[None],
                linear_velocity_world=state.linear_velocity[None],
                angular_velocity_world=state.angular_velocity[None],
                orientation_xyzw=state.orientation_xyzw[None],
                previous_action_physical=previous_action[None],
            )[0]
            raw_action, latency_ms = inference(
                model, rgb_history, depth_history, valid_history, features, device
            )
            raw_action = np.clip(raw_action, -ACTION_LIMITS, ACTION_LIMITS)
            policy_action, terminal_blend = terminal_capture_action(
                raw_action,
                local_goal_body,
                route_state.remaining_m,
            )
            command_world = Rotation.from_quat(state.orientation_xyzw).apply(policy_action[:3])
            factual = adapter.execute_velocity_command(
                command_world, float(policy_action[3]), args.action_duration_s
            )
            frame = adapter.get_depth()
            state = adapter.get_kinematics()
            collision = adapter.get_collision_info()
            dt = float(frame.timestamp) - previous_timestamp
            if dt <= 0:
                raise RuntimeError(f"timestamp regression at step {step}: {dt}")
            if bool(factual["stale_action"]):
                raise RuntimeError(f"stale_action at step {step}")
            if bool(collision["has_collided"]):
                raise RuntimeError(f"collision at step {step}")
            goal_distance = float(np.linalg.norm(state.position - contract["goal"]))
            entry = {
                "step": step,
                "sim_time": float(frame.timestamp),
                "dt_s": dt,
                "goal_distance_m": goal_distance,
                "route_progress_m": route_state.progress_m,
                "remaining_route_m": route_state.remaining_m,
                "cross_track_error_m": route_state.cross_track_error_m,
                "position_world_enu": np.asarray(state.position, dtype=float).tolist(),
                "local_goal_body_flu": np.asarray(local_goal_body, dtype=float).tolist(),
                "learned_action_body_flu": raw_action.tolist(),
                "policy_action_body_flu": policy_action.tolist(),
                "terminal_capture_blend": terminal_blend,
                "executed_action_body_flu": np.asarray(
                    factual["action_executed_body_flu"], dtype=float
                ).tolist(),
                "inference_latency_ms": latency_ms,
                "safety_intervened": bool(factual["safety_intervened"]),
            }
            result["steps"].append(entry)
            if route_state.cross_track_error_m > args.maximum_cross_track_m:
                raise RuntimeError(
                    f"cross-track gate exceeded at step {step}: {route_state.cross_track_error_m:.3f} m"
                )
            previous_action = policy_action.astype(np.float32)
            previous_timestamp = float(frame.timestamp)
            rgb_history.append(frame.rgb)
            depth_history.append(frame.depth_m)
            valid_history.append(frame.valid_mask)
            if goal_distance <= args.goal_tolerance_m:
                result["status"] = "PASS"
                result["success"] = True
                break
        else:
            result["status"] = "FAIL"
            result["success"] = False
            result["error"] = "maximum steps reached"
    except BaseException as error:
        result["status"] = "FAIL"
        result["success"] = False
        result["error"] = repr(error)
        raise
    finally:
        try:
            adapter.close()
            asyncio.run(stop_simulator())
        except Exception as error:
            result["cleanup_error"] = repr(error)
            result["status"] = "FAIL"
        steps = result["steps"]
        result["steps_completed"] = len(steps)
        result["minimum_goal_distance_m"] = min(
            (item["goal_distance_m"] for item in steps), default=float("inf")
        )
        result["maximum_cross_track_error_m"] = max(
            (item["cross_track_error_m"] for item in steps), default=0.0
        )
        result["safety_interventions"] = sum(item["safety_intervened"] for item in steps)
        result["stale_actions"] = 0
        if steps:
            latency = np.asarray([item["inference_latency_ms"] for item in steps])
            result["inference_latency_ms"] = {
                "mean": float(latency.mean()),
                "p95": float(np.percentile(latency, 95)),
                "maximum": float(latency.max()),
            }
        try:
            result["health_after"] = health()
        except Exception as error:
            result["health_after_error"] = repr(error)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "success",
                        "steps_completed",
                        "minimum_goal_distance_m",
                        "maximum_cross_track_error_m",
                        "safety_interventions",
                        "stale_actions",
                        "inference_latency_ms",
                        "error",
                    )
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
