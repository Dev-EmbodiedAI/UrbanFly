from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_wm_navigation.data.world_model_v3 import (
    PrivilegedStepLabels, StreamingWebDatasetWriter, WorldModelV3StepRecord,
)
from uav_wm_navigation.risk.cpa import cpa_risk_map
from uav_wm_navigation.simulators.urbanfly_sensor_packet import decode_urbanfly_sensor_packet
from uav_wm_navigation.types import EpisodeSpec, SafetyAudit


LIMITS = np.asarray([6.0, 6.0, 3.0, np.deg2rad(60.0)], dtype=np.float32)
BEHAVIORS = ("geometric_mpc_expert", "perturbed_expert", "active_near_miss", "failure_recovery", "random_exploration")


def parse_vector(value: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect synchronized UrbanFly world-model-v3 RGB-D shards")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--route-id")
    parser.add_argument("--route-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test", "calibration"))
    parser.add_argument("--tile-id", action="append")
    parser.add_argument("--start-nwu", type=parse_vector)
    parser.add_argument("--goal-nwu", type=parse_vector)
    parser.add_argument("--behavior", choices=BEHAVIORS)
    parser.add_argument("--scenario", default="single_uav_world_model")
    parser.add_argument("--zone-type", default="unknown")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--idle-timeout-s", type=float, default=30.0)
    parser.add_argument("--debug-timestamps", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.route_waypoints_nwu = None
    if args.route_manifest:
        if not args.route_id:
            parser.error("--route-id is required with --route-manifest")
        manifest = json.loads(args.route_manifest.read_text(encoding="utf-8"))
        route = next((item for item in manifest.get("routes", []) if item["route_id"] == args.route_id), None)
        if route is None:
            parser.error("route-id is absent from route manifest")
        args.start_nwu = tuple(route["start_nwu_m"]); args.goal_nwu = tuple(route["goal_nwu_m"])
        args.split = args.split or route["split"]; args.tile_id = args.tile_id or list(route["tile_ids"])
        args.zone_type = route.get("zone_type", args.zone_type); args.behavior = args.behavior or route["behavior"]
        args.seed = int(route["seed"]); args.route_waypoints_nwu = route["route_nwu_m"]
        args.episode_id = args.episode_id or f"collect-{args.route_id}-{args.seed}"
    missing = [name for name in ("episode_id", "route_id", "split", "tile_id", "start_nwu", "goal_nwu") if not getattr(args, name)]
    if missing: parser.error(f"missing route fields: {', '.join(missing)}")
    args.behavior = args.behavior or "geometric_mpc_expert"
    args.route_waypoints_nwu = args.route_waypoints_nwu or [list(args.start_nwu), list(args.goal_nwu)]
    return args


def nwu_to_world(value: tuple[float, float, float] | list[float]) -> list[float]:
    x, y, z = map(float, value)
    return [x, z, y]


def world_to_nwu(value: list[float]) -> np.ndarray:
    x, up, z = map(float, value)
    return np.asarray([x, z, up], dtype=np.float32)


def expert_action(
    header: dict, route: np.ndarray, waypoint_index: int, behavior: str,
    step: int, rng: np.random.Generator, depth_m: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, int]:
    position = world_to_nwu(header.get("vehicle_pose", {}).get("position", [0, 0, 0]))
    while waypoint_index < len(route) - 1 and np.linalg.norm(route[waypoint_index] - position) < 10.0:
        waypoint_index += 1
    target = route[waypoint_index]
    delta = target - position
    desired = np.clip(delta * 0.45, [-6, -6, -3], [6, 6, 3])
    horizontal = float(np.linalg.norm(desired[:2]))
    if horizontal > 6.0: desired[:2] *= 6.0 / horizontal
    yaw = np.deg2rad(float(header.get("yaw_degrees", 0.0)))
    forward = np.cos(yaw) * desired[0] + np.sin(yaw) * desired[1]
    left = -np.sin(yaw) * desired[0] + np.cos(yaw) * desired[1]
    target_yaw = np.arctan2(delta[1], delta[0]); yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    action = np.asarray([forward / 6.0, left / 6.0, desired[2] / 3.0, yaw_error / np.deg2rad(60)], dtype=np.float32)
    if behavior == "perturbed_expert":
        action += rng.normal(0, [0.10, 0.14, 0.08, 0.12]).astype(np.float32)
    elif behavior == "active_near_miss":
        midpoint = depth_m.shape[1] // 2
        left_clear = np.nanmedian(np.where(valid[:, :midpoint], depth_m[:, :midpoint], np.nan))
        right_clear = np.nanmedian(np.where(valid[:, midpoint:], depth_m[:, midpoint:], np.nan))
        toward_clutter = 1.0 if np.nan_to_num(left_clear, nan=120) < np.nan_to_num(right_clear, nan=120) else -1.0
        action[1] += 0.28 * toward_clutter
    elif behavior == "failure_recovery" and step < 25:
        action[1] += 0.45 * np.sin(step * 0.45); action[0] *= 0.65
    elif behavior == "random_exploration":
        action = rng.uniform([-0.25, -0.5, -0.25, -0.5], [0.65, 0.5, 0.25, 0.5]).astype(np.float32)
    front = depth_m[:, depth_m.shape[1] // 3: depth_m.shape[1] * 2 // 3]
    front_valid = valid[:, valid.shape[1] // 3: valid.shape[1] * 2 // 3]
    minimum = float(front[front_valid].min()) if front_valid.any() else 120.0
    if minimum < 3.5:
        action[0] = min(action[0], -0.25); action[2] = max(action[2], 0.35)
    if position[2] < 5.0: action[2] = max(action[2], 0.5)
    if position[2] > 115.0: action[2] = min(action[2], -0.5)
    return np.clip(action, -1, 1), waypoint_index


def action_from_header(header: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    world_model = header.get("world_model") or {}
    normalized = np.asarray(world_model.get("raw_action_normalized", [0, 0, 0, 0]), dtype=np.float32)
    executed_world = np.asarray(world_model.get("command_world_mps", [0, 0, 0]), dtype=np.float32)
    yaw = np.deg2rad(float(header.get("yaw_degrees", 0.0)))
    forward, left = np.array([np.cos(yaw), 0, np.sin(yaw)]), np.array([-np.sin(yaw), 0, np.cos(yaw)])
    executed = np.asarray([
        executed_world @ forward, executed_world @ left, executed_world[1],
        np.deg2rad(float(world_model.get("yaw_rate_degrees_s", 0.0))),
    ], dtype=np.float32)
    return np.clip(normalized, -1, 1), executed, world_model


def actor_labels(header: dict, actors: list[dict]) -> tuple[list[dict], np.ndarray]:
    pose = np.asarray(header.get("vehicle_pose", {}).get("position", [0, 0, 0]), dtype=np.float32)
    ego_velocity = np.asarray(header.get("linear_velocity_world_mps", [0, 0, 0]), dtype=np.float32)
    yaw = np.deg2rad(float(header.get("yaw_degrees", 0.0)))
    forward, left = np.array([np.cos(yaw), 0, np.sin(yaw)]), np.array([-np.sin(yaw), 0, np.cos(yaw)])
    relative_positions, relative_velocities = [], []
    normalized = []
    for actor in actors:
        position = np.asarray(actor.get("pos", [0, 0, 0]), dtype=np.float32)
        velocity = np.asarray(actor.get("vel", [0, 0, 0]), dtype=np.float32)
        relative, relative_velocity = position - pose, velocity - ego_velocity
        relative_positions.append([relative @ forward, relative @ left, relative[1]])
        relative_velocities.append([relative_velocity @ forward, relative_velocity @ left, relative_velocity[1]])
        normalized.append({
            "actor_id": int(actor.get("id", -1)), "actor_type": str(actor.get("actor_type", "unknown")),
            "position": position.tolist(), "velocity": velocity.tolist(),
            "bbox_extent": actor.get("bbox_extent", [0.5, 0.5, 0.5]), "scripted": bool(actor.get("scripted", True)),
        })
    risk = cpa_risk_map(np.asarray(relative_positions), np.asarray(relative_velocities)) if normalized else np.zeros(34, dtype=np.float32)
    return normalized, risk


async def run(args: argparse.Namespace) -> Path:
    spec = EpisodeSpec(
        episode_id=args.episode_id, route_id=args.route_id, split=args.split,
        tile_ids=tuple(args.tile_id), scenario=args.scenario, seed=args.seed,
        start_nwu_m=args.start_nwu, goal_nwu_m=args.goal_nwu,
        actor_script_id=f"actors-{args.seed}",
    )
    writer = StreamingWebDatasetWriter(
        args.output, shard_prefix=args.episode_id, max_samples_per_shard=args.shard_size,
        provenance={"collector": Path(__file__).name, "episode_spec": spec.__dict__ if hasattr(spec, "__dict__") else {
            "episode_id": spec.episode_id, "route_id": spec.route_id, "split": spec.split,
            "tile_ids": spec.tile_ids, "scenario": spec.scenario, "seed": spec.seed,
        }}, resume=args.resume,
    )
    previous_action = np.zeros(4, dtype=np.float32)
    last_distance, collision_count, dwell_frames = None, 0, 0
    resume_from_step = writer.next_step_id(args.episode_id)
    local_step = 0
    if args.resume and resume_from_step > 0:
        print(json.dumps({
            "resume_episode": args.episode_id,
            "next_step_id": resume_from_step,
            "last_committed_sim_time": writer.last_sim_time(args.episode_id),
        }), flush=True)
    latest_actors: list[dict] = []
    route = np.asarray(args.route_waypoints_nwu, dtype=np.float32)
    waypoint_index = 1 if len(route) > 1 else 0
    rng = np.random.default_rng(args.seed)
    action_step = 0
    episode_ready = False
    episode_ack_sim_time: float | None = None
    awaiting_first_episode_frame = True
    last_sensor_sim_time = -np.inf
    duplicate_sensor_frames = 0
    last_unique_wall_time = time.monotonic()
    inspected_sensor_frames = 0
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(args.url, max_msg_size=8 * 1024 * 1024) as ws:
                await ws.send_json({"type": "policy_subscribe", "payload": {"mode": "data_collection", "schema": "urbanfly-world-model-v3"}})
                await ws.send_json({"type": "select_scenario", "payload": {"name": args.scenario}})
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        parsed = json.loads(message.data)
                        message_type = parsed.get("type"); payload = parsed.get("payload", {})
                        if message_type == "sim_state": latest_actors = payload.get("actors", [])
                        elif message_type == "scenario_start":
                            await ws.send_json({"type": "policy_episode_config", "payload": {
                                "drone_id": "WM-UAV-01", "start_world_m": nwu_to_world(args.start_nwu),
                                "goal_world_m": nwu_to_world(args.goal_nwu), "yaw_degrees": 0.0,
                                "policy_family": f"collector_{args.behavior}", "shield_enabled": True,
                                "episode_seed": args.seed, "dynamic_actor_density": 1.0,
                            }})
                        elif message_type == "policy_episode_ack":
                            episode_ready = True
                            episode_ack_sim_time = float(payload.get("sim_time", 0.0))
                            awaiting_first_episode_frame = True
                            last_sensor_sim_time = -np.inf
                            last_unique_wall_time = time.monotonic()
                        elif message_type == "error": raise RuntimeError(payload.get("message", "UrbanFly error"))
                        continue
                    if message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    if not episode_ready:
                        continue
                    decoded = decode_urbanfly_sensor_packet(message.data, episode_id=args.episode_id, previous_action=previous_action)
                    observation, header = decoded.observation, decoded.header
                    if args.debug_timestamps and inspected_sensor_frames < 20:
                        print(json.dumps({
                            "sensor_sequence": int(header.get("sequence", -1)),
                            "sensor_sim_time": observation.sim_time,
                            "last_unique_sim_time": last_sensor_sim_time,
                        }), flush=True)
                    inspected_sensor_frames += 1
                    # An RGB-D encode started just before a scenario reset can
                    # complete after policy_episode_ack.  Reject that stale
                    # previous-epoch packet using the acknowledged simulator
                    # time; otherwise its large timestamp would mask every new
                    # episode frame as non-monotonic.
                    if awaiting_first_episode_frame:
                        acknowledged = 0.0 if episode_ack_sim_time is None else episode_ack_sim_time
                        if not acknowledged - 0.5 <= observation.sim_time <= acknowledged + 30.0:
                            duplicate_sensor_frames += 1
                            continue
                        awaiting_first_episode_frame = False
                    # Browser capture can outpace the fixed-step simulator for a
                    # render tick and publish the same synchronized state twice.
                    # A duplicate is neither an independent transition nor valid
                    # training data, so drop it before assigning step ids/actions.
                    if observation.sim_time <= last_sensor_sim_time + 1e-9:
                        duplicate_sensor_frames += 1
                        if time.monotonic() - last_unique_wall_time > args.idle_timeout_s:
                            raise TimeoutError(
                                "RGB-D bridge produced no increasing simulation timestamp "
                                f"for {args.idle_timeout_s:.1f}s"
                            )
                        continue
                    last_sensor_sim_time = observation.sim_time
                    last_unique_wall_time = time.monotonic()
                    observation.step_id = local_step
                    raw, executed, world_model = action_from_header(header)
                    distance = float(np.linalg.norm(observation.goal_body_flu_m))
                    progress = 0.0 if last_distance is None else last_distance - distance
                    last_distance = distance
                    dwell_frames = dwell_frames + 1 if distance <= 3.0 else 0
                    success = dwell_frames >= 20
                    valid_depth = observation.depth_m[observation.depth_valid_mask]
                    clearance = float(valid_depth.min()) if valid_depth.size else 0.0
                    current_count = int(world_model.get("actual_collision_count", 0))
                    collision = bool(world_model.get("collision", False) or current_count > collision_count)
                    collision_count = max(collision_count, current_count)
                    done = collision or success or local_step + 1 >= args.max_frames
                    reward = progress + 50.0 * success - 40.0 * collision - 0.02 - 0.25 * max(0.0, 3.0 - clearance)
                    raw_physical = raw * LIMITS
                    reasons = tuple(world_model.get("safety_intervention_reasons", []))
                    audit = SafetyAudit(
                        episode_id=args.episode_id, step_id=local_step, sim_time=observation.sim_time,
                        raw_action_normalized=raw, raw_action_physical=raw_physical,
                        executed_action_physical=executed, intervened=bool(world_model.get("safety_intervened", False)),
                        reasons=reasons, action_delta_l2=float(np.linalg.norm(executed - raw_physical)),
                        minimum_depth_m=clearance, predicted_risk=float(world_model.get("predicted_risk", 0.0)),
                    )
                    actors, cpa = actor_labels(header, latest_actors)
                    if local_step % 2 == 0:
                        next_action, waypoint_index = expert_action(
                            header, route, waypoint_index, args.behavior,
                            local_step, rng, observation.depth_m, observation.depth_valid_mask,
                        )
                        await ws.send_json({"type": "policy_action", "payload": {
                            "drone_id": "WM-UAV-01", "step_id": action_step,
                            "policy_family": f"collector_{args.behavior}",
                            "action_normalized": next_action.tolist(), "inference_latency_ms": 0.0,
                            "predicted_risk": float(cpa.max()), "shield_enabled": True, "timeout_s": 0.45,
                        }})
                        action_step += 1
                    if local_step < resume_from_step:
                        previous_action = raw
                        local_step += 1
                        continue
                    writer.append(WorldModelV3StepRecord(
                        observation=observation, episode_spec=spec, action_time=observation.sim_time,
                        raw_action_normalized=raw, executed_action_physical=executed, reward=reward,
                        collision=collision, success=success, done=done, minimum_clearance_m=clearance,
                        safety_audit=audit, privileged=PrivilegedStepLabels(
                            tile_id=args.tile_id[0], zone_type=args.zone_type,
                            dynamic_actor_states=actors, cpa_risk_map=cpa,
                            dynamics_parameters=header.get("dynamics", {}),
                            label_provenance={"source": "live_browser_bridge", "actor_truth": "sim_state"},
                        ),
                    ))
                    previous_action, local_step = raw, local_step + 1
                    if local_step % 100 == 0: print(json.dumps({"frames": local_step, "distance_m": distance, "cpa_max": float(cpa.max())}), flush=True)
                    if done: break
    except BaseException:
        writer.abort()
        raise
    manifest = writer.close()
    print(json.dumps({
        "manifest": str(manifest.resolve()), "frames": local_step,
        "duplicate_sensor_frames_dropped": duplicate_sensor_frames,
    }))
    return manifest


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
