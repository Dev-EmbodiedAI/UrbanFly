from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_wm_navigation.data.world_model_v2 import (
    WebDatasetShardWriter,
    WorldModelStepRecord,
)
from uav_wm_navigation.simulators.urbanfly_sensor_packet import (
    decode_urbanfly_sensor_packet,
)
from uav_wm_navigation.types import SafetyAudit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect UI-independent synchronized UrbanFly RGB-D shards."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=1000)
    return parser.parse_args()


def geometric_action_from_header(header: dict) -> tuple[np.ndarray, np.ndarray]:
    world_model = header.get("world_model") or {}
    normalized = world_model.get("raw_action_normalized")
    executed_world = np.asarray(
        world_model.get("command_world_mps", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    yaw = np.deg2rad(float(header.get("yaw_degrees", 0.0)))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    executed_body = np.asarray(
        [
            cosine * executed_world[0] + sine * executed_world[2],
            -sine * executed_world[0] + cosine * executed_world[2],
            executed_world[1],
            np.deg2rad(float(world_model.get("yaw_rate_degrees_s", 0.0))),
        ],
        dtype=np.float32,
    )
    if normalized is None:
        raw_body = world_model.get(
            "raw_action_physical_body_flu", executed_body
        )
        raw_body = np.asarray(raw_body, dtype=np.float32)
        if raw_body.shape == (3,):
            raw_body = np.r_[raw_body, 0.0].astype(np.float32)
        normalized = raw_body / np.asarray(
            [6.0, 6.0, 3.0, np.deg2rad(60.0)], dtype=np.float32
        )
    return np.clip(np.asarray(normalized, dtype=np.float32), -1.0, 1.0), executed_body


async def run(args: argparse.Namespace) -> Path:
    writer = WebDatasetShardWriter(
        args.output,
        shard_prefix=args.episode_id,
        max_samples_per_shard=args.shard_size,
    )
    previous_action = np.zeros(4, dtype=np.float32)
    last_goal_distance = None
    last_collision_count = 0
    local_step = 0
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                args.url, max_msg_size=8 * 1024 * 1024
            ) as ws:
                await ws.send_json(
                    {
                        "type": "policy_subscribe",
                        "payload": {
                            "mode": "data_collection",
                            "schema": "urbanfly-world-model-v2",
                        },
                    }
                )
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    decoded = decode_urbanfly_sensor_packet(
                        message.data,
                        episode_id=args.episode_id,
                        previous_action=previous_action,
                    )
                    observation = decoded.observation
                    observation.step_id = local_step
                    header = decoded.header
                    raw_action, executed_action = geometric_action_from_header(
                        header
                    )
                    world_model = header.get("world_model") or {}
                    goal_distance = float(
                        np.linalg.norm(observation.goal_body_flu_m)
                    )
                    progress = (
                        0.0
                        if last_goal_distance is None
                        else last_goal_distance - goal_distance
                    )
                    last_goal_distance = goal_distance
                    minimum_depth = float(
                        observation.depth_m[observation.depth_valid_mask].min()
                    )
                    collision_count = int(
                        world_model.get("actual_collision_count", 0)
                    )
                    collision = (
                        bool(world_model.get("collision", False))
                        or collision_count > last_collision_count
                    )
                    last_collision_count = max(
                        last_collision_count, collision_count
                    )
                    reward = (
                        progress
                        - 40.0 * collision
                        - 0.02
                        - 0.25 * max(0.0, 3.0 - minimum_depth)
                    )
                    reasons = tuple(
                        world_model.get("safety_intervention_reasons", [])
                    )
                    raw_physical = raw_action * np.asarray(
                        [6.0, 6.0, 3.0, np.deg2rad(60.0)],
                        dtype=np.float32,
                    )
                    audit = SafetyAudit(
                        episode_id=args.episode_id,
                        step_id=local_step,
                        sim_time=observation.sim_time,
                        raw_action_normalized=raw_action,
                        raw_action_physical=raw_physical,
                        executed_action_physical=executed_action,
                        intervened=bool(world_model.get("safety_intervened", False)),
                        reasons=reasons,
                        action_delta_l2=float(
                            np.linalg.norm(executed_action - raw_physical)
                        ),
                        minimum_depth_m=minimum_depth,
                        predicted_risk=float(
                            world_model.get("predicted_risk", 0.0)
                        ),
                    )
                    writer.append(
                        WorldModelStepRecord(
                            observation=observation,
                            raw_action=raw_action,
                            executed_action=executed_action,
                            reward=reward,
                            collision=collision,
                            minimum_clearance_m=minimum_depth,
                            zone_type="unknown",
                            dynamics_parameters=header.get("dynamics", {}),
                            safety_audit=audit,
                            privileged_labels={
                                "source": "live_browser_bridge",
                                "world_model_teacher": world_model.get(
                                    "backend", "none"
                                ),
                            },
                        )
                    )
                    previous_action = raw_action
                    local_step += 1
                    if local_step % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "frames": local_step,
                                    "sim_time": observation.sim_time,
                                    "minimum_depth_m": minimum_depth,
                                }
                            ),
                            flush=True,
                        )
                    if local_step >= args.max_frames:
                        break
    finally:
        manifest = writer.close()
    print(json.dumps({"manifest": str(manifest), "frames": local_step}))
    return manifest


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
