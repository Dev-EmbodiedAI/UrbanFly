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

from uav_wm_navigation.simulators.urbanfly_sensor_packet import (
    decode_urbanfly_sensor_packet,
)
from uav_wm_navigation.world_models.tdmpc2_continuous import (
    TDMPC2ContinuousPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained TD-MPC2 policy against UrbanFly's live RGB-D bridge."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--drone-id", default="UAV-01")
    parser.add_argument("--device", default=None)
    parser.add_argument("--shield", choices=("on", "off"), default="on")
    parser.add_argument("--max-actions", type=int, default=0)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    policy = TDMPC2ContinuousPolicy(
        checkpoint=args.checkpoint,
        device=args.device,
    )
    previous_action = np.zeros(4, dtype=np.float32)
    episode_id = f"urbanfly-live-{args.drone_id}"
    policy.reset(episode_id)
    action_step = 0
    last_action_time = -np.inf
    last_sim_time = -np.inf
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(args.url, max_msg_size=8 * 1024 * 1024) as ws:
            await ws.send_json(
                {
                    "type": "policy_subscribe",
                    "payload": {
                        "schema": "urbanfly-world-model-v2",
                        "policy_family": "tdmpc2_continuous",
                    },
                }
            )
            async for message in ws:
                if message.type != aiohttp.WSMsgType.BINARY:
                    continue
                decoded = decode_urbanfly_sensor_packet(
                    message.data,
                    episode_id=episode_id,
                    previous_action=previous_action,
                )
                observation = decoded.observation
                if observation.sim_time < last_sim_time:
                    episode_id = f"urbanfly-live-{args.drone_id}-{action_step}"
                    policy.reset(episode_id)
                    observation.episode_id = episode_id
                    last_action_time = -np.inf
                last_sim_time = observation.sim_time
                if observation.sim_time - last_action_time < 0.19:
                    continue
                policy.observe(observation)
                action = policy.act(deterministic=True)
                diagnostics = policy.diagnostics()
                await ws.send_json(
                    {
                        "type": "policy_action",
                        "payload": {
                            "drone_id": args.drone_id,
                            "step_id": action_step,
                            "policy_family": "tdmpc2_continuous",
                            "action_normalized": action.normalized.tolist(),
                            "inference_latency_ms": diagnostics["latency_ms"],
                            "predicted_risk": diagnostics[
                                "maximum_predicted_risk"
                            ],
                            "shield_enabled": args.shield == "on",
                            "timeout_s": 0.45,
                        },
                    }
                )
                print(
                    json.dumps(
                        {
                            "sim_time": observation.sim_time,
                            "step_id": action_step,
                            "action": action.normalized.tolist(),
                            "latency_ms": diagnostics["latency_ms"],
                            "predicted_risk": diagnostics[
                                "maximum_predicted_risk"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                previous_action = action.normalized.copy()
                last_action_time = observation.sim_time
                action_step += 1
                if args.max_actions and action_step >= args.max_actions:
                    return


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
