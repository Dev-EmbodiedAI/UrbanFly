#!/usr/bin/env python
"""在 Swarm 程序化数字孪生中运行 Agent→World Model→执行→反馈闭环。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.digital_twin import (  # noqa: E402
    GoalConditionedWorldModelPolicy,
    SwarmDigitalTwinAdapter,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swarm-root", type=Path, required=True)
    parser.add_argument("--environment", choices=("city", "open", "mountain", "village", "forest"), required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--drones", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    challenge_type = {"city": 1, "open": 2, "mountain": 3, "village": 4, "forest": 6}[args.environment]
    adapter = SwarmDigitalTwinAdapter(
        args.swarm_root,
        challenge_type=challenge_type,
        seed=args.seed,
        num_drones=args.drones,
        gui=args.gui,
    )
    world_model_policy = GoalConditionedWorldModelPolicy()
    started = time.perf_counter()
    report = {
        "schema": "urbanfly-cross-environment-digital-twin-navigation-v1",
        "status": "RUNNING",
        "environment": f"swarm:{args.environment}",
        "seed": args.seed,
        "requested_drones": args.drones,
        "architecture": [
            "Agent exact start/goal mission assignment",
            "goal-conditioned local policy candidates",
            "kinematic/depth/separation World Model reranking",
            "native Swarm PyBullet execution and collision",
            "fresh public-contract observation feedback",
        ],
        "control_contract": "cf_swarm_autopilot submission_zip.v1 [N,128,128,1]+[N,190]->[N,5]",
        "benchmark_eligible": False,
        "benchmark_exclusion_reason": "digital-twin mission mode exposes exact goals to the high-level Agent",
        "steps": 0,
        "agent_observations": 0,
        "world_model_decisions": 0,
        "executions": 0,
        "fresh_feedbacks": 0,
        "trajectory_every_50_steps": [],
    }
    last_info: dict = {}
    try:
        mission, observation = adapter.reset()
        assigned_goals = world_model_policy.reset(mission)
        report["mission"] = {
            "episode_id": mission.episode_id,
            "starts_enu_m": mission.starts_enu_m.tolist(),
            "native_goal_pool_enu_m": mission.goals_enu_m.tolist(),
            "agent_assigned_goals_enu_m": assigned_goals.tolist(),
            "agent_provider": mission.agent_provider,
            "privileged_goal_mode": mission.privileged_goal_mode,
            "metadata": dict(mission.metadata),
        }
        report["agent_observations"] = 1
        resolved = np.zeros(observation.drone_count, dtype=bool)
        minimum_goal_distance = np.linalg.norm(assigned_goals - observation.positions_enu_m, axis=1)
        minimum_predicted_clearance = np.full(observation.drone_count, np.inf)
        minimum_predicted_separation = np.full(observation.drone_count, np.inf)
        for step in range(args.max_steps):
            decision = world_model_policy.act(observation)
            decision.action[resolved] = 0.0
            report["world_model_decisions"] += 1
            previous_sequence = observation.sequence
            previous_timestamp = observation.timestamp_s
            feedback = adapter.step(decision.action)
            report["executions"] += 1
            observation = feedback.observation
            if observation.sequence != previous_sequence + 1:
                raise RuntimeError("Swarm feedback sequence is not the next control step")
            if observation.timestamp_s <= previous_timestamp:
                raise RuntimeError("Swarm feedback timestamp is stale")
            report["fresh_feedbacks"] += 1
            report["agent_observations"] += 1
            report["steps"] = step + 1
            resolved |= np.asarray(feedback.per_drone_success, dtype=bool)
            resolved |= np.asarray(feedback.per_drone_collision, dtype=bool)
            resolved |= np.asarray([reason != "NONE" for reason in feedback.per_drone_failure_reason], dtype=bool)
            distances = np.linalg.norm(assigned_goals - observation.positions_enu_m, axis=1)
            minimum_goal_distance = np.minimum(minimum_goal_distance, distances)
            minimum_predicted_clearance = np.minimum(minimum_predicted_clearance, decision.predicted_clearance_m)
            minimum_predicted_separation = np.minimum(minimum_predicted_separation, decision.predicted_minimum_separation_m)
            last_info = dict(feedback.raw_info)
            if step % 50 == 0:
                report["trajectory_every_50_steps"].append({
                    "step": step,
                    "timestamp_s": observation.timestamp_s,
                    "positions_enu_m": observation.positions_enu_m.tolist(),
                    "attitude_rpy_rad": observation.state[:, 3:6].tolist(),
                    "goal_distance_m": distances.tolist(),
                    "executed_action": decision.action.tolist(),
                    "selected_candidate": decision.selected_candidate.tolist(),
                    "predicted_clearance_m": decision.predicted_clearance_m.tolist(),
                    "predicted_minimum_separation_m": decision.predicted_minimum_separation_m.tolist(),
                })
            if feedback.terminated or feedback.truncated:
                break
        score = adapter.score(last_info) if last_info else None
        success = tuple(bool(item) for item in last_info.get("per_drone_success", [False] * args.drones))
        collisions = tuple(bool(item) for item in last_info.get("per_drone_collision", [False] * args.drones))
        reasons = tuple(str(item) for item in last_info.get("per_drone_failure_reason", ["NONE"] * args.drones))
        report.update({
            "success": all(success),
            "per_drone_success": list(success),
            "per_drone_collision": list(collisions),
            "per_drone_failure_reason": list(reasons),
            "minimum_goal_distance_m": minimum_goal_distance.tolist(),
            "minimum_predicted_clearance_m": minimum_predicted_clearance.tolist(),
            "minimum_predicted_separation_m": minimum_predicted_separation.tolist(),
            "native_score": score,
        })
        causal = (
            report["world_model_decisions"]
            == report["executions"]
            == report["fresh_feedbacks"]
            and report["agent_observations"] == report["fresh_feedbacks"] + 1
        )
        report["causal_chain_complete"] = causal
        report["status"] = "PASS" if report["success"] and not any(collisions) and causal else "FAIL"
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        raise
    finally:
        adapter.close()
        report["wall_time_s"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in (
        "status", "environment", "steps", "success", "per_drone_success",
        "per_drone_collision", "per_drone_failure_reason", "native_score",
        "causal_chain_complete", "wall_time_s", "error",
    )}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
