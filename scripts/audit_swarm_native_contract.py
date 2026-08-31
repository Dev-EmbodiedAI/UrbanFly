#!/usr/bin/env python3
"""审计 Swarm 原生多机 contract、环境矩阵、评分与碰撞链路。

该脚本不复制或修改 Swarm 源码。它从 ``--swarm-root`` 导入上游仓库，
使用上游任务生成器、PyBullet 环境、policy validator 和评分函数运行审计。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ENVIRONMENTS = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    6: "forest",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swarm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--upstream-commit", default="unknown")
    return parser.parse_args()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _make_task(task_gen: Any, challenge_type: int, n_drones: int, seed: int) -> Any:
    # screening_task exposes the drone-count override; reuse the exact upstream
    # type resolver so Mountain/Village retain their seed-dependent ranges.
    params = task_gen._resolve_params(seed, challenge_type)
    return task_gen.screening_task(
        sim_dt=0.02,
        seed=seed,
        challenge_type=challenge_type,
        distance_range=(float(params["r_min"]), float(params["r_max"])),
        family_id="cf_swarm_autopilot",
        moving_platform=False,
        n_drones=n_drones,
    )


def _teammate_presence_counts(state: np.ndarray) -> list[int]:
    teammate = state[:, 141:190].reshape(state.shape[0], 7, 7)
    return [int(np.count_nonzero(row[:, 6] > 0.5)) for row in teammate]


def _matrix_row(
    *,
    task_gen: Any,
    make_env_with_initial_obs: Any,
    runtime_family_for_task: Any,
    validate_action_output: Any,
    action_space: dict[str, Any],
    challenge_type: int,
    n_drones: int,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    task = _make_task(task_gen, challenge_type, n_drones, seed)
    env, observation = make_env_with_initial_obs(task, gui=False)
    try:
        depth = np.asarray(observation["depth"])
        state = np.asarray(observation["state"])
        action = np.zeros((n_drones, 5), dtype=np.float32)
        validate_action_output(action, action_space, num_drones=n_drones)

        clue_centres = state[:, 0:3] + state[:, 138:141]
        clue_spread = float(np.max(np.linalg.norm(clue_centres - clue_centres[0], axis=1)))
        presence_counts = _teammate_presence_counts(state)

        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False
        executed_steps = 0
        for _ in range(max(1, steps)):
            observation, reward, terminated, truncated, info = env.step(action)
            executed_steps += 1
            if terminated or truncated:
                break

        family = runtime_family_for_task(task)
        scoring = family.score_swarm(task, info)
        checks = {
            "task_count": int(task.num_drones) == n_drones,
            "depth_shape": tuple(depth.shape) == (n_drones, 128, 128, 1),
            "state_shape": tuple(state.shape) == (n_drones, 190),
            "depth_range": bool(np.all(np.isfinite(depth)) and depth.min() >= 0.0 and depth.max() <= 1.0),
            "state_finite": bool(np.all(np.isfinite(state))),
            "action_contract": True,
            "shared_clue": clue_spread <= 1e-4,
            "teammate_slots": presence_counts == [n_drones - 1] * n_drones,
            "collision_vector": len(info.get("per_drone_collision", ())) == n_drones,
            "score_vector": len(scoring["per_drone_final_score"]) == n_drones,
            "score_range": 0.0 <= float(scoring["final_score"]) <= 1.0,
            "score_is_mean": math.isclose(
                float(scoring["final_score"]),
                float(np.mean(scoring["per_drone_final_score"])),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ),
        }
        return {
            "environment": ENVIRONMENTS[challenge_type],
            "challenge_type": challenge_type,
            "n_drones": n_drones,
            "seed": seed,
            "steps": executed_steps,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "depth_shape": list(depth.shape),
            "state_shape": list(state.shape),
            "shared_clue_spread_m": clue_spread,
            "teammate_presence_counts": presence_counts,
            "per_drone_collision": [bool(value) for value in info["per_drone_collision"]],
            "final_score": float(scoring["final_score"]),
            "checks": checks,
            "pass": all(checks.values()),
        }
    finally:
        env.close()


def _collision_probe(
    *,
    task_gen: Any,
    make_env_with_initial_obs: Any,
    runtime_family_for_task: Any,
    seed: int,
) -> dict[str, Any]:
    import pybullet as bullet

    task = _make_task(task_gen, challenge_type=2, n_drones=2, seed=seed)
    env, _ = make_env_with_initial_obs(task, gui=False)
    try:
        client = env.getPyBulletClient()
        first_position, first_orientation = bullet.getBasePositionAndOrientation(
            env.DRONE_IDS[0], physicsClientId=client
        )
        bullet.resetBasePositionAndOrientation(
            env.DRONE_IDS[1], first_position, first_orientation, physicsClientId=client
        )
        bullet.resetBaseVelocity(
            env.DRONE_IDS[0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=client
        )
        bullet.resetBaseVelocity(
            env.DRONE_IDS[1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=client
        )
        contact_count = 0
        for _ in range(5):
            bullet.stepSimulation(physicsClientId=client)
            contacts = bullet.getContactPoints(
                bodyA=env.DRONE_IDS[0], bodyB=env.DRONE_IDS[1], physicsClientId=client
            )
            contact_count = max(contact_count, len(contacts))
            env._check_collision(0)
            env._check_collision(1)
            if bool(np.any(env._d_collision)):
                break
        _, _, _, _, info = env.step(np.zeros((2, 5), dtype=np.float32))
        scoring = runtime_family_for_task(task).score_swarm(task, info)
        collisions = [bool(value) for value in info["per_drone_collision"]]
        reasons = [str(value) for value in info["per_drone_failure_reason"]]
        collision_detected = any(collisions)
        collision_reached_score = all(
            score <= 0.0100001
            for score, collided in zip(scoring["per_drone_final_score"], collisions)
            if collided
        )
        return {
            "environment": "open",
            "n_drones": 2,
            "seed": seed,
            "method": "将第 2 架无人机重置到第 1 架的同一位姿后执行一个原生仿真步",
            "pybullet_contact_count": contact_count,
            "per_drone_collision": collisions,
            "per_drone_failure_reason": reasons,
            "per_drone_final_score": [float(value) for value in scoring["per_drone_final_score"]],
            "final_score": float(scoring["final_score"]),
            "collision_detected": collision_detected,
            "collision_reached_score": collision_reached_score,
            "pass": collision_detected and collision_reached_score,
        }
    finally:
        env.close()


def main() -> None:
    args = _arguments()
    swarm_root = args.swarm_root.resolve()
    if not (swarm_root / "swarm").is_dir():
        raise FileNotFoundError(f"不是有效的 Swarm 源码目录: {swarm_root}")
    sys.path.insert(0, str(swarm_root))

    from swarm.challenge_families import runtime_family_for_task
    from swarm.domain_model import get_policy_interface_contract
    from swarm.policy_interface import validate_action_output
    from swarm.utils.env_factory import make_env_with_initial_obs
    from swarm.validator import task_gen

    contract = get_policy_interface_contract("cf_swarm_autopilot", "submission_zip.v1")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for challenge_type in ENVIRONMENTS:
        for n_drones in range(2, 9):
            seed = args.seed + challenge_type * 100 + n_drones
            rows.append(
                _matrix_row(
                    task_gen=task_gen,
                    make_env_with_initial_obs=make_env_with_initial_obs,
                    runtime_family_for_task=runtime_family_for_task,
                    validate_action_output=validate_action_output,
                    action_space=contract["action_space"],
                    challenge_type=challenge_type,
                    n_drones=n_drones,
                    seed=seed,
                    steps=args.steps,
                )
            )

    collision = _collision_probe(
        task_gen=task_gen,
        make_env_with_initial_obs=make_env_with_initial_obs,
        runtime_family_for_task=runtime_family_for_task,
        seed=args.seed + 9000,
    )
    failed_rows = [
        {"environment": row["environment"], "n_drones": row["n_drones"], "checks": row["checks"]}
        for row in rows
        if not row["pass"]
    ]
    bullet_version = _version("swarm-bullet3")
    limitations = [
        "本报告直接运行原生 Python/PyBullet 链路，不包含 Docker/Cap'n Proto RPC 隔离层。",
        "零动作短步矩阵用于验证环境、contract、shared clue、碰撞字段和评分接线，不代表策略成功率。",
    ]
    if bullet_version != "2.0.0.3":
        limitations.append(
            f"当前 Windows 运行使用 swarm-bullet3={bullet_version}；上游锁定版本 2.0.0.3 "
            "未发布 Windows 包，最终复现必须在 Linux/Docker 上复核。"
        )
    report = {
        "schema": "urbanfly-swarm-native-contract-audit-v1",
        "family_id": "cf_swarm_autopilot",
        "upstream_root": str(swarm_root),
        "upstream_commit": args.upstream_commit,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "swarm-bullet3": bullet_version,
            "swarm-drone-gym": _version("swarm-drone-gym"),
            "gymnasium": _version("gymnasium"),
            "matches_upstream_locked_bullet": bullet_version == "2.0.0.3",
        },
        "contract": {
            "depth_shape": ["N", 128, 128, 1],
            "state_shape": ["N", 190],
            "action_shape": ["N", 5],
            "drone_count_range": [2, 8],
        },
        "matrix": {
            "environments": list(ENVIRONMENTS.values()),
            "drone_counts": list(range(2, 9)),
            "cases": len(rows),
            "passed": sum(bool(row["pass"]) for row in rows),
            "failed": len(failed_rows),
            "failed_rows": failed_rows,
        },
        "collision_probe": collision,
        "rows": rows,
        "elapsed_sec": time.perf_counter() - started,
        "pass": not failed_rows and collision["pass"],
        "limitations": limitations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("schema", "matrix", "collision_probe", "elapsed_sec", "pass")}, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
