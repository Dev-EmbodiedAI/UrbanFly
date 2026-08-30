from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.envs import UAVEnvConfig, UAVWorldModelEnv
from uav_wm_navigation.simulators import build_simulator
from uav_wm_navigation.utils.config import load_yaml


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark canonical single-UAV simulator throughput.")
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/benchmark_single_uav.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset-samples", type=int)
    parser.add_argument("--sensor-samples", type=int)
    parser.add_argument("--policy-steps", type=int)
    parser.add_argument("--connection-timeout-s", type=float)
    args = parser.parse_args()
    sim_config = load_yaml(args.sim_config)
    benchmark = load_yaml(args.benchmark_config)
    if args.connection_timeout_s is not None:
        sim_config["connection_timeout_s"] = float(args.connection_timeout_s)
        sim_config["sensor_timeout_s"] = float(args.connection_timeout_s)
    backend = str(sim_config.get("backend", "unknown"))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "backend": backend,
        "simulator_config": str(args.sim_config.resolve()),
        "status": "failed",
        "canonical_world_frame": "NWU",
        "canonical_body_frame": "FLU",
        "action": "body velocity [forward,left,up,yaw_rate]",
        "headless_unattended": True if backend == "mock" else "backend-dependent",
    }
    simulator = None
    env = None
    try:
        simulator = build_simulator(sim_config)
        env_values = benchmark["env"]
        env = UAVWorldModelEnv(
            simulator,
            config=UAVEnvConfig(
                physics_hz=int(env_values["physics_hz"]), sensor_hz=int(env_values["sensor_hz"]),
                policy_hz=int(env_values["policy_hz"]), success_radius_m=float(env_values["success_radius_m"]),
                success_dwell_s=float(env_values["success_dwell_s"]), max_episode_s=float(env_values["max_episode_s"]),
                depth_clip_m=float(sim_config.get("depth_max_m", 120.0)),
            ),
            seed=int(sim_config.get("seed", 0)),
        )
        goal = np.asarray(benchmark.get("goal_nwu", [1000.0, 0.0, 2.0]), dtype=np.float32)
        resets = int(args.reset_samples or benchmark.get("reset_samples", 20))
        sensor_samples = int(args.sensor_samples or benchmark.get("sensor_samples", 200))
        policy_steps = int(args.policy_steps or benchmark.get("policy_steps", 500))
        warmup = int(benchmark.get("warmup_steps", 10))
        reset_latency_ms = []
        for index in range(resets):
            started = time.perf_counter()
            env.reset(
                goal_nwu=goal, episode_id=f"throughput-reset-{index:04d}",
                scenario=str(benchmark.get("scenario", sim_config.get("scenario", "OpenSpace"))),
                difficulty=str(benchmark.get("difficulty", "easy")),
            )
            reset_latency_ms.append((time.perf_counter() - started) * 1000.0)
        sensor_latency_ms = []
        for _ in range(sensor_samples):
            started = time.perf_counter()
            simulator.get_sensor_frame()
            sensor_latency_ms.append((time.perf_counter() - started) * 1000.0)
        observation, info = env.reset(
            goal_nwu=goal, episode_id="throughput-policy", scenario=str(benchmark.get("scenario", "OpenSpace")),
            difficulty=str(benchmark.get("difficulty", "easy")),
        )
        action = np.asarray(benchmark.get("action_normalized", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        for _ in range(warmup):
            observation, _, terminated, truncated, info = env.step(action, shield_enabled=False)
            if terminated or truncated:
                observation, info = env.reset(goal_nwu=goal, scenario="OpenSpace")
        policy_latency_ms = []
        sim_time_before = float(simulator.get_sim_time())
        policy_started = time.perf_counter()
        for _ in range(policy_steps):
            step_started = time.perf_counter()
            observation, _, terminated, truncated, info = env.step(action, shield_enabled=False)
            policy_latency_ms.append((time.perf_counter() - step_started) * 1000.0)
            if terminated or truncated:
                observation, info = env.reset(goal_nwu=goal, scenario="OpenSpace")
        policy_wall_s = time.perf_counter() - policy_started
        sim_advanced_s = float(simulator.get_sim_time()) - sim_time_before
        physics_steps = policy_steps * int(env_values["physics_hz"]) // int(env_values["policy_hz"])
        sensor_frames = policy_steps * int(env_values["sensor_hz"]) // int(env_values["policy_hz"])
        result.update({
            "status": "complete",
            "reset_samples": resets,
            "reset_latency_ms": distribution(reset_latency_ms),
            "sensor_samples": sensor_samples,
            "sensor_latency_ms": distribution(sensor_latency_ms),
            "sensor_fps_standalone": float(1000.0 / np.mean(sensor_latency_ms)),
            "policy_steps": policy_steps,
            "policy_steps_per_sec": float(policy_steps / policy_wall_s),
            "physics_steps_per_sec": float(physics_steps / policy_wall_s),
            "sensor_frames_per_sec_in_policy_loop": float(sensor_frames / policy_wall_s),
            "policy_step_latency_ms": distribution(policy_latency_ms),
            "simulated_seconds_per_wall_second": float(sim_advanced_s / policy_wall_s),
            "websocket_policy_roundtrip_ms": distribution(policy_latency_ms) if backend == "urbanfly_websocket" else None,
            "frequency_contract": {
                "physics_hz": int(env_values["physics_hz"]), "sensor_hz": int(env_values["sensor_hz"]),
                "policy_hz": int(env_values["policy_hz"]),
            },
        })
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        elif simulator is not None:
            simulator.close()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
