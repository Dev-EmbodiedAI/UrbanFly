from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _bootstrap  # noqa: F401
from uav_wm_navigation.control import WAMMPCController
from uav_wm_navigation.envs.urbanfly_world_model_env import UrbanFlyEnvConfig, UrbanFlyWorldModelEnv
from uav_wm_navigation.evaluation import binary_auroc
from uav_wm_navigation.planners import MPPICostWeights, MPPIPlanner
from uav_wm_navigation.simulators import build_simulator
from uav_wm_navigation.types import ActionLimits, EpisodeSpec
from uav_wm_navigation.utils.config import load_yaml
from uav_wm_navigation.world_models import JEPAWorldModelAdapter


def latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_debug_step(output: Path, step: int, decision) -> None:
    plan = decision.plan
    if plan is None or plan.candidate_positions is None:
        return
    path = output / "mpc_debug" / f"step_{step:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        candidate_actions=plan.candidate_action_sequences.cpu().numpy(),
        candidate_positions=plan.candidate_positions.cpu().numpy(),
        candidate_costs=plan.candidate_costs.cpu().numpy(),
        best_actions=plan.action_sequence.cpu().numpy(),
        best_positions=plan.predicted_positions.cpu().numpy(),
        collision_probability=plan.predicted_collision_probability.cpu().numpy(),
    )
    path.with_suffix(".json").write_text(
        json.dumps(_json_ready({
            "total_cost": plan.total_cost,
            "cost_components": plan.cost_components,
            "diagnostics": decision.diagnostics,
        }), indent=2),
        encoding="utf-8",
    )


def save_episode_plot(
    path: Path,
    executed: np.ndarray,
    goal: np.ndarray,
    actors: list,
    last_plan,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    if last_plan is not None and last_plan.candidate_positions is not None:
        candidates = last_plan.candidate_positions.cpu().numpy()
        stride = max(1, len(candidates) // 64)
        for candidate in candidates[::stride]:
            axis.plot(candidate[:, 0], candidate[:, 1], color="#9ecae1", alpha=0.18, linewidth=0.7)
        best = last_plan.predicted_positions.cpu().numpy()
        axis.plot(best[:, 0], best[:, 1], color="#ff7f0e", linewidth=2.0, label="best planned")
    axis.plot(executed[:, 0], executed[:, 1], color="#1f77b4", linewidth=2.2, label="executed")
    axis.scatter(executed[-1, 0], executed[-1, 1], color="#1f77b4", s=35, label="UAV current")
    axis.scatter(goal[0], goal[1], marker="*", color="#2ca02c", s=180, label="goal")
    for index, actor in enumerate(actors):
        radius = float(max(actor.bbox_extent[0], actor.bbox_extent[1]))
        axis.add_patch(plt.Circle(actor.position[:2], radius, color="#d62728", alpha=0.28,
                                  label="obstacle" if index == 0 else None))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("NWU x / m")
    axis.set_ylabel("NWU y / m")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_stack(config: dict, sim_config: dict, checkpoint: Path | None, device: torch.device, seed: int):
    model, provenance = JEPAWorldModelAdapter.from_config(config, checkpoint=checkpoint, map_location=device)
    mppi = config["mppi"]
    cost = MPPICostWeights(**config.get("cost", {}))
    action = config["action"]
    planner = MPPIPlanner(
        horizon=int(mppi["horizon"]),
        dt=float(mppi["dt"]),
        num_samples=int(mppi["num_samples"]),
        num_iterations=int(mppi["num_iterations"]),
        temperature=float(mppi["temperature"]),
        noise_sigma=tuple(mppi["noise_sigma"]),
        action_min=tuple(action["min"]),
        action_max=tuple(action["max"]),
        warm_start=bool(mppi.get("warm_start", True)),
        cost_weights=cost,
        save_debug=bool(mppi.get("save_mpc_debug", False)),
        seed=seed,
        device=device,
    )
    wam = config["wam_mpc"]
    controller = WAMMPCController(
        model,
        planner,
        history=int(wam.get("history", 4)),
        depth_max_m=float(wam.get("depth_max_m", sim_config.get("depth_max_m", 20.0))),
        depth_shape=tuple(wam.get("depth_shape", sim_config.get("depth_shape", [96, 160]))),
        device=device,
    )
    return controller, provenance


def run_episode(
    episode_index: int,
    config: dict,
    sim_config: dict,
    checkpoint: Path | None,
    device: torch.device,
    output: Path,
    debug: bool,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    seed = int(sim_config.get("seed", 0)) + episode_index
    simulator_config = {**sim_config, "seed": seed}
    simulator = build_simulator(simulator_config)
    env_values = config["env"]
    limits = ActionLimits(*[float(value) for value in config["action"]["physical_limits"]])
    env = UrbanFlyWorldModelEnv(
        simulator,
        config=UrbanFlyEnvConfig(
            physics_hz=int(env_values["physics_hz"]),
            sensor_hz=int(env_values["sensor_hz"]),
            policy_hz=int(env_values["policy_hz"]),
            success_radius_m=float(env_values["success_radius_m"]),
            success_dwell_s=float(env_values["success_dwell_s"]),
            max_episode_s=float(env_values["max_episode_s"]),
            depth_clip_m=float(config["wam_mpc"]["depth_max_m"]),
            action_limits=limits,
        ),
        seed=seed,
    )
    controller, provenance = build_stack(config, simulator_config, checkpoint, device, seed)
    lateral = float(config.get("evaluation", {}).get("randomize_lateral_start_m", 0.0))
    start = np.asarray([0.0, ((episode_index % 3) - 1) * lateral, 0.0], dtype=np.float32)
    goal = np.asarray(sim_config.get("goal_nwu", [12.0, 0.0, 2.0]), dtype=np.float32)
    spec = EpisodeSpec(
        episode_id=f"wam-mpc-{episode_index:03d}", route_id=f"mock-route-{episode_index:03d}",
        split="test", tile_ids=(str(sim_config.get("scenario", "StaticObstacle")),),
        scenario=str(sim_config.get("scenario", "StaticObstacle")), seed=seed,
        start_nwu_m=tuple(float(value) for value in start), goal_nwu_m=tuple(float(value) for value in goal),
    )
    episode_dir = output / f"episode_{episode_index:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    observation, info = env.reset(goal_nwu=goal, episode_spec=spec)
    controller.reset()
    positions = [info["state"].position.copy()]
    actions: list[np.ndarray] = []
    speeds: list[float] = []
    jerks: list[float] = []
    predicted_risks: list[float] = []
    collision_labels: list[int] = []
    latencies = {"encoder": [], "rollout": [], "mppi": [], "total": []}
    success = collision = truncated = False
    last_plan = None
    errors: list[str] = []
    max_steps = int(config.get("evaluation", {}).get("max_steps", 300))
    try:
        for step in range(max_steps):
            state = info["state"]
            decision = controller.plan(observation, state, goal)
            if decision.error:
                errors.append(decision.error)
            diagnostics = decision.diagnostics
            for source, target in (
                ("encoder_latency_ms", "encoder"),
                ("rollout_latency_ms", "rollout"),
                ("optimization_latency_ms", "mppi"),
                ("total_planning_latency_ms", "total"),
            ):
                if source in diagnostics:
                    latencies[target].append(float(diagnostics[source]))
            if decision.plan is not None:
                last_plan = decision.plan
                if debug:
                    save_debug_step(episode_dir, step, decision)
            observation, _, terminated, truncated, info = env.step(
                decision.action_normalized,
                predicted_risk=decision.predicted_risk,
                shield_enabled=bool(env_values.get("safety_shield", True)),
            )
            positions.append(info["state"].position.copy())
            actions.append(decision.action_normalized.copy())
            speeds.append(float(np.linalg.norm(info["state"].linear_velocity)))
            jerks.append(float(info["jerk_mps3"]))
            predicted_risks.append(float(decision.predicted_risk))
            collision_labels.append(int(info["collision"]))
            success, collision = bool(info["success"]), bool(info["collision"])
            if terminated or truncated:
                break
            if controller.failure_count >= int(env_values.get("max_consecutive_planner_failures", 3)):
                errors.append("planner failure limit reached; episode terminated after hover fallback")
                break
    finally:
        actors = simulator.get_actor_states()
        env.close()
    trace = np.stack(positions)
    path_length = float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum()) if len(trace) > 1 else 0.0
    action_array = np.stack(actions) if actions else np.zeros((0, 4), dtype=np.float32)
    smoothness = float(np.mean(np.sum(np.diff(action_array, axis=0) ** 2, axis=1))) if len(actions) > 1 else 0.0
    episode = {
        "episode": episode_index,
        "provenance": provenance,
        "steps": len(actions),
        "success": success,
        "collision": collision,
        "truncated": truncated,
        "goal_distance_m": float(np.linalg.norm(goal - trace[-1])),
        "path_length_m": path_length,
        "flight_time_s": float(info.get("sim_time", 0.0)),
        "average_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
        "mean_jerk_mps3": float(np.mean(jerks)) if jerks else 0.0,
        "control_smoothness": smoothness,
        "fallback_count": controller.failure_count,
        "prediction_errors": {
            key: float(np.mean([item[key] for item in controller.prediction_errors]))
            for key in ("latent_prediction_error", "position_prediction_error_m", "velocity_prediction_error_mps")
            if controller.prediction_errors
        },
        "latency_ms": {name: latency_summary(values) for name, values in latencies.items()},
        "maximum_predicted_collision_probability": max(predicted_risks, default=0.0),
        "collision_probe_accuracy": (
            float(np.mean((np.asarray(predicted_risks) >= 0.5) == np.asarray(collision_labels, dtype=bool)))
            if collision_labels else math.nan
        ),
        "collision_probe_auroc": (
            binary_auroc(np.asarray(collision_labels), np.asarray(predicted_risks))
            if len(set(collision_labels)) > 1 else math.nan
        ),
        "collision_probe_probabilities": predicted_risks,
        "collision_labels": collision_labels,
        "errors": errors,
        "executed_trajectory_nwu": trace.tolist(),
    }
    (episode_dir / "metrics.json").write_text(json.dumps(_json_ready(episode), indent=2), encoding="utf-8")
    if bool(config.get("evaluation", {}).get("save_visualization", True)):
        save_episode_plot(episode_dir / "trajectory.png", trace, goal, actors, last_plan)
    return episode, latencies


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate single-UAV JEPA latent world model + vectorized MPPI.")
    parser.add_argument("--config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/wam_mpc_jepa.yaml")
    parser.add_argument("--sim-config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/simulator_mock.yaml")
    parser.add_argument("--world-model", choices=["jepa"], default="jepa")
    parser.add_argument("--planner", choices=["mppi"], default="mppi")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-samples", type=int, help="MPPI sample ablation override (for example 32..1024).")
    parser.add_argument("--horizon", type=int, help="Prediction-horizon ablation override.")
    parser.add_argument("--iterations", type=int, help="MPPI optimization-iteration override.")
    parser.add_argument("--max-steps", type=int, help="Per-episode control-step budget override.")
    parser.add_argument("--output-dir", type=Path, default=_bootstrap.PROJECT_ROOT / "outputs/wam_mpc_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow-untrained", action="store_true",
                        help="Interface/safety smoke test only; never treat this as a learned-model experiment.")
    args = parser.parse_args()
    config, sim_config = load_yaml(args.config), load_yaml(args.sim_config)
    for value, key in (
        (args.num_samples, "num_samples"), (args.horizon, "horizon"), (args.iterations, "num_iterations")
    ):
        if value is not None:
            if value <= 0:
                parser.error(f"--{key.replace('_', '-')} must be positive")
            config["mppi"][key] = value
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("--max-steps must be positive")
        config.setdefault("evaluation", {})["max_steps"] = args.max_steps
    if args.checkpoint is None and not args.allow_untrained:
        parser.error("--checkpoint is required unless --allow-untrained is explicitly set")
    if args.debug:
        config["mppi"]["save_mpc_debug"] = True
    episodes = int(args.episodes or config.get("evaluation", {}).get("episodes", 5))
    if episodes <= 0:
        parser.error("--episodes must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    results = []
    aggregate_latencies = {"encoder": [], "rollout": [], "mppi": [], "total": []}
    reset_failures = []
    for episode_index in range(episodes):
        try:
            result, latencies = run_episode(
                episode_index, config, sim_config, args.checkpoint, device, output, args.debug
            )
            results.append(result)
            for name in aggregate_latencies:
                aggregate_latencies[name].extend(latencies[name])
            print(json.dumps({key: result[key] for key in (
                "episode", "success", "collision", "goal_distance_m", "path_length_m", "fallback_count"
            )}), flush=True)
        except Exception as error:  # reset/backend failure must not kill the evaluation matrix
            reset_failures.append({"episode": episode_index, "error": f"{type(error).__name__}: {error}"})
            print(json.dumps(reset_failures[-1]), flush=True)
    completed = len(results)
    all_collision_labels = np.asarray(
        [label for item in results for label in item["collision_labels"]], dtype=np.int64
    )
    all_collision_probabilities = np.asarray(
        [value for item in results for value in item["collision_probe_probabilities"]], dtype=np.float64
    )
    summary = {
        "status": "smoke_only_untrained" if args.checkpoint is None else "checkpoint_evaluation",
        "world_model": args.world_model,
        "planner": args.planner,
        "requested_episodes": episodes,
        "completed_episodes": completed,
        "reset_failures": reset_failures,
        "success_rate": float(np.mean([item["success"] for item in results])) if results else math.nan,
        "collision_rate": float(np.mean([item["collision"] for item in results])) if results else math.nan,
        "mean_goal_distance_m": float(np.mean([item["goal_distance_m"] for item in results])) if results else math.nan,
        "mean_path_length_m": float(np.mean([item["path_length_m"] for item in results])) if results else math.nan,
        "mean_flight_time_s": float(np.mean([item["flight_time_s"] for item in results])) if results else math.nan,
        "mean_average_speed_mps": float(np.mean([item["average_speed_mps"] for item in results])) if results else math.nan,
        "mean_jerk_mps3": float(np.mean([item["mean_jerk_mps3"] for item in results])) if results else math.nan,
        "mean_control_smoothness": float(np.mean([item["control_smoothness"] for item in results])) if results else math.nan,
        "collision_probe_accuracy": (
            float(np.mean((all_collision_probabilities >= 0.5) == all_collision_labels))
            if len(all_collision_labels) else math.nan
        ),
        "collision_probe_auroc": (
            binary_auroc(all_collision_labels, all_collision_probabilities)
            if len(np.unique(all_collision_labels)) > 1 else math.nan
        ),
        "latency_ms": {name: latency_summary(values) for name, values in aggregate_latencies.items()},
        "gpu_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
        ),
        "wall_time_s": time.time() - started,
        "note": (
            "No checkpoint was supplied: results validate interfaces, vectorization, receding horizon and fallback only."
            if args.checkpoint is None else "Metrics use the supplied checkpoint."
        ),
        "episodes": results,
    }
    (output / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_ready({key: value for key, value in summary.items() if key != "episodes"}), indent=2))
    return 0 if completed == episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
