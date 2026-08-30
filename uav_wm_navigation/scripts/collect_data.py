from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.control import SafetyFilter, TrajectoryExecutor
from uav_wm_navigation.data import collect_episode, create_grouped_splits, validate_episode
from uav_wm_navigation.planners import MockCandidatePlanner, YOPOAdapter
from uav_wm_navigation.simulators import build_simulator
from uav_wm_navigation.utils.config import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect resumable HDF5 v2 episodes with formal-stage safeguards.")
    parser.add_argument("--config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/data_collection.yaml")
    parser.add_argument("--sim-config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/simulator_mock.yaml")
    parser.add_argument("--planner-config", type=Path)
    parser.add_argument("--route-manifest", type=Path)
    parser.add_argument("--stage", choices=["calibration", "pilot", "train", "validation", "test", "ood"], default="pilot")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--episodes", "--num-episodes", dest="episodes", type=int)
    parser.add_argument("--seed", type=int, help="Override the deterministic collection seed.")
    parser.add_argument("--max-steps", type=int, help="Maximum high-level decisions per episode.")
    parser.add_argument(
        "--collection-mode", choices=("expert", "perturbed_expert", "safe_exploration"), default="expert"
    )
    parser.add_argument("--randomize-start-goal", action="store_true")
    parser.add_argument("--perturbation-std", type=float, nargs=4, metavar=("FWD", "LEFT", "UP", "YAW"))
    parser.add_argument("--perturbation-bound", type=float, nargs=4, metavar=("FWD", "LEFT", "UP", "YAW"))
    args = parser.parse_args()
    config, sim_config = load_yaml(args.config), load_yaml(args.sim_config)
    formal = bool(config.get("require_real_yopo", False))
    if formal and args.planner_config is None: raise ValueError("formal collection requires --planner-config with real YOPO weights")
    if args.stage in {"train", "validation", "test"} and str(sim_config.get("map")) != "Town10HD":
        raise ValueError("formal train/validation/test collection is restricted to Town10HD")
    if args.stage == "ood" and str(sim_config.get("map")) != "Town05":
        raise ValueError("OOD collection is restricted to Town05")
    planner = YOPOAdapter(load_yaml(args.planner_config)) if args.planner_config else MockCandidatePlanner()
    routes = json.loads(args.route_manifest.read_text(encoding="utf-8"))["routes"] if args.route_manifest else []
    requested_split = {"train": "train", "validation": "validation", "test": "test"}.get(args.stage)
    if requested_split and routes:
        routes = [route for route in routes if route.get("split") == requested_split]
        if not routes:
            raise ValueError(f"route manifest has no routes for split {requested_split}")
    elif args.stage in {"pilot", "calibration"} and routes:
        # Pilot/calibration are tuning data and must not consume held-out zones.
        routes = [route for route in routes if route.get("split") in {None, "train"}]
    base_seed = int(args.seed if args.seed is not None else config.get("seed", 7))
    requested = int(args.episodes or config.get("stages", {}).get(args.stage, config.get("episodes", 8)))
    if requested <= 0:
        raise ValueError("num episodes must be positive")
    rng = np.random.default_rng(base_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True); paths = []
    failures = []
    split_name = {"validation": "validation", "test": "test", "train": "train"}.get(args.stage)
    for index in range(requested):
        episode_id = f"{sim_config.get('map', 'Mock')}_{args.stage}_{index:05d}"
        final_path = args.output_dir / f"{episode_id}.h5"
        if final_path.exists(): validate_episode(final_path); paths.append(final_path); continue
        route = routes[index % len(routes)] if routes else {}
        scenario = route.get("scenario", config["scenarios"][index % len(config["scenarios"])])
        difficulty = route.get("difficulty", config.get("difficulties", ["easy"])[
            (index // len(config["scenarios"])) % len(config.get("difficulties", ["easy"]))
        ])
        seed = int(route.get("seed", base_seed + index))
        simulator_config = {**sim_config, "seed": seed, "scenario": scenario}
        if route.get("initial_yaw_nwu_deg") is not None:
            simulator_config["initial_yaw_nwu_deg"] = float(route["initial_yaw_nwu_deg"])
        if route.get("scripted_crossing") is not None:
            simulator_config["scripted_crossing"] = route["scripted_crossing"]
        if route.get("scripted_obstacles"):
            simulator_config["scripted_obstacles"] = route["scripted_obstacles"]
        simulator = build_simulator(simulator_config)
        safety_config = dict(config.get("safety", {}))
        route_start = route.get("start_nwu", simulator_config.get("flight_start_nwu"))
        if args.randomize_start_goal and not route:
            randomization = config.get("randomization", {})
            start_low = np.asarray(randomization.get("start_min_nwu", [0.0, -1.0, 2.0]), dtype=np.float64)
            start_high = np.asarray(randomization.get("start_max_nwu", [1.0, 1.0, 2.0]), dtype=np.float64)
            goal_low = np.asarray(randomization.get("goal_min_nwu", [8.0, -2.0, 2.0]), dtype=np.float64)
            goal_high = np.asarray(randomization.get("goal_max_nwu", [12.0, 2.0, 2.0]), dtype=np.float64)
            route_start = rng.uniform(start_low, start_high)
            goal = rng.uniform(goal_low, goal_high)
            if np.linalg.norm(goal - route_start) < float(randomization.get("minimum_goal_distance_m", 5.0)):
                goal[0] = goal_high[0]
        else:
            goal = np.asarray(route.get("goal_nwu", sim_config.get("goal_nwu", [12.0, 0.0, 2.0])), dtype=np.float64)
        if route_start is not None:
            safety_config["target_altitude_m"] = float(route_start[2])
            safety_config["min_altitude_m"] = float(simulator_config.get("minimum_altitude_nwu", route_start[2] - 1.0))
            safety_config["max_altitude_m"] = float(simulator_config.get("maximum_altitude_nwu", route_start[2] + 8.0))
            safety_config["acceleration_reference"] = str(simulator_config.get("acceleration_reference", "commanded"))
        default_std = {
            "expert": (0.0, 0.0, 0.0, 0.0),
            "perturbed_expert": (0.6, 0.6, 0.25, 0.12),
            "safe_exploration": (0.25, 0.25, 0.12, 0.06),
        }[args.collection_mode]
        default_bound = {
            "expert": (0.0, 0.0, 0.0, 0.0),
            "perturbed_expert": (1.5, 1.5, 0.7, 0.3),
            "safe_exploration": (0.6, 0.6, 0.3, 0.15),
        }[args.collection_mode]
        executor = TrajectoryExecutor(
            simulator, SafetyFilter(safety_config), 1.0 / float(config.get("control_hz", 20)),
            float(simulator_config.get("trajectory_position_kp", 1.2)),
            action_noise_std=tuple(args.perturbation_std or default_std),
            action_noise_bound=tuple(args.perturbation_bound or default_bound),
            seed=seed,
        )
        metadata = {
            "map": sim_config.get("map", "Mock"), "stage": args.stage, "split": split_name,
            "spatial_zone": route.get("spatial_zone", args.stage), "route_id": route.get("route_id", episode_id),
            "corridor_id": route.get("corridor_id", route.get("route_id", episode_id)),
            "scenario_script": f"{scenario}:{difficulty}:{seed}", "route_nwu": route.get("waypoints_nwu", []),
            "start_nwu": None if route_start is None else np.asarray(route_start).tolist(),
            "scene_id": f"{sim_config.get('map', 'Mock')}:{scenario}", "scene_seed": seed,
            "collection_mode": args.collection_mode,
        }
        try:
            path = collect_episode(
                simulator, planner, executor, args.output_dir, episode_id, goal,
                steps=int(args.max_steps or config.get("steps_per_episode", round(float(config.get("episode_duration_s", 40)) * float(config.get("planner_hz", 5))))),
                future_horizon=int(config.get("future_horizon", 5)), scenario=scenario, seed=seed,
                difficulty=difficulty, metadata=metadata,
                required_candidate_count=int(config.get("required_candidate_count", 15)) if formal else None,
                planning_period_s=1.0 / float(config.get("planner_hz", 5)),
                collection_mode=args.collection_mode,
            )
            print(validate_episode(path)); paths.append(path)
        except Exception as error:
            failures.append({"episode_id": episode_id, "error": f"{type(error).__name__}: {error}"})
            print(json.dumps(failures[-1]), flush=True)
    all_formal = sorted(args.output_dir.glob("Town10HD_*.h5"))
    if all_formal: create_grouped_splits(all_formal, args.output_dir / "splits.json", base_seed)
    elif paths: create_grouped_splits(paths, args.output_dir / "splits.json", base_seed)
    collisions, near_obstacle, lengths, successes, synchronization_errors = [], [], [], [], []
    for path in paths:
        with h5py.File(path, "r") as handle:
            collisions.extend(handle["labels/collision"][:].astype(bool).tolist())
            near_obstacle.extend((handle["labels/minimum_clearance"][:] < float(config.get("near_obstacle_m", 3.0))).tolist())
            lengths.append(int(handle["timestamp"].shape[0]))
            synchronization_errors.extend(np.abs(handle["timestamps/sensor"][:] - handle["timestamps/state"][:]).tolist())
        metadata_path = path.with_suffix(".metadata.json")
        episode_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        successes.append(bool(episode_metadata.get("success", False)))
    summary = {
        "requested_episodes": requested,
        "completed_episodes": len(paths),
        "failed_episodes": failures,
        "automatic_reset_count": max(len(paths) - 1, 0),
        "collection_mode": args.collection_mode,
        "teacher": planner.__class__.__name__,
        "collision_positive_ratio": float(np.mean(collisions)) if collisions else math.nan,
        "near_obstacle_transition_ratio": float(np.mean(near_obstacle)) if near_obstacle else math.nan,
        "success_rate": float(np.mean(successes)) if successes else math.nan,
        "episode_length": {
            "min": min(lengths, default=0), "mean": float(np.mean(lengths)) if lengths else math.nan,
            "median": float(np.median(lengths)) if lengths else math.nan,
            "p95": float(np.percentile(lengths, 95)) if lengths else math.nan,
            "max": max(lengths, default=0),
        },
        "sensor_state_alignment_ms": {
            "mean": float(np.mean(synchronization_errors) * 1000) if synchronization_errors else math.nan,
            "p95": float(np.percentile(synchronization_errors, 95) * 1000) if synchronization_errors else math.nan,
            "max": float(np.max(synchronization_errors) * 1000) if synchronization_errors else math.nan,
        },
        "hdf5_readback_valid": len(paths) == requested and not failures,
        "action_t_to_state_t_plus_1_valid": len(paths) == requested and not failures,
    }
    (args.output_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if len(paths) == requested else 1


if __name__ == "__main__":
    raise SystemExit(main())
