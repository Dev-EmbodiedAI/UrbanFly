from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation

import _bootstrap  # noqa: F401
from uav_wm_navigation.control import (
    RiskReranker, RouteManager, SafetyFilter, TrajectoryExecutor, rank_route_consistent_candidates,
)
from uav_wm_navigation.evaluation import (
    navigation_error,
    polyline_length,
    success_weighted_path_length,
)
from uav_wm_navigation.planners import PlanningContext, YOPOAdapter
from uav_wm_navigation.simulators import (
    MockSimulator,
    UrbanFlyWebSocketAdapter,
)
from uav_wm_navigation.utils.config import load_yaml
from uav_wm_navigation.utils.runlog import create_run_dir, write_manifest
from uav_wm_navigation.types import RiskPrediction
from uav_wm_navigation.world_models import (
    TDMPC2CandidateAssistant,
    build_world_model,
    mc_dropout_predict,
)


def load_world_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_world_model(config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, config, checkpoint, device


def ttc_predictions(candidates, sensor, state):
    height, width = sensor.depth_m.shape
    rotation = Rotation.from_quat(state.orientation_xyzw).as_matrix()
    predictions = []
    for candidate in candidates:
        delta_body = rotation.T @ (candidate.positions[-1] - state.position)
        yaw = float(np.arctan2(delta_body[1], max(delta_body[0], 1e-6)))
        column = int(np.clip((0.5 - yaw / np.pi) * width, 0, width - 1))
        band = sensor.depth_m[height // 3 : 2 * height // 3, max(0, column - 4) : min(width, column + 5)]
        valid = band[np.isfinite(band) & (band > 0)]
        clearance = float(np.percentile(valid, 10)) if valid.size else 0.0
        forward_speed = max(float(np.max(np.linalg.norm(candidate.velocities, axis=1))), 0.1)
        ttc = clearance / forward_speed
        risk = float(np.exp(-ttc / 1.5))
        progress = float(np.linalg.norm(state.position - candidate.positions[-1]))
        predictions.append(RiskPrediction(risk, clearance, progress, risk, 0.0))
    return predictions


def oracle_predictions(candidates, simulator, goal):
    predictions = []
    obstacles = getattr(simulator, "obstacles", [])
    for candidate in candidates:
        clearance = float("inf")
        collision = False
        for obstacle in obstacles:
            distances = np.linalg.norm(candidate.positions - obstacle.center[None], axis=1) - obstacle.radius - 0.25
            clearance = min(clearance, float(distances.min()))
            collision |= bool((distances <= 0).any())
        if not np.isfinite(clearance):
            clearance = 20.0
        progress = float(np.linalg.norm(goal - candidate.positions[0]) - np.linalg.norm(goal - candidate.positions[-1]))
        predictions.append(RiskPrediction(float(collision), max(clearance, 0.0), progress, float(collision), 0.0))
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real-weight YOPO baseline with all network candidates exposed.")
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--planner-config", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/evaluation.yaml")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scenario")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--map")
    parser.add_argument("--goal-nwu", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--method",
        choices=[
            "yopo",
            "yopo_ttc",
            "yopo_gru",
            "yopo_transformer",
            "yopo_dreamerv3",
            "yopo_jepa",
            "yopo_tdmpc2",
            "yopo_occflow",
            "oracle",
        ],
        default="yopo",
    )
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=_bootstrap.PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    sim_config, planner_config, evaluation = map(load_yaml, [args.sim_config, args.planner_config, args.evaluation_config])
    if args.seed is not None: sim_config["seed"] = args.seed
    if args.scenario is not None: sim_config["scenario"] = args.scenario
    if args.map is not None: sim_config["map"] = args.map
    if args.goal_nwu is not None:
        sim_config["goal_nwu"] = list(map(float, args.goal_nwu))
        start = sim_config.get("flight_start_nwu", sim_config.get("initial_position_nwu"))
        if start is None:
            raise ValueError("--goal-nwu requires flight_start_nwu or initial_position_nwu in the simulator config")
        sim_config["route_nwu"] = [list(map(float, start)), list(map(float, args.goal_nwu))]
    backend = sim_config["backend"]
    if backend == "mock":
        simulator = MockSimulator(seed=int(sim_config.get("seed", 0)), scenario=str(sim_config.get("scenario", "StaticObstacle")))
    elif backend == "urbanfly_websocket":
        simulator = UrbanFlyWebSocketAdapter(sim_config)
        simulator.policy_family = args.method
    else:
        raise ValueError(f"unsupported simulator backend {backend!r}")
    planner = YOPOAdapter(planner_config)
    safety_config = dict(evaluation["safety"])
    safety_config.update(sim_config.get("safety_overrides", {}))
    safety = SafetyFilter(safety_config)
    if sim_config.get("acceleration_reference") is not None:
        safety.acceleration_reference = str(sim_config["acceleration_reference"])
    if sim_config.get("flight_start_nwu") is not None:
        flight_start = np.asarray(sim_config["flight_start_nwu"], dtype=np.float64)
        safety.target_altitude = float(sim_config.get("target_altitude_nwu", flight_start[2]))
    if sim_config.get("minimum_altitude_nwu") is not None:
        safety.min_altitude = float(sim_config["minimum_altitude_nwu"])
    if sim_config.get("maximum_altitude_nwu") is not None:
        safety.max_altitude = float(sim_config["maximum_altitude_nwu"])
    if sim_config.get("bounds_min_nwu") is not None:
        safety.bounds_min = np.asarray(sim_config["bounds_min_nwu"], dtype=np.float64)
    if sim_config.get("bounds_max_nwu") is not None:
        safety.bounds_max = np.asarray(sim_config["bounds_max_nwu"], dtype=np.float64)
    executor = TrajectoryExecutor(
        simulator, safety, float(sim_config.get("control_dt", 0.1)),
        float(sim_config.get("trajectory_position_kp", 1.2)),
        yaw_kp=sim_config.get("yaw_kp"),
        yaw_deadband_degrees=float(sim_config.get("yaw_deadband_degrees", 0.0)),
        yaw_rate_smoothing_alpha=float(sim_config.get("yaw_rate_smoothing_alpha", 1.0)),
        velocity_smoothing_alpha=float(sim_config.get("velocity_smoothing_alpha", 1.0)),
        route_lateral_velocity_scale=float(sim_config.get("route_lateral_velocity_scale", 1.0)),
    )
    run_dir = create_run_dir(args.output_root, f"yopo_{backend}")
    write_manifest(run_dir, {"simulator": sim_config, "planner": planner_config, "evaluation": evaluation})
    goal = np.asarray(sim_config.get("goal_nwu", [12.0, 0.0, 2.0]), dtype=np.float64)
    reranker = RiskReranker(evaluation["rerank_weights"])
    world_model = world_model_config = checkpoint = device = None
    tdmpc2_assistant = None
    if args.method in {"yopo_gru", "yopo_transformer", "yopo_dreamerv3", "yopo_jepa", "yopo_occflow"}:
        if args.world_model_checkpoint is None:
            raise ValueError(f"{args.method} requires --world-model-checkpoint")
        world_model, world_model_config, checkpoint, device = load_world_model(args.world_model_checkpoint)
        if world_model_config["model"] not in args.method:
            raise ValueError(f"checkpoint model {world_model_config['model']} does not match method {args.method}")
        reranker = RiskReranker(evaluation["rerank_weights"], checkpoint.get("normalization"))
    elif args.method == "yopo_tdmpc2":
        if args.world_model_checkpoint is None:
            raise ValueError("yopo_tdmpc2 requires --world-model-checkpoint")
        tdmpc2_assistant = TDMPC2CandidateAssistant(
            args.world_model_checkpoint,
            horizon_steps=int(evaluation.get("tdmpc2_horizon_steps", 15)),
            discount=float(evaluation.get("tdmpc2_discount", 0.97)),
            risk_weight=float(evaluation.get("tdmpc2_risk_weight", 8.0)),
        )
        tdmpc2_assistant.reset(
            f"{sim_config.get('map', 'MockTown')}-{sim_config.get('scenario', 'scenario')}-seed{sim_config.get('seed', 0)}"
        )
    state_history, depth_history = [], []
    route_manager = None
    decisions, trajectory, visual_frames = [], [], []
    collision_any = False
    success = False
    termination_reason = "max_steps"
    previous_trajectory_id = None
    episode_started = time.perf_counter()
    continuous_recording = None
    recording_active = False
    episode_start_position = np.zeros(3, dtype=np.float64)
    final_position = np.zeros(3, dtype=np.float64)
    try:
        simulator.connect()
        simulator.reset()
        if sim_config.get("initial_position_nwu") is not None:
            simulator.set_initial_pose(np.asarray(sim_config["initial_position_nwu"], dtype=np.float64))
        simulator.configure_scenario(
            str(sim_config.get("scenario", "StreetCanyon")), args.difficulty, int(sim_config.get("seed", 0))
        )
        simulator.set_goal(goal)
        simulator.takeoff()
        if sim_config.get("flight_start_nwu") is not None:
            simulator.set_initial_pose(np.asarray(sim_config["flight_start_nwu"], dtype=np.float64))
            simulator.execute_velocity_command(np.zeros(3), 0.0, 0.2)
            if hasattr(simulator, "stabilize_at_altitude"):
                simulator.stabilize_at_altitude(float(safety.target_altitude), 1.5)
        episode_start_position = simulator.get_kinematics().position.astype(np.float64)
        final_position = episode_start_position.copy()
        executor.reset()
        recording_config = sim_config.get("continuous_recording", {})
        if bool(recording_config.get("enabled", False)) and hasattr(simulator, "start_continuous_recording"):
            simulator.start_continuous_recording(
                run_dir / "chase_continuous_raw.mp4",
                fps=float(recording_config.get("fps", 30.0)),
                metadata_path=run_dir / "chase_continuous_raw.json",
            )
            recording_active = True
        episode_started = time.perf_counter()
        best_route_progress = 0.0
        stagnant_steps = 0
        for step in range(args.steps):
            simulator.step_scenario(time.perf_counter() - episode_started)
            sensor = simulator.get_depth()
            # UrbanFly's browser bridge publishes an RGB-D/state packet.  Read
            # the sensor first so its adapter returns the exactly paired state;
            # the other backends remain semantically unchanged.
            state = simulator.get_kinematics()
            if route_manager is None:
                route_points = np.asarray(sim_config.get("route_nwu", [state.position.tolist(), goal.tolist()]), dtype=np.float64)
                route_manager = RouteManager(route_points)
            route_manager.update(state.position)
            if route_manager.progress_m > best_route_progress + float(sim_config.get("stagnation_progress_m", 0.30)):
                best_route_progress = route_manager.progress_m
                stagnant_steps = 0
            else:
                stagnant_steps += 1
            local_goal = route_manager.local_goal()
            started = time.perf_counter()
            candidates = planner.plan(PlanningContext(sensor, state, local_goal))
            latency = (time.perf_counter() - started) * 1000.0
            state_vector = np.concatenate([state.position, state.orientation_xyzw, state.linear_velocity, state.angular_velocity]).astype(np.float32)
            state_history.append(state_vector); depth_history.append(sensor.depth_m.astype(np.float32))
            history_size = int(world_model_config["history"]) if world_model_config else 1
            state_history[:] = state_history[-history_size:]; depth_history[:] = depth_history[-history_size:]
            while len(state_history) < history_size:
                state_history.insert(0, state_history[0].copy()); depth_history.insert(0, depth_history[0].copy())
            rotation = Rotation.from_quat(state.orientation_xyzw).as_matrix()
            local_candidates = np.stack([(item.positions - state.position[None]) @ rotation for item in candidates])
            decision = None
            predictions = None
            control_predictions = None
            fallback_applied = False
            model_latency = 0.0
            route_guard_metrics = []
            route_guard_reason = "disabled"
            tdmpc2_predicted_return = None
            if args.method == "yopo":
                selected = int(np.argmin([item.yopo_cost for item in candidates]))
                predicted_risk = 0.0
            else:
                model_started = time.perf_counter()
                if args.method == "yopo_ttc":
                    predictions = ttc_predictions(candidates, sensor, state)
                elif args.method == "oracle":
                    predictions = oracle_predictions(candidates, simulator, goal)
                elif args.method == "yopo_tdmpc2":
                    predictions, tdmpc2_predicted_return, model_latency = (
                        tdmpc2_assistant.predict(
                            candidates,
                            sensor,
                            state,
                            local_goal,
                        )
                    )
                else:
                    trajectories = []
                    trajectory_steps = int(world_model_config.get("trajectory_steps", 16))
                    for candidate in candidates:
                        positions = (candidate.positions - state.position[None]) @ rotation
                        velocities = candidate.velocities @ rotation
                        accelerations = candidate.accelerations @ rotation
                        source = np.linspace(0.0, 1.0, len(positions)); target = np.linspace(0.0, 1.0, trajectory_steps)
                        combined = np.concatenate([positions, velocities, accelerations], axis=-1)
                        combined = np.stack([np.interp(target, source, combined[:, column]) for column in range(9)], axis=-1)
                        trajectories.append(combined)
                    trajectories = np.stack(trajectories)
                    depth_max_m = float(world_model_config.get("depth_max_m", 20.0))
                    depth_batch = np.stack(depth_history)
                    depth_batch = np.nan_to_num(depth_batch, nan=depth_max_m, posinf=depth_max_m, neginf=0.0)
                    depth_batch = np.clip(depth_batch, 0.0, depth_max_m) / depth_max_m
                    if depth_batch.shape[-2:] != (96, 160):
                        depth_batch = np.stack([cv2.resize(frame, (160, 96), interpolation=cv2.INTER_NEAREST) for frame in depth_batch])
                    inputs = (
                        torch.from_numpy(depth_batch.astype(np.float32)[None, :, None]).to(device),
                        torch.from_numpy(np.stack(state_history)[None]).to(device),
                        torch.from_numpy(((local_goal - state.position) @ rotation).astype(np.float32)[None]).to(device),
                        torch.from_numpy(trajectories.astype(np.float32)[None]).to(device),
                    )
                    output = mc_dropout_predict(
                        world_model, inputs, int(evaluation.get("mc_dropout_samples", world_model_config.get("mc_dropout_samples", 5))),
                        checkpoint.get("calibration", {}),
                    )
                    predictions = [RiskPrediction(
                        float(output["collision_probability"][0, i]), float(output["minimum_clearance"][0, i]),
                        float(output["goal_progress"][0, i]), float(output["failure_probability"][0, i]),
                        float(output["uncertainty"][0, i]),
                    ) for i in range(len(candidates))]
                if args.method != "yopo_tdmpc2":
                    model_latency = (time.perf_counter() - model_started) * 1000.0
                decision = reranker.rank(candidates, predictions, model_latency, float(evaluation["inference_timeout_ms"]), float(evaluation["max_risk"]))
                selected = decision.selected_index
                predicted_risk = 1.0 if selected < 0 else predictions[selected].collision_probability
                control_predictions = predictions
            display_scores = np.asarray(
                decision.total_scores if decision is not None else [item.yopo_cost for item in candidates],
                dtype=np.float64,
            )
            if decision is not None and selected < 0:
                fallback_predictions = ttc_predictions(candidates, sensor, state)
                fallback_yopo = np.asarray([item.yopo_cost for item in candidates], dtype=np.float64)
                low, high = float(np.min(fallback_yopo)), float(np.max(fallback_yopo))
                fallback_yopo = (fallback_yopo - low) / max(high - low, 1e-8)
                fallback_risk = np.asarray([item.collision_probability for item in fallback_predictions])
                fallback_clearance = np.asarray([item.minimum_clearance for item in fallback_predictions])
                c_low, c_high = float(np.min(fallback_clearance)), float(np.max(fallback_clearance))
                fallback_clearance = (fallback_clearance - c_low) / max(c_high - c_low, 1e-8)
                display_scores = fallback_yopo + 4.0 * fallback_risk - fallback_clearance
                control_predictions = fallback_predictions
                fallback_applied = True
                decision.reason = f"{decision.reason}+fallback_yopo_ttc"
                decision.used_fallback = True
            route_guard_config = sim_config.get("route_consistency", {})
            if bool(route_guard_config.get("enabled", False)):
                guarded_selected, guarded_ranking, guarded_scores, route_guard_metrics, route_guard_reason = (
                    rank_route_consistent_candidates(
                        candidates, route_manager, display_scores, route_guard_config, previous_trajectory_id,
                    )
                )
                selected = guarded_selected
                display_scores = np.asarray(guarded_scores, dtype=np.float64)
                if decision is not None:
                    decision.selected_index = selected
                    decision.reranked = guarded_ranking
                    decision.total_scores = guarded_scores
                    decision.reason = f"{decision.reason}+{route_guard_reason}"
                    decision.used_fallback = bool(decision.used_fallback or fallback_applied or selected < 0)
            predicted_risk = (
                0.0 if control_predictions is None or selected < 0
                else control_predictions[selected].collision_probability
            )
            execution = [] if selected < 0 else executor.execute_prefix(
                candidates[selected], float(sim_config.get("planning_prefix_s", 0.2)), predicted_risk, sensor,
                heading_target_nwu=local_goal,
            )
            if selected >= 0:
                planner.commit_selected(candidates[selected], float(sim_config.get("planning_prefix_s", 0.2)))
                previous_trajectory_id = candidates[selected].trajectory_id
            if selected < 0:
                simulator.execute_velocity_command(np.zeros(3), 0.0, float(sim_config.get("planning_prefix_s", 0.2)))
            current = simulator.get_kinematics()
            final_position = current.position.astype(np.float64)
            collision_report = simulator.get_collision_info()
            collision_now = bool(collision_report.get("has_collided", False))
            collision_any |= collision_now
            distance = float(np.linalg.norm(goal - current.position))
            decisions.append({
                "step": step, "candidate_count": len(candidates), "selected_index": selected,
                "costs": [item.yopo_cost for item in candidates], "latency_ms": latency,
                "model_latency_ms": model_latency, "total_planning_latency_ms": latency + model_latency,
                "execution": execution, "goal_distance_m": distance,
                "route_completion": route_manager.completion, "local_goal_nwu": local_goal.tolist(),
                "method": args.method,
                "tdmpc2_predicted_return": (
                    None
                    if tdmpc2_predicted_return is None
                    else tdmpc2_predicted_return.tolist()
                ),
                "elapsed_s": float(time.perf_counter() - episode_started),
                "route_guard": {
                    "reason": route_guard_reason,
                    "eligible_count": int(sum(item.get("eligible", 0.0) > 0.5 for item in route_guard_metrics)),
                    "selected_metrics": None if selected < 0 or not route_guard_metrics else route_guard_metrics[selected],
                    "candidates": route_guard_metrics,
                },
                "rerank": None if decision is None else {
                    "selected_index": decision.selected_index, "original_ranking": decision.original_ranking,
                    "reranked": decision.reranked, "scores": decision.total_scores,
                    "reason": decision.reason, "used_fallback": decision.used_fallback,
                },
            })
            trajectory.append(current.position.tolist())
            visual_rgb = simulator.get_visualization_rgb()
            visual_frames.append({
                "rgb": (
                    sensor.rgb.astype(np.uint8) if sensor.rgb is not None
                    else np.zeros((*sensor.depth_m.shape, 3), dtype=np.uint8)
                ),
                "third_person_rgb": (
                    visual_rgb.astype(np.uint8) if visual_rgb is not None
                    else np.zeros((*sensor.depth_m.shape, 3), dtype=np.uint8)
                ),
                "depth": sensor.depth_m.astype(np.float32),
                "candidates": local_candidates.astype(np.float32),
                "candidates_world": np.stack([item.positions for item in candidates]).astype(np.float32),
                "camera_intrinsics": (
                    sensor.camera_intrinsics.astype(np.float32)
                    if sensor.camera_intrinsics is not None
                    else np.eye(3, dtype=np.float32)
                ),
                "camera_pose_nwu": (
                    sensor.camera_pose_nwu.astype(np.float32)
                    if sensor.camera_pose_nwu is not None
                    else np.eye(4, dtype=np.float32)
                ),
                "planning_position_nwu": state.position.astype(np.float32),
                "selected": np.int16(selected),
                "yopo_cost": np.asarray([item.yopo_cost for item in candidates], dtype=np.float32),
                "collision_probability": np.asarray(
                    [item.collision_probability for item in predictions] if predictions is not None
                    else np.full(len(candidates), np.nan), dtype=np.float32,
                ),
                "total_score": np.asarray(
                    display_scores,
                    dtype=np.float32,
                ),
                "position_nwu": current.position.astype(np.float32),
                "velocity_nwu": current.linear_velocity.astype(np.float32),
                "goal_nwu": goal.astype(np.float32),
                "elapsed_s": np.float64(time.perf_counter() - episode_started),
                "method": np.asarray(args.method),
            })
            goal_tolerance = float(sim_config.get("goal_tolerance_m", evaluation.get("goal_tolerance_m", 1.0)))
            if route_manager.reached(current.position, goal_tolerance):
                success = True
                termination_reason = "goal_reached"
                break
            if collision_now:
                termination_reason = "collision"
                break
            if stagnant_steps >= int(sim_config.get("stagnation_timeout_steps", 30)):
                termination_reason = "stagnation_timeout"
                break
    finally:
        try:
            if recording_active:
                continuous_recording = simulator.stop_continuous_recording()
                recording_active = False
        finally:
            try:
                if getattr(simulator, "client", None) is not None:
                    simulator.land()
            finally:
                simulator.close()
    visual_path = run_dir / "yopo_visualization.npz"
    if visual_frames:
        np.savez_compressed(visual_path, **{
            name: np.stack([frame[name] for frame in visual_frames]) for name in visual_frames[0]
        })
    latency_values = np.asarray([item["total_planning_latency_ms"] for item in decisions], dtype=np.float64)
    steady_latency = latency_values[1:] if latency_values.size > 1 else latency_values
    if visual_frames:
        positions_array = np.stack([frame["position_nwu"] for frame in visual_frames])
        velocities_array = np.stack([frame["velocity_nwu"] for frame in visual_frames])
        depth_array = np.stack([frame["depth"] for frame in visual_frames])
        depth_valid = np.isfinite(depth_array) & (depth_array > 0.0)
        depth_max = float(sim_config.get("depth_max_m", 20.0))
        path_length = float(np.linalg.norm(np.diff(positions_array, axis=0), axis=1).sum()) if len(positions_array) > 1 else 0.0
        speeds = np.linalg.norm(velocities_array, axis=1)
        target_altitude = float(safety.target_altitude) if safety.target_altitude is not None else float(positions_array[0, 2])
        selected_values = np.asarray([frame["selected"] for frame in visual_frames])
        elapsed_values = np.asarray([frame["elapsed_s"] for frame in visual_frames], dtype=np.float64)
        elapsed_duration = float(elapsed_values[-1] - elapsed_values[0]) if len(elapsed_values) > 1 else 0.0
        position_speeds = (
            np.linalg.norm(np.diff(positions_array, axis=0), axis=1) / np.maximum(np.diff(elapsed_values), 1e-3)
            if len(positions_array) > 1 else np.zeros(1)
        )
        route_audit = RouteManager(route_manager.waypoints)
        raw_route_progress, route_lateral = [], []
        audit_reference = 0.0
        for position in positions_array:
            projected_progress, cross_track = route_audit.project_nearest(position, audit_reference)
            raw_route_progress.append(projected_progress); route_lateral.append(cross_track)
            audit_reference = max(audit_reference, projected_progress)
        route_progress_delta = np.diff(np.asarray(raw_route_progress, dtype=np.float64))
        regressions = np.clip(-route_progress_delta, 0.0, None)
        route_advance = max(float(raw_route_progress[-1] - raw_route_progress[0]), 0.0)
        quality = {
            "frame_count": int(len(visual_frames)),
            "path_length_m": path_length,
            "net_displacement_m": float(np.linalg.norm(positions_array[-1] - positions_array[0])),
            "mean_speed_mps": path_length / max(elapsed_duration, 1e-3),
            "max_speed_mps": float(position_speeds.max()),
            "reported_mean_speed_mps": float(speeds.mean()),
            "reported_max_speed_mps": float(speeds.max()),
            "recording_duration_s": elapsed_duration,
            "altitude_error_p95_m": float(np.percentile(np.abs(positions_array[:, 2] - target_altitude), 95)),
            "depth_saturated_fraction": float((depth_valid & (depth_array >= depth_max - 0.05)).mean()),
            "depth_near_15m_fraction": float((depth_valid & (depth_array < 15.0)).mean()),
            "selected_candidate_changes": int(np.count_nonzero(np.diff(selected_values))),
            "third_person_available": bool(any(np.any(frame["third_person_rgb"]) for frame in visual_frames)),
            "route_backtracking_step_fraction": float(np.mean(route_progress_delta < -0.05)) if route_progress_delta.size else 0.0,
            "route_total_regression_m": float(regressions.sum()),
            "route_maximum_single_regression_m": float(regressions.max()) if regressions.size else 0.0,
            "route_lateral_error_p95_m": float(np.percentile(route_lateral, 95)),
            "route_path_efficiency": route_advance / max(path_length, 1e-6),
        }
        playback_fps = float(sim_config.get("visualization_fps", 5.0))
        playback_duration = max(len(positions_array) / max(playback_fps, 1e-3), 1e-3)
        playback_speeds = (
            np.linalg.norm(np.diff(positions_array, axis=0), axis=1) * playback_fps
            if len(positions_array) > 1 else np.zeros(1)
        )
        quality["playback_fps"] = playback_fps
        quality["playback_mean_speed_mps"] = path_length / playback_duration
        quality["playback_peak_speed_mps"] = float(playback_speeds.max()) if playback_speeds.size else 0.0
    else:
        quality = {"frame_count": 0}
    quality["route_completion"] = 0.0 if route_manager is None else float(route_manager.completion)
    quality["collision_free"] = not collision_any
    quality["goal_reached"] = success
    if continuous_recording is not None:
        quality["continuous_recording"] = {
            key: value for key, value in continuous_recording.items() if key != "frames"
        }
    gates = sim_config.get("visualization_acceptance", {})
    checks = {
        "minimum_frames": quality.get("frame_count", 0) >= int(gates.get("minimum_frames", 60)),
        "minimum_path_length": quality.get("path_length_m", 0.0) >= float(gates.get("minimum_path_length_m", 10.0)),
        "minimum_net_displacement": quality.get("net_displacement_m", 0.0) >= float(gates.get("minimum_net_displacement_m", 0.0)),
        "minimum_wall_clock_mean_speed": quality.get("mean_speed_mps", 0.0) >= float(gates.get("minimum_wall_clock_mean_speed_mps", 0.0)),
        "minimum_playback_mean_speed": quality.get("playback_mean_speed_mps", 0.0) >= float(gates.get("minimum_playback_mean_speed_mps", 0.0)),
        "minimum_playback_peak_speed": quality.get("playback_peak_speed_mps", 0.0) >= float(gates.get("minimum_playback_peak_speed_mps", 0.0)),
        "maximum_altitude_error": quality.get("altitude_error_p95_m", float("inf")) <= float(gates.get("maximum_altitude_error_p95_m", 0.35)),
        "maximum_depth_saturation": quality.get("depth_saturated_fraction", 1.0) <= float(gates.get("maximum_depth_saturated_fraction", 0.35)),
        "minimum_near_depth": quality.get("depth_near_15m_fraction", 0.0) >= float(gates.get("minimum_near_depth_fraction", 0.30)),
        "third_person_camera": (not bool(gates.get("require_third_person", False))) or bool(quality.get("third_person_available", False)),
        "minimum_route_completion": quality.get("route_completion", 0.0) >= float(gates.get("minimum_route_completion", 0.0)),
        "collision_free": (not bool(gates.get("require_collision_free", False))) or bool(quality.get("collision_free", False)),
        "goal_reached": (not bool(gates.get("require_goal_reached", False))) or bool(quality.get("goal_reached", False)),
        "maximum_backtracking_fraction": quality.get("route_backtracking_step_fraction", 1.0) <= float(gates.get("maximum_backtracking_step_fraction", 1.0)),
        "maximum_total_regression": quality.get("route_total_regression_m", float("inf")) <= float(gates.get("maximum_total_regression_m", float("inf"))),
        "maximum_single_regression": quality.get("route_maximum_single_regression_m", float("inf")) <= float(gates.get("maximum_single_regression_m", float("inf"))),
        "maximum_lateral_error": quality.get("route_lateral_error_p95_m", float("inf")) <= float(gates.get("maximum_lateral_error_p95_m", float("inf"))),
        "minimum_route_efficiency": quality.get("route_path_efficiency", 0.0) >= float(gates.get("minimum_route_path_efficiency", 0.0)),
    }
    acceptance = {"passed": bool(all(checks.values())), "checks": checks, "metrics": quality}
    (run_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    executed_positions = np.asarray(
        [episode_start_position.tolist(), *trajectory],
        dtype=np.float64,
    )
    executed_path_length_m = polyline_length(executed_positions)
    shortest_path_length_m = (
        float(route_manager.total_length_m)
        if route_manager is not None
        else float(np.linalg.norm(goal - episode_start_position))
    )
    navigation_error_m = navigation_error(final_position, goal)
    spl = success_weighted_path_length(
        success,
        executed_path_length_m,
        shortest_path_length_m,
    )
    summary = {
        "success": success, "collision": collision_any, "backend": backend, "method": args.method,
        "sr": float(success),
        "navigation_error_m": navigation_error_m,
        "ne_m": navigation_error_m,
        "path_length_m": executed_path_length_m,
        "shortest_path_length_m": shortest_path_length_m,
        "spl": spl,
        "spl_reference": (
            "configured_route_polyline"
            if route_manager is not None
            else "start_goal_euclidean"
        ),
        "start_position_nwu": episode_start_position.tolist(),
        "final_position_nwu": final_position.tolist(),
        "goal_position_nwu": goal.tolist(),
        "termination_reason": termination_reason,
        "seed": int(sim_config.get("seed", 0)), "scenario": sim_config.get("scenario"),
        "difficulty": args.difficulty, "map": sim_config.get("map", "MockTown"),
        "real_yopo_loaded": True,
        "candidate_count": planner.candidate_count, "steps": len(decisions), "trajectory_nwu": trajectory,
        "route_completion": 0.0 if route_manager is None else route_manager.completion,
        "decisions_path": str(run_dir / "decisions.json"),
        "visualization_data_path": str(visual_path),
        "planning_latency_p50_ms": float(np.percentile(latency_values, 50)) if latency_values.size else 0.0,
        "planning_latency_p95_ms": float(np.percentile(latency_values, 95)) if latency_values.size else 0.0,
        "cold_start_latency_ms": float(latency_values[0]) if latency_values.size else 0.0,
        "steady_state_latency_p50_ms": float(np.percentile(steady_latency, 50)) if steady_latency.size else 0.0,
        "steady_state_latency_p95_ms": float(np.percentile(steady_latency, 95)) if steady_latency.size else 0.0,
        "fallback_rate": float(np.mean([
            bool(item["rerank"] and item["rerank"]["used_fallback"]) for item in decisions
        ])) if decisions else 0.0,
        "rerank_rate": float(np.mean([
            bool(item["rerank"] and item["rerank"]["reranked"]) for item in decisions
        ])) if decisions else 0.0,
        "continuous_recording": None if continuous_recording is None else {
            key: value for key, value in continuous_recording.items() if key != "frames"
        },
        "visualization_acceptance": acceptance,
    }
    (run_dir / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
