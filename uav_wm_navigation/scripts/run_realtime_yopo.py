from __future__ import annotations

import argparse
import faulthandler
import json
import os
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.control import RealtimeYOPORunner, SafetyFilter
from uav_wm_navigation.planners import YOPOAdapter
from uav_wm_navigation.simulators import build_simulator
from uav_wm_navigation.utils.config import load_yaml
from uav_wm_navigation.utils.runlog import create_run_dir, write_manifest
from uav_wm_navigation.world_models import CandidateWorldModelRuntime, V3CandidateWorldModelRuntime


def _safety(sim_config: dict, speed_mps: float) -> SafetyFilter:
    config = {
        "max_speed_mps": speed_mps,
        "max_acceleration_mps2": 6.0,
        "max_yaw_rate_rps": 1.5,
        "emergency_depth_m": 0.40,
        "slow_depth_m": 1.20,
        "clearance_percentile": 2.0,
        "depth_roi_fraction": [0.25, 0.82, 0.28, 0.72],
        "collision_debounce_steps": 1,
        "acceleration_reference": "commanded",
    }
    config.update(sim_config.get("safety_overrides", {}))
    start = np.asarray(sim_config["flight_start_nwu"], dtype=float)
    config.update({
        "target_altitude_m": float(sim_config.get("target_altitude_nwu", start[2])),
        "min_altitude_m": float(sim_config.get("minimum_altitude_nwu", start[2] - 0.7)),
        "max_altitude_m": float(sim_config.get("maximum_altitude_nwu", start[2] + 6.0)),
        "bounds_min_nwu": sim_config.get("bounds_min_nwu", [-1e6, -1e6, -1e6]),
        "bounds_max_nwu": sim_config.get("bounds_max_nwu", [1e6, 1e6, 1e6]),
    })
    return SafetyFilter(config)


def run(args: argparse.Namespace) -> tuple[Path, dict]:
    sim_config = load_yaml(args.sim_config)
    if args.seed is not None:
        sim_config["seed"] = int(args.seed)
    planner_config = load_yaml(args.planner_config)
    evaluation = load_yaml(args.evaluation_config) if args.evaluation_config else {}
    speed = float(args.speed if args.speed is not None else planner_config.get("velocity", 4.5))
    planner_config["velocity"] = speed
    planner_config["plan_from_reference"] = False
    sim_config["parallel_rpc_clients"] = True
    sim_config["velocity_command_blocking"] = False
    sim_config["velocity_command_hold_s"] = max(float(sim_config.get("velocity_command_hold_s", 0.20)), 0.20)
    goal = np.asarray(sim_config["goal_nwu"], dtype=np.float64)
    start = np.asarray(sim_config["flight_start_nwu"], dtype=np.float64)
    route_nwu = np.asarray(sim_config.get("route_nwu", [start.tolist(), goal.tolist()]), dtype=np.float64)
    if route_nwu.ndim != 2 or route_nwu.shape[0] < 2 or route_nwu.shape[1] != 3:
        raise ValueError("route_nwu must contain at least two [x,y,z] waypoints")
    if np.linalg.norm(route_nwu[0] - start) > float(sim_config.get("route_start_tolerance_m", 3.0)):
        raise ValueError("flight_start_nwu must lie near the first route waypoint")
    if np.linalg.norm(route_nwu[-1] - goal) > float(sim_config.get("route_goal_tolerance_m", 1e-3)):
        raise ValueError("goal_nwu must equal the final route waypoint")
    method = str(args.method).lower()
    run_dir = create_run_dir(args.output_root, f"realtime_{method}_{speed:g}mps")
    write_manifest(run_dir, {
        "simulator": sim_config, "planner": planner_config, "speed_mps": speed,
        "method": method, "evaluation": evaluation,
        "world_model_checkpoint": None if args.world_model_checkpoint is None else str(args.world_model_checkpoint.resolve()),
    })
    simulator = build_simulator(sim_config)
    planner = YOPOAdapter(planner_config)
    realtime_config = dict(sim_config.get("realtime", {}))
    world_model_runtime = None
    if method != "yopo":
        if args.world_model_checkpoint is None:
            raise ValueError(f"{method} requires --world-model-checkpoint")
        if method in {"tdmpc2_visual", "dreamer_rssm_v3", "vjepa2_1_uav"}:
            world_model_runtime = V3CandidateWorldModelRuntime(
                args.world_model_checkpoint,
                family=method,
                weights=evaluation.get("rerank_weights_v3"),
                timeout_ms=float(evaluation.get("inference_timeout_ms_v3", 150.0)),
                max_risk=float(evaluation.get("max_risk", 0.75)),
                device=args.device,
                horizon_steps=15,
            )
        else:
            world_model_runtime = CandidateWorldModelRuntime(
                args.world_model_checkpoint,
                weights=evaluation.get("rerank_weights", {
                    "yopo": 1.0, "collision": 4.0, "failure": 2.0,
                    "progress": 1.0, "clearance": 1.0, "uncertainty": 1.0,
                }),
                timeout_ms=float(evaluation.get("inference_timeout_ms", 80.0)),
                max_risk=float(evaluation.get("max_risk", 0.75)),
                mc_dropout_samples=args.mc_dropout_samples,
                device=args.device,
                local_goal_lookahead_m=float(evaluation.get("world_model_goal_lookahead_m", 10.0)),
            )
        if world_model_runtime.family != method:
            raise ValueError(f"checkpoint family {world_model_runtime.family} does not match --method {method}")
    runner = RealtimeYOPORunner(
        simulator, planner, _safety(sim_config, speed), goal, realtime_config,
        world_model_runtime=world_model_runtime,
        route_nwu=route_nwu,
    )
    recording_started = False
    recording = None
    termination = "not_started"
    collision = False
    success = False
    monitoring: list[dict] = []
    started = time.perf_counter()
    try:
        simulator.connect()
        simulator.reset()
        simulator.set_initial_pose(start)
        simulator.configure_scenario(
            str(sim_config.get("scenario", "RandomForestField")), str(args.difficulty), int(sim_config.get("seed", 0))
        )
        simulator.set_goal(goal)
        simulator.takeoff()
        simulator.set_initial_pose(start)
        simulator.execute_velocity_command(np.zeros(3), 0.0, 0.1)
        if hasattr(simulator, "stabilize_at_altitude"):
            simulator.stabilize_at_altitude(float(sim_config.get("target_altitude_nwu", start[2])), 2.0)
        baseline_collision = simulator.get_collision_info()
        runner.collision_baseline_timestamp = int(baseline_collision.get("timestamp", 0))
        started = time.perf_counter()
        runner.start()
        ready = False
        ready_deadline = time.perf_counter() + float(realtime_config.get("controller_startup_timeout_s", 8.0))
        while time.perf_counter() < ready_deadline:
            if runner.errors:
                break
            with runner._data_lock:
                ready = bool(runner.plan_records) and any(sample.plan_sequence_id >= 0 for sample in runner.control_records)
            if ready:
                break
            time.sleep(0.02)
        if not ready:
            raise RuntimeError(f"realtime controller did not receive a valid YOPO plan: {runner.errors}")
        baseline_collision = simulator.get_collision_info()
        runner.collision_baseline_timestamp = int(baseline_collision.get("timestamp", 0))
        runner.begin_flight_interval()
        # Start all camera streams at the same accepted-run boundary as
        # telemetry.  Earlier versions included controller warm-up frames,
        # producing a 2--3 s visual/trajectory offset.
        if hasattr(simulator, "start_synchronized_recording"):
            simulator.start_synchronized_recording(run_dir, fps=float(sim_config.get("recording_fps", 30.0)))
            recording_started = True
        started = time.perf_counter()
        best_progress = 0.0
        goal_dwell_started: float | None = None
        regression_started: float | None = None
        while time.perf_counter() - started < float(args.max_duration_s):
            elapsed = time.perf_counter() - started
            simulator.step_scenario(elapsed)
            state = simulator.get_kinematics()
            collision_report = simulator.get_collision_info()
            collision = bool(collision_report.get("has_collided", False)) and (
                "timestamp" not in collision_report
                or int(collision_report.get("timestamp", 0)) > runner.collision_baseline_timestamp
            )
            distance = float(np.linalg.norm(goal - state.position))
            route = runner.observe_route(state)
            progress = float(route.nearest_progress_m)
            best_progress = max(best_progress, progress)
            regression = best_progress - progress
            planner_status = runner.planner_status()
            plan_gap = planner_status["plan_gap_s"]
            monitoring.append({
                "elapsed_s": elapsed, "position_nwu": state.position.tolist(),
                "velocity_nwu": state.linear_velocity.tolist(), "goal_distance_m": distance,
                "local_goal_nwu": route.local_goal_nwu.tolist(),
                "route_progress_m": route.progress_m,
                "route_nearest_progress_m": route.nearest_progress_m,
                "route_completion": route.progress_m / runner.route.total_length_m,
                "cross_track_error_m": route.cross_track_error_m,
                "route_segment_index": route.segment_index,
                "route_remaining_m": route.remaining_m,
                "agl_m": float(state.position[2] - float(realtime_config.get("ground_altitude_nwu", 0.0))),
                "regression_from_best_m": regression, "collision": collision,
                **planner_status,
                "collision_report": collision_report,
            })
            if collision:
                termination = "collision"
                break
            if runner.errors:
                termination = "worker_error"
                break
            maximum_plan_gap = float(realtime_config.get("maximum_plan_gap_s", 0.5))
            if (
                plan_gap is not None
                and elapsed > float(realtime_config.get("planner_watchdog_grace_s", 1.0))
                and float(plan_gap) > maximum_plan_gap
            ):
                termination = "planner_stalled"
                runner._fail(
                    "planner-watchdog",
                    RuntimeError(
                        f"planner produced no plan for {float(plan_gap):.3f}s "
                        f"(stage={planner_status['planner_stage']})"
                    ),
                )
                break
            if distance <= float(sim_config.get("goal_tolerance_m", 3.0)):
                goal_dwell_started = goal_dwell_started or time.perf_counter()
                if time.perf_counter() - goal_dwell_started >= float(sim_config.get("success_dwell_s", 2.0)):
                    termination = "goal_reached_and_stable"
                    success = True
                    break
            else:
                goal_dwell_started = None
            if regression > float(realtime_config.get("maximum_sustained_regression_m", 1.5)):
                regression_started = regression_started or time.perf_counter()
                if time.perf_counter() - regression_started > float(realtime_config.get("regression_timeout_s", 1.0)):
                    termination = "sustained_backtracking"
                    break
            else:
                regression_started = None
            # Collision/goal supervision is not part of the 50 Hz controller.
            # Keeping it at 10 Hz avoids two synchronous RPCs competing with
            # the dedicated actuator worker.
            monitor_hz = max(float(realtime_config.get("monitor_hz", 10.0)), 1.0)
            time.sleep(1.0 / monitor_hz)
        else:
            termination = "timeout"
    except BaseException as exc:
        termination = "exception"
        runner._fail("main", exc)
    finally:
        runner.stop(termination)
        try:
            simulator.execute_velocity_command(np.zeros(3), 0.0, 0.2)
        except Exception:
            pass
        if recording_started:
            try:
                recording = simulator.stop_synchronized_recording()
            except Exception as exc:
                recording = {"error": repr(exc)}
        try:
            simulator.land()
        except Exception:
            pass
        simulator.close()
    runner.save_telemetry(run_dir / "telemetry.h5")
    metrics = runner.metrics()
    monitor_progress = np.asarray([item["route_nearest_progress_m"] for item in monitoring], dtype=np.float64)
    monitor_regression = np.clip(-np.diff(monitor_progress), 0.0, None)
    metrics["estimated_state_regression_m"] = float(metrics.get("estimated_state_regression_m", 0.0))
    metrics["total_regression_m"] = float(monitor_regression.sum()) if monitor_regression.size else 0.0
    metrics["backtracking_fraction"] = float(np.mean(monitor_regression > 0.01)) if monitor_regression.size else 0.0
    recording_summary = recording
    if isinstance(recording, dict):
        recording_summary = {
            stream: ({key: value for key, value in value.items() if key != "frames"} if isinstance(value, dict) else value)
            for stream, value in recording.items()
        }
    navigation_error_m = monitoring[-1]["goal_distance_m"] if monitoring else float(np.linalg.norm(goal - start))
    path_length_m = float(metrics.get("path_length_m", 0.0))
    shortest_path_m = float(runner.route.total_length_m)
    spl = float(success) * shortest_path_m / max(shortest_path_m, path_length_m, 1e-6)
    metrics.update({
        "success": success, "collision": collision, "termination_reason": termination,
        "goal_distance_m": navigation_error_m,
        "navigation_error_m": navigation_error_m,
        "ne_m": navigation_error_m,
        "shortest_path_m": shortest_path_m,
        "spl": spl,
        "route_completion": 1.0 if success else (
            0.0 if not monitoring else float(np.clip(monitoring[-1]["route_completion"], 0.0, 1.0))
        ),
        "route_length_m": runner.route.total_length_m,
        "maximum_cross_track_error_m": max(
            (float(item["cross_track_error_m"]) for item in monitoring), default=0.0
        ),
        "agl_p05_m": float(np.percentile([item["agl_m"] for item in monitoring], 5)) if monitoring else None,
        "agl_p95_m": float(np.percentile([item["agl_m"] for item in monitoring], 95)) if monitoring else None,
        "speed_target_mps": speed, "candidate_count": planner.candidate_count,
        "method": method,
        "checkpoint": str(planner.checkpoint), "recording": recording_summary,
        "world_model_checkpoint": None if world_model_runtime is None else str(world_model_runtime.checkpoint_path),
    })
    acceptance = {
        "candidate_count_15": planner.candidate_count == 15,
        "raw_yopo_only": method == "yopo",
        "world_model_loaded": method == "yopo" or world_model_runtime is not None,
        "collision_free": not collision,
        "goal_reached": success,
        "route_completion_95pct": metrics["route_completion"] >= 0.95,
        "end_to_end_p95_under_180ms": metrics["planner_latency_p95_ms"] is not None and metrics["planner_latency_p95_ms"] < 180.0,
        "sensor_rate_at_least_9hz": metrics["sensor_rate_hz"] >= 9.0,
        "planner_rate_at_least_4_5hz": metrics["planner_rate_hz"] >= 4.5,
        "control_rate_at_least_45hz": metrics["control_rate_hz"] >= 45.0,
        "maximum_plan_gap_under_0_5s": metrics["maximum_plan_gap_s"] <= float(
            realtime_config.get("maximum_plan_gap_s", 0.5)
        ),
        "backtracking_under_0_75m": metrics["total_regression_m"] <= 0.75,
    }
    if method != "yopo":
        # This is an experimental comparison, so reranking is expected.  The
        # pure-YOPO invariant is only an acceptance condition for group A.
        acceptance.pop("raw_yopo_only")
    summary = {"run_dir": str(run_dir), "metrics": metrics, "acceptance": acceptance, "passed": all(acceptance.values())}
    (run_dir / "monitoring.json").write_text(json.dumps(monitoring, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return run_dir, summary


def main() -> int:
    debug_stack_interval = float(os.environ.get("UAV_WM_DEBUG_STACK_INTERVAL_S", "0"))
    if debug_stack_interval > 0:
        faulthandler.dump_traceback_later(debug_stack_interval, repeat=True)
    parser = argparse.ArgumentParser(description="Run asynchronous YOPO or YOPO+world-model UrbanFly navigation.")
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--planner-config", type=Path, required=True)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--method", choices=["yopo", "dreamerv3", "jepa", "tdmpc2_visual", "dreamer_rssm_v3", "vjepa2_1_uav"], default="yopo")
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--evaluation-config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs" / "evaluation_dreamer_jepa.yaml")
    parser.add_argument("--mc-dropout-samples", type=int)
    parser.add_argument("--device")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, help="Override the simulator/scenario seed recorded in the run manifest.")
    parser.add_argument("--max-duration-s", type=float, default=90.0)
    parser.add_argument("--output-root", type=Path, default=_bootstrap.PROJECT_ROOT / "outputs" / "realtime_yopo")
    args = parser.parse_args()
    _, summary = run(args)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
