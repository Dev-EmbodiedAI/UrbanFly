#!/usr/bin/env python
"""Run one held-out Helsinki latent-world-model flight and record its real compositor."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import aiohttp
import cv2
import h5py
import imageio_ffmpeg
import numpy as np
from scipy.spatial.transform import Rotation
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "uav_wm_navigation" / "src"))

from backend.engine.helsinki_frames import (  # noqa: E402
    backend_world_to_enu,
    enu_to_backend_world,
)
from backend.agents.helsinki_closed_loop import (  # noqa: E402
    AgentStatus,
    HelsinkiAgentWorldModelRuntime,
    SemanticMissionPlan,
    WorldModelActionDecision,
)
from backend.engine.helsinki_navigation import HelsinkiNavigationStack  # noqa: E402
from backend.digital_twin import HelsinkiDigitalTwinAdapter  # noqa: E402
from uav_wm_navigation.control.route_manager import PolylineRoute  # noqa: E402
from urbanfly_vln.navigation_world_model import (  # noqa: E402
    load_navigation_world_model_checkpoint,
)
from urbanfly_vln.observation_policy import (  # noqa: E402
    ACTION_LIMITS,
    load_observation_policy_checkpoint,
)
from urbanfly_vln.observation_policy_data import (  # noqa: E402
    load_qa_episode_records,
    public_state_features,
)
from scripts.run_helsinki_observation_policy import terminal_capture_action  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--policy", type=Path, required=True)
    result.add_argument("--world-model", type=Path, required=True)
    result.add_argument("--qa", type=Path)
    result.add_argument("--episode", type=int, default=97)
    result.add_argument("--scene", type=Path, default=ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km")
    result.add_argument("--custom-start-backend", type=float, nargs=3)
    result.add_argument("--custom-goal-backend", type=float, nargs=3)
    result.add_argument("--via-backend", type=float, nargs=3, action="append", default=[])
    result.add_argument("--semantic-mission-plan", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-steps", type=int, default=1400)
    # The observation policy and latent world model were trained and validated
    # against the 0.5 s Helsinki command horizon.  A 0.1 s horizon materially
    # changes the closed-loop trajectory and can accumulate cross-track error.
    result.add_argument("--action-duration-s", type=float, default=0.5)
    result.add_argument("--goal-tolerance-m", type=float, default=3.0)
    result.add_argument("--maximum-cross-track-m", type=float, default=15.0)
    result.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    result.add_argument("--recording-layout", default="rgbd_world_model")
    result.add_argument("--video-speed", type=float, default=1.0)
    result.add_argument("--video-name", default="helsinki_rooftop_to_ground_world_model.mp4")
    return result


def health() -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=3) as response:
        return json.load(response)


async def stop_simulator() -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect("http://127.0.0.1:8765/ws") as socket:
            await socket.send_json({"type": "control", "payload": {"action": "stop"}})
            await asyncio.sleep(0.2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_contract(record) -> dict[str, object]:
    with h5py.File(record.path, "r") as handle:
        route = np.asarray(handle["episode/global_route_world"][:], dtype=np.float64)
        start = np.asarray(handle["episode/start_world"][:], dtype=np.float64)
        goal = np.asarray(handle["episode/global_goal_world"][:], dtype=np.float64)
        orientation = np.asarray(handle["state/orientation_xyzw"][0], dtype=np.float64)
    return {
        "route": route,
        "start": start,
        "goal": goal,
        "initial_yaw": float(Rotation.from_quat(orientation).as_euler("xyz")[2]),
    }


def policy_latent_and_action(model, rgb_history, depth_history, valid_history, state_features, device):
    while len(rgb_history) < model.config.history_frames:
        rgb_history.appendleft(rgb_history[0].copy())
        depth_history.appendleft(depth_history[0].copy())
        valid_history.appendleft(valid_history[0].copy())
    rgb = torch.from_numpy(np.stack(rgb_history)).unsqueeze(0).to(device)
    depth = torch.from_numpy(np.stack(depth_history)).unsqueeze(0).to(device)
    valid = torch.from_numpy(np.stack(valid_history)).unsqueeze(0).to(device)
    state = torch.from_numpy(state_features).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        latent = model.encode(rgb, depth, valid, state)
        normalized = model.head(latent)
        action = normalized * model.action_std + model.action_mean
        limits = torch.as_tensor(ACTION_LIMITS, device=device)
        action = torch.maximum(torch.minimum(action, limits), -limits)
    return latent, action[0], started


def candidate_actions(base: np.ndarray) -> np.ndarray:
    offsets = np.asarray(
        [
            [0, 0, 0, 0],
            [0.35, 0, 0, 0], [-0.35, 0, 0, 0],
            [0, 0.30, 0, 0], [0, -0.30, 0, 0],
            [0, 0, 0.20, 0], [0, 0, -0.20, 0],
            [0, 0, 0, 0.08], [0, 0, 0, -0.08],
            [0.25, 0.20, 0, 0], [0.25, -0.20, 0, 0],
            [-0.20, 0.20, 0, 0], [-0.20, -0.20, 0, 0],
            [0.20, 0, 0.15, 0], [0.20, 0, -0.15, 0],
        ],
        dtype=np.float32,
    )
    return np.clip(base[None] + offsets, -ACTION_LIMITS, ACTION_LIMITS)


def candidate_trajectory(position, orientation, delta_body, horizon_scale: float = 12.0):
    endpoint = np.asarray(position) + Rotation.from_quat(orientation).apply(delta_body * horizon_scale)
    fractions = np.linspace(0.0, 1.0, 12)[1:]
    return [
        enu_to_backend_world(np.asarray(position) * (1.0 - fraction) + endpoint * fraction).tolist()
        for fraction in fractions
    ]


def rerank(world_model, latent, base_action, position, orientation, device):
    candidates = candidate_actions(base_action.detach().cpu().numpy())
    tensor = torch.from_numpy(candidates).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = world_model.predict(latent, tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    physical = prediction["physical_mean"][0].cpu().numpy()
    std = prediction["physical_std"][0].cpu().numpy()
    uncertainty = np.linalg.norm(std[:, :3], axis=1) + std[:, 3] + 0.1 * std[:, 4]
    deviation = np.linalg.norm((candidates - candidates[0]) / ACTION_LIMITS, axis=1)
    clearance_penalty = np.square(np.maximum(0.0, 5.0 - physical[:, 4]))
    scores = physical[:, 3] - 0.06 * clearance_penalty - 0.55 * uncertainty - 0.08 * deviation
    safe = (physical[:, 4] >= 2.5) & (uncertainty <= 0.8)
    safe[0] = True
    gated_scores = np.where(safe, scores, -np.inf)
    raw_selected = int(np.argmax(gated_scores))
    selected = raw_selected if gated_scores[raw_selected] >= gated_scores[0] + 0.002 else 0
    predicted_clearance = physical[:, 4]
    risk = 1.0 / (1.0 + np.exp(np.clip((predicted_clearance - 3.0) / 0.6, -30.0, 30.0)))
    next_latent = prediction["next_latent_mean"][0, selected].cpu().numpy()
    top_candidates = []
    for index in range(15):
        top_candidates.append({
            "candidate_index": index,
            "score": float(scores[index]),
            "collision_probability": float(risk[index]),
            "uncertainty": float(uncertainty[index]),
            "predicted_collision": bool(predicted_clearance[index] < 2.5),
            "trajectory_world_m": candidate_trajectory(position, orientation, physical[index, :3]),
        })
    return {
        "candidates": candidates,
        "selected_index": selected,
        "raw_selected_index": raw_selected,
        "selected_action": candidates[selected],
        "selected_risk": float(risk[selected]),
        "selected_uncertainty": float(uncertainty[selected]),
        "scores": scores,
        "physical": physical,
        "next_latent": next_latent,
        "top_candidates": top_candidates,
        "latency_ms": latency_ms,
    }


def inspect_video(output: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(output))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_stats = []
    # Some H.264 backends cannot seek to the exact advertised final frame.  A
    # sample one second before EOF still validates the final recorded segment.
    for index in sorted(set([0, max(0, frames // 2), max(0, frames - 30)])):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            sample_stats.append({"frame": index, "mean": float(frame.mean()), "std": float(frame.std())})
    capture.release()
    valid = width == 1920 and height == 1080 and fps >= 20 and frames >= 30 and len(sample_stats) == 3 and all(item["std"] > 5 for item in sample_stats)
    return {
        "status": "PASS" if valid else "FAIL",
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_s": frames / max(fps, 1e-9),
        "sample_frame_statistics": sample_stats,
        "continuous_browser_recording": True,
        "screenshot_stitching": False,
    }


def transcode_and_qa(source: Path, output: Path, speed: float = 1.0) -> dict[str, object]:
    if not math.isfinite(speed) or speed <= 0 or speed > 10:
        raise ValueError("video speed must be in (0, 10]")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
    ]
    if abs(speed - 1.0) > 1e-9:
        command.extend(["-vf", f"setpts=PTS/{speed:.8f},fps=30"])
    command.extend([
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ])
    subprocess.run(command, check=True)
    result = inspect_video(output)
    result["playback_speed"] = float(speed)
    return result


def main() -> None:
    args = parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    before = health()
    if int((before.get("clients") or {}).get("policy", 0)) != 0:
        raise RuntimeError("an existing policy client is active")
    surfaces = [item for item in before.get("surfaces", []) if float(item.get("age_s", 99)) < 5]
    if len(surfaces) != 1 or not bool(surfaces[0].get("scene_ready")):
        raise RuntimeError("exactly one fresh, ready real sensor surface is required")
    semantic_report = None
    semantic_mission = None
    if args.semantic_mission_plan:
        semantic_report = json.loads(args.semantic_mission_plan.read_text(encoding="utf-8"))
        semantic_mission = SemanticMissionPlan.from_report(semantic_report)
    custom_requested = args.custom_start_backend is not None or args.custom_goal_backend is not None
    if custom_requested:
        if args.custom_start_backend is None or args.custom_goal_backend is None:
            raise ValueError("custom start and goal must be supplied together")
        stack = HelsinkiNavigationStack.load(args.scene, enable_triangle_geometry=True)
        start_backend = np.asarray(args.custom_start_backend, dtype=np.float64)
        goal_backend = np.asarray(args.custom_goal_backend, dtype=np.float64)
        explicit_via = [np.asarray(point, dtype=np.float64) for point in args.via_backend]
        if semantic_mission is not None:
            semantic_via = [
                np.asarray(point, dtype=np.float64)
                for point in semantic_mission.ordered_waypoints_backend
            ]
            if explicit_via and (
                len(explicit_via) != len(semantic_via)
                or any(not np.allclose(left, right, atol=1e-6) for left, right in zip(explicit_via, semantic_via))
            ):
                raise ValueError("explicit via points do not match the gated semantic mission plan")
            via_points = semantic_via
        else:
            via_points = explicit_via
            semantic_mission = SemanticMissionPlan.deterministic([*via_points, goal_backend])
        mission_points = [
            start_backend,
            *via_points,
            goal_backend,
        ]
        segments = []
        route_backend_parts = []
        for segment_index, (segment_start, segment_goal) in enumerate(
            zip(mission_points[:-1], mission_points[1:])
        ):
            segment = stack.plan(
                segment_start,
                segment_goal,
                expert_mode="high_altitude",
                allow_layer_transitions=True,
            )
            segments.append(segment)
            route_backend_parts.append(
                segment.trajectory if segment_index == 0 else segment.trajectory[1:]
            )
        route_backend = np.concatenate(route_backend_parts, axis=0)
        route_enu = backend_world_to_enu(route_backend)
        start_enu = backend_world_to_enu(start_backend)
        goal_enu = backend_world_to_enu(goal_backend)
        mission_waypoints_enu = [backend_world_to_enu(point) for point in [*via_points, goal_backend]]
        direction = route_enu[1] - route_enu[0]
        contract = {
            "route": route_enu,
            "start": start_enu,
            "goal": goal_enu,
            "initial_yaw": float(math.atan2(direction[1], direction[0])),
        }
        episode_index = None
        episode_id = "HelsinkiCentral1km_custom_multipoint_long_range"
        task_type = "long_range"
        training_membership = "custom 1 km route; not a Dataset v1 training episode"
        planning_report = {
            "path_length_m": float(sum(segment.path_length_m for segment in segments)),
            "straight_planar_distance_m": float(np.linalg.norm((goal_enu - start_enu)[:2])),
            "trajectory_points": int(len(route_backend)),
            "waypoints_backend": [point.tolist() for point in mission_points],
            "segment_count": len(segments),
            "segments": [
                {
                    "path_length_m": float(segment.path_length_m),
                    "validation": segment.validation,
                    "triangle_validation": segment.triangle_validation,
                }
                for segment in segments
            ],
        }
        if semantic_report is not None:
            planning_report["semantic_mission_plan"] = semantic_report
            planning_report["semantic_waypoints_consumed_by_route"] = True
    else:
        if semantic_mission is not None:
            raise ValueError("semantic mission plan requires a custom start and goal")
        if args.qa is None:
            raise ValueError("--qa is required when a canonical episode is selected")
        records = load_qa_episode_records(args.qa)
        record = next((item for item in records if item.episode_index == args.episode), None)
        if record is None or record.task_type != "rooftop_to_ground":
            raise ValueError("selected episode must be a canonical rooftop_to_ground route")
        contract = episode_contract(record)
        episode_index = record.episode_index
        episode_id = record.episode_id
        task_type = record.task_type
        training_membership = "held-out episode 080-099; episode 097 was not used for policy/world-model training"
        planning_report = None
        goal_backend = enu_to_backend_world(contract["goal"])
        semantic_mission = SemanticMissionPlan.deterministic([goal_backend])
        mission_waypoints_enu = [np.asarray(contract["goal"], dtype=np.float64)]
    route = PolylineRoute(
        contract["route"], normal_lookahead_m=20.0, turn_lookahead_m=20.0,
        lookahead_speed_gain_s=0.0, maximum_lookahead_m=20.0,
    )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    policy, policy_metadata = load_observation_policy_checkpoint(args.policy, device=device)
    world_model, world_metadata = load_navigation_world_model_checkpoint(args.world_model, device=device)
    if world_metadata.get("status") != "OFFLINE_PASS":
        raise RuntimeError("world model has not passed offline validation")
    agent_runtime = HelsinkiAgentWorldModelRuntime(
        semantic_mission,
        mission_waypoints_enu,
        semantic_waypoint_tolerance_m=25.0,
        final_goal_tolerance_m=args.goal_tolerance_m,
    )
    rgb_history = deque(maxlen=policy.config.history_frames)
    depth_history = deque(maxlen=policy.config.history_frames)
    valid_history = deque(maxlen=policy.config.history_frames)
    previous_action = np.zeros(4, dtype=np.float32)
    goal_distances: list[float] = []
    cross_track_errors: list[float] = []
    inference_latencies_ms: list[float] = []
    safety_interventions = 0
    result = {
        "schema": "urbanfly-helsinki-world-model-video-qa-v1",
        "status": "RUNNING",
        "episode_index": episode_index,
        "episode_id": episode_id,
        "task_type": task_type,
        "training_membership": training_membership,
        "policy": str(args.policy.resolve()),
        "world_model": str(args.world_model.resolve()),
        "world_model_control_authority": "15-candidate local action reranker only",
        "agent_control_authority": "semantic mission sequencing and fail-closed action authorization",
        "semantic_mission_consumed": semantic_report is not None,
        "backend_safety_shield": True,
        "global_route_source": (
            "frozen Helsinki global planner; semantic mission planned segment by segment"
            if custom_requested
            else "frozen planner route stored in canonical Dataset v1"
        ),
        "action_duration_s": float(args.action_duration_s),
        "steps": 0,
        "selection_changed_steps": 0,
        "world_model_rerank_steps": 0,
        "latent_visualization_steps": 0,
        "trajectory_every_10_steps": [],
        "health_before": before,
        "planning": planning_report,
        "recording_layout": args.recording_layout,
        "requested_video_speed": args.video_speed,
    }
    adapter = HelsinkiDigitalTwinAdapter({
        "websocket_url": "ws://127.0.0.1:8765/ws",
        "urbanfly_scenario": "single_uav_world_model",
        "vehicle_name": "WM-UAV-01",
        "policy_family": "helsinki_latent_world_model_v1",
        "backend_safety_shield": True,
        "policy_lockstep": True,
        "sensor_timeout_s": 20.0,
        "command_timeout_s": 5.0,
        "dynamic_actor_density": 0.0,
        "episode_duration_s": 900.0,
        "initial_yaw_enu_radians": contract["initial_yaw"],
    })
    recording = None
    recording_started = False
    source_video = None
    source_manifest = None
    try:
        initial = adapter.connect_and_reset(
            task_type=task_type,
            split="world_model_video",
            seed=20260830 + (args.episode or 0),
            start_enu_m=contract["start"],
            goal_enu_m=contract["goal"],
        )
        frame = initial.frame
        state = initial.kinematics
        rgb_history.append(frame.rgb); depth_history.append(frame.depth_m); valid_history.append(frame.valid_mask)
        agent_runtime.begin(
            observation_timestamp_s=float(frame.timestamp),
            position_enu=state.position,
        )
        adapter.start_synchronized_recording(
            output_dir,
            fps=30.0,
            layout=args.recording_layout,
        )
        recording_started = True
        for step in range(args.max_steps):
            speed = float(np.linalg.norm(state.linear_velocity))
            route_state = route.observe(state.position, speed_mps=speed)
            local_goal_body = Rotation.from_quat(state.orientation_xyzw).inv().apply(route_state.local_goal_nwu - state.position)
            features = public_state_features(
                local_goal_body=local_goal_body[None],
                linear_velocity_world=state.linear_velocity[None],
                angular_velocity_world=state.angular_velocity[None],
                orientation_xyzw=state.orientation_xyzw[None],
                previous_action_physical=previous_action[None],
            )[0]
            latent, base_action, policy_started = policy_latent_and_action(
                policy, rgb_history, depth_history, valid_history, features, device
            )
            decision = rerank(world_model, latent, base_action, state.position, state.orientation_xyzw, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_latency_ms = (time.perf_counter() - policy_started) * 1000.0
            selected_action = np.asarray(decision["selected_action"], dtype=np.float64)
            selected_action, terminal_blend = terminal_capture_action(
                selected_action, local_goal_body, route_state.remaining_m
            )
            agent_runtime.authorize_world_model(
                WorldModelActionDecision.create(
                    step=step,
                    selected_index=int(decision["selected_index"]),
                    candidate_count=len(decision["candidates"]),
                    predicted_risk=float(decision["selected_risk"]),
                    uncertainty=float(decision["selected_uncertainty"]),
                    action_body_flu=selected_action,
                )
            )
            visualization = {
                "decision_sequence": step,
                "candidate_count": 15,
                "selected_index": decision["selected_index"],
                "raw_selected_index": decision["raw_selected_index"],
                "selection_method": "learned_latent_dynamics_ensemble",
                "control_authority": "candidate_reranker_only",
                "selected_trajectory_world_m": decision["top_candidates"][decision["selected_index"]]["trajectory_world_m"],
                "top_candidates": decision["top_candidates"],
                "planner_latency_ms": total_latency_ms,
                "predicted_risk": decision["selected_risk"],
                "ensemble_uncertainty": decision["selected_uncertainty"],
                "latent_state": latent[0].detach().cpu().numpy().tolist(),
                "predicted_next_latent": decision["next_latent"].tolist(),
            }
            adapter.publish_policy_visualization(visualization)
            command_world = Rotation.from_quat(state.orientation_xyzw).apply(selected_action[:3])
            execution = adapter.step_velocity(
                command_world,
                float(selected_action[3]),
                args.action_duration_s,
                inference_latency_ms=total_latency_ms,
                predicted_risk=decision["selected_risk"],
            )
            factual = execution.factual_action
            frame = execution.observation.frame
            state = execution.observation.kinematics
            collision = execution.collision
            goal_distance = float(np.linalg.norm(state.position - contract["goal"]))
            agent_directive = agent_runtime.accept_execution_feedback(
                step=step,
                feedback_timestamp_s=float(frame.timestamp),
                position_enu=state.position,
                executed_action_body_flu=np.asarray(
                    factual["action_executed_body_flu"], dtype=np.float64
                ),
                stale_action=bool(factual["stale_action"]),
                collision=bool(collision["has_collided"]),
                safety_intervened=bool(factual["safety_intervened"]),
            )
            goal_distances.append(goal_distance)
            cross_track_errors.append(float(route_state.cross_track_error_m))
            inference_latencies_ms.append(total_latency_ms)
            safety_interventions += int(bool(factual["safety_intervened"]))
            result["steps"] = step + 1
            result["world_model_rerank_steps"] += 1
            result["latent_visualization_steps"] += 1
            result["selection_changed_steps"] += int(decision["selected_index"] != 0)
            if step % 10 == 0:
                result["trajectory_every_10_steps"].append({
                    "step": step,
                    "sim_time": float(frame.timestamp),
                    "position_world_enu": np.asarray(state.position, dtype=float).tolist(),
                    "goal_distance_m": goal_distance,
                    "remaining_route_m": float(route_state.remaining_m),
                    "cross_track_error_m": float(route_state.cross_track_error_m),
                    "selected_candidate": int(decision["selected_index"]),
                    "predicted_clearance_m": float(decision["physical"][decision["selected_index"], 4]),
                    "ensemble_uncertainty": float(decision["selected_uncertainty"]),
                    "terminal_capture_blend": float(terminal_blend),
                })
            if route_state.cross_track_error_m > args.maximum_cross_track_m:
                reason = f"cross-track gate exceeded: {route_state.cross_track_error_m:.3f} m"
                agent_runtime.abort(reason)
                raise RuntimeError(reason)
            previous_action = selected_action.astype(np.float32)
            rgb_history.append(frame.rgb); depth_history.append(frame.depth_m); valid_history.append(frame.valid_mask)
            if agent_directive.status is AgentStatus.COMPLETE:
                result["success"] = True
                break
        else:
            result["success"] = False
            raise RuntimeError("maximum steps reached")
        result["collision"] = False
        result["stale_action_count"] = 0
        result["agent_closed_loop"] = agent_runtime.snapshot()
        result["safety_interventions"] = safety_interventions
    except BaseException as error:
        if agent_runtime.status is AgentStatus.RUNNING:
            agent_runtime.abort(repr(error))
        result["status"] = "FAIL"
        result["success"] = False
        result["error"] = repr(error)
        raise
    finally:
        if recording_started:
            try:
                recording = adapter.stop_synchronized_recording()
            except Exception as error:
                result["recording_error"] = repr(error)
        try:
            adapter.close(); asyncio.run(stop_simulator())
        except Exception as error:
            result["cleanup_error"] = repr(error)
        if recording and isinstance(recording.get("runtime"), dict):
            runtime = recording["runtime"]
            source_video = Path(runtime["video_path"])
            source_manifest = Path(runtime["manifest_path"])
            try:
                result["video"] = transcode_and_qa(
                    source_video,
                    output_dir / args.video_name,
                    speed=args.video_speed,
                )
            except Exception as error:
                result["recording_error"] = repr(error)
            finally:
                source_video.unlink(missing_ok=True)
                source_manifest.unlink(missing_ok=True)
        result["minimum_goal_distance_m"] = min(goal_distances, default=float("inf"))
        result["maximum_cross_track_error_m"] = max(cross_track_errors, default=0.0)
        result["safety_interventions"] = safety_interventions
        result["stale_action_count"] = 0
        result["agent_closed_loop"] = agent_runtime.snapshot()
        if inference_latencies_ms:
            latency = np.asarray(inference_latencies_ms, dtype=np.float64)
            result["inference_latency_ms"] = {
                "mean": float(np.mean(latency)),
                "p95": float(np.percentile(latency, 95)),
                "maximum": float(np.max(latency)),
            }
        loop_qa = result["agent_closed_loop"]
        if result.get("success") and result.get("video", {}).get("status") == "PASS" and result["world_model_rerank_steps"] == result["steps"] and result["latent_visualization_steps"] == result["steps"] and result["selection_changed_steps"] > 0 and loop_qa["status"] == "COMPLETE" and loop_qa["causal_chain_complete"] and loop_qa["fresh_feedbacks"] == result["steps"] and not result.get("cleanup_error"):
            result["status"] = "PASS"
        elif result.get("status") == "RUNNING":
            result["status"] = "FAIL"
        try:
            result["health_after"] = health()
        except Exception as error:
            result["health_after_error"] = repr(error)
        result["policy_checkpoint_status"] = policy_metadata.get("status")
        result["world_model_checkpoint_status"] = world_metadata.get("status")
        result["output_artifacts"] = [args.video_name, "flight_qa.json"]
        (output_dir / "flight_qa.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: result.get(key) for key in ("status", "success", "steps", "selection_changed_steps", "world_model_rerank_steps", "video")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
