#!/usr/bin/env python3
"""Continuous real-Helsinki Dataset v1 collector (no mock fallback)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import _bootstrap  # noqa: F401

ROOT = _bootstrap.PROJECT_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_frames import (  # noqa: E402
    backend_world_to_enu,
    enu_to_backend_world,
)
from backend.engine.helsinki_navigation import HelsinkiNavigationStack  # noqa: E402
from backend.engine.helsinki_spatial_split import HelsinkiSpatialSplit  # noqa: E402
from backend.engine.helsinki_urban_sampling import HelsinkiUrbanDensity  # noqa: E402
from backend.engine.local_goal import LocalGoalSelector  # noqa: E402
from scripts.verify_helsinki_low_altitude_expert import (  # noqa: E402
    DifficultTaskSampler,
    TASK_TYPES,
)
from uav_wm_navigation.data.helsinki_dataset_v1 import (  # noqa: E402
    HelsinkiDatasetV1Writer,
    HelsinkiTransition,
    validate_helsinki_dataset_v1,
)
from uav_wm_navigation.data.helsinki_dataset_v1_qa import (  # noqa: E402
    audit_helsinki_collection,
)
from uav_wm_navigation.simulators.helsinki_websocket_adapter import (  # noqa: E402
    HelsinkiWebSocketAdapter,
)
from uav_wm_navigation.utils.config import load_yaml  # noqa: E402


SCENE_ROOT = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"


def _point_at_progress(path: np.ndarray, progress_m: float) -> np.ndarray:
    segment = np.diff(path, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    target = float(np.clip(progress_m, 0.0, cumulative[-1]))
    index = min(max(int(np.searchsorted(cumulative, target, side="right") - 1), 0), len(path) - 2)
    alpha = 0.0 if lengths[index] <= 1e-9 else (target - cumulative[index]) / lengths[index]
    return path[index] + alpha * segment[index]


def _route_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _plan_smoke_tasks(stack, density, partition, seed: int, count: int) -> list[dict]:
    sampler = DifficultTaskSampler(stack, seed=seed, urban_density=density)
    split_mask = partition.masks(stack.low_altitude_grid)["train"]
    planned: list[dict] = []
    accepted_starts: list[np.ndarray] = []
    accepted_goals: list[np.ndarray] = []
    candidate_index = 0
    for episode_index in range(int(count)):
        task_type = TASK_TYPES[episode_index % len(TASK_TYPES)]
        accepted = None
        # Rank a frozen-sampler pool by endpoint-cell rarity and separation
        # before invoking the frozen planner. Small runs keep a larger retry
        # pool because they still enforce the historical 15 m hard spacing;
        # production runs use coverage ranking directly. This changes only
        # collector orchestration.
        pool_size = 80 if int(count) <= 50 else 16
        candidate_pool = []
        for _ in range(pool_size):
            task = sampler.sample(
                candidate_index,
                task_type,
                spatial_stratum="train_urban",
                start_spatial_mask=split_mask & density.non_open_mask,
                goal_spatial_mask=split_mask & density.non_open_mask,
                distance_range_override=(60.0, 210.0),
            )
            candidate_index += 1
            start = np.asarray(task.start, dtype=np.float64)
            goal = np.asarray(task.goal, dtype=np.float64)
            cell_size_m = 50.0
            start_cell = tuple(np.floor(start[[0, 2]] / cell_size_m).astype(int))
            goal_cell = tuple(np.floor(goal[[0, 2]] / cell_size_m).astype(int))
            if accepted_starts:
                start_distance = min(np.linalg.norm(start - item) for item in accepted_starts)
                goal_distance = min(np.linalg.norm(goal - item) for item in accepted_goals)
                used_start_cells = {
                    tuple(np.floor(item[[0, 2]] / cell_size_m).astype(int))
                    for item in accepted_starts
                }
                used_goal_cells = {
                    tuple(np.floor(item[[0, 2]] / cell_size_m).astype(int))
                    for item in accepted_goals
                }
            else:
                start_distance = goal_distance = 1000.0
                used_start_cells = set()
                used_goal_cells = set()
            coverage_score = (
                1000.0 * float(start_cell not in used_start_cells)
                + 1000.0 * float(goal_cell not in used_goal_cells)
                + min(start_distance, 100.0)
                + min(goal_distance, 100.0)
            )
            candidate_pool.append((coverage_score, task, start, goal))

        for attempt, (_, task, start, goal) in enumerate(
            sorted(candidate_pool, key=lambda item: item[0], reverse=True)
        ):
            # Expand endpoint coverage without modifying the frozen sampler.
            # Keep a deterministic 15 m separation while candidates are
            # plentiful, then relax it only to avoid turning coverage into a
            # brittle planning failure.
            if int(count) <= 50 and attempt < 60 and accepted_starts:
                start_distance = min(np.linalg.norm(start - item) for item in accepted_starts)
                goal_distance = min(np.linalg.norm(goal - item) for item in accepted_goals)
                if start_distance < 15.0 or goal_distance < 15.0:
                    continue
            plan = stack.plan(
                start,
                goal,
                expert_mode="low_altitude_3d",
                altitude_min_m=task.altitude_min_m,
                altitude_max_m=task.altitude_max_m,
                allow_layer_transitions=False,
            )
            if partition.assign_backend_route(plan.trajectory) != "train":
                continue
            accepted = {
                "task": asdict(task),
                "plan": plan,
            }
            break
        if accepted is None:
            raise RuntimeError(
                f"FAIL CLOSED: could not plan a train-isolated {task_type} task"
            )
        planned.append(accepted)
        accepted_starts.append(np.asarray(accepted["task"]["start"], dtype=np.float64))
        accepted_goals.append(np.asarray(accepted["task"]["goal"], dtype=np.float64))
    return planned


def _manifest_record(item: dict) -> dict:
    plan = item["plan"]
    return {
        "task": item["task"],
        "route_backend": np.asarray(plan.trajectory, dtype=np.float64).tolist(),
        "route_enu": backend_world_to_enu(plan.trajectory).tolist(),
        "planner_mode": plan.planner_mode,
        "expert_mode": plan.expert_mode,
        "planning_time_ms": float(plan.planning_time_ms),
        "path_length_m": float(plan.path_length_m),
        "triangle_validation": plan.triangle_validation,
    }


def _load_task_manifest(
    path: Path,
    stack,
    partition,
    expected_count: int,
    episode_index_offset: int = 0,
) -> list[dict]:
    all_records = json.loads(path.resolve().read_text(encoding="utf-8"))
    stop = int(episode_index_offset) + int(expected_count)
    if (
        not isinstance(all_records, list)
        or int(episode_index_offset) < 0
        or len(all_records) < stop
    ):
        raise ValueError(
            "task manifest does not contain the requested episode slice "
            f"[{episode_index_offset}:{stop}]"
        )
    records = all_records[int(episode_index_offset):stop]
    planned = []
    task_counts = {task_type: 0 for task_type in TASK_TYPES}
    for index, record in enumerate(records):
        task = dict(record["task"])
        task_type = str(task.get("task_type", ""))
        if task_type not in task_counts:
            raise ValueError(f"manifest record {index} has unknown task type {task_type!r}")
        route = np.asarray(record["route_backend"], dtype=np.float64)
        if route.ndim != 2 or route.shape[1:] != (3,) or len(route) < 2:
            raise ValueError(f"manifest record {index} has an invalid route")
        if not np.isfinite(route).all():
            raise ValueError(f"manifest record {index} route is non-finite")
        if partition.assign_backend_route(route) != "train":
            raise ValueError(f"manifest record {index} leaks outside the train split")
        heightmap = stack.validate_path(
            route,
            altitude_min_m=float(task["altitude_min_m"]),
            altitude_max_m=float(task["altitude_max_m"]),
        )
        if not heightmap["path_valid"]:
            raise ValueError(f"manifest record {index} failed heightmap readback")
        triangle = stack.local_triangle_geometry.trajectory_query(
            route, stack.required_clearance
        ).as_dict()
        if bool(triangle["collision"]):
            raise ValueError(f"manifest record {index} failed triangle readback")
        planned.append({
            "task": task,
            "plan": SimpleNamespace(
                trajectory=route,
                planner_mode=str(record["planner_mode"]),
                expert_mode=str(record.get("expert_mode", "low_altitude_3d")),
                planning_time_ms=float(record.get("planning_time_ms", 0.0)),
                path_length_m=_route_length(route),
                triangle_validation=triangle,
            ),
        })
        task_counts[task_type] += 1
    if max(task_counts.values()) - min(task_counts.values()) > 1:
        raise ValueError(f"task manifest is not balanced: {task_counts}")
    return planned


def _initial_yaw_enu(route_enu: np.ndarray) -> float:
    for delta in np.diff(route_enu, axis=0):
        if np.linalg.norm(delta[:2]) > 0.25:
            return float(math.atan2(delta[1], delta[0]))
    return 0.0


def _body_action_command(
    state,
    tracking_target_enu: np.ndarray,
    remaining_distance_m: float,
    max_speed: float,
) -> tuple[np.ndarray, float]:
    delta = tracking_target_enu - state.position
    distance = float(np.linalg.norm(delta))
    speed = min(max_speed, 0.65 * distance, max(0.8, remaining_distance_m * 0.5))
    velocity = delta / max(distance, 1e-6) * speed
    velocity[2] = float(np.clip(velocity[2], -1.5, 1.5))
    yaw_current = Rotation.from_quat(state.orientation_xyzw).as_euler("xyz")[2]
    yaw_target = yaw_current if np.linalg.norm(delta[:2]) <= 0.25 else math.atan2(delta[1], delta[0])
    yaw_error = (yaw_target - yaw_current + math.pi) % (2.0 * math.pi) - math.pi
    yaw_rate = float(np.clip(1.5 * yaw_error, -math.radians(45.0), math.radians(45.0)))
    return velocity, yaw_rate


def _episode_metrics(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        positions = handle["state/position_world"][:]
        next_position = handle["next_state/position_world"][-1]
        all_positions = np.vstack([positions, next_position])
        speed = np.linalg.norm(handle["state/linear_velocity"][:], axis=1)
        dt = handle["timestamps/dt"][:]
        orientation = handle["state/orientation_xyzw"][:]
        final_orientation = handle["next_state/orientation_xyzw"][-1]
        yaw = Rotation.from_quat(orientation).as_euler("xyz")[:, 2]
        final_yaw = Rotation.from_quat(final_orientation).as_euler("xyz")[2]
        yaw_all = np.unwrap(np.r_[yaw, final_yaw])
        commanded = handle["actions/commanded_body_flu"][:]
        executed_actions = handle["actions/executed_body_flu"][:]
        stale_action = handle["labels/stale_action"][:]
        return {
            "num_steps": int(len(dt)),
            "path_length_m": float(np.linalg.norm(np.diff(all_positions, axis=0), axis=1).sum()),
            "flight_time_s": float(dt.sum()),
            "minimum_clearance_m": float(handle["labels/minimum_clearance"][:].min()),
            "mean_speed_mps": float(speed.mean()),
            "maximum_speed_mps": float(speed.max()),
            "mean_dt_s": float(dt.mean()),
            "p95_dt_s": float(np.percentile(dt, 95)),
            "rgb_frames": int(handle["observations/rgb_front"].shape[0]),
            "depth_frames": int(handle["observations/depth_front"].shape[0]),
            "state_count": int(handle["state/position_world"].shape[0]),
            "next_state_count": int(handle["next_state/position_world"].shape[0]),
            "commanded_action_count": int(commanded.shape[0]),
            "executed_action_count": int(executed_actions.shape[0]),
            "yaw_start_degrees_enu": float(np.degrees(yaw_all[0])),
            "yaw_end_degrees_enu": float(np.degrees(yaw_all[-1])),
            "yaw_net_change_degrees_enu": float(np.degrees(yaw_all[-1] - yaw_all[0])),
            "commanded_yaw_integral_degrees": float(
                np.degrees(np.sum(executed_actions[:, 3] * dt))
            ),
            "stale_action_count": int(np.count_nonzero(stale_action)),
            "initial_sim_time_s": float(handle["timestamps/sim"][0]),
            "final_sim_time_s": float(handle["timestamps/next_sim"][-1]),
            "initial_position_world": positions[0].tolist(),
            "final_position_world": next_position.tolist(),
            "initial_speed_mps": float(speed[0]),
            "initial_acceleration_mps2": float(
                np.linalg.norm(handle["state/linear_acceleration"][0])
            ),
            "initial_route_progress_m": float(handle["route/progress"][0]),
        }


def collect_one(
    adapter: HelsinkiWebSocketAdapter,
    stack: HelsinkiNavigationStack,
    task_record: dict,
    output_dir: Path,
    episode_index: int,
    config: dict,
) -> dict:
    task = task_record["task"]
    plan = task_record["plan"]
    route_backend = np.asarray(plan.trajectory, dtype=np.float64)
    route_enu = backend_world_to_enu(route_backend)
    start_enu, goal_enu = route_enu[0], route_enu[-1]
    episode_id = f"HelsinkiCentral1km_real_smoke_{episode_index:03d}_{task['task_type']}"
    adapter.config.update(
        {
            "seed": int(config["smoke_seed"]) + episode_index,
            "initial_yaw_enu_radians": _initial_yaw_enu(route_enu),
            "dynamic_actor_density": float(config.get("dynamic_actor_density", 0.0)),
        }
    )
    adapter.reset()
    adapter.configure_scenario(task["task_type"], "smoke", int(config["smoke_seed"]) + episode_index)
    adapter.set_initial_pose(start_enu)
    adapter.set_goal(goal_enu)
    adapter.takeoff()

    local_selector = LocalGoalSelector(float(config["local_goal_lookahead_m"]))
    writer = HelsinkiDatasetV1Writer(
        output_dir,
        episode_id,
        {
            "scene_id": "HelsinkiCentral1km",
            "scene_seed": int(config["smoke_seed"]) + episode_index,
            "task_type": task["task_type"],
            "collection_mode": "expert",
            "start_world": start_enu.tolist(),
            "global_goal_world": goal_enu.tolist(),
            "spatial_split": "train",
            "urban_region_type": task["spatial_stratum"],
            "lookahead_distance_m": float(config["local_goal_lookahead_m"]),
            "global_route_world": route_enu.tolist(),
            "global_route_backend": route_backend.tolist(),
            "expert_planning_result": {
                "result": "PLANNED",
                "planner_mode": plan.planner_mode,
                "expert_mode": plan.expert_mode,
                "planning_time_ms": plan.planning_time_ms,
                "path_length_m": plan.path_length_m,
                "triangle_validation": plan.triangle_validation,
            },
            "transition_timeline": [
                "synchronized sensor/state t",
                "privileged route + LocalGoalSelector",
                "action_commanded_t",
                "backend safety/controller -> action_executed_t",
                "6-DOF sim integration",
                "synchronized sensor/state t+1",
            ],
        },
    )
    control_period = 1.0 / float(config["policy_frequency_hz"])
    route_length = _route_length(route_backend)
    max_steps = int(math.ceil(route_length / 2.5 / control_period) + 300)
    previous_progress = 0.0
    first_policy_step_id = None
    episode_started = time.perf_counter()
    failure_reason = "timeout"
    sensor = adapter.get_depth()
    state = adapter.get_kinematics()
    for step_index in range(max_steps):
        backend_position = enu_to_backend_world(state.position)
        backend_velocity = enu_to_backend_world(state.linear_velocity)
        yaw_enu = Rotation.from_quat(state.orientation_xyzw).as_euler("xyz")[2]
        selection = local_selector.select(
            backend_position,
            backend_velocity,
            -math.degrees(yaw_enu),
            route_backend,
        )
        if selection.route_progress_m + 0.25 < previous_progress:
            raise RuntimeError("FAIL CLOSED: Local Goal route progress regressed")
        previous_progress = selection.route_progress_m
        tracking_backend = _point_at_progress(
            route_backend,
            selection.route_progress_m + float(config["control_lookahead_m"]),
        )
        tracking_enu = backend_world_to_enu(tracking_backend)
        command_velocity, command_yaw_rate = _body_action_command(
            state,
            tracking_enu,
            selection.remaining_distance_m,
            float(config["maximum_speed_mps"]),
        )
        action_record = adapter.execute_velocity_command(
            command_velocity, command_yaw_rate, control_period
        )
        if first_policy_step_id is None:
            first_policy_step_id = int(action_record["step_id"])
        action_timestamp = float(action_record["action_timestamp"])
        if abs(action_timestamp - state.timestamp) <= 1e-6:
            action_timestamp = float(state.timestamp)
        next_sensor = adapter.get_depth()
        next_state = adapter.get_kinematics()
        if abs(next_sensor.timestamp - next_state.timestamp) > 1e-6:
            raise RuntimeError("FAIL CLOSED: next RGB-D and state timestamps differ")
        if not state.timestamp <= action_timestamp < next_state.timestamp:
            raise RuntimeError(
                "FAIL CLOSED: action/state off-by-one detected "
                f"at step={step_index}: state_t={state.timestamp!r}, "
                f"action_t={action_timestamp!r}, next_state_t={next_state.timestamp!r}"
            )
        collision_info = adapter.get_collision_info()
        backend_next = enu_to_backend_world(next_state.position)
        triangle = stack.local_triangle_geometry.segment_query(
            backend_position,
            backend_next,
            stack.drone_radius,
        )
        collision = bool(collision_info.get("has_collided", False) or triangle.collision)
        next_selection = local_selector.select(
            backend_next,
            enu_to_backend_world(next_state.linear_velocity),
            -math.degrees(Rotation.from_quat(next_state.orientation_xyzw).as_euler("xyz")[2]),
            route_backend,
        )
        success = bool(np.linalg.norm(next_state.position - goal_enu) <= float(config["goal_tolerance_m"]))
        timeout = step_index == max_steps - 1 and not success and not collision
        terminated = bool(success or collision)
        reward = (
            next_selection.route_progress_m - selection.route_progress_m
            + (5.0 if success else 0.0)
            - (10.0 if collision else 0.0)
        )
        writer.append(
            HelsinkiTransition(
                sensor=sensor,
                state=state,
                next_state=next_state,
                wall_timestamp=time.time(),
                action_timestamp=action_timestamp,
                global_goal_world=goal_enu,
                local_goal_world=backend_world_to_enu(selection.local_goal_world),
                local_goal_body=selection.local_goal_body_flu,
                global_route_progress=selection.route_progress_m,
                remaining_route_distance=selection.remaining_distance_m,
                action_commanded=action_record["action_commanded_body_flu"],
                action_executed=action_record["action_executed_body_flu"],
                reward=reward,
                collision=collision,
                minimum_clearance=triangle.minimum_distance_m,
                success=success,
                terminated=terminated,
                truncated=timeout,
                safety_intervened=bool(action_record["safety_intervened"]),
                stale_action=bool(action_record["stale_action"]),
            )
        )
        if success:
            failure_reason = ""
            break
        if collision:
            failure_reason = "collision"
            break
        if timeout:
            break
        sensor, state = next_sensor, next_state
    writer.metadata.update(
        {
            "failure_reason": failure_reason,
            "wall_duration_s": time.perf_counter() - episode_started,
            "success": failure_reason == "",
            "collision": failure_reason == "collision",
            "timeout": failure_reason == "timeout",
        }
    )
    path = writer.close()
    integrity = validate_helsinki_dataset_v1(path)
    metrics = _episode_metrics(path)
    expected_initial_yaw = _initial_yaw_enu(route_enu)
    actual_initial_yaw = math.radians(metrics["yaw_start_degrees_enu"])
    initial_yaw_error = (
        actual_initial_yaw - expected_initial_yaw + math.pi
    ) % (2.0 * math.pi) - math.pi
    partial_path = path.with_suffix(".h5.partial")
    return {
        "episode_id": episode_id,
        "task_type": task["task_type"],
        "start_world": start_enu.tolist(),
        "goal_world": goal_enu.tolist(),
        "success": failure_reason == "",
        "collision": failure_reason == "collision",
        "timeout": failure_reason == "timeout",
        "failure_reason": failure_reason,
        **metrics,
        "action_alignment": "PASS" if integrity["checks"]["action_state_temporal_consistency"] else "FAIL",
        "local_goal_qa": "PASS" if integrity["checks"]["local_goal_progression"] else "FAIL",
        "yaw_qa": "PASS" if integrity["checks"]["yaw_rate_orientation_consistency"] else "FAIL",
        "hdf5_readback": integrity["status"],
        "integrity_checks": integrity["checks"],
        "reset_evidence": {
            "start_position_error_m": float(
                np.linalg.norm(np.asarray(metrics["initial_position_world"]) - start_enu)
            ),
            "initial_yaw_error_degrees": float(abs(math.degrees(initial_yaw_error))),
            "first_policy_step_id": first_policy_step_id,
            "action_buffer_reset": bool(first_policy_step_id == 0),
            # The adapter only returns after the synchronized packet reports
            # this same new step id.  stale_action may still be true when the
            # command expires before the slower RGB-D capture; that is an
            # in-transition timeout label, not evidence of an old episode
            # command leaking into the new episode.
            "new_policy_step_observed": bool(first_policy_step_id == 0),
            "writer_flush_closed": bool(path.exists() and not partial_path.exists()),
        },
        "path": str(path),
    }


def _reset_transition(previous: dict, current: dict, adapter: HelsinkiWebSocketAdapter) -> dict:
    evidence = current["reset_evidence"]
    state_reset = bool(
        evidence["start_position_error_m"] <= 0.25
        and current["initial_speed_mps"] <= 0.25
        and current["initial_acceleration_mps2"] <= 0.25
    )
    controller_reset = bool(
        current["initial_speed_mps"] <= 0.25
        and current["initial_acceleration_mps2"] <= 0.25
        and evidence["action_buffer_reset"]
        and evidence["new_policy_step_observed"]
    )
    yaw_reset = bool(evidence["initial_yaw_error_degrees"] <= 2.0)
    local_goal_reset = bool(current["initial_route_progress_m"] <= 1.0)
    timestamps_legal = bool(
        current["integrity_checks"]["timestamps_monotonic"]
        and current["integrity_checks"]["action_state_temporal_consistency"]
        and (
            current["initial_sim_time_s"] > previous["final_sim_time_s"]
            or current["initial_sim_time_s"] <= 2.0
        )
    )
    writer_flush = bool(previous["reset_evidence"]["writer_flush_closed"])
    connection_persistence = bool(adapter._connected and adapter._thread is not None)
    goal_switched = bool(
        not np.allclose(previous["goal_world"], current["goal_world"], atol=1e-6)
    )
    checks = {
        "state_reset": state_reset,
        "controller_reset": controller_reset,
        "yaw_reset": yaw_reset,
        "local_goal_reset": local_goal_reset,
        "action_buffer_reset": bool(evidence["action_buffer_reset"]),
        "stale_action_inheritance_absent": bool(evidence["new_policy_step_observed"]),
        "timestamps_legal": timestamps_legal,
        "writer_flush": writer_flush,
        "connection_persistence": connection_persistence,
        "goal_switched": goal_switched,
    }
    return {
        "from_episode": previous["episode_id"],
        "to_episode": current["episode_id"],
        "previous_final_state": previous["final_position_world"],
        "new_initial_state": current["initial_position_world"],
        "previous_final_sim_time_s": previous["final_sim_time_s"],
        "new_initial_sim_time_s": current["initial_sim_time_s"],
        "checks": checks,
        "automatic_reset": "PASS" if all(checks.values()) else "FAIL",
    }


def _summary_payload(
    requested_episodes: int,
    records: list[dict],
    reset_transitions: list[dict],
) -> dict:
    return {
        "requested_episodes": int(requested_episodes),
        "completed_episodes": len(records),
        "automatic_reset_count": len(reset_transitions),
        "reset_transitions": reset_transitions,
        "collector_process_ids": sorted({r["collector_process_id"] for r in records}),
        "connection_thread_ids": sorted({r["connection_thread_id"] for r in records}),
        "scene": "HelsinkiCentral1km",
        "simulator": "UrbanFly browser WebGL RGB-D + backend 6-DOF",
        "planner": "frozen Helsinki low-altitude 3-D privileged expert",
        "sampler": "frozen geometry-derived Helsinki urban sampler",
        "local_goal": "frozen LocalGoalSelector",
        "collision_source": "Helsinki L18 triangle mesh oracle",
        "writer": "HDF5 urbanfly-helsinki-dataset-v1",
        "teacher": "FrozenHelsinkiPrivilegedExpert",
        "mock_candidate_planner_used": False,
        "records": records,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _checkpoint_technical_pass(report: dict) -> bool:
    checks = report["gate_checks"]
    return bool(all(checks[name] for name in (
        "episode_count",
        "balanced_task_counts",
        "reset_count_and_success",
        "corrupted_hdf5_zero",
        "partial_count_zero",
        "cross_episode_stale_action_zero",
        "all_stale_executed_actions_factual_hover",
        "all_dataset_integrity_checks",
    )))


def _require_episode_integrity(record: dict, output_dir: Path) -> None:
    """Stop before another reset if the just-closed episode is not intact."""
    checks = record.get("integrity_checks", {})
    if record.get("hdf5_readback") != "PASS" or not checks or not all(checks.values()):
        raise RuntimeError("FAIL CLOSED: per-episode HDF5/schema/readback failure")
    if not record.get("reset_evidence", {}).get("action_buffer_reset", False):
        raise RuntimeError("FAIL CLOSED: new episode did not observe policy step zero")
    if list(output_dir.glob("*.partial")):
        raise RuntimeError("FAIL CLOSED: partial file remains after episode close")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs" / "helsinki_dataset_v1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--task-manifest-in",
        type=Path,
        help="reuse and independently revalidate an already-reviewed route manifest",
    )
    parser.add_argument(
        "--episode-index-offset",
        type=int,
        default=0,
        help="absolute episode index for continuation batches using a reviewed manifest",
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if not 1 <= args.episodes <= 500:
        raise ValueError("this entry point supports 1..500 continuous real episodes")
    if args.episode_index_offset < 0 or args.episode_index_offset + args.episodes > 500:
        raise ValueError("episode index slice must stay within Dataset v1 indices 0..499")
    if args.episode_index_offset and args.task_manifest_in is None:
        raise ValueError("continuation batches require --task-manifest-in")
    stack = HelsinkiNavigationStack.load(SCENE_ROOT, enable_triangle_geometry=True)
    density = HelsinkiUrbanDensity(stack.low_altitude_grid)
    partition = HelsinkiSpatialSplit(float(config["spatial_guard_m"]))
    tasks = (
        _load_task_manifest(
            args.task_manifest_in,
            stack,
            partition,
            int(args.episodes),
            int(args.episode_index_offset),
        )
        if args.task_manifest_in is not None
        else _plan_smoke_tasks(
            stack, density, partition, int(config["smoke_seed"]), int(args.episodes)
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_manifest = [_manifest_record(item) for item in tasks]
    _write_json_atomic(args.output_dir / "smoke_tasks.json", task_manifest)
    if args.prepare_only:
        print(json.dumps({"status": "PASS", "prepared_tasks": len(tasks)}, indent=2))
        return 0
    adapter = HelsinkiWebSocketAdapter(
        {
            "websocket_url": args.url,
            "urbanfly_scenario": "single_uav_world_model",
            "vehicle_name": "WM-UAV-01",
            "policy_family": "frozen_helsinki_privileged_expert",
            "backend_safety_shield": bool(config["backend_safety_shield"]),
            "policy_lockstep": True,
            "sensor_timeout_s": 20.0,
            "command_timeout_s": 5.0,
            "dynamic_actor_density": float(config.get("dynamic_actor_density", 0.0)),
        }
    )
    records = []
    reset_transitions = []
    progress_path = args.output_dir / "collection_progress.json"

    def write_progress(status: str, current_episode: int | None = None) -> None:
        _write_json_atomic(progress_path, {
            **_summary_payload(int(args.episodes), records, reset_transitions),
            "status": status,
            "collector_process_id": os.getpid(),
            "episode_index_offset": int(args.episode_index_offset),
            "current_episode_index": current_episode,
            "updated_unix_s": time.time(),
        })

    checkpoint_targets = {
        value
        for value in (5, 10, 25, 50, 100, 250, 500)
        if value <= int(args.episodes)
    }
    try:
        adapter.connect()
        connection_thread_id = id(adapter._thread)
        for local_index, task in enumerate(tasks):
            episode_index = int(args.episode_index_offset) + local_index
            write_progress("COLLECTING", episode_index)
            record = collect_one(
                adapter, stack, task, args.output_dir, episode_index, config
            )
            record["collector_process_id"] = int(os.getpid())
            record["connection_thread_id"] = connection_thread_id
            record["connection_alive_after_episode"] = bool(
                adapter._connected and id(adapter._thread) == connection_thread_id
            )
            if records:
                reset_record = _reset_transition(records[-1], record, adapter)
                reset_transitions.append(reset_record)
            records.append(record)
            write_progress("EPISODE_CLOSED", episode_index)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            _require_episode_integrity(record, args.output_dir)
            if records and reset_transitions and reset_transitions[-1]["to_episode"] == record["episode_id"]:
                if reset_transitions[-1]["automatic_reset"] != "PASS":
                    failure_qa = audit_helsinki_collection(
                        args.output_dir,
                        expected_episodes=len(records),
                        reset_transitions=reset_transitions,
                        output_path=args.output_dir / f"checkpoint_{len(records):03d}_failure_qa.json",
                    )
                    raise RuntimeError(
                        "FAIL CLOSED: automatic reset QA failed: "
                        + json.dumps(failure_qa["reset"], ensure_ascii=False)
                    )
            if len(records) in checkpoint_targets:
                checkpoint_path = args.output_dir / f"checkpoint_{len(records):03d}_qa.json"
                checkpoint = audit_helsinki_collection(
                    args.output_dir,
                    expected_episodes=len(records),
                    reset_transitions=reset_transitions,
                    output_path=checkpoint_path,
                )
                print(json.dumps({
                    "checkpoint": len(records),
                    "status": checkpoint["status"],
                    "technical_pass": _checkpoint_technical_pass(checkpoint),
                    "success_rate": checkpoint["success_rate"],
                    "collision_count": checkpoint["collision_count"],
                    "stale_action_ratio": checkpoint["stale_action"]["ratio"],
                    "maximum_stale_action_burst": checkpoint["stale_action"]["maximum_burst"],
                    "path": str(checkpoint_path.resolve()),
                }, ensure_ascii=False), flush=True)
                if not _checkpoint_technical_pass(checkpoint):
                    raise RuntimeError(
                        f"FAIL CLOSED: technical Dataset QA failed at checkpoint {len(records)}"
                    )
    except BaseException as error:
        failure = _summary_payload(int(args.episodes), records, reset_transitions)
        failure.update({
            "status": "FAIL",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        _write_json_atomic(args.output_dir / "collection_failure.json", failure)
        write_progress("FAIL")
        raise
    finally:
        adapter.close()
    summary = _summary_payload(int(args.episodes), records, reset_transitions)
    summary_path = args.output_dir / "collection_summary.json"
    _write_json_atomic(summary_path, summary)
    final_qa_path = args.output_dir / "independent_collection_qa.json"
    final_qa = audit_helsinki_collection(
        args.output_dir,
        expected_episodes=int(args.episodes),
        reset_transitions=reset_transitions,
        output_path=final_qa_path,
    )
    summary["independent_collection_qa"] = str(final_qa_path.resolve())
    summary["all_pass"] = final_qa["status"] == "PASS"
    _write_json_atomic(summary_path, summary)
    write_progress("PASS" if summary["all_pass"] else "FAIL")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
