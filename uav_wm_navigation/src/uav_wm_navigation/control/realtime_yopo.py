from __future__ import annotations

import json
import math
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Generic, TypeVar

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from uav_wm_navigation.planners.base import PlanningContext
from uav_wm_navigation.simulators.base import SimulatorAdapter
from uav_wm_navigation.types import (
    ControlSample,
    TimestampedSensorFrame,
    TrajectoryPlan,
    VehicleState,
)
from uav_wm_navigation.control.safety_filter import SafetyFilter
from uav_wm_navigation.control.route_manager import PolylineRoute, RouteProjection


T = TypeVar("T")


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = map(float, quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class LatestValue(Generic[T]):
    """A one-element mailbox: readers never build a stale processing queue."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: T | None = None
        self._version = 0

    def publish(self, value: T) -> int:
        with self._condition:
            self._value = value
            self._version += 1
            self._condition.notify_all()
            return self._version

    def get(self) -> tuple[T | None, int]:
        with self._condition:
            return self._value, self._version

    def wait_newer(self, version: int, timeout_s: float) -> tuple[T | None, int]:
        with self._condition:
            self._condition.wait_for(lambda: self._version > version, timeout=max(float(timeout_s), 0.0))
            return self._value, self._version


def _interp_trajectory(trajectory, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.linspace(0.0, float(trajectory.duration), len(trajectory.positions), dtype=np.float64)
    query = float(np.clip(elapsed_s, 0.0, trajectory.duration))
    position = np.array([np.interp(query, times, trajectory.positions[:, axis]) for axis in range(3)])
    velocity = np.array([np.interp(query, times, trajectory.velocities[:, axis]) for axis in range(3)])
    acceleration = np.array([np.interp(query, times, trajectory.accelerations[:, axis]) for axis in range(3)])
    return position, velocity, acceleration


def _calculate_yaw_official(
    velocity: np.ndarray, goal_direction: np.ndarray, last_yaw: float, dt: float, max_yaw_rate_factor: float = 0.5,
) -> tuple[float, float]:
    """Port of policy.poly_solver.calculate_yaw from the official repository."""
    velocity_direction = velocity / (float(np.linalg.norm(velocity)) + 1e-5)
    goal_distance = float(np.linalg.norm(goal_direction))
    normalized_goal = goal_direction / (goal_distance + 1e-5)
    goal_yaw = float(np.arctan2(normalized_goal[1], normalized_goal[0]))
    delta_yaw = (goal_yaw - last_yaw + np.pi) % (2.0 * np.pi) - np.pi
    weight = 6.0 * abs(delta_yaw) / np.pi
    desired_direction = velocity_direction + weight * normalized_goal
    desired_yaw = float(np.arctan2(desired_direction[1], desired_direction[0])) if goal_distance > 0.5 else last_yaw
    yaw_difference = (desired_yaw - last_yaw + np.pi) % (2.0 * np.pi) - np.pi
    maximum_change = max_yaw_rate_factor * np.pi * dt
    yaw_change = float(np.clip(yaw_difference, -maximum_change, maximum_change))
    yaw = float((last_yaw + yaw_change + np.pi) % (2.0 * np.pi) - np.pi)
    return yaw, yaw_change / dt


class RealtimeYOPORunner:
    """Asynchronous pure-YOPO receding-horizon runner.

    Sensor, planner, controller and actuator workers are decoupled so the
    controller never waits for network inference or video encoding.
    """

    def __init__(
        self,
        simulator: SimulatorAdapter,
        planner,
        safety_filter: SafetyFilter,
        goal_nwu: np.ndarray,
        config: dict,
        world_model_runtime=None,
        route_nwu: np.ndarray | None = None,
    ) -> None:
        self.simulator = simulator
        self.planner = planner
        self.safety = safety_filter
        self.goal = np.asarray(goal_nwu, dtype=np.float64)
        self.config = dict(config)
        route_points = np.asarray(
            route_nwu if route_nwu is not None else np.stack([self.goal, self.goal + np.array([1.0, 0.0, 0.0])]),
            dtype=np.float64,
        )
        if route_nwu is None:
            # The first state is not available at construction time.  The
            # caller should pass route_nwu for realtime runs; this fallback is
            # retained for narrow unit tests that only exercise mailboxes.
            route_points = np.stack([self.goal - np.array([1.0, 0.0, 0.0]), self.goal])
        self.route = PolylineRoute(
            route_points,
            normal_lookahead_m=float(config.get("route_minimum_lookahead_m", 12.0)),
            turn_lookahead_m=float(config.get("route_turn_lookahead_m", 12.0)),
            turn_threshold_degrees=float(config.get("route_turn_threshold_degrees", 30.0)),
            turn_awareness_distance_m=float(config.get("route_turn_awareness_distance_m", 20.0)),
            lookahead_speed_gain_s=float(config.get("route_lookahead_speed_gain_s", 1.5)),
            maximum_lookahead_m=float(config.get("route_maximum_lookahead_m", 22.0)),
        )
        self._route_lock = threading.Lock()
        self.world_model_runtime = world_model_runtime
        self.method = "yopo" if world_model_runtime is None else f"yopo_{world_model_runtime.family}"
        self.sensor_hz = float(config.get("sensor_hz", 15.0))
        self.state_hz = float(config.get("state_hz", 50.0))
        self.planner_hz = float(config.get("planner_hz", 10.0))
        self.control_hz = float(config.get("control_hz", 50.0))
        self.sensor_stale_s = float(config.get("sensor_stale_s", 0.35))
        self.plan_stale_s = float(config.get("plan_stale_s", 0.75))
        self.position_kp = float(config.get("position_kp", 1.2))
        self.yaw_kp = float(config.get("yaw_kp", 2.0))
        self.yaw_deadband = math.radians(float(config.get("yaw_deadband_degrees", 2.0)))
        self.yaw_alpha = float(np.clip(config.get("yaw_smoothing_alpha", 0.2), 0.0, 1.0))
        self.ttc_stop_s = float(config.get("ttc_stop_s", 0.8))
        self.velocity_kd = float(config.get("velocity_kd", 0.8))
        self.acceleration_feedforward_s = float(config.get("acceleration_feedforward_s", 0.20))
        self.inline_actuation = bool(config.get("inline_actuation", False))
        self.depth_roi = np.asarray(config.get("depth_roi_fraction", [0.30, 0.72, 0.38, 0.62]), dtype=float)
        self.sensor_box: LatestValue[TimestampedSensorFrame] = LatestValue()
        self.state_box: LatestValue[VehicleState] = LatestValue()
        self.plan_box: LatestValue[TrajectoryPlan] = LatestValue()
        self.command_box: LatestValue[tuple[int, np.ndarray, float]] = LatestValue()
        self.stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started_monotonic = 0.0
        self._stopped_monotonic = 0.0
        self._errors: list[dict[str, str]] = []
        self._data_lock = threading.Lock()
        self.sensor_records: list[dict] = []
        self.plan_records: list[dict] = []
        self.control_records: list[ControlSample] = []
        self.actuation_records: list[dict[str, float | int]] = []
        self._controller_stop_reason = "running"
        self._planner_status_lock = threading.Lock()
        self._planner_stage = "not_started"
        self._planner_heartbeat_monotonic_s = 0.0
        self._last_plan_monotonic_s = 0.0
        self.collision_baseline_timestamp = int(config.get("collision_baseline_timestamp", 0))

    def _fail(self, worker: str, exc: BaseException) -> None:
        with self._data_lock:
            self._errors.append({"worker": worker, "error": repr(exc)})
        if worker == "planner":
            self._set_planner_status("failed")
        self.stop_event.set()

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("realtime runner is already started")
        self.stop_event.clear()
        # Camera callbacks can copy several megabytes per frame.  A shorter
        # interpreter switch interval prevents those callbacks from starving
        # the 20 ms reference publisher on Windows.
        if bool(self.config.get("low_latency_thread_scheduling", True)):
            sys.setswitchinterval(0.001)
        self._started_monotonic = time.perf_counter()
        self._set_planner_status("starting")
        self._stopped_monotonic = 0.0
        workers = [("state", self._state_loop), ("sensor", self._sensor_loop), ("planner", self._planner_loop)]
        workers.append(("controller", self._control_loop))
        if not self.inline_actuation:
            workers.append(("actuator", self._actuator_loop))
        for name, target in workers:
            thread = threading.Thread(target=target, name=f"yopo-{name}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def stop(self, reason: str = "requested") -> None:
        self._controller_stop_reason = reason
        self.stop_event.set()
        self._stopped_monotonic = time.perf_counter()
        for thread in self._threads:
            thread.join(timeout=10.0)
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            self._errors.append({"worker": "shutdown", "error": f"threads did not stop: {alive}"})
        self._threads.clear()

    @property
    def errors(self) -> list[dict[str, str]]:
        with self._data_lock:
            return list(self._errors)

    def begin_flight_interval(self) -> None:
        """Discard controller-process warm-up and start the accepted run clock."""
        with self._data_lock:
            self.sensor_records.clear()
            self.plan_records.clear()
            self.control_records.clear()
            self.actuation_records.clear()
        self._started_monotonic = time.perf_counter()
        self._stopped_monotonic = 0.0
        with self._planner_status_lock:
            self._last_plan_monotonic_s = 0.0
        if self.world_model_runtime is not None:
            self.world_model_runtime.reset()

    def _set_planner_status(self, stage: str, *, published: bool = False) -> None:
        now = time.perf_counter()
        with self._planner_status_lock:
            self._planner_stage = str(stage)
            self._planner_heartbeat_monotonic_s = now
            if published:
                self._last_plan_monotonic_s = now

    def planner_status(self, now_monotonic_s: float | None = None) -> dict[str, float | str | bool | None]:
        now = time.perf_counter() if now_monotonic_s is None else float(now_monotonic_s)
        with self._planner_status_lock:
            heartbeat = self._planner_heartbeat_monotonic_s
            last_plan = self._last_plan_monotonic_s
            stage = self._planner_stage
        return {
            "planner_stage": stage,
            "planner_thread_alive": any(thread.name == "yopo-planner" and thread.is_alive() for thread in self._threads),
            "planner_heartbeat_age_s": None if heartbeat <= 0.0 else max(now - heartbeat, 0.0),
            "plan_gap_s": None if last_plan <= 0.0 else max(now - last_plan, 0.0),
            "planner_heartbeat_monotonic_s": heartbeat,
            "last_plan_monotonic_s": last_plan,
        }

    def observe_route(self, state: VehicleState) -> RouteProjection:
        speed = float(np.linalg.norm(state.linear_velocity[:2]))
        with self._route_lock:
            return self.route.observe(state.position, speed)

    def _sensor_loop(self) -> None:
        period = 1.0 / max(self.sensor_hz, 1e-6)
        sequence = 0
        next_tick = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                sensor = self.simulator.get_depth()
                state, _ = self.state_box.get()
                if state is None:
                    state, _ = self.state_box.wait_newer(0, 0.1)
                if state is None:
                    continue
                received = time.perf_counter()
                frame = TimestampedSensorFrame(sequence, received, sensor, state)
                minimum_valid_depth = float(self.config.get("minimum_valid_depth_m", 0.8))
                planning_valid = sensor.valid_mask & np.isfinite(sensor.depth_m) & (sensor.depth_m >= minimum_valid_depth)
                valid_fraction = float(np.mean(planning_valid))
                with self._data_lock:
                    self.sensor_records.append({
                        "sequence_id": sequence,
                        "monotonic_s": received,
                        "sensor_timestamp": float(sensor.timestamp),
                        "state_timestamp": float(state.timestamp),
                        "synchronization_error_s": frame.synchronization_error_s,
                        "valid_fraction": valid_fraction,
                        "accepted": valid_fraction > 0.0,
                    })
                # A renderer can emit a transient blank depth image during a
                # hitch. Never feed it to YOPO; retaining the last accepted
                # frame lets the existing 0.35 s stale guard hover safely if
                # valid depth does not recover.
                if valid_fraction > 0.0:
                    self.sensor_box.publish(frame)
                sequence += 1
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                elif delay < -period:
                    next_tick = time.perf_counter()
        except BaseException as exc:
            self._fail("sensor", exc)

    def _state_loop(self) -> None:
        period = 1.0 / max(self.state_hz, 1e-6)
        next_tick = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                self.state_box.publish(self.simulator.get_kinematics())
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                elif delay < -period:
                    next_tick = time.perf_counter()
        except BaseException as exc:
            self._fail("state", exc)

    def _planner_loop(self) -> None:
        minimum_period = 1.0 / max(self.planner_hz, 1e-6)
        last_sensor_version = 0
        last_planned_at = -1e9
        sequence = 0
        try:
            while not self.stop_event.is_set():
                self._set_planner_status("waiting_for_sensor")
                frame, version = self.sensor_box.wait_newer(last_sensor_version, 0.1)
                if frame is None or version <= last_sensor_version:
                    continue
                now = time.perf_counter()
                if now - last_planned_at < minimum_period:
                    continue
                last_sensor_version = version
                route = self.observe_route(frame.state)
                local_goal = route.local_goal_nwu
                self._set_planner_status("yopo_inference")
                started = time.perf_counter()
                candidates = tuple(self.planner.plan(PlanningContext(frame.sensor, frame.state, local_goal)))
                yopo_latency_ms = (time.perf_counter() - started) * 1000.0
                if len(candidates) != 15:
                    raise RuntimeError(f"official YOPO must expose 15 candidates, got {len(candidates)}")
                raw_selected = int(np.argmin([candidate.yopo_cost for candidate in candidates]))
                selected = raw_selected
                model_latency_ms = 0.0
                model_metadata: dict[str, object] = {
                    "raw_selected_index": raw_selected, "used_fallback": False, "fallback_reason": "none",
                }
                selection_method = "yopo_raw_argmin"
                if self.world_model_runtime is not None:
                    self._set_planner_status("world_model_rerank")
                    decision, predictions, model_latency_ms = self.world_model_runtime.rank(frame, candidates, local_goal)
                    if decision.selected_index >= 0 and not decision.used_fallback:
                        selected = int(decision.selected_index)
                    else:
                        selected = raw_selected
                    selection_method = f"{self.world_model_runtime.family}_rerank"
                    model_metadata.update({
                        "used_fallback": bool(decision.used_fallback),
                        "fallback_reason": str(decision.reason) if decision.used_fallback else "none",
                        "reranked": selected != raw_selected,
                        "total_scores": np.asarray(decision.total_scores, dtype=np.float32),
                        "collision_probability": np.asarray([p.collision_probability for p in predictions], dtype=np.float32),
                        "failure_probability": np.asarray([p.failure_probability for p in predictions], dtype=np.float32),
                        "minimum_clearance": np.asarray([p.minimum_clearance for p in predictions], dtype=np.float32),
                        "goal_progress": np.asarray([p.goal_progress for p in predictions], dtype=np.float32),
                        "uncertainty": np.asarray([p.uncertainty for p in predictions], dtype=np.float32),
                    })
                latency_ms = (time.perf_counter() - started) * 1000.0
                created = time.perf_counter()
                plan = TrajectoryPlan(
                    sequence_id=sequence, sensor_sequence_id=frame.sequence_id,
                    created_monotonic_s=created, valid_until_monotonic_s=created + self.plan_stale_s,
                    candidates=candidates, selected_index=selected, planner_latency_ms=latency_ms,
                    local_goal_nwu=local_goal, selection_method=selection_method, metadata=model_metadata,
                )
                self.plan_box.publish(plan)
                self.simulator.publish_planner_visualization(
                    candidates,
                    selected_index=selected,
                    decision_sequence=sequence,
                    metadata={
                        **model_metadata,
                        "selection_method": selection_method,
                        "planner_latency_ms": latency_ms,
                    },
                )
                depth_mm = np.clip(
                    np.nan_to_num(frame.sensor.depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0,
                    0.0, 65535.0,
                ).astype(np.uint16)
                with self._data_lock:
                    self.plan_records.append({
                        "sequence_id": sequence, "sensor_sequence_id": frame.sequence_id,
                        "elapsed_s": created - self._started_monotonic, "planner_latency_ms": latency_ms,
                        "yopo_latency_ms": yopo_latency_ms, "model_latency_ms": model_latency_ms,
                        "planning_position_nwu": frame.state.position.copy(),
                        "planning_velocity_nwu": frame.state.linear_velocity.copy(),
                        "local_goal_nwu": local_goal.copy(),
                        "route_progress_s_m": route.progress_m,
                        "cross_track_error_m": route.cross_track_error_m,
                        "route_segment_index": route.segment_index,
                        "route_remaining_m": route.remaining_m,
                        "route_lookahead_m": route.lookahead_m,
                        "distance_to_turn_m": np.nan if route.distance_to_turn_m is None else route.distance_to_turn_m,
                        "agl_m": float(frame.state.position[2] - float(self.config.get("ground_altitude_nwu", 0.0))),
                        "planner_heartbeat": sequence,
                        "planner_stage": "published",
                        "camera_intrinsics": None if frame.sensor.camera_intrinsics is None else frame.sensor.camera_intrinsics.copy(),
                        "camera_pose_nwu": None if frame.sensor.camera_pose_nwu is None else frame.sensor.camera_pose_nwu.copy(),
                        "depth_mm": depth_mm, "selected_index": selected,
                        "raw_selected_index": raw_selected, "selection_method": selection_method,
                        **model_metadata,
                        "costs": np.asarray([candidate.yopo_cost for candidate in candidates], dtype=np.float32),
                        "candidate_positions_nwu": np.stack([candidate.positions for candidate in candidates]),
                        "candidate_velocities_nwu": np.stack([candidate.velocities for candidate in candidates]),
                    })
                last_planned_at = created
                self._set_planner_status("published", published=True)
                sequence += 1
        except BaseException as exc:
            self._fail("planner", exc)

    def _ttc(self, sensor, command_velocity: np.ndarray) -> float:
        speed = float(np.linalg.norm(command_velocity[:2]))
        if speed < 0.1:
            return float("inf")
        height, width = sensor.depth_m.shape
        top, bottom, left, right = self.depth_roi
        y0, y1 = int(top * height), max(int(bottom * height), int(top * height) + 1)
        x0, x1 = int(left * width), max(int(right * width), int(left * width) + 1)
        roi = sensor.depth_m[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if not valid.size:
            return 0.0
        return float(np.percentile(valid, 2.0)) / speed

    def _control_loop(self) -> None:
        if bool(self.config.get("windows_high_priority_control_thread", True)) and sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST
            except Exception:
                pass
        dt = 1.0 / max(self.control_hz, 1e-6)
        sequence = 0
        previous_plan_id = -1
        control_time_s = 0.0
        last_yaw: float | None = None
        next_tick = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                frame, _ = self.sensor_box.get()
                plan, _ = self.plan_box.get()
                if frame is None:
                    # No observation exists yet; wait briefly without issuing
                    # a competing state RPC from the 50 Hz command thread.
                    self.stop_event.wait(min(dt, 0.01))
                    next_tick = time.perf_counter()
                    continue
                observation_age = max(0.0, now - frame.received_monotonic_s)
                observed = frame.state
                state = VehicleState(
                    timestamp=observed.timestamp + observation_age,
                    position=observed.position + observed.linear_velocity * observation_age,
                    orientation_xyzw=observed.orientation_xyzw.copy(),
                    linear_velocity=observed.linear_velocity.copy(),
                    angular_velocity=observed.angular_velocity.copy(),
                    linear_acceleration=observed.linear_acceleration.copy(),
                    vehicle_name=observed.vehicle_name,
                )
                reference_position = state.position.copy()
                reference_velocity = np.zeros(3, dtype=np.float64)
                reference_acceleration = np.zeros(3, dtype=np.float64)
                mode = "hover"
                reasons: list[str] = []
                plan_id = -1
                desired_yaw = float(Rotation.from_quat(state.orientation_xyzw).as_euler("ZYX")[0])
                yaw_rate = 0.0
                command = np.zeros(3, dtype=np.float64)
                if frame is None or now - frame.received_monotonic_s > self.sensor_stale_s:
                    reasons.append("sensor_stale")
                elif plan is None or now > plan.valid_until_monotonic_s:
                    reasons.append("plan_stale")
                else:
                    plan_id = plan.sequence_id
                    if plan_id != previous_plan_id:
                        previous_plan_id = plan_id
                        control_time_s = 0.0
                    # Official control_pub increments a fixed 0.02 s timer and
                    # hard-switches to the newly solved polynomial under a lock.
                    control_time_s = min(control_time_s + dt, float(plan.selected.duration))
                    reference_position, reference_velocity, reference_acceleration = _interp_trajectory(
                        plan.selected, control_time_s
                    )
                    if hasattr(self.planner, "update_control_reference"):
                        self.planner.update_control_reference(
                            reference_position, reference_velocity, reference_acceleration
                        )
                    # Translate YOPO's P/V/A reference into the adapter's
                    # velocity command with feedback and acceleration feed-forward.
                    desired = (
                        reference_velocity
                        + self.position_kp * (reference_position - state.position)
                        + self.velocity_kd * (reference_velocity - state.linear_velocity)
                        + self.acceleration_feedforward_s * reference_acceleration
                    )
                    current_yaw = float(Rotation.from_quat(state.orientation_xyzw).as_euler("ZYX")[0])
                    if last_yaw is None:
                        last_yaw = current_yaw
                    desired_yaw, yaw_rate = _calculate_yaw_official(
                        reference_velocity, plan.local_goal_nwu - reference_position, last_yaw, dt
                    )
                    last_yaw = desired_yaw
                    ttc = self._ttc(frame.sensor, desired)
                    predicted_risk = 1.0 if ttc < self.ttc_stop_s else 0.0
                    if predicted_risk:
                        reasons.append("ttc_emergency")
                    # Collision polling stays in the monitor thread. Querying it
                    # here would serialize a second RPC into every 20 ms cycle.
                    filtered = self.safety.apply(desired, yaw_rate, state, frame.sensor, dt, predicted_risk, False)
                    command = np.asarray(filtered.velocity, dtype=np.float64)
                    yaw_rate = float(filtered.yaw_rate)
                    mode = filtered.mode
                    reasons.extend(filtered.reasons)
                if not self.inline_actuation:
                    self.command_box.publish((sequence, command.copy(), yaw_rate))
                sample = ControlSample(
                    sequence_id=sequence, monotonic_s=now, plan_sequence_id=plan_id,
                    reference_position_nwu=reference_position, reference_velocity_nwu=reference_velocity,
                    reference_acceleration_nwu=reference_acceleration,
                    actual_position_nwu=state.position, actual_velocity_nwu=state.linear_velocity,
                    command_velocity_nwu=command, desired_yaw_rad=desired_yaw,
                    command_yaw_rate_rps=yaw_rate, mode=mode, reasons=tuple(dict.fromkeys(reasons)),
                )
                with self._data_lock:
                    self.control_records.append(sample)
                if self.inline_actuation:
                    actuation_started = time.perf_counter()
                    self.simulator.execute_velocity_command(command, yaw_rate, dt)
                    actuation_finished = time.perf_counter()
                    with self._data_lock:
                        self.actuation_records.append({
                            "control_sequence_id": int(sequence),
                            "started_monotonic_s": actuation_started,
                            "finished_monotonic_s": actuation_finished,
                            "rpc_latency_ms": (actuation_finished - actuation_started) * 1000.0,
                        })
                sequence += 1
                next_tick += dt
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                elif delay < -dt:
                    next_tick = time.perf_counter()
        except BaseException as exc:
            self._fail("controller", exc)

    def _actuator_loop(self) -> None:
        """Forward the newest 50 Hz setpoint through the simulator adapter."""
        version = 0
        try:
            while not self.stop_event.is_set():
                command, new_version = self.command_box.wait_newer(version, 0.05)
                if command is None or new_version <= version:
                    continue
                sequence_id, velocity, yaw_rate = command
                started = time.perf_counter()
                self.simulator.execute_velocity_command(velocity, yaw_rate, 1.0 / max(self.control_hz, 1e-6))
                finished = time.perf_counter()
                with self._data_lock:
                    self.actuation_records.append({
                        "control_sequence_id": int(sequence_id), "started_monotonic_s": started,
                        "finished_monotonic_s": finished, "rpc_latency_ms": (finished - started) * 1000.0,
                    })
                version = new_version
        except BaseException as exc:
            self._fail("actuator", exc)

    def metrics(self) -> dict[str, object]:
        with self._data_lock:
            controls = list(self.control_records)
            plans = list(self.plan_records)
            sensors = list(self.sensor_records)
            actuations = list(self.actuation_records)
        stopped = self._stopped_monotonic or time.perf_counter()
        elapsed = max(stopped - self._started_monotonic, 1e-9)
        if controls:
            positions = np.stack([item.actual_position_nwu for item in controls]).astype(np.float64)
            velocities = np.stack([item.actual_velocity_nwu for item in controls]).astype(np.float64)
            commands = np.stack([item.command_velocity_nwu for item in controls]).astype(np.float64)
            times = np.asarray([item.monotonic_s for item in controls], dtype=np.float64)
            speed = np.linalg.norm(velocities, axis=1)
            audit_route = PolylineRoute(
                self.route.waypoints,
                normal_lookahead_m=self.route.normal_lookahead_m,
                turn_lookahead_m=self.route.turn_lookahead_m,
                turn_threshold_degrees=math.degrees(self.route.turn_threshold),
                turn_awareness_distance_m=self.route.turn_awareness_distance_m,
                lookahead_speed_gain_s=self.route.lookahead_speed_gain_s,
                maximum_lookahead_m=self.route.maximum_lookahead_m,
            )
            route_audit = [audit_route.observe(position, float(current_speed)) for position, current_speed in zip(positions, speed)]
            progress = np.asarray([item.progress_m for item in route_audit], dtype=np.float64)
            regression = np.clip(-np.diff(progress), 0.0, None)
            lateral = np.asarray([item.cross_track_error_m for item in route_audit], dtype=np.float64)
            acceleration = np.linalg.norm(np.diff(commands, axis=0) / (1.0 / max(self.control_hz, 1e-6)), axis=1)
            yaw_rates = np.asarray([item.command_yaw_rate_rps for item in controls])
        else:
            positions = np.empty((0, 3)); speed = progress = lateral = acceleration = yaw_rates = np.asarray([])
            regression = np.asarray([])
        plan_latency = np.asarray([item["planner_latency_ms"] for item in plans], dtype=float)
        yopo_latency = np.asarray([item.get("yopo_latency_ms", item["planner_latency_ms"]) for item in plans], dtype=float)
        model_latency = np.asarray([item.get("model_latency_ms", 0.0) for item in plans], dtype=float)
        actuation_latency = np.asarray([item["rpc_latency_ms"] for item in actuations], dtype=float)
        plan_times = np.asarray([item["elapsed_s"] for item in plans], dtype=np.float64)
        plan_gaps = np.diff(np.r_[0.0, plan_times, elapsed]) if plan_times.size else np.asarray([elapsed])
        return {
            "elapsed_s": elapsed,
            "sensor_frames": len(sensors), "sensor_rate_hz": len(sensors) / elapsed,
            "planner_cycles": len(plans), "planner_rate_hz": len(plans) / elapsed,
            "maximum_plan_gap_s": float(np.max(plan_gaps)),
            "control_cycles": len(controls), "control_rate_hz": len(controls) / elapsed,
            "actuator_updates": len(actuations), "actuator_rate_hz": len(actuations) / elapsed,
            "actuator_rpc_latency_p95_ms": float(np.percentile(actuation_latency, 95)) if actuation_latency.size else None,
            "planner_latency_p50_ms": float(np.percentile(plan_latency, 50)) if plan_latency.size else None,
            "planner_latency_p95_ms": float(np.percentile(plan_latency, 95)) if plan_latency.size else None,
            "yopo_latency_p95_ms": float(np.percentile(yopo_latency, 95)) if yopo_latency.size else None,
            "model_latency_p95_ms": float(np.percentile(model_latency, 95)) if model_latency.size else None,
            "rerank_rate": float(np.mean([bool(item.get("reranked", False)) for item in plans])) if plans else 0.0,
            "fallback_rate": float(np.mean([bool(item.get("used_fallback", False)) for item in plans])) if plans else 0.0,
            "method": self.method,
            "speed_mean_mps": float(np.mean(speed)) if speed.size else 0.0,
            "speed_p50_mps": float(np.percentile(speed, 50)) if speed.size else 0.0,
            "speed_p95_mps": float(np.percentile(speed, 95)) if speed.size else 0.0,
            "path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0,
            "net_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])) if len(positions) > 1 else 0.0,
            "estimated_state_regression_m": float(regression.sum()) if regression.size else 0.0,
            "total_regression_m": float(regression.sum()) if regression.size else 0.0,
            "backtracking_fraction": float(np.mean(regression > 0.01)) if regression.size else 0.0,
            "lateral_p95_m": float(np.percentile(lateral, 95)) if lateral.size else 0.0,
            "command_acceleration_p95_mps2": float(np.percentile(acceleration, 95)) if acceleration.size else 0.0,
            "yaw_rate_p95_rps": float(np.percentile(np.abs(yaw_rates), 95)) if yaw_rates.size else 0.0,
            "errors": self.errors, "stop_reason": self._controller_stop_reason,
        }

    def save_telemetry(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._data_lock:
            plans = list(self.plan_records)
            controls = list(self.control_records)
            actuations = list(self.actuation_records)
        with h5py.File(output_path, "w") as handle:
            handle.attrs["schema"] = "uav-wm-nav-yopo-realtime-v1"
            handle.attrs["goal_nwu"] = self.goal
            handle.attrs["route_nwu"] = self.route.waypoints
            handle.attrs["route_id"] = self.route.route_id
            handle.attrs["config_json"] = json.dumps(self.config)
            plan_group = handle.create_group("plans")
            if plans:
                for name in (
                    "sequence_id", "sensor_sequence_id", "elapsed_s", "planner_latency_ms",
                    "yopo_latency_ms", "model_latency_ms", "raw_selected_index",
                    "planning_position_nwu", "planning_velocity_nwu", "local_goal_nwu",
                    "route_progress_s_m", "cross_track_error_m", "route_segment_index",
                    "route_remaining_m", "route_lookahead_m", "distance_to_turn_m", "agl_m",
                    "planner_heartbeat", "selected_index", "costs",
                    "candidate_positions_nwu", "candidate_velocities_nwu",
                ):
                    plan_group.create_dataset(name, data=np.stack([np.asarray(item[name]) for item in plans]), compression="gzip")
                for name in (
                    "total_scores", "collision_probability", "failure_probability", "minimum_clearance",
                    "goal_progress", "uncertainty",
                ):
                    if name in plans[0]:
                        plan_group.create_dataset(name, data=np.stack([np.asarray(item[name]) for item in plans]), compression="gzip")
                string_dtype = h5py.string_dtype(encoding="utf-8")
                plan_group.create_dataset("selection_method", data=np.asarray([item["selection_method"] for item in plans], dtype=string_dtype))
                plan_group.create_dataset("planner_stage", data=np.asarray([item["planner_stage"] for item in plans], dtype=string_dtype))
                plan_group.create_dataset("fallback_reason", data=np.asarray([item.get("fallback_reason", "none") for item in plans], dtype=string_dtype))
                plan_group.create_dataset("used_fallback", data=np.asarray([item.get("used_fallback", False) for item in plans], dtype=np.uint8))
                plan_group.create_dataset(
                    "depth_mm", data=np.stack([item["depth_mm"] for item in plans]),
                    compression="gzip", shuffle=True, chunks=(1, *plans[0]["depth_mm"].shape),
                )
                if plans[0]["camera_intrinsics"] is not None:
                    plan_group.create_dataset("camera_intrinsics", data=np.stack([item["camera_intrinsics"] for item in plans]))
                if plans[0]["camera_pose_nwu"] is not None:
                    plan_group.create_dataset("camera_pose_nwu", data=np.stack([item["camera_pose_nwu"] for item in plans]))
            control_group = handle.create_group("control")
            if controls:
                for name in (
                    "sequence_id", "monotonic_s", "plan_sequence_id", "reference_position_nwu",
                    "reference_velocity_nwu", "reference_acceleration_nwu", "actual_position_nwu", "actual_velocity_nwu",
                    "command_velocity_nwu", "desired_yaw_rad", "command_yaw_rate_rps",
                ):
                    control_group.create_dataset(name, data=np.stack([np.asarray(getattr(item, name)) for item in controls]))
                string_dtype = h5py.string_dtype(encoding="utf-8")
                control_group.create_dataset("mode", data=np.asarray([item.mode for item in controls], dtype=string_dtype))
                control_group.create_dataset(
                    "reasons", data=np.asarray(["|".join(item.reasons) for item in controls], dtype=string_dtype)
                )
            actuation_group = handle.create_group("actuation")
            if actuations:
                for name in ("control_sequence_id", "started_monotonic_s", "finished_monotonic_s", "rpc_latency_ms"):
                    actuation_group.create_dataset(name, data=np.asarray([item[name] for item in actuations]))
