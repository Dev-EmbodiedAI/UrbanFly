"""统一 action contract 上的起终点导航 policy + 轻量预测式 World Model。

这是可执行的工程 baseline：Agent 给出唯一目标分配，policy 生成局部候选，
World Model 用运动学、深度和机间预测距离进行一步前视重排。它不冒充已训练
的 Helsinki latent model，也不具备公开 Swarm benchmark 资格。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import DigitalTwinMission, DigitalTwinObservation


@dataclass(frozen=True, slots=True)
class WorldModelBatchDecision:
    action: np.ndarray
    selected_candidate: np.ndarray
    predicted_next_position_enu_m: np.ndarray
    predicted_clearance_m: np.ndarray
    predicted_minimum_separation_m: np.ndarray
    candidate_scores: np.ndarray


class GoalConditionedWorldModelPolicy:
    """动态 2–8 机的预测式导航基线，输出标准 ``[N,5]`` action。"""

    def __init__(self, *, maximum_speed_mps: float = 3.0, prediction_dt_s: float = 0.35) -> None:
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.prediction_dt_s = float(prediction_dt_s)
        self.assigned_goals: np.ndarray | None = None
        self.cruise_altitudes_m: np.ndarray | None = None
        self.maximum_altitudes_m: np.ndarray | None = None
        self.previous_timestamp_s: float | None = None
        self.previous_action: np.ndarray | None = None
        self.escape_target_altitudes_m: np.ndarray | None = None
        self.escape_origins_xy_m: np.ndarray | None = None
        self.escape_climb_used: np.ndarray | None = None
        self.scan_target_yaw_rad: np.ndarray | None = None
        self.best_horizontal_goal_distance_m: np.ndarray | None = None
        self.steps_without_progress: np.ndarray | None = None
        self.progress_escape_used: np.ndarray | None = None
        self.decisions = 0

    def reset(self, mission: DigitalTwinMission) -> np.ndarray:
        distances = np.linalg.norm(
            mission.starts_enu_m[:, None, :] - mission.goals_enu_m[None, :, :], axis=2
        )
        rows, columns = linear_sum_assignment(distances)
        goals = np.empty_like(mission.goals_enu_m)
        goals[rows] = mission.goals_enu_m[columns]
        self.assigned_goals = goals
        self.cruise_altitudes_m = np.maximum(
            mission.starts_enu_m[:, 2] + 1.5,
            goals[:, 2] + 2.0,
        )
        self.maximum_altitudes_m = np.maximum(
            mission.starts_enu_m[:, 2], goals[:, 2]
        ) + 15.0
        self.previous_timestamp_s = None
        self.previous_action = np.zeros((len(goals), 5), dtype=np.float32)
        self.escape_target_altitudes_m = np.full(len(goals), np.nan, dtype=np.float64)
        self.escape_origins_xy_m = mission.starts_enu_m[:, :2].astype(np.float64).copy()
        self.escape_climb_used = np.zeros(len(goals), dtype=bool)
        self.scan_target_yaw_rad = np.full(len(goals), np.nan, dtype=np.float64)
        self.best_horizontal_goal_distance_m = np.linalg.norm(
            goals[:, :2] - mission.starts_enu_m[:, :2], axis=1
        ).astype(np.float64)
        self.steps_without_progress = np.zeros(len(goals), dtype=np.int64)
        self.progress_escape_used = np.zeros(len(goals), dtype=bool)
        self.decisions = 0
        return goals.copy()

    def act(self, observation: DigitalTwinObservation) -> WorldModelBatchDecision:
        if (
            self.assigned_goals is None
            or self.cruise_altitudes_m is None
            or self.maximum_altitudes_m is None
            or self.escape_target_altitudes_m is None
            or self.escape_origins_xy_m is None
            or self.escape_climb_used is None
            or self.scan_target_yaw_rad is None
            or self.best_horizontal_goal_distance_m is None
            or self.steps_without_progress is None
            or self.progress_escape_used is None
        ):
            raise RuntimeError("World Model policy must be reset with a mission")
        if observation.drone_count != len(self.assigned_goals):
            raise ValueError("observation drone count changed inside an episode")
        if self.previous_timestamp_s is not None and observation.timestamp_s <= self.previous_timestamp_s:
            raise RuntimeError("World Model received stale observation")
        self.previous_timestamp_s = observation.timestamp_s
        positions = observation.state[:, 0:3].astype(np.float64)
        yaw = observation.state[:, 5].astype(np.float64)
        n = len(positions)
        # 所有平移候选都必须落在当前相机的可观测视场内。candidate 0 是
        # 朝目标方向（裁剪到视场边缘），其余候选用于沿障碍两侧绕行。
        view_offsets = np.asarray([0.0, 0.34, -0.34, 0.68, -0.68], dtype=np.float64)
        candidates = np.zeros((n, len(view_offsets), 5), dtype=np.float64)
        predicted = np.zeros((n, len(view_offsets), 3), dtype=np.float64)
        # 视场之外是“未知”，不能当成无障碍。0.75 m 会迫使 policy 先原地转向，
        # 等下一帧 Depth 真正覆盖目标航向后再平移。
        clearance = np.full((n, len(view_offsets)), 0.75, dtype=np.float64)
        goal_distance = np.linalg.norm(self.assigned_goals - positions, axis=1)
        horizontal_goal_distance = np.linalg.norm(
            self.assigned_goals[:, :2] - positions[:, :2], axis=1
        )
        for drone in range(n):
            if (
                horizontal_goal_distance[drone]
                < self.best_horizontal_goal_distance_m[drone] - 0.25
            ):
                self.best_horizontal_goal_distance_m[drone] = horizontal_goal_distance[drone]
                self.steps_without_progress[drone] = 0
            else:
                self.steps_without_progress[drone] += 1
        phases: list[str] = []
        phase_targets: list[np.ndarray] = []

        for drone in range(n):
            goal = self.assigned_goals[drone]
            goal_delta = goal - positions[drone]
            horizontal = float(np.linalg.norm(goal_delta[:2]))
            altitude_distance_m = float(np.clip(observation.state[drone, 137], 0.0, 1.0) * 20.0)
            if altitude_distance_m < 1.0 and horizontal > 1.2:
                phase = "takeoff"
                target = positions[drone].copy()
                target[2] = max(self.cruise_altitudes_m[drone], positions[drone, 2] + 1.2)
                speed = 0.18
            elif horizontal > 30.0:
                phase = "cruise"
                target = goal.copy()
                target[2] = self.cruise_altitudes_m[drone]
                speed = float(np.clip(horizontal / 20.0, 0.35, 0.85))
            elif horizontal > 0.35:
                phase = "approach"
                target = goal.copy()
                target[2] += 0.4
                # The native contract requires 0.5 s of stable platform contact.
                # Preserve enough episode budget for that physical landing gate:
                # a 1.02 m/s minimum approach (old 0.34 command) can reach the
                # platform geometrically at the final frame yet still TIMEOUT
                # before the contact dwell completes. 1.38 m/s remains well
                # below the 3 m/s environment limit and is still reduced by the
                # depth/turn safety caps below whenever clearance is constrained.
                speed = float(np.clip(horizontal / 12.0, 0.46, 0.68))
            else:
                phase = "land"
                target = goal.copy()
                # Native landing accepts at most 0.5 m/s vertical velocity and
                # then requires 0.5 s stable contact. Keep the commanded norm
                # below that hard gate (0.165 * 3 = 0.495 m/s) while avoiding
                # an asymptotically slow final 20 cm that contacts too late.
                speed = float(
                    np.clip(
                        abs(target[2] - positions[drone, 2]) / 1.25,
                        0.08,
                        0.165,
                    )
                )
            phases.append(phase)
            phase_targets.append(target.copy())
            delta = target - positions[drone]
            target_distance = float(np.linalg.norm(delta))
            direct_yaw = math.atan2(goal_delta[1], goal_delta[0]) if horizontal > 1e-6 else yaw[drone]
            vertical_limit = 0.95 if phase == "approach" else 0.45
            vertical = float(
                np.clip(
                    delta[2] / max(target_distance, 1e-6),
                    -vertical_limit,
                    vertical_limit,
                )
            )
            target_heading_error = (direct_yaw - yaw[drone] + math.pi) % (2.0 * math.pi) - math.pi
            visible_target_heading = yaw[drone] + float(
                np.clip(target_heading_error, -0.68, 0.68)
            )
            for candidate, view_offset in enumerate(view_offsets):
                if phase == "takeoff":
                    heading = yaw[drone]
                    direction = np.asarray([0.0, 0.0, 1.0])
                elif phase == "land":
                    heading = direct_yaw
                    direction = delta.copy()
                else:
                    heading = (
                        visible_target_heading
                        if candidate == 0
                        else yaw[drone] + view_offset
                    )
                    heading = (heading + math.pi) % (2.0 * math.pi) - math.pi
                    direction = np.asarray([math.cos(heading), math.sin(heading), vertical])
                direction /= max(float(np.linalg.norm(direction)), 1e-9)
                candidates[drone, candidate, :3] = direction
                candidates[drone, candidate, 3] = speed
                candidates[drone, candidate, 4] = float(np.clip(heading / math.pi, -1.0, 1.0))
                predicted[drone, candidate] = (
                    positions[drone]
                    + direction * speed * self.maximum_speed_mps * self.prediction_dt_s
                )
                relative = (heading - yaw[drone] + math.pi) % (2 * math.pi) - math.pi
                if phase in {"takeoff", "land"}:
                    # 起降阶段主要由向下高度与平台接触约束；前视 Depth 不用于
                    # 给垂直运动制造虚假的横向障碍。
                    clearance[drone, candidate] = 20.0
                elif abs(relative) <= math.pi / 4:
                    # PyBullet view matrix 的屏幕右轴是 body -Y；ENU 中正相对
                    # 航向（逆时针/左转）因此落在图像左侧，列映射必须取反。
                    column = int(
                        np.clip((-relative / (math.pi / 2) + 0.5) * 127, 0, 127)
                    )
                    # 用接近机体投影宽度的窗口，而不是单条中心射线；这对树干、
                    # 灯杆等细障碍尤其关键。5% 分位数既保留少量深度噪声鲁棒性，
                    # 又能把擦边碰撞纳入预测。
                    band = observation.depth[
                        drone,
                        44:84,
                        max(0, column - 10): min(128, column + 11),
                        0,
                    ]
                    clearance[drone, candidate] = 0.5 + 19.5 * float(
                        np.percentile(band, 5)
                    )

        predicted_separation = np.full((n, len(view_offsets)), np.inf, dtype=np.float64)
        scores = np.empty((n, len(view_offsets)), dtype=np.float64)
        for drone in range(n):
            current_distance = goal_distance[drone]
            for candidate in range(len(view_offsets)):
                if phases[drone] == "takeoff":
                    score_target = positions[drone].copy()
                    score_target[2] = self.cruise_altitudes_m[drone]
                else:
                    score_target = phase_targets[drone]
                current_distance = float(np.linalg.norm(score_target - positions[drone]))
                next_distance = float(np.linalg.norm(score_target - predicted[drone, candidate]))
                separation = min(
                    (
                        float(np.linalg.norm(predicted[drone, candidate] - positions[other]))
                        for other in range(n)
                        if other != drone
                    ),
                    default=float("inf"),
                )
                predicted_separation[drone, candidate] = separation
                # 3 m 是以 2.55 m/s 巡航时仍有制动余量的安全门槛。惩罚从这里
                # 开始，而不是等到几乎接触（旧值 1.2 m）才介入。
                obstacle_penalty = max(0.0, 3.0 - clearance[drone, candidate]) ** 2 * 2.5
                separation_penalty = max(0.0, 1.5 - separation) ** 2 * 6.0
                candidate_heading = candidates[drone, candidate, 4] * math.pi
                relative_turn = (candidate_heading - yaw[drone] + math.pi) % (2.0 * math.pi) - math.pi
                turn_penalty = 0.04 * abs(relative_turn)
                scores[drone, candidate] = (
                    current_distance - next_distance - obstacle_penalty - separation_penalty - turn_penalty
                )

        selected = np.argmax(scores, axis=1)
        action = candidates[np.arange(n), selected].astype(np.float32)
        chosen_clearance = clearance[np.arange(n), selected]
        chosen_heading = action[:, 4].astype(np.float64) * math.pi
        heading_error = (chosen_heading - yaw + math.pi) % (2.0 * math.pi) - math.pi
        escape_climb = np.zeros(n, dtype=bool)
        terminal_clear_approach = np.zeros(n, dtype=bool)
        safety_speed_cap = np.ones(n, dtype=np.float32)
        overhead_clearance = np.empty(n, dtype=np.float64)
        for drone in range(n):
            overhead_band = observation.depth[drone, 4:36, 48:80, 0]
            overhead_clearance[drone] = 0.5 + 19.5 * float(
                np.percentile(overhead_band, 10)
            )

        for drone in range(n):
            if phases[drone] not in {"takeoff", "land"}:
                if (
                    self.escape_climb_used[drone]
                    and float(
                        np.linalg.norm(
                            positions[drone, :2] - self.escape_origins_xy_m[drone]
                        )
                    ) > 4.0
                ):
                    self.escape_climb_used[drone] = False
                active_escape_altitude = self.escape_target_altitudes_m[drone]
                if math.isfinite(active_escape_altitude):
                    if positions[drone, 2] < active_escape_altitude - 0.2:
                        action[drone, :3] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
                        action[drone, 3] = 0.12
                        action[drone, 4] = float(
                            np.clip(yaw[drone] / math.pi, -1.0, 1.0)
                        )
                        safety_speed_cap[drone] = 0.12
                        escape_climb[drone] = True
                        continue
                    self.escape_target_altitudes_m[drone] = np.nan
                active_scan_yaw = self.scan_target_yaw_rad[drone]
                if math.isfinite(active_scan_yaw):
                    scan_error = (
                        active_scan_yaw - yaw[drone] + math.pi
                    ) % (2.0 * math.pi) - math.pi
                    if abs(scan_error) > math.radians(8.0):
                        action[drone, :4] = 0.0
                        action[drone, 4] = float(
                            np.clip(active_scan_yaw / math.pi, -1.0, 1.0)
                        )
                        safety_speed_cap[drone] = 0.0
                        continue
                    self.scan_target_yaw_rad[drone] = np.nan
                # 整个可见扇区均被近距离障碍封死时，二维绕行没有信息增益。
                # 只有图像上方带也确认有净空时才允许单次升高 3 m；横向移动
                # 4 m 之前禁止重复爬升，避免在树冠下无信息地连续上升。
                full_view_blocked = float(np.max(clearance[drone])) < 1.25
                stalled_with_escape_available = (
                    self.steps_without_progress[drone] >= 150
                    and not self.progress_escape_used[drone]
                )
                if (
                    (full_view_blocked or stalled_with_escape_available)
                    and overhead_clearance[drone] > 3.0
                    and not self.escape_climb_used[drone]
                    and positions[drone, 2] < self.maximum_altitudes_m[drone] - 0.3
                ):
                    escape_target = min(
                        self.maximum_altitudes_m[drone],
                        positions[drone, 2] + 3.0,
                    )
                    self.cruise_altitudes_m[drone] = max(
                        self.cruise_altitudes_m[drone], escape_target
                    )
                    self.escape_target_altitudes_m[drone] = escape_target
                    self.escape_origins_xy_m[drone] = positions[drone, :2]
                    self.escape_climb_used[drone] = True
                    if stalled_with_escape_available:
                        # A generic progress-triggered terrain escape is useful
                        # once, but repeated climbs can consume the entire
                        # horizon in forested scenes. Truly blocked depth views
                        # remain eligible again after horizontal cooldown.
                        self.progress_escape_used[drone] = True
                    self.best_horizontal_goal_distance_m[drone] = horizontal_goal_distance[drone]
                    self.steps_without_progress[drone] = 0
                    action[drone, :3] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
                    action[drone, 3] = 0.12
                    action[drone, 4] = float(np.clip(yaw[drone] / math.pi, -1.0, 1.0))
                    safety_speed_cap[drone] = 0.12
                    escape_climb[drone] = True
                    continue
                if float(np.max(clearance[drone])) < 0.9:
                    scan_target = (yaw[drone] + 0.68 + math.pi) % (
                        2.0 * math.pi
                    ) - math.pi
                    self.scan_target_yaw_rad[drone] = scan_target
                    action[drone, :4] = 0.0
                    action[drone, 4] = float(
                        np.clip(scan_target / math.pi, -1.0, 1.0)
                    )
                    safety_speed_cap[drone] = 0.0
                    continue
                # 感知对齐：大角度转向时仅允许低速侧移，沿障碍边缘获取新视角；
                # 不能高速盲飞，也不能完全停住而陷入左右扫描死锁。
                terminal_clear_approach[drone] = (
                    phases[drone] == "approach"
                    and horizontal_goal_distance[drone] < 6.0
                    # The landing pad is intentionally visible in forward
                    # depth and must be approached. Keep the same 2.5 m safety
                    # gate used by the UrbanFly dataset while still braking for
                    # anything closer than that corridor margin.
                    and chosen_clearance[drone] >= 2.5
                )
                if terminal_clear_approach[drone]:
                    # Within a depth-verified clear terminal corridor, world-
                    # frame translation can continue while yaw settles. This
                    # reserves time for the native 0.5 s contact-stability gate
                    # without weakening any obstacle-proximate turn limit.
                    safety_speed_cap[drone] = 0.46
                elif abs(heading_error[drone]) > math.radians(22.0):
                    safety_speed_cap[drone] = 0.08
                else:
                    # 净空风险直接约束执行速度，而不只参与候选排序。
                    if chosen_clearance[drone] < 1.4:
                        safety_speed_cap[drone] = 0.045
                    elif chosen_clearance[drone] < 2.2:
                        safety_speed_cap[drone] = 0.10
                    elif chosen_clearance[drone] < 3.5:
                        safety_speed_cap[drone] = 0.22
                    elif chosen_clearance[drone] < 6.0:
                        safety_speed_cap[drone] = 0.46
                    # 转向时相机视场与惯性速度方向并不重合，必须先减速再转。
                    if abs(heading_error[drone]) > math.radians(16.0):
                        # With more than 2.2 m of measured clearance, 0.54 m/s
                        # still leaves several seconds of braking margin while
                        # avoiding the long near-stationary arcs that exhaust a
                        # 60 s mission before the platform-contact dwell. The
                        # <2.2 m and blind-turn gates above stay unchanged.
                        turn_cap = 0.22 if chosen_clearance[drone] >= 3.5 else 0.18
                        safety_speed_cap[drone] = min(safety_speed_cap[drone], turn_cap)
                    elif abs(heading_error[drone]) > math.radians(9.0):
                        turn_cap = 0.32 if chosen_clearance[drone] >= 3.5 else 0.28
                        safety_speed_cap[drone] = min(safety_speed_cap[drone], turn_cap)

        previous = self.previous_action
        if previous is None or previous.shape != action.shape:
            previous = observation.state[:, 132:137].astype(np.float32)
        if self.decisions == 0:
            previous[:, 4] = np.clip(yaw / math.pi, -1.0, 1.0).astype(np.float32)
        # 限制加减速与方向突变，避免局部避障时的大倾角瞬态。
        # 只限制加速；安全刹车不得被“平滑”逻辑延迟。
        # The depth-verified terminal corridor may recover speed faster after a
        # one-frame conservative brake. Any newly observed risk still brakes
        # immediately because only the upper acceleration bound is relaxed.
        acceleration_increment = np.where(terminal_clear_approach, 0.04, 0.012)
        action[:, 3] = np.minimum(action[:, 3], previous[:, 3] + acceleration_increment)
        action[:, 3] = np.minimum(action[:, 3], safety_speed_cap)
        action[:, 3] = np.clip(action[:, 3], 0.0, 1.0)
        previous_yaw = previous[:, 4].astype(np.float64) * math.pi
        desired_yaw = action[:, 4].astype(np.float64) * math.pi
        yaw_delta = (desired_yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
        smoothed_yaw = previous_yaw + 0.14 * yaw_delta
        smoothed_yaw = (smoothed_yaw + math.pi) % (2.0 * math.pi) - math.pi
        action[:, 4] = np.clip(smoothed_yaw / math.pi, -1.0, 1.0).astype(np.float32)
        for drone in range(n):
            desired = action[drone, :3].astype(np.float64)
            prior = previous[drone, :3].astype(np.float64)
            if (
                not escape_climb[drone]
                and float(np.linalg.norm(desired)) > 1e-6
                and float(np.linalg.norm(prior)) > 1e-6
            ):
                blended = 0.88 * prior + 0.12 * desired
                norm = float(np.linalg.norm(blended))
                if norm > 1e-6:
                    action[drone, :3] = (blended / norm).astype(np.float32)
        self.previous_action = action.copy()

        selected_predicted = positions + (
            action[:, :3].astype(np.float64)
            * action[:, 3:4].astype(np.float64)
            * self.maximum_speed_mps
            * self.prediction_dt_s
        )
        self.decisions += 1
        return WorldModelBatchDecision(
            action=action,
            selected_candidate=selected.astype(np.int64),
            predicted_next_position_enu_m=selected_predicted.astype(np.float32),
            predicted_clearance_m=chosen_clearance.astype(np.float32),
            predicted_minimum_separation_m=predicted_separation[np.arange(n), selected].astype(np.float32),
            candidate_scores=scores.astype(np.float32),
        )
