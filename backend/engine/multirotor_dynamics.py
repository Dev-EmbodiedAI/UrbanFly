"""
UrbanFly 四旋翼 6-DOF 刚体动力学
================================

采用控制器、执行器与刚体物理分层：
路径跟踪器给出期望位置和速度，位置外环产生期望加速度，SO(3) 姿态
控制器产生力矩，X 构型混控器转换为四个电机指令，最后由刚体积分器更新
位置、速度、四元数和角速度。

坐标系为 UrbanFly 本地 ENU 风格世界坐标：X/Z 为水平面，Y 向上。
机体系 X 为机头、Y 为上方（旋翼总推力方向）、Z 为机体侧向。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


GRAVITY = 9.80665
# 50 Hz 刚体/执行器子步。外层仍为 20 Hz，2× 仿真时每步拆为 5 个子步；
# 这对最短 40 ms 电机时间常数仍满足稳定积分，同时可让 30 机保持交互速率。
MAX_PHYSICS_SUBSTEP = 0.02
EPSILON = 1e-9


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude < EPSILON:
        return fallback.copy()
    return vector / magnitude


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize(q, np.array([1.0, 0.0, 0.0, 0.0]))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        q = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            q = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            q = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            q = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return _normalize(q, np.array([1.0, 0.0, 0.0, 0.0]))


def yaw_to_quaternion(yaw_degrees: float) -> np.ndarray:
    """UrbanFly 正偏航由 +X 转向 +Z，因此绕世界 Y 轴使用负右手角。"""
    half = -np.radians(yaw_degrees) * 0.5
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=float)


def quaternion_to_euler_degrees(q: np.ndarray) -> np.ndarray:
    """返回适合 UI 的 roll/pitch/yaw；姿态内部仍始终使用四元数。"""
    rotation = quaternion_to_matrix(q)
    forward = rotation[:, 0]
    up = rotation[:, 1]
    yaw = np.degrees(np.arctan2(forward[2], forward[0]))
    pitch = np.degrees(np.arctan2(forward[1], np.hypot(forward[0], forward[2])))
    right_reference = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right_reference = _normalize(right_reference, np.array([0.0, 0.0, 1.0]))
    roll = np.degrees(np.arctan2(np.dot(up, right_reference), up[1]))
    return np.array([roll, pitch, yaw], dtype=float)


@dataclass
class MultirotorParameters:
    mass: float
    inertia: np.ndarray
    arm_length: float
    max_thrust_per_motor: float
    motor_time_constant: float
    yaw_moment_ratio: float
    linear_drag: np.ndarray
    angular_drag: np.ndarray
    position_gain: float
    velocity_gain: float
    attitude_gain: np.ndarray
    angular_rate_gain: np.ndarray
    hover_power_w: float
    avionics_power_w: float
    thrust_coefficient: float = 1.0e-5

    @classmethod
    def from_dict(cls, values: Dict) -> "MultirotorParameters":
        data = dict(values)
        for key in ("inertia", "linear_drag", "angular_drag", "attitude_gain", "angular_rate_gain"):
            data[key] = np.asarray(data[key], dtype=float)
        return cls(**data)


class MultirotorDynamics:
    """每架无人机独立持有一个实例，以保存电机和姿态内部状态。"""

    def __init__(self, parameters: MultirotorParameters):
        self.parameters = parameters
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.angular_velocity = np.zeros(3, dtype=float)
        self.motor_omega = np.zeros(4, dtype=float)
        self.initialized = False
        self.last_power_w = parameters.avionics_power_w
        self.last_total_thrust = 0.0
        self.last_motor_thrusts = np.zeros(4, dtype=float)
        self._mixer = self._build_mixer()
        self._mixer_inverse = np.linalg.pinv(self._mixer)

    def _build_mixer(self) -> np.ndarray:
        arm = self.parameters.arm_length / np.sqrt(2.0)
        # front-left, front-right, rear-right, rear-left
        positions = np.array(
            [[arm, 0, arm], [arm, 0, -arm], [-arm, 0, -arm], [-arm, 0, arm]],
            dtype=float,
        )
        spin = np.array([1.0, -1.0, 1.0, -1.0])
        mixer = np.zeros((4, 4), dtype=float)
        mixer[0, :] = 1.0
        mixer[1, :] = -positions[:, 2]
        mixer[2, :] = spin * self.parameters.yaw_moment_ratio
        mixer[3, :] = positions[:, 0]
        return mixer

    def initialize(self, yaw_degrees: float) -> None:
        self.orientation = yaw_to_quaternion(yaw_degrees)
        self.angular_velocity[:] = 0.0
        hover_thrust = self.parameters.mass * GRAVITY / 4.0
        hover_omega = np.sqrt(hover_thrust / self.parameters.thrust_coefficient)
        self.motor_omega[:] = hover_omega
        self.initialized = True

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        desired_yaw_degrees: float,
        wind_velocity: Iterable[float],
        payload_mass: float,
        max_acceleration: float,
        dt: float,
    ) -> Dict[str, np.ndarray | float]:
        if not self.initialized:
            self.initialize(desired_yaw_degrees)

        pos = np.asarray(position, dtype=float).copy()
        vel = np.asarray(velocity, dtype=float).copy()
        wind = np.asarray(wind_velocity, dtype=float)
        steps = max(1, int(np.ceil(dt / MAX_PHYSICS_SUBSTEP)))
        sub_dt = dt / steps
        acceleration = np.zeros(3)

        for _ in range(steps):
            pos, vel, acceleration = self._substep(
                pos,
                vel,
                np.asarray(target_position, dtype=float),
                np.asarray(target_velocity, dtype=float),
                desired_yaw_degrees,
                wind,
                max(0.0, payload_mass),
                max_acceleration,
                sub_dt,
            )

        euler = quaternion_to_euler_degrees(self.orientation)
        return {
            "position": pos,
            "velocity": vel,
            "acceleration": acceleration,
            "orientation": self.orientation.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            "motor_omega": self.motor_omega.copy(),
            "motor_thrusts": self.last_motor_thrusts.copy(),
            "total_thrust": self.last_total_thrust,
            "power_w": self.last_power_w,
            "roll": euler[0],
            "pitch": euler[1],
            "yaw": euler[2],
        }

    def _substep(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        desired_yaw_degrees: float,
        wind_velocity: np.ndarray,
        payload_mass: float,
        max_acceleration: float,
        dt: float,
    ):
        params = self.parameters
        total_mass = params.mass + payload_mass

        position_error = target_position - position
        velocity_error = target_velocity - velocity
        acceleration_command = (
            params.position_gain * position_error
            + params.velocity_gain * velocity_error
        )
        horizontal = np.array([acceleration_command[0], 0.0, acceleration_command[2]])
        horizontal_magnitude = float(np.linalg.norm(horizontal))
        if horizontal_magnitude > max_acceleration:
            horizontal *= max_acceleration / horizontal_magnitude
            acceleration_command[0] = horizontal[0]
            acceleration_command[2] = horizontal[2]
        acceleration_command[1] = np.clip(
            acceleration_command[1],
            -max_acceleration,
            max_acceleration,
        )

        gravity = np.array([0.0, -GRAVITY, 0.0])
        required_specific_force = acceleration_command - gravity
        desired_up = _normalize(required_specific_force, np.array([0.0, 1.0, 0.0]))
        yaw_radians = np.radians(desired_yaw_degrees)
        desired_forward_flat = np.array(
            [np.cos(yaw_radians), 0.0, np.sin(yaw_radians)]
        )
        desired_forward = desired_forward_flat - desired_up * np.dot(
            desired_forward_flat, desired_up
        )
        desired_forward = _normalize(desired_forward, np.array([1.0, 0.0, 0.0]))
        desired_side = _normalize(
            np.cross(desired_forward, desired_up),
            np.array([0.0, 0.0, 1.0]),
        )
        desired_rotation = np.column_stack(
            (desired_forward, desired_up, desired_side)
        )

        rotation = quaternion_to_matrix(self.orientation)
        attitude_matrix_error = 0.5 * (
            desired_rotation.T @ rotation - rotation.T @ desired_rotation
        )
        attitude_error = np.array(
            [
                attitude_matrix_error[2, 1],
                attitude_matrix_error[0, 2],
                attitude_matrix_error[1, 0],
            ]
        )
        desired_moment = (
            -params.attitude_gain * attitude_error
            - params.angular_rate_gain * self.angular_velocity
        )

        desired_collective = total_mass * float(np.linalg.norm(required_specific_force))
        desired_collective = np.clip(
            desired_collective,
            0.0,
            params.max_thrust_per_motor * 4.0,
        )
        desired_wrench = np.array(
            [
                desired_collective,
                desired_moment[0],
                desired_moment[1],
                desired_moment[2],
            ]
        )
        desired_motor_thrusts = self._mixer_inverse @ desired_wrench
        desired_motor_thrusts = np.clip(
            desired_motor_thrusts,
            0.0,
            params.max_thrust_per_motor,
        )
        desired_omega = np.sqrt(
            desired_motor_thrusts / params.thrust_coefficient
        )
        motor_alpha = 1.0 - np.exp(-dt / max(params.motor_time_constant, 1e-3))
        self.motor_omega += (desired_omega - self.motor_omega) * motor_alpha

        motor_thrusts = params.thrust_coefficient * self.motor_omega**2
        actual_wrench = self._mixer @ motor_thrusts
        total_thrust = float(actual_wrench[0])
        body_moment = actual_wrench[1:4]

        relative_air_velocity = velocity - wind_velocity
        drag_force_world = -params.linear_drag * relative_air_velocity * np.abs(
            relative_air_velocity
        )
        thrust_world = rotation @ np.array([0.0, total_thrust, 0.0])
        acceleration = (
            thrust_world + drag_force_world
        ) / total_mass + gravity

        inertia = params.inertia
        angular_acceleration = (
            body_moment
            - np.cross(self.angular_velocity, inertia * self.angular_velocity)
            - params.angular_drag * self.angular_velocity
        ) / inertia
        self.angular_velocity += angular_acceleration * dt
        self.angular_velocity = np.clip(self.angular_velocity, -8.0, 8.0)

        omega_quaternion = np.array(
            [0.0, *self.angular_velocity],
            dtype=float,
        )
        quaternion_derivative = 0.5 * quaternion_multiply(
            self.orientation,
            omega_quaternion,
        )
        self.orientation = _normalize(
            self.orientation + quaternion_derivative * dt,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

        velocity += acceleration * dt
        position += velocity * dt

        hover_omega = np.sqrt(
            params.mass * GRAVITY / (4.0 * params.thrust_coefficient)
        )
        normalized_cube = float(
            np.mean(np.maximum(self.motor_omega / hover_omega, 0.0) ** 3)
        )
        self.last_power_w = (
            params.avionics_power_w
            + params.hover_power_w * normalized_cube
            + payload_mass * 12.0
        )
        self.last_total_thrust = total_thrust
        self.last_motor_thrusts = motor_thrusts
        return position, velocity, acceleration

    def handle_collision(self, normal: np.ndarray | None = None) -> None:
        """消除撞击法向速度并显著衰减角速度，避免数值穿透后继续发散。"""
        self.angular_velocity *= 0.25
        if normal is not None:
            normal = _normalize(np.asarray(normal, dtype=float), np.array([0.0, 1.0, 0.0]))
            rotation = quaternion_to_matrix(self.orientation)
            current_up = rotation[:, 1]
            corrected_up = _normalize(current_up + normal * 0.35, normal)
            forward = rotation[:, 0]
            forward -= corrected_up * np.dot(forward, corrected_up)
            forward = _normalize(forward, np.array([1.0, 0.0, 0.0]))
            side = _normalize(np.cross(forward, corrected_up), np.array([0.0, 0.0, 1.0]))
            self.orientation = matrix_to_quaternion(
                np.column_stack((forward, corrected_up, side))
            )
