"""Explicit coordinate contracts for Helsinki Dataset v1.

The licensed Helsinki asset is stored in the renderer/backend frame
``[east, up, south]``.  Its negative Z axis is geographic north.  Dataset v1
uses a conventional right-handed ENU world frame ``[east, north, up]`` and a
right-handed FLU body frame ``[forward, left, up]``.

These functions are deliberately small and dependency-free so every boundary
(planner, renderer, WebSocket adapter, writer and tests) uses the same signed
permutation instead of relying on comments or axis names.
"""

from __future__ import annotations

import math

import numpy as np


CANONICAL_WORLD_FRAME = "ENU"
CANONICAL_BODY_FRAME = "FLU"
CAMERA_FRAME = "RDF"
QUATERNION_ORDER = "xyzw"

# backend = [east, up, south] = ENU @ ENU_TO_BACKEND.T
ENU_TO_BACKEND = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
BACKEND_TO_ENU = ENU_TO_BACKEND.T


def _vectors(value: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1:] != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must end in a finite dimension of length 3")
    return array


def backend_world_to_enu(value: np.ndarray | list[float]) -> np.ndarray:
    """Convert renderer/backend ``[east, up, south]`` vectors to ENU."""

    return _vectors(value, "backend world vector") @ BACKEND_TO_ENU.T


def enu_to_backend_world(value: np.ndarray | list[float]) -> np.ndarray:
    """Convert canonical ENU vectors to renderer/backend coordinates."""

    return _vectors(value, "ENU vector") @ ENU_TO_BACKEND.T


def backend_yaw_to_enu_radians(yaw_degrees: float) -> float:
    """Convert backend yaw (east toward south, degrees) to ENU yaw radians."""

    yaw = float(yaw_degrees)
    if not math.isfinite(yaw):
        raise ValueError("yaw_degrees must be finite")
    return -math.radians(yaw)


def enu_yaw_to_backend_degrees(yaw_radians: float) -> float:
    yaw = float(yaw_radians)
    if not math.isfinite(yaw):
        raise ValueError("yaw_radians must be finite")
    return -math.degrees(yaw)


def body_flu_yaw_rate_to_backend_degrees(yaw_rate_radians_s: float) -> float:
    """Convert canonical positive-left FLU yaw rate to backend yaw degrees/s."""

    yaw_rate = float(yaw_rate_radians_s)
    if not math.isfinite(yaw_rate):
        raise ValueError("yaw_rate_radians_s must be finite")
    return -math.degrees(yaw_rate)


def backend_yaw_rate_to_body_flu_radians(yaw_rate_degrees_s: float) -> float:
    """Convert backend positive-toward-south yaw rate to positive-left FLU."""

    yaw_rate = float(yaw_rate_degrees_s)
    if not math.isfinite(yaw_rate):
        raise ValueError("yaw_rate_degrees_s must be finite")
    return -math.radians(yaw_rate)


def enu_delta_to_body_flu(
    delta_enu: np.ndarray | list[float],
    yaw_enu_radians: float,
) -> np.ndarray:
    """Rotate an ENU displacement into canonical body FLU."""

    delta = _vectors(delta_enu, "ENU delta")
    yaw = float(yaw_enu_radians)
    if not math.isfinite(yaw):
        raise ValueError("yaw_enu_radians must be finite")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    east, north, up = np.moveaxis(delta, -1, 0)
    return np.stack(
        [
            cosine * east + sine * north,
            -sine * east + cosine * north,
            up,
        ],
        axis=-1,
    )


def body_flu_to_enu(
    value: np.ndarray | list[float],
    yaw_enu_radians: float,
) -> np.ndarray:
    body = _vectors(value, "body FLU vector")
    yaw = float(yaw_enu_radians)
    if not math.isfinite(yaw):
        raise ValueError("yaw_enu_radians must be finite")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    forward, left, up = np.moveaxis(body, -1, 0)
    return np.stack(
        [
            cosine * forward - sine * left,
            sine * forward + cosine * left,
            up,
        ],
        axis=-1,
    )


def backend_delta_to_body_flu(
    delta_backend: np.ndarray | list[float],
    yaw_backend_degrees: float,
) -> np.ndarray:
    return enu_delta_to_body_flu(
        backend_world_to_enu(delta_backend),
        backend_yaw_to_enu_radians(yaw_backend_degrees),
    )


def coordinate_metadata() -> dict[str, object]:
    """Dataset metadata shared by every Dataset v1 episode."""

    return {
        "world_frame": {
            "name": CANONICAL_WORLD_FRAME,
            "axes": ["east", "north", "up"],
            "handedness": "right",
            "units": "meter",
        },
        "backend_world_frame": {
            "name": "HELSINKI_RENDERER_Y_UP",
            "axes": ["east", "up", "south"],
            "north_axis": "negative_z",
            "units": "meter",
        },
        "body_frame": {
            "name": CANONICAL_BODY_FRAME,
            "axes": ["forward", "left", "up"],
            "handedness": "right",
        },
        "camera_frame": {
            "name": CAMERA_FRAME,
            "axes": ["right", "down", "forward"],
            "optical_axis": "positive_z",
        },
        "quaternion_order": QUATERNION_ORDER,
        "linear_velocity_frame": CANONICAL_WORLD_FRAME,
        "angular_velocity_frame": CANONICAL_WORLD_FRAME,
        "linear_acceleration_frame": CANONICAL_WORLD_FRAME,
        "action_frame": CANONICAL_BODY_FRAME,
        "action_order": ["forward_mps", "left_mps", "up_mps", "yaw_rate_rps"],
        "yaw_rate_positive": "left_ccw_about_body_up",
    }
