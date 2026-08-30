from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


NED_TO_NWU = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def ned_to_nwu(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64) @ NED_TO_NWU.T


def nwu_to_ned(vector: np.ndarray) -> np.ndarray:
    return ned_to_nwu(vector)


def frd_to_flu(vector: np.ndarray) -> np.ndarray:
    return ned_to_nwu(vector)


def flu_to_frd(vector: np.ndarray) -> np.ndarray:
    return ned_to_nwu(vector)


def rotation_ned_frd_to_nwu_flu(rotation_ned_frd: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_ned_frd, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")
    return NED_TO_NWU @ rotation @ NED_TO_NWU


def quaternion_ned_frd_to_nwu_flu(quaternion_xyzw: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64)).as_matrix()
    return Rotation.from_matrix(rotation_ned_frd_to_nwu_flu(rotation)).as_quat()


def quaternion_nwu_flu_to_ned_frd(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return quaternion_ned_frd_to_nwu_flu(quaternion_xyzw)


def camera_rdf_to_body_flu(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("camera points must end in dimension 3")
    right, down, forward = np.moveaxis(points, -1, 0)
    return np.stack([forward, -right, -down], axis=-1)


def body_flu_to_camera_rdf(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("body points must end in dimension 3")
    forward, left, up = np.moveaxis(points, -1, 0)
    return np.stack([-left, -up, forward], axis=-1)


