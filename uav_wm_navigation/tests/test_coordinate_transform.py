import numpy as np
from scipy.spatial.transform import Rotation

from uav_wm_navigation.control.coordinate_transform import (
    body_flu_to_camera_rdf, camera_rdf_to_body_flu,
    ned_to_nwu, nwu_to_ned, quaternion_ned_frd_to_nwu_flu,
)


def test_ned_nwu_roundtrip_and_units() -> None:
    vector = np.array([12.0, -3.0, 5.0])
    assert np.allclose(nwu_to_ned(ned_to_nwu(vector)), vector)
    assert np.allclose(ned_to_nwu(vector), [12.0, 3.0, -5.0])


def test_camera_body_roundtrip_and_forward_axis() -> None:
    camera = np.array([[2.0, 3.0, 10.0]])  # right, down, optical forward
    body = camera_rdf_to_body_flu(camera)
    assert np.allclose(body, [[10.0, -2.0, -3.0]])
    assert np.allclose(body_flu_to_camera_rdf(body), camera)


def test_quaternion_rotation_conversion_is_proper() -> None:
    quaternion = Rotation.from_euler("ZYX", [0.3, -0.2, 0.1]).as_quat()
    converted = quaternion_ned_frd_to_nwu_flu(quaternion)
    matrix = Rotation.from_quat(converted).as_matrix()
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(matrix), 1.0)
