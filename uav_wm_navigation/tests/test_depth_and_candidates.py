import numpy as np
import pytest

from uav_wm_navigation.planners.polynomial import sample_quintic
from uav_wm_navigation.planners.yopo_adapter import preprocess_depth
from uav_wm_navigation.data.labels import build_depth_clearance_query
from uav_wm_navigation.types import CandidateTrajectory, SensorFrame


def test_depth_preprocessing_shape_range_and_invalid_fill() -> None:
    depth = np.full((48, 80), 10.0, dtype=np.float32)
    depth[5:10, 5:10] = np.nan
    depth[0, 0] = 0.0
    output = preprocess_depth(depth, 160, 96, 20.0, 0.8)
    assert output.shape == (1, 1, 96, 160)
    assert np.isfinite(output).all()
    assert 0.0 <= output.min() <= output.max() <= 1.0


def test_all_invalid_depth_rejected() -> None:
    with pytest.raises(ValueError, match="no valid"):
        preprocess_depth(np.full((10, 10), np.nan), 10, 10, 20.0, 0.8)


def test_candidate_and_quintic_boundary_conditions() -> None:
    p, v, a = sample_quintic(np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3), 2.0, 21)
    candidate = CandidateTrajectory("a", p, v, a, 2.0, 0.2, np.ones(21, dtype=bool))
    assert candidate.positions.shape == (21, 3)
    assert np.allclose(candidate.positions[0], 0.0, atol=1e-6)
    assert np.allclose(candidate.positions[-1], 1.0, atol=1e-5)
    assert np.allclose(candidate.velocities[[0, -1]], 0.0, atol=1e-5)


def test_visibility_aware_depth_clearance_query() -> None:
    depth = np.full((9, 9), 5.0, dtype=np.float32)
    intrinsics = np.array([[4.5, 0, 4], [0, 4.5, 4], [0, 0, 1]], dtype=np.float32)
    sensor = SensorFrame(
        0.0, depth, np.ones_like(depth, dtype=bool),
        camera_intrinsics=intrinsics, camera_pose_nwu=np.eye(4, dtype=np.float32),
    )
    query = build_depth_clearance_query(sensor, stride=1)
    clearance = query(np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
    assert clearance[0] > 2.5
    assert clearance[1] < 0.1
    assert clearance[2] == 0.0
    assert np.isnan(clearance[3])
