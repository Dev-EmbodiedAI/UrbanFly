import numpy as np

from urbanfly_vln.corridor_world_model import (
    DepthCameraModel,
    DepthWorldModelMPC,
    LocalObstacleMemory,
    WorldModelConfig,
    depth_to_body_points,
    robust_front_clearance,
)


def test_depth_projection_center_pixel_points_forward() -> None:
    camera = DepthCameraModel(width=5, height=5, horizontal_fov_deg=90.0)
    config = WorldModelConfig(point_stride=1, vertical_keep_m=10.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[2, 2] = 4.0

    points = depth_to_body_points(depth, camera, config)

    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], [4.0, 0.0, 0.0], atol=1e-6)


def test_robust_front_clearance_ignores_invalid_depth() -> None:
    depth = np.full((20, 20), np.inf, dtype=np.float32)
    depth[8:12, 8:12] = 3.5
    depth[9, 9] = np.nan

    assert 3.49 <= robust_front_clearance(depth, max_depth_m=18.0) <= 3.51


def test_world_model_steers_away_from_left_blocking_obstacle() -> None:
    model = DepthWorldModelMPC()
    x = np.linspace(2.0, 5.0, 30, dtype=np.float32)
    obstacle = np.column_stack((x, np.full_like(x, -0.7), np.zeros_like(x)))

    action, diagnostics = model.choose_action(
        obstacle_points_body_m=obstacle,
        measured_velocity_body_mps=np.zeros(2, dtype=np.float32),
        current_corridor_y_m=0.0,
    )

    assert action[1] > 0.0
    assert diagnostics["predicted_collision"] is False


def test_world_model_cruises_straight_in_clear_centered_corridor() -> None:
    model = DepthWorldModelMPC()

    action, diagnostics = model.choose_action(
        obstacle_points_body_m=np.empty((0, 3), dtype=np.float32),
        measured_velocity_body_mps=np.zeros(2, dtype=np.float32),
        current_corridor_y_m=0.0,
    )

    np.testing.assert_allclose(action, [2.2, 0.0], atol=1e-6)
    assert diagnostics["safe_candidate_count"] == diagnostics["candidate_count"]


def test_obstacle_memory_retains_and_extrudes_a_passed_surface() -> None:
    memory = LocalObstacleMemory()
    surface = np.asarray([[4.0, -0.5, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    memory.update(surface, np.asarray([0.0, 0.0, 0.0]), yaw_rad=0.0)

    remembered = memory.points_body(np.asarray([4.5, 0.0, 0.0]), yaw_rad=0.0)

    assert remembered.shape[0] >= 6
    assert np.min(remembered[:, 0]) < 0.0
    assert np.max(remembered[:, 0]) >= 1.0
