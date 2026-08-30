from pathlib import Path

import numpy as np
import trimesh

from backend.engine.triangle_geometry import TriangleMeshLocalCollision


def _box_collision() -> TriangleMeshLocalCollision:
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    _ = mesh.triangles_tree
    return TriangleMeshLocalCollision(mesh, Path("synthetic_box"), 0.0)


def test_triangle_surface_distance_and_sphere_collision():
    collision = _box_collision()
    assert collision.distance_to_surface(np.array([2.0, 0.0, 0.0])) == 1.0
    assert not collision.is_collision(np.array([2.0, 0.0, 0.0]), drone_radius=0.9)
    assert collision.is_collision(np.array([2.0, 0.0, 0.0]), drone_radius=1.0)


def test_segment_query_detects_crossing_triangle_surface():
    result = _box_collision().segment_query(
        np.array([-2.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        drone_radius=0.1,
    )
    assert result.collision
    assert result.minimum_distance_m <= 0.1
    assert result.collision_position is not None
    assert result.triangle_index is not None


def test_trajectory_query_preserves_free_path():
    result = _box_collision().trajectory_query(
        np.array([[-2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]]),
        drone_radius=0.5,
    )
    assert not result.collision
    assert result.minimum_distance_m >= 0.99
    assert result.sample_count > 3
