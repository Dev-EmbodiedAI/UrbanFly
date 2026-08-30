import json
from pathlib import Path

import numpy as np

from backend.engine.collision import (
    DenseSignedDistanceField,
    HierarchicalStaticCollisionMap,
    SparseStaticCollisionMap,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CITY_COLLISION = REPO_ROOT / "data" / "citygs_collision" / "Residence"


def _synthetic_hierarchy():
    distance = np.full((12, 12, 12), 8.0, dtype=np.float16)
    distance[5:7, 5:7, 5:7] = -1.0
    global_esdf = DenseSignedDistanceField(
        distance=distance,
        origin=np.zeros(3, dtype=np.float32),
        resolution=1.0,
        truncation=8.0,
    )
    local_surface = SparseStaticCollisionMap(
        coords=np.array([[8, 5, 5]], dtype=np.uint16),
        origin=np.zeros(3, dtype=np.float32),
        resolution=1.0,
        shape=distance.shape,
    )
    return HierarchicalStaticCollisionMap(global_esdf, local_surface)


def test_dense_esdf_is_signed_and_rejects_out_of_bounds():
    hierarchy = _synthetic_hierarchy()
    assert hierarchy.global_esdf.clearance(np.array([5.5, 5.5, 5.5])) < 0.0
    assert hierarchy.global_esdf.clearance(np.array([2.5, 2.5, 2.5])) > 0.0
    assert hierarchy.global_esdf.clearance(np.array([-1.0, 2.5, 2.5])) < 0.0


def test_swept_query_catches_obstacle_between_endpoints():
    hierarchy = _synthetic_hierarchy()
    start = np.array([2.5, 5.5, 5.5])
    end = np.array([10.5, 5.5, 5.5])
    assert hierarchy.clearance(start) > 0.5
    assert hierarchy.clearance(end) > 0.5

    collides, clearance, hit = hierarchy.sweep_collides(
        start,
        end,
        safety_radius=0.5,
        step=0.2,
    )
    assert collides
    assert clearance < 0.5
    assert hit is not None
    assert 4.0 < hit[0] < 9.5


def test_local_esdf_tiles_are_bounded_and_cached():
    hierarchy = _synthetic_hierarchy()
    query = np.array([8.5, 5.5, 5.5])
    first = hierarchy.local_tiles.clearance(query)
    second = hierarchy.local_tiles.clearance(query)
    stats = hierarchy.local_tiles.stats()

    assert first == second
    assert stats["resident_blocks"] == 1
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["resident_megabytes"] <= 0.01


def test_citygs_collision_artifacts_are_closed_and_metric():
    metadata = json.loads(
        (CITY_COLLISION / "collision_geometry.json").read_text(encoding="utf-8")
    )
    assert (CITY_COLLISION / "city_collision.glb").stat().st_size > 1_000_000
    assert (CITY_COLLISION / "global_esdf.npz").stat().st_size > 1_000_000
    assert metadata["buildings"]["watertight"] is True
    assert metadata["buildings"]["winding_consistent"] is True
    assert metadata["buildings"]["triangles"] > 500_000
    assert metadata["global_esdf"]["resolution_m"] == 1.0
    assert metadata["local_esdf"]["resolution_m"] == 0.25
    assert metadata["local_esdf"]["storage"] == "runtime_lru_tiles"
