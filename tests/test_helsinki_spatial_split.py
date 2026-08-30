import numpy as np

from backend.engine.helsinki_spatial_split import HelsinkiSpatialSplit


def test_split_is_episode_level_and_buffered() -> None:
    split = HelsinkiSpatialSplit(guard_m=20.0)
    assert split.assign_backend_route(np.asarray([[0, 10, -250], [20, 10, -150]])) == "test"
    assert split.assign_backend_route(np.asarray([[0, 10, 0], [20, 10, 200]])) == "train"
    assert split.assign_backend_route(np.asarray([[0, 10, 350], [20, 10, 450]])) == "validation"
    assert split.assign_backend_route(np.asarray([[0, 10, -150], [20, 10, 0]])) is None
    assert split.assign_backend_position(np.asarray([0, 10, -100])) is None


def test_split_canonical_north_relation() -> None:
    split = HelsinkiSpatialSplit()
    manifest_bounds = {
        key: [-bounds[1], -bounds[0]]
        for key, bounds in (
            (name, split.interior_backend_z_bounds(name))
            for name in ("train", "validation", "test")
        )
    }
    assert manifest_bounds["test"] == [120.0, 280.0]
    assert manifest_bounds["train"] == [-280.0, 80.0]
    assert manifest_bounds["validation"] == [-480.0, -320.0]
