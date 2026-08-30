import json
from pathlib import Path

import h5py
import numpy as np

from uav_wm_navigation.control import SafetyFilter, TrajectoryExecutor
from uav_wm_navigation.data import WorldModelDataset, collect_episode, create_grouped_splits, validate_episode
from uav_wm_navigation.planners import MockCandidatePlanner
from uav_wm_navigation.simulators import MockSimulator


def test_hdf5_roundtrip_split_and_mock_closed_loop(tmp_path: Path) -> None:
    simulator = MockSimulator(seed=3)
    planner = MockCandidatePlanner(candidate_count=3, horizon_steps=8)
    executor = TrajectoryExecutor(simulator, SafetyFilter({"max_acceleration_mps2": 20}), 0.1)
    path = collect_episode(simulator, planner, executor, tmp_path, "episode_000", np.array([4.0, 0.0, 2.0]), steps=7, future_horizon=3)
    report = validate_episode(path)
    assert report["status"] == "valid" and report["steps"] > 1
    with h5py.File(path, "r") as handle:
        assert handle["candidates/positions"].shape[1:] == (3, 8, 3)
        assert handle["depth_m"].shape[-2:] == (96, 160)
    splits = create_grouped_splits([path], tmp_path / "splits.json", 1)
    assert splits["train"] == [str(path.resolve())]
    dataset = WorldModelDataset(splits["train"], history=2)
    sample = dataset[0]
    assert sample["depth"].shape == (2, 1, 96, 160)
    assert sample["trajectories"].shape == (3, 8, 9)
    assert json.loads((tmp_path / "episode_000.metadata.json").read_text())["scenario"] == "StaticObstacle"


def test_grouped_split_keeps_repeated_corridor_seeds_together(tmp_path: Path) -> None:
    paths = []
    for corridor_index in range(5):
        for seed in range(2):
            path = tmp_path / f"corridor_{corridor_index}_seed_{seed}.h5"
            path.touch()
            path.with_suffix(".metadata.json").write_text(json.dumps({
                "map": "Town10HD", "spatial_zone": "A", "corridor_id": f"c{corridor_index}",
                "route_id": f"route_{corridor_index}_{seed}",
                "scenario_script": f"DynamicCrossing:hard:{seed}",
            }))
            paths.append(path)
    split = create_grouped_splits(paths, tmp_path / "splits.json", seed=7)
    ownership = {}
    for split_name in ("train", "validation", "test"):
        for value in split[split_name]:
            corridor = Path(value).stem.split("_seed_")[0]
            ownership.setdefault(corridor, set()).add(split_name)
    assert all(len(splits) == 1 for splits in ownership.values())
