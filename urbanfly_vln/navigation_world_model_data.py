from __future__ import annotations

from collections import OrderedDict

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from .observation_policy_data import HelsinkiEpisodeRecord, public_state_features


class HelsinkiLatentTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """Consecutive public-observation transitions from canonical Dataset v1."""

    def __init__(
        self,
        records: tuple[HelsinkiEpisodeRecord, ...],
        *,
        history_frames: int = 2,
        cache_episodes: int = 2,
    ) -> None:
        if not records:
            raise ValueError("transition dataset requires at least one episode")
        self.records = records
        self.history_frames = int(history_frames)
        self.cache_episodes = int(cache_episodes)
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._index = [
            (record_index, step)
            for record_index, record in enumerate(records)
            for step in range(max(0, record.steps - 1))
        ]
        if not self._index:
            raise ValueError("transition dataset contains no consecutive steps")

    def __len__(self) -> int:
        return len(self._index)

    def _load_episode(self, record_index: int) -> dict[str, np.ndarray]:
        cached = self._cache.pop(record_index, None)
        if cached is not None:
            self._cache[record_index] = cached
            return cached
        record = self.records[record_index]
        with h5py.File(record.path, "r") as handle:
            commanded = np.asarray(handle["actions/commanded_body_flu"][:], dtype=np.float32)
            executed = np.asarray(handle["actions/executed_body_flu"][:], dtype=np.float32)
            previous = np.zeros_like(commanded)
            previous[1:] = commanded[:-1]
            orientation = np.asarray(handle["state/orientation_xyzw"][:], dtype=np.float64)
            position = np.asarray(handle["state/position_world"][:], dtype=np.float64)
            world_delta = np.asarray(handle["next_state/position_world"][:], dtype=np.float64) - position
            arrays = {
                "rgb": np.asarray(handle["observations/rgb_front"][:], dtype=np.uint8),
                "depth": np.asarray(handle["observations/depth_front"][:], dtype=np.float32),
                "valid": np.asarray(handle["observations/depth_valid"][:], dtype=bool),
                "state": public_state_features(
                    local_goal_body=handle["goal/local_goal_body"][:],
                    linear_velocity_world=handle["state/linear_velocity"][:],
                    angular_velocity_world=handle["state/angular_velocity"][:],
                    orientation_xyzw=orientation,
                    previous_action_physical=previous,
                ),
                "action": executed,
                "delta_position_body": Rotation.from_quat(orientation).inv().apply(world_delta).astype(np.float32),
                "progress_delta": np.diff(
                    np.asarray(handle["route/progress"][:], dtype=np.float32), append=float(handle["route/progress"][-1])
                ).astype(np.float32),
                "next_clearance": np.concatenate(
                    (
                        np.asarray(handle["labels/minimum_clearance"][1:], dtype=np.float32),
                        np.asarray(handle["labels/minimum_clearance"][-1:], dtype=np.float32),
                    )
                ),
            }
        self._cache[record_index] = arrays
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return arrays

    def _history(self, step: int) -> np.ndarray:
        return np.maximum(np.arange(step - self.history_frames + 1, step + 1), 0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, step = self._index[index]
        arrays = self._load_episode(record_index)
        current = self._history(step)
        following = self._history(step + 1)
        return {
            "rgb": torch.from_numpy(arrays["rgb"][current].copy()),
            "depth_m": torch.from_numpy(arrays["depth"][current].copy()),
            "depth_valid": torch.from_numpy(arrays["valid"][current].copy()),
            "public_state": torch.from_numpy(arrays["state"][step].copy()),
            "next_rgb": torch.from_numpy(arrays["rgb"][following].copy()),
            "next_depth_m": torch.from_numpy(arrays["depth"][following].copy()),
            "next_depth_valid": torch.from_numpy(arrays["valid"][following].copy()),
            "next_public_state": torch.from_numpy(arrays["state"][step + 1].copy()),
            "executed_action": torch.from_numpy(arrays["action"][step].copy()),
            "physical_target": torch.from_numpy(
                np.concatenate(
                    (
                        arrays["delta_position_body"][step],
                        arrays["progress_delta"][step : step + 1],
                        arrays["next_clearance"][step : step + 1],
                    )
                ).astype(np.float32)
            ),
            "episode_index": torch.as_tensor(self.records[record_index].episode_index),
            "step": torch.as_tensor(step),
        }
