from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from .observation_policy import ACTION_LIMITS


EPISODE_INDEX = re.compile(r"_(\d{3})_")


@dataclass(frozen=True)
class HelsinkiEpisodeRecord:
    episode_index: int
    episode_id: str
    task_type: str
    path: Path
    steps: int


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[HelsinkiEpisodeRecord, ...]
    validation: tuple[HelsinkiEpisodeRecord, ...]


def load_qa_episode_records(qa_path: Path) -> list[HelsinkiEpisodeRecord]:
    qa_path = qa_path.resolve()
    report = json.loads(qa_path.read_text(encoding="utf-8"))
    if report.get("schema") == "urbanfly-helsinki-canonical-dataset-v1-qa":
        if (
            report.get("status") != "PASS"
            or report.get("episode_count") != 100
            or report.get("stale_action_count") != 0
            or report.get("collision_count") != 0
            or report.get("partial_count") != 0
            or report.get("all_hdf5_readback_pass") is not True
        ):
            raise ValueError("canonical dataset QA is not a full PASS")
        manifest_path = Path(report["manifest"]).resolve()
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report.get("schema") != "urbanfly-helsinki-canonical-dataset-v1":
            raise ValueError("canonical dataset manifest schema is invalid")
    if report.get("status") != "PASS":
        raise ValueError("refusing to train from a QA report that is not PASS")
    if report.get("schema") != "urbanfly-helsinki-canonical-dataset-v1" and not bool(
        (report.get("gate_checks") or {}).get("stale_action_zero")
    ):
        raise ValueError("refusing to train from a dataset with stale_action")
    if report.get("corrupted_hdf5") or report.get("partial_files"):
        raise ValueError("refusing to train from corrupt or partial data")
    records: list[HelsinkiEpisodeRecord] = []
    seen: set[int] = set()
    for item in report.get("episodes", []):
        match = EPISODE_INDEX.search(str(item["episode_id"]))
        if match is None:
            raise ValueError(f"episode index is missing from {item['episode_id']}")
        index = int(match.group(1))
        path = Path(item["path"]).resolve()
        if index in seen or not path.is_file():
            raise ValueError(f"duplicate or missing episode {index:03d}: {path}")
        integrity = item.get("integrity_status", item.get("hdf5_readback"))
        if integrity != "PASS":
            raise ValueError(f"episode {index:03d} failed integrity QA")
        seen.add(index)
        records.append(
            HelsinkiEpisodeRecord(
                episode_index=index,
                episode_id=str(item["episode_id"]),
                task_type=str(item["task_type"]),
                path=path,
                steps=int(item["steps"]),
            )
        )
    records.sort(key=lambda item: item.episode_index)
    if [item.episode_index for item in records] != list(range(len(records))):
        raise ValueError("QA episodes are not unique and contiguous from zero")
    return records


def tail_episode_split(
    records: list[HelsinkiEpisodeRecord],
    *,
    validation_episodes: int = 20,
) -> EpisodeSplit:
    if validation_episodes <= 0 or validation_episodes >= len(records):
        raise ValueError("validation_episodes must leave non-empty train and validation sets")
    boundary = len(records) - validation_episodes
    train = tuple(item for item in records if item.episode_index < boundary)
    validation = tuple(item for item in records if item.episode_index >= boundary)
    if len(train) + len(validation) != len(records):
        raise RuntimeError("episode split lost records")
    return EpisodeSplit(train=train, validation=validation)


def action_statistics(records: tuple[HelsinkiEpisodeRecord, ...]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(4, dtype=np.float64)
    squared = np.zeros(4, dtype=np.float64)
    for record in records:
        with h5py.File(record.path, "r") as handle:
            action = np.asarray(handle["actions/commanded_body_flu"][:], dtype=np.float64)
        count += len(action)
        total += action.sum(axis=0)
        squared += np.square(action).sum(axis=0)
    if count == 0:
        raise ValueError("no actions available")
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), np.asarray([0.1, 0.05, 0.05, 0.02]))
    return mean.astype(np.float32), std.astype(np.float32)


def public_state_features(
    *,
    local_goal_body: np.ndarray,
    linear_velocity_world: np.ndarray,
    angular_velocity_world: np.ndarray,
    orientation_xyzw: np.ndarray,
    previous_action_physical: np.ndarray,
) -> np.ndarray:
    rotation = Rotation.from_quat(np.asarray(orientation_xyzw, dtype=np.float64))
    linear_body = rotation.inv().apply(np.asarray(linear_velocity_world, dtype=np.float64))
    angular_body = rotation.inv().apply(np.asarray(angular_velocity_world, dtype=np.float64))
    gravity_world = np.zeros_like(linear_body)
    gravity_world[..., 2] = -1.0
    gravity_body = rotation.inv().apply(gravity_world)
    previous_normalized = np.clip(
        np.asarray(previous_action_physical, dtype=np.float64) / ACTION_LIMITS,
        -1.0,
        1.0,
    )
    return np.concatenate(
        (
            np.asarray(local_goal_body, dtype=np.float64),
            linear_body,
            angular_body,
            gravity_body,
            previous_normalized,
        ),
        axis=-1,
    ).astype(np.float32)


class HelsinkiObservationPolicyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: tuple[HelsinkiEpisodeRecord, ...],
        *,
        history_frames: int = 2,
        cache_episodes: int = 2,
        seed: int = 20260829,
    ) -> None:
        if not records:
            raise ValueError("dataset requires at least one episode")
        if history_frames < 1 or cache_episodes < 1:
            raise ValueError("history_frames and cache_episodes must be positive")
        self.records = records
        self.history_frames = int(history_frames)
        self.cache_episodes = int(cache_episodes)
        self.seed = int(seed)
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._sample_index: list[tuple[int, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        order = np.arange(len(self.records))
        if epoch > 0:
            np.random.default_rng(self.seed + int(epoch)).shuffle(order)
        self._sample_index = [
            (int(record_index), step)
            for record_index in order
            for step in range(self.records[int(record_index)].steps)
        ]

    def __len__(self) -> int:
        return len(self._sample_index)

    def _load_episode(self, record_index: int) -> dict[str, np.ndarray]:
        cached = self._cache.pop(record_index, None)
        if cached is not None:
            self._cache[record_index] = cached
            return cached
        record = self.records[record_index]
        with h5py.File(record.path, "r") as handle:
            commanded = np.asarray(handle["actions/commanded_body_flu"][:], dtype=np.float32)
            previous = np.zeros_like(commanded)
            previous[1:] = commanded[:-1]
            arrays = {
                "rgb": np.asarray(handle["observations/rgb_front"][:], dtype=np.uint8),
                "depth": np.asarray(handle["observations/depth_front"][:], dtype=np.float32),
                "valid": np.asarray(handle["observations/depth_valid"][:], dtype=bool),
                "action": commanded,
                "state": public_state_features(
                    local_goal_body=handle["goal/local_goal_body"][:],
                    linear_velocity_world=handle["state/linear_velocity"][:],
                    angular_velocity_world=handle["state/angular_velocity"][:],
                    orientation_xyzw=handle["state/orientation_xyzw"][:],
                    previous_action_physical=previous,
                ),
            }
        if len(arrays["action"]) != record.steps:
            raise ValueError(f"QA/HDF5 step mismatch for {record.episode_id}")
        self._cache[record_index] = arrays
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, step = self._sample_index[index]
        arrays = self._load_episode(record_index)
        history = np.maximum(
            np.arange(step - self.history_frames + 1, step + 1),
            0,
        )
        return {
            "rgb": torch.from_numpy(arrays["rgb"][history].copy()),
            "depth_m": torch.from_numpy(arrays["depth"][history].copy()),
            "depth_valid": torch.from_numpy(arrays["valid"][history].copy()),
            "public_state": torch.from_numpy(arrays["state"][step].copy()),
            "target_action": torch.from_numpy(arrays["action"][step].copy()),
            "episode_index": torch.as_tensor(self.records[record_index].episode_index),
            "step": torch.as_tensor(step),
        }
