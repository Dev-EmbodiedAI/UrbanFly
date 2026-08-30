from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .visual_world_model import action_from_delta


@dataclass(frozen=True)
class VisualFlightEpisode:
    run_dir: Path
    plan_ids: tuple[int, ...]
    actions: np.ndarray
    states: np.ndarray
    rewards: np.ndarray
    risks: np.ndarray
    continues: np.ndarray
    depth_max_m: float

    def __len__(self) -> int:
        return len(self.plan_ids)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def discover_visual_runs(roots: list[Path]) -> list[Path]:
    runs: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if (root / "observations" / "rgb").is_dir():
            runs.add(root)
        if root.is_dir():
            for rgb_dir in root.rglob("observations/rgb"):
                candidate = rgb_dir.parent.parent
                if (candidate / "long_range_trajectory.csv").exists():
                    runs.add(candidate)
    return sorted(runs)


def load_visual_episode(
    run_dir: Path,
    *,
    depth_max_m: float = 20.0,
    risk_depth_m: float = 8.0,
    bottom_crop_fraction: float = 1.0 / 3.0,
) -> VisualFlightEpisode:
    run_dir = run_dir.resolve()
    rows = _read_csv(run_dir / "long_range_trajectory.csv")
    route = np.asarray(
        [[_number(row, axis) for axis in ("x", "y", "z")] for row in _read_csv(run_dir / "global_route.csv")],
        dtype=np.float32,
    )
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(_number(row, "replan_step")), []).append(row)
    available = []
    for plan_id in sorted(grouped):
        rgb = run_dir / "observations" / "rgb" / f"plan_{plan_id:04d}.png"
        depth = run_dir / "observations" / "depth" / f"plan_{plan_id:04d}.npy"
        if rgb.exists() and depth.exists():
            available.append(plan_id)
    if len(available) < 2:
        raise ValueError(f"{run_dir} has fewer than two paired RGB-D planning observations")

    depth_statistics: dict[int, tuple[float, float]] = {}
    for plan_id in available:
        depth = np.load(run_dir / "observations" / "depth" / f"plan_{plan_id:04d}.npy").astype(np.float32)
        valid_height = max(1, round(depth.shape[0] * (1.0 - bottom_crop_fraction)))
        valid = depth[:valid_height]
        valid = valid[np.isfinite(valid) & (valid > 0.05) & (valid <= depth_max_m)]
        depth_statistics[plan_id] = (
            float(np.min(valid)) if valid.size else 0.0,
            float(np.percentile(valid, 5)) if valid.size else 0.0,
        )

    actions, states, rewards, risks, continues = [], [], [], [], []
    for index, plan_id in enumerate(available):
        row = grouped[plan_id][0]
        next_row = grouped[available[min(index + 1, len(available) - 1)]][0]
        position = np.asarray([_number(row, axis) for axis in ("x", "y", "z")], dtype=np.float32)
        waypoint = min(max(int(_number(row, "target_waypoint_idx")), 0), len(route) - 1)
        actions.append(action_from_delta(route[waypoint] - position))
        velocity = np.asarray([_number(row, axis) for axis in ("vx", "vy", "vz")], dtype=np.float32)
        minimum, p05 = depth_statistics[plan_id]
        states.append(
            np.asarray(
                [
                    *(velocity / 10.0),
                    np.linalg.norm(velocity) / 10.0,
                    p05 / depth_max_m,
                    minimum / depth_max_m,
                    _number(row, "local_goal_distance_m") / 80.0,
                    _number(row, "final_goal_distance_m") / 200.0,
                    waypoint / max(len(route) - 1, 1),
                    _number(row, "terminal_mode"),
                    _number(row, "selected_score") / 10.0,
                    _number(row, "velocity_scale", 1.0),
                ],
                dtype=np.float32,
            )
        )
        progress = _number(row, "final_goal_distance_m") - _number(next_row, "final_goal_distance_m")
        collision = _number(next_row, "collision_events") > _number(row, "collision_events")
        near_miss = depth_statistics[available[min(index + 1, len(available) - 1)]][1] < risk_depth_m
        risk = float(collision or near_miss)
        rewards.append(float(np.clip(progress / 5.0 - 2.0 * risk, -5.0, 5.0)))
        risks.append(risk)
        continues.append(float(index + 1 < len(available) and not collision))
    return VisualFlightEpisode(
        run_dir=run_dir,
        plan_ids=tuple(available),
        actions=np.stack(actions),
        states=np.stack(states),
        rewards=np.asarray(rewards, dtype=np.float32),
        risks=np.asarray(risks, dtype=np.float32),
        continues=np.asarray(continues, dtype=np.float32),
        depth_max_m=float(depth_max_m),
    )


@lru_cache(maxsize=1024)
def _load_observation(
    rgb_path: str,
    depth_path: str,
    image_size: int,
    depth_max_m: float,
    bottom_crop_fraction: float,
) -> np.ndarray:
    rgb_source = Image.open(rgb_path).convert("RGB")
    rgb_height = max(1, round(rgb_source.height * (1.0 - bottom_crop_fraction)))
    rgb = rgb_source.crop((0, 0, rgb_source.width, rgb_height)).resize((image_size, image_size))
    rgb_array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    depth = np.load(depth_path).astype(np.float32)
    depth_height = max(1, round(depth.shape[0] * (1.0 - bottom_crop_fraction)))
    depth = depth[:depth_height]
    depth = np.clip(depth / depth_max_m, 0.0, 1.0)
    depth_image = Image.fromarray((depth * 255).astype(np.uint8)).resize((image_size, image_size))
    depth_array = np.asarray(depth_image, dtype=np.float32)[None] / 255.0
    return np.concatenate([rgb_array, depth_array]).astype(np.float32)


class VisualSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        episodes: list[VisualFlightEpisode],
        *,
        sequence_length: int = 16,
        stride: int = 4,
        image_size: int = 64,
        bottom_crop_fraction: float = 1.0 / 3.0,
    ) -> None:
        self.episodes = episodes
        self.sequence_length = int(sequence_length)
        self.image_size = int(image_size)
        self.bottom_crop_fraction = float(bottom_crop_fraction)
        self.windows: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(episodes):
            for start in range(0, len(episode) - self.sequence_length + 1, max(int(stride), 1)):
                self.windows.append((episode_index, start))
        if not self.windows:
            raise ValueError("no episode is long enough for the requested sequence length")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, start = self.windows[index]
        episode = self.episodes[episode_index]
        stop = start + self.sequence_length
        observations = []
        for plan_id in episode.plan_ids[start:stop]:
            observations.append(
                _load_observation(
                    str(episode.run_dir / "observations" / "rgb" / f"plan_{plan_id:04d}.png"),
                    str(episode.run_dir / "observations" / "depth" / f"plan_{plan_id:04d}.npy"),
                    self.image_size,
                    episode.depth_max_m,
                    self.bottom_crop_fraction,
                )
            )
        return {
            "observations": torch.from_numpy(np.stack(observations)),
            "actions": torch.from_numpy(episode.actions[start:stop]),
            "states": torch.from_numpy(episode.states[start:stop]),
            "rewards": torch.from_numpy(episode.rewards[start:stop]),
            "risks": torch.from_numpy(episode.risks[start:stop]),
            "continues": torch.from_numpy(episode.continues[start:stop]),
        }


class DirectTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        episodes: list[VisualFlightEpisode],
        *,
        image_size: int = 64,
        bottom_crop_fraction: float = 1.0 / 3.0,
    ) -> None:
        self.episodes = episodes
        self.image_size = int(image_size)
        self.bottom_crop_fraction = float(bottom_crop_fraction)
        self.transitions = [
            (episode_index, step)
            for episode_index, episode in enumerate(episodes)
            for step in range(len(episode) - 1)
        ]
        if not self.transitions:
            raise ValueError("no direct world-model transitions are available")

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, step = self.transitions[index]
        episode = self.episodes[episode_index]
        plan_id = episode.plan_ids[step]
        observation = _load_observation(
            str(episode.run_dir / "observations" / "rgb" / f"plan_{plan_id:04d}.png"),
            str(episode.run_dir / "observations" / "depth" / f"plan_{plan_id:04d}.npy"),
            self.image_size,
            episode.depth_max_m,
            self.bottom_crop_fraction,
        )
        return {
            "observations": torch.from_numpy(observation),
            "states": torch.from_numpy(episode.states[step]),
            "actions": torch.from_numpy(episode.actions[step]),
            "next_states": torch.from_numpy(episode.states[step + 1]),
            "rewards": torch.as_tensor(episode.rewards[step]),
        }


def dataset_manifest(episodes: list[VisualFlightEpisode]) -> dict[str, object]:
    return {
        "format": "urbanfly-visual-flight-dataset-v1",
        "runs": [str(episode.run_dir) for episode in episodes],
        "frames": sum(len(episode) for episode in episodes),
        "risk_frames": int(sum(float(episode.risks.sum()) for episode in episodes)),
    }
