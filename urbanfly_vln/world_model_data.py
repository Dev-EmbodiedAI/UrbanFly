from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .risk_world_model import instruction_embedding


BASE_FEATURE_DIM = 10
TARGET_DIM = 6


@dataclass(frozen=True)
class WorldModelSample:
    features: np.ndarray
    target: np.ndarray
    risk: float
    source: str
    scene_id: str
    instruction: str
    replan_step: int


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    strategy: str
    train_sources: tuple[str, ...]
    validation_sources: tuple[str, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        result = float(row.get(key, default))
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def samples_from_run(
    run_dir: Path,
    *,
    language_dimensions: int = 16,
    risk_horizon: int = 3,
    near_miss_depth_m: float = 5.0,
) -> list[WorldModelSample]:
    """Build action-conditioned transitions without crossing replan boundaries."""
    run_dir = run_dir.resolve()
    trajectory_path = run_dir / "long_range_trajectory.csv"
    route_path = run_dir / "global_route.csv"
    summary_path = run_dir / "long_range_summary.json"
    if not trajectory_path.exists() or not route_path.exists():
        raise FileNotFoundError(f"{run_dir} must contain trajectory and global route CSV files")

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    instruction = str(summary.get("instruction", "")).strip()
    scene_id = str(summary.get("carla_map") or Path(str(summary.get("dataset_root", "unknown"))).name)
    language = instruction_embedding(instruction, language_dimensions).astype(np.float32)
    route = np.asarray(
        [[_value(row, axis) for axis in ("x", "y", "z")] for row in _read_csv(route_path)],
        dtype=np.float32,
    )
    if len(route) < 2:
        raise ValueError(f"{route_path} must contain at least two route points")

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in _read_csv(trajectory_path):
        grouped.setdefault(int(_value(row, "replan_step")), []).append(row)
    ordered = [(key, grouped[key]) for key in sorted(grouped)]
    samples: list[WorldModelSample] = []
    for group_index, ((replan_step, current_group), (_, next_group)) in enumerate(zip(ordered[:-1], ordered[1:])):
        current = current_group[-1]
        next_row = next_group[-1]
        position = np.asarray([_value(current, axis) for axis in ("x", "y", "z")], dtype=np.float32)
        next_position = np.asarray([_value(next_row, axis) for axis in ("x", "y", "z")], dtype=np.float32)
        waypoint_idx = min(max(int(_value(current, "target_waypoint_idx")), 0), len(route) - 1)
        action_vector = route[waypoint_idx] - position
        action_distance = float(np.linalg.norm(action_vector))
        speed = float(np.linalg.norm([_value(current, axis) for axis in ("vx", "vy", "vz")]))
        next_speed = float(np.linalg.norm([_value(next_row, axis) for axis in ("vx", "vy", "vz")]))
        p05 = _value(current_group[0], "p05_depth_m", 20.0)
        next_p05 = _value(next_group[0], "p05_depth_m", 20.0)
        state = np.asarray(
            [
                speed,
                _value(current, "vz"),
                p05,
                _value(current, "local_goal_distance_m", action_distance),
                _value(current, "final_goal_distance_m"),
                waypoint_idx / max(len(route) - 1, 1),
            ],
            dtype=np.float32,
        )
        action = np.concatenate([action_vector, [action_distance]]).astype(np.float32)
        delta_position = next_position - position
        direction = action_vector / max(action_distance, 1e-6)
        projected_progress = float(np.dot(delta_position, direction))
        target = np.asarray([*delta_position, next_speed, next_p05, projected_progress], dtype=np.float32)

        future_groups = ordered[group_index : group_index + max(risk_horizon, 1) + 1]
        future_rows = [row for _, group in future_groups for row in group]
        baseline_collisions = _value(current_group[0], "collision_events")
        collision_risk = any(_value(row, "collision_events") > baseline_collisions for row in future_rows)
        depth_risk = any(_value(row, "p05_depth_m", 20.0) < near_miss_depth_m for row in future_rows)
        features = np.concatenate([state, action, language]).astype(np.float32)
        samples.append(
            WorldModelSample(
                features=features,
                target=target,
                risk=float(collision_risk or depth_risk),
                source=run_dir.name,
                scene_id=scene_id,
                instruction=instruction,
                replan_step=replan_step,
            )
        )
    return samples


def stack_samples(samples: list[WorldModelSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("at least one world-model sample is required")
    return (
        np.stack([sample.features for sample in samples]).astype(np.float32),
        np.stack([sample.target for sample in samples]).astype(np.float32),
        np.asarray([sample.risk for sample in samples], dtype=np.float32),
    )


def grouped_split(
    samples: list[WorldModelSample],
    *,
    validation_sources: set[str] | None = None,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> DatasetSplit:
    """Split whole runs when possible; a single run uses a chronological tail split."""
    if len(samples) < 2:
        raise ValueError("at least two samples are required for a train/validation split")
    sources = sorted({sample.source for sample in samples})
    explicit = set(validation_sources or ())
    unknown = explicit.difference(sources)
    if unknown:
        raise ValueError(f"unknown validation sources: {sorted(unknown)}")

    if explicit:
        val_sources = explicit
        strategy = "explicit_run_holdout"
    elif len(sources) > 1:
        rng = np.random.default_rng(seed)
        shuffled = list(sources)
        rng.shuffle(shuffled)
        holdout_count = min(max(1, round(len(sources) * validation_fraction)), len(sources) - 1)
        val_sources = set(shuffled[-holdout_count:])
        strategy = "grouped_run_holdout"
    else:
        split_at = min(max(1, int(len(samples) * (1.0 - validation_fraction))), len(samples) - 1)
        return DatasetSplit(
            train_indices=np.arange(split_at, dtype=np.int64),
            validation_indices=np.arange(split_at, len(samples), dtype=np.int64),
            strategy="chronological_tail_holdout",
            train_sources=(sources[0],),
            validation_sources=(sources[0],),
        )

    train = np.asarray([i for i, sample in enumerate(samples) if sample.source not in val_sources], dtype=np.int64)
    validation = np.asarray([i for i, sample in enumerate(samples) if sample.source in val_sources], dtype=np.int64)
    if not len(train) or not len(validation):
        raise ValueError("split produced an empty train or validation partition")
    return DatasetSplit(
        train_indices=train,
        validation_indices=validation,
        strategy=strategy,
        train_sources=tuple(sorted({samples[i].source for i in train})),
        validation_sources=tuple(sorted({samples[i].source for i in validation})),
    )
