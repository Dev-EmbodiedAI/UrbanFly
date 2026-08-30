from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math


Vector3 = list[float]


def _vector3(name: str, value: Vector3) -> None:
    if len(value) != 3 or not all(math.isfinite(float(item)) for item in value):
        raise ValueError(f"{name} must contain three finite numbers")


@dataclass
class Step:
    time_s: float
    position: Vector3
    velocity: Vector3
    action_delta: Vector3
    goal_distance_m: float
    min_depth_m: float
    p05_depth_m: float
    collision: bool
    replan_step: int = 0
    target_waypoint_idx: int = 0
    rgb_path: str | None = None
    depth_path: str | None = None

    def validate(self) -> None:
        _vector3("position", self.position)
        _vector3("velocity", self.velocity)
        _vector3("action_delta", self.action_delta)
        if self.time_s < 0 or self.goal_distance_m < 0:
            raise ValueError("time and goal distance must be non-negative")


@dataclass
class Episode:
    episode_id: str
    scene_id: str
    instruction: str
    route: list[Vector3]
    steps: list[Step]
    split: str = "train"
    instruction_source: str = "geometry_template"
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "urbanfly-vln-0.1"

    def validate(self) -> None:
        if not self.episode_id or not self.instruction.strip():
            raise ValueError("episode_id and instruction are required")
        if self.split not in {"train", "val_seen", "val_unseen", "test"}:
            raise ValueError(f"unsupported split: {self.split}")
        if len(self.route) < 2 or len(self.steps) < 2:
            raise ValueError("an episode needs at least two route points and two steps")
        for point in self.route:
            _vector3("route point", point)
        previous_time = -1.0
        for step in self.steps:
            step.validate()
            if step.time_s < previous_time:
                raise ValueError("step timestamps must be monotonic")
            previous_time = step.time_s

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Episode":
        payload = dict(payload)
        payload["steps"] = [Step(**item) for item in payload["steps"]]
        episode = cls(**payload)
        episode.validate()
        return episode

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> "Episode":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
