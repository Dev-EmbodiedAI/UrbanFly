from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ScriptedDynamicActor:
    actor_id: int
    actor_type: str
    origin: np.ndarray
    direction: np.ndarray
    speed_mps: float
    half_extent: np.ndarray
    travel_m: float
    phase_s: float
    zone_type: str
    position: np.ndarray
    velocity: np.ndarray

    def update(self, sim_time_s: float) -> None:
        period = 2.0 * self.travel_m / self.speed_mps
        phase = (float(sim_time_s) + self.phase_s) % period
        forward = phase <= period * 0.5
        distance = self.speed_mps * (phase if forward else period - phase)
        sign = 1.0 if forward else -1.0
        self.position = self.origin + self.direction * distance
        self.velocity = self.direction * (sign * self.speed_mps)

    def to_dict(self) -> dict:
        return {
            "id": self.actor_id,
            "actor_type": self.actor_type,
            "pos": self.position.tolist(),
            "vel": self.velocity.tolist(),
            "bbox_extent": self.half_extent.tolist(),
            "scripted": True,
            "zone_type": self.zone_type,
        }


class ScriptedActorField:
    """Deterministic replayable street actors for RGB-D and CPA labels."""

    def __init__(self) -> None:
        self.actors: list[ScriptedDynamicActor] = []
        self.seed = 20260731

    def reset(self, bounds: tuple[np.ndarray, np.ndarray], *, seed: int, density: float = 1.0) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        lower, upper = (np.asarray(item, dtype=float) for item in bounds)
        span = upper - lower
        if np.any(span <= 0):
            self.actors = []
            return
        if float(density) <= 0.0:
            self.actors = []
            return
        count_vehicle = max(1, int(round(12 * float(np.clip(density, 0.0, 2.0)))))
        count_pedestrian = max(1, int(round(8 * float(np.clip(density, 0.0, 2.0)))))
        ground = lower[1] + 0.9
        actors: list[ScriptedDynamicActor] = []
        for index in range(count_vehicle + count_pedestrian):
            vehicle = index < count_vehicle
            horizontal_x = bool(index % 2)
            direction = np.array([1.0, 0.0, 0.0]) if horizontal_x else np.array([0.0, 0.0, 1.0])
            travel = float((span[0] if horizontal_x else span[2]) * rng.uniform(0.18, 0.42))
            origin = np.array([
                rng.uniform(lower[0] + 0.1 * span[0], upper[0] - 0.1 * span[0]),
                ground if vehicle else ground - 0.25,
                rng.uniform(lower[2] + 0.1 * span[2], upper[2] - 0.1 * span[2]),
            ])
            extent = np.array([2.2, 0.9, 0.95]) if vehicle else np.array([0.35, 0.9, 0.35])
            speed = float(rng.uniform(3.0, 8.0) if vehicle else rng.uniform(1.0, 2.0))
            actor = ScriptedDynamicActor(
                actor_id=index + 1, actor_type="vehicle" if vehicle else "pedestrian",
                origin=origin, direction=direction, speed_mps=speed, half_extent=extent,
                travel_m=max(travel, 5.0), phase_s=float(rng.uniform(0.0, 60.0)),
                zone_type="transport" if vehicle else "public_space",
                position=origin.copy(), velocity=direction * speed,
            )
            actor.update(0.0)
            actors.append(actor)
        self.actors = actors

    def update(self, sim_time_s: float) -> None:
        for actor in self.actors:
            actor.update(sim_time_s)

    def snapshot(self) -> list[dict]:
        return [actor.to_dict() for actor in self.actors]
