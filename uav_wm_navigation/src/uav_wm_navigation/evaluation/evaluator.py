from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class EpisodeMetrics:
    success: bool
    collision: bool
    dynamic_collision: bool
    minimum_clearance: float
    path_length: float
    flight_time: float
    goal_progress: float
    replanning_count: int
    emergency_stop_count: int
    inference_latency_ms: float
    control_frequency_hz: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

