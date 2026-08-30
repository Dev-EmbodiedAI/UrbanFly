from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from uav_wm_navigation.types import BodyVelocityAction, EpisodeSpec, WorldModelObservation


class ContinuousWorldModelPolicy(ABC):
    """Common runtime API for learned 5 Hz UrbanFly policies.

    This interface is intentionally separate from the legacy YOPO candidate
    reranker protocol.  Implementations must produce a continuous normalized
    body-frame action, not merely reorder externally generated trajectories.
    """

    @abstractmethod
    def reset(self, episode: str | EpisodeSpec) -> None:
        ...


def episode_id_from_spec(episode: str | EpisodeSpec) -> str:
    return episode.episode_id if isinstance(episode, EpisodeSpec) else str(episode)

    @abstractmethod
    def observe(self, observation: WorldModelObservation) -> None:
        ...

    @abstractmethod
    def act(self, deterministic: bool = True) -> BodyVelocityAction:
        ...

    @abstractmethod
    def predict(self, action_sequences: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate normalized action sequences shaped [batch, horizon, 4]."""
        ...

    @abstractmethod
    def diagnostics(self) -> dict[str, Any]:
        ...


def validate_action_sequences(action_sequences: np.ndarray) -> np.ndarray:
    value = np.asarray(action_sequences, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 4:
        raise ValueError("action_sequences must have shape [batch, horizon, 4]")
    if not np.isfinite(value).all():
        raise ValueError("action_sequences contain non-finite values")
    return np.clip(value, -1.0, 1.0)
