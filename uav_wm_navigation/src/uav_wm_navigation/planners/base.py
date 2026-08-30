from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from uav_wm_navigation.types import CandidateTrajectory, SensorFrame, VehicleState


@dataclass(slots=True)
class PlanningContext:
    sensor: SensorFrame
    state: VehicleState
    local_goal_nwu: np.ndarray


class CandidatePlanner(ABC):
    @abstractmethod
    def plan(self, context: PlanningContext) -> list[CandidateTrajectory]: ...

