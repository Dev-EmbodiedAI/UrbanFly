from __future__ import annotations

import numpy as np

from uav_wm_navigation.planners.base import CandidatePlanner, PlanningContext
from uav_wm_navigation.planners.polynomial import sample_quintic
from uav_wm_navigation.types import CandidateTrajectory


class MockCandidatePlanner(CandidatePlanner):
    def __init__(self, candidate_count: int = 5, horizon_steps: int = 16, duration: float = 2.0, speed: float = 2.0) -> None:
        self.candidate_count = candidate_count
        self.horizon_steps = horizon_steps
        self.duration = duration
        self.speed = speed

    def plan(self, context: PlanningContext) -> list[CandidateTrajectory]:
        state = context.state
        direction = np.asarray(context.local_goal_nwu, dtype=np.float64) - state.position
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        base_yaw = float(np.arctan2(direction[1], direction[0]))
        offsets = np.linspace(-0.55, 0.55, self.candidate_count)
        candidates = []
        for index, offset in enumerate(offsets):
            heading = np.array([np.cos(base_yaw + offset), np.sin(base_yaw + offset), 0.15 * direction[2]])
            heading /= max(float(np.linalg.norm(heading)), 1e-6)
            endpoint = state.position + heading * self.speed * self.duration
            end_velocity = heading * self.speed
            positions, velocities, accelerations = sample_quintic(
                state.position, state.linear_velocity, state.linear_acceleration,
                endpoint, end_velocity, np.zeros(3), self.duration, self.horizon_steps,
            )
            candidates.append(CandidateTrajectory(
                trajectory_id=f"mock-{index}", positions=positions, velocities=velocities,
                accelerations=accelerations, duration=self.duration,
                yopo_cost=float(abs(offset)), valid_mask=np.ones(self.horizon_steps, dtype=bool),
                metadata={"source": "mock", "heading_offset_rad": float(offset)},
            ))
        return candidates

