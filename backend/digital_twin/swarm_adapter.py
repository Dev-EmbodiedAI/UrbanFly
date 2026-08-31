"""Swarm ``cf_swarm_autopilot`` 的程序化数字孪生环境适配器。

上游源码保持外部依赖；本文件不复制 Swarm 场景或修改其物理、碰撞和评分。
``privileged_goal_mode`` 仅用于数字孪生起终点任务演示，绝不能写成公开 contract
benchmark 成绩；正式 benchmark 仍只能使用 shared noisy clue。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

from .contracts import DigitalTwinFeedback, DigitalTwinMission, DigitalTwinObservation


_DISTANCE_RANGES = {
    1: (22.0, 40.0),
    2: (28.0, 65.0),
    3: (65.0, 95.0),
    4: (28.0, 50.0),
    6: (22.0, 40.0),
}


class SwarmDigitalTwinAdapter:
    environment_names = {1: "city", 2: "open", 3: "mountain", 4: "village", 6: "forest"}

    def __init__(
        self,
        upstream_root: str | Path,
        *,
        challenge_type: int,
        seed: int,
        num_drones: int = 2,
        sim_dt: float = 1.0 / 50.0,
        gui: bool = False,
    ) -> None:
        self.upstream_root = Path(upstream_root).resolve()
        if not (self.upstream_root / "swarm" / "__init__.py").is_file():
            raise FileNotFoundError(f"Swarm upstream root is invalid: {self.upstream_root}")
        if challenge_type not in self.environment_names:
            raise ValueError("Swarm environment must be city/open/mountain/village/forest")
        if not 2 <= num_drones <= 8:
            raise ValueError("Swarm drone count must be in [2,8]")
        self.challenge_type = int(challenge_type)
        self.seed = int(seed)
        self.num_drones = int(num_drones)
        self.sim_dt = float(sim_dt)
        self.gui = bool(gui)
        self.environment_id = f"swarm:{self.environment_names[self.challenge_type]}"
        self.episode_id = f"{self.environment_id}:seed-{self.seed}:n-{self.num_drones}"
        self.env: Any | None = None
        self.task: Any | None = None
        self.sequence = 0
        self._last_timestamp_s: float | None = None
        self.mission: DigitalTwinMission | None = None

    def reset(self) -> tuple[DigitalTwinMission, DigitalTwinObservation]:
        if self.env is not None:
            raise RuntimeError("adapter is already active")
        root = str(self.upstream_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from swarm.utils.env_factory import make_env
        from swarm.validator.task_gen import screening_task

        self.task = screening_task(
            self.sim_dt,
            self.seed,
            challenge_type=self.challenge_type,
            distance_range=_DISTANCE_RANGES[self.challenge_type],
            family_id="cf_swarm_autopilot",
            moving_platform=False,
            n_drones=self.num_drones,
        )
        self.env = make_env(self.task, gui=self.gui)
        observation, info = self.env.reset(seed=self.task.map_seed)
        starts = np.asarray(
            getattr(self.env, "_swarm_adjusted_starts", self.task.starts), dtype=np.float32
        )
        goals = np.asarray(self.env.GOAL_POSES, dtype=np.float32)
        self.mission = DigitalTwinMission(
            environment_id=self.environment_id,
            episode_id=self.episode_id,
            starts_enu_m=starts,
            goals_enu_m=goals,
            agent_provider="deterministic_goal_assignment",
            privileged_goal_mode=True,
            metadata={
                "scope": "digital_twin_start_to_goal_navigation",
                "benchmark_eligible": False,
                "benchmark_exclusion_reason": "exact goals are Agent mission inputs",
                "challenge_type": self.challenge_type,
                "map_seed": self.seed,
                "upstream_family": "cf_swarm_autopilot",
            },
        )
        self.sequence = 0
        self._last_timestamp_s = 0.0
        return self.mission, self._observation(observation, info)

    def step(self, action: np.ndarray) -> DigitalTwinFeedback:
        if self.env is None:
            raise RuntimeError("adapter must be reset before step")
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (self.num_drones, 5) or not np.isfinite(values).all():
            raise ValueError(f"action must be finite [{self.num_drones},5]")
        if np.any(values[:, :3] < -1.0) or np.any(values[:, :3] > 1.0):
            raise ValueError("direction is outside [-1,1]")
        if np.any(values[:, 3] < 0.0) or np.any(values[:, 3] > 1.0):
            raise ValueError("speed is outside [0,1]")
        if np.any(values[:, 4] < -1.0) or np.any(values[:, 4] > 1.0):
            raise ValueError("yaw is outside [-1,1]")
        observation, reward, terminated, truncated, info = self.env.step(values)
        self.sequence += 1
        wrapped = self._observation(observation, info)
        timestamp = wrapped.timestamp_s
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise RuntimeError("Swarm observation timestamp did not advance")
        self._last_timestamp_s = timestamp
        success = tuple(bool(item) for item in info.get("per_drone_success", [False] * self.num_drones))
        collisions = tuple(bool(item) for item in info.get("per_drone_collision", [False] * self.num_drones))
        reasons = tuple(str(item) for item in info.get("per_drone_failure_reason", ["NONE"] * self.num_drones))
        return DigitalTwinFeedback(
            observation=wrapped,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            per_drone_success=success,
            per_drone_collision=collisions,
            per_drone_failure_reason=reasons,
            raw_info=dict(info),
        )

    def score(self, info: dict[str, Any]) -> dict[str, Any]:
        if self.env is None or self.task is None:
            raise RuntimeError("adapter is not active")
        return self.env.family_runtime.score_swarm(self.task, info)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _observation(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> DigitalTwinObservation:
        depth = np.asarray(observation["depth"], dtype=np.float32)
        state = np.asarray(observation["state"], dtype=np.float32)
        return DigitalTwinObservation(
            environment_id=self.environment_id,
            episode_id=self.episode_id,
            sequence=self.sequence,
            timestamp_s=self.sequence * self.sim_dt,
            depth=depth,
            state=state,
            metadata={"native_info": dict(info), "public_contract": "submission_zip.v1"},
        )
