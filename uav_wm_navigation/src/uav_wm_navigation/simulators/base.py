from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from uav_wm_navigation.types import ActorState, CandidateTrajectory, SensorFrame, VehicleState


class SimulatorAdapter(ABC):
    """Canonical single-UAV simulator boundary.

    Every implementation exposes world-NWU state/commands and metric sensors.
    The historical method names remain abstract for compatibility; the clearer
    aliases below are the contract new AirSim/Isaac/Gazebo adapters should use.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def takeoff(self) -> None: ...

    @abstractmethod
    def land(self) -> None: ...

    @abstractmethod
    def get_depth(self) -> SensorFrame: ...

    @abstractmethod
    def get_rgb(self) -> np.ndarray | None: ...

    def get_visualization_rgb(self) -> np.ndarray | None:
        return self.get_rgb()

    @abstractmethod
    def get_kinematics(self) -> VehicleState: ...

    @abstractmethod
    def get_collision_info(self) -> dict[str, object]: ...

    @abstractmethod
    def get_timestamp(self) -> float: ...

    @abstractmethod
    def set_goal(self, goal_nwu: np.ndarray) -> None: ...

    @abstractmethod
    def execute_velocity_command(self, velocity_nwu: np.ndarray, yaw_rate: float, duration: float) -> None: ...

    # Backward-compatible canonical aliases. Algorithms should normally use
    # UAVWorldModelEnv rather than call these methods directly.
    def get_vehicle_state(self) -> VehicleState:
        return self.get_kinematics()

    def get_sensor_frame(self) -> SensorFrame:
        return self.get_depth()

    def get_collision(self) -> bool:
        return bool(self.get_collision_info().get("has_collided", False))

    def get_sim_time(self) -> float:
        return self.get_timestamp()

    def send_velocity_command(
        self,
        forward_mps: float,
        left_mps: float,
        up_mps: float,
        yaw_rate_rps: float,
        duration_s: float,
    ) -> None:
        """Send a canonical body-FLU command through the NWU backend API."""

        state = self.get_vehicle_state()
        x, y, z, w = state.orientation_xyzw.astype(np.float64)
        yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        velocity_nwu = np.asarray(
            [
                cosine * forward_mps - sine * left_mps,
                sine * forward_mps + cosine * left_mps,
                up_mps,
            ],
            dtype=np.float64,
        )
        self.execute_velocity_command(velocity_nwu, float(yaw_rate_rps), float(duration_s))

    def execute_trajectory(self, trajectory: CandidateTrajectory, dt: float) -> None:
        for index in np.flatnonzero(trajectory.valid_mask):
            self.execute_velocity_command(trajectory.velocities[index], 0.0, dt)

    def get_actor_states(self) -> list[ActorState]:
        """Return dynamic/static actor snapshots when the backend exposes them."""
        return []

    def publish_planner_visualization(
        self,
        candidates: tuple[CandidateTrajectory, ...] | list[CandidateTrajectory],
        *,
        selected_index: int,
        decision_sequence: int,
        metadata: dict[str, object],
    ) -> None:
        """Publish auditable planner telemetry when a simulator has a live UI.

        This is deliberately a no-op for headless simulators.  It never alters
        the selected trajectory or the executed command.
        """

        return None

    def configure_scenario(self, scenario: str, difficulty: str, seed: int) -> None:
        return None

    def step_scenario(self, elapsed_s: float) -> None:
        """Advance deterministic scripted actors to an episode-relative time."""
        return None

    def set_initial_pose(self, position_nwu: np.ndarray) -> None:
        return None

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def continue_simulation(self) -> None: ...

    def close(self) -> None:
        return None
