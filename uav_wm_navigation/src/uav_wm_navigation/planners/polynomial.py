from __future__ import annotations

import numpy as np


def sample_quintic(
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    end_position: np.ndarray,
    end_velocity: np.ndarray,
    end_acceleration: np.ndarray,
    duration: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if duration <= 0 or steps < 2:
        raise ValueError("duration must be positive and steps must be at least two")
    p0, v0, a0, pf, vf, af = [np.asarray(v, dtype=np.float64) for v in (
        start_position, start_velocity, start_acceleration, end_position, end_velocity, end_acceleration
    )]
    c0 = p0
    c1 = v0
    c2 = a0 / 2.0
    t = float(duration)
    matrix = np.array([[t**3, t**4, t**5], [3*t**2, 4*t**3, 5*t**4], [6*t, 12*t**2, 20*t**3]])
    rhs = np.stack([pf - (c0 + c1*t + c2*t*t), vf - (c1 + 2*c2*t), af - 2*c2])
    c3, c4, c5 = np.linalg.solve(matrix, rhs)
    times = np.linspace(0.0, t, steps, dtype=np.float64)[:, None]
    positions = c0 + c1*times + c2*times**2 + c3*times**3 + c4*times**4 + c5*times**5
    velocities = c1 + 2*c2*times + 3*c3*times**2 + 4*c4*times**3 + 5*c5*times**4
    accelerations = 2*c2 + 6*c3*times + 12*c4*times**2 + 20*c5*times**3
    return positions.astype(np.float32), velocities.astype(np.float32), accelerations.astype(np.float32)

