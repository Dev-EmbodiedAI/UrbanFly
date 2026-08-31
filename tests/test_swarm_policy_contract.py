import math

import numpy as np
import pytest

from backend.integrations.swarm_policy import (
    CanonicalDroneState,
    SwarmPolicyEncoder,
    enu_to_urbanfly_world,
    normalized_swarm_action_to_urbanfly,
    urbanfly_world_to_enu,
)


def _drone(index, position, velocity=(0.0, 0.0, 0.0)):
    return CanonicalDroneState(
        drone_id=f"UAV-{index}",
        position_enu_m=np.asarray(position),
        orientation_rpy_rad=np.asarray([0.01 * index, -0.02 * index, 0.1 * index]),
        linear_velocity_enu_mps=np.asarray(velocity),
        angular_velocity_rpy_radps=np.asarray([0.0, 0.0, 0.01 * index]),
        altitude_distance_m=2.0 + index,
    )


def test_urbanfly_enu_roundtrip_and_yaw_conversion():
    source = np.asarray([10.0, 30.0, -20.0])
    np.testing.assert_allclose(urbanfly_world_to_enu(source), [10.0, -20.0, 30.0])
    np.testing.assert_allclose(enu_to_urbanfly_world(urbanfly_world_to_enu(source)), source)
    drone = CanonicalDroneState.from_urbanfly(
        drone_id="UAV-A",
        position_eun_m=source,
        roll_rad=0.1,
        pitch_rad=-0.2,
        yaw_degrees=90.0,
        velocity_eun_mps=[1.0, 3.0, 2.0],
        angular_velocity_eun_radps=[0.1, 0.3, 0.2],
        altitude_distance_m=12.0,
    )
    assert math.isclose(float(drone.orientation_rpy_rad[2]), math.pi / 2, rel_tol=1e-6)


def test_encoder_matches_190d_layout_shared_clue_and_nearest_teammates():
    drones = [
        _drone(0, [0.0, 0.0, 10.0], [1.0, 0.0, 0.0]),
        _drone(1, [5.0, 0.0, 10.0], [0.0, 1.0, 0.0]),
        _drone(2, [2.0, 0.0, 10.0], [0.0, 0.0, 1.0]),
    ]
    history = np.arange(3 * 25 * 5, dtype=np.float32).reshape(3, 25, 5) / 1000.0
    depth = np.full((3, 128, 128), 10.0, dtype=np.float32)
    observation = SwarmPolicyEncoder().encode(
        depth_m=depth,
        drones=drones,
        shared_clue_enu_m=[20.0, 5.0, 0.0],
        action_history=history,
    )
    assert observation.depth.shape == (3, 128, 128, 1)
    assert observation.state.shape == (3, 190)
    np.testing.assert_allclose(observation.depth, 0.5)
    np.testing.assert_allclose(observation.state[0, 12:137], history[0].reshape(-1))
    np.testing.assert_allclose(observation.state[0, 138:141], [20.0, 5.0, -10.0])
    # UAV-2 (2 m) precedes UAV-1 (5 m) in nearest-neighbour slots.
    np.testing.assert_allclose(observation.state[0, 141:148], [2, 0, 0, -1, 0, 1, 1])
    np.testing.assert_allclose(observation.state[0, 148:155], [5, 0, 0, -1, 1, 0, 1])
    assert np.count_nonzero(observation.state[0, 155:190]) == 0


def test_depth_invalid_values_become_far_plane_and_altitude_is_clipped():
    depth = np.full((2, 128, 128, 1), 40.0, dtype=np.float32)
    depth[0, 0, 0, 0] = np.nan
    observation = SwarmPolicyEncoder(max_depth_m=20.0).encode(
        depth_m=depth,
        drones=[_drone(0, [0, 0, 0]), _drone(1, [1, 0, 0])],
        shared_clue_enu_m=[0, 0, 0],
    )
    assert float(observation.depth.min()) == 1.0
    assert float(observation.depth.max()) == 1.0


def test_action_mapping_validates_bounds_and_preserves_coordinate_meaning():
    action = np.asarray([[1, -0.5, 0.25, 0.8, 0.5], [-1, 0, -0.25, 0, -1]], dtype=np.float32)
    mapped = normalized_swarm_action_to_urbanfly(action)
    np.testing.assert_allclose(mapped["direction_eun"][0], [1.0, 0.25, -0.5])
    np.testing.assert_allclose(mapped["absolute_yaw_degrees"], [90.0, -180.0])
    with pytest.raises(ValueError):
        normalized_swarm_action_to_urbanfly(np.asarray([[2, 0, 0, 1, 0]] * 2))
