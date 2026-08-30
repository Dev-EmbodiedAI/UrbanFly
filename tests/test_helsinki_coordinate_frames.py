import math
from types import SimpleNamespace

import numpy as np

from backend.engine.helsinki_frames import (
    backend_yaw_rate_to_body_flu_radians,
    backend_delta_to_body_flu,
    backend_world_to_enu,
    body_flu_to_enu,
    body_flu_yaw_rate_to_backend_degrees,
    coordinate_metadata,
    enu_delta_to_body_flu,
    enu_to_backend_world,
)
from backend.engine.simulator import Simulator


def test_asset_backend_and_canonical_world_directions() -> None:
    origin = np.asarray([10.0, 20.0, 30.0])
    assert np.allclose(enu_to_backend_world(origin), [10.0, 30.0, -20.0])
    assert np.allclose(backend_world_to_enu([10.0, 30.0, -20.0]), origin)

    backend_origin = np.asarray([0.0, 0.0, 0.0])
    assert np.allclose(
        backend_world_to_enu(backend_origin + [1.0, 0.0, 0.0]),
        [1.0, 0.0, 0.0],
    )  # east
    assert np.allclose(
        backend_world_to_enu(backend_origin + [0.0, 0.0, -1.0]),
        [0.0, 1.0, 0.0],
    )  # north is negative renderer Z
    assert np.allclose(
        backend_world_to_enu(backend_origin + [0.0, 1.0, 0.0]),
        [0.0, 0.0, 1.0],
    )  # up


def test_body_flu_is_right_handed_and_roundtrips() -> None:
    yaw = math.radians(30.0)
    value = np.asarray([4.0, -2.0, 1.0])
    assert np.allclose(
        enu_delta_to_body_flu(body_flu_to_enu(value, yaw), yaw),
        value,
    )
    # Backend yaw zero faces east; geographic left is renderer negative Z.
    assert np.allclose(backend_delta_to_body_flu([0.0, 0.0, -5.0], 0.0), [0.0, 5.0, 0.0])
    positive_left = math.radians(45.0)
    backend_rate = body_flu_yaw_rate_to_backend_degrees(positive_left)
    assert backend_rate == -45.0
    assert backend_yaw_rate_to_body_flu_radians(backend_rate) == positive_left


def test_dataset_coordinate_metadata_is_complete() -> None:
    metadata = coordinate_metadata()
    assert metadata["world_frame"]["axes"] == ["east", "north", "up"]
    assert metadata["body_frame"]["axes"] == ["forward", "left", "up"]
    assert metadata["camera_frame"]["axes"] == ["right", "down", "forward"]
    for key in (
        "quaternion_order",
        "linear_velocity_frame",
        "angular_velocity_frame",
        "action_frame",
    ):
        assert metadata[key]


def test_backend_policy_positive_left_moves_geographic_left() -> None:
    simulator = Simulator.__new__(Simulator)
    simulator.drones = [
        SimpleNamespace(id="uav", yaw=0.0, safety_radius=1.0)
    ]
    simulator._external_policy_commands = {}
    simulator.time = 12.0
    accepted = simulator.set_external_policy_action(
        "uav",
        [0.0, 0.5, 0.0, 0.0],
        step_id=0,
        policy_family="coordinate_test",
    )
    # Facing east at yaw zero: left is north, which is backend negative Z.
    assert np.allclose(accepted["command_world_mps"], [0.0, 0.0, -3.0])
