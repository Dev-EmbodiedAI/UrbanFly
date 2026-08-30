from __future__ import annotations

import numpy as np

from uav_wm_navigation.simulators.urbanfly_websocket_adapter import (
    UrbanFlyWebSocketAdapter,
    nwu_to_urbanfly_world,
    urbanfly_world_to_nwu,
)


def test_urbanfly_world_nwu_conversion_is_explicit_and_invertible() -> None:
    world = np.array([12.0, 7.5, -4.0])
    nwu = urbanfly_world_to_nwu(world)
    assert np.allclose(nwu, [12.0, -4.0, 7.5])
    assert np.allclose(nwu_to_urbanfly_world(nwu), world)


def test_vertical_velocity_maps_to_urbanfly_up_axis() -> None:
    assert np.allclose(
        nwu_to_urbanfly_world(np.array([2.0, -3.0, 1.25])),
        [2.0, 1.25, -3.0],
    )


def test_reset_clears_cross_episode_action_and_kinematics_state(monkeypatch) -> None:
    adapter = UrbanFlyWebSocketAdapter({})
    monkeypatch.setattr(adapter, "_select_scenario", lambda: None)
    adapter._action_ack_step = 87
    adapter._action_ack = {"step_id": 87, "accepted_sim_time": 41.2}
    adapter._last_velocity_time = 41.2
    adapter._last_velocity_nwu[:] = [1.0, 2.0, 3.0]
    adapter._last_acceleration_nwu[:] = [4.0, 5.0, 6.0]
    adapter._last_canonical_state = object()

    adapter.reset()

    assert adapter._action_ack_step == -2
    assert adapter._action_ack is None
    assert adapter._last_velocity_time is None
    assert np.array_equal(adapter._last_velocity_nwu, np.zeros(3))
    assert np.array_equal(adapter._last_acceleration_nwu, np.zeros(3))
    assert not hasattr(adapter, "_last_canonical_state")
