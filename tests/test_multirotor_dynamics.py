import numpy as np

from backend.config import DRONE_TYPES, MULTIROTOR_DYNAMICS
from backend.engine.models import DroneState, DroneStateData, Waypoint
from backend.engine.collision import HeightmapStaticCollisionMap
from backend.engine.multirotor_dynamics import (
    MultirotorDynamics,
    MultirotorParameters,
)
from backend.engine.scenario import ScenarioEngine
from backend.engine.simulator import Simulator


def _standard_model():
    return MultirotorDynamics(
        MultirotorParameters.from_dict(MULTIROTOR_DYNAMICS["standard"])
    )


def test_standard_multirotor_holds_hover():
    model = _standard_model()
    position = np.array([0.0, 20.0, 0.0])
    velocity = np.zeros(3)

    for _ in range(500):
        frame = model.step(
            position=position,
            velocity=velocity,
            target_position=np.array([0.0, 20.0, 0.0]),
            target_velocity=np.zeros(3),
            desired_yaw_degrees=0.0,
            wind_velocity=np.zeros(3),
            payload_mass=0.0,
            max_acceleration=3.0,
            dt=0.01,
        )
        position = frame["position"]
        velocity = frame["velocity"]

    assert np.linalg.norm(position - np.array([0.0, 20.0, 0.0])) < 0.05
    assert np.linalg.norm(velocity) < 0.05
    assert abs(np.linalg.norm(frame["orientation"]) - 1.0) < 1e-6
    assert np.all(frame["motor_thrusts"] > 0)


def test_standard_multirotor_generates_tilt_and_forward_motion():
    model = _standard_model()
    position = np.array([0.0, 20.0, 0.0])
    velocity = np.zeros(3)

    for _ in range(300):
        frame = model.step(
            position=position,
            velocity=velocity,
            target_position=np.array([100.0, 20.0, 0.0]),
            target_velocity=np.array([12.0, 0.0, 0.0]),
            desired_yaw_degrees=0.0,
            wind_velocity=np.zeros(3),
            payload_mass=0.0,
            max_acceleration=3.0,
            dt=0.01,
        )
        position = frame["position"]
        velocity = frame["velocity"]

    assert position[0] > 4.0
    assert velocity[0] > 2.0
    assert abs(frame["pitch"]) > 3.0
    assert abs(position[1] - 20.0) < 1.0


def test_simulator_snapshot_exposes_actuator_and_attitude_state():
    type_config = DRONE_TYPES["standard"]
    drone = DroneStateData(
        id="UAV-TEST",
        drone_type="standard",
        position=np.array([0.0, 20.0, 0.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        yaw=0.0,
        battery_remaining=type_config["battery_capacity"],
        payload_current=0.0,
        state=DroneState.EN_ROUTE,
        **{
            key: value
            for key, value in type_config.items()
            if key not in {"label", "color"}
        },
    )
    drone.path = [Waypoint(position=np.array([60.0, 20.0, 0.0]))]
    simulator = Simulator()
    simulator.drones = [drone]

    for _ in range(20):
        simulator._update_drone_dynamics(drone, 0.05)

    payload = drone.to_dict()
    assert payload["dynamics_model"] == "urbanfly_fast_multirotor_6dof"
    assert len(payload["orientation"]) == 4
    assert len(payload["motor_omega"]) == 4
    assert payload["total_thrust"] > 0
    assert payload["power_w"] > 0
    assert payload["pos"][0] > 0


def test_heightmap_collision_sweeps_against_building_surface():
    height = np.zeros((21, 21), dtype=np.float32)
    height[8:13, 8:13] = 18.0
    collision = HeightmapStaticCollisionMap(
        height=height,
        origin_x=-10.0,
        origin_z=-10.0,
        resolution=1.0,
    )

    assert collision.collides(np.array([0.0, 10.0, 0.0]), 1.0)[0]
    assert not collision.collides(np.array([0.0, 24.0, 0.0]), 1.0)[0]
    hit, clearance, point = collision.sweep_collides(
        np.array([-8.0, 10.0, 0.0]),
        np.array([8.0, 10.0, 0.0]),
        safety_radius=1.0,
    )
    assert hit
    assert clearance < 0.0
    assert point is not None


def test_single_uav_acceptance_scenario_flies_scripted_6dof_route():
    scenario_engine = ScenarioEngine.create_default()
    scenario = scenario_engine.get_scenario("single_uav_dynamics")
    assert scenario is not None
    assert len(scenario.drones) == 1

    collision = HeightmapStaticCollisionMap(
        height=np.zeros((401, 401), dtype=np.float32),
        origin_x=-100.0,
        origin_z=-100.0,
        resolution=1.0,
    )
    simulator = Simulator(static_collision_map=collision)
    simulator.initialize_scenario(scenario)

    assert len(simulator.drones) == 1
    drone = simulator.drones[0]
    assert len(drone.path) >= 7
    assert all(wp.metadata.get("static_clearance_validated") for wp in drone.path)

    start = drone.position.copy()
    for _ in range(250):
        simulator.step()

    assert np.linalg.norm(drone.position - start) > 5.0
    assert drone.dynamics_model == "urbanfly_fast_multirotor_6dof"
    assert np.all(np.isfinite(drone.orientation_quaternion))
    assert np.all(drone.motor_omega > 0)
    assert drone.power_w > 0
