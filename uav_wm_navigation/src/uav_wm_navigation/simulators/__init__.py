from .base import SimulatorAdapter
from .mock_simulator import MockSimulator
from .urbanfly_websocket_adapter import UrbanFlyWebSocketAdapter
from .helsinki_websocket_adapter import HelsinkiWebSocketAdapter


def build_simulator(config: dict):
    backend = str(config.get("backend", "mock"))
    if backend == "mock":
        return MockSimulator(
            seed=int(config.get("seed", 0)), scenario=str(config.get("scenario", "StreetCanyon")),
            depth_shape=tuple(config.get("depth_shape", [96, 160])),
            depth_max_m=float(config.get("depth_max_m", 20.0)),
            control_dt=float(config.get("control_dt", 0.1)),
            vehicle_name=str(config.get("vehicle_name", "SimpleFlight")),
        )
    if backend == "urbanfly_websocket": return UrbanFlyWebSocketAdapter(config)
    if backend == "helsinki_websocket": return HelsinkiWebSocketAdapter(config)
    raise ValueError(f"unsupported simulator backend {backend!r}")

__all__ = [
    "MockSimulator",
    "SimulatorAdapter",
    "UrbanFlyWebSocketAdapter",
    "HelsinkiWebSocketAdapter",
    "build_simulator",
]
