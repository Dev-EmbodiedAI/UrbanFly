from __future__ import annotations

import argparse
import json
import socket
import sys
import traceback
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.simulators import MockSimulator, UrbanFlyWebSocketAdapter
from uav_wm_navigation.utils.config import load_yaml
from uav_wm_navigation.utils.runlog import create_run_dir, write_manifest


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a safe simulator lifecycle smoke test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=_bootstrap.PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_dir = create_run_dir(args.output_root, f"smoke_{config.get('backend', 'unknown')}")
    write_manifest(run_dir, config)
    backend = config.get("backend")
    if backend == "mock":
        simulator = MockSimulator(
            seed=int(config.get("seed", 0)), depth_shape=tuple(config.get("depth_shape", [96, 160])),
            depth_max_m=float(config.get("depth_max_m", 20.0)), scenario=str(config.get("scenario", "StaticObstacle")),
        )
    elif backend == "urbanfly_websocket":
        simulator = UrbanFlyWebSocketAdapter(config)
    else:
        raise ValueError(f"unsupported backend: {backend}")
    result = {"backend": backend, "success": False, "ports": {}}
    if backend == "urbanfly_websocket":
        result["ports"][str(config.get("port", 8765))] = port_open(
            str(config.get("host", "127.0.0.1")), int(config.get("port", 8765))
        )
    try:
        simulator.connect()
        simulator.reset()
        if "flight_start_nwu" in config:
            simulator.set_initial_pose(np.asarray(config["flight_start_nwu"], dtype=np.float64))
        simulator.takeoff()
        if "flight_start_nwu" in config:
            simulator.set_initial_pose(np.asarray(config["flight_start_nwu"], dtype=np.float64))
        if hasattr(simulator, "stabilize_at_altitude") and "target_altitude_nwu" in config:
            simulator.stabilize_at_altitude(float(config["target_altitude_nwu"]), 2.0)
        state0 = simulator.get_kinematics()
        sensor = simulator.get_depth()
        velocity = np.asarray(config.get("smoke_velocity_nwu", [0.5, 0.0, 0.0]), dtype=np.float64)
        simulator.execute_velocity_command(velocity, 0.0, float(config.get("smoke_duration_s", 0.5)))
        state1 = simulator.get_kinematics()
        collision = simulator.get_collision_info()
        simulator.land()
        result.update({
            "success": True, "initial_position_nwu": state0.position.tolist(), "final_position_nwu": state1.position.tolist(),
            "depth_shape": list(sensor.depth_m.shape), "valid_depth_fraction": float(sensor.valid_mask.mean()),
            "collision": collision,
        })
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            simulator.land()
        except Exception:
            pass
        simulator.close()
        (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), **result}, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
