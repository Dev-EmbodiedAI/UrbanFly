from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np


METHODS = {
    "yopo_direct": [None],
    "yopo_tdmpc2_visual": [101, 202, 303],
    "yopo_dreamer_rssm": [101, 202, 303],
    "yopo_vjepa2_1": [101, 202, 303],
    "arr_fly": [101, 202, 303],
    "geometric_mpc_teacher": [None],
}


def canonical_hash(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def perturbation(group: str, rng: np.random.Generator) -> dict:
    base = {
        "appearance": {"exposure_ev": 0.0, "fog_density": 0.0, "color_temperature_k": 6500.0, "camera_noise_std": 0.0, "frame_drop_probability": 0.0},
        "dynamics": {"wind_nwu_mps": [0.0, 0.0, 0.0], "mass_scale": 1.0, "drag_scale": 1.0, "motor_delay_ms": 0.0, "control_jitter_ms": 0.0},
        "dynamic_actor_density": 1.0,
    }
    if group == "appearance_sensor":
        base["appearance"] = {
            "exposure_ev": float(rng.uniform(-1.5, 1.5)),
            "fog_density": float(rng.uniform(0.01, 0.12)),
            "color_temperature_k": float(rng.uniform(3500.0, 9000.0)),
            "camera_noise_std": float(rng.uniform(0.005, 0.035)),
            "frame_drop_probability": float(rng.uniform(0.0, 0.12)),
        }
    elif group == "dynamics":
        direction = float(rng.uniform(-np.pi, np.pi)); speed = float(rng.uniform(1.0, 7.0))
        base["dynamics"] = {
            "wind_nwu_mps": [float(speed * np.cos(direction)), float(speed * np.sin(direction)), float(rng.uniform(-0.5, 0.5))],
            "mass_scale": float(rng.uniform(0.75, 1.35)),
            "drag_scale": float(rng.uniform(0.7, 1.4)),
            "motor_delay_ms": float(rng.uniform(20.0, 120.0)),
            "control_jitter_ms": float(rng.uniform(0.0, 35.0)),
        }
    elif group == "canyon_dynamic_stress":
        base["appearance"]["fog_density"] = float(rng.uniform(0.0, 0.04))
        base["dynamics"]["wind_nwu_mps"] = [float(rng.uniform(-4.0, 4.0)), float(rng.uniform(-4.0, 4.0)), 0.0]
        base["dynamic_actor_density"] = 2.0
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the five-by-24 paired UrbanFly v3 evaluation routes")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = json.loads(args.train.read_text(encoding="utf-8"))
    test = json.loads(args.test.read_text(encoding="utf-8"))
    if len(train["routes"]) != 24 or len(test["routes"]) != 24:
        raise ValueError("base manifests must each contain exactly 24 routes")
    rng = np.random.default_rng(20260831)
    groups = [
        ("id_train_tiles", train["routes"]),
        ("unseen_tiles", test["routes"]),
        ("appearance_sensor", train["routes"]),
        ("dynamics", train["routes"]),
        ("canyon_dynamic_stress", test["routes"]),
    ]
    routes = []
    for group_index, (group, sources) in enumerate(groups):
        for index, source in enumerate(sources):
            route = deepcopy(source)
            route["base_route_id"] = source["route_id"]
            route["route_id"] = f"eval-{group_index + 1}-{group}-{index:02d}"
            route["group"] = group
            route["seed"] = 2026083100 + group_index * 100 + index
            route["actor_script_id"] = f"actors-{route['seed']}"
            route["perturbation"] = perturbation(group, rng)
            routes.append(route)
    payload = {
        "schema": "urbanfly-paired-evaluation-v3",
        "created_for": "2026-08 month-end quick comparison",
        "route_count": len(routes),
        "routes_per_group": 24,
        "groups": [item[0] for item in groups],
        "methods": METHODS,
        "shield_modes": [False, True],
        "sensor_hz": 10,
        "policy_hz": 5,
        "physics_hz": 50,
        "candidate_count": 15,
        "success": {"radius_m": 3.0, "dwell_s": 2.0, "collision_free": True},
        "source_manifests": {
            "train": {"path": str(args.train), "sha256": train["manifest_sha256"]},
            "test": {"path": str(args.test), "sha256": test["manifest_sha256"]},
        },
        "routes": routes,
    }
    payload["manifest_sha256"] = canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = sum(len(seeds) for seeds in METHODS.values()) * 2 * len(routes)
    print(json.dumps({"output": str(args.output.resolve()), "routes": len(routes), "jobs": jobs, "sha256": payload["manifest_sha256"]}))


if __name__ == "__main__":
    main()
