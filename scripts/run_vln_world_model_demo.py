#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.risk_world_model import LinearRiskWorldModel  # noqa: E402
from urbanfly_vln.schema import Episode, Step  # noqa: E402


def synthetic_episode(index: int, steps: int = 60) -> Episode:
    """Generate a deterministic smoke-test episode, not a claimed flight result."""
    rng = np.random.default_rng(100 + index)
    direction = 1.0 if index % 2 == 0 else -1.0
    route = [[0.0, 0.0, -15.0], [35.0, 8.0 * direction, -15.0], [75.0, 30.0 * direction, -15.0]]
    instruction = (
        f"Fly east, turn {'left' if direction > 0 else 'right'} after the first block, "
        "keep clear of moving traffic, and stop at the endpoint."
    )
    output: list[Step] = []
    position = np.array(route[0], dtype=np.float64)
    previous_velocity = np.zeros(3, dtype=np.float64)
    for tick in range(steps):
        progress = tick / max(steps - 1, 1)
        target = np.array(route[1] if progress < 0.5 else route[2], dtype=np.float64)
        delta = target - position
        action = delta / max(np.linalg.norm(delta), 1e-6) * (1.1 + 0.08 * rng.normal())
        action[2] = 0.0
        velocity = 0.75 * previous_velocity + 0.25 * action / 0.2
        obstacle_center = 0.42 + 0.03 * (index % 3)
        p05_depth = max(0.45, 8.0 - 8.0 * math.exp(-((progress - obstacle_center) / 0.09) ** 2))
        p05_depth += float(rng.normal(0.0, 0.12))
        collision = bool(p05_depth < 0.72 and index % 4 == 0)
        goal_distance = float(np.linalg.norm(np.array(route[-1]) - position))
        output.append(
            Step(
                time_s=tick * 0.2,
                position=position.tolist(),
                velocity=velocity.tolist(),
                action_delta=action.tolist(),
                goal_distance_m=goal_distance,
                min_depth_m=max(0.2, p05_depth * 0.55),
                p05_depth_m=max(0.2, p05_depth),
                collision=collision,
                replan_step=tick,
                target_waypoint_idx=1 if progress < 0.5 else 2,
            )
        )
        position = position + action
        previous_velocity = velocity
    return Episode(
        episode_id=f"synthetic-{index:03d}",
        scene_id="synthetic-smoke-test",
        instruction=instruction,
        route=route,
        steps=output,
        split="train" if index < 8 else "val_unseen",
        success=not any(step.collision for step in output),
        metadata={"synthetic": True, "purpose": "pipeline smoke test only"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline smoke test of the UrbanFly risk world-model contract.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "vln_world_model_demo")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = [synthetic_episode(index) for index in range(10)]
    model = LinearRiskWorldModel()
    train_metrics = model.fit(episodes[:8], risk_horizon=5, risk_depth_m=2.0)
    validation_metrics = model.evaluate(episodes[8:], risk_horizon=5, risk_depth_m=2.0)
    episodes[8].write_json(output_dir / "sample_episode.json")
    report = {
        "status": "PASS",
        "scope": "synthetic pipeline smoke test; not a scientific flight result",
        "contract": "instruction + state + candidate action -> next state + near-future risk",
        "train": train_metrics.__dict__,
        "val_unseen": validation_metrics.__dict__,
        "next_real_run": (
            "Record an UrbanFly flight, then convert the run with "
            "python -m urbanfly_vln.episode_builder."
        ),
    }
    report_path = output_dir / "demo_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
