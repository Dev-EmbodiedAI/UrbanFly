#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.direct_visual_world_model import DirectVisualWorldModel, DirectWorldModelConfig  # noqa: E402
from urbanfly_vln.visual_world_model_data import (  # noqa: E402
    DirectTransitionDataset,
    discover_visual_runs,
    load_visual_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the direct visual flight world model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--data-root", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    config = DirectWorldModelConfig(**payload["config"])
    model = DirectVisualWorldModel(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    runs = [path.resolve() for path in args.run_dir] + discover_visual_runs(args.data_root)
    if not runs:
        runs = [Path(path) for path in payload["metadata"]["validation"]["runs"]]
    episodes = [load_visual_episode(path) for path in sorted(set(runs))]
    dataset = DirectTransitionDataset(
        episodes,
        image_size=config.image_size,
        bottom_crop_fraction=config.bottom_crop_fraction,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    state_errors, reward_errors = [], []
    with torch.inference_mode():
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items()}
            output = model(batch["observations"], batch["states"], batch["actions"])
            state_errors.append((output["next_state"] - batch["next_states"]).cpu().numpy())
            reward_errors.append((output["reward"] - batch["rewards"]).abs().cpu().numpy())
    error = np.concatenate(state_errors)
    metrics = {
        "format": "urbanfly-direct-visual-world-model-eval-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "runs": len(episodes),
        "transitions": len(dataset),
        "state_normalized_rmse": float(np.sqrt(np.mean(error**2))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(error[:, :3] ** 2)) * 10.0),
        "p05_depth_rmse_m": float(np.sqrt(np.mean(error[:, 4] ** 2)) * 20.0),
        "final_distance_rmse_m": float(np.sqrt(np.mean(error[:, 7] ** 2)) * 200.0),
        "reward_mae": float(np.mean(np.concatenate(reward_errors))),
    }
    output_path = args.output or args.checkpoint.with_suffix(".eval.json")
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
