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

from urbanfly_vln.visual_world_model import RSSMState, VisualRSSM, VisualWorldModelConfig  # noqa: E402
from urbanfly_vln.visual_world_model_data import (  # noqa: E402
    VisualSequenceDataset,
    discover_visual_runs,
    load_visual_episode,
)
from urbanfly_vln.world_model_metrics import json_ready, risk_report  # noqa: E402


def threshold_sweep(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    reports = [risk_report(labels, probabilities, float(threshold)) for threshold in np.linspace(0.0, 1.0, 101)]
    best_f1 = max(reports, key=lambda report: (report["f1"], report["recall"]))
    safety_candidates = [report for report in reports if report["recall"] >= 0.8]
    safety = max(safety_candidates, key=lambda report: report["threshold"]) if safety_candidates else reports[0]
    return {"best_f1": best_f1, "recall_at_least_0_8": safety}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out posterior and open-loop visual RSSM predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, action="append", default=[])
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--depth-max-m", type=float, default=20.0)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    model = VisualRSSM(VisualWorldModelConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    runs = [path.resolve() for path in args.run_dir]
    runs.extend(discover_visual_runs(args.data_root))
    if not runs:
        runs = [Path(path) for path in payload.get("metadata", {}).get("validation", {}).get("runs", [])]
    episodes = [load_visual_episode(path, depth_max_m=args.depth_max_m) for path in sorted(set(runs))]
    dataset = VisualSequenceDataset(
        episodes,
        sequence_length=args.sequence_length,
        stride=args.sequence_length,
        image_size=model.config.image_size,
        bottom_crop_fraction=model.config.bottom_crop_fraction,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    posterior_state_errors, posterior_depth_errors, posterior_rgb_errors = [], [], []
    imagined_state_errors, imagined_reward_errors = [], []
    imagined_risks, imagined_risk_labels = [], []
    with torch.inference_mode():
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                posterior = model.observe(batch["observations"], batch["actions"], sample=False)
                reconstruction = model.decoder(posterior["feature"])
                initial = RSSMState(posterior["deter"][:, 0], posterior["stoch"][:, 0])
                imagined = model.imagine(initial, batch["actions"][:, :-1], sample=False)
            posterior_state_errors.append((posterior["state"] - batch["states"]).float().cpu().numpy())
            posterior_rgb_errors.append(
                (reconstruction[:, :, :3] - batch["observations"][:, :, :3]).abs().float().cpu().numpy()
            )
            posterior_depth_errors.append(
                (reconstruction[:, :, 3:] - batch["observations"][:, :, 3:]).abs().float().cpu().numpy()
            )
            imagined_state_errors.append(
                (imagined["state"] - batch["states"][:, 1:]).float().cpu().numpy()
            )
            imagined_reward_errors.append(
                (imagined["reward"] - batch["rewards"][:, :-1]).abs().float().cpu().numpy()
            )
            imagined_risks.append(torch.sigmoid(imagined["risk_logit"]).float().cpu().numpy().reshape(-1))
            imagined_risk_labels.append(batch["risks"][:, :-1].float().cpu().numpy().reshape(-1))

    posterior_state = np.concatenate(posterior_state_errors).reshape(-1, model.config.state_dim)
    imagined_state = np.concatenate(imagined_state_errors).reshape(-1, model.config.state_dim)
    probabilities = np.concatenate(imagined_risks)
    labels = np.concatenate(imagined_risk_labels)
    metrics = {
        "format": "urbanfly-visual-rssm-evaluation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "runs": len(episodes),
        "sequences": len(dataset),
        "frames": len(dataset) * args.sequence_length,
        "posterior": {
            "rgb_mae_0_1": float(np.mean(np.concatenate(posterior_rgb_errors))),
            "depth_mae_m": float(np.mean(np.concatenate(posterior_depth_errors)) * args.depth_max_m),
            "state_normalized_rmse": float(np.sqrt(np.mean(posterior_state**2))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(posterior_state[:, :3] ** 2)) * 10.0),
            "p05_depth_rmse_m": float(np.sqrt(np.mean(posterior_state[:, 4] ** 2)) * args.depth_max_m),
        },
        "open_loop_imagination": {
            "horizon": args.sequence_length - 1,
            "state_normalized_rmse": float(np.sqrt(np.mean(imagined_state**2))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(imagined_state[:, :3] ** 2)) * 10.0),
            "p05_depth_rmse_m": float(np.sqrt(np.mean(imagined_state[:, 4] ** 2)) * args.depth_max_m),
            "reward_mae": float(np.mean(np.concatenate(imagined_reward_errors))),
            "risk": risk_report(labels, probabilities, args.risk_threshold),
            "risk_threshold_sweep": threshold_sweep(labels, probabilities),
        },
    }
    metrics = json_ready(metrics)
    output = args.output or args.checkpoint.with_suffix(".eval.json")
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
