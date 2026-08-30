#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.latent_world_model import LatentWorldModelEnsemble  # noqa: E402
from urbanfly_vln.world_model_data import samples_from_run, stack_samples  # noqa: E402
from urbanfly_vln.world_model_metrics import json_ready, risk_report  # noqa: E402


def continuous_report(prediction: np.ndarray, target: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    return {
        "delta_position_rmse_m": float(np.sqrt(np.mean((prediction[:, :3] - target[:, :3]) ** 2))),
        "speed_rmse_mps": float(np.sqrt(np.mean((prediction[:, 3] - target[:, 3]) ** 2))),
        "p05_depth_rmse_m": float(np.sqrt(np.mean((prediction[:, 4] - target[:, 4]) ** 2))),
        "progress_rmse_m": float(np.sqrt(np.mean((prediction[:, 5] - target[:, 5]) ** 2))),
        "mean_epistemic_uncertainty": float(np.mean(uncertainty)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an UrbanFly world-model checkpoint on complete held-out runs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--risk-horizon", type=int, default=None, help="Override the checkpoint label horizon.")
    parser.add_argument(
        "--near-miss-depth-m",
        type=float,
        default=None,
        help="Override the checkpoint near-miss threshold for apples-to-apples comparisons.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = None if args.device == "auto" else torch.device(args.device)
    model = LatentWorldModelEnsemble.load(args.checkpoint.resolve(), device=device)

    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    label_config = payload.get("label_config", {})
    risk_horizon = (
        args.risk_horizon if args.risk_horizon is not None else int(label_config.get("risk_horizon", 3))
    )
    near_miss_depth_m = (
        args.near_miss_depth_m
        if args.near_miss_depth_m is not None
        else float(label_config.get("near_miss_depth_m", 5.0))
    )
    samples = []
    for run_dir in args.run_dir:
        samples.extend(
            samples_from_run(
                run_dir,
                language_dimensions=model.language_dimensions,
                risk_horizon=risk_horizon,
                near_miss_depth_m=near_miss_depth_m,
            )
        )
    x, y, risk = stack_samples(samples)
    started = time.perf_counter()
    prediction, probability, uncertainty = model.predict_feature_matrix(x)
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "format": "urbanfly-world-model-evaluation-0.2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_format": model.checkpoint_format,
        "examples": len(samples),
        "sources": sorted({sample.source for sample in samples}),
        "scenes": sorted({sample.scene_id for sample in samples}),
        "risk_horizon": risk_horizon,
        "near_miss_depth_m": near_miss_depth_m,
        "inference_ms_total": elapsed * 1000.0,
        "inference_ms_per_example": elapsed * 1000.0 / len(samples),
        **continuous_report(prediction, y, uncertainty),
        "risk": risk_report(risk, probability, args.risk_threshold),
    }
    per_source = {}
    for source in sorted({sample.source for sample in samples}):
        mask = np.asarray([sample.source == source for sample in samples])
        per_source[source] = {
            "examples": int(np.sum(mask)),
            **continuous_report(prediction[mask], y[mask], uncertainty[mask]),
            "risk": risk_report(risk[mask], probability[mask], args.risk_threshold),
        }
    report["per_source"] = per_source
    text = json.dumps(json_ready(report), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
