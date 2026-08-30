from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from train_world_model import make_dataset, model_forward
from uav_wm_navigation.evaluation import (
    binary_auprc, binary_auroc, brier_score, expected_calibration_error, pairwise_ranking_accuracy,
)
from uav_wm_navigation.world_models import build_world_model, validate_world_model_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate candidate risk, calibration, ranking and model dynamics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); config = checkpoint["config"]
    paths = json.loads(args.splits.read_text(encoding="utf-8"))[args.split]
    future = int(config.get("future_target_steps", 0)) if config["model"] in {"dreamerv3", "jepa"} else 0
    dataset = make_dataset(paths, config, future); loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 8)))
    model = build_world_model(config); model.load_state_dict(checkpoint["model"]); model.eval()
    calibration = checkpoint.get("calibration", {})
    collision_temperature = float(calibration.get("collision_temperature", 1.0))
    failure_temperature = float(calibration.get("failure_temperature", 1.0))
    labels, scores, clearance_error, progress_error = [], [], [], []
    target_scores, predicted_scores, valid_masks, unsafe_top1 = [], [], [], []
    dynamics_metrics: dict[str, list[float]] = {"dreamer_depth_mae": [], "jepa_latent_loss": [], "occupancy_iou": [], "flow_epe": []}
    with torch.inference_mode():
        for batch in loader:
            output = model_forward(model, config, batch, torch.device("cpu"))
            validate_world_model_output(output, batch["depth"].shape[0], batch["trajectories"].shape[1])
            valid = batch["label_valid_mask"].bool().numpy(); label = batch["collision"].numpy()
            probability = torch.sigmoid(output["collision_logits"] / collision_temperature).numpy()
            labels.extend(label[valid]); scores.extend(probability[valid])
            clearance_error.extend(np.abs(output["minimum_clearance"].numpy() - batch["minimum_clearance"].numpy())[valid])
            progress_error.extend(np.abs(output["goal_progress"].numpy() - batch["goal_progress"].numpy())[valid])
            target_score = 4*label + 2*batch["failure"].numpy() - batch["minimum_clearance"].numpy() - batch["goal_progress"].numpy()
            predicted_score = 4*probability + 2*torch.sigmoid(
                output["failure_logits"] / failure_temperature
            ).numpy() - output["minimum_clearance"].numpy() - output["goal_progress"].numpy()
            target_scores.append(target_score); predicted_scores.append(predicted_score); valid_masks.append(valid)
            choices = np.argmin(predicted_score, axis=1)
            unsafe_top1.extend([float(label[row, choice]) for row, choice in enumerate(choices)])
            if "predicted_future_depth" in output:
                target = torch.nn.functional.interpolate(
                    batch["future_depth"][:, :output["predicted_future_depth"].shape[1]].flatten(0, 1),
                    size=(48, 80), mode="nearest",
                ).reshape_as(output["predicted_future_depth"])
                dynamics_metrics["dreamer_depth_mae"].append(float((output["predicted_future_depth"] - target).abs().mean()))
            if "jepa_loss" in output: dynamics_metrics["jepa_latent_loss"].append(float(output["jepa_loss"]))
            if "occupancy_logits" in output and "occupancy" in batch:
                prediction = output["occupancy_logits"].sigmoid() >= 0.5; target = batch["occupancy"].bool()
                intersection = (prediction & target).sum().item(); union = (prediction | target).sum().item()
                dynamics_metrics["occupancy_iou"].append(intersection / max(union, 1))
                occupied = target[:, :, :2].amax(dim=2, keepdim=True)
                epe = torch.linalg.vector_norm(output["flow"] - batch["flow"], dim=2, keepdim=True)
                dynamics_metrics["flow_epe"].append(float((epe * occupied).sum() / occupied.sum().clamp_min(1)))
    label_array, score_array = np.asarray(labels), np.asarray(scores)
    metrics = {
        "model": config["model"], "split": args.split, "examples": int(label_array.size),
        "collision_auroc": binary_auroc(label_array, score_array),
        "collision_auprc": binary_auprc(label_array, score_array),
        "brier": brier_score(label_array, score_array), "ece": expected_calibration_error(label_array, score_array),
        "clearance_mae": float(np.mean(clearance_error)), "progress_mae": float(np.mean(progress_error)),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(
            np.concatenate(target_scores), np.concatenate(predicted_scores), np.concatenate(valid_masks)
        ),
        "unsafe_top1_rate": float(np.mean(unsafe_top1)),
        **{name: float(np.mean(values)) for name, values in dynamics_metrics.items() if values},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
