from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from train_world_model import make_dataset
from uav_wm_navigation.world_models import build_world_model


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / log_temperature.exp(), labels)
        loss.backward(); return loss
    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit validation-only temperature scaling and write a calibrated checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    future = int(config.get("future_target_steps", 0)) if config["model"] in {"dreamerv3", "jepa"} else 0
    dataset = make_dataset(splits["validation"], config, future)
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 8)))
    model = build_world_model(config); model.load_state_dict(checkpoint["model"]); model.eval()
    collision_logits, collision_labels, failure_logits, failure_labels = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            output = model(*(batch[key] for key in ("depth", "state", "goal", "trajectories")))
            valid = batch["label_valid_mask"].bool()
            collision_logits.append(output["collision_logits"][valid]); collision_labels.append(batch["collision"][valid])
            failure_logits.append(output["failure_logits"][valid]); failure_labels.append(batch["failure"][valid])
    checkpoint["calibration"] = {
        "collision_temperature": fit_temperature(torch.cat(collision_logits), torch.cat(collision_labels).float()),
        "failure_temperature": fit_temperature(torch.cat(failure_logits), torch.cat(failure_labels).float()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, args.output)
    print(json.dumps(checkpoint["calibration"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
