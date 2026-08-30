from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from uav_wm_navigation.data import DreamerSequenceDataset, WorldModelDataset
from uav_wm_navigation.evaluation import binary_auprc, expected_calibration_error
from uav_wm_navigation.utils.config import load_yaml
from uav_wm_navigation.world_models import WorldModelLoss, build_world_model, validate_world_model_output


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(paths: list[str], config: dict, future: int) -> WorldModelDataset:
    cache = config.get("occupancy_cache_dir") if config["model"] == "occflow" else None
    return WorldModelDataset(
        paths, history=int(config["history"]), depth_max_m=float(config.get("depth_max_m", 20.0)),
        future_observations=future, trajectory_steps=int(config.get("trajectory_steps", 16)),
        occupancy_cache_dir=cache,
    )


def model_forward(model, config: dict, batch: dict[str, torch.Tensor], device: torch.device):
    inputs = [batch[key].to(device) for key in ("depth", "state", "goal", "trajectories")]
    kwargs = {}
    if config["model"] in {"dreamerv3", "jepa"}:
        kwargs = {
            "future_depth": batch["future_depth"].to(device),
            "selected_index": batch["selected_index"].to(device),
            "future_valid_mask": batch["future_valid_mask"].to(device),
        }
    return model(*inputs, **kwargs)


def targets(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "collision", "failure", "minimum_clearance", "goal_progress", "label_valid_mask", "label_confidence",
        "occupancy", "flow",
    )
    return {name: batch[name].to(device) for name in names if name in batch}


@torch.no_grad()
def validate(model, loader, criterion, config, device) -> dict[str, float]:
    model.eval(); losses, labels, probabilities, clearance_errors, unsafe = [], [], [], [], []
    for batch in loader:
        output = model_forward(model, config, batch, device)
        validate_world_model_output(output, batch["depth"].shape[0], batch["trajectories"].shape[1])
        loss, _ = criterion(output, targets(batch, device)); losses.append(float(loss))
        valid = batch["label_valid_mask"].bool().numpy()
        label = batch["collision"].numpy(); probability = torch.sigmoid(output["collision_logits"]).cpu().numpy()
        labels.extend(label[valid].tolist()); probabilities.extend(probability[valid].tolist())
        clearance_error = np.abs(output["minimum_clearance"].cpu().numpy() - batch["minimum_clearance"].numpy())
        clearance_errors.extend(clearance_error[valid].tolist())
        selected = output["collision_logits"].argmin(dim=1).cpu().numpy()
        unsafe.extend([float(label[row, choice]) for row, choice in enumerate(selected)])
    label_array, probability_array = np.asarray(labels), np.asarray(probabilities)
    auprc = binary_auprc(label_array, probability_array) if len(np.unique(label_array)) > 1 else 0.0
    ece = expected_calibration_error(label_array, probability_array) if len(label_array) else 1.0
    clearance_mae = float(np.mean(clearance_errors)) if clearance_errors else 20.0
    unsafe_top1 = float(np.mean(unsafe)) if unsafe else 1.0
    composite = 0.45 * (1.0 - auprc) + 0.20 * ece + 0.20 * min(clearance_mae / 20.0, 1.0) + 0.15 * unsafe_top1
    return {
        "loss": float(np.mean(losses)) if losses else math.inf, "collision_auprc": auprc,
        "ece": ece, "clearance_mae": clearance_mae, "unsafe_top1": unsafe_top1,
        "composite": composite,
    }


def normalization_stats(dataset: WorldModelDataset) -> dict[str, list[float]]:
    values = {"yopo_cost": [], "minimum_clearance": [], "goal_progress": []}
    for index in range(len(dataset)):
        sample = dataset[index]; valid = sample["label_valid_mask"].bool()
        values["yopo_cost"].extend(sample["yopo_cost"].numpy().tolist())
        values["minimum_clearance"].extend(sample["minimum_clearance"][valid].numpy().tolist())
        values["goal_progress"].extend(sample["goal_progress"][valid].numpy().tolist())
    return {
        name: [float(np.percentile(items, 5)), float(np.percentile(items, 95))]
        for name, items in values.items() if items
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and validate an action-conditioned YOPO world model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--overfit-episodes", type=int, default=0)
    args = parser.parse_args()
    config = load_yaml(args.config); set_seed(args.seed)
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    future = int(config.get("future_target_steps", config.get("jepa_target_steps", 0))) if config["model"] in {"dreamerv3", "jepa"} else 0
    train_paths = split_payload["train"][: args.overfit_episodes or None]
    validation_paths = train_paths if args.overfit_episodes else split_payload["validation"]
    train_dataset = make_dataset(train_paths, config, future)
    validation_dataset = make_dataset(validation_paths, config, future)
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sequence_loader = None
    if config["model"] == "dreamerv3":
        sequence_dataset = DreamerSequenceDataset(
            train_paths, sequence_length=int(config.get("sequence_length", 16)),
            depth_max_m=float(config.get("depth_max_m", 20.0)),
            stride=int(config.get("sequence_stride", 4)),
        )
        if not len(sequence_dataset):
            raise ValueError("Dreamer training requires episodes at least sequence_length steps long")
        sequence_loader = DataLoader(
            sequence_dataset, batch_size=int(config.get("sequence_batch_size", batch_size)),
            shuffle=True, num_workers=0,
        )
    model = build_world_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    criterion = WorldModelLoss(config.get("loss_weights"), float(config.get("positive_weight", 1.0))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 1e-4))
    )
    epochs = int(args.epochs or config.get("epochs", 60))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    accumulation = max(1, math.ceil(16 / batch_size))
    patience = int(config.get("patience", 8)); stale = 0; best = math.inf; best_state = None; history = []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); running = []
        sequence_iterator = iter(sequence_loader) if sequence_loader is not None else None
        for batch_index, batch in enumerate(train_loader):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model_forward(model, config, batch, device)
                loss, parts = criterion(prediction, targets(batch, device))
                sequence_every = max(1, int(config.get("sequence_every_n_batches", 1)))
                if sequence_iterator is not None and batch_index % sequence_every == 0:
                    try:
                        sequence_batch = next(sequence_iterator)
                    except StopIteration:
                        sequence_iterator = iter(sequence_loader)
                        sequence_batch = next(sequence_iterator)
                    sequence_batch = {name: value.to(device) for name, value in sequence_batch.items()}
                    sequence_loss, sequence_parts = model.sequence_training_loss(
                        sequence_batch["depth"], sequence_batch["state"], sequence_batch["goal"],
                        sequence_batch["action"], sequence_batch["reward"], sequence_batch["continuation"],
                        sequence_batch["is_first"], burn_in=int(config.get("burn_in", 4)),
                    )
                    loss = loss + float(config.get("sequence_loss_weight", 0.5)) * sequence_loss
                    parts.update(sequence_parts)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                update_target = getattr(model, "update_target_encoder", None)
                if update_target is not None: update_target()
            running.append(float(loss.detach()))
        scheduler.step(); metrics = validate(model, validation_loader, criterion, config, device)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(running)), **metrics}; history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["composite"] < best - 1e-6:
            best, stale = metrics["composite"], 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if not args.overfit_episodes and stale >= patience: break
    if best_state is None: raise RuntimeError("training produced no checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state, "config": config, "training_history": history, "best_validation_composite": best,
        "normalization": normalization_stats(train_dataset),
        "calibration": {"collision_temperature": 1.0, "failure_temperature": 1.0},
        "split_manifest_sha256": split_payload.get("manifest_sha256", hashlib.sha256(args.splits.read_bytes()).hexdigest()),
        "training_seed": args.seed,
    }, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
