from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from uav_wm_navigation.data import WAMMPCTransitionDataset
from uav_wm_navigation.utils.config import load_yaml
from uav_wm_navigation.world_models import JEPAWorldModelAdapter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(model, batch, device, weights):
    observation = {
        "depth": batch["depth"].to(device),
        "state_history": batch["state_history"].to(device),
        "goal_body": batch["goal_body"].to(device),
    }
    next_observation = {
        "depth": batch["next_depth"].to(device),
        "state_history": batch["next_state_history"].to(device),
        "goal_body": batch["next_goal_body"].to(device),
    }
    latent = model.encode(observation)
    with torch.no_grad():
        target_latent = model.encode(next_observation)
    dt_values = batch["dt"].to(device)
    prediction = model.predict_step(
        latent,
        batch["planning_state"].to(device),
        batch["action"].to(device),
        dt=dt_values,
    )
    latent_loss = (2.0 - 2.0 * torch.nn.functional.cosine_similarity(
        prediction["latent"], target_latent, dim=-1
    )).mean()
    target_state = batch["next_planning_state"].to(device)
    state_loss = torch.nn.functional.smooth_l1_loss(prediction["state"][:, :6], target_state[:, :6])
    collision_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction["collision_logits"], batch["collision"].to(device).float()
    )
    total = (
        float(weights.get("latent", 1.0)) * latent_loss
        + float(weights.get("state", 1.0)) * state_loss
        + float(weights.get("collision", 1.0)) * collision_loss
    )
    return total, {"latent": latent_loss, "state": state_loss, "collision": collision_loss}


@torch.no_grad()
def validate(model, loader, device, weights) -> dict[str, float]:
    model.eval()
    values = []
    parts: dict[str, list[float]] = {"latent": [], "state": [], "collision": []}
    for batch in loader:
        loss, components = compute_loss(model, batch, device, weights)
        values.append(float(loss))
        for name, value in components.items():
            parts[name].append(float(value))
    return {
        "validation_loss": float(np.mean(values)) if values else float("inf"),
        **{f"validation_{name}_loss": float(np.mean(items)) for name, items in parts.items() if items},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train lightweight action-conditioned JEPA dynamics/probes; no pixel decoder.")
    parser.add_argument("--config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/wam_mpc_jepa.yaml")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path,
                        help="Optional existing candidate-JEPA checkpoint used to initialize the encoder.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    set_seed(args.seed)
    config = load_yaml(args.config)
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, initialization = JEPAWorldModelAdapter.from_config(
        config, checkpoint=args.encoder_checkpoint, map_location=device
    )
    model.to(device)
    if model.parameter_count > 30_000_000:
        raise ValueError(f"adapter exceeds the 30M lightweight ceiling: {model.parameter_count:,}")
    wam, action, training = config["wam_mpc"], config["action"], config["training"]
    dataset_kwargs = {
        "history": int(wam["history"]),
        "depth_max_m": float(wam["depth_max_m"]),
        "depth_shape": tuple(wam["depth_shape"]),
        "action_scale": tuple(action["physical_limits"]),
    }
    train_dataset = WAMMPCTransitionDataset(split_payload["train"], **dataset_kwargs)
    validation_paths = split_payload.get("validation") or split_payload["train"]
    validation_dataset = WAMMPCTransitionDataset(validation_paths, **dataset_kwargs)
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("training and validation transition datasets must be non-empty")
    batch_size = int(training.get("batch_size", 64))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=int(training.get("workers", 0)), pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or training.get("epochs", 60))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    weights = training.get("loss_weights", {})
    history = []
    best_loss = float("inf")
    best_state = None
    training_steps = 0
    for epoch in range(epochs):
        model.train()
        running = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, _ = compute_loss(model, batch, device, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            update_target = getattr(model.core, "update_target_encoder", None)
            if update_target is not None:
                update_target()
            training_steps += 1
            running.append(float(loss.detach()))
        scheduler.step()
        metrics = validate(model, validation_loader, device, weights)
        record = {"epoch": epoch + 1, "training_loss": float(np.mean(running)), **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["validation_loss"] < best_loss:
            best_loss = metrics["validation_loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "urbanfly-wam-mpc-v1",
        "family": "jepa_wam_mpc",
        "adapter_state": best_state,
        "config": config,
        "training_steps": training_steps,
        "training_history": history,
        "best_validation_loss": best_loss,
        "parameter_count": model.parameter_count,
        "initialization": initialization,
        "split_manifest_sha256": split_payload.get(
            "manifest_sha256", hashlib.sha256(args.splits.read_bytes()).hexdigest()
        ),
        "training_seed": args.seed,
        "pixel_reconstruction_loss": False,
    }, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
