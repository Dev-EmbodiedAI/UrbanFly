#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.observation_policy import (  # noqa: E402
    ACTION_NAMES,
    HelsinkiObservationPolicy,
    ObservationPolicyConfig,
    save_observation_policy_checkpoint,
)
from urbanfly_vln.observation_policy_data import (  # noqa: E402
    HelsinkiObservationPolicyDataset,
    action_statistics,
    load_qa_episode_records,
    tail_episode_split,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Train the compact Helsinki RGB-D/Local-Goal observation policy."
    )
    result.add_argument("--qa", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--validation-episodes", type=int, default=20)
    result.add_argument("--epochs", type=int, default=12)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--history-frames", type=int, default=2)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--seed", type=int, default=20260829)
    result.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    result.add_argument("--max-train-batches", type=int, default=0)
    result.add_argument("--max-validation-batches", type=int, default=0)
    return result


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in {"rgb", "depth_m", "depth_valid", "public_state", "target_action"}
    }


def evaluate(
    model: HelsinkiObservationPolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 0,
) -> dict[str, object]:
    model.eval()
    absolute = np.zeros(4, dtype=np.float64)
    squared_normalized = np.zeros(4, dtype=np.float64)
    baseline_absolute = np.zeros(4, dtype=np.float64)
    yaw_correct = 0
    yaw_count = 0
    count = 0
    latency_ms: list[float] = []
    with torch.inference_mode():
        for batch_index, raw in enumerate(loader, start=1):
            batch = move_batch(raw, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            predicted = model(
                batch["rgb"], batch["depth_m"], batch["depth_valid"], batch["public_state"]
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ms.append((time.perf_counter() - started) * 1000.0 / len(predicted))
            target = batch["target_action"]
            error = predicted - target
            absolute += error.abs().sum(dim=0).cpu().numpy()
            squared_normalized += torch.square(error / model.action_std).sum(dim=0).cpu().numpy()
            baseline = model.action_mean.expand_as(target)
            baseline_absolute += (baseline - target).abs().sum(dim=0).cpu().numpy()
            meaningful = target[:, 3].abs() >= 0.05
            yaw_correct += int((torch.sign(predicted[meaningful, 3]) == torch.sign(target[meaningful, 3])).sum())
            yaw_count += int(meaningful.sum())
            count += len(target)
            if max_batches and batch_index >= max_batches:
                break
    if count == 0:
        raise RuntimeError("validation produced no samples")
    mae = absolute / count
    baseline_mae = baseline_absolute / count
    return {
        "samples": count,
        "mae": dict(zip(ACTION_NAMES, mae.tolist())),
        "normalized_rmse": dict(zip(ACTION_NAMES, np.sqrt(squared_normalized / count).tolist())),
        "mean_action_baseline_mae": dict(zip(ACTION_NAMES, baseline_mae.tolist())),
        "relative_mae_to_mean_baseline": float(mae.sum() / max(baseline_mae.sum(), 1e-9)),
        "meaningful_yaw_sign_accuracy": float(yaw_correct / max(yaw_count, 1)),
        "meaningful_yaw_samples": yaw_count,
        "inference_latency_ms_per_sample_mean": float(np.mean(latency_ms)),
        "inference_latency_ms_per_sample_p95": float(np.percentile(latency_ms, 95)),
    }


def task_counts(records) -> dict[str, int]:
    return dict(sorted(Counter(record.task_type for record in records).items()))


def main() -> None:
    args = parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    records = load_qa_episode_records(args.qa)
    split = tail_episode_split(records, validation_episodes=args.validation_episodes)
    action_mean, action_std = action_statistics(split.train)
    config = ObservationPolicyConfig(history_frames=args.history_frames)
    train_dataset = HelsinkiObservationPolicyDataset(
        split.train, history_frames=config.history_frames, seed=args.seed
    )
    validation_dataset = HelsinkiObservationPolicyDataset(
        split.validation, history_frames=config.history_frames, seed=args.seed
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    model = HelsinkiObservationPolicy(
        config, action_mean=action_mean, action_std=action_std
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        samples = 0
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                predicted = model(
                    batch["rgb"], batch["depth_m"], batch["depth_valid"], batch["public_state"]
                )
                loss = model.loss(predicted, batch["target_action"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(predicted)
            samples += len(predicted)
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        metrics = evaluate(
            model,
            validation_loader,
            device,
            max_batches=args.max_validation_batches,
        )
        score = float(metrics["relative_mae_to_mean_baseline"])
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(samples, 1),
            "validation_relative_mae": score,
            "validation_yaw_sign_accuracy": float(metrics["meaningful_yaw_sign_accuracy"]),
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_metric:
            best_metric = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training produced no best state")
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, validation_loader, device)
    metadata = {
        "status": "OFFLINE_BASELINE_ONLY",
        "source_qa": str(args.qa.resolve()),
        "split": {
            "kind": "held-out episode tail; not an official spatial-unseen split",
            "train_episode_indices": [record.episode_index for record in split.train],
            "validation_episode_indices": [record.episode_index for record in split.validation],
            "train_task_counts": task_counts(split.train),
            "validation_task_counts": task_counts(split.validation),
        },
        "target": "actions/commanded_body_flu",
        "policy_inputs_exclude_privileged": True,
        "parameters": model.parameter_count,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "device": str(device),
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        ),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "validation": final_metrics,
        "history": history,
        "limitations": [
            "NOT TESTED online in the Helsinki simulator",
            "NOT TESTED on a second city or appearance domain",
            "100 source episodes are all labelled train in Dataset v1 metadata",
        ],
    }
    output = args.output.resolve()
    save_observation_policy_checkpoint(output, model, metadata=metadata)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(output), "metrics": str(metrics_path), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
