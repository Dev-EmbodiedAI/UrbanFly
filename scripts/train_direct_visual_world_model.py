#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.direct_visual_world_model import (  # noqa: E402
    DirectVisualWorldModel,
    DirectWorldModelConfig,
    save_direct_checkpoint,
)
from urbanfly_vln.visual_world_model_data import (  # noqa: E402
    DirectTransitionDataset,
    dataset_manifest,
    discover_visual_runs,
    load_visual_episode,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the direct RGB-D flight world model.")
    result.add_argument("--data-root", type=Path, action="append", default=[])
    result.add_argument("--run-dir", type=Path, action="append", default=[])
    result.add_argument("--validation-run-dir", type=Path, action="append", default=[])
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--preset", choices=["small", "medium", "large"], default="medium")
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--validation-fraction", type=float, default=0.2)
    result.add_argument("--depth-max-m", type=float, default=20.0)
    result.add_argument("--risk-depth-m", type=float, default=8.0)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--seed", type=int, default=29)
    result.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    result.add_argument("--no-amp", action="store_true")
    result.add_argument("--max-train-steps", type=int, default=0)
    return result


def split_runs(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    explicit_validation = {path.resolve() for path in args.validation_run_dir}
    runs = set(discover_visual_runs(args.data_root)) | {path.resolve() for path in args.run_dir}
    runs -= explicit_validation
    runs = sorted(runs)
    if explicit_validation:
        return runs, sorted(explicit_validation)
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    rng = random.Random(args.seed)
    rng.shuffle(runs)
    count = min(max(1, round(len(runs) * args.validation_fraction)), len(runs) - 1)
    return runs[:-count], runs[-count:]


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main() -> None:
    args = parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_amp = device.type == "cuda" and not args.no_amp
    train_runs, validation_runs = split_runs(args)
    episode_kwargs = {"depth_max_m": args.depth_max_m, "risk_depth_m": args.risk_depth_m}
    train_episodes = [load_visual_episode(path, **episode_kwargs) for path in train_runs]
    validation_episodes = [load_visual_episode(path, **episode_kwargs) for path in validation_runs]
    config = DirectWorldModelConfig.preset(args.preset)
    train_dataset = DirectTransitionDataset(
        train_episodes,
        image_size=config.image_size,
        bottom_crop_fraction=config.bottom_crop_fraction,
    )
    validation_dataset = DirectTransitionDataset(
        validation_episodes,
        image_size=config.image_size,
        bottom_crop_fraction=config.bottom_crop_fraction,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = DirectVisualWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: defaultdict[str, float] = defaultdict(float)
        examples = 0
        for step, raw in enumerate(train_loader, start=1):
            batch = move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss, metrics = model.loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 20.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["observations"])
            examples += count
            for key, value in metrics.items():
                totals[key] += float(value) * count
            if args.max_train_steps and step >= args.max_train_steps:
                break

        model.eval()
        validation_loss = 0.0
        validation_examples = 0
        with torch.inference_mode():
            for raw in validation_loader:
                batch = move(raw, device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss, _ = model.loss(batch)
                validation_loss += float(loss) * len(batch["observations"])
                validation_examples += len(batch["observations"])
                if args.max_train_steps:
                    break
        validation_loss /= max(validation_examples, 1)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(examples, 1),
            "train_state_loss": totals["state"] / max(examples, 1),
            "train_reward_loss": totals["reward"] / max(examples, 1),
            "validation_loss": validation_loss,
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metadata = {
        "preset": args.preset,
        "parameters": model.parameter_count,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        "train": dataset_manifest(train_episodes),
        "validation": dataset_manifest(validation_episodes),
    }
    output = args.output.resolve()
    save_direct_checkpoint(output, model, metadata=metadata)
    output.with_suffix(".history.json").write_text(
        json.dumps({"metadata": metadata, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
