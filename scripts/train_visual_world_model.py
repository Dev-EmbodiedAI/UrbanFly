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

from urbanfly_vln.visual_world_model import (  # noqa: E402
    VisualRSSM,
    VisualWorldModelConfig,
    save_visual_checkpoint,
)
from urbanfly_vln.visual_world_model_data import (  # noqa: E402
    VisualSequenceDataset,
    dataset_manifest,
    discover_visual_runs,
    load_visual_episode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the UrbanFly RGB-D RSSM world model.")
    parser.add_argument("--data-root", type=Path, action="append", default=[])
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--validation-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Continue from a visual-rssm-v1 checkpoint.")
    parser.add_argument("--preset", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=100.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=20.0)
    parser.add_argument("--risk-depth-m", type=float, default=8.0)
    parser.add_argument(
        "--risk-positive-weight",
        type=float,
        default=0.0,
        help="Positive BCE weight; 0 derives the negative/positive ratio from training runs.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 runs every batch; useful for smoke tests.")
    return parser


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def split_runs(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    explicit_train = {path.resolve() for path in args.run_dir}
    explicit_validation = {path.resolve() for path in args.validation_run_dir}
    discovered = set(discover_visual_runs(args.data_root))
    all_runs = sorted((discovered | explicit_train | explicit_validation) - explicit_validation)
    if explicit_validation:
        train_runs, validation_runs = all_runs, sorted(explicit_validation)
    else:
        if len(all_runs) < 2:
            raise ValueError("at least two RGB-D runs are required for run-level train/validation splitting")
        rng = random.Random(args.seed)
        rng.shuffle(all_runs)
        validation_count = min(max(1, round(len(all_runs) * args.validation_fraction)), len(all_runs) - 1)
        train_runs, validation_runs = all_runs[:-validation_count], all_runs[-validation_count:]
    if not train_runs or not validation_runs:
        raise ValueError("train or validation run split is empty")
    return train_runs, validation_runs


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    train_runs, validation_runs = split_runs(args)
    load_kwargs = {"depth_max_m": args.depth_max_m, "risk_depth_m": args.risk_depth_m}
    train_episodes = [load_visual_episode(path, **load_kwargs) for path in train_runs]
    validation_episodes = [load_visual_episode(path, **load_kwargs) for path in validation_runs]
    dataset_kwargs = {
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "image_size": 64,
        "bottom_crop_fraction": VisualWorldModelConfig.preset(args.preset).bottom_crop_fraction,
    }
    train_dataset = VisualSequenceDataset(train_episodes, **dataset_kwargs)
    validation_dataset = VisualSequenceDataset(validation_episodes, **dataset_kwargs)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    config = VisualWorldModelConfig.preset(args.preset)
    model = VisualRSSM(config).to(device)
    if args.resume is not None:
        resume_payload = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        resume_config = VisualWorldModelConfig(**resume_payload["config"])
        if resume_config != config:
            raise ValueError("--resume checkpoint configuration does not match --preset")
        model.load_state_dict(resume_payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_validation = float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    positives = sum(float(episode.risks.sum()) for episode in train_episodes)
    frames = sum(len(episode) for episode in train_episodes)
    risk_positive_weight = (
        args.risk_positive_weight
        if args.risk_positive_weight > 0.0
        else max((frames - positives) / max(positives, 1.0), 1.0)
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: defaultdict[str, float] = defaultdict(float)
        examples = 0
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss, metrics = model.loss(batch, risk_positive_weight=risk_positive_weight)
                scaled_loss = loss / max(args.gradient_accumulation, 1)
            scaler.scale(scaled_loss).backward()
            if step % max(args.gradient_accumulation, 1) == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_size = len(batch["observations"])
            examples += batch_size
            for key, value in metrics.items():
                totals[key] += float(value) * batch_size
            if args.max_train_steps and step >= args.max_train_steps:
                break

        model.eval()
        validation_total = 0.0
        validation_examples = 0
        with torch.inference_mode():
            for raw_batch in validation_loader:
                batch = move_batch(raw_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss, _ = model.loss(batch, risk_positive_weight=risk_positive_weight)
                validation_total += float(loss) * len(batch["observations"])
                validation_examples += len(batch["observations"])
                if args.max_train_steps:
                    break
        validation_loss = validation_total / max(validation_examples, 1)
        row = {
            "epoch": float(epoch),
            "train_loss": totals["loss"] / max(examples, 1),
            "validation_loss": validation_loss,
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    peak_vram = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    metadata = {
        "preset": args.preset,
        "parameters": model.parameter_count,
        "device": str(device),
        "amp": use_amp,
        "peak_vram_gb": peak_vram,
        "best_validation_loss": best_validation,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "risk_positive_weight": risk_positive_weight,
        "train": dataset_manifest(train_episodes),
        "validation": dataset_manifest(validation_episodes),
    }
    output = args.output.resolve()
    save_visual_checkpoint(output, model, epoch=args.epochs, metadata=metadata)
    output.with_suffix(".history.json").write_text(
        json.dumps({"history": history, "metadata": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
