#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.visual_world_model import VisualRSSM, VisualWorldModelConfig  # noqa: E402


def synthetic_batch(config: VisualWorldModelConfig, batch_size: int, sequence_length: int, device: torch.device):
    return {
        "observations": torch.rand(
            batch_size,
            sequence_length,
            config.image_channels,
            config.image_size,
            config.image_size,
            device=device,
        ),
        "actions": torch.rand(batch_size, sequence_length, config.action_dim, device=device) * 2 - 1,
        "states": torch.rand(batch_size, sequence_length, config.state_dim, device=device),
        "rewards": torch.randn(batch_size, sequence_length, device=device),
        "risks": torch.randint(0, 2, (batch_size, sequence_length), device=device).float(),
        "continues": torch.ones(batch_size, sequence_length, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure visual RSSM training and planning VRAM.")
    parser.add_argument("--preset", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    device = torch.device("cuda")
    use_amp = not args.no_amp
    config = VisualWorldModelConfig.preset(args.preset)
    model = VisualRSSM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batch = synthetic_batch(config, args.batch_size, args.sequence_length, device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss, _ = model.loss(batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    train_allocated = torch.cuda.max_memory_allocated() / 2**30
    train_reserved = torch.cuda.max_memory_reserved() / 2**30

    del optimizer, scaler, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    initial = model.initial(args.candidates, device)
    actions = torch.rand(args.candidates, args.horizon, config.action_dim, device=device) * 2 - 1
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        imagined = model.imagine(initial, actions)
        score = imagined["reward"].sum(1) - 8.0 * torch.sigmoid(imagined["risk_logit"]).sum(1)
        _ = int(score.argmax())
    torch.cuda.synchronize()
    planning_seconds = time.perf_counter() - started
    result = {
        "gpu": torch.cuda.get_device_name(),
        "preset": args.preset,
        "parameters": model.parameter_count,
        "parameter_millions": model.parameter_count / 1e6,
        "amp": use_amp,
        "training": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "steps": args.steps,
            "seconds_per_step": train_seconds / args.steps,
            "peak_allocated_gb": train_allocated,
            "peak_reserved_gb": train_reserved,
        },
        "planning": {
            "candidates": args.candidates,
            "horizon": args.horizon,
            "latency_ms": planning_seconds * 1000.0,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
