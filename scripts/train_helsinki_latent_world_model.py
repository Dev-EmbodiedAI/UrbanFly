#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.navigation_world_model import (  # noqa: E402
    HelsinkiLatentWorldModel,
    NavigationWorldModelConfig,
    save_navigation_world_model_checkpoint,
)
from urbanfly_vln.navigation_world_model_data import (  # noqa: E402
    HelsinkiLatentTransitionDataset,
)
from urbanfly_vln.observation_policy import load_observation_policy_checkpoint  # noqa: E402
from urbanfly_vln.observation_policy_data import (  # noqa: E402
    load_qa_episode_records,
    tail_episode_split,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the Helsinki action-conditioned latent world model.")
    result.add_argument("--qa", type=Path, required=True)
    result.add_argument("--policy", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--validation-episodes", type=int, default=20)
    result.add_argument("--epochs", type=int, default=12)
    result.add_argument("--batch-size", type=int, default=512)
    result.add_argument("--encode-batch-size", type=int, default=128)
    result.add_argument("--learning-rate", type=float, default=5e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--ensemble-size", type=int, default=3)
    result.add_argument("--hidden-dim", type=int, default=256)
    result.add_argument("--seed", type=int, default=20260830)
    result.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


@torch.inference_mode()
def encode_dataset(dataset, policy, device, batch_size: int) -> tuple[torch.Tensor, ...]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    latent_rows, action_rows, target_rows = [], [], []
    policy.eval()
    for raw in loader:
        current = policy.encode(
            raw["rgb"].to(device, non_blocking=True),
            raw["depth_m"].to(device, non_blocking=True),
            raw["depth_valid"].to(device, non_blocking=True),
            raw["public_state"].to(device, non_blocking=True),
        )
        following = policy.encode(
            raw["next_rgb"].to(device, non_blocking=True),
            raw["next_depth_m"].to(device, non_blocking=True),
            raw["next_depth_valid"].to(device, non_blocking=True),
            raw["next_public_state"].to(device, non_blocking=True),
        )
        target = torch.cat((raw["physical_target"].float(), (following - current).cpu()), dim=1)
        latent_rows.append(current.cpu())
        action_rows.append(raw["executed_action"].float())
        target_rows.append(target)
    return torch.cat(latent_rows), torch.cat(action_rows), torch.cat(target_rows)


@torch.inference_mode()
def evaluate(model, latent, action, target, train_target_mean, device) -> dict[str, float]:
    latent = latent.to(device)
    action = action.to(device)
    target = target.to(device)
    candidates = action[:, None]
    prediction = model.predict(latent, candidates)
    mean = torch.cat(
        (prediction["physical_mean"][:, 0], prediction["next_latent_mean"][:, 0] - latent),
        dim=1,
    )
    error = mean - target
    position_rmse = float(torch.sqrt(torch.mean(torch.square(error[:, :3]))).cpu())
    progress_rmse = float(torch.sqrt(torch.mean(torch.square(error[:, 3]))).cpu())
    clearance_mae = float(torch.mean(torch.abs(error[:, 4])).cpu())
    latent_delta_rmse = float(torch.sqrt(torch.mean(torch.square(error[:, 5:]))).cpu())
    baseline = train_target_mean.to(device)
    baseline_error = baseline[None] - target
    persistence_error = -target[:, 5:]
    zero_actions = torch.zeros_like(action)[:, None]
    zero_prediction = model.predict(latent, zero_actions)["physical_mean"][:, 0]
    action_sensitivity = torch.linalg.vector_norm(
        prediction["physical_mean"][:, 0, :4] - zero_prediction[:, :4], dim=1
    ).mean()
    physical_std = prediction["physical_std"][:, 0]
    return {
        "examples": int(len(target)),
        "position_delta_rmse_m": position_rmse,
        "position_delta_mean_baseline_rmse_m": float(torch.sqrt(torch.mean(torch.square(baseline_error[:, :3]))).cpu()),
        "route_progress_delta_rmse_m": progress_rmse,
        "route_progress_mean_baseline_rmse_m": float(torch.sqrt(torch.mean(torch.square(baseline_error[:, 3]))).cpu()),
        "next_clearance_mae_m": clearance_mae,
        "next_clearance_mean_baseline_mae_m": float(torch.mean(torch.abs(baseline_error[:, 4])).cpu()),
        "latent_delta_rmse": latent_delta_rmse,
        "latent_persistence_rmse": float(torch.sqrt(torch.mean(torch.square(persistence_error))).cpu()),
        "action_sensitivity_mean": float(action_sensitivity.cpu()),
        "ensemble_physical_std_mean": float(physical_std.mean().cpu()),
    }


def main() -> None:
    args = parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    records = load_qa_episode_records(args.qa)
    split = tail_episode_split(records, validation_episodes=args.validation_episodes)
    policy, policy_metadata = load_observation_policy_checkpoint(args.policy, device=device)
    train_source = HelsinkiLatentTransitionDataset(split.train, history_frames=policy.config.history_frames)
    validation_source = HelsinkiLatentTransitionDataset(split.validation, history_frames=policy.config.history_frames)
    encode_started = time.perf_counter()
    train_latent, train_action, train_target = encode_dataset(
        train_source, policy, device, args.encode_batch_size
    )
    validation_latent, validation_action, validation_target = encode_dataset(
        validation_source, policy, device, args.encode_batch_size
    )
    encode_s = time.perf_counter() - encode_started
    target_mean = train_target.mean(dim=0)
    target_std = train_target.std(dim=0, unbiased=False).clamp_min(1e-3)
    normalized_target = (train_target - target_mean) / target_std
    model = HelsinkiLatentWorldModel(
        NavigationWorldModelConfig(
            latent_dim=train_latent.shape[1],
            hidden_dim=args.hidden_dim,
            ensemble_size=args.ensemble_size,
        ),
        target_mean=target_mean,
        target_std=target_std,
    ).to(device)
    training_started = time.perf_counter()
    member_summaries = []
    for member_index in range(args.ensemble_size):
        torch.manual_seed(args.seed + member_index)
        dataset = TensorDataset(train_latent, train_action, normalized_target)
        generator = torch.Generator().manual_seed(args.seed + member_index)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
        optimizer = torch.optim.AdamW(
            model.members[member_index].parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best = float("inf")
        best_state = None
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            model.members[member_index].train()
            total, count = 0.0, 0
            for latent, action, target in loader:
                latent = latent.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                prediction = model.member_prediction(member_index, latent, action)
                physical_loss = torch.nn.functional.smooth_l1_loss(prediction[:, :5], target[:, :5])
                latent_loss = torch.nn.functional.smooth_l1_loss(prediction[:, 5:], target[:, 5:])
                loss = physical_loss + 0.15 * latent_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.members[member_index].parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(latent)
                count += len(latent)
            epoch_loss = total / max(count, 1)
            if epoch_loss < best:
                best = epoch_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.members[member_index].state_dict())
        if best_state is None:
            raise RuntimeError("world-model member did not produce a checkpoint")
        model.members[member_index].load_state_dict(best_state)
        model.members[member_index].eval()
        member_summaries.append({"member": member_index, "best_epoch": best_epoch, "best_train_loss": best})

    metrics = evaluate(
        model,
        validation_latent,
        validation_action,
        validation_target,
        target_mean,
        device,
    )
    ratios = {
        "position": metrics["position_delta_rmse_m"] / max(metrics["position_delta_mean_baseline_rmse_m"], 1e-9),
        "progress": metrics["route_progress_delta_rmse_m"] / max(metrics["route_progress_mean_baseline_rmse_m"], 1e-9),
        "clearance": metrics["next_clearance_mae_m"] / max(metrics["next_clearance_mean_baseline_mae_m"], 1e-9),
        "latent": metrics["latent_delta_rmse"] / max(metrics["latent_persistence_rmse"], 1e-9),
    }
    passed = all(np.isfinite(list(ratios.values()))) and all(value < 1.0 for value in ratios.values()) and metrics["action_sensitivity_mean"] > 1e-4
    metadata = {
        "status": "OFFLINE_PASS" if passed else "OFFLINE_FAIL",
        "scope": "one-step action-conditioned public-observation latent dynamics; not collision-risk qualified",
        "source_qa": str(args.qa.resolve()),
        "policy_checkpoint": str(args.policy.resolve()),
        "policy_checkpoint_sha256": file_sha256(args.policy.resolve()),
        "policy_status": policy_metadata.get("status"),
        "split": {
            "train_episode_indices": [item.episode_index for item in split.train],
            "validation_episode_indices": [item.episode_index for item in split.validation],
            "train_transitions": len(train_source),
            "validation_transitions": len(validation_source),
        },
        "model": {
            "parameters": model.parameter_count,
            "ensemble_size": args.ensemble_size,
            "latent_dim": model.config.latent_dim,
            "hidden_dim": model.config.hidden_dim,
        },
        "metrics": metrics,
        "relative_to_baseline": ratios,
        "member_training": member_summaries,
        "runtime": {
            "device": str(device),
            "encoding_s": encode_s,
            "training_s": time.perf_counter() - training_started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "limitations": [
            "Dataset v1 contains no collision and no clearance below 4.288824 m; learned collision probability is not claimed.",
            "Candidate reranking must remain local to the learned policy and behind the independent backend safety shield.",
            "Held-out episodes are not a second city or official spatial-unseen split.",
        ],
    }
    output = args.output.resolve()
    save_navigation_world_model_checkpoint(output, model, metadata=metadata)
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(output), **metadata}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
