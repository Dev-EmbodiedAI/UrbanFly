#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.latent_world_model import DynamicsMLP  # noqa: E402
from urbanfly_vln.world_model_data import grouped_split, samples_from_run, stack_samples  # noqa: E402
from urbanfly_vln.world_model_metrics import fit_ensemble_temperature, json_ready, risk_report  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the UrbanFly language-conditioned risk world-model ensemble.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Training/auto-split rollout directory.")
    parser.add_argument("--validation-run-dir", type=Path, action="append", default=[], help="Explicit held-out rollout directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample runs' transitions per member. Use --no-bootstrap for full-data deep ensembles.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--language-dim", type=int, default=16)
    parser.add_argument("--risk-horizon", type=int, default=3)
    parser.add_argument("--near-miss-depth-m", type=float, default=5.0)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr-patience", type=int, default=15)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--risk-loss-weight", type=float, default=0.7)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-train-risk-positives", type=int, default=1)
    parser.add_argument("--minimum-validation-risk-positives", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _load_samples(args: argparse.Namespace):
    common = {
        "language_dimensions": args.language_dim,
        "risk_horizon": args.risk_horizon,
        "near_miss_depth_m": args.near_miss_depth_m,
    }
    samples = []
    for run_dir in [*args.run_dir, *args.validation_run_dir]:
        samples.extend(samples_from_run(run_dir, **common))
    explicit = {path.resolve().name for path in args.validation_run_dir}
    split = grouped_split(
        samples,
        validation_sources=explicit or None,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    return samples, split


def _evaluate_members(
    models: list[DynamicsMLP],
    x_norm: np.ndarray,
    indices: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(x_norm[indices]).to(device)
    continuous, logits = [], []
    with torch.inference_mode():
        for model in models:
            output = model(tensor).cpu().numpy()
            continuous.append(output[:, :6] * y_scale + y_mean)
            logits.append(output[:, 6])
    members = np.stack(continuous)
    member_logits = np.stack(logits)
    uncertainty = np.std(members[:, :, 5], axis=0) + 0.25 * np.linalg.norm(np.std(members[:, :, :3], axis=0), axis=1)
    return members.mean(axis=0), member_logits, uncertainty


def _continuous_metrics(prediction: np.ndarray, target: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    return {
        "delta_position_rmse_m": float(np.sqrt(np.mean((prediction[:, :3] - target[:, :3]) ** 2))),
        "speed_rmse_mps": float(np.sqrt(np.mean((prediction[:, 3] - target[:, 3]) ** 2))),
        "p05_depth_rmse_m": float(np.sqrt(np.mean((prediction[:, 4] - target[:, 4]) ** 2))),
        "progress_rmse_m": float(np.sqrt(np.mean((prediction[:, 5] - target[:, 5]) ** 2))),
        "mean_epistemic_uncertainty": float(np.mean(uncertainty)),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.ensemble_size < 1 or args.language_dim < 0 or args.risk_horizon < 1:
        raise ValueError("ensemble size and risk horizon must be positive; language dim must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)

    samples, split = _load_samples(args)
    x, y, risk = stack_samples(samples)
    train_idx, val_idx = split.train_indices, split.validation_indices
    x_mean, x_scale = x[train_idx].mean(0), x[train_idx].std(0)
    y_mean, y_scale = y[train_idx].mean(0), y[train_idx].std(0)
    x_scale[x_scale < 1e-5] = 1.0
    y_scale[y_scale < 1e-5] = 1.0
    x_norm = ((x - x_mean) / x_scale).astype(np.float32)
    y_norm = ((y - y_mean) / y_scale).astype(np.float32)
    positives = float(risk[train_idx].sum())
    validation_positives = int(risk[val_idx].sum())
    if positives < args.minimum_train_risk_positives:
        raise ValueError(
            f"training split has only {int(positives)} risk positives; "
            f"requires {args.minimum_train_risk_positives}"
        )
    if validation_positives < args.minimum_validation_risk_positives:
        raise ValueError(
            f"validation split has only {validation_positives} risk positives; "
            f"requires {args.minimum_validation_risk_positives}"
        )
    pos_weight = max((len(train_idx) - positives) / max(positives, 1.0), 1.0)

    model_states: list[dict[str, torch.Tensor]] = []
    trained_models: list[DynamicsMLP] = []
    histories: list[list[dict[str, float]]] = []
    for member in range(args.ensemble_size):
        member_seed = args.seed + member
        torch.manual_seed(member_seed)
        rng = np.random.default_rng(member_seed)
        member_indices = (
            rng.choice(train_idx, size=len(train_idx), replace=True)
            if args.bootstrap
            else rng.permutation(train_idx)
        )
        dataset = TensorDataset(
            torch.from_numpy(x_norm[member_indices]),
            torch.from_numpy(y_norm[member_indices]),
            torch.from_numpy(risk[member_indices, None]),
        )
        loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)
        model = DynamicsMLP(
            args.hidden_dim,
            input_dim=x.shape[1],
            dropout=args.dropout,
            layer_norm=args.layer_norm,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_learning_rate,
        )
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        val_x = torch.from_numpy(x_norm[val_idx]).to(device)
        val_y = torch.from_numpy(y_norm[val_idx]).to(device)
        val_risk = torch.from_numpy(risk[val_idx, None]).to(device)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            for xb, yb, rb in loader:
                xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
                output = model(xb)
                dynamics_loss = nn.functional.smooth_l1_loss(output[:, :6], yb)
                risk_loss = bce(output[:, 6:7], rb)
                loss = dynamics_loss + args.risk_loss_weight * risk_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss_sum += float(loss.detach()) * len(xb)

            model.eval()
            with torch.inference_mode():
                val_output = model(val_x)
                val_dynamics = nn.functional.smooth_l1_loss(val_output[:, :6], val_y)
                val_risk_loss = bce(val_output[:, 6:7], val_risk)
                val_loss = float(val_dynamics + args.risk_loss_weight * val_risk_loss)
            scheduler.step(val_loss)
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss_sum / len(dataset),
                    "val_loss": val_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if stale_epochs >= args.patience:
                break

        if best_state is None:
            raise RuntimeError("training did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        trained_models.append(model)
        model_states.append({key: value.detach().cpu() for key, value in best_state.items()})
        histories.append(history)

    prediction, member_logits, uncertainty = _evaluate_members(
        trained_models, x_norm, val_idx, y_mean, y_scale, device
    )
    temperature = fit_ensemble_temperature(risk[val_idx], member_logits)
    probability = np.mean(
        1.0 / (1.0 + np.exp(-np.clip(member_logits / temperature, -30.0, 30.0))), axis=0
    )
    metrics: dict[str, object] = {
        "format": "urbanfly-world-model-metrics-0.2",
        "samples": len(x),
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "train_risk_positives": int(positives),
        "validation_risk_positives": validation_positives,
        "split_strategy": split.strategy,
        "train_sources": list(split.train_sources),
        "validation_sources": list(split.validation_sources),
        "scenes": sorted({sample.scene_id for sample in samples}),
        "risk_horizon": args.risk_horizon,
        "near_miss_depth_m": args.near_miss_depth_m,
        "risk_temperature": temperature,
        "ensemble_size": args.ensemble_size,
        "device": str(device),
        "optimization": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "layer_norm": args.layer_norm,
            "bootstrap": args.bootstrap,
            "member_best_val_loss": [min(row["val_loss"] for row in history) for history in histories],
            "member_last_val_loss": [history[-1]["val_loss"] for history in histories],
            "member_best_epoch": [
                int(min(history, key=lambda row: row["val_loss"])["epoch"]) for history in histories
            ],
            "member_last_to_best_ratio": [
                history[-1]["val_loss"] / max(min(row["val_loss"] for row in history), 1e-12)
                for history in histories
            ],
            "member_epochs": [int(history[-1]["epoch"]) for history in histories],
        },
        **_continuous_metrics(prediction, y[val_idx], uncertainty),
        "risk": risk_report(risk[val_idx], probability, args.risk_threshold),
    }
    per_source: dict[str, object] = {}
    for source in split.validation_sources:
        mask = np.asarray([samples[index].source == source for index in val_idx])
        if np.any(mask):
            per_source[source] = {
                **_continuous_metrics(prediction[mask], y[val_idx][mask], uncertainty[mask]),
                "risk": risk_report(risk[val_idx][mask], probability[mask], args.risk_threshold),
            }
    metrics["per_validation_source"] = per_source

    metrics = json_ready(metrics)
    payload = {
        "format": "urbanfly-latent-world-model-v2",
        "feature_version": "state-action-language-v2",
        "hidden_dim": args.hidden_dim,
        "input_dim": int(x.shape[1]),
        "language_dimensions": args.language_dim,
        "architecture": {"dropout": args.dropout, "layer_norm": args.layer_norm},
        "risk_temperature": temperature,
        "model_state_dicts": model_states,
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(),
        "y_scale": y_scale.tolist(),
        "label_config": {"risk_horizon": args.risk_horizon, "near_miss_depth_m": args.near_miss_depth_m},
        "training_config": vars(args) | {"run_dir": [str(path.resolve()) for path in args.run_dir], "validation_run_dir": [str(path.resolve()) for path in args.validation_run_dir]},
        "metrics": metrics,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    output.with_suffix(".history.json").write_text(
        json.dumps({"members": histories, "log_interval_epochs": 1}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, history in enumerate(histories):
        epochs = [entry["epoch"] for entry in history]
        axes[0].plot(epochs, [entry["train_loss"] for entry in history], alpha=0.65, label=f"member {index + 1}")
        axes[1].plot(epochs, [entry["val_loss"] for entry in history], alpha=0.65)
    axes[0].set_title("Bootstrap training loss")
    axes[1].set_title("Held-out run validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".training_curves.png"), dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
