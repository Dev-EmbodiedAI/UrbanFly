from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import _bootstrap  # noqa: F401
from uav_wm_navigation.data import WorldModelDataset
from uav_wm_navigation.world_models import build_world_model


def choose_sample(dataset: WorldModelDataset, model: torch.nn.Module, device: torch.device) -> tuple[int, dict, dict]:
    """Choose a held-out frame with a dangerous candidate and non-trivial model contrast."""
    best: tuple[float, int, dict, dict] | None = None
    for index in np.linspace(0, len(dataset) - 1, min(len(dataset), 160), dtype=int):
        sample = dataset[int(index)]
        if not bool((sample["collision"] > 0.5).any()):
            continue
        inputs = [sample[name][None].to(device) for name in ("depth", "state", "goal", "trajectories")]
        with torch.inference_mode():
            output = model(*inputs)
            probability = torch.sigmoid(output["collision_logits"][0]).cpu().numpy()
        contrast = float(np.percentile(probability, 90) - np.percentile(probability, 10))
        dangerous_probability = float(probability[sample["collision"].numpy() > 0.5].max())
        score = contrast + dangerous_probability
        payload = {"collision_probability": probability}
        if best is None or score > best[0]:
            best = (score, int(index), sample, payload)
    if best is None:
        raise RuntimeError("held-out split contains no dangerous candidate in the inspected samples")
    return best[1], best[2], best[3]


def occlusion_sensitivity(
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor],
    candidate_index: int,
    device: torch.device,
    patch: int,
) -> tuple[np.ndarray, float]:
    depth = sample["depth"].clone()
    state, goal, trajectories = (sample[name][None].to(device) for name in ("state", "goal", "trajectories"))
    with torch.inference_mode():
        baseline = float(model(depth[None].to(device), state, goal, trajectories)["collision_logits"][0, candidate_index])
    height, width = depth.shape[-2:]
    variants, locations = [], []
    for y0 in range(0, height, patch):
        for x0 in range(0, width, patch):
            perturbed = depth.clone()
            perturbed[-1, 0, y0:min(y0 + patch, height), x0:min(x0 + patch, width)] = 0.0
            variants.append(perturbed)
            locations.append((y0, min(y0 + patch, height), x0, min(x0 + patch, width)))
    values = []
    with torch.inference_mode():
        for begin in range(0, len(variants), 32):
            batch = torch.stack(variants[begin:begin + 32]).to(device)
            count = batch.shape[0]
            output = model(
                batch,
                state.expand(count, -1, -1),
                goal.expand(count, -1),
                trajectories.expand(count, -1, -1, -1),
            )
            values.extend(output["collision_logits"][:, candidate_index].cpu().numpy().tolist())
    heatmap = np.zeros((height, width), dtype=np.float32)
    for value, (y0, y1, x0, x1) in zip(values, locations):
        heatmap[y0:y1, x0:x1] = abs(float(value) - baseline)
    if float(heatmap.max()) > 0:
        heatmap /= float(heatmap.max())
    return heatmap, baseline


def candidate_token_attention(
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor],
    candidate_index: int,
    device: torch.device,
) -> np.ndarray:
    """Capture real multi-head attention over context + candidate time tokens."""
    layer = model.predictor.layers[0]
    original_forward = layer.self_attn.forward
    captured: list[torch.Tensor] = []

    def recording_forward(*call_args, **call_kwargs):
        call_kwargs["need_weights"] = True
        call_kwargs["average_attn_weights"] = False
        result, weights = original_forward(*call_args, **call_kwargs)
        captured.append(weights.detach().cpu())
        return result, weights

    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    layer.self_attn.forward = recording_forward
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode():
            model(*(sample[name][None].to(device) for name in ("depth", "state", "goal", "trajectories")))
    finally:
        layer.self_attn.forward = original_forward
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    if not captured:
        raise RuntimeError("failed to capture JEPA transformer attention weights")
    # The encoder flattens [B,N] to B*N. Here B=1, so the candidate is the
    # leading batch index. Average the four real attention heads for display.
    return captured[0][candidate_index].mean(dim=0).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an honest image-space sensitivity map for the lightweight JEPA.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = dict(checkpoint["config"])
    model = build_world_model(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    dataset = WorldModelDataset(
        split_payload["test"],
        history=int(config["history"]),
        depth_max_m=float(config["depth_max_m"]),
        future_observations=0,
        trajectory_steps=int(config["trajectory_steps"]),
    )
    index, sample, prediction = choose_sample(dataset, model, torch.device(args.device))
    probabilities = prediction["collision_probability"]
    dangerous = sample["collision"].numpy() > 0.5
    candidate_index = int(np.argmax(np.where(dangerous, probabilities, -1.0)))
    heatmap, baseline_logit = occlusion_sensitivity(
        model, sample, candidate_index, torch.device(args.device), int(args.patch_size)
    )
    token_attention = candidate_token_attention(model, sample, candidate_index, torch.device(args.device))

    depth_m = sample["depth"][-1, 0].numpy() * float(config["depth_max_m"])
    depth_color = cv2.cvtColor(
        cv2.applyColorMap(np.clip(depth_m / float(config["depth_max_m"]) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )
    heat_color = cv2.cvtColor(
        cv2.applyColorMap(np.clip(heatmap * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB,
    )
    overlay = np.clip(0.58 * depth_color + 0.42 * heat_color, 0, 255).astype(np.uint8)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4), constrained_layout=True)
    depth_image = axes[0, 0].imshow(depth_m, cmap="turbo", vmin=0, vmax=float(config["depth_max_m"]))
    axes[0, 0].set_title("真实度量深度输入（最后一帧）")
    fig.colorbar(depth_image, ax=axes[0, 0], fraction=0.045, label="m")
    sensitivity_image = axes[0, 1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title("局部遮挡敏感度：碰撞 logit 变化")
    fig.colorbar(sensitivity_image, ax=axes[0, 1], fraction=0.045, label="归一化重要性")
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("深度 + 风险归因叠加（亮区对判断更关键）")
    colors = np.where(dangerous, "#D64545", "#3A7CA5")
    axes[1, 1].bar(np.arange(len(probabilities)), probabilities, color=colors)
    axes[1, 1].axvline(candidate_index, color="#F2A900", lw=3, label=f"解释候选 {candidate_index}")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("YOPO candidate ID")
    axes[1, 1].set_ylabel("校准前碰撞概率")
    axes[1, 1].set_title("15 候选风险；红色为几何危险标签")
    axes[1, 1].legend(frameon=False)
    for axis in axes[:2].flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Action-Conditioned JEPA 图像空间解释：遮挡敏感度，不冒充 ViT self-attention",
        fontsize=16,
        fontweight="bold",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "jepa_image_occlusion_sensitivity.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)

    token_labels = ["context"] + [f"t{step}" for step in range(token_attention.shape[0] - 1)]
    token_figure, token_axis = plt.subplots(figsize=(8.4, 7.0), constrained_layout=True)
    token_image = token_axis.imshow(token_attention, cmap="magma", vmin=0.0)
    token_axis.set_xticks(np.arange(len(token_labels)), token_labels, rotation=45, ha="right")
    token_axis.set_yticks(np.arange(len(token_labels)), token_labels)
    token_axis.set_xlabel("被关注的 context / candidate 时间 token")
    token_axis.set_ylabel("当前查询 token")
    token_axis.set_title(
        f"JEPA 第1层真实多头注意力均值：候选 {candidate_index}\n"
        "因果 mask 使上三角为零；动作 token 只能读取当前及过去",
        fontweight="bold",
    )
    token_figure.colorbar(token_image, ax=token_axis, fraction=0.046, label="attention weight")
    token_output = args.output_dir / "jepa_candidate_token_attention.png"
    token_figure.savefig(token_output, dpi=220)
    plt.close(token_figure)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split_manifest": str(args.splits.resolve()),
        "dataset_index": index,
        "episode_path": str(dataset.index[index][0]),
        "episode_step": int(dataset.index[index][1]),
        "candidate_index": candidate_index,
        "candidate_collision_label": bool(dangerous[candidate_index]),
        "baseline_collision_logit": baseline_logit,
        "baseline_collision_probability": float(probabilities[candidate_index]),
        "patch_size": int(args.patch_size),
        "interpretation": (
            "Absolute change in the selected candidate collision logit after masking one patch "
            "of the newest depth frame to zero. This is occlusion sensitivity, not ViT attention."
        ),
        "output": str(output.resolve()),
        "token_attention_output": str(token_output.resolve()),
        "token_attention": (
            "Actual first-layer multi-head attention weights averaged across heads for the selected "
            "candidate's context and causal action-time tokens. It is not image-patch attention."
        ),
    }
    (args.output_dir / "jepa_image_occlusion_sensitivity.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
