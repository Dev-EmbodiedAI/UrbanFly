from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def steady_p95(path: Path) -> float:
    rows = load(path)[1:]
    return float(np.percentile([row["total_planning_latency_ms"] for row in rows], 95))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Pilot accuracy and real-time trade-offs.")
    parser.add_argument("--dreamer", type=Path, required=True)
    parser.add_argument("--jepa", type=Path, required=True)
    parser.add_argument("--occflow", type=Path, required=True)
    parser.add_argument("--jepa-decisions", type=Path, required=True)
    parser.add_argument("--occflow-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ["Dreamer-style\nRSSM", "Action-conditioned\nJEPA", "OccFlow-WM"]
    metrics = [load(args.dreamer), load(args.jepa), load(args.occflow)]
    auroc = [item["collision_auroc"] for item in metrics]
    auprc = [item["collision_auprc"] for item in metrics]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor="#090d14")
    x = np.arange(3); width = 0.34
    axes[0].bar(x - width / 2, auroc, width, label="AUROC", color="#43c6f9")
    axes[0].bar(x + width / 2, auprc, width, label="AUPRC", color="#ff9f43")
    axes[0].set_xticks(x, names); axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Candidate collision prediction")
    axes[0].set_title("Held-out Pilot test (1845 labeled candidates)")
    axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=0.18)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    jepa_latency = steady_p95(args.jepa_decisions)
    occflow_latency = steady_p95(args.occflow_decisions)
    axes[1].scatter([jepa_latency], [auprc[1]], s=170, color="#66ff99", label="JEPA")
    axes[1].scatter([occflow_latency], [auprc[2]], s=170, color="#ff6b6b", marker="X",
                    label="OccFlow (full run collided)")
    axes[1].axvspan(0, 80, color="#2ecc71", alpha=0.09)
    axes[1].axvline(80, color="#ffd166", linestyle="--", linewidth=2, label="80 ms budget")
    axes[1].annotate(f"{jepa_latency:.1f} ms", (jepa_latency, auprc[1]), xytext=(8, 10), textcoords="offset points")
    axes[1].annotate(f"{occflow_latency:.1f} ms", (occflow_latency, auprc[2]), xytext=(8, 10), textcoords="offset points")
    axes[1].set(xlabel="Steady end-to-end planning P95 / ms", ylabel="Collision AUPRC",
                title="Accuracy versus closed-loop latency")
    axes[1].set_xlim(0, max(230, occflow_latency + 25)); axes[1].set_ylim(0, 0.38)
    axes[1].grid(alpha=0.18); axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("YOPO + world model Pilot: measured trade-off on RTX 5060 Laptop GPU", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
