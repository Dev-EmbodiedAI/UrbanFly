from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import _bootstrap  # noqa: F401
from uav_wm_navigation.evaluation import paired_bootstrap_interval


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def mcnemar_exact(left: np.ndarray, right: np.ndarray) -> float:
    left_only = int(np.sum(left & ~right)); right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    return float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue) if discordant else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired YOPO-vs-world-model tests with Holm correction.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="yopo")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    frame = frame[(frame["returncode"] == 0) & (frame["status"] == "complete")]
    keys = ["map", "scenario", "difficulty", "seed"]
    baseline = frame[frame.method == args.baseline].set_index(keys)
    comparisons, raw_p = [], []
    for method in sorted(set(frame.method) - {args.baseline}):
        treatment = frame[frame.method == method].set_index(keys)
        common = baseline.index.intersection(treatment.index)
        if not len(common):
            continue
        left, right = baseline.loc[common], treatment.loc[common]
        collision_delta = right.collision.astype(float).to_numpy() - left.collision.astype(float).to_numpy()
        success_delta = right.success.astype(float).to_numpy() - left.success.astype(float).to_numpy()
        completion_delta = right.route_completion.astype(float).to_numpy() - left.route_completion.astype(float).to_numpy()
        collision_p = mcnemar_exact(left.collision.astype(bool).to_numpy(), right.collision.astype(bool).to_numpy())
        success_p = mcnemar_exact(left.success.astype(bool).to_numpy(), right.success.astype(bool).to_numpy())
        completion_p = float(wilcoxon(completion_delta).pvalue) if np.any(completion_delta) else 1.0
        raw_p.extend([collision_p, success_p, completion_p])
        comparisons.append({
            "method": method, "pairs": len(common),
            "collision_rate_delta": float(collision_delta.mean()),
            "collision_delta_ci95": paired_bootstrap_interval(collision_delta),
            "success_rate_delta": float(success_delta.mean()),
            "route_completion_delta": float(completion_delta.mean()),
            "p_collision_mcnemar": collision_p, "p_success_mcnemar": success_p,
            "p_completion_wilcoxon": completion_p,
        })
    adjusted = holm_adjust(raw_p)
    offset = 0
    for row in comparisons:
        row["p_collision_holm"], row["p_success_holm"], row["p_completion_holm"] = adjusted[offset:offset + 3]
        offset += 3
    payload = {"baseline": args.baseline, "paired_keys": keys, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
