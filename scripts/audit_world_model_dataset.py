#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbanfly_vln.world_model_data import samples_from_run  # noqa: E402


def audit_run(run_dir: Path, risk_horizon: int, near_miss_depth_m: float) -> dict[str, object]:
    samples = samples_from_run(
        run_dir,
        language_dimensions=0,
        risk_horizon=risk_horizon,
        near_miss_depth_m=near_miss_depth_m,
    )
    p05 = np.asarray([sample.features[2] for sample in samples], dtype=np.float64)
    risk = np.asarray([sample.risk for sample in samples], dtype=np.float64)
    prevalence = float(np.mean(risk))
    warnings = []
    if not samples[0].instruction:
        warnings.append("missing_instruction")
    if prevalence > 0.8:
        warnings.append("risk_prevalence_above_80_percent_check_camera_mask")
    if prevalence == 0.0:
        warnings.append("no_positive_risk_windows")
    if prevalence == 1.0:
        warnings.append("no_negative_risk_windows")
    if float(np.median(p05)) < near_miss_depth_m:
        warnings.append("median_depth_below_near_miss_threshold")
    return {
        "source": run_dir.resolve().name,
        "path": str(run_dir.resolve()),
        "scene": samples[0].scene_id,
        "samples": len(samples),
        "risk_positives": int(np.sum(risk)),
        "risk_prevalence": prevalence,
        "p05_depth_m": {
            "minimum": float(np.min(p05)),
            "q05": float(np.quantile(p05, 0.05)),
            "median": float(np.median(p05)),
            "q95": float(np.quantile(p05, 0.95)),
        },
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UrbanFly rollout suitability for world-model training.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--risk-horizon", type=int, default=3)
    parser.add_argument("--near-miss-depth-m", type=float, default=5.0)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    runs = [audit_run(path, args.risk_horizon, args.near_miss_depth_m) for path in args.run_dir]
    report = {
        "format": "urbanfly-world-model-data-audit-0.2",
        "risk_horizon": args.risk_horizon,
        "near_miss_depth_m": args.near_miss_depth_m,
        "runs": runs,
        "warning_count": sum(len(run["warnings"]) for run in runs),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(text, encoding="utf-8")
    print(text)
    if args.fail_on_warning and report["warning_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
