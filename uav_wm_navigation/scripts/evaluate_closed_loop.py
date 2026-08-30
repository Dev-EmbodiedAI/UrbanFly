from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.evaluation import (
    aggregate_navigation_metrics,
    paired_bootstrap_interval,
    wilson_interval,
)
from uav_wm_navigation.utils.config import load_yaml


def latest_summary(root: Path) -> dict:
    candidates = sorted(root.rglob("summary.json"), key=lambda path: path.stat().st_mtime)
    return json.loads(candidates[-1].read_text(encoding="utf-8")) if candidates else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the paired 4-method × map × scenario × difficulty × seed matrix.")
    parser.add_argument("--evaluation-config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/evaluation.yaml")
    parser.add_argument("--sim-config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/simulator_mock.yaml")
    parser.add_argument("--planner-config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/planner_yopo.yaml")
    parser.add_argument("--dreamerv3-checkpoint", type=Path)
    parser.add_argument("--jepa-checkpoint", type=Path)
    parser.add_argument("--tdmpc2-checkpoint", type=Path)
    parser.add_argument("--occflow-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--maps", nargs="*")
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--difficulties", nargs="*")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--steps",
        type=int,
        help="Override the per-episode control-step budget for smoke validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluation = load_yaml(args.evaluation_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = args.methods or evaluation["methods"]
    maps = args.maps or evaluation["maps"]
    scenarios = args.scenarios or evaluation["scenarios"]
    difficulties = args.difficulties or evaluation["difficulties"]
    seeds = args.seeds or evaluation["seeds"]
    checkpoints = {
        "yopo_dreamerv3": args.dreamerv3_checkpoint,
        "yopo_jepa": args.jepa_checkpoint,
        "yopo_tdmpc2": args.tdmpc2_checkpoint,
        "yopo_occflow": args.occflow_checkpoint,
    }
    matrix = list(product(methods, maps, scenarios, difficulties, seeds))
    if args.max_runs is not None:
        matrix = matrix[: args.max_runs]
    manifest = {
        "methods": methods, "maps": maps, "scenarios": scenarios, "difficulties": difficulties,
        "seeds": seeds, "planned_runs": len(matrix), "paired_key": ["map", "scenario", "difficulty", "seed"],
        "note": "Run each UrbanFly map as a separate paired batch.",
    }
    (args.output_dir / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for run_index, (method, map_name, scenario, difficulty, seed) in enumerate(matrix):
        checkpoint = checkpoints.get(method)
        run_root = args.output_dir / f"{run_index:04d}_{method}_{map_name}_{scenario}_{difficulty}_seed{seed}"
        command = [
            sys.executable, str(_bootstrap.PROJECT_ROOT / "scripts/run_yopo_baseline.py"),
            "--sim-config", str(args.sim_config), "--planner-config", str(args.planner_config),
            "--evaluation-config", str(args.evaluation_config), "--steps", str(
                args.steps if args.steps is not None else evaluation.get("max_steps", 120)
            ),
            "--output-root", str(run_root), "--method", method, "--map", map_name,
            "--scenario", scenario, "--difficulty", difficulty, "--seed", str(seed),
        ]
        if method in checkpoints:
            if checkpoint is None:
                rows.append({"method": method, "map": map_name, "scenario": scenario, "difficulty": difficulty,
                             "seed": seed, "returncode": 2, "status": "missing_checkpoint"})
                continue
            command.extend(["--world-model-checkpoint", str(checkpoint)])
        if args.dry_run:
            rows.append({"method": method, "map": map_name, "scenario": scenario, "difficulty": difficulty,
                         "seed": seed, "returncode": 0, "status": "planned", "command": subprocess.list2cmdline(command)})
            continue
        process = subprocess.run(command, text=True, capture_output=True)
        summary = latest_summary(run_root)
        rows.append({
            "method": method, "map": map_name, "scenario": scenario, "difficulty": difficulty, "seed": seed,
            "returncode": process.returncode, "status": "complete" if process.returncode == 0 else "failed",
            "success": summary.get("success"), "collision": summary.get("collision"),
            "sr": summary.get("sr"),
            "navigation_error_m": summary.get("navigation_error_m"),
            "ne_m": summary.get("ne_m"),
            "path_length_m": summary.get("path_length_m"),
            "shortest_path_length_m": summary.get("shortest_path_length_m"),
            "spl": summary.get("spl"),
            "route_completion": summary.get("route_completion"),
            "planning_latency_p50_ms": summary.get("planning_latency_p50_ms"),
            "planning_latency_p95_ms": summary.get("planning_latency_p95_ms"),
            "steady_state_latency_p95_ms": summary.get("steady_state_latency_p95_ms"),
            "fallback_rate": summary.get("fallback_rate"), "rerank_rate": summary.get("rerank_rate"),
            "summary_path": str(next(iter(sorted(run_root.rglob("summary.json"))), "")),
            "stderr": process.stderr[-1000:],
        })
        print(json.dumps(rows[-1], ensure_ascii=False))
    json_path = args.output_dir / "closed_loop_results.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output_dir / "closed_loop_results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if (
            row.get("status") == "complete"
            and row.get("success") is not None
            and row.get("navigation_error_m") is not None
            and row.get("spl") is not None
        ):
            grouped[str(row["method"])].append(row)
    main_table = []
    for method in methods:
        episodes = grouped.get(method, [])
        if not episodes:
            main_table.append(
                {
                    "method": method,
                    "episodes": 0,
                    "status": "no_completed_episodes",
                }
            )
            continue
        successes = np.asarray([bool(row["success"]) for row in episodes])
        errors = np.asarray(
            [float(row["navigation_error_m"]) for row in episodes],
            dtype=np.float64,
        )
        spl_values = np.asarray(
            [float(row["spl"]) for row in episodes],
            dtype=np.float64,
        )
        aggregate = aggregate_navigation_metrics(successes, errors, spl_values)
        sr, sr_low, sr_high = wilson_interval(
            int(successes.sum()),
            len(successes),
        )
        ne, ne_low, ne_high = paired_bootstrap_interval(
            errors,
            seed=int(evaluation.get("bootstrap_seed", 20260731)),
        )
        spl_mean, spl_low, spl_high = paired_bootstrap_interval(
            spl_values,
            seed=int(evaluation.get("bootstrap_seed", 20260731)) + 1,
        )
        main_table.append(
            {
                "method": method,
                **aggregate,
                "status": "complete",
                "sr": sr,
                "sr_ci95": [sr_low, sr_high],
                "ne_m": ne,
                "ne_m_ci95": [ne_low, ne_high],
                "spl": spl_mean,
                "spl_ci95": [spl_low, spl_high],
            }
        )
    (args.output_dir / "navigation_main_table.json").write_text(
        json.dumps(main_table, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    table_fields = [
        "method",
        "status",
        "episodes",
        "successes",
        "sr",
        "sr_ci95",
        "ne_m",
        "ne_m_ci95",
        "spl",
        "spl_ci95",
    ]
    with (args.output_dir / "navigation_main_table.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(main_table)
    print(json.dumps({"navigation_main_table": main_table}, ensure_ascii=False))
    return 0 if args.dry_run or all(row["returncode"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
