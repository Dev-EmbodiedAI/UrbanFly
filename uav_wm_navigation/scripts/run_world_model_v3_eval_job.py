from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_wm_navigation.evaluation.paired_benchmark import load_evaluation_manifest


METHOD_RUNTIME = {
    "yopo_direct": "yopo",
    "yopo_tdmpc2_visual": "tdmpc2_visual",
    "yopo_dreamer_rssm": "dreamer_rssm_v3",
    "yopo_vjepa2_1": "vjepa2_1_uav",
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one preregistered UrbanFly v3 evaluation job")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--shield", choices=("on", "off"), required=True)
    parser.add_argument("--checkpoint-registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-config", type=Path, default=ROOT / "configs/planner_yopo.yaml")
    parser.add_argument("--max-duration-s", type=float, default=180.0)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = load_evaluation_manifest(args.manifest)
    route = next((item for item in manifest["routes"] if item["route_id"] == args.route_id), None)
    if route is None:
        raise ValueError("route is not in the frozen evaluation manifest")
    registered_seeds = manifest["methods"].get(args.method)
    if registered_seeds is None or args.model_seed not in registered_seeds:
        raise ValueError("method/model seed pair is not preregistered")
    if args.method not in METHOD_RUNTIME:
        raise NotImplementedError(
            f"{args.method} requires its dedicated policy runner; refusing to substitute YOPO or random weights"
        )
    checkpoint = None
    if args.method != "yopo_direct":
        if args.checkpoint_registry is None:
            raise ValueError("learned methods require --checkpoint-registry")
        registry = json.loads(args.checkpoint_registry.read_text(encoding="utf-8"))
        checkpoint_value = registry.get(args.method, {}).get(str(args.model_seed))
        if not checkpoint_value:
            raise FileNotFoundError(f"no registered checkpoint for {args.method} seed {args.model_seed}")
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    base_config = yaml.safe_load((ROOT / "configs/simulator_urbanfly_websocket.yaml").read_text(encoding="utf-8"))
    base_config.update({
        "flight_start_nwu": route["start_nwu_m"], "goal_nwu": route["goal_nwu_m"],
        "route_nwu": route["route_nwu_m"], "seed": int(route["seed"]),
        "goal_tolerance_m": 3.0, "success_dwell_s": 2.0,
        "backend_safety_shield": args.shield == "on",
        "dynamic_actor_density": float(route["perturbation"]["dynamic_actor_density"]),
        "appearance_perturbation": route["perturbation"]["appearance"],
        "dynamics_perturbation": route["perturbation"]["dynamics"],
        "policy_family": args.method,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    sim_config = args.output / "resolved_simulator_config.yaml"
    sim_config.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable, str(ROOT / "scripts/run_realtime_yopo.py"),
        "--sim-config", str(sim_config), "--planner-config", str(args.planner_config),
        "--evaluation-config", str(ROOT / "configs/evaluation_world_model_v3.yaml"),
        "--method", METHOD_RUNTIME[args.method], "--seed", str(route["seed"]),
        "--max-duration-s", str(args.max_duration_s), "--output-root", str(args.output / "runtime"),
    ]
    if checkpoint is not None:
        command.extend(["--world-model-checkpoint", str(checkpoint)])
    if args.device:
        command.extend(["--device", args.device])
    job = {
        "schema": "urbanfly-evaluation-job-v3", "manifest_sha256": manifest["manifest_sha256"],
        "route_id": args.route_id, "group": route["group"], "method": args.method,
        "model_seed": args.model_seed, "shield_enabled": args.shield == "on",
        "episode_seed": int(route["seed"]), "command": subprocess.list2cmdline(command),
        "checkpoint": None if checkpoint is None else str(checkpoint),
    }
    atomic_json(args.output / "job.json", job)
    if args.dry_run:
        print(json.dumps(job, ensure_ascii=False))
        return 0
    process = subprocess.run(command, text=True, capture_output=True)
    summaries = sorted((args.output / "runtime").rglob("summary.json"), key=lambda item: item.stat().st_mtime)
    if not summaries:
        atomic_json(args.output / "failure.json", {**job, "returncode": process.returncode, "stderr": process.stderr[-4000:]})
        return process.returncode or 1
    summary = json.loads(summaries[-1].read_text(encoding="utf-8")); metrics = summary["metrics"]
    result = {
        "schema": "urbanfly-paired-result-v3", "route_id": args.route_id, "group": route["group"],
        "method": args.method, "model_seed": args.model_seed, "shield_enabled": args.shield == "on",
        "episode_seed": int(route["seed"]), "success": bool(metrics["success"]),
        "collision": bool(metrics["collision"]), "collision_count": int(bool(metrics["collision"])),
        "navigation_error_m": float(metrics["navigation_error_m"]), "spl": float(metrics["spl"]),
        "path_length_m": float(metrics["path_length_m"]), "shortest_path_m": float(metrics["shortest_path_m"]),
        "latency_p95_ms": metrics["planner_latency_p95_ms"],
        "intervention_steps": int(metrics.get("safety_intervention_steps", 0)),
        "decision_steps": int(metrics.get("planner_cycles", 0)),
        "termination_reason": metrics["termination_reason"], "runtime_summary": str(summaries[-1]),
        "runner_returncode": process.returncode,
    }
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
