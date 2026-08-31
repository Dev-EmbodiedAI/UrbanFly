"""跨环境数字孪生闭环结果的独立一致性审计。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EXPECTED_SWARM_ENVIRONMENTS = (
    "swarm:city",
    "swarm:open",
    "swarm:mountain",
    "swarm:village",
    "swarm:forest",
)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cross_environment_reports(
    report_paths: Sequence[str | Path],
    *,
    expected_environments: Iterable[str] = EXPECTED_SWARM_ENVIRONMENTS,
    policy_sources: Sequence[str | Path] = (),
    upstream_commit: str | None = None,
) -> dict:
    """Fail closed unless all reports are comparable, causal, safe successes."""

    paths = tuple(Path(path).resolve() for path in report_paths)
    expected = tuple(expected_environments)
    if len(paths) != len(expected):
        raise ValueError(f"需要 {len(expected)} 份环境报告，实际 {len(paths)} 份")

    reports: list[dict] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") != "urbanfly-cross-environment-digital-twin-navigation-v1":
            raise ValueError(f"{path.name}: schema 不匹配")
        reports.append(report)

    environments = tuple(str(report.get("environment")) for report in reports)
    if len(set(environments)) != len(environments) or set(environments) != set(expected):
        raise ValueError(f"环境必须唯一且完整: {environments}")

    seeds = {int(report["seed"]) for report in reports}
    drone_counts = {int(report["requested_drones"]) for report in reports}
    contracts = {str(report["control_contract"]) for report in reports}
    if len(seeds) != 1 or len(drone_counts) != 1 or len(contracts) != 1:
        raise ValueError("seed、无人机数量和 control contract 必须一致")

    per_environment: dict[str, dict] = {}
    total_drones = 0
    total_successes = 0
    total_collisions = 0
    total_steps = 0
    total_decisions = 0
    for path, report in zip(paths, reports):
        environment = str(report["environment"])
        successes = tuple(bool(item) for item in report.get("per_drone_success", ()))
        collisions = tuple(bool(item) for item in report.get("per_drone_collision", ()))
        reasons = tuple(str(item) for item in report.get("per_drone_failure_reason", ()))
        drone_count = int(report["requested_drones"])
        if not (len(successes) == len(collisions) == len(reasons) == drone_count):
            raise ValueError(f"{path.name}: per-drone 长度不一致")
        steps = int(report.get("steps", -1))
        decisions = int(report.get("world_model_decisions", -1))
        executions = int(report.get("executions", -1))
        feedbacks = int(report.get("fresh_feedbacks", -1))
        observations = int(report.get("agent_observations", -1))
        causal_counts_pass = (
            steps > 0
            and decisions == executions == feedbacks == steps
            and observations == feedbacks + 1
            and bool(report.get("causal_chain_complete"))
        )
        clearances = np.asarray(report.get("minimum_predicted_clearance_m", ()), dtype=float)
        separations = np.asarray(report.get("minimum_predicted_separation_m", ()), dtype=float)
        native_score = report.get("native_score") or {}
        final_score = float(native_score.get("final_score", float("nan")))
        gates = {
            "status_pass": report.get("status") == "PASS",
            "all_drones_success": bool(report.get("success")) and all(successes),
            "zero_collision": not any(collisions),
            "no_failure_reason": all(reason == "NONE" for reason in reasons),
            "causal_counts_pass": causal_counts_pass,
            "exact_goal_mode_not_benchmark": report.get("benchmark_eligible") is False,
            "clearance_finite_nonnegative": (
                clearances.shape == (drone_count,)
                and bool(np.isfinite(clearances).all())
                and bool((clearances >= 0.0).all())
            ),
            "separation_finite_positive": (
                separations.shape == (drone_count,)
                and bool(np.isfinite(separations).all())
                and bool((separations > 0.0).all())
            ),
            "native_score_valid": bool(np.isfinite(final_score) and 0.0 <= final_score <= 1.0),
        }
        if not all(gates.values()):
            failed = [name for name, passed in gates.items() if not passed]
            raise ValueError(f"{path.name}: QA gate 失败: {failed}")
        total_drones += drone_count
        total_successes += sum(successes)
        total_collisions += sum(collisions)
        total_steps += steps
        total_decisions += decisions
        per_environment[environment] = {
            "report": str(path),
            "report_sha256": sha256_file(path),
            "steps": steps,
            "successes": sum(successes),
            "drones": drone_count,
            "collisions": sum(collisions),
            "native_score": final_score,
            "minimum_predicted_clearance_m": float(np.min(clearances)),
            "minimum_predicted_separation_m": float(np.min(separations)),
            "gates": gates,
        }

    source_hashes = {}
    newest_source_mtime_ns = 0
    for source in (Path(item).resolve() for item in policy_sources):
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hashes[str(source)] = sha256_file(source)
        newest_source_mtime_ns = max(newest_source_mtime_ns, source.stat().st_mtime_ns)
    reports_newer_than_sources = all(
        path.stat().st_mtime_ns >= newest_source_mtime_ns for path in paths
    )
    if source_hashes and not reports_newer_than_sources:
        raise ValueError("至少一份报告早于最终 policy source，不能证明同版本回归")

    ordered = {environment: per_environment[environment] for environment in expected}
    return {
        "schema": "urbanfly-cross-environment-digital-twin-qa-v1",
        "status": "PASS",
        "scope": "Swarm exact-goal digital-twin navigation; one fixed seed per environment",
        "environments": list(expected),
        "seed": next(iter(seeds)),
        "drones_per_environment": next(iter(drone_counts)),
        "control_contract": next(iter(contracts)),
        "episodes": len(reports),
        "total_drones": total_drones,
        "successful_drones": total_successes,
        "collisions": total_collisions,
        "total_steps": total_steps,
        "world_model_decisions": total_decisions,
        "all_causal_chains_complete": True,
        "reports_newer_than_policy_sources": reports_newer_than_sources,
        "policy_source_sha256": source_hashes,
        "upstream_commit": upstream_commit,
        "per_environment": ordered,
        "limitations": [
            "exact goals are exposed to the high-level Agent, so these runs are not formal Swarm benchmark entries",
            "the Swarm World Model is an analytic one-step predictive baseline, not the learned Helsinki latent checkpoint",
            "one fixed seed and two UAVs per environment do not establish statistical generalization",
        ],
    }
