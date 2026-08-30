from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .metrics import navigation_error, paired_bootstrap_interval, success_weighted_path_length, wilson_interval


SCHEMA = "urbanfly-paired-evaluation-v3"
RESULT_SCHEMA = "urbanfly-paired-result-v3"


def _canonical_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    if payload.get("manifest_sha256") != _canonical_hash(payload):
        raise ValueError("evaluation manifest SHA-256 mismatch")
    if payload.get("route_count") != 120 or len(payload.get("routes", [])) != 120:
        raise ValueError("formal quick evaluation must contain exactly 120 routes")
    groups = defaultdict(int)
    for route in payload["routes"]:
        groups[route["group"]] += 1
    if set(groups.values()) != {24} or len(groups) != 5:
        raise ValueError("formal quick evaluation must contain five groups of 24")
    return payload


def expected_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for route in manifest["routes"]:
        for method, seeds in manifest["methods"].items():
            for model_seed in seeds:
                for shield_enabled in manifest["shield_modes"]:
                    jobs.append({
                        "route_id": route["route_id"], "group": route["group"],
                        "method": method, "model_seed": model_seed,
                        "shield_enabled": bool(shield_enabled), "episode_seed": int(route["seed"]),
                    })
    return jobs


def _job_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (item["route_id"], item["method"], item.get("model_seed"), bool(item["shield_enabled"]))


def normalize_result(record: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    if result.get("schema") not in (None, RESULT_SCHEMA):
        raise ValueError(f"invalid result schema for {result.get('route_id')}")
    result["schema"] = RESULT_SCHEMA
    success = bool(result.get("success", False))
    collision = bool(result.get("collision", False)) or int(result.get("collision_count", 0)) > 0
    if success and collision:
        raise ValueError("success cannot be true for a collided episode")
    if "navigation_error_m" not in result:
        result["navigation_error_m"] = navigation_error(
            np.asarray(result["final_position_nwu_m"]), np.asarray(route["goal_nwu_m"])
        )
    if "spl" not in result:
        result["spl"] = success_weighted_path_length(
            success, float(result["path_length_m"]), float(route["shortest_path_m"])
        )
    result["success"] = success
    result["collision"] = collision
    result["shortest_path_m"] = float(route["shortest_path_m"])
    result["group"] = route["group"]
    for field in ("navigation_error_m", "spl", "path_length_m"):
        if not np.isfinite(float(result[field])):
            raise ValueError(f"non-finite {field} in {result['route_id']}")
    if not 0.0 <= float(result["spl"]) <= 1.0 + 1e-6:
        raise ValueError("SPL must lie in [0, 1]")
    return result


def validate_results(
    records: Iterable[dict[str, Any]], manifest: dict[str, Any], *, allow_incomplete: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routes = {item["route_id"]: item for item in manifest["routes"]}
    expected = {_job_key(item): item for item in expected_jobs(manifest)}
    normalized, seen = [], set()
    for raw in records:
        key = _job_key(raw)
        if key not in expected:
            raise ValueError(f"result is not preregistered: {key}")
        if key in seen:
            raise ValueError(f"duplicate evaluation result: {key}")
        if int(raw.get("episode_seed", -1)) != expected[key]["episode_seed"]:
            raise ValueError(f"episode seed mismatch: {key}")
        normalized.append(normalize_result(raw, routes[raw["route_id"]]))
        seen.add(key)
    missing = [job for key, job in expected.items() if key not in seen]
    if missing and not allow_incomplete:
        raise RuntimeError(f"formal result is incomplete: {len(missing)} of {len(expected)} jobs are missing")
    return normalized, missing


def _bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    return paired_bootstrap_interval(np.asarray(values, dtype=np.float64), seed=seed)


def summarize_results(records: list[dict[str, Any]], *, seed: int = 20260831) -> list[dict[str, Any]]:
    groups: dict[tuple[str, Any, bool, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        groups[(item["method"], item.get("model_seed"), bool(item["shield_enabled"]), item["group"])].append(item)
    output = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(str(value) for value in pair[0])):
        method, model_seed, shield, group = key
        successes = np.asarray([bool(item["success"]) for item in items])
        ne = np.asarray([float(item["navigation_error_m"]) for item in items])
        spl = np.asarray([float(item["spl"]) for item in items])
        sr, sr_low, sr_high = wilson_interval(int(successes.sum()), len(items))
        ne_mean, ne_low, ne_high = _bootstrap_mean(ne, seed)
        spl_mean, spl_low, spl_high = _bootstrap_mean(spl, seed + 1)
        output.append({
            "method": method, "model_seed": model_seed, "shield_enabled": shield, "group": group,
            "episodes": len(items), "successes": int(successes.sum()),
            "sr": sr, "sr_wilson_95": [sr_low, sr_high],
            "ne_m": ne_mean, "ne_bootstrap_95": [ne_low, ne_high],
            "spl": spl_mean, "spl_bootstrap_95": [spl_low, spl_high],
            "collisions": int(sum(bool(item["collision"]) for item in items)),
            "intervention_rate": sum(int(item.get("intervention_steps", 0)) for item in items) / max(1, sum(int(item.get("decision_steps", 0)) for item in items)),
            "latency_p95_ms": float(np.percentile([float(item.get("latency_p95_ms", item.get("latency_ms", 0.0))) for item in items], 95)),
        })
    return output


def paired_deltas(records: list[dict[str, Any]], *, baseline: str = "yopo_direct") -> list[dict[str, Any]]:
    by_context: dict[tuple[str, bool, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in records:
        context = (item["route_id"], bool(item["shield_enabled"]), item["group"])
        seed_label = "none" if item.get("model_seed") is None else str(item["model_seed"])
        by_context[context][f"{item['method']}:{seed_label}"] = item
    output = []
    labels = sorted({label for values in by_context.values() for label in values if not label.startswith(f"{baseline}:")})
    for label in labels:
        for shield in (False, True):
            diffs = {"sr": [], "ne_m": [], "spl": []}
            for (_, context_shield, _), values in by_context.items():
                if context_shield != shield or label not in values or f"{baseline}:none" not in values:
                    continue
                model, direct = values[label], values[f"{baseline}:none"]
                diffs["sr"].append(float(model["success"]) - float(direct["success"]))
                diffs["ne_m"].append(float(model["navigation_error_m"]) - float(direct["navigation_error_m"]))
                diffs["spl"].append(float(model["spl"]) - float(direct["spl"]))
            if diffs["sr"]:
                output.append({
                    "method_seed": label, "baseline": baseline, "shield_enabled": shield, "paired_routes": len(diffs["sr"]),
                    **{f"delta_{metric}": paired_bootstrap_interval(np.asarray(values), seed=20260831 + index) for index, (metric, values) in enumerate(diffs.items())},
                })
    return output
