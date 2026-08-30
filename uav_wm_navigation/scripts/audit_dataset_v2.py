from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.data import validate_episode
from uav_wm_navigation.types import LabelSource
from uav_wm_navigation.utils.config import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply formal Pilot quality gates to HDF5 v2 episodes.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/data_collection_formal.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = load_yaml(args.config); gates = config["quality_gates"]
    episodes = sorted(args.dataset_dir.glob("*.h5")); failures = []; selected = Counter(); candidates = 0
    dangerous = depth_valid = depth_total = body_contamination = body_total = 0
    skews, planners, split_corridors, dynamic_confidences, dynamic_replay_matches = [], Counter(), {}, [], []
    for path in episodes:
        try: validate_episode(path)
        except Exception as error: failures.append(f"{path.name}: {error}"); continue
        metadata_path = path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        planners[metadata.get("planner", "missing")] += 1
        split = metadata.get("split"); corridor = metadata.get("corridor_id")
        if split and corridor: split_corridors.setdefault(split, set()).add(corridor)
        with h5py.File(path, "r") as handle:
            count = handle["candidates/positions"].shape[1]
            if count != int(config["required_candidate_count"]): failures.append(f"{path.name}: candidate_count={count}")
            choice = handle["selected_index"][:]
            selected.update(map(int, choice)); candidates += len(choice) * count
            dangerous += int(handle["labels/candidate_collision"][:].sum())
            valid = handle["depth_valid_mask"][:].astype(bool); depth_valid += int(valid.sum()); depth_total += valid.size
            depth = handle["depth_m"][:]
            # A close road user/tree is valid data, not drone-body leakage.
            # Body contamination must be camera-fixed and persistent.  Count
            # pixels below 0.3 m in >=80% of an episode, restricted to the
            # lower image where a rotor/landing gear could physically appear.
            lower = depth[:, int(depth.shape[1] * 0.7):]
            persistent_body = (np.isfinite(lower) & (lower < 0.3)).mean(axis=0) >= 0.8
            body_contamination += int(persistent_body.sum()); body_total += persistent_body.size
            skews.extend(np.abs(handle["timestamps/sensor"][:] - handle["timestamps/state"][:]) * 1000.0)
            sources = handle["labels/candidate_source"][:]
            confidence = handle["labels/candidate_confidence"][:]
            dynamic = np.isin(sources, [int(LabelSource.SCRIPTED_ACTOR), int(LabelSource.ACTOR_CONSTANT_VELOCITY)])
            dynamic_confidences.extend(confidence[dynamic].tolist())
            if "actors/id" in handle and handle["actors/id"].shape[0] > 1:
                actor_ids = handle["actors/id"][:]
                actor_positions = handle["actors/position"][:]
                actor_velocities = handle["actors/velocity"][:]
                actor_valid = handle["actors/valid_mask"][:].astype(bool)
                scripted = handle["actors/scripted"][:].astype(bool)
                actor_times = handle["timestamp"][:]
                for time_index in range(len(actor_times) - 1):
                    dt = float(actor_times[time_index + 1] - actor_times[time_index])
                    next_lookup = {
                        int(actor_ids[time_index + 1, item]): actor_positions[time_index + 1, item]
                        for item in np.flatnonzero(actor_valid[time_index + 1])
                    }
                    next_velocity_lookup = {
                        int(actor_ids[time_index + 1, item]): actor_velocities[time_index + 1, item]
                        for item in np.flatnonzero(actor_valid[time_index + 1])
                    }
                    for actor_index in np.flatnonzero(actor_valid[time_index] & scripted[time_index]):
                        actor_id = int(actor_ids[time_index, actor_index])
                        if actor_id not in next_lookup:
                            continue
                        # Exclude the single sample where a scripted actor
                        # starts/stops; constant-velocity projection is only a
                        # meaningful check inside a motion segment.
                        if np.linalg.norm(actor_velocities[time_index, actor_index] - next_velocity_lookup[actor_id]) > 0.2:
                            continue
                        projected = actor_positions[time_index, actor_index] + actor_velocities[time_index, actor_index] * dt
                        dynamic_replay_matches.append(float(np.linalg.norm(projected - next_lookup[actor_id])) <= 0.5)
    split_manifest_path = args.dataset_dir / "splits.json"
    if split_manifest_path.exists():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        manifest_corridors: dict[str, set[str]] = {}
        for split_name in ("train", "validation", "test"):
            corridors = set()
            for episode_path in split_manifest.get(split_name, []):
                metadata_path = Path(episode_path).with_suffix(".metadata.json")
                if metadata_path.exists():
                    corridor = json.loads(metadata_path.read_text(encoding="utf-8")).get("corridor_id")
                    if corridor:
                        corridors.add(str(corridor))
            manifest_corridors[split_name] = corridors
        for left_index, left_name in enumerate(("train", "validation", "test")):
            for right_name in ("train", "validation", "test")[left_index + 1:]:
                overlap = manifest_corridors[left_name] & manifest_corridors[right_name]
                if overlap:
                    failures.append(f"split manifest corridor leakage {left_name}/{right_name}: {sorted(overlap)[:5]}")
    total_selected = sum(selected.values()); selected_fraction = max(selected.values(), default=0) / max(total_selected, 1)
    metrics = {
        "episodes": len(episodes), "planner_counts": dict(planners),
        "depth_valid_rate": depth_valid / max(depth_total, 1),
        "body_mask_contamination": body_contamination / max(body_total, 1),
        "sensor_state_skew_p95_ms": float(np.percentile(skews, 95)) if skews else float("inf"),
        "dangerous_candidate_fraction": dangerous / max(candidates, 1),
        "max_selected_lattice_fraction": selected_fraction, "selected_histogram": dict(selected),
        "dynamic_label_mean_confidence": float(np.mean(dynamic_confidences)) if dynamic_confidences else None,
        "dynamic_replay_agreement": float(np.mean(dynamic_replay_matches)) if dynamic_replay_matches else None,
        "partial_files": len(list(args.dataset_dir.glob("*.partial"))),
        "split_manifest_checked": split_manifest_path.exists(),
    }
    if config.get("require_real_yopo") and set(planners) != {"YOPOAdapter"}: failures.append("not every episode used YOPOAdapter")
    if metrics["depth_valid_rate"] < gates["depth_valid_rate_min"]: failures.append("depth_valid_rate below gate")
    if metrics["body_mask_contamination"] > gates["body_mask_contamination_max"]: failures.append("body contamination above gate")
    if metrics["sensor_state_skew_p95_ms"] > gates["sensor_state_skew_p95_ms_max"]: failures.append("sensor/state skew above gate")
    if selected_fraction > gates["max_selected_lattice_fraction"]: failures.append("selected lattice distribution is collapsed")
    replay_agreement = metrics["dynamic_replay_agreement"]
    if replay_agreement is None or replay_agreement < gates["dynamic_replay_agreement_min"]:
        failures.append("dynamic actor replay agreement missing or below gate")
    low, high = gates["dangerous_candidate_fraction"]
    if not low <= metrics["dangerous_candidate_fraction"] <= high: failures.append("dangerous candidate fraction outside gate")
    split_names = sorted(split_corridors)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = split_corridors[left] & split_corridors[right]
            if overlap: failures.append(f"corridor leakage {left}/{right}: {sorted(overlap)[:5]}")
    if metrics["partial_files"]: failures.append("partial episode files remain")
    report = {"status": "pass" if not failures else "fail", "metrics": metrics, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
