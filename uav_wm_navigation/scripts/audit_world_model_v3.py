from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from uav_wm_navigation.data.splits import load_split_manifest


PRIVILEGED_KEYS = {
    "tile_id", "zone_type", "dynamic_actor_states", "cpa_risk_map",
    "static_esdf_m", "teacher_action_normalized", "candidate_outcomes",
    "appearance_parameters", "dynamics_parameters", "counterfactual_parent_id",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit UrbanFly world-model-v3 quality and leakage gates")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--split-manifest", type=Path, default=_bootstrap.PROJECT_ROOT / "configs/urbanfly_spatial_split_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-every", type=int, default=1)
    args = parser.parse_args()
    split_manifest = load_split_manifest(args.split_manifest)
    tile_splits = {tile["id"]: tile["split"] for tile in split_manifest["tiles"]}
    failures: list[str] = []
    skews, frames, damaged = [], 0, 0
    route_splits: dict[str, set[str]] = defaultdict(set)
    parent_splits: dict[str, set[str]] = defaultdict(set)
    last_step: dict[str, int] = {}
    last_time: dict[str, float] = {}
    previous_actors: dict[str, tuple[float, dict[int, dict]]] = {}
    actor_errors: list[float] = []
    formats: set[str] = set()
    for manifest_value in args.manifests:
        manifest_path = manifest_value.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "urbanfly-world-model-v3" or not manifest.get("complete"):
            failures.append(f"{manifest_path.name}: incomplete or wrong schema")
            continue
        if not manifest.get("policy_inputs_exclude_privileged"):
            failures.append(f"{manifest_path.name}: privileged input exclusion not declared")
        for descriptor in manifest.get("shards", []):
            shard = manifest_path.parent / descriptor["name"]
            state = manifest_path.parent / descriptor["state_table"]
            formats.add(str(descriptor["state_format"]))
            for path, expected in ((shard, descriptor["sha256"]), (state, descriptor["state_sha256"])):
                if not path.exists():
                    failures.append(f"missing {path}")
                elif sha256(path) != expected:
                    failures.append(f"checksum mismatch {path.name}")
            if not shard.exists():
                continue
            with tarfile.open(shard) as archive:
                members = {member.name: member for member in archive if member.isfile()}
                public_names = sorted(name for name in members if name.endswith(".json") and not name.endswith(".privileged.json"))
                if len(public_names) != int(descriptor["samples"]):
                    failures.append(f"{shard.name}: descriptor/sample count mismatch")
                for public_name in public_names:
                    key = public_name[:-5]
                    try:
                        extracted = archive.extractfile(members[public_name]); assert extracted is not None
                        public = json.load(extracted)
                        if PRIVILEGED_KEYS & public.keys():
                            failures.append(f"{key}: privileged key leaked into public JSON")
                        episode, step = str(public["episode_id"]), int(public["step_id"])
                        expected_step = last_step.get(episode, -1) + 1
                        if step != expected_step:
                            failures.append(f"{episode}: expected step {expected_step}, got {step}")
                        sim_time = float(public["sim_time"])
                        if sim_time <= last_time.get(episode, -np.inf):
                            failures.append(f"{episode}: non-increasing sim time")
                        last_step[episode], last_time[episode] = step, sim_time
                        skew = abs(float(public["sensor_time"]) - float(public["state_time"])) * 1000.0
                        skews.append(skew)
                        route_splits[str(public["route_id"])].add(str(public["split"]))
                        private_name = f"{key}.privileged.json"
                        if private_name in members:
                            extracted = archive.extractfile(members[private_name]); assert extracted is not None
                            private = json.load(extracted)
                            tile_id, split = str(private.get("tile_id", "")), str(public["split"])
                            if split != "calibration" and tile_splits.get(tile_id) != split:
                                failures.append(f"{key}: tile {tile_id} belongs to {tile_splits.get(tile_id)}, not {split}")
                            parent = private.get("counterfactual_parent_id")
                            if parent: parent_splits[str(parent)].add(split)
                            actors = {int(item["actor_id"]): item for item in private.get("dynamic_actor_states", [])}
                            previous = previous_actors.get(episode)
                            if previous is not None:
                                previous_time, old = previous
                                dt = sim_time - previous_time
                                for actor_id in old.keys() & actors.keys():
                                    old_actor, new_actor = old[actor_id], actors[actor_id]
                                    old_velocity = np.asarray(old_actor["velocity"], dtype=float)
                                    new_velocity = np.asarray(new_actor["velocity"], dtype=float)
                                    if np.linalg.norm(old_velocity - new_velocity) <= 0.2:
                                        projected = np.asarray(old_actor["position"], dtype=float) + old_velocity * dt
                                        actor_errors.append(float(np.linalg.norm(projected - np.asarray(new_actor["position"], dtype=float))))
                            previous_actors[episode] = (sim_time, actors)
                        if frames % max(args.verify_every, 1) == 0:
                            for suffix, mode in (("rgb.jpg", cv2.IMREAD_COLOR), ("depth.png", cv2.IMREAD_UNCHANGED), ("depth_valid.png", cv2.IMREAD_GRAYSCALE)):
                                member = members.get(f"{key}.{suffix}")
                                if member is None:
                                    raise ValueError(f"missing {suffix}")
                                extracted = archive.extractfile(member); assert extracted is not None
                                if cv2.imdecode(np.frombuffer(extracted.read(), np.uint8), mode) is None:
                                    raise ValueError(f"cannot decode {suffix}")
                    except Exception as error:
                        damaged += 1
                        failures.append(f"{shard.name}/{key}: {error}")
                    frames += 1
    for route_id, splits in route_splits.items():
        if len(splits) > 1:
            failures.append(f"route leakage {route_id}: {sorted(splits)}")
    for parent_id, splits in parent_splits.items():
        if len(splits) > 1:
            failures.append(f"counterfactual parent leakage {parent_id}: {sorted(splits)}")
    partials = sorted({str(path) for manifest in args.manifests for path in manifest.resolve().parent.glob("*.partial")})
    if partials:
        failures.append(f"unfinished partial files: {len(partials)}")
    corrupt_rate = damaged / max(frames, 1)
    skew_max = max(skews, default=float("inf"))
    if corrupt_rate >= 0.001:
        failures.append(f"corrupt frame rate {corrupt_rate:.6f} is not below 0.1%")
    if skew_max > 110.0:
        failures.append(f"maximum sensor/state skew {skew_max:.3f} ms exceeds 110 ms")
    report = {
        "status": "pass" if not failures else "fail",
        "schema": "urbanfly-world-model-v3-audit-v1",
        "frames": frames, "episodes": len(last_step), "routes": len(route_splits),
        "corrupt_frames": damaged, "corrupt_rate": corrupt_rate,
        "sensor_state_skew_max_ms": skew_max,
        "sensor_state_skew_p95_ms": float(np.percentile(skews, 95)) if skews else None,
        "actor_replay_error_p95_m": float(np.percentile(actor_errors, 95)) if actor_errors else None,
        "state_formats": sorted(formats), "partial_files": partials,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
