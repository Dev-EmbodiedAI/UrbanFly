#!/usr/bin/env python
"""Build and verify one canonical Helsinki Dataset v1 directory.

The canonical episode files are NTFS hardlinks, so consolidation does not
duplicate the approximately 1 GiB active dataset.  Source HDF5 deletion is a
separate, explicitly gated mode that is allowed only after canonical QA PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "uav_wm_navigation" / "src"))

from uav_wm_navigation.data.helsinki_dataset_v1 import (  # noqa: E402
    validate_helsinki_dataset_v1,
)


SCHEMA = "urbanfly-helsinki-canonical-dataset-v1"
EPISODE_INDEX = re.compile(r"_(\d{3})_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def selected_episodes(qa_path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(qa_path.read_text(encoding="utf-8"))
    gates = report.get("gate_checks") or {}
    required = (
        "episode_count",
        "unique_contiguous_episode_ids",
        "corrupted_hdf5_zero",
        "partial_count_zero",
        "stale_action_zero",
        "cross_episode_stale_action_zero",
        "all_dataset_integrity_checks",
    )
    if report.get("status") != "PASS" or not all(gates.get(key) is True for key in required):
        raise ValueError("source replacement-aware QA is not a full PASS")
    episodes = list(report.get("episodes") or [])
    if len(episodes) != 100:
        raise ValueError("source QA must select exactly 100 episodes")
    episodes.sort(key=lambda item: int(EPISODE_INDEX.search(item["episode_id"]).group(1)))
    indices = [int(EPISODE_INDEX.search(item["episode_id"]).group(1)) for item in episodes]
    if indices != list(range(100)):
        raise ValueError("source QA episode IDs are not unique and contiguous")
    return report, episodes


def create(args: argparse.Namespace) -> None:
    qa_path = args.source_qa.resolve()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if dataset_root not in qa_path.parents:
        raise ValueError("source QA must live under dataset_root")
    if output.parent != dataset_root:
        raise ValueError("canonical output must be a direct child of dataset_root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"canonical output is not empty: {output}")
    source_report, selected = selected_episodes(qa_path)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    records = []
    task_counts: Counter[str] = Counter()
    for item in selected:
        source = Path(item["path"]).resolve()
        if dataset_root not in source.parents or not source.is_file():
            raise ValueError(f"unsafe or missing source HDF5: {source}")
        index = int(EPISODE_INDEX.search(item["episode_id"]).group(1))
        destination = episodes_dir / f"episode_{index:03d}_{item['task_type']}.h5"
        if destination.exists():
            raise FileExistsError(destination)
        os.link(source, destination)
        if not os.path.samefile(source, destination):
            raise RuntimeError(f"hardlink verification failed: {destination}")
        integrity = validate_helsinki_dataset_v1(destination)
        if integrity.get("status") != "PASS":
            raise RuntimeError(f"independent HDF5 readback failed: {destination}")
        with h5py.File(destination, "r") as handle:
            stale = int(np.asarray(handle["labels/stale_action"][:], dtype=np.uint8).sum())
            collision = int(np.asarray(handle["labels/collision"][:], dtype=np.uint8).sum())
            success = bool(np.asarray(handle["labels/success"][:], dtype=np.uint8).any())
            steps = int(len(handle["actions/commanded_body_flu"]))
        if stale or collision or not success or steps != int(item["steps"]):
            raise RuntimeError(f"canonical episode gate failed: {destination}")
        digest = sha256(destination)
        task_counts[str(item["task_type"])] += 1
        records.append(
            {
                "episode_index": index,
                "episode_id": item["episode_id"],
                "task_type": item["task_type"],
                "path": str(destination),
                "source_path": str(source),
                "size_bytes": destination.stat().st_size,
                "sha256": digest,
                "steps": steps,
                "hdf5_readback": "PASS",
            }
        )
        print(f"{index:03d}: hardlink + HDF5 readback PASS", flush=True)
    partials = list(output.rglob("*.partial"))
    task_gate = task_counts == Counter(
        {
            "building_blocked": 20,
            "street_canyon": 20,
            "rooftop_to_ground": 20,
            "ground_to_rooftop": 20,
            "rooftop_to_rooftop": 20,
        }
    )
    status = "PASS" if len(records) == 100 and task_gate and not partials else "FAIL"
    manifest = {
        "schema": SCHEMA,
        "status": status,
        "source_qa": str(qa_path),
        "storage": "NTFS hardlinks at canonicalization time",
        "episodes": records,
    }
    qa = {
        "schema": f"{SCHEMA}-qa",
        "status": status,
        "episode_count": len(records),
        "unique_contiguous_episode_ids": [item["episode_index"] for item in records] == list(range(100)),
        "transition_count": sum(item["steps"] for item in records),
        "task_counts": dict(sorted(task_counts.items())),
        "all_hdf5_readback_pass": all(item["hdf5_readback"] == "PASS" for item in records),
        "stale_action_count": 0,
        "collision_count": 0,
        "success_count": 100,
        "partial_count": len(partials),
        "source_metrics": {
            "clearance_m": source_report["clearance_m"],
            "dt_s": source_report["dt_s"],
            "reset": {
                "expected_total_episode_boundaries": source_report["reset"]["expected_total_episode_boundaries"],
                "automatic_reset_passes": source_report["reset"]["automatic_reset_passes"],
                "fresh_boundary_passes": source_report["reset"]["fresh_boundary_passes"],
            },
        },
        "content_bytes": sum(item["size_bytes"] for item in records),
        "manifest": str(output / "dataset_manifest.json"),
        "source_hdf5_cleanup": "NOT RUN",
    }
    atomic_json(output / "dataset_manifest.json", manifest)
    atomic_json(output / "dataset_qa.json", qa)
    if status != "PASS":
        raise RuntimeError("canonical dataset QA failed")
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)


def cleanup(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if output.parent != dataset_root:
        raise ValueError("canonical output must be a direct child of dataset_root")
    manifest_path = output / "dataset_manifest.json"
    qa_path = output / "dataset_qa.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or qa.get("status") != "PASS":
        raise ValueError("canonical PASS is required before cleanup")
    canonical = {Path(item["path"]).resolve(): item for item in manifest["episodes"]}
    if len(canonical) != 100:
        raise ValueError("canonical manifest must contain 100 unique files")
    for path, item in canonical.items():
        if not path.is_file() or output not in path.parents:
            raise ValueError(f"missing or unsafe canonical episode: {path}")
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"canonical hash/readback changed: {path}")
        if validate_helsinki_dataset_v1(path).get("status") != "PASS":
            raise RuntimeError(f"canonical HDF5 readback changed: {path}")
    reset = qa["source_metrics"]["reset"]
    qa["source_metrics"]["reset"] = {
        "expected_total_episode_boundaries": reset["expected_total_episode_boundaries"],
        "automatic_reset_passes": reset["automatic_reset_passes"],
        "fresh_boundary_passes": reset["fresh_boundary_passes"],
    }
    source_files = sorted(
        path.resolve()
        for path in dataset_root.rglob("*.h5")
        if output not in path.resolve().parents
    )
    for path in source_files:
        if dataset_root not in path.parents:
            raise RuntimeError(f"unsafe cleanup target: {path}")
    bytes_unlinked = sum(path.stat().st_size for path in source_files)
    for path in source_files:
        path.unlink()
    remaining_source = [
        path for path in dataset_root.rglob("*.h5") if output not in path.resolve().parents
    ]
    if remaining_source:
        raise RuntimeError("source HDF5 cleanup was incomplete")
    qa["source_hdf5_cleanup"] = "PASS"
    qa["source_hdf5_links_or_files_removed"] = int(
        qa.get("source_hdf5_links_or_files_removed", 0)
    ) + len(source_files)
    qa["source_hdf5_apparent_bytes_unlinked"] = int(
        qa.get("source_hdf5_apparent_bytes_unlinked", 0)
    ) + bytes_unlinked
    qa["post_cleanup_canonical_hdf5_count"] = len(list((output / "episodes").glob("*.h5")))
    qa["post_cleanup_partial_count"] = len(list(dataset_root.rglob("*.partial")))
    if qa["post_cleanup_canonical_hdf5_count"] != 100 or qa["post_cleanup_partial_count"] != 0:
        qa["status"] = "FAIL"
    atomic_json(qa_path, qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)
    if qa["status"] != "PASS":
        raise RuntimeError("post-cleanup canonical QA failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "cleanup-source-h5"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-qa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "create":
        if args.source_qa is None:
            parser.error("create requires --source-qa")
        create(args)
    else:
        cleanup(args)


if __name__ == "__main__":
    main()
