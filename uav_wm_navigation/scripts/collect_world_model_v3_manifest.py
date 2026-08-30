from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect every route in a frozen UrbanFly v3 manifest, with crash recovery")
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--max-routes", type=int)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    route_manifest = json.loads(args.route_manifest.read_text(encoding="utf-8"))
    routes = route_manifest["routes"][: args.max_routes]
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "collection_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if args.resume and status_path.exists() else {
        "schema": "urbanfly-v3-collection-status", "route_manifest": str(args.route_manifest.resolve()),
        "route_manifest_sha256": route_manifest["manifest_sha256"], "complete": [], "failed": [],
    }
    completed_ids = {item["route_id"] for item in status["complete"]}
    for ordinal, route in enumerate(routes, 1):
        if route["route_id"] in completed_ids:
            continue
        episode_id = f"collect-{route['route_id']}-{route['seed']}"
        episode_dir = args.output / route["route_id"]
        command = [
            sys.executable, str(ROOT / "scripts/collect_urbanfly_world_model_v3_live.py"),
            "--url", args.url, "--output", str(episode_dir), "--episode-id", episode_id,
            "--route-manifest", str(args.route_manifest), "--route-id", route["route_id"],
            "--max-frames", str(args.max_frames), "--shard-size", str(args.shard_size),
        ]
        recovery = episode_dir / f"{episode_id}.recovery.json"
        if args.resume and recovery.exists(): command.append("--resume")
        if args.dry_run:
            print(subprocess.list2cmdline(command)); continue
        process = subprocess.run(command, text=True, capture_output=True)
        manifest = episode_dir / f"{episode_id}.manifest.json"
        if process.returncode != 0 or not manifest.is_file():
            status["failed"].append({"route_id": route["route_id"], "returncode": process.returncode, "stderr": process.stderr[-4000:]})
            atomic_json(status_path, status)
            print(json.dumps(status["failed"][-1], ensure_ascii=False), flush=True)
            return process.returncode or 1
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        record = {"route_id": route["route_id"], "manifest": str(manifest.resolve()), "sha256": sha256(manifest), "samples": payload["samples"]}
        status["complete"].append(record); completed_ids.add(route["route_id"]); status["failed"] = [item for item in status["failed"] if item["route_id"] != route["route_id"]]
        atomic_json(status_path, status); print(json.dumps({"progress": f"{ordinal}/{len(routes)}", **record}), flush=True)
    if not args.dry_run:
        status["formal_complete"] = len(status["complete"]) == len(routes)
        status["samples"] = sum(item["samples"] for item in status["complete"])
        atomic_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
