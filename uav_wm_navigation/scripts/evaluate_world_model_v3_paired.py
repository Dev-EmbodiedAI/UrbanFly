from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_wm_navigation.evaluation.paired_benchmark import (
    expected_jobs, load_evaluation_manifest, paired_deltas, summarize_results, validate_results,
)


def read_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
        records.extend(payload)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and aggregate the preregistered UrbanFly v3 paired benchmark")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Label output preliminary; formal output rejects missing jobs")
    parser.add_argument("--write-job-list", action="store_true")
    args = parser.parse_args()
    manifest = load_evaluation_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = expected_jobs(manifest)
    if args.write_job_list:
        (args.output / "expected_jobs.jsonl").write_text("".join(json.dumps(item) + "\n" for item in jobs), encoding="utf-8")
    records = read_records(args.results)
    normalized, missing = validate_results(records, manifest, allow_incomplete=args.allow_incomplete)
    payload = {
        "schema": "urbanfly-paired-summary-v3",
        "formal_complete": not missing,
        "manifest_sha256": manifest["manifest_sha256"],
        "received_jobs": len(normalized), "expected_jobs": len(jobs), "missing_jobs": len(missing),
        "summary": summarize_results(normalized) if normalized else [],
        "paired_deltas_vs_yopo_direct": paired_deltas(normalized) if normalized else [],
    }
    (args.output / "paired_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "missing_jobs.jsonl").write_text("".join(json.dumps(item) + "\n" for item in missing), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "received": len(normalized), "expected": len(jobs), "missing": len(missing), "formal_complete": not missing}))


if __name__ == "__main__":
    main()
