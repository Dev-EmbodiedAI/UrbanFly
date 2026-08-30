from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_wm_navigation.evaluation.paired_benchmark import expected_jobs, load_evaluation_manifest, validate_results
from uav_wm_navigation.evaluation.world_model_report import write_world_model_report


def read_records(paths: list[Path]) -> list[dict]:
    output = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        output.extend(json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the auditable UrbanFly v3 HTML/SVG/PDF report")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = load_evaluation_manifest(args.manifest)
    records, missing = validate_results(read_records(args.results), manifest, allow_incomplete=args.allow_incomplete)
    paths = write_world_model_report(
        records, args.output, expected_jobs=len(expected_jobs(manifest)),
        manifest_sha256=manifest["manifest_sha256"],
    )
    (args.output / "missing_jobs.jsonl").write_text("".join(json.dumps(item) + "\n" for item in missing), encoding="utf-8")
    print(json.dumps({name: str(path) for name, path in paths.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
