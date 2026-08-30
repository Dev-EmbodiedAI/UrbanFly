#!/usr/bin/env python3
"""Independently audit a directory of Helsinki Dataset v1 HDF5 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import _bootstrap  # noqa: F401

ROOT = _bootstrap.PROJECT_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_wm_navigation.data.helsinki_dataset_v1_qa import (  # noqa: E402
    audit_helsinki_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary_path = args.summary or args.output_dir / "collection_summary.json"
    resets = []
    if summary_path.exists():
        resets = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "reset_transitions", []
        )
    destination = args.output or args.output_dir / "independent_collection_qa.json"
    report = audit_helsinki_collection(
        args.output_dir,
        expected_episodes=args.expected_episodes,
        reset_transitions=resets,
        output_path=destination,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
