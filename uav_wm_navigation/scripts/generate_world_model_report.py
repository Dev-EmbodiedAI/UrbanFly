from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_wm_navigation.evaluation.world_model_report import (
    write_world_model_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, help="JSON list or JSONL records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.records.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    paths = write_world_model_report(records, args.output)
    print(json.dumps({name: str(path) for name, path in paths.items()}))


if __name__ == "__main__":
    main()
