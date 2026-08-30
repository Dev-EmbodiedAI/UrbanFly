from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the lowest validation-composite seed without reading test metrics.")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({
            "path": str(path.resolve()), "training_seed": int(checkpoint["training_seed"]),
            "validation_composite": float(checkpoint["best_validation_composite"]),
            "split_manifest_sha256": str(checkpoint["split_manifest_sha256"]),
        })
    if len({row["split_manifest_sha256"] for row in rows}) != 1:
        raise ValueError("candidate seeds were not trained on the same frozen split manifest")
    selected = min(rows, key=lambda row: row["validation_composite"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["path"], args.output)
    payload = {"selection_rule": "minimum validation composite; test metrics unread", "selected": selected, "candidates": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
