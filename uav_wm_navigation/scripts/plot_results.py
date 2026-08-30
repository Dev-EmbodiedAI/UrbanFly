from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot closed-loop run completion by method.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    totals = Counter(row["method"] for row in rows)
    passed = Counter(row["method"] for row in rows if int(row["returncode"]) == 0)
    methods = sorted(totals)
    rates = [passed[name] / totals[name] for name in methods]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(methods, rates, color="#2563eb")
    axis.set_ylim(0, 1); axis.set_ylabel("Executable run success rate"); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); args.output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.output, dpi=180); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
