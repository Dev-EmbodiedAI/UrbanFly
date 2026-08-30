from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from uav_wm_navigation.evaluation.yopo_visualizer import render_yopo_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an official-YOPO-inspired candidate visualization.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--title", default="YOPO candidate evaluation")
    args = parser.parse_args()
    print(render_yopo_frame(args.input, args.output, args.frame, args.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
