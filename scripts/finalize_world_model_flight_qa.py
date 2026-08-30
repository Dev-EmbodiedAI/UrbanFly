#!/usr/bin/env python
"""Re-run independent MP4 readback and finalize an existing flight QA report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_helsinki_world_model_video import inspect_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report["video"] = inspect_video(args.video.resolve())
    gates = {
        "navigation_success": bool(report.get("success")),
        "collision_free": not bool(report.get("collision", True)),
        "zero_stale_action": int(report.get("stale_action_count", -1)) == 0,
        "world_model_used_every_step": int(report.get("world_model_rerank_steps", -1))
        == int(report.get("steps", -2)),
        "latent_visualized_every_step": int(report.get("latent_visualization_steps", -1))
        == int(report.get("steps", -2)),
        "world_model_changed_action": int(report.get("selection_changed_steps", 0)) > 0,
        "goal_tolerance_met": float(report.get("minimum_goal_distance_m", float("inf"))) <= 3.0,
        "cross_track_gate_met": float(report.get("maximum_cross_track_error_m", float("inf"))) <= 15.0,
        "video_readback": report["video"]["status"] == "PASS",
    }
    report["final_gates"] = gates
    report["status"] = "PASS" if all(gates.values()) else "FAIL"
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "gates": gates, "video": report["video"]}, indent=2))


if __name__ == "__main__":
    main()
