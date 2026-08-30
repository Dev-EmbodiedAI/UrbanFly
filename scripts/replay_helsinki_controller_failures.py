#!/usr/bin/env python3
"""Replay the four controller failures from the frozen 200-task benchmark."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_navigation import HelsinkiNavigationStack  # noqa: E402
from scripts.verify_helsinki_low_altitude_expert import (  # noqa: E402
    LowAltitudeTask,
    _jsonable,
    _run_task,
)


FAILED_INDICES = (38, 79, 121, 133)


def main() -> int:
    scene = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
    benchmark = ROOT / "outputs" / "helsinki_low_altitude_expert"
    output = ROOT / "outputs" / "helsinki_controller_failure_replay_after_fix"
    output.mkdir(parents=True, exist_ok=True)
    paths = output / "paths"
    paths.mkdir(exist_ok=True)
    raw_tasks = json.loads((benchmark / "qualification_tasks.json").read_text(encoding="utf-8"))
    accepted = {field.name for field in fields(LowAltitudeTask)}
    stack = HelsinkiNavigationStack.load(scene)
    records = []
    for index in FAILED_INDICES:
        task = LowAltitudeTask(**{key: value for key, value in raw_tasks[index].items() if key in accepted})
        record = _run_task(stack, task, paths / f"task_{index:03d}.npz")
        records.append(record)
        print(
            f"replay {index}: {record['result']} rmse={record.get('tracking_rmse_m')} "
            f"ceiling_samples={record.get('executed_height_violation_samples')}",
            flush=True,
        )
    report = {
        "source_qualification_failed_indices": list(FAILED_INDICES),
        "controller_configuration": "one generic low-altitude configuration; no task-specific branch",
        "success_count": sum(record.get("execution_success", False) for record in records),
        "collision_count": sum(record.get("collision", False) for record in records),
        "ceiling_violation_episode_count": sum(
            int(record.get("executed_height_violation_samples", 0)) > 0 for record in records
        ),
        "records": records,
    }
    (output / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if report["success_count"] == len(FAILED_INDICES) else 2


if __name__ == "__main__":
    raise SystemExit(main())
