from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .schema import Episode, Step


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _safe_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def read_route(run_dir: Path, rows: list[dict[str, str]]) -> list[list[float]]:
    route_path = run_dir / "global_route.csv"
    if route_path.exists():
        route_rows = _read_csv(route_path)
        route = [[_safe_float(row, axis) for axis in ("x", "y", "z")] for row in route_rows]
        if len(route) >= 2:
            return route
    # Backward compatibility for older runs that did not persist the global route.
    stride = max(1, len(rows) // 12)
    route = [
        [_safe_float(row, "x"), _safe_float(row, "y"), _safe_float(row, "z")]
        for row in rows[::stride]
    ]
    last = [_safe_float(rows[-1], axis) for axis in ("x", "y", "z")]
    if not route or route[-1] != last:
        route.append(last)
    return route


def geometry_instruction(route: list[list[float]]) -> str:
    points = np.asarray(route, dtype=np.float64)
    delta = points[-1, :2] - points[0, :2]
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
    directions = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]
    direction = directions[int((angle + 22.5) // 45.0) % 8]
    distance = float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())

    turn_words: list[str] = []
    vectors = np.diff(points[:, :2], axis=0)
    for first, second in zip(vectors[:-1], vectors[1:]):
        if np.linalg.norm(first) < 1e-6 or np.linalg.norm(second) < 1e-6:
            continue
        cross = float(first[0] * second[1] - first[1] * second[0])
        cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
        if cosine < 0.75:
            turn_words.append("left" if cross > 0 else "right")
    turns = ", then ".join(f"turn {word}" for word in turn_words[:3])
    turn_clause = f"; {turns}" if turns else ""
    altitude = abs(float(points[0, 2]))
    return (
        f"Fly {direction} for about {distance:.0f} meters at roughly {altitude:.0f} meters altitude"
        f"{turn_clause}. Keep clear of buildings and moving traffic, then stop at the route endpoint."
    )


def build_episode(run_dir: Path, split: str = "train", risk_depth_m: float = 2.0) -> Episode:
    summary_path = run_dir / "long_range_summary.json"
    trajectory_path = run_dir / "long_range_trajectory.csv"
    if not summary_path.exists() or not trajectory_path.exists():
        raise FileNotFoundError("run directory must contain long_range_summary.json and long_range_trajectory.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_csv(trajectory_path)
    if len(rows) < 2:
        raise ValueError("trajectory must contain at least two rows")
    route = read_route(run_dir, rows)
    provided_instruction = str(summary.get("instruction", "")).strip()
    instruction = provided_instruction or geometry_instruction(route)

    steps: list[Step] = []
    previous_collisions = 0.0
    for index, row in enumerate(rows):
        current = np.array([_safe_float(row, axis) for axis in ("x", "y", "z")], dtype=np.float64)
        next_row = rows[min(index + 1, len(rows) - 1)]
        following = np.array([_safe_float(next_row, axis) for axis in ("x", "y", "z")], dtype=np.float64)
        collision_count = _safe_float(row, "collision_events")
        collision = collision_count > previous_collisions
        previous_collisions = max(previous_collisions, collision_count)
        replan = int(_safe_float(row, "replan_step"))
        rgb = run_dir / "observations" / "rgb" / f"plan_{replan:04d}.png"
        depth = run_dir / "observations" / "depth" / f"plan_{replan:04d}.npy"
        p05 = _safe_float(row, "p05_depth_m")
        steps.append(
            Step(
                time_s=_safe_float(row, "time_s"),
                position=current.tolist(),
                velocity=[_safe_float(row, axis) for axis in ("vx", "vy", "vz")],
                action_delta=(following - current).tolist(),
                goal_distance_m=_safe_float(row, "final_goal_distance_m"),
                min_depth_m=_safe_float(row, "min_depth_m"),
                p05_depth_m=p05,
                collision=collision,
                replan_step=replan,
                target_waypoint_idx=int(_safe_float(row, "target_waypoint_idx")),
                rgb_path=str(rgb.relative_to(run_dir)) if rgb.exists() else None,
                depth_path=str(depth.relative_to(run_dir)) if depth.exists() else None,
            )
        )

    digest = hashlib.sha1(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    episode = Episode(
        episode_id=f"urbanfly-{digest}",
        scene_id=str(summary.get("carla_map") or Path(str(summary.get("dataset_root", "Town03"))).name),
        instruction=instruction,
        route=route,
        steps=steps,
        split=split,
        instruction_source="user_semantic" if provided_instruction else "geometry_template",
        success=bool(summary.get("success", False)),
        metadata={
            "source_run": str(run_dir.resolve()),
            "instruction_limit": "Geometry template only; replace with landmark-grounded human/paraphrased instructions.",
            "risk_depth_m": risk_depth_m,
            "collision_steps": int(summary.get("collision_steps", 0)),
        },
    )
    episode.validate()
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an UrbanFly YOPO run into UrbanFly-VLN episode JSON.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val_seen", "val_unseen", "test"], default="train")
    args = parser.parse_args()
    episode = build_episode(args.run_dir.resolve(), split=args.split)
    episode.write_json(args.output.resolve())
    print(json.dumps({"episode": episode.episode_id, "steps": len(episode.steps), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
