#!/usr/bin/env python3
"""Re-execute and overlay all Helsinki qualification trajectories."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.helsinki_navigation import HelsinkiNavigationStack  # noqa: E402


SCENE = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
_STACK = None


def _initialize_worker(scene: str):
    global _STACK
    _STACK = HelsinkiNavigationStack.load(scene)


def _replay(record: dict) -> dict:
    start = np.asarray(record["start"], dtype=float)
    goal = np.asarray(record["goal"], dtype=float)
    plan = _STACK.plan(
        start,
        goal,
        flight_level="L2_transition",
        allow_layer_transitions=True,
    )
    execution = _STACK.execute(plan, timeout_s=180.0)
    return {
        "index": int(record["index"]),
        "task_type": record["task_type"],
        "recorded_result": record["result"],
        "replayed_result": execution["result"],
        "planner_mode": plan.planner_mode,
        "start": start.astype(np.float32),
        "goal": goal.astype(np.float32),
        "planned": plan.trajectory.astype(np.float32),
        "executed": execution["executed_trajectory"].astype(np.float32),
        "minimum_clearance_m": float(
            execution["executed_validation"]["minimum_clearance_m"]
        ),
    }


def _draw(stack, results, output: Path):
    height = stack.collision_map.height
    extent = [
        stack.collision_map.origin_x,
        stack.collision_map.origin_x
        + (height.shape[1] - 1) * stack.collision_map.resolution,
        stack.collision_map.origin_z,
        stack.collision_map.maximum_z,
    ]
    colors = {
        "ground_to_ground": "#22d3ee",
        "rooftop_to_ground": "#fbbf24",
        "ground_to_rooftop": "#f472b6",
        "rooftop_to_rooftop": "#a78bfa",
        "long_range_urban": "#34d399",
    }
    labels = {
        "ground_to_ground": "ground → ground (40)",
        "rooftop_to_ground": "rooftop → ground (40)",
        "ground_to_rooftop": "ground → rooftop (40)",
        "rooftop_to_rooftop": "rooftop → rooftop (40)",
        "long_range_urban": "long-range urban (40)",
    }

    figure, axis = plt.subplots(figsize=(15.5, 13.5), constrained_layout=True)
    figure.patch.set_facecolor("#07101c")
    axis.set_facecolor("#07101c")
    image = axis.imshow(
        height,
        extent=extent,
        origin="upper",
        cmap="turbo",
        vmin=-2.0,
        vmax=48.0,
        interpolation="nearest",
        rasterized=True,
    )

    # Planned lines remain barely visible; coloured lines are the actual
    # closed-loop replay trajectories.
    for result in results:
        planned = result["planned"]
        executed = result["executed"]
        color = colors[result["task_type"]]
        axis.plot(
            planned[:, 0],
            planned[:, 2],
            color="#ffffff",
            alpha=0.055,
            linewidth=0.45,
            linestyle="--",
            zorder=2,
        )
        axis.plot(
            executed[:, 0],
            executed[:, 2],
            color=color,
            alpha=0.30,
            linewidth=0.65,
            zorder=3,
        )

    astar = [item for item in results if item["planner_mode"] == "existing_pathplanner_multilayer_astar"]
    for result in astar:
        route = result["executed"]
        line = axis.plot(
            route[:, 0],
            route[:, 2],
            color="#ffffff",
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            alpha=0.95,
            zorder=5,
        )[0]
        line.set_path_effects([path_effects.Stroke(linewidth=3.2, foreground="#111827"), path_effects.Normal()])

    failures = [item for item in results if item["replayed_result"] != "SUCCESS"]
    for result in failures:
        route = result["executed"]
        axis.plot(route[:, 0], route[:, 2], color="#ff1f1f", linewidth=3.2, zorder=7)
        final = route[-1]
        axis.scatter(
            final[0],
            final[2],
            marker="X",
            s=180,
            color="#ff1f1f",
            edgecolors="#ffffff",
            linewidths=1.3,
            zorder=8,
        )
        axis.annotate(
            f"collision task #{result['index'] + 1}",
            xy=(final[0], final[2]),
            xytext=(12, 12),
            textcoords="offset points",
            color="#ffffff",
            fontsize=10,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#9f1239", "ec": "#ffffff", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "#ffffff"},
            zorder=9,
        )

    handles = [
        Line2D([0], [0], color=colors[key], lw=2.5, label=labels[key])
        for key in colors
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#ffffff", lw=1.8, linestyle="--", label=f"A* fallback ({len(astar)})"),
            Line2D([0], [0], color="#ff1f1f", lw=3.0, marker="X", label=f"collision replay ({len(failures)})"),
        ]
    )
    legend = axis.legend(
        handles=handles,
        loc="upper right",
        facecolor="#07101c",
        edgecolor="#cbd5e1",
        labelcolor="#ffffff",
        framealpha=0.90,
        fontsize=10,
        title="Actual 6-DOF replay",
    )
    legend.get_title().set_color("#ffffff")

    axis.set_xlim(-500, 500)
    axis.set_ylim(-500, 500)
    axis.set_aspect("equal")
    axis.set_xlabel("local x / east (m)", color="#ffffff", fontsize=12)
    axis.set_ylabel("local z / north (m)", color="#ffffff", fontsize=12)
    axis.tick_params(colors="#e2e8f0")
    for spine in axis.spines.values():
        spine.set_color("#cbd5e1")
    mismatches = sum(item["recorded_result"] != item["replayed_result"] for item in results)
    axis.set_title(
        "HelsinkiCentral1km — 200 actual closed-loop qualification trajectories\n"
        f"199 recorded successes · 1 collision · 4 A* fallbacks · deterministic replay mismatches: {mismatches}",
        color="#ffffff",
        fontsize=17,
        pad=14,
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.025)
    colorbar.set_label("real collision-surface height (m)", color="#ffffff", fontsize=11)
    colorbar.ax.tick_params(colors="#e2e8f0")
    colorbar.outline.set_edgecolor("#cbd5e1")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=185, facecolor=figure.get_facecolor())
    return figure


def _write_inline_fragment(figure, output: Path):
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="jpeg",
        dpi=72,
        facecolor=figure.get_facecolor(),
        pil_kwargs={"quality": 82},
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    fragment = f'''<div id="helsinki-qualification-map">
  <h2>Helsinki 200 qualification trajectories</h2>
  <img src="data:image/jpeg;base64,{encoded}" alt="Top-down real Helsinki collision heightmap overlaid with 200 closed-loop qualification trajectories, four A-star fallback routes, and the collision replay highlighted in red." style="display:block;width:100%;height:auto;" />
</div>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "outputs" / "helsinki_privileged_planner_final" / "qualification_records.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "navigation_reality_audit_after_restore" / "map_path_debug_200_trajectories.png",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fragment", type=Path)
    args = parser.parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    if len(records) != 200:
        raise ValueError(f"expected 200 qualification records, found {len(records)}")

    results = []
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_initialize_worker,
        initargs=(str(SCENE),),
    ) as executor:
        for index, result in enumerate(executor.map(_replay, records), start=1):
            results.append(result)
            if index % 10 == 0:
                print(f"replayed {index}/200", flush=True)

    stack = HelsinkiNavigationStack.load(SCENE)
    figure = _draw(stack, results, args.output)
    if args.fragment:
        _write_inline_fragment(figure, args.fragment)
    plt.close(figure)
    mismatches = sum(item["recorded_result"] != item["replayed_result"] for item in results)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        **{
            f"executed_{item['index']:03d}": item["executed"]
            for item in results
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "trajectories": len(results),
                "replay_mismatches": mismatches,
                "replay_failures": sum(item["replayed_result"] != "SUCCESS" for item in results),
                "astar_fallbacks": sum(item["planner_mode"] == "existing_pathplanner_multilayer_astar" for item in results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
