"""
Optimized benchmark runner for the real Empty.umap-derived city.

This script keeps the same evaluation dimensions as the original
run_midterm_benchmark.py, but reduces duplicated runs and uses lighter
baseline hyper-parameters so the full suite can complete reliably on the
real-city scene.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_midterm_benchmark as bench


OUTPUT_JSON = ROOT / "data" / "midterm_benchmark.json"
OUTPUT_MD = ROOT / "data" / "midterm_benchmark_summary.md"
TASK_COUNT = 84


def make_paper_allocators():
    stc_kwargs = dict(max_iterations=18, max_bundle_size=14, use_residual_repair=True)
    return {
        "STC-RCBBA": bench.CBBAAllocator(**stc_kwargs, use_priority_term=True, use_corridor_term=True, use_robust_consensus=True),
        "原始CBBA": bench.CBBAAllocator(
            max_iterations=8,
            max_bundle_size=6,
            use_priority_term=True,
            use_corridor_term=False,
            use_robust_consensus=False,
            use_residual_repair=False,
            display_name="原始CBBA",
        ),
        "Hungarian": bench.HungarianAllocator(),
        "Greedy": bench.GreedyAllocator(max_tasks_per_drone=4),
        "Auction": bench.AuctionAllocator(max_rounds=14, epsilon=0.12),
        "Genetic": bench.GeneticAllocator(population_size=16, generations=12),
        "Market": bench.MarketAllocator(),
        "PSO": bench.PSOAllocator(n_particles=18, n_iterations=20),
        "GWO": bench.GWOAllocator(n_wolves=18, n_iterations=20),
        "ACO": bench.ACOAllocator(n_ants=16, n_iterations=18),
        "WOA": bench.WOAAllocator(n_whales=18, n_iterations=20),
        "SA": bench.SAAllocator(T_init=320.0, T_min=0.1, alpha=0.88, steps_per_T=12),
        "DE": bench.DEAllocator(pop_size=16, n_iterations=20),
    }


def make_paper_ablation_allocators():
    stc_kwargs = dict(max_iterations=18, max_bundle_size=14, use_residual_repair=True)
    return {
        "去掉优先级紧迫项": bench.CBBAAllocator(
            **stc_kwargs,
            use_priority_term=False,
            use_corridor_term=True,
            use_robust_consensus=True,
            display_name="去掉优先级紧迫项",
        ),
        "去掉通信鲁棒共识": bench.CBBAAllocator(
            **stc_kwargs,
            use_priority_term=True,
            use_corridor_term=True,
            use_robust_consensus=False,
            display_name="去掉通信鲁棒共识",
        ),
        "去掉走廊冲突代价": bench.CBBAAllocator(
            **stc_kwargs,
            use_priority_term=True,
            use_corridor_term=False,
            use_robust_consensus=True,
            display_name="去掉走廊冲突代价",
        ),
    }


def copy_rows_for_scenario(rows, algorithms, scenario_name):
    copied = []
    for row in rows:
        if row["algorithm"] not in algorithms:
            continue
        item = copy.deepcopy(row)
        item["scenario"] = scenario_name
        copied.append(item)
    return copied


def compute_reference_path_example(drones, tasks):
    planner = bench.load_planner_if_available()
    if planner is None or not tasks:
        return {}

    anchor_drone = next((d for d in drones if d.drone_type == "standard"), drones[0])
    ranked_tasks = sorted(
        tasks,
        key=lambda t: float(bench.np.linalg.norm(t.delivery_pos[[0, 2]] - anchor_drone.position[[0, 2]])),
        reverse=True,
    )
    chosen_tasks = ranked_tasks[:2]
    if not chosen_tasks:
        return {}

    path, _ = planner.plan_bundle_path(anchor_drone, chosen_tasks, start_time=bench.CURRENT_TIME)
    metrics = planner.estimate_path_metrics(path, cruise_speed=anchor_drone.cruise_speed, preferred_layer=chosen_tasks[0].airspace_level)
    debug = planner._last_debug
    raw_skeleton = [bench.np.array(p, dtype=float) for p in debug.get("skeleton", [])]
    smooth = [bench.np.array(p, dtype=float) for p in debug.get("smooth", [])]

    return {
        "drone_id": anchor_drone.id,
        "task_ids": [t.id for t in chosen_tasks],
        "planner_metrics": metrics,
        "debug": debug,
        "raw_curvature": bench.curvature_cost(raw_skeleton),
        "smooth_curvature": bench.curvature_cost(smooth),
    }


def markdown_table(rows, columns):
    header = "| " + " | ".join(name for _, name in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [header, sep]
    for row in rows:
        vals = []
        for key, _ in columns:
            val = row[key]
            vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join(body)


def to_jsonable(obj):
    return bench.to_jsonable(obj)


def main():
    layout = bench.load_city_layout()
    buildings = layout["buildings"]
    density_meta = bench.build_density_grid(buildings)
    hotspots = bench.make_hotspots(layout)

    drones = bench.build_drones()
    tasks_main = bench.build_tasks(layout, hotspots, TASK_COUNT, dynamic_bias=1.24)
    tasks_dynamic = bench.build_tasks(layout, hotspots, TASK_COUNT, dynamic_bias=1.05)
    main_graph = bench.make_comm_graph(drones, buildings, "occlusion")

    algorithms = make_paper_allocators()
    print("== 主算法对比：真实城市高密度遮挡场景 ==")
    main_results = bench.run_algorithm_suite(algorithms, drones, tasks_main, density_meta, main_graph, "dense_occlusion")
    bench.annotate_composite_scores(main_results)

    print("\n== 通信退化对比：复用遮挡场景并补跑三类退化链路 ==")
    comm_algorithms = ("STC-RCBBA", "原始CBBA", "Auction", "Hungarian")
    comm_results = copy_rows_for_scenario(main_results, comm_algorithms, "building_occlusion")
    for scenario_name, scenario_key in [
        ("full_comm", "ideal"),
        ("intermittent_comm", "intermittent"),
        ("local_island", "islanded"),
    ]:
        graph = bench.make_comm_graph(drones, buildings, scenario_key)
        rows = bench.run_algorithm_suite(
            {name: algorithms[name] for name in comm_algorithms},
            drones,
            tasks_main,
            density_meta,
            graph,
            scenario_name,
        )
        comm_results.extend(rows)

    print("\n== 场景复杂度对比：复用街谷基线并补跑常规密度与动态注入 ==")
    complexity_algorithms = ("STC-RCBBA", "原始CBBA", "Auction", "Hungarian")
    complexity_results = copy_rows_for_scenario(main_results, complexity_algorithms, "street_canyon_dense")
    for scenario_name, tasks_scene, graph in [
        ("regular_density", bench.build_tasks(layout, hotspots, TASK_COUNT, dynamic_bias=1.36), bench.make_comm_graph(drones, buildings, "ideal")),
        ("dynamic_injection", tasks_dynamic, bench.make_comm_graph(drones, buildings, "intermittent")),
    ]:
        rows = bench.run_algorithm_suite(
            {name: algorithms[name] for name in complexity_algorithms},
            drones,
            tasks_scene,
            density_meta,
            graph,
            scenario_name,
        )
        complexity_results.extend(rows)

    print("\n== 消融实验：复用主算法基线并补跑三类去除项 ==")
    ablation_results_raw = []
    ablation_algorithms = make_paper_ablation_allocators()
    ablation_results_raw.extend(
        bench.run_algorithm_suite(ablation_algorithms, drones, tasks_main, density_meta, main_graph, "dense_ablation")
    )
    stc_row = next(copy.deepcopy(row) for row in main_results if row["algorithm"] == "STC-RCBBA")
    ablation_metrics = [stc_row] + ablation_results_raw
    path_example = compute_reference_path_example(drones, tasks_main)
    ablation_rows = bench.build_ablation_rows(ablation_metrics, path_example)

    payload = {
        "meta": {
            "seed": bench.RANDOM_SEED,
            "current_time": bench.CURRENT_TIME,
            "task_count": TASK_COUNT,
            "city_source": layout["source_path"],
            "building_count": len(buildings),
            "fleet_composition": bench.BENCHMARK_FLEET,
            "runner": "run_midterm_benchmark_realcity.py",
        },
        "hotspots": {k: v.tolist() for k, v in hotspots.items()},
        "main_results": main_results,
        "communication_results": comm_results,
        "complexity_results": complexity_results,
        "ablation_results": ablation_rows,
        "path_example": path_example,
        "sample_city": {
            "buildings": buildings,
            "drones": [{"id": d.id, "drone_type": d.drone_type, "position": d.position.tolist()} for d in drones],
            "tasks": [t.to_dict() for t in tasks_main],
            "comm_graph_occlusion": main_graph.tolist(),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    ranked = sorted(main_results, key=lambda row: row["composite_score"], reverse=True)
    OUTPUT_MD.write_text(
        "# 中期报告实验结果摘要\n\n"
        "## 主算法对比\n\n"
        + markdown_table(
            ranked,
            [
                ("algorithm", "算法"),
                ("composite_score", "综合评分"),
                ("weighted_completion_rate", "加权完成率"),
                ("time_window_rate", "时间窗满足率"),
                ("utility_energy_ratio", "单位收益能耗比"),
                ("corridor_conflicts", "走廊冲突"),
                ("runtime_ms", "耗时(ms)"),
            ],
        )
        + "\n\n## 消融结果\n\n"
        + markdown_table(
            ablation_rows,
            [
                ("variant", "变体"),
                ("weighted_completion_rate", "加权完成率"),
                ("time_window_rate", "时间窗满足率"),
                ("utility_energy_ratio", "单位收益能耗比"),
                ("conflict_suppression", "冲突抑制"),
                ("smoothness_index", "平滑度"),
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n结果已写入: {OUTPUT_JSON}")
    print(f"摘要已写入: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
