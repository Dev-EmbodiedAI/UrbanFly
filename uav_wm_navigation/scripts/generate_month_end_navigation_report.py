from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


METHOD_LABELS = {
    "yopo": "直接端到端规划",
    "yopo_dreamerv3": "规划 + Dreamer",
    "yopo_jepa": "规划 + JEPA",
    "yopo_tdmpc2": "规划 + TD-MPC2",
}
METHOD_COLORS = {
    "yopo": "#3288bd",
    "yopo_dreamerv3": "#e9903a",
    "yopo_jepa": "#8b67d9",
    "yopo_tdmpc2": "#3aad68",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="SMOKE")
    return parser.parse_args()


def resolve_summary(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def fmt_interval(value: object, digits: int = 2) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "—"
    return f"[{float(value[0]):.{digits}f}, {float(value[1]):.{digits}f}]"


def metric_svg(table: list[dict]) -> str:
    width, height = 920, 330
    left, plot_width = 190, 190
    ne_left = 650
    rows = []
    maximum_ne = max(
        [float(row.get("ne_m", 0.0)) for row in table if row.get("episodes", 0)]
        + [1.0]
    )
    for index, row in enumerate(table):
        method = str(row["method"])
        y = 62 + index * 62
        color = METHOD_COLORS.get(method, "#777777")
        sr = float(row.get("sr", 0.0))
        spl = float(row.get("spl", 0.0))
        ne = float(row.get("ne_m", 0.0))
        rows.append(
            f'<text x="12" y="{y + 16}" class="label">{html.escape(METHOD_LABELS.get(method, method))}</text>'
            f'<rect x="{left}" y="{y}" width="{plot_width * sr:.2f}" height="22" fill="{color}" opacity=".82"/>'
            f'<text x="{left + plot_width * sr + 6:.2f}" y="{y + 16}" class="value">{sr:.2f}</text>'
            f'<rect x="{left + 230}" y="{y}" width="{plot_width * spl:.2f}" height="22" fill="{color}" opacity=".82"/>'
            f'<text x="{left + 230 + plot_width * spl + 6:.2f}" y="{y + 16}" class="value">{spl:.2f}</text>'
            f'<rect x="{ne_left}" y="{y}" width="{190 * ne / maximum_ne:.2f}" height="22" fill="{color}" opacity=".82"/>'
            f'<text x="{ne_left + 190 * ne / maximum_ne + 6:.2f}" y="{y + 16}" class="value">{ne:.2f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="SR SPL NE comparison">'
        '<style>.label,.value,.axis{font:13px system-ui,sans-serif;fill:#20242a}.axis{fill:#626a73}.grid{stroke:#d9dde2;stroke-width:1}</style>'
        f'<text x="{left}" y="28" class="axis">SR ↑</text>'
        f'<text x="{left + 230}" y="28" class="axis">SPL ↑</text>'
        f'<text x="{ne_left}" y="28" class="axis">NE ↓ (m)</text>'
        + "".join(rows)
        + "</svg>"
    )


def trajectory_svg(rows: list[dict]) -> str:
    by_scenario: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in rows:
        summary_path = resolve_summary(str(row.get("summary_path", "")))
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        by_scenario[str(row.get("scenario", "unknown"))].append((row, summary))
    if not by_scenario:
        return ""
    panel_width, panel_height = 430, 300
    panels = []
    for panel_index, (scenario, episodes) in enumerate(sorted(by_scenario.items())):
        traces = []
        all_points = []
        for row, summary in episodes:
            start = np.asarray(summary["start_position_nwu"], dtype=np.float64)
            trajectory = np.asarray(summary.get("trajectory_nwu", []), dtype=np.float64)
            points = (
                np.vstack([start, trajectory])
                if trajectory.size
                else start.reshape(1, 3)
            )
            all_points.append(points[:, :2])
            traces.append((row, summary, points[:, :2]))
        joined = np.vstack(all_points)
        minimum = joined.min(axis=0)
        maximum = joined.max(axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        pad = span * 0.08
        minimum -= pad
        maximum += pad
        span = maximum - minimum

        def project(points: np.ndarray) -> str:
            x = 35 + (points[:, 0] - minimum[0]) / span[0] * (panel_width - 70)
            y = 45 + (maximum[1] - points[:, 1]) / span[1] * (panel_height - 85)
            return " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in zip(x, y))

        x_offset = panel_index * panel_width
        content = [
            f'<g transform="translate({x_offset},0)">',
            f'<text x="18" y="24" class="panel-title">{html.escape(scenario)}</text>',
            f'<rect x="20" y="35" width="{panel_width - 40}" height="{panel_height - 60}" class="panel"/>',
        ]
        for row, summary, points in traces:
            method = str(row["method"])
            color = METHOD_COLORS.get(method, "#777777")
            dash = "" if bool(summary.get("success")) else ' stroke-dasharray="6 4"'
            content.append(
                f'<polyline points="{project(points)}" fill="none" stroke="{color}" '
                f'stroke-width="3"{dash}/>'
            )
        content.append("</g>")
        panels.append("".join(content))
    total_width = panel_width * len(panels)
    legend = []
    x = 18
    for method in METHOD_LABELS:
        legend.append(
            f'<line x1="{x}" y1="326" x2="{x + 24}" y2="326" stroke="{METHOD_COLORS[method]}" stroke-width="4"/>'
            f'<text x="{x + 31}" y="331" class="legend">{html.escape(METHOD_LABELS[method])}</text>'
        )
        x += 190
    return (
        f'<svg viewBox="0 0 {total_width} 350" role="img" aria-label="paired trajectory comparison">'
        '<style>.panel{fill:#f7f8fa;stroke:#d9dde2}.panel-title{font:600 14px system-ui,sans-serif;fill:#20242a}'
        '.legend{font:12px system-ui,sans-serif;fill:#4f5862}</style>'
        + "".join(panels)
        + "".join(legend)
        + "</svg>"
    )


def main() -> int:
    args = parse_args()
    matrix_dir = args.matrix_dir.resolve()
    table = json.loads(
        (matrix_dir / "navigation_main_table.json").read_text(encoding="utf-8")
    )
    rows = json.loads(
        (matrix_dir / "closed_loop_results.json").read_text(encoding="utf-8")
    )
    output = (args.output or matrix_dir / "month_end_navigation_report.html").resolve()
    metric = metric_svg(table)
    trajectories = trajectory_svg(rows)
    table_rows = []
    for row in table:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(METHOD_LABELS.get(str(row['method']), str(row['method'])))}</td>"
            f"<td>{int(row.get('episodes', 0))}</td>"
            f"<td>{float(row.get('sr', math.nan)):.3f}</td>"
            f"<td>{fmt_interval(row.get('sr_ci95'), 3)}</td>"
            f"<td>{float(row.get('ne_m', math.nan)):.3f}</td>"
            f"<td>{fmt_interval(row.get('ne_m_ci95'), 3)}</td>"
            f"<td>{float(row.get('spl', math.nan)):.3f}</td>"
            f"<td>{fmt_interval(row.get('spl_ci95'), 3)}</td>"
            "</tr>"
        )
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UrbanFly 月末导航对比实验</title>
<style>
body{{font:14px/1.55 system-ui,sans-serif;color:#20242a;background:#fff;margin:28px auto;max-width:1120px;padding:0 20px}}
h1,h2{{line-height:1.2}}.badge{{display:inline-block;padding:4px 9px;border-radius:999px;background:#fff1cc;color:#6a4b00;font-weight:700}}
.warning{{border-left:4px solid #e9903a;padding:8px 12px;background:#fff8ec}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #d9dde2;text-align:right}}th:first-child,td:first-child{{text-align:left}}
svg{{width:100%;height:auto}}code{{background:#f1f3f5;padding:1px 4px}}
</style>
</head>
<body>
<span class="badge">{html.escape(args.label)}</span>
<h1>端到端规划与世界模型辅助规划对比</h1>
<p class="warning">本页为流程验证结果。样本数不足时不得用于宣称方法优劣；所有失败航迹均保留。</p>
<h2>主指标</h2>
<p>SR：进入目标容差的成功比例；NE：最终位置到目标的欧氏距离；SPL：成功率加权的参考路径效率。</p>
<table>
<thead><tr><th>方法</th><th>N</th><th>SR</th><th>SR 95% CI</th><th>NE (m)</th><th>NE 95% CI</th><th>SPL</th><th>SPL 95% CI</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
{metric}
<h2>同种子航迹</h2>
<p>实线表示成功，虚线表示失败。</p>
{trajectories}
<h2>审计口径</h2>
<ul>
<li>四种方法共享地图、场景、难度、随机种子和候选轨迹集合。</li>
<li>Dreamer、JEPA、TD-MPC2仅允许重排候选轨迹，不能生成另一套隐藏控制。</li>
<li>原始逐航迹数据保存在 <code>closed_loop_results.json</code>，汇总保存在 <code>navigation_main_table.json</code>。</li>
</ul>
</body>
</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(output), "episodes": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
