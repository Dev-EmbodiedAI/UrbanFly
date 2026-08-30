from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import paired_bootstrap_interval, wilson_interval


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(
            str(record["method"]), str(record.get("model_seed", "none")),
            str(record.get("group", "all")), bool(record.get("shield_enabled", False)),
        )].append(record)
    summaries = []
    for (method, model_seed, group, shield), items in sorted(grouped.items()):
        success = np.asarray([bool(item.get("success", False)) for item in items])
        ne = np.asarray([_number(item.get("navigation_error_m", item.get("ne_m"))) for item in items])
        spl = np.asarray([_number(item.get("spl")) for item in items])
        sr, sr_low, sr_high = wilson_interval(int(success.sum()), len(items))
        ne_mean, ne_low, ne_high = paired_bootstrap_interval(ne, seed=20260831)
        spl_mean, spl_low, spl_high = paired_bootstrap_interval(spl, seed=20260832)
        latencies = np.asarray([_number(item.get("latency_p95_ms", item.get("latency_ms"))) for item in items])
        clearance = np.asarray([_number(item.get("minimum_clearance_m")) for item in items])
        decisions = sum(int(item.get("decision_steps", 0)) for item in items)
        interventions = sum(int(item.get("intervention_steps", 0)) for item in items)
        summaries.append({
            "method": method, "model_seed": model_seed, "group": group, "shield_enabled": shield,
            "episodes": len(items), "successes": int(success.sum()),
            "sr": sr, "sr_ci95_low": sr_low, "sr_ci95_high": sr_high,
            "ne_m": ne_mean, "ne_ci95_low": ne_low, "ne_ci95_high": ne_high,
            "spl": spl_mean, "spl_ci95_low": spl_low, "spl_ci95_high": spl_high,
            "collisions": sum(bool(item.get("collision", False)) for item in items),
            "minimum_clearance_mean_m": float(clearance.mean()),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "intervention_rate": interventions / max(decisions, 1),
        })
    return summaries


def _main_svg(summaries: list[dict[str, Any]]) -> str:
    width, height, left, top = 1040, 520, 110, 62
    rows = len(summaries); bar_area = max(1, height - top - 50)
    row_height = bar_area / max(rows, 1)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="SR NE SPL comparison">',
        '<rect width="100%" height="100%" fill="#07131c"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#dceaf4}.muted{fill:#91a9b8;font-size:12px}.label{font-size:11px}</style>',
        '<text x="30" y="32" font-size="20" font-weight="700">UrbanFly paired navigation: SR / NE / SPL</text>',
    ]
    if not summaries:
        elements.append('<text x="30" y="90" class="muted">No completed evaluation records. No metric has been fabricated.</text>')
    for index, item in enumerate(summaries):
        y = top + index * row_height + row_height * 0.5
        label = f"{item['method']} s={item['model_seed']} {item['group']} {'shield' if item['shield_enabled'] else 'raw'}"
        elements.append(f'<text x="20" y="{y + 4:.1f}" class="label">{html.escape(label[:58])}</text>')
        x0, chart_width = 430, 540
        sr_width = item["sr"] * chart_width
        spl_width = item["spl"] * chart_width
        elements.append(f'<rect x="{x0}" y="{y-8:.1f}" width="{sr_width:.1f}" height="6" fill="#54c7ff"><title>SR {item["sr"]:.3f}</title></rect>')
        elements.append(f'<rect x="{x0}" y="{y+2:.1f}" width="{spl_width:.1f}" height="6" fill="#72e0ae"><title>SPL {item["spl"]:.3f}</title></rect>')
        elements.append(f'<text x="{x0+chart_width+8}" y="{y+4:.1f}" class="label">SR {item["sr"]:.2f} · NE {item["ne_m"]:.1f}m · SPL {item["spl"]:.2f}</text>')
    elements.append('</svg>')
    return ''.join(elements)


def _write_pdf(summaries: list[dict[str, Any]], path: Path) -> bool:
    if not summaries:
        return False
    import matplotlib.pyplot as plt

    labels = [f"{item['method']}\n{item['group']}\n{'shield' if item['shield_enabled'] else 'raw'}" for item in summaries]
    y = np.arange(len(summaries))
    figure, axes = plt.subplots(1, 3, figsize=(16, max(5, len(summaries) * 0.28)), sharey=True)
    for axis, key, title, color in zip(axes, ("sr", "ne_m", "spl"), ("SR (higher)", "NE metres (lower)", "SPL (higher)"), ("#3288bd", "#d95f5f", "#53a567")):
        axis.barh(y, [item[key] for item in summaries], color=color)
        axis.set_title(title); axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels, fontsize=6); axes[0].invert_yaxis()
    figure.tight_layout(); figure.savefig(path, bbox_inches="tight"); plt.close(figure)
    return True


def write_world_model_report(
    records: list[dict[str, Any]], output_dir: str | Path, *,
    title: str = "UrbanFly 单机世界模型辅助城市飞行实验报告",
    expected_jobs: int | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Path]:
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    summaries = summarize_records(records) if records else []
    complete = expected_jobs is None or len(records) == expected_jobs
    status = {
        "schema": "urbanfly-world-model-report-v3", "formal_complete": complete,
        "received_jobs": len(records), "expected_jobs": expected_jobs,
        "manifest_sha256": manifest_sha256, "summary": summaries,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output / "sr_ne_spl_main_table.csv"
    fields = list(summaries[0]) if summaries else ["method", "model_seed", "group", "shield_enabled", "episodes", "sr", "ne_m", "spl"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    svg_path = output / "sr_ne_spl_comparison.svg"
    svg_path.write_text(_main_svg(summaries), encoding="utf-8")
    pdf_path = output / "sr_ne_spl_comparison.pdf"
    pdf_written = _write_pdf(summaries, pdf_path)
    rows = ''.join(
        '<tr>'
        f'<td>{html.escape(item["method"])}</td><td>{html.escape(item["model_seed"])}</td><td>{html.escape(item["group"])}</td><td>{"开启" if item["shield_enabled"] else "关闭"}</td>'
        f'<td>{item["episodes"]}</td><td>{item["sr"]:.1%} [{item["sr_ci95_low"]:.1%}, {item["sr_ci95_high"]:.1%}]</td>'
        f'<td>{item["ne_m"]:.2f} [{item["ne_ci95_low"]:.2f}, {item["ne_ci95_high"]:.2f}]</td>'
        f'<td>{item["spl"]:.3f} [{item["spl_ci95_low"]:.3f}, {item["spl_ci95_high"]:.3f}]</td>'
        f'<td>{item["collisions"]}</td><td>{item["latency_p95_ms"]:.1f}</td></tr>'
        for item in summaries
    ) or '<tr><td colspan="10">尚无已完成回合；报告不会生成虚构指标。</td></tr>'
    failures = [item for item in records if not bool(item.get("success", False))]
    failure_rows = ''.join(
        f'<tr><td>{html.escape(str(item.get("route_id", "")))}</td><td>{html.escape(str(item.get("method", "")))}</td>'
        f'<td>{html.escape(str(item.get("termination_reason", item.get("failure_reason", "未说明"))))}</td>'
        f'<td>{html.escape(str(item.get("video_path", "未录制")))}</td></tr>' for item in failures
    ) or '<tr><td colspan="4">无失败记录。</td></tr>'
    status_text = (
        f'正式结果完整：{len(records)}/{expected_jobs} 个预注册回合。' if complete
        else f'当前为进度报告：仅完成 {len(records)}/{expected_jobs} 个预注册回合，不可作为正式结论。'
    )
    report_path = output / "report.html"
    report_path.write_text(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;background:#07131c;color:#dceaf4;margin:32px}}.card{{background:#0d202c;border:1px solid #294151;border-radius:12px;padding:18px;margin:18px 0}}.warn{{color:#ffd166}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #294151;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}a{{color:#54c7ff}}</style></head><body>
<h1>{html.escape(title)}</h1><p class="{'card' if complete else 'card warn'}">{html.escape(status_text)}<br>评测清单 SHA-256：{html.escape(str(manifest_sha256 or '未提供'))}</p>
<div class="card"><object data="{svg_path.name}" type="image/svg+xml" width="100%"></object></div>
<div class="card"><h2>预注册主指标</h2><table><thead><tr><th>方法</th><th>训练种子</th><th>评测组</th><th>安全层</th><th>回合</th><th>SR (Wilson 95%)</th><th>NE m (bootstrap 95%)</th><th>SPL (bootstrap 95%)</th><th>碰撞</th><th>P95 ms</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="card"><h2>全部失败案例</h2><table><thead><tr><th>路线</th><th>方法</th><th>终止原因</th><th>连续运行时视频</th></tr></thead><tbody>{failure_rows}</tbody></table></div>
<div class="card"><h2>可追溯文件</h2><p><a href="{summary_path.name}">JSON 摘要</a> · <a href="{csv_path.name}">CSV 主表</a> · <a href="{svg_path.name}">SVG 图</a>{' · <a href="' + pdf_path.name + '">PDF 图</a>' if pdf_written else ''}</p></div>
</body></html>''', encoding="utf-8")
    paths = {"html": report_path, "summary": summary_path, "csv": csv_path, "svg": svg_path}
    if pdf_written: paths["pdf"] = pdf_path
    return paths
