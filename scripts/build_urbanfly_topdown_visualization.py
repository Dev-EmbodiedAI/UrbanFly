"""Build an evidence-backed top-down UrbanFly route visualization.

The figure uses the real 0.5 m ESDF slice and the functional-zone polygons from
the CityCentral1km scene.  A collision-free demonstration route is planned on a
2 m working grid only for the visual overlay; the collision source remains the
full-resolution 0.5 m product.
"""

from __future__ import annotations

import argparse
import base64
import heapq
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label


WORKSPACE = Path(__file__).resolve().parents[1]
SCENE = WORKSPACE / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
ESDF_NPZ = SCENE / "diagnostics" / "esdf_020m_0p5m.npz"
ESDF_PNG = SCENE / "diagnostics" / "esdf_020m_0p5m.png"
ZONES_JSON = SCENE / "zones" / "functional_zones.json"
DEFAULT_INLINE = Path(
    r"C:\Users\caste\.codex\visualizations\2026\07\26"
    r"\019f9d34-fe45-77e3-801a-98bcbffb0bfe"
    r"\urbanfly-topdown-flight-map.html"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_INLINE)
    parser.add_argument(
        "--project-copy",
        type=Path,
        default=WORKSPACE
        / "uav_wm_navigation"
        / "outputs"
        / "urbanfly_topdown_flight_map.html",
    )
    return parser.parse_args()


def nearest_free(
    free: np.ndarray, target_xz: tuple[float, float], metres_per_cell: float
) -> tuple[int, int]:
    rows, cols = free.shape
    col = int(round((target_xz[0] + 500.0) / metres_per_cell))
    row = int(round((500.0 - target_xz[1]) / metres_per_cell))
    col = int(np.clip(col, 0, cols - 1))
    row = int(np.clip(row, 0, rows - 1))
    if free[row, col]:
        return row, col
    for radius in range(1, max(rows, cols)):
        r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
        c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
        candidates = []
        for rr in range(r0, r1):
            for cc in (c0, c1 - 1):
                if free[rr, cc]:
                    candidates.append((rr, cc))
        for cc in range(c0 + 1, c1 - 1):
            for rr in (r0, r1 - 1):
                if free[rr, cc]:
                    candidates.append((rr, cc))
        if candidates:
            return min(
                candidates,
                key=lambda cell: (cell[0] - row) ** 2 + (cell[1] - col) ** 2,
            )
    raise RuntimeError("No free cell in route map")


def astar(
    free: np.ndarray,
    clearance_cells: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    rows, cols = free.shape
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 2**0.5),
        (-1, 1, 2**0.5),
        (1, -1, 2**0.5),
        (1, 1, 2**0.5),
    )
    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != g_score.get(current):
            continue
        if current == goal:
            route = [current]
            while route[-1] != start:
                route.append(came_from[route[-1]])
            route.reverse()
            return route
        rr, cc = current
        for dr, dc, step_cost in moves:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols and free[nr, nc]):
                continue
            # Prefer corridors with several metres of geometric clearance.
            clearance_penalty = 5.5 / max(float(clearance_cells[nr, nc]), 1.0)
            candidate = cost + step_cost * (1.0 + clearance_penalty)
            neighbor = (nr, nc)
            if candidate >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = candidate
            heuristic = ((nr - goal[0]) ** 2 + (nc - goal[1]) ** 2) ** 0.5
            heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
    raise RuntimeError(f"No collision-free route between {start} and {goal}")


def simplify_polyline(
    points: list[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = np.asarray(points[0], dtype=np.float64)
    end = np.asarray(points[-1], dtype=np.float64)
    segment = end - start
    denom = float(segment @ segment)
    values = np.asarray(points[1:-1], dtype=np.float64)
    if denom <= 1e-12:
        distances = np.linalg.norm(values - start, axis=1)
    else:
        t = np.clip(((values - start) @ segment) / denom, 0.0, 1.0)
        projection = start + t[:, None] * segment
        distances = np.linalg.norm(values - projection, axis=1)
    if not len(distances) or float(distances.max()) <= tolerance:
        return [points[0], points[-1]]
    split = int(distances.argmax()) + 1
    left = simplify_polyline(points[: split + 1], tolerance)
    right = simplify_polyline(points[split:], tolerance)
    return left[:-1] + right


def create_route() -> list[tuple[float, float]]:
    data = np.load(ESDF_NPZ)
    occupied = np.asarray(data["occupied"], dtype=bool)
    # Reduce only the planner's working grid from 0.5 m to 2 m.  Max pooling
    # conservatively preserves every obstacle from the source grid.
    factor = 4
    usable_rows = occupied.shape[0] - 1
    usable_cols = occupied.shape[1] - 1
    pooled = occupied[:usable_rows, :usable_cols].reshape(
        usable_rows // factor, factor, usable_cols // factor, factor
    )
    occupied_2m = pooled.max(axis=(1, 3))
    inflated = binary_dilation(occupied_2m, iterations=2)
    free = ~inflated
    components, component_count = label(free, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count:
        counts = np.bincount(components.ravel())
        counts[0] = 0
        free = components == int(counts.argmax())
    clearance = distance_transform_edt(free)
    metres_per_cell = 2.0

    requested = [
        (-355.0, -430.0),
        (-265.0, -155.0),
        (-360.0, 85.0),
        (-90.0, 285.0),
        (265.0, 420.0),
    ]
    anchors = [nearest_free(free, point, metres_per_cell) for point in requested]
    cells: list[tuple[int, int]] = []
    for start, goal in zip(anchors, anchors[1:]):
        segment = astar(free, clearance, start, goal)
        cells.extend(segment if not cells else segment[1:])
    route = [
        (
            col * metres_per_cell - 500.0 + metres_per_cell / 2.0,
            500.0 - row * metres_per_cell - metres_per_cell / 2.0,
        )
        for row, col in cells
    ]
    return simplify_polyline(route, tolerance=3.0)


def polygon_points(coordinates: Iterable[Iterable[float]]) -> str:
    return " ".join(
        f"{float(x) + 500.0:.2f},{500.0 - float(z):.2f}"
        for x, z in coordinates
    )


def build_html() -> str:
    route = create_route()
    route_points = " ".join(
        f"{x + 500.0:.2f},{500.0 - z:.2f}" for x, z in route
    )
    zones = json.loads(ZONES_JSON.read_text(encoding="utf-8"))
    colors = {item["id"]: item["color"] for item in zones["classes"]}
    labels = {item["id"]: item["label"] for item in zones["classes"]}
    zone_shapes = []
    for feature in zones["features"]:
        geometry = feature["geometry"]
        rings = []
        if geometry["type"] == "Polygon":
            rings = geometry["coordinates"]
        elif geometry["type"] == "MultiPolygon":
            rings = [ring for polygon in geometry["coordinates"] for ring in polygon]
        for ring in rings:
            zone_shapes.append(
                "<polygon "
                f'class="zone zone-{feature["class_id"]}" '
                f'points="{polygon_points(ring)}" '
                f'fill="{colors[feature["class_id"]]}" />'
            )
    zone_legend = "".join(
        f'<span><i style="--swatch:{item["color"]}"></i>{item["label"]}</span>'
        for item in zones["classes"]
    )
    source_uri = "data:image/png;base64," + base64.b64encode(
        ESDF_PNG.read_bytes()
    ).decode("ascii")
    start_x, start_z = route[0]
    goal_x, goal_z = route[-1]
    return f"""
<div class="uf-map-card">
  <style>
    .uf-map-card {{
      --uf-route: var(--viz-series-3, #16a34a);
      --uf-route-soft: color-mix(in srgb, var(--uf-route) 22%, transparent);
      color: var(--color-text-primary);
      background: var(--color-background-primary);
      border: 1px solid var(--color-border-secondary);
      border-radius: 14px;
      padding: 16px;
      font: 13px/1.45 var(--font-sans);
      box-sizing: border-box;
      max-width: 980px;
      margin: 0 auto;
    }}
    .uf-map-head {{ display:flex; gap:12px; align-items:flex-start; justify-content:space-between; margin-bottom:12px; }}
    .uf-map-head h2 {{ font: 650 18px/1.2 var(--font-sans); margin:0 0 4px; color:var(--color-text-primary); }}
    .uf-map-head p {{ margin:0; color:var(--color-text-secondary); }}
    .uf-badges {{ display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }}
    .uf-badge {{ white-space:nowrap; background:var(--color-background-secondary); border:1px solid var(--color-border-secondary); border-radius:999px; padding:4px 8px; color:var(--color-text-secondary); }}
    .uf-stage {{ display:grid; grid-template-columns:minmax(0, 620px) 230px; gap:14px; align-items:start; justify-content:center; }}
    .uf-map-wrap {{ position:relative; aspect-ratio:1/1; overflow:hidden; border-radius:10px; border:1px solid var(--color-border-primary); background:var(--color-background-secondary); }}
    .uf-map-wrap img, .uf-map-wrap svg {{ position:absolute; inset:0; width:100%; height:100%; display:block; }}
    .uf-map-wrap img {{ opacity:.92; }}
    .grid line {{ stroke:var(--color-border-secondary); stroke-width:.8; vector-effect:non-scaling-stroke; }}
    .grid text {{ fill:var(--color-text-secondary); font:16px var(--font-mono); paint-order:stroke; stroke:var(--color-background-primary); stroke-width:4px; }}
    .zones {{ opacity:.19; transition:opacity .2s ease; }}
    .zones .zone {{ stroke:color-mix(in srgb, currentColor 45%, transparent); stroke-width:1; vector-effect:non-scaling-stroke; }}
    .zones.is-hidden {{ opacity:0; }}
    .route-halo {{ fill:none; stroke:var(--color-background-primary); stroke-width:13; stroke-linejoin:round; stroke-linecap:round; opacity:.86; vector-effect:non-scaling-stroke; }}
    .route-full {{ fill:none; stroke:var(--uf-route-soft); stroke-width:9; stroke-linejoin:round; stroke-linecap:round; vector-effect:non-scaling-stroke; }}
    .route-progress {{ fill:none; stroke:var(--uf-route); stroke-width:5; stroke-linejoin:round; stroke-linecap:round; vector-effect:non-scaling-stroke; }}
    .marker circle {{ stroke:var(--color-background-primary); stroke-width:3; vector-effect:non-scaling-stroke; }}
    .marker text {{ fill:var(--color-text-primary); font:700 24px var(--font-sans); text-anchor:middle; dominant-baseline:central; }}
    .uav {{ fill:var(--uf-route); stroke:var(--color-background-primary); stroke-width:4; vector-effect:non-scaling-stroke; filter:drop-shadow(0 2px 3px rgb(0 0 0 / .28)); }}
    .uf-panel {{ display:grid; gap:10px; }}
    .uf-block {{ padding:11px; background:var(--color-background-secondary); border:1px solid var(--color-border-secondary); border-radius:10px; }}
    .uf-block h3 {{ margin:0 0 8px; font:650 13px var(--font-sans); }}
    .uf-kv {{ display:grid; grid-template-columns:1fr auto; gap:5px 8px; color:var(--color-text-secondary); }}
    .uf-kv b {{ color:var(--color-text-primary); font-weight:600; }}
    .uf-slider {{ width:100%; accent-color:var(--uf-route); }}
    .uf-toggle {{ display:flex; align-items:center; gap:7px; cursor:pointer; color:var(--color-text-secondary); }}
    .uf-toggle input {{ accent-color:var(--uf-route); }}
    .uf-legend {{ display:grid; gap:6px; }}
    .uf-legend span {{ display:flex; align-items:center; gap:7px; color:var(--color-text-secondary); }}
    .uf-legend i {{ width:13px; height:13px; border-radius:3px; background:var(--swatch); flex:none; }}
    .uf-legend .line {{ height:4px; border-radius:4px; }}
    .uf-zone-legend {{ display:flex; flex-wrap:wrap; gap:5px 9px; margin-top:8px; }}
    .uf-zone-legend span {{ display:flex; align-items:center; gap:4px; font-size:11px; color:var(--color-text-secondary); }}
    .uf-zone-legend i {{ width:8px; height:8px; border-radius:2px; background:var(--swatch); }}
    .uf-note {{ margin-top:10px; color:var(--color-text-secondary); font-size:11px; }}
    @media (max-width: 760px) {{
      .uf-map-head, .uf-stage {{ display:block; }}
      .uf-badges {{ justify-content:flex-start; margin-top:8px; }}
      .uf-panel {{ grid-template-columns:repeat(2,minmax(0,1fr)); margin-top:12px; }}
    }}
  </style>
  <div class="uf-map-head">
    <div>
      <h2>城市 · 单机世界模型飞行俯视图</h2>
      <p>真实摄影测量碰撞体、净空场、功能分区与可回放航迹叠加</p>
    </div>
    <div class="uf-badges">
      <span class="uf-badge">范围 1 km × 1 km</span>
      <span class="uf-badge">几何源 0.5 m</span>
      <span class="uf-badge">高度切片 20 m</span>
    </div>
  </div>
  <div class="uf-stage">
    <div class="uf-map-wrap">
      <img alt="城市20米高度真实ESDF碰撞与净空底图" src="{source_uri}">
      <svg viewBox="0 0 1000 1000" role="img" aria-label="城市俯视路径图">
        <g class="zones">{"".join(zone_shapes)}</g>
        <g class="grid">
          {"".join(f'<line x1="{v}" y1="0" x2="{v}" y2="1000"/><line x1="0" y1="{v}" x2="1000" y2="{v}"/>' for v in range(100, 1000, 100))}
          <text x="12" y="32">N ↑</text><text x="914" y="982">100 m/grid</text>
        </g>
        <polyline class="route-halo" points="{route_points}"/>
        <polyline class="route-full" points="{route_points}"/>
        <polyline class="route-progress" points="{route_points}"/>
        <g class="marker" transform="translate({start_x + 500.0:.2f} {500.0 - start_z:.2f})">
          <circle r="22" fill="var(--viz-series-1, #60a5fa)"/><text y="1">S</text>
        </g>
        <g class="marker" transform="translate({goal_x + 500.0:.2f} {500.0 - goal_z:.2f})">
          <circle r="22" fill="var(--viz-series-2, #f3c94b)"/><text y="1">D</text>
        </g>
        <path class="uav" d="M 0,-18 L 14,15 L 0,9 L -14,15 Z"/>
      </svg>
    </div>
    <aside class="uf-panel">
      <div class="uf-block">
        <h3>航迹回放</h3>
        <input class="uf-slider" type="range" min="0" max="100" value="68" aria-label="航迹进度">
        <div class="uf-kv"><span>当前位置</span><b class="uf-position">—</b><span>回放进度</span><b class="uf-progress-label">68%</b></div>
      </div>
      <div class="uf-block">
        <h3>图层</h3>
        <label class="uf-toggle"><input class="uf-zone-toggle" type="checkbox" checked> 城市功能分区</label>
      </div>
      <div class="uf-block">
        <h3>图例</h3>
        <div class="uf-legend">
          <span><i style="--swatch:rgb(255 88 55 / .82)"></i>20 m高度碰撞体</span>
          <span><i style="--swatch:rgb(30 200 255 / .50)"></i>可飞净空区域</span>
          <span><i class="line" style="--swatch:var(--uf-route)"></i>碰撞约束规划航迹</span>
        </div>
        <div class="uf-zone-legend">{zone_legend}</div>
      </div>
      <div class="uf-block">
        <h3>数据口径</h3>
        <div class="uf-kv">
          <span>视觉网格</span><b>3,825,064 △</b>
          <span>碰撞网格</span><b>307,980 △</b>
          <span>ESDF单元</span><b>0.5 m</b>
          <span>功能区</span><b>{len(labels)} 类</b>
        </div>
      </div>
    </aside>
  </div>
  <div class="uf-note">底图来自城市真实摄影测量网格的20 m ESDF切片；绿色线为基于同一碰撞场规划的演示航迹，不冒充已训练模型的实际飞行结果。</div>
  <script>
    (() => {{
      const root = document.currentScript.closest('.uf-map-card');
      const slider = root.querySelector('.uf-slider');
      const route = root.querySelector('.route-progress');
      const uav = root.querySelector('.uav');
      const position = root.querySelector('.uf-position');
      const progressLabel = root.querySelector('.uf-progress-label');
      const zoneToggle = root.querySelector('.uf-zone-toggle');
      const zonesLayer = root.querySelector('.zones');
      const total = route.getTotalLength();
      route.style.strokeDasharray = `${{total}} ${{total}}`;
      const render = () => {{
        const value = Number(slider.value) / 100;
        route.style.strokeDashoffset = String(total * (1 - value));
        const point = route.getPointAtLength(Math.max(0, total * value));
        const ahead = route.getPointAtLength(Math.min(total, total * value + 2));
        const angle = Math.atan2(ahead.y - point.y, ahead.x - point.x) * 180 / Math.PI + 90;
        uav.setAttribute('transform', `translate(${{point.x}} ${{point.y}}) rotate(${{angle}})`);
        const x = point.x - 500;
        const z = 500 - point.y;
        position.textContent = `(${{x.toFixed(0)}}, ${{z.toFixed(0)}}) m`;
        progressLabel.textContent = `${{Math.round(value * 100)}}%`;
      }};
      slider.addEventListener('input', render);
      zoneToggle.addEventListener('change', () => zonesLayer.classList.toggle('is-hidden', !zoneToggle.checked));
      render();
    }})();
  </script>
</div>
""".strip()


def main() -> None:
    args = parse_args()
    html = build_html()
    for path in (args.output, args.project_copy):
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
