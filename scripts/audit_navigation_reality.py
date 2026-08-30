"""Fail-closed audit and top-down visualization for UrbanFly navigation assets.

This script deliberately does not synthesize a map.  It follows the same
asset precedence as ``backend.server.server.main`` and only reports/plots
files that are actually present on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.engine.collision import (  # noqa: E402
    DenseSignedDistanceField,
    HeightmapStaticCollisionMap,
    HierarchicalStaticCollisionMap,
    SparseStaticCollisionMap,
)


def _exists(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}


def _load_real_collision_map() -> tuple[object | None, dict[str, object]]:
    data = ROOT / "data"
    helsinki = data / "helsinki_mesh" / "HelsinkiCentral1km"
    manifest_path = helsinki / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        heightmap_path = helsinki / manifest["collision"]["heightmap"]["uri"]
        if heightmap_path.is_file():
            value = HeightmapStaticCollisionMap.load(heightmap_path)
            return value, {
                "runtime_selection": "helsinki_conservative_heightmap",
                "source": str(heightmap_path.resolve()),
                "resolution_m": value.resolution,
                "shape": list(value.shape),
                "signed": False,
                "three_dimensional": False,
            }

    city = data / "citygs_collision" / "Residence"
    global_path = city / "global_esdf.npz"
    local_path = city / "local_collision_sparse.npz"
    if global_path.is_file() and local_path.is_file():
        global_esdf = DenseSignedDistanceField.load(global_path)
        local = SparseStaticCollisionMap.load(local_path)
        value = HierarchicalStaticCollisionMap(global_esdf, local)
        return value, {
            "runtime_selection": "citygs_hierarchical_esdf",
            "source": [str(global_path.resolve()), str(local_path.resolve())],
            "resolution_m": {"global": global_esdf.resolution, "local": local.resolution},
            "shape": list(global_esdf.shape),
            "signed": True,
            "three_dimensional": True,
        }
    return None, {
        "runtime_selection": "none",
        "source": None,
        "signed": False,
        "three_dimensional": False,
        "reason": "configured collision assets are absent; backend runs without a static scene",
    }


def _route(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("route_nwu_m", payload.get("route_nwu", payload.get("path")))
    if points is None:
        raise ValueError("route JSON needs route_nwu_m, route_nwu, or path")
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 2:
        raise ValueError("route must contain at least two 3-D points")
    return value


def _visualize(collision_map: object | None, metadata: dict[str, object], route: np.ndarray | None, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    if isinstance(collision_map, HeightmapStaticCollisionMap):
        extent = [
            collision_map.origin_x,
            collision_map.origin_x + collision_map.shape[1] * collision_map.resolution,
            collision_map.maximum_z - collision_map.shape[0] * collision_map.resolution,
            collision_map.maximum_z,
        ]
        image = axis.imshow(collision_map.height, origin="lower", extent=extent, cmap="terrain")
        figure.colorbar(image, ax=axis, label="highest collision surface (m)")
        axis.set_title("Runtime collision heightmap (real on-disk asset)")
    elif isinstance(collision_map, HierarchicalStaticCollisionMap):
        field = collision_map.global_esdf
        altitude = route[0, 1] if route is not None else float(field.origin[1] + field.shape[1] * field.resolution * 0.5)
        y_index = int(np.clip(np.floor((altitude - field.origin[1]) / field.resolution), 0, field.shape[1] - 1))
        values = field.distance[:, y_index, :].astype(np.float32).T
        extent = [field.origin[0], field.origin[0] + field.shape[0] * field.resolution,
                  field.origin[2], field.origin[2] + field.shape[2] * field.resolution]
        image = axis.imshow(values, origin="lower", extent=extent, cmap="RdYlBu", vmin=-8, vmax=8)
        axis.contour(values, levels=[0.0], colors="black", linewidths=0.7,
                     extent=extent, origin="lower")
        figure.colorbar(image, ax=axis, label="signed distance (m)")
        axis.set_title(f"Runtime global ESDF slice at y={altitude:.1f} m")
    else:
        axis.set_facecolor("#fff4f4")
        axis.text(0.5, 0.58, "NO RUNTIME MAP ASSET", ha="center", va="center",
                  transform=axis.transAxes, fontsize=22, weight="bold", color="#b42318")
        axis.text(0.5, 0.43, str(metadata.get("reason", "map unavailable")), ha="center", va="center",
                  transform=axis.transAxes, fontsize=11, wrap=True)
        axis.set_title("UrbanFly map/debug visualization: verification blocked")
        axis.set_xlim(-1, 1)
        axis.set_ylim(-1, 1)
    if route is not None:
        axis.plot(route[:, 0], route[:, 2], color="#2e5aac", linewidth=2.0, label="planned path")
        axis.scatter(route[0, 0], route[0, 2], c="#00a878", s=80, label="start", zorder=5)
        axis.scatter(route[-1, 0], route[-1, 2], c="#d62828", s=90, marker="*", label="goal", zorder=5)
        axis.legend(loc="best")
    axis.set_xlabel("world x / north (m)")
    axis.set_ylabel("world z / west (m)")
    axis.set_aspect("equal", adjustable="box")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict runtime map/planner asset audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "navigation_reality_audit")
    parser.add_argument("--route-json", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    collision_map, map_metadata = _load_real_collision_map()
    route = _route(args.route_json)
    city = ROOT / "data" / "citygs_collision" / "Residence"
    helsinki = ROOT / "data" / "helsinki_mesh" / "HelsinkiCentral1km"
    yopo_root = ROOT / "YOPO-YOPO-Simple" / "YOPO"
    checkpoint = yopo_root / "saved" / "urbanfly" / "YOPO_0" / "epoch24.pth"
    report = {
        "verdict": "RUNTIME_GEOMETRY_AVAILABLE" if collision_map is not None else "RUNTIME_GEOMETRY_MISSING",
        "map": map_metadata,
        "required_assets": {
            "helsinki_manifest": _exists(helsinki / "manifest.json"),
            "citygs_global_esdf": _exists(city / "global_esdf.npz"),
            "citygs_local_surface": _exists(city / "local_collision_sparse.npz"),
            "citygs_collision_mesh": _exists(city / "city_collision.glb"),
            "yopo_source": {"path": str(yopo_root.resolve()), "exists": yopo_root.is_dir()},
            "yopo_checkpoint": _exists(checkpoint),
        },
        "planner_runtime": {
            "backend_global_planner_available": (ROOT / "data" / "scene" / "occupancy_grid.npz").is_file(),
            "backend_server_global_planner_for_helsinki": False,
            "yopo_teacher_loadable": checkpoint.is_file() and yopo_root.is_dir(),
        },
        "tree_safety": {
            "explicit_tree_collision_layer": False,
            "verified_in_runtime_map": False,
            "verdict": "TREE SAFETY NOT CURRENTLY RELIABLE",
        },
        "visualization": str((output / "map_path_debug.png").resolve()),
    }
    _visualize(collision_map, map_metadata, route, output / "map_path_debug.png")
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if collision_map is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
