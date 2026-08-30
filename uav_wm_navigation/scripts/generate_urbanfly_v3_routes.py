from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_wm_navigation.data.splits import load_split_manifest, validate_route_split


DISTANCE_BINS = ((50, 150), (150, 300), (300, 600), (600, 900))
BEHAVIORS = (
    ("geometric_mpc_expert", 0.40), ("perturbed_expert", 0.25),
    ("active_near_miss", 0.20), ("failure_recovery", 0.10),
    ("random_exploration", 0.05),
)


def astar(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    neighbors = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (1, 1, 2**0.5))
    queue = [(0.0, start)]
    cost = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]; path.append(current)
            return path[::-1]
        for dr, dc, step in neighbors:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < mask.shape[0] and 0 <= nxt[1] < mask.shape[1] and mask[nxt]):
                continue
            proposal = cost[current] + step
            if proposal >= cost.get(nxt, float("inf")):
                continue
            cost[nxt], parent[nxt] = proposal, current
            heuristic = np.hypot(nxt[0] - goal[0], nxt[1] - goal[1])
            heapq.heappush(queue, (proposal + heuristic, nxt))
    return None


def point_in_ring(point: np.ndarray, ring: list[list[float]]) -> bool:
    x, y = point; inside = False
    for left, right in zip(ring, ring[1:] + ring[:1]):
        x1, y1 = left; x2, y2 = right
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def zone_at(point: np.ndarray, zoning: dict) -> str:
    for feature in zoning["features"]:
        polygons = [feature["geometry"]["coordinates"]] if feature["geometry"]["type"] == "Polygon" else feature["geometry"]["coordinates"]
        for polygon in polygons:
            if polygon and point_in_ring(point, polygon[0]):
                return str(feature["class_id"])
    return "public_space"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen, ESDF-validated v3 collection routes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--altitude", type=float, choices=(10.0, 20.0, 30.0, 40.0), default=30.0)
    parser.add_argument("--grid-resolution", type=float, default=10.0)
    parser.add_argument(
        "--skip-zone-coverage",
        action="store_true",
        help="Do not force the first routes to cover all eight zone classes. Useful for a split that does not contain every class.",
    )
    args = parser.parse_args()
    split_path = ROOT / "configs/urbanfly_spatial_split_v1.json"
    split_manifest = load_split_manifest(split_path)
    scene_root = ROOT.parent / "data/helsinki_mesh/HelsinkiCentral1km"
    esdf_path = scene_root / f"diagnostics/esdf_{int(args.altitude):03d}m_0p5m.npz"
    esdf_payload = np.load(esdf_path)
    esdf = esdf_payload["signed_distance_m"]
    rng = np.random.default_rng(args.seed)
    resolution = float(args.grid_resolution)
    coordinates = np.arange(-500.0 + resolution / 2, 500.0, resolution)
    mask = np.zeros((len(coordinates), len(coordinates)), dtype=bool)
    allowed_tiles = [tile for tile in split_manifest["tiles"] if tile["split"] == args.split]
    for row, y in enumerate(coordinates):
        for column, x in enumerate(coordinates):
            inside = any(
                np.all(np.asarray([x, y]) >= np.asarray(tile["route_inner_bounds_nwu_m"])[0])
                and np.all(np.asarray([x, y]) <= np.asarray(tile["route_inner_bounds_nwu_m"])[1])
                for tile in allowed_tiles
            )
            esdf_column = int(round((x + 500.0) / 0.5))
            esdf_row = int(round((500.0 - y) / 0.5))
            mask[row, column] = inside and float(esdf[esdf_row, esdf_column]) >= 3.0
    free = np.argwhere(mask)
    zoning = json.loads((scene_root / "zones/functional_zones.json").read_text(encoding="utf-8"))
    required_zones = [item["id"] for item in zoning["classes"]]
    free_zones = np.asarray([zone_at(np.asarray([coordinates[column], coordinates[row]]), zoning) for row, column in free])
    target_per_bin = [args.count // 4 + int(index < args.count % 4) for index in range(4)]
    behavior_pool = [name for name, fraction in BEHAVIORS for _ in range(round(args.count * fraction))]
    while len(behavior_pool) < args.count: behavior_pool.append("geometric_mpc_expert")
    rng.shuffle(behavior_pool)
    routes = []
    for bin_index, ((minimum, maximum), target) in enumerate(zip(DISTANCE_BINS, target_per_bin)):
        created, attempts = 0, 0
        while created < target and attempts < 100_000:
            attempts += 1
            route_index = len(routes)
            requested_zone = (
                required_zones[route_index]
                if not args.skip_zone_coverage and route_index < len(required_zones)
                else None
            )
            zone_options = np.flatnonzero(free_zones == requested_zone) if requested_zone else np.arange(len(free))
            if not len(zone_options):
                raise RuntimeError(f"no ESDF-safe grid cell is available for required zone {requested_zone}")
            start = tuple(map(int, free[rng.choice(zone_options)]))
            euclidean = np.linalg.norm((free - np.asarray(start)) * resolution, axis=1)
            options = np.flatnonzero((euclidean >= minimum * 0.72) & (euclidean <= maximum))
            if not len(options): continue
            goal = tuple(map(int, free[rng.choice(options)]))
            grid_path = astar(mask, start, goal)
            if not grid_path: continue
            length = sum(np.hypot(b[0] - a[0], b[1] - a[1]) * resolution for a, b in zip(grid_path, grid_path[1:]))
            if not minimum <= length <= maximum: continue
            dense = np.asarray([[coordinates[column], coordinates[row], args.altitude] for row, column in grid_path], dtype=np.float32)
            tiles = validate_route_split(dense, args.split, split_manifest)
            indices = sorted(set([0, *range(5, len(dense) - 1, 5), len(dense) - 1]))
            waypoints = dense[indices]
            path_zones = sorted({zone_at(point[:2], zoning) for point in dense[::max(1, len(dense) // 40)]})
            routes.append({
                "route_id": f"{args.split}-{route_index:04d}", "split": args.split,
                "seed": int(args.seed + route_index), "distance_bin_m": [minimum, maximum],
                "shortest_path_m": float(length), "altitude_m": args.altitude,
                "behavior": behavior_pool[route_index], "tile_ids": list(tiles),
                "zone_type": requested_zone or zone_at(dense[len(dense) // 2, :2], zoning),
                "zone_types": path_zones,
                "start_nwu_m": dense[0].tolist(), "goal_nwu_m": dense[-1].tolist(),
                "route_nwu_m": waypoints.tolist(), "esdf_clearance_min_m": float(min(
                    esdf[int(round((500.0 - point[1]) / 0.5)), int(round((point[0] + 500.0) / 0.5))] for point in dense
                )),
            })
            created += 1
        if created != target:
            raise RuntimeError(f"could not create distance bin {minimum}-{maximum}: {created}/{target}")
    payload = {
        "schema": "urbanfly-route-manifest-v3", "seed": args.seed,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "esdf_source": str(esdf_path.resolve()), "routes": routes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "routes": len(routes), "sha256": payload["manifest_sha256"]}))


if __name__ == "__main__":
    main()
