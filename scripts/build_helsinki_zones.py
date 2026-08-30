#!/usr/bin/env python3
"""Download and localize official Helsinki functional zoning data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


WFS_URL = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
PLAN_UNIT_LAYER = "avoindata:Kaavayksikot"
MASTER_PLAN_LAYER = "avoindata:Yleiskaava2016_100m_ruudut"

FUNCTION_CLASSES = {
    "AK": {
        "id": "residential",
        "label": "住宅区",
        "color": "#62b7ff",
        "flight_cost": 1.25,
        "policy": "居民敏感区，降低噪声并提高最低巡航高度",
    },
    "K": {
        "id": "commercial_office",
        "label": "商业/办公区",
        "color": "#ffb45f",
        "flight_cost": 1.0,
        "policy": "配送需求密集区，允许常规航路",
    },
    "Y": {
        "id": "public_service",
        "label": "公共服务区",
        "color": "#b68cff",
        "flight_cost": 1.4,
        "policy": "学校、医疗及公共设施候选敏感区",
    },
    "V": {
        "id": "green_recreation",
        "label": "绿地/休闲区",
        "color": "#59d58b",
        "flight_cost": 1.15,
        "policy": "休闲活动区，限制低空盘旋",
    },
    "L1": {
        "id": "transport",
        "label": "交通区",
        "color": "#f7df6e",
        "flight_cost": 0.85,
        "policy": "可作为优先走廊候选，但需避让道路设施",
    },
    "W": {
        "id": "water",
        "label": "水域",
        "color": "#38c9e8",
        "flight_cost": 0.95,
        "policy": "低地面人员风险，但考虑风场与迫降风险",
    },
    "ET": {
        "id": "special",
        "label": "特殊用途区",
        "color": "#ff6e78",
        "flight_cost": 1.6,
        "policy": "默认高代价，任务规划前需人工复核",
    },
    "L": {
        "id": "public_space",
        "label": "公共空间",
        "color": "#b6c5cf",
        "flight_cost": 1.05,
        "policy": "公共开放空间，按人群密度动态调整代价",
    },
}

FALLBACK_CLASS = {
    "id": "other",
    "label": "其他规划用途",
    "color": "#9aa5ad",
    "flight_cost": 1.2,
    "policy": "未分类规划单元，使用保守默认代价",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    return parser.parse_args()


def request_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_geojson(
    session: requests.Session,
    layer: str,
    bbox: tuple[float, float, float, float],
) -> dict:
    bbox_text = ",".join(str(value) for value in bbox) + ",EPSG:3879"
    response = session.get(
        WFS_URL,
        params={
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": layer,
            "outputFormat": "application/json",
            "srsName": "EPSG:3879",
            "bbox": bbox_text,
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def clip_ring(
    ring: list[list[float]],
    bbox: tuple[float, float, float, float],
) -> list[list[float]]:
    minimum_x, minimum_y, maximum_x, maximum_y = bbox

    def clip(
        points: list[list[float]],
        inside: Callable[[list[float]], bool],
        intersection: Callable[[list[float], list[float]], list[float]],
    ) -> list[list[float]]:
        if not points:
            return []
        output: list[list[float]] = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    def intersect_x(
        a: list[float],
        b: list[float],
        x: float,
    ) -> list[float]:
        ratio = 0.0 if b[0] == a[0] else (x - a[0]) / (b[0] - a[0])
        return [x, a[1] + ratio * (b[1] - a[1])]

    def intersect_y(
        a: list[float],
        b: list[float],
        y: float,
    ) -> list[float]:
        ratio = 0.0 if b[1] == a[1] else (y - a[1]) / (b[1] - a[1])
        return [a[0] + ratio * (b[0] - a[0]), y]

    points = [list(point[:2]) for point in ring]
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    points = clip(
        points,
        lambda point: point[0] >= minimum_x,
        lambda a, b: intersect_x(a, b, minimum_x),
    )
    points = clip(
        points,
        lambda point: point[0] <= maximum_x,
        lambda a, b: intersect_x(a, b, maximum_x),
    )
    points = clip(
        points,
        lambda point: point[1] >= minimum_y,
        lambda a, b: intersect_y(a, b, minimum_y),
    )
    points = clip(
        points,
        lambda point: point[1] <= maximum_y,
        lambda a, b: intersect_y(a, b, maximum_y),
    )
    if len(points) < 3:
        return []
    points.append(points[0])
    return points


def clip_geometry(geometry: dict, bbox: tuple[float, float, float, float]) -> dict | None:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        rings = [clip_ring(ring, bbox) for ring in coordinates]
        rings = [ring for ring in rings if ring]
        return {"type": "Polygon", "coordinates": rings} if rings else None
    if geometry_type == "MultiPolygon":
        polygons = []
        for polygon in coordinates:
            rings = [clip_ring(ring, bbox) for ring in polygon]
            rings = [ring for ring in rings if ring]
            if rings:
                polygons.append(rings)
        return (
            {"type": "MultiPolygon", "coordinates": polygons}
            if polygons
            else None
        )
    return None


def localize_geometry(
    geometry: dict,
    center_x: float,
    center_y: float,
) -> dict:
    def transform(value: Any) -> Any:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            return [
                round(float(value[0]) - center_x, 3),
                round(center_y - float(value[1]), 3),
            ]
        if isinstance(value, list):
            return [transform(item) for item in value]
        return value

    return {
        "type": geometry["type"],
        "coordinates": transform(geometry["coordinates"]),
    }


def ring_area(ring: list[list[float]]) -> float:
    return abs(
        sum(
            ring[index][0] * ring[index + 1][1]
            - ring[index + 1][0] * ring[index][1]
            for index in range(len(ring) - 1)
        )
        * 0.5
    )


def geometry_area(geometry: dict) -> float:
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    return sum(
        max(
            0.0,
            ring_area(polygon[0])
            - sum(ring_area(ring) for ring in polygon[1:]),
        )
        for polygon in polygons
        if polygon
    )


def main() -> None:
    args = parse_args()
    scene = args.scene.resolve()
    manifest_path = scene / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    origin_x, origin_y = manifest["original_bounds"]["minimum"]
    maximum_x, maximum_y = manifest["original_bounds"]["maximum"]
    srs_origin_x, srs_origin_y, _ = manifest["source"]["srs_origin"]
    bbox = (
        srs_origin_x + origin_x,
        srs_origin_y + origin_y,
        srs_origin_x + maximum_x,
        srs_origin_y + maximum_y,
    )
    center_x = srs_origin_x + manifest["local_frame"]["center_original_xy"][0]
    center_y = srs_origin_y + manifest["local_frame"]["center_original_xy"][1]

    session = request_session()
    plan_units = fetch_geojson(session, PLAN_UNIT_LAYER, bbox)
    master_plan = fetch_geojson(session, MASTER_PLAN_LAYER, bbox)

    features = []
    class_counts: Counter[str] = Counter()
    class_area: defaultdict[str, float] = defaultdict(float)
    used_classes: dict[str, dict] = {}
    for source_feature in plan_units.get("features", []):
        clipped = clip_geometry(source_feature.get("geometry") or {}, bbox)
        if clipped is None:
            continue
        localized = localize_geometry(clipped, center_x, center_y)
        properties = source_feature.get("properties") or {}
        code = properties.get("kayttotarkoitusluokka_koodi") or ""
        category = FUNCTION_CLASSES.get(code, FALLBACK_CLASS)
        area = geometry_area(localized)
        if area < 0.5:
            continue
        used_classes[category["id"]] = category
        class_counts[category["id"]] += 1
        class_area[category["id"]] += area
        features.append(
            {
                "id": source_feature.get("id"),
                "class_id": category["id"],
                "source_code": code,
                "source_label": properties.get("kayttotarkoitusluokka"),
                "area_m2": round(area, 2),
                "address": properties.get("osoite"),
                "geometry": localized,
            }
        )

    master_features = []
    for source_feature in master_plan.get("features", []):
        clipped = clip_geometry(source_feature.get("geometry") or {}, bbox)
        if clipped is None:
            continue
        properties = source_feature.get("properties") or {}
        localized = localize_geometry(clipped, center_x, center_y)
        master_features.append(
            {
                "id": source_feature.get("id"),
                "source_code": properties.get("kayttark"),
                "source_label": properties.get("kayttark_selite"),
                "geometry": localized,
            }
        )

    classes = []
    for category in FUNCTION_CLASSES.values():
        if category["id"] not in used_classes:
            continue
        classes.append(
            {
                **category,
                "feature_count": class_counts[category["id"]],
                "area_m2": round(class_area[category["id"]], 2),
            }
        )
    if FALLBACK_CLASS["id"] in used_classes:
        classes.append(
            {
                **FALLBACK_CLASS,
                "feature_count": class_counts[FALLBACK_CLASS["id"]],
                "area_m2": round(class_area[FALLBACK_CLASS["id"]], 2),
            }
        )

    output = {
        "schema_version": 1,
        "scene_name": manifest["scene_name"],
        "source": {
            "provider": "City of Helsinki, City Survey Services",
            "license": "CC BY 4.0",
            "wfs": WFS_URL,
            "layers": [PLAN_UNIT_LAYER, MASTER_PLAN_LAYER],
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "note": "几何与用途代码来自官方数据；flight_cost 和 policy 是 UrbanFly 仿真默认策略，不是现实法规。",
        },
        "coordinate_frame": "UrbanFly local X/Z metres",
        "extent_m": {
            "minimum": [
                -manifest["operation_size_m"] / 2,
                -manifest["operation_size_m"] / 2,
            ],
            "maximum": [
                manifest["operation_size_m"] / 2,
                manifest["operation_size_m"] / 2,
            ],
        },
        "classes": classes,
        "features": features,
        "master_plan_context": master_features,
    }
    output_dir = scene / "zones"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "functional_zones.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest["zoning"] = {
        "uri": "zones/functional_zones.json",
        "source": "City of Helsinki official WFS",
        "license": "CC BY 4.0",
        "feature_count": len(features),
        "classes": [
            {
                "id": item["id"],
                "label": item["label"],
                "feature_count": item["feature_count"],
                "area_m2": item["area_m2"],
            }
            for item in classes
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["zoning"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
