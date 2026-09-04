"""Persistent storage and validation for Helsinki business semantic nodes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "urbanfly-business-nodes-v1"
SCENE = "HelsinkiCentral1km"
COORDINATE_FRAME = "world_enu"
AXIS_ORDER = ["east", "up", "north"]
NODE_LIMITS = {
    "supply": 100,
    "delivery": 100,
    "resupply": 20,
    "drone_origin": 20,
}


def empty_document() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "scene": SCENE,
        "coordinate_frame": COORDINATE_FRAME,
        "axis_order": AXIS_ORDER,
        "nodes": [],
    }


def validate_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("semantic node document must be an object")
    nodes = document.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("semantic node nodes must be a list")
    normalized = empty_document()
    seen_ids: set[str] = set()
    counts = {node_type: 0 for node_type in NODE_LIMITS}
    for item in nodes:
        if not isinstance(item, dict):
            raise ValueError("semantic node must be an object")
        node_id = str(item.get("id", ""))
        node_type = str(item.get("type", ""))
        position = item.get("position")
        if not node_id or node_id in seen_ids:
            raise ValueError("semantic node ids must be non-empty and unique")
        if node_type not in NODE_LIMITS:
            raise ValueError(f"unsupported semantic node type: {node_type}")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"invalid position for semantic node: {node_id}")
        try:
            coordinates = [float(value) for value in position]
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid position for semantic node: {node_id}") from error
        if not all(value == value and abs(value) != float("inf") for value in coordinates):
            raise ValueError(f"invalid position for semantic node: {node_id}")
        counts[node_type] += 1
        if counts[node_type] > NODE_LIMITS[node_type]:
            raise ValueError(f"semantic node limit exceeded for type: {node_type}")
        seen_ids.add(node_id)
        normalized["nodes"].append({
            "id": node_id,
            "type": node_type,
            "position": coordinates,
            "qa_status": str(item.get("qa_status", "UNCHECKED")),
        })
    return normalized


class SemanticNodeStore:
    """Small atomic JSON store used by the local UrbanFly server."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return empty_document()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return validate_document(json.load(handle))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"cannot load semantic nodes: {error}") from error

    def save(self, document: Any) -> dict[str, Any]:
        normalized = validate_document(document)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(normalized, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return normalized
