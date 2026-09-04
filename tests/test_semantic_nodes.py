from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.server.semantic_nodes import SemanticNodeStore, validate_document


def test_semantic_node_document_is_normalized_and_atomic():
    with tempfile.TemporaryDirectory(dir=Path.cwd() / "tmp") as directory:
        path = Path(directory) / "semantic" / "helsinki_business_nodes.json"
        store = SemanticNodeStore(path)
        saved = store.save({
            "nodes": [{
                "id": "送货点_001",
                "type": "delivery",
                "position": [1, 2, 3],
            }]
        })
        assert saved["coordinate_frame"] == "world_enu"
        assert saved["nodes"][0]["qa_status"] == "UNCHECKED"
        assert json.loads(path.read_text(encoding="utf-8"))["nodes"][0]["position"] == [1.0, 2.0, 3.0]
        assert store.load() == saved


def test_semantic_node_document_rejects_duplicates_and_limits():
    with pytest.raises(ValueError, match="unique"):
        validate_document({"nodes": [
            {"id": "供货点_001", "type": "supply", "position": [0, 0, 0]},
            {"id": "供货点_001", "type": "supply", "position": [1, 0, 0]},
        ]})
    with pytest.raises(ValueError, match="limit"):
        validate_document({"nodes": [
            {"id": f"补给点_{index:03d}", "type": "resupply", "position": [index, 0, 0]}
            for index in range(1, 22)
        ]})
