"""Collector-only fail-closed regression; no simulator is started."""
import importlib.util
from pathlib import Path
import sys

import pytest


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("helsinki_collection_guard", scripts / "collect_helsinki_dataset_v1.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def valid_record():
    return {"hdf5_readback": "PASS", "integrity_checks": {"timestamps": True},
            "reset_evidence": {"action_buffer_reset": True}}


def test_closed_episode_passes(tmp_path):
    collector._require_episode_integrity(valid_record(), tmp_path)


@pytest.mark.parametrize("failure", ["readback", "schema", "missing_checks", "reset", "partial"])
def test_bad_episode_stops_before_next_reset(tmp_path, failure):
    record = valid_record()
    if failure == "readback": record["hdf5_readback"] = "FAIL"
    if failure == "schema": record["integrity_checks"]["timestamps"] = False
    if failure == "missing_checks": record["integrity_checks"] = {}
    if failure == "reset": record["reset_evidence"]["action_buffer_reset"] = False
    if failure == "partial": (tmp_path / "episode.h5.partial").touch()
    with pytest.raises(RuntimeError, match="FAIL CLOSED"):
        collector._require_episode_integrity(record, tmp_path)
