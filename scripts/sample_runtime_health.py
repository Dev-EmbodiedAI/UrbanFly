"""Save a bounded passive performance sample; never sends simulation controls."""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--samples", default=10, type=int)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
samples = []
for index in range(args.samples):
    with urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=3) as response:
        samples.append(json.load(response))
    if index + 1 < args.samples:
        time.sleep(2)
fps = [surface["presentation"].get("fps", 0) for sample in samples
       for surface in sample["surfaces"] if surface["age_s"] < 5 and surface["scene_ready"]]
summary = {"sample_count": len(samples), "fps_median": statistics.median(fps) if fps else None,
           "fps_min": min(fps) if fps else None, "fps_max": max(fps) if fps else None}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"summary": summary, "samples": samples}, indent=2), encoding="utf-8")
print(json.dumps(summary))
