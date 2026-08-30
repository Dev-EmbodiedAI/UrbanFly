#!/usr/bin/env python
"""GPU benchmark Qwen3-VL on real canonical Helsinki RGB-D negative clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.semantic_fleet import (  # noqa: E402
    OpenAICompatibleQwenVLClient,
    SemanticEvent,
    SemanticEventGate,
)


MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgbd_montage(rgb: np.ndarray, depth_m: np.ndarray) -> Image.Image:
    rgb_image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    clip_m = float(np.percentile(depth_m[valid], 95)) if valid.any() else 120.0
    clip_m = max(1.0, min(120.0, clip_m))
    depth_gray = np.zeros(depth_m.shape, dtype=np.uint8)
    depth_gray[valid] = np.clip(
        255.0 * (1.0 - depth_m[valid] / clip_m), 0.0, 255.0
    ).astype(np.uint8)
    depth_image = ImageOps.colorize(
        Image.fromarray(depth_gray, mode="L"),
        black="#06131d",
        white="#ffcf66",
    )
    montage = Image.new("RGB", (rgb_image.width * 2, rgb_image.height), "black")
    montage.paste(rgb_image, (0, 0))
    montage.paste(depth_image, (rgb_image.width, 0))
    return montage.resize((640, 180), Image.Resampling.NEAREST)


def load_clip(path: Path, history_frames: int = 4) -> tuple[list[Image.Image], dict]:
    with h5py.File(path, "r") as handle:
        count = int(handle["observations/rgb_front"].shape[0])
        indices = np.linspace(0, count - 1, history_frames).round().astype(int)
        images = [
            rgbd_montage(
                handle["observations/rgb_front"][index],
                handle["observations/depth_front"][index],
            )
            for index in indices
        ]
        telemetry = {
            "episode": path.stem,
            "frame_indices": indices.tolist(),
            "sim_times_s": [
                float(handle["timestamps/sim"][index]) for index in indices
            ],
            "positions_world_enu": [
                np.asarray(handle["state/position_world"][index], dtype=float).tolist()
                for index in indices
            ],
        }
    return images, telemetry


def prompt(telemetry: dict) -> str:
    return (
        "You are a conservative semantic safety observer for a UAV fleet. "
        "Each image is a time-ordered RGB view on the left and aligned depth "
        "visualization on the right. Report only a NEW event when evidence is strong. "
        "Static buildings, trees, roads, parked objects, shadows, and ordinary depth "
        "edges are NOT temporary obstacles. A temporary obstacle requires change "
        "corroborated in at least two frames. A no-fly zone requires an authoritative "
        "notice; weather and drone failure require matching telemetry. "
        "Return exactly one JSON object with key events and no markdown. "
        "Allowed event_type values: temporary_obstacle, no_fly_zone, "
        "weather_hazard, drone_failure, goal_landmark. Every event requires "
        "position [east,up,north] metres, numeric radius_m, numeric confidence "
        "in [0,1], numeric severity in [0,1], numeric ttl_s, "
        "evidence and affected_task_ids. Independent support fields are computed by the "
        "backend and must not be invented. Keep evidence under 20 words and the entire "
        "response concise. "
        "Never output a flight command. "
        "If evidence is insufficient return {\"events\":[]}. Telemetry="
        + json.dumps(telemetry, separators=(",", ":"))
    )


def infer(model, processor, images: list[Image.Image], text: str) -> tuple[str, float]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    result = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return result.strip(), latency_ms


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal Qwen benchmark")
    episode_paths = sorted(args.dataset.glob("episodes/*.h5"))[: args.samples]
    if len(episode_paths) != args.samples:
        raise RuntimeError(f"requested {args.samples} clips, found {len(episode_paths)}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model_load_s = time.perf_counter() - load_started
    gate = SemanticEventGate()
    clips = []
    for path in episode_paths:
        images, telemetry = load_clip(path)
        raw, latency_ms = infer(model, processor, images, prompt(telemetry))
        parsed = None
        json_error = None
        schema_error = None
        events = []
        decisions = []
        raw_proposal_count = 0
        try:
            parsed = OpenAICompatibleQwenVLClient._parse_json_object(raw)
            proposals = parsed.get("events", [])
            if not isinstance(proposals, list):
                raise ValueError("events is not a list")
            raw_proposal_count = len(proposals)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            json_error = repr(error)
            proposals = []
        try:
            for proposal in proposals:
                # Match production: generative self-claims can never satisfy
                # independent RGB-D/telemetry/authority corroboration.
                proposal = {
                    **proposal,
                    "temporal_support_count": 0,
                    "depth_support": False,
                    "telemetry_support": False,
                    "authoritative_notice": False,
                }
                event = SemanticEvent.from_mapping(
                    proposal,
                    default_timestamp_s=telemetry["sim_times_s"][-1],
                    default_source_drone_id="QWEN-BENCH",
                )
                events.append(event.to_dict())
                decision = gate.validate(
                    event,
                    now_s=telemetry["sim_times_s"][-1],
                    known_drone_ids={"QWEN-BENCH"},
                )
                decisions.append({
                    "event_id": event.event_id,
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                })
        except (KeyError, TypeError, ValueError) as error:
            schema_error = repr(error)
        clips.append({
            "episode": path.stem,
            "latency_ms": latency_ms,
            "json_valid": json_error is None,
            "json_error": json_error,
            "schema_valid": json_error is None and schema_error is None,
            "schema_error": schema_error,
            "raw_proposal_count": raw_proposal_count,
            "event_count": len(events),
            "events": events,
            "gate_decisions": decisions,
            "raw_response": raw,
        })
    latencies = np.asarray([item["latency_ms"] for item in clips], dtype=float)
    false_positive_clips = sum(item["raw_proposal_count"] > 0 for item in clips)
    accepted_false_positive_clips = sum(
        any(decision["accepted"] for decision in item["gate_decisions"])
        for item in clips
    )
    perception_pass = all(item["schema_valid"] for item in clips) and false_positive_clips == 0
    safety_gate_pass = accepted_false_positive_clips == 0
    report = {
        "benchmark": "urbanfly_qwen3_vl_2b_real_helsinki_negative_v1",
        "status": "PASS" if perception_pass and safety_gate_pass else "FAIL",
        "perception_status": "PASS" if perception_pass else "FAIL",
        "safety_gate_status": "PASS" if safety_gate_pass else "FAIL",
        "scope": "real canonical Helsinki RGB-D negative clips; no dynamic event positives",
        "explicitly_not_tested": [
            "positive-event precision/recall/F1",
            "live WebSocket inference",
            "closed-loop Qwen-triggered replanning",
        ],
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": sha256(args.model / "model.safetensors"),
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "model_load_s": model_load_s,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "clips": len(clips),
        "frames_per_clip": 4,
        "json_valid_clips": sum(item["json_valid"] for item in clips),
        "schema_valid_clips": sum(item["schema_valid"] for item in clips),
        "false_positive_clips": false_positive_clips,
        "accepted_false_positive_clips": accepted_false_positive_clips,
        "latency_ms": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
        "results": clips,
    }
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--model", type=Path, default=ROOT / "models" / "qwen3_vl_2b_instruct"
    )
    result.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "outputs" / "helsinki_dataset_v1" / "main_100_zero_stale_v1",
    )
    result.add_argument("--samples", type=int, default=5)
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "qwen_fleet_system_v1" / "qwen_negative_perception_qa.json",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "clips": report["clips"],
        "json_valid_clips": report["json_valid_clips"],
        "false_positive_clips": report["false_positive_clips"],
        "latency_ms": report["latency_ms"],
        "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
