#!/usr/bin/env python
"""Ask Qwen API for a gated, low-rate waypoint order.

The public/release path uses an OpenAI-compatible Qwen endpoint and never
stores API keys. An explicit ``--direct-model`` remains available only for
reproducing historical offline experiments; it is never bundled or pushed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib import request


def _completion_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    parsed = json.loads(match.group(0) if match else raw)
    if not isinstance(parsed, dict):
        raise ValueError("Qwen mission response must be one JSON object")
    return parsed


def _call_api(*, endpoint: str, api_key: str, model: str, prompt: str,
              timeout_s: float) -> tuple[dict, float]:
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    call = request.Request(
        _completion_url(endpoint),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(call, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000.0
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return _extract_json(str(content)), latency_ms


def _call_local(model_path: Path, prompt: str) -> tuple[dict, float, float, str]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    started_load = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    load_s = time.perf_counter() - started_load
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    raw = processor.batch_decode(
        generated[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    digest = hashlib.sha256((model_path / "model.safetensors").read_bytes()).hexdigest()
    return _extract_json(raw), latency_ms, load_s, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--candidate", action="append", required=True,
                        help="LABEL:EAST:SOUTH:ALTITUDE")
    parser.add_argument("--monotonic-east", action="store_true")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get(
            "URBANFLY_QWEN_ENDPOINT",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("URBANFLY_QWEN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY"),
    )
    parser.add_argument("--api-model", default="qwen-plus")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--direct-model", type=Path, default=None)
    args = parser.parse_args()

    candidates: dict[str, list[float]] = {}
    for item in args.candidate:
        label, east, south, altitude = item.split(":")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", label):
            raise ValueError(f"invalid waypoint label: {label}")
        candidates[label] = [float(east), float(altitude), float(south)]
    labels = list(candidates)
    prompt = (
        "You are the slow semantic mission planner for a Helsinki UAV digital twin. "
        "You cannot issue flight-control actions. Choose an order for every supplied "
        "waypoint exactly once to make a purposeful non-straight inspection route. "
        + ("The east coordinate must strictly increase at every waypoint. "
           if args.monotonic_east else "")
        + "Return JSON only with keys waypoint_order and rationale; rationale must be "
        "at most 20 words. Mission: " + args.request
        + "\nCandidates in backend [east, up, south] metres: "
        + json.dumps(candidates, separators=(",", ":"))
    )

    if args.direct_model is not None:
        parsed, latency_ms, load_s, digest = _call_local(args.direct_model, prompt)
        provider, api_called, model_name = "local_transformers", False, "Qwen/Qwen3-VL-2B-Instruct"
    else:
        if not args.api_key:
            raise RuntimeError(
                "Qwen API key is not configured; set DASHSCOPE_API_KEY or "
                "URBANFLY_QWEN_API_KEY. Keys are never written to reports."
            )
        parsed, latency_ms = _call_api(
            endpoint=args.endpoint, api_key=args.api_key, model=args.api_model,
            prompt=prompt, timeout_s=args.timeout_s,
        )
        provider, api_called, model_name = "qwen_openai_compatible_api", True, args.api_model
        load_s, digest = None, None

    order = parsed.get("waypoint_order")
    gate_pass = (
        isinstance(order, list)
        and len(order) == len(labels)
        and set(order) == set(labels)
        and all(isinstance(item, str) for item in order)
    )
    if gate_pass and args.monotonic_east:
        east_values = [candidates[item][0] for item in order]
        gate_pass = all(a < b for a, b in zip(east_values[:-1], east_values[1:]))
    if not gate_pass:
        raise RuntimeError(f"Qwen waypoint proposal failed deterministic gate: {parsed!r}")

    report = {
        "schema": "urbanfly-qwen-mission-waypoint-plan-v2",
        "status": "PASS",
        "provider": provider,
        "api_called": api_called,
        "api_endpoint": args.endpoint if api_called else None,
        "api_key_stored": False,
        "model": model_name,
        "model_weight_sha256": digest,
        "control_authority": "semantic waypoint ordering only; no flight actions",
        "request": args.request,
        "candidates_backend": candidates,
        "waypoint_order": order,
        "ordered_waypoints_backend": [candidates[label] for label in order],
        "rationale": str(parsed.get("rationale", ""))[:500],
        "deterministic_gate": "PASS",
        "monotonic_east_gate": bool(args.monotonic_east),
        "model_load_s": load_s,
        "inference_latency_ms": latency_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
