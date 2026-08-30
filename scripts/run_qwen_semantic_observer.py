#!/usr/bin/env python
"""Run Qwen-VL as a slow, non-controlling semantic observer for UrbanFly.

The process subscribes to synchronized browser RGB-D packets, builds a compact
RGB/depth montage history, calls either the pinned local Transformers model or
an OpenAI-compatible Qwen endpoint, and sends only structured event proposals
back to the deterministic backend gate. It never sends flight-control actions
and writes no model or image artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import defaultdict, deque
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import zlib

import aiohttp
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.semantic_fleet import (  # noqa: E402
    ObservationPacket,
    OpenAICompatibleQwenVLClient,
    SemanticEvent,
)


class DirectTransformersQwenVLClient:
    """Single-process local Qwen client for Windows/laptop deployments."""

    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("direct Qwen observer requires CUDA")
        self.torch = torch
        self.model_name = "Qwen/Qwen3-VL-2B-Instruct"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            Path(model_path),
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            Path(model_path), local_files_only=True
        )
        self.last_latency_ms = None

    @staticmethod
    def _image(data_url: str) -> Image.Image:
        prefix, encoded = data_url.split(",", 1)
        if not prefix.startswith("data:image/"):
            raise ValueError("Qwen frame must be an image data URL")
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

    def analyze(self, packet: ObservationPacket) -> list[SemanticEvent]:
        images = [self._image(item) for item in packet.frame_data_urls[-4:]]
        content = [{"type": "image", "image": image} for image in images]
        content.append({
            "type": "text",
            "text": OpenAICompatibleQwenVLClient._prompt(packet),
        })
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                use_cache=True,
            )
        self.torch.cuda.synchronize()
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        raw = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = OpenAICompatibleQwenVLClient._parse_json_object(raw)
        proposals = parsed.get("events", [])
        if not isinstance(proposals, list):
            raise ValueError("Qwen response field 'events' must be a list")
        return [
            SemanticEvent.from_mapping(
                proposal,
                default_timestamp_s=packet.timestamp_s,
                default_source_drone_id=packet.drone_id,
            )
            for proposal in proposals
        ]


def decode_rgbd_montage(packet: bytes) -> tuple[dict, str]:
    """Validate one UFWM packet and return metadata plus a JPEG data URL."""

    if len(packet) < 8 or packet[:4] != b"UFWM":
        raise ValueError("invalid UrbanFly sensor packet magic")
    header_length = int.from_bytes(packet[4:8], "little")
    if header_length <= 0 or 8 + header_length > len(packet):
        raise ValueError("invalid UrbanFly sensor packet header length")
    header = json.loads(packet[8 : 8 + header_length].decode("utf-8"))
    if header.get("schema") != "urbanfly-sensor-packet-v2":
        raise ValueError("unsupported UrbanFly sensor packet schema")
    width, height = int(header["width"]), int(header["height"])
    rgb_length = int(header["rgb_length"])
    depth_length = int(header["depth_length"])
    payload_start = 8 + header_length
    if payload_start + rgb_length + depth_length != len(packet):
        raise ValueError("UrbanFly sensor packet length mismatch")
    rgb_payload = packet[payload_start : payload_start + rgb_length]
    codec = str(header.get("rgb_codec", "jpeg_q95"))
    if codec == "raw_rgb8":
        expected = width * height * 3
        if len(rgb_payload) != expected:
            raise ValueError("raw RGB payload size mismatch")
        rgb = Image.frombytes("RGB", (width, height), rgb_payload)
    elif codec.startswith("jpeg"):
        rgb = Image.open(BytesIO(rgb_payload)).convert("RGB")
    else:
        raise ValueError(f"unsupported RGB codec: {codec}")

    depth_payload = packet[payload_start + rgb_length :]
    compression = str(header.get("depth_compression", "none"))
    if compression == "deflate":
        depth_payload = zlib.decompress(depth_payload)
    elif compression != "none":
        raise ValueError(f"unsupported depth compression: {compression}")
    depth_u16 = np.frombuffer(depth_payload, dtype="<u2")
    if depth_u16.size != width * height:
        raise ValueError("depth payload size mismatch")
    depth_m = depth_u16.reshape(height, width).astype(np.float32)
    depth_m *= float(header.get("depth_scale_m", 0.01))
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    clip_m = float(np.percentile(depth_m[valid], 95)) if valid.any() else 120.0
    clip_m = max(1.0, min(120.0, clip_m))
    depth_gray = np.zeros((height, width), dtype=np.uint8)
    depth_gray[valid] = np.clip(
        255.0 * (1.0 - depth_m[valid] / clip_m), 0.0, 255.0
    ).astype(np.uint8)
    depth = ImageOps.colorize(
        Image.fromarray(depth_gray, mode="L"),
        black="#06131d",
        white="#ffcf66",
    )
    montage = Image.new("RGB", (rgb.width * 2, rgb.height), "black")
    montage.paste(rgb, (0, 0))
    montage.paste(depth.resize(rgb.size, Image.Resampling.BILINEAR), (rgb.width, 0))
    if montage.width > 960:
        target_height = round(montage.height * 960 / montage.width)
        montage = montage.resize((960, target_height), Image.Resampling.LANCZOS)
    encoded = BytesIO()
    montage.save(encoded, format="JPEG", quality=82, optimize=True)
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
    header["depth_valid_ratio"] = float(valid.mean())
    header["depth_p95_m"] = clip_m
    return header, data_url


def compact_telemetry(header: dict) -> dict:
    keys = (
        "sequence",
        "sim_time",
        "vehicle_name",
        "goal_body_flu_m",
        "linear_velocity_body_flu_mps",
        "angular_velocity_body_flu_rps",
        "depth_valid_ratio",
        "depth_p95_m",
    )
    return {key: header[key] for key in keys if key in header}


async def analyze_and_propose(
    socket,
    client: OpenAICompatibleQwenVLClient,
    *,
    observer: str,
    timestamp_s: float,
    telemetry: dict,
    frames: tuple[str, ...],
) -> None:
    packet = ObservationPacket(
        timestamp_s=timestamp_s,
        drone_id=observer,
        telemetry=telemetry,
        frame_data_urls=frames,
    )
    events = await asyncio.to_thread(client.analyze, packet)
    await socket.send_json({
        "type": "semantic_event_proposal",
        "payload": {
            "observer_drone_id": observer,
            "observation_timestamp_s": timestamp_s,
            "events": [event.to_dict() for event in events],
            "model": getattr(client, "model_name", getattr(client, "model", "unknown")),
            "inference_latency_ms": getattr(client, "last_latency_ms", None),
        },
    })


async def run(args) -> None:
    if args.direct_model is None and not args.api_key:
        raise RuntimeError(
            "Qwen API key is not configured; set DASHSCOPE_API_KEY or "
            "URBANFLY_QWEN_API_KEY. Keys are never stored by UrbanFly."
        )
    client = (
        DirectTransformersQwenVLClient(args.direct_model)
        if args.direct_model is not None
        else OpenAICompatibleQwenVLClient(
            args.endpoint,
            model=args.model,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
        )
    )
    histories: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=4))
    last_query_sim_time: dict[str, float] = defaultdict(lambda: -float("inf"))
    pending: set[asyncio.Task] = set()

    def finish_inference(task: asyncio.Task) -> None:
        pending.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            print(
                json.dumps(
                    {
                        "type": "semantic_observer_inference_error",
                        "error": f"{type(error).__name__}: {error}",
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(args.ws_url, max_msg_size=8 * 1024 * 1024) as socket:
            await socket.send_json({
                "type": "policy_subscribe",
                "payload": {
                    "mode": "semantic_observer",
                    "lockstep": False,
                    "sends_control_actions": False,
                },
            })
            async for message in socket:
                if message.type == aiohttp.WSMsgType.BINARY:
                    header, frame = decode_rgbd_montage(message.data)
                    observer = str(header.get("vehicle_name", ""))
                    sim_time = float(header.get("sim_time", -1.0))
                    if not observer or sim_time < 0:
                        continue
                    histories[observer].append(frame)
                    if sim_time - last_query_sim_time[observer] < args.interval_s:
                        continue
                    if len(pending) >= args.max_concurrent:
                        continue
                    last_query_sim_time[observer] = sim_time
                    task = asyncio.create_task(analyze_and_propose(
                        socket,
                        client,
                        observer=observer,
                        timestamp_s=sim_time,
                        telemetry=compact_telemetry(header),
                        frames=tuple(histories[observer]),
                    ))
                    pending.add(task)
                    task.add_done_callback(finish_inference)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(message.data)
                    if data.get("type") == "semantic_event_ack":
                        print(json.dumps(data["payload"], ensure_ascii=False))
                    elif data.get("type") == "error":
                        print(json.dumps(data, ensure_ascii=False), file=sys.stderr)
                elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ws-url", default="http://127.0.0.1:8765/ws")
    result.add_argument(
        "--endpoint",
        default=os.environ.get(
            "URBANFLY_QWEN_ENDPOINT",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        help="OpenAI-compatible base URL ending in /v1",
    )
    result.add_argument(
        "--direct-model",
        type=Path,
        default=None,
        help="Optional explicit local checkpoint; API mode is the release default",
    )
    result.add_argument(
        "--api-key",
        default=os.environ.get("URBANFLY_QWEN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY"),
    )
    result.add_argument("--model", default="qwen3-vl-plus")
    result.add_argument("--interval-s", type=float, default=2.0)
    result.add_argument("--timeout-s", type=float, default=8.0)
    result.add_argument("--max-concurrent", type=int, default=1)
    return result


if __name__ == "__main__":
    asyncio.run(run(parser().parse_args()))
