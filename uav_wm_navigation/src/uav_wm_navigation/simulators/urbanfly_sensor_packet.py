from __future__ import annotations

import json
import zlib
from dataclasses import dataclass

import cv2
import numpy as np

from uav_wm_navigation.types import WorldModelObservation


@dataclass(frozen=True, slots=True)
class DecodedUrbanFlyPacket:
    header: dict
    observation: WorldModelObservation


def decode_urbanfly_sensor_packet(
    packet: bytes,
    *,
    episode_id: str,
    previous_action: np.ndarray | None = None,
) -> DecodedUrbanFlyPacket:
    """Decode the binary browser-to-policy RGB-D bridge without extra copies."""

    if len(packet) < 8 or packet[:4] != b"UFWM":
        raise ValueError("invalid UrbanFly sensor packet magic")
    header_length = int.from_bytes(packet[4:8], "little")
    if header_length <= 0 or 8 + header_length > len(packet):
        raise ValueError("invalid UrbanFly sensor packet header length")
    header = json.loads(packet[8 : 8 + header_length].decode("utf-8"))
    if header.get("schema") != "urbanfly-sensor-packet-v2":
        raise ValueError("unsupported UrbanFly sensor packet schema")
    rgb_length = int(header["rgb_length"])
    depth_length = int(header["depth_length"])
    payload_start = 8 + header_length
    if payload_start + rgb_length + depth_length != len(packet):
        raise ValueError("UrbanFly sensor packet length mismatch")
    height, width = int(header["height"]), int(header["width"])
    rgb_payload = np.frombuffer(
        packet, dtype=np.uint8, count=rgb_length, offset=payload_start
    )
    rgb_codec = str(header.get("rgb_codec", "jpeg_q95"))
    if rgb_codec == "raw_rgb8":
        if rgb_payload.size != height * width * 3:
            raise ValueError("decoded UrbanFly raw RGB shape mismatch")
        rgb = rgb_payload.reshape(height, width, 3).copy()
    elif rgb_codec.startswith("jpeg"):
        rgb_bgr = cv2.imdecode(rgb_payload, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise ValueError("failed to decode UrbanFly RGB JPEG")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported RGB codec: {rgb_codec}")
    depth_payload = packet[payload_start + rgb_length :]
    compression = header.get("depth_compression", "none")
    if compression == "deflate":
        depth_payload = zlib.decompress(depth_payload)
    elif compression != "none":
        raise ValueError(f"unsupported depth compression: {compression}")
    depth_u16 = np.frombuffer(depth_payload, dtype="<u2")
    if depth_u16.size != height * width:
        raise ValueError("decoded UrbanFly depth shape mismatch")
    depth_m = (
        depth_u16.reshape(height, width).astype(np.float32)
        * float(header["depth_scale_m"])
    )
    intrinsics_data = header["intrinsics"]
    intrinsics = np.asarray(
        [
            [intrinsics_data["fx"], 0.0, intrinsics_data["cx"]],
            [0.0, intrinsics_data["fy"], intrinsics_data["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    observation = WorldModelObservation(
        episode_id=episode_id,
        step_id=int(header["sequence"]),
        sim_time=float(header["sim_time"]),
        rgb=rgb,
        depth_m=depth_m,
        depth_valid_mask=np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= 120.0),
        goal_body_flu_m=np.asarray(header["goal_body_flu_m"], dtype=np.float32),
        linear_velocity_body_flu_mps=np.asarray(
            header["linear_velocity_body_flu_mps"], dtype=np.float32
        ),
        angular_velocity_body_flu_rps=np.asarray(
            header["angular_velocity_body_flu_rps"], dtype=np.float32
        ),
        gravity_body_flu=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        previous_action=(
            np.zeros(4, dtype=np.float32)
            if previous_action is None
            else np.asarray(previous_action, dtype=np.float32)
        ),
        sensor_timestamp=float(header["sim_time"]),
        state_timestamp=float(header["sim_time"]),
        camera_intrinsics=intrinsics,
        camera_extrinsics_body=np.eye(4, dtype=np.float32),
    )
    return DecodedUrbanFlyPacket(header=header, observation=observation)
