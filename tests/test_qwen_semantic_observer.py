import base64
from io import BytesIO
import json
import zlib

import numpy as np
from PIL import Image

from scripts.run_qwen_semantic_observer import decode_rgbd_montage


def test_rgbd_packet_becomes_compact_qwen_montage_without_disk_artifact():
    width, height = 8, 4
    rgb = Image.new("RGB", (width, height), (30, 80, 140))
    rgb_buffer = BytesIO()
    rgb.save(rgb_buffer, format="JPEG", quality=90)
    rgb_payload = rgb_buffer.getvalue()
    depth = np.arange(1, width * height + 1, dtype="<u2").tobytes()
    depth_payload = zlib.compress(depth)
    header = {
        "schema": "urbanfly-sensor-packet-v2",
        "width": width,
        "height": height,
        "rgb_length": len(rgb_payload),
        "depth_length": len(depth_payload),
        "rgb_codec": "jpeg_q90",
        "depth_compression": "deflate",
        "depth_scale_m": 0.1,
        "vehicle_name": "UAV-A",
        "sim_time": 1.5,
        "sequence": 7,
    }
    encoded_header = json.dumps(header).encode("utf-8")
    packet = (
        b"UFWM"
        + len(encoded_header).to_bytes(4, "little")
        + encoded_header
        + rgb_payload
        + depth_payload
    )
    decoded_header, data_url = decode_rgbd_montage(packet)
    assert decoded_header["depth_valid_ratio"] == 1.0
    assert data_url.startswith("data:image/jpeg;base64,")
    montage = Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
    assert montage.size == (width * 2, height)
