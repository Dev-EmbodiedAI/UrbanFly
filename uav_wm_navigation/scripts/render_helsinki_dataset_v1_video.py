#!/usr/bin/env python3
"""Render a visual audit video directly from one Dataset v1 HDF5 episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def _text(frame: np.ndarray, value: str, x: int, y: int, scale: float = 0.55) -> None:
    cv2.putText(
        frame,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (238, 243, 248),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.episode, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        rgb = handle["observations/rgb_front"][:]
        depth = handle["observations/depth_front"][:]
        sim = handle["timestamps/sim"][:]
        dt = handle["timestamps/dt"][:]
        position = handle["state/position_world"][:]
        velocity = handle["state/linear_velocity"][:]
        remaining = handle["route/remaining_distance"][:]
        clearance = handle["labels/minimum_clearance"][:]
        commanded = handle["actions/commanded_body_flu"][:]
        executed = handle["actions/executed_body_flu"][:]
        collision = handle["labels/collision"][:]
        success = handle["labels/success"][:]

    if len(rgb) == 0 or len(rgb) != len(depth):
        raise ValueError("episode must contain matching non-empty RGB-D arrays")
    simulation_duration = float(np.sum(dt))
    fps = float(np.clip(len(rgb) / max(simulation_duration, 1e-6), 1.0, 30.0))
    view_width, view_height = 480, 270
    panel_height = 150
    output_size = (view_width * 2, view_height + panel_height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")
    try:
        for index in range(len(rgb)):
            rgb_bgr = cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR)
            rgb_view = cv2.resize(rgb_bgr, (view_width, view_height), interpolation=cv2.INTER_NEAREST)
            depth_u8 = np.clip(depth[index] / 120.0 * 255.0, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
            depth_view = cv2.resize(depth_color, (view_width, view_height), interpolation=cv2.INTER_NEAREST)
            _text(rgb_view, "REAL WEBGL RGB", 14, 25, 0.65)
            _text(depth_view, "METRIC DEPTH 0-120 m", 14, 25, 0.65)
            frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            frame[:view_height, :view_width] = rgb_view
            frame[:view_height, view_width:] = depth_view
            speed = float(np.linalg.norm(velocity[index]))
            time_value = float(sim[index] - sim[0])
            _text(frame, f"Dataset v1 factual replay | {metadata['task_type']} | frame {index + 1}/{len(rgb)}", 15, 296, 0.62)
            _text(frame, f"sim t={time_value:6.2f}s  dt={dt[index]:.3f}s  speed={speed:.2f}m/s  clearance={clearance[index]:.2f}m", 15, 322)
            _text(frame, f"ENU position=({position[index,0]:.2f}, {position[index,1]:.2f}, {position[index,2]:.2f})  route remaining={remaining[index]:.2f}m", 15, 348)
            _text(frame, "commanded FLU=" + np.array2string(commanded[index], precision=2, suppress_small=True), 15, 374, 0.48)
            _text(frame, "executed  FLU=" + np.array2string(executed[index], precision=2, suppress_small=True), 15, 399, 0.48)
            status = "SUCCESS" if success[index] else ("COLLISION" if collision[index] else "FLYING")
            color = (70, 220, 120) if status != "COLLISION" else (40, 40, 240)
            cv2.putText(frame, status, (810, 402), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()

    print(
        json.dumps(
            {
                "status": "PASS",
                "source": str(args.episode.resolve()),
                "output": str(args.output.resolve()),
                "frames": int(len(rgb)),
                "fps": fps,
                "simulation_duration_s": simulation_duration,
                "provenance": "reconstructed directly from synchronized HDF5 RGB-D and telemetry",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
