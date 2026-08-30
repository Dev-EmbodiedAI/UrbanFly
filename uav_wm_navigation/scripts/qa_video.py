from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a rendered flight video and create a four-frame contact sheet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.input.resolve()))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frames < 4 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("invalid or incomplete video stream")
    selected = [0, frames // 3, 2 * frames // 3, frames - 1]
    samples, means = [], []
    for frame_index in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"failed to decode frame {frame_index}")
        means.append(float(frame.mean()))
        cv2.putText(frame, f"frame {frame_index + 1}/{frames}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 255, 255), 3, cv2.LINE_AA)
        samples.append(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA))
    capture.release()
    contact_sheet = np.vstack([np.hstack(samples[:2]), np.hstack(samples[2:])])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output.resolve()), contact_sheet):
        raise OSError(f"could not write {args.output}")
    report = {
        "passed": bool(min(means) > 8.0 and width == 1920 and height == 1080),
        "frames": frames, "width": width, "height": height, "fps": fps,
        "duration_s": frames / fps, "sample_brightness": means,
        "mp4_bytes": args.input.stat().st_size, "contact_sheet": str(args.output.resolve()),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
