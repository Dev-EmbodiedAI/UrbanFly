from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def source_timing(run: Path) -> tuple[Path, float, float]:
    video = run / "front_camera_source.mp4"
    metadata = json.loads((run / "front_camera_source.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    encoded_duration = max((frames - 1) / max(fps, 1e-6), 1e-6)
    simulation_duration = float(metadata["simulation_duration_s"])
    return video, simulation_duration / encoded_duration, simulation_duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Render synchronized real front-camera views for the three ablations.")
    parser.add_argument("--yopo", type=Path, required=True)
    parser.add_argument("--dreamer", type=Path, required=True)
    parser.add_argument("--jepa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [args.yopo, args.dreamer, args.jepa]
    labels = ["A  YOPO", "B  YOPO + DreamerV3-style RSSM", "C  YOPO + Action-Conditioned JEPA"]
    timing = [source_timing(run) for run in runs]
    maximum_duration = max(item[2] for item in timing)
    font = "C\\:/Windows/Fonts/arial.ttf"
    filters = []
    for index, (_, scale, _) in enumerate(timing):
        filters.append(
            f"[{index}:v]setpts={scale:.10f}*PTS,"
            "scale=640:360:force_original_aspect_ratio=decrease,"
            "pad=640:360:(ow-iw)/2:(oh-ih)/2:black,fps=60,"
            "tpad=stop_mode=clone:stop_duration=10,"
            "drawbox=x=0:y=0:w=iw:h=52:color=black@0.62:t=fill,"
            f"drawtext=fontfile='{font}':text='{labels[index]}':x=18:y=13:fontsize=25:fontcolor=white[v{index}]"
        )
    filters.append(
        f"[v0][v1][v2]hstack=inputs=3,trim=duration={maximum_duration:.6f},setpts=PTS-STARTPTS[out]"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    for video, _, _ in timing:
        command += ["-i", str(video)]
    command += [
        "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ]
    subprocess.run(command, check=True)
    capture = cv2.VideoCapture(str(args.output))
    result = {
        "output": str(args.output.resolve()),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / max(float(capture.get(cv2.CAP_PROP_FPS)), 1e-6),
        "source_simulation_durations_s": [item[2] for item in timing],
        "time_scale_factors": [item[1] for item in timing],
        "interpretation": (
            "Three real, time-correct front-camera streams from the same route and seed. "
            "Shorter streams hold their last frame; no playback speed-up is used."
        ),
    }
    capture.release()
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
