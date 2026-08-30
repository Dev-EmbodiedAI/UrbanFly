from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble captured CityGS PNG frames into MP4 and GIF.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-gif", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--gif-max-width", type=int, default=1280)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths = sorted(args.frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {args.frames_dir}")

    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_gif.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(
        args.output_video,
        frames,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )

    gif_frames: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[1] > args.gif_max_width:
            scale = args.gif_max_width / frame.shape[1]
            resized = Image.fromarray(frame).resize(
                (args.gif_max_width, max(int(round(frame.shape[0] * scale)), 1)),
                resample=Image.Resampling.LANCZOS,
            )
            gif_frames.append(np.asarray(resized))
        else:
            gif_frames.append(frame)
    imageio.mimsave(args.output_gif, gif_frames, fps=args.fps)


if __name__ == "__main__":
    main()
