"""Create a progressively streamable browser splat from a CityGS PLY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_citygs_splats import open_ply_memmap, sigmoid

SH_C0 = 0.28209479177387814
SPLAT_DTYPE = np.dtype(
    [
        ("position", "<f4", (3,)),
        ("scale", "<f4", (3,)),
        ("color", "u1", (4,)),
        ("rotation", "u1", (4,)),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a progressive CityGS web splat.")
    parser.add_argument(
        "--ply",
        type=Path,
        default=Path(
            r"C:\Users\caste\Downloads\Residence\residence_c20_r4_light_60_vq\point_cloud.ply"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/citygs_visualization/assets/Residence_web_1m.splat"),
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--opacity", type=float, default=0.08)
    parser.add_argument("--scale-multiplier", type=float, default=2.2)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices = open_ply_memmap(args.ply)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.output.open("wb") as output:
        for start in range(0, len(vertices), args.chunk_size):
            stop = min(len(vertices), start + args.chunk_size)
            chunk = vertices[start:stop]
            alpha = sigmoid(np.asarray(chunk["opacity"], dtype=np.float32))
            global_indices = np.arange(start, stop, dtype=np.int64)
            mask = (global_indices % args.stride == 0) & (alpha >= args.opacity)
            if not np.any(mask):
                continue

            selected = chunk[mask]
            selected_alpha = alpha[mask]
            count = len(selected)
            rows = np.empty(count, dtype=SPLAT_DTYPE)
            rows["position"] = np.column_stack(
                (selected["x"], selected["y"], selected["z"])
            ).astype(np.float32)
            rows["scale"] = np.exp(
                np.column_stack(
                    (selected["scale_0"], selected["scale_1"], selected["scale_2"])
                ).astype(np.float32)
            ) * float(args.scale_multiplier)

            dc = np.column_stack(
                (selected["f_dc_0"], selected["f_dc_1"], selected["f_dc_2"])
            ).astype(np.float32)
            rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
            rows["color"][:, :3] = np.rint(rgb * 255.0).astype(np.uint8)
            rows["color"][:, 3] = np.rint(selected_alpha * 255.0).astype(np.uint8)

            quaternion = np.column_stack(
                (
                    selected["rot_0"],
                    selected["rot_1"],
                    selected["rot_2"],
                    selected["rot_3"],
                )
            ).astype(np.float32)
            quaternion /= np.linalg.norm(
                quaternion, axis=1, keepdims=True
            ).clip(min=1e-8)
            rows["rotation"] = np.clip(
                np.rint(quaternion * 128.0 + 128.0), 0, 255
            ).astype(np.uint8)
            output.write(rows.tobytes())
            written += count
            print(
                f"[Web splat] {stop:,}/{len(vertices):,}; "
                f"{written:,} Gaussians written"
            )

    print(
        f"[Web splat] wrote {args.output} "
        f"({written:,} Gaussians, {args.output.stat().st_size / 1_000_000:.1f} MB)"
    )


if __name__ == "__main__":
    main()
