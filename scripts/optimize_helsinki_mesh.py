#!/usr/bin/env python3
"""Apply lossless Meshopt compression to a built Helsinki mesh scene."""

from __future__ import annotations

import argparse
import json
import subprocess
import shutil
from pathlib import Path


def optimized_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_meshopt{path.suffix}")


def optimize(cli: Path, source: Path) -> Path:
    target = optimized_path(source)
    executable = [str(cli)]
    if cli.suffix.lower() == ".js":
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node.js is required to run glTF Transform")
        executable = [node, str(cli)]
    command = executable + [
        "optimize",
        str(source),
        str(target),
        "--compress",
        "meshopt",
        "--meshopt-level",
        "high",
        "--simplify",
        "false",
        "--texture-compress",
        "false",
        "--palette",
        "false",
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    args = parser.parse_args()

    scene = args.scene.resolve()
    manifest_path = scene / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def optimize_tiles(section: dict, directory: str) -> tuple[int, int]:
        raw_bytes = 0
        optimized_bytes = 0
        for tile in section["tiles"]:
            source = scene / directory / tile["uri"]
            raw_bytes += source.stat().st_size
            target = optimize(args.cli.resolve(), source)
            tile["uri"] = target.name
            tile["bytes"] = target.stat().st_size
            tile["compression"] = "EXT_meshopt_compression"
            optimized_bytes += target.stat().st_size
        section["bytes"] = optimized_bytes
        section["raw_bytes"] = raw_bytes
        section["compression"] = {
            "extension": "EXT_meshopt_compression",
            "geometry_simplified": False,
            "textures_reencoded": False,
            "ratio": optimized_bytes / raw_bytes,
        }
        return raw_bytes, optimized_bytes

    raw_total, optimized_total = optimize_tiles(manifest["visual"], "visual")
    overview = manifest["visual"].get("overview")
    if overview and overview.get("tiles"):
        optimize_tiles(overview, "overview")

    collision = manifest["collision"]
    collision_source = scene / collision["uri"]
    collision_target = optimize(args.cli.resolve(), collision_source)
    collision["uri"] = str(
        collision_target.relative_to(scene)
    ).replace("\\", "/")
    collision["bytes"] = collision_target.stat().st_size
    collision["compression"] = "EXT_meshopt_compression"

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Visual GLB: {raw_total / 1048576:.2f} MiB -> "
        f"{optimized_total / 1048576:.2f} MiB"
    )


if __name__ == "__main__":
    main()
