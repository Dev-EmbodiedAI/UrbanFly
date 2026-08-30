from __future__ import annotations

import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_run_dir(root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root.expanduser().resolve() / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_manifest(run_dir: Path, config: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(item) for item in sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        **(extra or {}),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

