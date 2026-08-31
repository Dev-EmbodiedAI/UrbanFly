#!/usr/bin/env python
"""汇总并严格审计五类 Swarm 数字孪生闭环结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.digital_twin.qa import audit_cross_environment_reports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--policy-source", action="append", type=Path, default=[])
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    commit_file = args.upstream_root.resolve() / ".git"
    upstream_commit = args.upstream_commit
    if upstream_commit is None and commit_file.exists():
        import subprocess

        upstream_commit = subprocess.check_output(
            ["git", "-C", str(args.upstream_root.resolve()), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    result = audit_cross_environment_reports(
        args.report,
        policy_sources=args.policy_source,
        upstream_commit=upstream_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "episodes": result["episodes"],
        "successful_drones": result["successful_drones"],
        "total_drones": result["total_drones"],
        "collisions": result["collisions"],
        "total_steps": result["total_steps"],
        "upstream_commit": result["upstream_commit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
