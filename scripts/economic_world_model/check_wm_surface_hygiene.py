#!/usr/bin/env python3
"""Run a full WM surface hygiene sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model.wm_surface_hygiene import (  # noqa: E402
    DEFAULT_TARGET_ROOTS,
    run_wm_surface_hygiene,
)


def _git_changed_paths(repo_root: Path) -> list[str]:
    commands = [
        ["git", "diff", "--name-only", "HEAD", "--"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    paths: list[str] = []
    for command in commands:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            paths.extend(line.strip() for line in proc.stdout.splitlines() if line.strip())
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep WM code/docs/scripts/manifests for hygiene drift"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/wm_surface_hygiene",
    )
    parser.add_argument(
        "--target-root",
        action="append",
        default=[],
        help="Override target roots. May be passed multiple times.",
    )
    parser.add_argument("--large-python-line-threshold", type=int, default=2000)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    target_roots = tuple(args.target_root) if args.target_root else DEFAULT_TARGET_ROOTS
    report = run_wm_surface_hygiene(
        repo_root=repo_root,
        output_dir=args.output_dir,
        changed_paths=_git_changed_paths(repo_root),
        target_roots=target_roots,
        large_python_line_threshold=args.large_python_line_threshold,
    )
    summary_keys = [
        "status",
        "scanned_file_count",
        "python_file_count",
        "doc_file_count",
        "manifest_file_count",
        "receipt_count",
        "blocking_issue_count",
        "advisory_issue_count",
        "risky_true_claim_count",
        "protected_change_count",
        "oversized_python_file_count",
        "todo_marker_count",
        "output_paths",
    ]
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    return 0 if report["blocking_issue_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
