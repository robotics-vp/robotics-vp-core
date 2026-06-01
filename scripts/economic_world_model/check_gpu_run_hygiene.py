#!/usr/bin/env python3
"""Validate GPU/provider/loop/training manifests before launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model.gpu_run_hygiene import (  # noqa: E402
    run_gpu_run_hygiene,
)


def _manifest_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.manifest]
    for directory in args.manifest_dir:
        paths.extend(sorted(Path(directory).rglob("manifest.json")))
        paths.extend(sorted(Path(directory).glob("*.json")))
    return sorted({path for path in paths if path.exists()})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check GPU/provider/loop/training run manifest hygiene"
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Manifest JSON file to validate. May be passed multiple times.",
    )
    parser.add_argument(
        "--manifest-dir",
        action="append",
        default=[],
        help="Directory containing manifest JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/gpu_run_hygiene",
    )
    args = parser.parse_args()

    paths = _manifest_paths(args)
    if not paths:
        parser.error("no manifest files found")
    report = run_gpu_run_hygiene(manifest_paths=paths, output_dir=args.output_dir)
    summary_keys = [
        "status",
        "manifest_count",
        "receipt_count",
        "blocking_issue_count",
        "advisory_issue_count",
        "safe_to_queue_count",
        "unsafe_to_queue_count",
        "output_paths",
    ]
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    return 0 if report["blocking_issue_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
