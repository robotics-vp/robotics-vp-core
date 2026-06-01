#!/usr/bin/env python3
"""Prepare a RunPod launch manifest for provider, loop, or training profiles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.runpod import RUNPOD_LAUNCH_PROFILE_IDS, write_runpod_launch_manifest  # noqa: E402
from src.world_model.economic_world_model.gpu_run_hygiene import (  # noqa: E402
    validate_gpu_run_manifest_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=RUNPOD_LAUNCH_PROFILE_IDS, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--commit-sha", default="")
    parser.add_argument("--volume-id", default=os.environ.get("RUNPOD_VOLUME_ID", ""))
    parser.add_argument("--template", default=os.environ.get("RUNPOD_TEMPLATE_ID", ""))
    parser.add_argument("--image", default="")
    parser.add_argument("--output-root", default=".agent/runs")
    args = parser.parse_args()

    payload = write_runpod_launch_manifest(
        profile_id=args.profile,
        output_root=args.output_root,
        run_id=args.run_id,
        branch=args.branch,
        commit_sha=args.commit_sha,
        volume_id=args.volume_id or None,
        template=args.template,
        image=args.image,
    )
    receipts = validate_gpu_run_manifest_payload(
        payload["manifest"],
        manifest_path=payload["manifest_path"],
    )
    blocking = [
        receipt.to_dict()
        for receipt in receipts
        if not receipt.passed and receipt.severity == "blocking"
    ]
    summary = {
        "status": "ok_runpod_launch_manifest_prepared"
        if not blocking
        else "blocked_runpod_launch_manifest_hygiene_failed",
        "run_id": payload["run_id"],
        "profile_id": payload["profile_id"],
        "manifest_path": payload["manifest_path"],
        "launch_command_path": payload["launch_command_path"],
        "blocking_issue_count": len(blocking),
        "blocking_issues": blocking,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
