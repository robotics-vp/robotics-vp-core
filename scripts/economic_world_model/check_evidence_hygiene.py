#!/usr/bin/env python3
"""Check Economic WM claim evidence, artifact freshness, and retention policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model.evidence_hygiene import (  # noqa: E402
    DEFAULT_ADVISORY_LOCAL_ARTIFACT_BYTES,
    DEFAULT_MAX_LOCAL_ARTIFACT_BYTES,
    run_economic_wm_evidence_hygiene,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Economic WM artifact claims and receipt hygiene"
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/economic_world_model",
        help="Artifact root to scan for JSON/JSONL receipts.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/evidence_hygiene",
        help="Directory for generated hygiene reports and receipts.",
    )
    parser.add_argument(
        "--max-local-artifact-bytes",
        type=int,
        default=DEFAULT_MAX_LOCAL_ARTIFACT_BYTES,
    )
    parser.add_argument(
        "--advisory-local-artifact-bytes",
        type=int,
        default=DEFAULT_ADVISORY_LOCAL_ARTIFACT_BYTES,
    )
    args = parser.parse_args()

    report = run_economic_wm_evidence_hygiene(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        max_local_artifact_bytes=args.max_local_artifact_bytes,
        advisory_local_artifact_bytes=args.advisory_local_artifact_bytes,
    )
    summary_keys = [
        "status",
        "scanned_file_count",
        "claim_receipt_count",
        "stale_receipt_count",
        "retention_receipt_count",
        "blocking_issue_count",
        "advisory_issue_count",
        "provider_gpu_hardware_claims_blocked",
        "artifact_refs_resolved",
        "retention_policy_passed",
        "output_paths",
    ]
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    return 0 if report["blocking_issue_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
