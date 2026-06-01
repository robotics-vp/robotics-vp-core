#!/usr/bin/env python3
"""Build the local provider bring-up readiness ledger."""

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

from src.runpod.provider_readiness_ledger import write_provider_readiness_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/runpod/provider_readiness_ledger")
    parser.add_argument("--volume-id", default=os.environ.get("RUNPOD_VOLUME_ID", ""))
    parser.add_argument("--api-key-present", action="store_true")
    args = parser.parse_args()

    summary = write_provider_readiness_report(
        args.output_dir,
        volume_id=args.volume_id or None,
        api_key="present" if args.api_key_present else None,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "report_id": summary["report_id"],
                "entry_count": summary["entry_count"],
                "json_path": summary["json_path"],
                "markdown_path": summary["markdown_path"],
                "provider_execution_attempted": summary["provider_execution_attempted"],
                "promotion_eligible": summary["promotion_eligible"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
