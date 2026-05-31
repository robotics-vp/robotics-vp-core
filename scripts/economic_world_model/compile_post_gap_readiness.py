#!/usr/bin/env python3
"""Compile post-gap Economic WM readiness manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model import (  # noqa: E402
    save_post_gap_readiness_bundle,
)


def run_compile_post_gap_readiness(*, output_dir: str | Path) -> dict[str, Any]:
    return save_post_gap_readiness_bundle(output_dir=output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile planning-only post-gap readiness artifacts"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/post_gap_readiness",
        help="Directory for readiness artifacts.",
    )
    args = parser.parse_args()
    payload = run_compile_post_gap_readiness(output_dir=args.output_dir)
    summary_keys = [
        "status",
        "all_post_gap_items_manifested",
        "ready_for_august_gpu_window",
        "gpu_day_one_runbook_count",
        "external_dataset_count",
        "corpus_prep_artifact_count",
        "benchmark_gate_count",
        "provider_runtime_packaging_count",
        "replay_loop_count",
        "g1_r1_purchase_readiness_count",
        "evidence_hygiene_count",
        "launch_authority_granted",
        "external_download_executed",
        "provider_executed",
        "gpu_training_executed",
        "promotion_eligible",
        "phase7_constraint_honored",
    ]
    print(json.dumps({key: payload[key] for key in summary_keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
