#!/usr/bin/env python3
"""Emit Perception benchmark evidence from provider invocation receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.world_model.perception_grounding.benchmark_evidence_emitter import (  # noqa: E402
    PROVIDER_ADAPTER_BENCHMARK_SEAM_TYPES,
    emit_provider_adapter_benchmark_evidence,
)


def _parse_optional_bool(value: str) -> Optional[bool]:
    normalized = value.strip().lower()
    if normalized == "infer":
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected one of: infer, true, false")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Perception provider invocation receipts and persist a "
            "provider-adapter benchmark-evidence artifact."
        ),
    )
    parser.add_argument(
        "--provider-receipts",
        required=True,
        help=(
            "JSON containing a provider_invocation_receipt_v1, a list of "
            "receipts, or a Perception state metadata payload with "
            "provider_adapter_receipts."
        ),
    )
    parser.add_argument(
        "--provider-kind",
        required=True,
        choices=sorted(PROVIDER_ADAPTER_BENCHMARK_SEAM_TYPES),
        help="Provider adapter seam kind to aggregate.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for perception_benchmark_evidence_v1 JSON.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional provider-adapter checkpoint reference.",
    )
    parser.add_argument(
        "--training-manifest",
        default=None,
        help="Optional training_runtime_manifest_v1 path to link into metadata.",
    )
    parser.add_argument(
        "--metric-report",
        default=None,
        help=(
            "Optional JSON object with held-out metrics. Receipt-only evidence "
            "stays provisional unless this report or the explicit override says "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--evidence-source-provisional",
        type=_parse_optional_bool,
        default=None,
        help="Override evidence provenance: infer, true, or false. Default: infer.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path for emission summary JSON.",
    )
    args = parser.parse_args()

    emission = emit_provider_adapter_benchmark_evidence(
        provider_receipts_path=args.provider_receipts,
        provider_kind=args.provider_kind,
        output_path=args.output,
        checkpoint_path=args.checkpoint_path,
        training_manifest_path=args.training_manifest,
        metric_report_path=args.metric_report,
        evidence_source_provisional=args.evidence_source_provisional,
    )

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(emission.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    evidence = emission.evidence.to_dict()
    print(
        json.dumps(
            {
                "schema_version": emission.schema_version,
                "provider_kind": emission.provider_kind,
                "source_record_count": evidence["source_record_count"],
                "matched_receipt_count": emission.matched_receipt_count,
                "success_count": emission.success_count,
                "fallback_count": emission.fallback_count,
                "evidence_source_provisional": evidence["evidence_source_provisional"],
                "benchmark_evidence_present": evidence["benchmark_evidence_present"],
                "promotion_eligible": evidence["promotion_eligible"],
                "evidence_digest": emission.evidence_digest,
                "output_path": emission.output_path,
                "checkpoint_ref_status": emission.checkpoint_ref_status,
                "training_manifest_ref_status": (emission.training_manifest_ref_status),
                "metric_report_ref_status": emission.metric_report_ref_status,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
