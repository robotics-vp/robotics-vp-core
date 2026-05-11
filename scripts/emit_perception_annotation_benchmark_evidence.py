#!/usr/bin/env python3
"""Emit Perception benchmark evidence from persisted annotation exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.world_model.perception_grounding.benchmark_evidence_emitter import (  # noqa: E402
    ANNOTATION_BENCHMARK_SEAM_TYPES,
    emit_annotation_benchmark_evidence,
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
            "Evaluate a Perception seam on an annotation-export JSON and "
            "persist a typed benchmark-evidence artifact."
        ),
    )
    parser.add_argument(
        "--annotation-export",
        required=True,
        help="Path to annotation_export_v2 JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for perception_benchmark_evidence_v1 JSON.",
    )
    parser.add_argument(
        "--seam-type",
        default="scene_graph_transformer",
        choices=sorted(ANNOTATION_BENCHMARK_SEAM_TYPES),
        help="Seam to evaluate against the annotation export.",
    )
    parser.add_argument(
        "--seam-id",
        default=None,
        help="Optional seam id for registry metadata.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional seam checkpoint. Missing checkpoints use fresh init truth.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional registry checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for seam evaluation.",
    )
    parser.add_argument(
        "--evidence-source-provisional",
        type=_parse_optional_bool,
        default=None,
        help="Override token provenance: infer, true, or false. Default: infer.",
    )
    parser.add_argument(
        "--held-out-fraction",
        type=float,
        default=0.2,
        help="Held-out fraction for annotation benchmark evaluation.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path for emission summary JSON.",
    )
    args = parser.parse_args()

    emission = emit_annotation_benchmark_evidence(
        annotation_export_path=args.annotation_export,
        seam_type=args.seam_type,
        output_path=args.output,
        seam_id=args.seam_id,
        checkpoint_path=args.checkpoint_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        evidence_source_provisional=args.evidence_source_provisional,
        held_out_fraction=args.held_out_fraction,
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
                "seam_type": emission.seam_type,
                "source_record_count": evidence["source_record_count"],
                "evidence_source_provisional": evidence[
                    "evidence_source_provisional"
                ],
                "benchmark_evidence_present": evidence[
                    "benchmark_evidence_present"
                ],
                "promotion_eligible": evidence["promotion_eligible"],
                "evidence_digest": emission.evidence_digest,
                "output_path": emission.output_path,
                "checkpoint_ref_status": emission.checkpoint_ref_status,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
