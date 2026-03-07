#!/usr/bin/env python3
"""Evaluate recurring promotion evidence for regal and advisor nodes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay.dataset import load_replay_dataset
from src.replay.receipt_ingest import resolve_receipt_label_bundle
from src.regality.promotion_policy import load_regal_promotion_policy
from src.regality.promotion_reporting import (
    DEFAULT_PROMOTION_NODE_IDS,
    build_promotion_evidence_report,
    write_promotion_evidence_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate regal promotion readiness")
    parser.add_argument("--replay-dataset-dir", required=True, type=str)
    parser.add_argument("--promotion-policy", default="configs/regality/promotion_default.yaml", type=str)
    parser.add_argument("--receipt-label-dir", default=None, type=str)
    parser.add_argument("--receipt-label-mode", default="synthetic_shadow", type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    dataset = load_replay_dataset(args.replay_dataset_dir)
    policy = load_regal_promotion_policy(args.promotion_policy)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_bundle = resolve_receipt_label_bundle(
        dataset=dataset,
        receipt_label_dir=args.receipt_label_dir,
        allow_synthetic=True,
        label_mode=args.receipt_label_mode,
    )

    report = build_promotion_evidence_report(
        dataset=dataset,
        promotion_policy=policy,
        receipt_bundle=receipt_bundle,
        node_ids=DEFAULT_PROMOTION_NODE_IDS,
        evidence_pointers={
            "dataset_dir": str(args.replay_dataset_dir),
            "receipt_label_dir": str(args.receipt_label_dir) if args.receipt_label_dir else "synthetic_generated",
        },
    )
    paths = write_promotion_evidence_report(output_root, report)
    summary = dict(report.summary)
    summary["artifacts"] = paths
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
