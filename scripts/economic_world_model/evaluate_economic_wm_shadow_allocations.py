#!/usr/bin/env python3
"""Evaluate shadow-only Economic WM allocation candidates over local rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model import (  # noqa: E402
    EconomicWMShadowAllocationEval,
    evaluate_economic_wm_shadow_allocations_from_paths,
)


def _write_markdown(path: Path, eval_report: EconomicWMShadowAllocationEval) -> None:
    payload = eval_report.to_dict()
    lines = [
        "# Economic WM Shadow Allocation Eval",
        "",
        f"- Eval ID: `{payload['eval_id']}`",
        f"- Scaffold ID: `{payload['scaffold_id']}`",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Recommended candidate: `{payload['recommended_candidate']}`",
        f"- Row count: `{payload['row_count']}`",
        f"- Benchmark-ready rows: `{payload['benchmark_ready_count']}`",
        f"- Shadow-only rows: `{payload['shadow_only_count']}`",
        f"- Authority class: `{payload['authority_class']}`",
        f"- Reward math mutation: `{str(payload['reward_math_mutation']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        "",
        "## Candidates",
    ]
    for candidate in payload["candidates"]:
        lines.extend(
            [
                f"### `{candidate['label']}`",
                f"- allowed: `{str(candidate['allowed']).lower()}`",
                f"- expected_value: `{candidate['expected_value']}`",
                f"- rationale: {candidate['rationale']}",
            ]
        )
        denials = candidate["denial_reasons"]
        if denials:
            lines.append("- denial reasons:")
            lines.extend(f"  - `{reason}`" for reason in denials)
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a shadow-only recommendation. It does not execute allocation, mutate reward math, run GPU training, or promote a model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(
    *,
    output_root: Path,
    scaffold_report_path: Optional[str | Path],
    corpus_manifest_path: Optional[str | Path],
    rows_path: Optional[str | Path],
    run_rows_if_missing: bool,
) -> tuple[Path, Path, Path]:
    if corpus_manifest_path is None or rows_path is None:
        expected_rows_root = output_root / "training_rows"
        expected_manifest = (
            expected_rows_root / "economic_wm_training_corpus_manifest_v1.json"
        )
        expected_rows = expected_rows_root / "economic_wm_replay_feature_rows_v1.jsonl"
        if run_rows_if_missing:
            from scripts.economic_world_model.materialize_economic_wm_training_rows import (
                run_materialize_economic_wm_training_rows,
            )

            row_payload = run_materialize_economic_wm_training_rows(
                output_dir=expected_rows_root,
                scaffold_report_path=scaffold_report_path,
            )
            manifest_path = Path(row_payload["artifact_refs"]["manifest_path"])
            resolved_rows_path = Path(row_payload["artifact_refs"]["rows_path"])
            resolved_scaffold_path = Path(
                row_payload["artifact_refs"]["scaffold_report_path"]
            )
        else:
            manifest_path = (
                expected_manifest
                if corpus_manifest_path is None
                else Path(corpus_manifest_path)
            )
            resolved_rows_path = expected_rows if rows_path is None else Path(rows_path)
            if scaffold_report_path is None:
                resolved_scaffold_path = (
                    output_root / "scaffold" / "economic_wm_scaffold_report_v1.json"
                )
            else:
                resolved_scaffold_path = Path(scaffold_report_path)
    else:
        manifest_path = Path(corpus_manifest_path)
        resolved_rows_path = Path(rows_path)
        if scaffold_report_path is None:
            resolved_scaffold_path = (
                output_root / "scaffold" / "economic_wm_scaffold_report_v1.json"
            )
        else:
            resolved_scaffold_path = Path(scaffold_report_path)
    return resolved_scaffold_path, manifest_path, resolved_rows_path


def run_evaluate_economic_wm_shadow_allocations(
    *,
    output_dir: str | Path,
    scaffold_report_path: Optional[str | Path] = None,
    corpus_manifest_path: Optional[str | Path] = None,
    rows_path: Optional[str | Path] = None,
    run_rows_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    scaffold_path, manifest_path, resolved_rows_path = _resolve_inputs(
        output_root=output_root,
        scaffold_report_path=scaffold_report_path,
        corpus_manifest_path=corpus_manifest_path,
        rows_path=rows_path,
        run_rows_if_missing=run_rows_if_missing,
    )
    eval_path = output_root / "economic_wm_shadow_allocation_eval_v1.json"
    markdown_path = output_root / "economic_wm_shadow_allocation_eval_v1.md"
    eval_report = evaluate_economic_wm_shadow_allocations_from_paths(
        scaffold_report_path=scaffold_path,
        corpus_manifest_path=manifest_path,
        rows_path=resolved_rows_path,
        output_path=eval_path,
        metadata={"source": "evaluate_economic_wm_shadow_allocations_script"},
    )
    payload = eval_report.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "eval_path": str(eval_path),
        "markdown_path": str(markdown_path),
    }
    eval_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMShadowAllocationEval.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_shadow_allocation_eval",
        help="Directory for shadow allocation eval artifacts.",
    )
    parser.add_argument("--scaffold-report", default=None)
    parser.add_argument("--corpus-manifest", default=None)
    parser.add_argument("--rows", default=None)
    parser.add_argument(
        "--no-run-rows",
        action="store_true",
        help="Do not materialize rows when corpus inputs are missing.",
    )
    args = parser.parse_args()
    payload = run_evaluate_economic_wm_shadow_allocations(
        output_dir=args.output_dir,
        scaffold_report_path=args.scaffold_report,
        corpus_manifest_path=args.corpus_manifest,
        rows_path=args.rows,
        run_rows_if_missing=not args.no_run_rows,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["recommended_candidate"] and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
