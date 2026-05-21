#!/usr/bin/env python3
"""Prepare Economic WM teacher/provider evidence contracts from local row evidence."""

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
    EconomicWMTeacherProviderContract,
    build_economic_wm_teacher_provider_contract_from_paths,
)


def _write_markdown(path: Path, contract: EconomicWMTeacherProviderContract) -> None:
    payload = contract.to_dict()
    lines = [
        "# Economic WM Teacher/Provider Evidence Contract",
        "",
        f"- Contract ID: `{payload['contract_id']}`",
        f"- Scaffold ID: `{payload['scaffold_id']}`",
        f"- Allocation eval ID: `{payload['allocation_eval_id']}`",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Authority class: `{payload['authority_class']}`",
        f"- Provider bring-up ready: `{str(payload['provider_bringup_ready']).lower()}`",
        f"- GPU training ready: `{str(payload['gpu_training_ready']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        f"- Reward math mutation: `{str(payload['reward_math_mutation']).lower()}`",
        "",
        "## Aggregate scores",
    ]
    for key, value in payload["aggregate_scores"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Requirements"])
    for requirement in payload["requirements"]:
        lines.extend(
            [
                f"### `{requirement['requirement_key']}`",
                f"- provider family: `{requirement['provider_family']}`",
                f"- evidence kind: `{requirement['evidence_kind']}`",
                f"- current status: `{requirement['current_status']}`",
                f"- satisfaction score: `{requirement['satisfaction_score']}`",
            ]
        )
        if requirement["blockers"]:
            lines.append("- blockers:")
            lines.extend(f"  - `{blocker}`" for blocker in requirement["blockers"])
        if requirement["local_prep_actions"]:
            lines.append("- local prep actions:")
            lines.extend(
                f"  - `{action}`" for action in requirement["local_prep_actions"]
            )
    lines.extend(["", "## Recommended next actions"])
    for action in payload["recommended_next_actions"]:
        lines.append(f"- `{action}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is contract prep only. It does not run a provider, run GPU training, promote a model, or mutate reward math.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(
    *,
    output_root: Path,
    scaffold_report_path: Optional[str | Path],
    allocation_eval_path: Optional[str | Path],
    corpus_manifest_path: Optional[str | Path],
    rows_path: Optional[str | Path],
    run_eval_if_missing: bool,
) -> tuple[Path, Path, Path, Path]:
    if (
        allocation_eval_path is None
        or corpus_manifest_path is None
        or rows_path is None
    ):
        if run_eval_if_missing:
            from scripts.economic_world_model.evaluate_economic_wm_shadow_allocations import (
                run_evaluate_economic_wm_shadow_allocations,
            )

            eval_payload = run_evaluate_economic_wm_shadow_allocations(
                output_dir=output_root / "shadow_allocation_eval",
                scaffold_report_path=scaffold_report_path,
            )
            artifact_refs = dict(eval_payload.get("artifact_refs", {}) or {})
            resolved_scaffold_path = Path(artifact_refs["scaffold_report_path"])
            resolved_eval_path = Path(artifact_refs["eval_path"])
            resolved_manifest_path = Path(artifact_refs["corpus_manifest_path"])
            resolved_rows_path = Path(artifact_refs["rows_path"])
        else:
            resolved_scaffold_path = Path(
                scaffold_report_path
                or output_root / "scaffold" / "economic_wm_scaffold_report_v1.json"
            )
            resolved_eval_path = Path(
                allocation_eval_path
                or output_root
                / "shadow_allocation_eval"
                / "economic_wm_shadow_allocation_eval_v1.json"
            )
            resolved_manifest_path = Path(
                corpus_manifest_path
                or output_root
                / "training_rows"
                / "economic_wm_training_corpus_manifest_v1.json"
            )
            resolved_rows_path = Path(
                rows_path
                or output_root
                / "training_rows"
                / "economic_wm_replay_feature_rows_v1.jsonl"
            )
    else:
        resolved_scaffold_path = Path(
            scaffold_report_path
            or output_root / "scaffold" / "economic_wm_scaffold_report_v1.json"
        )
        resolved_eval_path = Path(allocation_eval_path)
        resolved_manifest_path = Path(corpus_manifest_path)
        resolved_rows_path = Path(rows_path)
    return (
        resolved_scaffold_path,
        resolved_eval_path,
        resolved_manifest_path,
        resolved_rows_path,
    )


def run_prepare_economic_wm_teacher_provider_contracts(
    *,
    output_dir: str | Path,
    scaffold_report_path: Optional[str | Path] = None,
    allocation_eval_path: Optional[str | Path] = None,
    corpus_manifest_path: Optional[str | Path] = None,
    rows_path: Optional[str | Path] = None,
    run_eval_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    scaffold_path, eval_path, manifest_path, resolved_rows_path = _resolve_inputs(
        output_root=output_root,
        scaffold_report_path=scaffold_report_path,
        allocation_eval_path=allocation_eval_path,
        corpus_manifest_path=corpus_manifest_path,
        rows_path=rows_path,
        run_eval_if_missing=run_eval_if_missing,
    )
    contract_path = output_root / "economic_wm_teacher_provider_contract_v1.json"
    markdown_path = output_root / "economic_wm_teacher_provider_contract_v1.md"
    contract = build_economic_wm_teacher_provider_contract_from_paths(
        scaffold_report_path=scaffold_path,
        allocation_eval_path=eval_path,
        corpus_manifest_path=manifest_path,
        rows_path=resolved_rows_path,
        output_path=contract_path,
        metadata={"source": "prepare_economic_wm_teacher_provider_contracts_script"},
    )
    payload = contract.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "contract_path": str(contract_path),
        "markdown_path": str(markdown_path),
    }
    contract_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMTeacherProviderContract.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_teacher_provider_contracts",
        help="Directory for teacher/provider evidence contract artifacts.",
    )
    parser.add_argument("--scaffold-report", default=None)
    parser.add_argument("--allocation-eval", default=None)
    parser.add_argument("--corpus-manifest", default=None)
    parser.add_argument("--rows", default=None)
    parser.add_argument(
        "--no-run-eval",
        action="store_true",
        help="Do not run the shadow allocation evaluator if inputs are missing.",
    )
    args = parser.parse_args()
    payload = run_prepare_economic_wm_teacher_provider_contracts(
        output_dir=args.output_dir,
        scaffold_report_path=args.scaffold_report,
        allocation_eval_path=args.allocation_eval,
        corpus_manifest_path=args.corpus_manifest,
        rows_path=args.rows,
        run_eval_if_missing=not args.no_run_eval,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload["promotion_eligible"] and payload["requirements"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
