#!/usr/bin/env python3
"""Materialize local Economic WM replay feature rows from scaffold evidence."""

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
    EconomicWMTrainingCorpusManifest,
    load_economic_wm_scaffold_report,
    materialize_economic_wm_training_corpus_from_paths,
)


def _write_markdown(path: Path, manifest: EconomicWMTrainingCorpusManifest) -> None:
    payload = manifest.to_dict()
    lines = [
        "# Economic WM Training Rows",
        "",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Scaffold ID: `{payload['scaffold_id']}`",
        f"- Row count: `{payload['row_count']}`",
        f"- Benchmark-ready rows: `{payload['benchmark_ready_count']}`",
        f"- Shadow-only rows: `{payload['shadow_only_count']}`",
        f"- Readiness class: `{payload['readiness_class']}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        "",
        "## Training blockers",
    ]
    blockers = payload["training_blockers"]
    lines.append("- none" if not blockers else "")
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These rows are local scaffold artifacts. They are not a GPU training run, provider bring-up, promotion run, or reward-math change.",
        ]
    )
    path.write_text("\n".join(lines).replace("\n\n\n", "\n\n") + "\n", encoding="utf-8")


def _resolve_scaffold_and_admission_paths(
    *,
    output_root: Path,
    scaffold_report_path: Optional[str | Path],
    admission_log_path: Optional[str | Path],
    run_scaffold_if_missing: bool,
) -> tuple[Path, Path]:
    if scaffold_report_path is None:
        expected_scaffold = (
            output_root / "scaffold" / "economic_wm_scaffold_report_v1.json"
        )
        if run_scaffold_if_missing:
            from scripts.economic_world_model.build_economic_wm_scaffold import (
                run_build_economic_wm_scaffold,
            )

            scaffold_payload = run_build_economic_wm_scaffold(
                output_dir=output_root / "scaffold"
            )
            scaffold_path = Path(
                scaffold_payload["artifact_refs"]["scaffold_report_path"]
            )
        else:
            scaffold_path = expected_scaffold
    else:
        scaffold_path = Path(scaffold_report_path)

    if admission_log_path is not None:
        return scaffold_path, Path(admission_log_path)

    scaffold_report = load_economic_wm_scaffold_report(scaffold_path)
    bridge_manifest = dict(
        scaffold_report.artifact_refs.get("bridge_manifest", {}) or {}
    )
    candidate = bridge_manifest.get("admission_log_path")
    if not candidate:
        candidate = (
            output_root
            / "scaffold"
            / "entry_preflight"
            / "stage1_bridge_readiness_sweep"
            / "stage1"
            / "governed_video"
            / "proposal_admission_v1.jsonl"
        )
    return scaffold_path, Path(candidate)


def run_materialize_economic_wm_training_rows(
    *,
    output_dir: str | Path,
    scaffold_report_path: Optional[str | Path] = None,
    admission_log_path: Optional[str | Path] = None,
    run_scaffold_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    scaffold_path, admission_path = _resolve_scaffold_and_admission_paths(
        output_root=output_root,
        scaffold_report_path=scaffold_report_path,
        admission_log_path=admission_log_path,
        run_scaffold_if_missing=run_scaffold_if_missing,
    )
    rows_path = output_root / "economic_wm_replay_feature_rows_v1.jsonl"
    manifest_path = output_root / "economic_wm_training_corpus_manifest_v1.json"
    markdown_path = output_root / "economic_wm_training_corpus_manifest_v1.md"
    manifest = materialize_economic_wm_training_corpus_from_paths(
        scaffold_report_path=scaffold_path,
        admission_log_path=admission_path,
        rows_path=rows_path,
        manifest_path=manifest_path,
        metadata={"source": "materialize_economic_wm_training_rows_script"},
    )
    payload = manifest.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "manifest_path": str(manifest_path),
        "rows_path": str(rows_path),
        "markdown_path": str(markdown_path),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMTrainingCorpusManifest.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_training_rows",
        help="Directory for row corpus artifacts.",
    )
    parser.add_argument(
        "--scaffold-report",
        default=None,
        help="Optional existing economic_wm_scaffold_report_v1.json.",
    )
    parser.add_argument(
        "--admission-log",
        default=None,
        help="Optional Stage-1 proposal_admission_v1.jsonl path.",
    )
    parser.add_argument(
        "--no-run-scaffold",
        action="store_true",
        help="Do not run the scaffold builder when no scaffold report is provided.",
    )
    args = parser.parse_args()
    payload = run_materialize_economic_wm_training_rows(
        output_dir=args.output_dir,
        scaffold_report_path=args.scaffold_report,
        admission_log_path=args.admission_log,
        run_scaffold_if_missing=not args.no_run_scaffold,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["row_count"] > 0 and not payload["promotion_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
