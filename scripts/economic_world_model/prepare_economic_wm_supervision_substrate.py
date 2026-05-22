#!/usr/bin/env python3
"""Materialize typed Economic WM supervision records from Phase-5 joins."""

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
    EconomicWMSupervisionManifest,
    build_economic_wm_supervision_substrate_from_paths,
)


def _write_markdown(path: Path, manifest: EconomicWMSupervisionManifest) -> None:
    payload = manifest.to_dict()
    lines = [
        "# Economic WM Supervision Substrate",
        "",
        f"- Manifest ID: `{payload['manifest_id']}`",
        f"- Phase-5 manifest ID: `{payload['phase5_manifest_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Records: `{payload['record_count']}`",
        f"- Ready records: `{payload['ready_record_count']}`",
        f"- Counterfactual evals: `{payload['counterfactual_eval_count']}`",
        f"- Value target packs: `{payload['value_target_pack_count']}`",
        f"- Value ledger receipts: `{payload['value_ledger_receipt_count']}`",
        f"- Ready for shadow outcome loop: `{str(payload['ready_for_shadow_outcome_loop']).lower()}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        f"- Reward math mutation: `{str(payload['reward_math_mutation']).lower()}`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This materializes typed counterfactual/value supervision records from existing Phase-5 refs. It does not train, invoke providers, promote outputs, or mutate reward math.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_phase5(
    *, output_root: Path, phase5_prep_path: Optional[str | Path], run_if_missing: bool
) -> Path:
    path = Path(
        phase5_prep_path
        or "artifacts/economic_world_model/economic_wm_phase5_local_prep/economic_wm_phase5_local_prep_manifest_v1.json"
    )
    if path.exists():
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    from scripts.economic_world_model.prepare_economic_wm_phase5_local_prep import (
        run_prepare_economic_wm_phase5_local_prep,
    )

    payload = run_prepare_economic_wm_phase5_local_prep(
        output_dir=output_root / "phase5_local_prep"
    )
    return Path(payload["artifact_refs"]["manifest_path"])


def run_prepare_economic_wm_supervision_substrate(
    *,
    output_dir: str | Path,
    phase5_prep_path: Optional[str | Path] = None,
    run_dependencies_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_phase5 = _resolve_phase5(
        output_root=output_root,
        phase5_prep_path=phase5_prep_path,
        run_if_missing=run_dependencies_if_missing,
    )
    manifest_path = output_root / "economic_wm_supervision_manifest_v1.json"
    records_path = output_root / "economic_wm_supervision_records_v1.jsonl"
    markdown_path = output_root / "economic_wm_supervision_substrate_v1.md"
    manifest = build_economic_wm_supervision_substrate_from_paths(
        phase5_prep_path=resolved_phase5,
        manifest_path=manifest_path,
        records_path=records_path,
        metadata={"source": "prepare_economic_wm_supervision_substrate_script"},
    )
    payload = manifest.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "manifest_path": str(manifest_path),
        "records_path": str(records_path),
        "markdown_path": str(markdown_path),
        "phase5_prep_path": str(resolved_phase5),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMSupervisionManifest.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_supervision_substrate",
    )
    parser.add_argument("--phase5-prep", default=None)
    parser.add_argument("--no-run-dependencies", action="store_true")
    args = parser.parse_args()
    payload = run_prepare_economic_wm_supervision_substrate(
        output_dir=args.output_dir,
        phase5_prep_path=args.phase5_prep,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] == "ok"
        and payload["ready_for_shadow_outcome_loop"]
        and not payload["ready_for_training"]
        and not payload["promotion_eligible"]
        and not payload["reward_math_mutation"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
