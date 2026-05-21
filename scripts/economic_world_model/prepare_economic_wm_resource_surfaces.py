#!/usr/bin/env python3
"""Prepare Economic WM Phase-5 resource/compute receipt surfaces."""

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
    EconomicWMResourceIngestionManifest,
    build_economic_wm_resource_surfaces_from_paths,
)


def _write_markdown(path: Path, manifest: EconomicWMResourceIngestionManifest) -> None:
    payload = manifest.to_dict()
    lines = [
        "# Economic WM Resource Surfaces",
        "",
        f"- Manifest ID: `{payload['manifest_id']}`",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Row count: `{payload['row_count']}`",
        f"- Receipt count: `{payload['receipt_count']}`",
        f"- Companion-compute contracts: `{payload['contract_count']}`",
        f"- Degraded-mode runbooks: `{payload['runbook_count']}`",
        f"- Queue telemetry surfaces: `{payload['telemetry_surface_count']}`",
        f"- Ready for Phase-5 local prep: `{str(payload['ready_for_phase5_local_prep']).lower()}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        f"- Reward math mutation: `{str(payload['reward_math_mutation']).lower()}`",
        "",
        "## Economic WM ingestion slots",
    ]
    lines.extend(f"- `{slot}`" for slot in payload["economic_wm_ingestion_slots"])
    lines.extend(["", "## Allocatable budget objects"])
    lines.extend(f"- `{obj}`" for obj in payload["allocatable_budget_objects"])
    lines.extend(["", "## Blockers"])
    lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These are local receipt schemas and ingestion slots for capacity, latency, thermal, battery, companion compute, degraded modes, and queues. They do not run GPU training, provider bring-up, live control, promotion, or reward-math mutation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_rows(
    *,
    output_root: Path,
    corpus_manifest_path: Optional[str | Path],
    rows_path: Optional[str | Path],
    run_rows_if_missing: bool,
) -> tuple[Path, Path]:
    manifest = Path(
        corpus_manifest_path
        or "artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json"
    )
    rows = Path(
        rows_path
        or "artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl"
    )
    if manifest.exists() and rows.exists():
        return manifest, rows
    if not run_rows_if_missing:
        missing = manifest if not manifest.exists() else rows
        raise FileNotFoundError(missing)

    from scripts.economic_world_model.materialize_economic_wm_training_rows import (
        run_materialize_economic_wm_training_rows,
    )

    payload = run_materialize_economic_wm_training_rows(
        output_dir=output_root / "training_rows"
    )
    return (
        Path(payload["artifact_refs"]["manifest_path"]),
        Path(payload["artifact_refs"]["rows_path"]),
    )


def run_prepare_economic_wm_resource_surfaces(
    *,
    output_dir: str | Path,
    corpus_manifest_path: Optional[str | Path] = None,
    rows_path: Optional[str | Path] = None,
    run_rows_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path, resolved_rows_path = _resolve_rows(
        output_root=output_root,
        corpus_manifest_path=corpus_manifest_path,
        rows_path=rows_path,
        run_rows_if_missing=run_rows_if_missing,
    )
    resource_manifest_path = (
        output_root / "economic_wm_resource_ingestion_manifest_v1.json"
    )
    receipts_path = output_root / "economic_wm_resource_receipts_v1.jsonl"
    contracts_path = output_root / "economic_wm_companion_compute_contracts_v1.jsonl"
    runbooks_path = output_root / "economic_wm_degraded_mode_runbooks_v1.jsonl"
    telemetry_path = output_root / "economic_wm_queue_telemetry_surfaces_v1.jsonl"
    markdown_path = output_root / "economic_wm_resource_surfaces_v1.md"
    manifest = build_economic_wm_resource_surfaces_from_paths(
        corpus_manifest_path=manifest_path,
        rows_path=resolved_rows_path,
        manifest_path=resource_manifest_path,
        receipts_path=receipts_path,
        contracts_path=contracts_path,
        degraded_runbooks_path=runbooks_path,
        telemetry_surfaces_path=telemetry_path,
        metadata={"source": "prepare_economic_wm_resource_surfaces_script"},
    )
    payload = manifest.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "manifest_path": str(resource_manifest_path),
        "receipts_path": str(receipts_path),
        "contracts_path": str(contracts_path),
        "degraded_runbooks_path": str(runbooks_path),
        "telemetry_surfaces_path": str(telemetry_path),
        "markdown_path": str(markdown_path),
        "corpus_manifest_path": str(manifest_path),
        "rows_path": str(resolved_rows_path),
    }
    resource_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(
        markdown_path, EconomicWMResourceIngestionManifest.from_dict(payload)
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_resource_surfaces",
        help="Directory for Phase-5 resource surface artifacts.",
    )
    parser.add_argument("--corpus-manifest", default=None)
    parser.add_argument("--rows", default=None)
    parser.add_argument(
        "--no-run-rows",
        action="store_true",
        help="Do not materialize rows if resource-surface inputs are missing.",
    )
    args = parser.parse_args()
    payload = run_prepare_economic_wm_resource_surfaces(
        output_dir=args.output_dir,
        corpus_manifest_path=args.corpus_manifest,
        rows_path=args.rows,
        run_rows_if_missing=not args.no_run_rows,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] == "ok"
        and payload["ready_for_phase5_local_prep"]
        and not payload["ready_for_training"]
        and not payload["promotion_eligible"]
        and not payload["reward_math_mutation"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
