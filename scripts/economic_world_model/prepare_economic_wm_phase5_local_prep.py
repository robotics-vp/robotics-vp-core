#!/usr/bin/env python3
"""Prepare Economic WM Phase-5 local datapack/temporal/join rows."""

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
    EconomicWMPhase5LocalPrepManifest,
    build_economic_wm_phase5_local_prep_from_paths,
)


def _write_markdown(path: Path, manifest: EconomicWMPhase5LocalPrepManifest) -> None:
    payload = manifest.to_dict()
    lines = [
        "# Economic WM Phase-5 Local Prep",
        "",
        f"- Manifest ID: `{payload['manifest_id']}`",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Lower-WM preflight ID: `{payload['lower_wm_preflight_id']}`",
        f"- Resource ingestion manifest ID: `{payload['resource_ingestion_manifest_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Composition rows: `{payload['composition_row_count']}`",
        f"- Counterfactual/value joins: `{payload['counterfactual_value_join_count']}`",
        f"- Temporal windows: `{payload['temporal_window_count']}`",
        f"- Authority class: `{payload['authority_class']}`",
        f"- Ready for trainer scaffold: `{str(payload['ready_for_trainer_scaffold']).lower()}`",
        f"- Ready for GPU training: `{str(payload['ready_for_gpu_training']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        f"- Reward math mutation: `{str(payload['reward_math_mutation']).lower()}`",
        "",
        "## Row families",
        "",
        "- `economic_wm_datapack_composition_row_v1`",
        "- `economic_wm_counterfactual_value_join_row_v1`",
        "- `economic_wm_temporal_window_row_v1`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This pass deepens local Economic WM ingestion beyond Stage-1, but remains scaffold-only. It does not run GPU training, provider bring-up, live allocation, promotion, or reward-math mutation.",
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


def _resolve_lower_wm_preflight(
    *,
    output_root: Path,
    lower_wm_preflight_path: Optional[str | Path],
    canonical_consumption_rows_path: Optional[str | Path],
    corpus_manifest_path: Path,
    rows_path: Path,
    run_if_missing: bool,
) -> tuple[Path, Path]:
    preflight = Path(
        lower_wm_preflight_path
        or "artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.json"
    )
    consumption_rows = Path(
        canonical_consumption_rows_path
        or "artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_canonical_consumption_rows_v1.jsonl"
    )
    if preflight.exists() and consumption_rows.exists():
        return preflight, consumption_rows
    if not run_if_missing:
        missing = preflight if not preflight.exists() else consumption_rows
        raise FileNotFoundError(missing)
    from scripts.economic_world_model.prepare_economic_wm_lower_wm_consumption_preflight import (  # noqa: E501
        run_prepare_economic_wm_lower_wm_consumption_preflight,
    )

    payload = run_prepare_economic_wm_lower_wm_consumption_preflight(
        output_dir=output_root / "lower_wm_consumption_preflight",
        corpus_manifest_path=corpus_manifest_path,
        rows_path=rows_path,
    )
    return (
        Path(payload["artifact_refs"]["preflight_path"]),
        Path(payload["artifact_refs"]["consumption_rows_path"]),
    )


def _resolve_resource_surfaces(
    *,
    output_root: Path,
    resource_manifest_path: Optional[str | Path],
    resource_receipts_path: Optional[str | Path],
    queue_telemetry_surfaces_path: Optional[str | Path],
    corpus_manifest_path: Path,
    rows_path: Path,
    run_if_missing: bool,
) -> tuple[Path, Path, Path]:
    manifest = Path(
        resource_manifest_path
        or "artifacts/economic_world_model/economic_wm_resource_surfaces/economic_wm_resource_ingestion_manifest_v1.json"
    )
    receipts = Path(
        resource_receipts_path
        or "artifacts/economic_world_model/economic_wm_resource_surfaces/economic_wm_resource_receipts_v1.jsonl"
    )
    telemetry = Path(
        queue_telemetry_surfaces_path
        or "artifacts/economic_world_model/economic_wm_resource_surfaces/economic_wm_queue_telemetry_surfaces_v1.jsonl"
    )
    if manifest.exists() and receipts.exists() and telemetry.exists():
        return manifest, receipts, telemetry
    if not run_if_missing:
        for path in (manifest, receipts, telemetry):
            if not path.exists():
                raise FileNotFoundError(path)
    from scripts.economic_world_model.prepare_economic_wm_resource_surfaces import (
        run_prepare_economic_wm_resource_surfaces,
    )

    payload = run_prepare_economic_wm_resource_surfaces(
        output_dir=output_root / "resource_surfaces",
        corpus_manifest_path=corpus_manifest_path,
        rows_path=rows_path,
    )
    return (
        Path(payload["artifact_refs"]["manifest_path"]),
        Path(payload["artifact_refs"]["receipts_path"]),
        Path(payload["artifact_refs"]["telemetry_surfaces_path"]),
    )


def run_prepare_economic_wm_phase5_local_prep(
    *,
    output_dir: str | Path,
    corpus_manifest_path: Optional[str | Path] = None,
    rows_path: Optional[str | Path] = None,
    lower_wm_preflight_path: Optional[str | Path] = None,
    canonical_consumption_rows_path: Optional[str | Path] = None,
    resource_manifest_path: Optional[str | Path] = None,
    resource_receipts_path: Optional[str | Path] = None,
    queue_telemetry_surfaces_path: Optional[str | Path] = None,
    window_size: int = 2,
    run_dependencies_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path, resolved_rows_path = _resolve_rows(
        output_root=output_root,
        corpus_manifest_path=corpus_manifest_path,
        rows_path=rows_path,
        run_rows_if_missing=run_dependencies_if_missing,
    )
    resolved_lower_preflight, resolved_consumption_rows = _resolve_lower_wm_preflight(
        output_root=output_root,
        lower_wm_preflight_path=lower_wm_preflight_path,
        canonical_consumption_rows_path=canonical_consumption_rows_path,
        corpus_manifest_path=resolved_manifest_path,
        rows_path=resolved_rows_path,
        run_if_missing=run_dependencies_if_missing,
    )
    resolved_resource_manifest, resolved_receipts, resolved_telemetry = (
        _resolve_resource_surfaces(
            output_root=output_root,
            resource_manifest_path=resource_manifest_path,
            resource_receipts_path=resource_receipts_path,
            queue_telemetry_surfaces_path=queue_telemetry_surfaces_path,
            corpus_manifest_path=resolved_manifest_path,
            rows_path=resolved_rows_path,
            run_if_missing=run_dependencies_if_missing,
        )
    )
    phase5_manifest_path = (
        output_root / "economic_wm_phase5_local_prep_manifest_v1.json"
    )
    composition_rows_path = (
        output_root / "economic_wm_datapack_composition_rows_v1.jsonl"
    )
    joins_path = output_root / "economic_wm_counterfactual_value_joins_v1.jsonl"
    windows_path = output_root / "economic_wm_temporal_windows_v1.jsonl"
    markdown_path = output_root / "economic_wm_phase5_local_prep_v1.md"
    manifest = build_economic_wm_phase5_local_prep_from_paths(
        corpus_manifest_path=resolved_manifest_path,
        rows_path=resolved_rows_path,
        lower_wm_preflight_path=resolved_lower_preflight,
        canonical_consumption_rows_path=resolved_consumption_rows,
        resource_manifest_path=resolved_resource_manifest,
        resource_receipts_path=resolved_receipts,
        queue_telemetry_surfaces_path=resolved_telemetry,
        manifest_path=phase5_manifest_path,
        composition_rows_path=composition_rows_path,
        counterfactual_value_joins_path=joins_path,
        temporal_windows_path=windows_path,
        window_size=window_size,
        metadata={"source": "prepare_economic_wm_phase5_local_prep_script"},
    )
    payload = manifest.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "manifest_path": str(phase5_manifest_path),
        "composition_rows_path": str(composition_rows_path),
        "counterfactual_value_joins_path": str(joins_path),
        "temporal_windows_path": str(windows_path),
        "markdown_path": str(markdown_path),
    }
    phase5_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMPhase5LocalPrepManifest.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_phase5_local_prep",
        help="Directory for Economic WM Phase-5 local prep artifacts.",
    )
    parser.add_argument("--corpus-manifest", default=None)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--lower-wm-preflight", default=None)
    parser.add_argument("--lower-wm-consumption-rows", default=None)
    parser.add_argument("--resource-manifest", default=None)
    parser.add_argument("--resource-receipts", default=None)
    parser.add_argument("--queue-telemetry-surfaces", default=None)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument(
        "--no-run-dependencies",
        action="store_true",
        help="Do not run missing row/lower-WM/resource dependency CLIs.",
    )
    args = parser.parse_args()
    payload = run_prepare_economic_wm_phase5_local_prep(
        output_dir=args.output_dir,
        corpus_manifest_path=args.corpus_manifest,
        rows_path=args.rows,
        lower_wm_preflight_path=args.lower_wm_preflight,
        canonical_consumption_rows_path=args.lower_wm_consumption_rows,
        resource_manifest_path=args.resource_manifest,
        resource_receipts_path=args.resource_receipts,
        queue_telemetry_surfaces_path=args.queue_telemetry_surfaces,
        window_size=args.window_size,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] == "ok"
        and payload["ready_for_trainer_scaffold"]
        and not payload["ready_for_gpu_training"]
        and not payload["promotion_eligible"]
        and not payload["reward_math_mutation"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
