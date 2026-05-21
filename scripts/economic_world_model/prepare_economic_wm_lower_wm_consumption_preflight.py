#!/usr/bin/env python3
"""Prepare Economic WM lower-WM canonical-consumption preflight artifacts."""

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
    EconomicWMLowerWMConsumptionPreflight,
    build_economic_wm_lower_wm_consumption_preflight_from_paths,
)


def _write_markdown(
    path: Path, preflight: EconomicWMLowerWMConsumptionPreflight
) -> None:
    payload = preflight.to_dict()
    lines = [
        "# Economic WM Lower-WM Consumption Preflight",
        "",
        f"- Preflight ID: `{payload['preflight_id']}`",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Row count: `{payload['row_count']}`",
        f"- All required WMs referenced: `{str(payload['all_required_wms_referenced']).lower()}`",
        f"- Ready for neural manifest: `{str(payload['ready_for_neural_manifest']).lower()}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        "",
        "## Required WMs",
    ]
    lines.extend(f"- `{key}`" for key in payload["required_wm_keys"])
    lines.extend(["", "## Aggregate counts"])
    for key, value in payload["aggregate_counts"].items():
        lines.append(f"- `{key}`: {value}")
    if payload["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This preflight only proves Economic WM rows carry canonical lower-WM state references. It does not run GPU training, provider bring-up, benchmark promotion, or reward-math mutation.",
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

    from scripts.economic_world_model.materialize_economic_wm_training_rows import (  # noqa: E501
        run_materialize_economic_wm_training_rows,
    )

    payload = run_materialize_economic_wm_training_rows(
        output_dir=output_root / "training_rows"
    )
    return (
        Path(payload["artifact_refs"]["manifest_path"]),
        Path(payload["artifact_refs"]["rows_path"]),
    )


def run_prepare_economic_wm_lower_wm_consumption_preflight(
    *,
    output_dir: str | Path,
    corpus_manifest_path: Optional[str | Path] = None,
    rows_path: Optional[str | Path] = None,
    run_rows_if_missing: bool = True,
    compile_missing_refs: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path, resolved_rows_path = _resolve_rows(
        output_root=output_root,
        corpus_manifest_path=corpus_manifest_path,
        rows_path=rows_path,
        run_rows_if_missing=run_rows_if_missing,
    )
    preflight_path = output_root / "economic_wm_lower_wm_consumption_preflight_v1.json"
    consumption_rows_path = (
        output_root / "economic_wm_canonical_consumption_rows_v1.jsonl"
    )
    markdown_path = output_root / "economic_wm_lower_wm_consumption_preflight_v1.md"
    preflight = build_economic_wm_lower_wm_consumption_preflight_from_paths(
        corpus_manifest_path=manifest_path,
        rows_path=resolved_rows_path,
        output_dir=output_root,
        preflight_path=preflight_path,
        consumption_rows_path=consumption_rows_path,
        compile_missing_refs=compile_missing_refs,
        metadata={
            "source": "prepare_economic_wm_lower_wm_consumption_preflight_script"
        },
    )
    payload = preflight.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "preflight_path": str(preflight_path),
        "consumption_rows_path": str(consumption_rows_path),
        "markdown_path": str(markdown_path),
        "corpus_manifest_path": str(manifest_path),
        "rows_path": str(resolved_rows_path),
    }
    preflight_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(
        markdown_path, EconomicWMLowerWMConsumptionPreflight.from_dict(payload)
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight",
        help="Directory for lower-WM consumption preflight artifacts.",
    )
    parser.add_argument("--corpus-manifest", default=None)
    parser.add_argument("--rows", default=None)
    parser.add_argument(
        "--no-run-rows",
        action="store_true",
        help="Do not materialize training rows if row inputs are missing.",
    )
    parser.add_argument(
        "--no-compile-missing-refs",
        action="store_true",
        help="Do not compile local canonical lower-WM reference artifacts when rows lack direct refs.",
    )
    args = parser.parse_args()
    payload = run_prepare_economic_wm_lower_wm_consumption_preflight(
        output_dir=args.output_dir,
        corpus_manifest_path=args.corpus_manifest,
        rows_path=args.rows,
        run_rows_if_missing=not args.no_run_rows,
        compile_missing_refs=not args.no_compile_missing_refs,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] == "ok"
        and payload["all_required_wms_referenced"]
        and payload["ready_for_neural_manifest"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
