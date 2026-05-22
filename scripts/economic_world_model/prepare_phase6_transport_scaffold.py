#!/usr/bin/env python3
"""Prepare local Phase-6.0-6.2 cross-WM transport scaffold artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.economic_world_model.sweep_economic_wm_lower_wm_maturity import (  # noqa: E402
    run_sweep_economic_wm_lower_wm_maturity,
)
from src.world_model.economic_world_model import (  # noqa: E402
    load_economic_wm_lower_wm_maturity_rows,
    load_economic_wm_lower_wm_maturity_sweep,
    load_economic_wm_phase5_local_prep_manifest,
)
from src.world_model.transport import (  # noqa: E402
    build_per_wm_transformer_registry,
    build_wm_transport_contract_pack,
    build_wm_transport_phase6_scaffold_report,
    build_wm_transport_roundtrip_receipts,
    build_wm_transport_training_rows,
    save_per_wm_transformer_registry,
    save_wm_transport_contract_pack,
    save_wm_transport_phase6_scaffold_report,
    save_wm_transport_roundtrip_receipts,
    save_wm_transport_training_rows,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/economic_world_model/phase6_transport_scaffold")
DEFAULT_PHASE5_PREP = Path(
    "artifacts/economic_world_model/economic_wm_phase5_local_prep/"
    "economic_wm_phase5_local_prep_manifest_v1.json"
)
DEFAULT_MATURITY_SWEEP = Path(
    "artifacts/economic_world_model/economic_wm_lower_wm_maturity_sweep/"
    "economic_wm_lower_wm_maturity_sweep_v1.json"
)
DEFAULT_MATURITY_ROWS = Path(
    "artifacts/economic_world_model/economic_wm_lower_wm_maturity_sweep/"
    "economic_wm_lower_wm_maturity_rows_v1.jsonl"
)


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6 Transport Scaffold Artifact",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Contracts: `{payload['contract_count']}`",
        f"- Transformers: `{payload['transformer_count']}`",
        f"- Round-trip receipts: `{payload['roundtrip_receipt_count']}`",
        f"- Training rows: `{payload['training_row_count']}`",
        f"- Ready for Phase 6.3 neural scaffold: `{str(payload['ready_for_phase6_3_neural_scaffold']).lower()}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        "",
        "## Boundary",
        "",
        "This artifact proves local contracts, per-WM transformer posture, row shapes,",
        "and round-trip/topology/uncertainty receipt surfaces. It does not train",
        "transport bridges, invoke providers, execute hardware, promote outputs,",
        "grant live control, or mutate frozen reward/trust/`w_econ`/lambda math.",
        "",
        "## Artifact refs",
        "",
    ]
    for key, value in sorted(dict(payload.get("artifact_refs", {}) or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(
    *,
    output_dir: Path,
    phase5_prep_path: Optional[str | Path],
    maturity_sweep_path: Optional[str | Path],
    maturity_rows_path: Optional[str | Path],
    run_dependencies_if_missing: bool,
) -> tuple[Path, Path, Path]:
    phase5 = Path(phase5_prep_path or DEFAULT_PHASE5_PREP)
    sweep = Path(maturity_sweep_path or DEFAULT_MATURITY_SWEEP)
    rows = Path(maturity_rows_path or DEFAULT_MATURITY_ROWS)
    if phase5.exists() and sweep.exists() and rows.exists():
        return phase5, sweep, rows
    if not run_dependencies_if_missing:
        missing = [str(path) for path in [phase5, sweep, rows] if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase-6 transport inputs: " + ", ".join(missing)
        )
    result = run_sweep_economic_wm_lower_wm_maturity(
        output_dir=output_dir.parent / "economic_wm_lower_wm_maturity_sweep",
        run_dependencies_if_missing=True,
    )
    sweep = Path(result["artifact_refs"]["sweep_path"])
    rows = Path(result["artifact_refs"]["maturity_rows_path"])
    return phase5, sweep, rows


def run_prepare_phase6_transport_scaffold(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase5_prep_path: Optional[str | Path] = None,
    maturity_sweep_path: Optional[str | Path] = None,
    maturity_rows_path: Optional[str | Path] = None,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase5_path, sweep_path, rows_path = _resolve_inputs(
        output_dir=output,
        phase5_prep_path=phase5_prep_path,
        maturity_sweep_path=maturity_sweep_path,
        maturity_rows_path=maturity_rows_path,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    phase5 = load_economic_wm_phase5_local_prep_manifest(phase5_path)
    maturity_sweep = load_economic_wm_lower_wm_maturity_sweep(sweep_path)
    maturity_rows = load_economic_wm_lower_wm_maturity_rows(rows_path)

    pack_path = output / "wm_transport_contract_pack_v1.json"
    contracts_path = output / "wm_transport_bridge_contracts_v1.jsonl"
    registry_path = output / "per_wm_transport_transformer_registry_v1.json"
    receipts_path = output / "wm_transport_roundtrip_receipts_v1.jsonl"
    training_manifest_path = output / "wm_transport_training_manifest_v1.json"
    training_rows_path = output / "wm_transport_training_rows_v1.jsonl"
    report_path = output / "wm_transport_phase6_scaffold_report_v1.json"
    markdown_path = output / "wm_transport_phase6_scaffold_v1.md"

    artifact_refs = {
        "phase5_manifest_path": str(phase5_path),
        "maturity_sweep_path": str(sweep_path),
        "maturity_rows_path": str(rows_path),
        "pack_path": str(pack_path),
        "contracts_path": str(contracts_path),
        "registry_path": str(registry_path),
        "roundtrip_receipts_path": str(receipts_path),
        "training_manifest_path": str(training_manifest_path),
        "training_rows_path": str(training_rows_path),
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
    }

    pack, contracts = build_wm_transport_contract_pack(
        maturity_sweep=maturity_sweep,
        maturity_rows=maturity_rows,
        phase5_manifest=phase5,
        contract_path=contracts_path,
        artifact_refs=artifact_refs,
    )
    save_wm_transport_contract_pack(pack_path=pack_path, pack=pack, contracts=contracts)

    registry = build_per_wm_transformer_registry(
        contract_pack_id=pack.pack_id,
        contracts=contracts,
        artifact_refs=artifact_refs,
    )
    save_per_wm_transformer_registry(registry_path, registry)

    receipts = build_wm_transport_roundtrip_receipts(
        contracts=contracts, transformer_registry=registry
    )
    save_wm_transport_roundtrip_receipts(receipts_path, receipts)

    training_manifest, training_rows = build_wm_transport_training_rows(
        contract_pack=pack,
        contracts=contracts,
        transformer_registry=registry,
        roundtrip_receipts=receipts,
        rows_path=training_rows_path,
        artifact_refs=artifact_refs,
    )
    save_wm_transport_training_rows(
        manifest_path=training_manifest_path,
        manifest=training_manifest,
        rows=training_rows,
    )

    report = build_wm_transport_phase6_scaffold_report(
        contract_pack=pack,
        transformer_registry=registry,
        training_manifest=training_manifest,
        roundtrip_receipt_count=len(receipts),
        artifact_refs=artifact_refs,
    )
    save_wm_transport_phase6_scaffold_report(report_path, report)
    payload = report.to_dict()
    _write_markdown(markdown_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase5-prep", default=None)
    parser.add_argument("--maturity-sweep", default=None)
    parser.add_argument("--maturity-rows", default=None)
    parser.add_argument(
        "--no-run-dependencies",
        action="store_true",
        help="Fail instead of materializing missing prerequisite artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_prepare_phase6_transport_scaffold(
        output_dir=args.output_dir,
        phase5_prep_path=args.phase5_prep,
        maturity_sweep_path=args.maturity_sweep,
        maturity_rows_path=args.maturity_rows,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )


if __name__ == "__main__":
    main()
