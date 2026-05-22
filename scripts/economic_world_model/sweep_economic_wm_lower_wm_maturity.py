#!/usr/bin/env python3
"""Sweep lower-WM maturity for Economic WM canonical refs."""

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
    EconomicWMLowerWMMaturitySweep,
    build_economic_wm_lower_wm_maturity_sweep_from_paths,
)


def _write_markdown(path: Path, sweep: EconomicWMLowerWMMaturitySweep) -> None:
    payload = sweep.to_dict()
    lines = [
        "# Economic WM Lower-WM Maturity Sweep",
        "",
        f"- Sweep ID: `{payload['sweep_id']}`",
        f"- Phase-5 manifest ID: `{payload['phase5_manifest_id']}`",
        f"- Lower-WM preflight ID: `{payload['lower_wm_preflight_id']}`",
        f"- Resource manifest ID: `{payload['resource_manifest_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Maturity rows: `{payload['maturity_row_count']}`",
        f"- Structural-ready refs: `{payload['structural_ready_count']}`",
        f"- Production-ready refs: `{payload['production_ready_count']}`",
        f"- Ready for Phase-6 contracts: `{str(payload['ready_for_phase6_contracts']).lower()}`",
        f"- Ready for production: `{str(payload['ready_for_production']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        "",
        "## Aggregate counts",
    ]
    for key, value in payload["aggregate_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Blockers"])
    lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This sweep distinguishes structural canonical-ref readiness from production maturity. It does not promote lower WMs, run providers, run hardware, or grant Economic WM authority.",
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


def run_sweep_economic_wm_lower_wm_maturity(
    *,
    output_dir: str | Path,
    phase5_prep_path: Optional[str | Path] = None,
    lower_wm_preflight_path: Optional[str | Path] = None,
    canonical_consumption_rows_path: Optional[str | Path] = None,
    resource_manifest_path: Optional[str | Path] = None,
    run_dependencies_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_phase5 = _resolve_phase5(
        output_root=output_root,
        phase5_prep_path=phase5_prep_path,
        run_if_missing=run_dependencies_if_missing,
    )
    lower_preflight = Path(
        lower_wm_preflight_path
        or "artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.json"
    )
    consumption_rows = Path(
        canonical_consumption_rows_path
        or "artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_canonical_consumption_rows_v1.jsonl"
    )
    resource_manifest = Path(
        resource_manifest_path
        or "artifacts/economic_world_model/economic_wm_resource_surfaces/economic_wm_resource_ingestion_manifest_v1.json"
    )
    for required in (lower_preflight, consumption_rows, resource_manifest):
        if not required.exists():
            if not run_dependencies_if_missing:
                raise FileNotFoundError(required)
            raise FileNotFoundError(
                f"Missing dependency {required}; run Phase-5 local prep/resource builders first"
            )
    sweep_path = output_root / "economic_wm_lower_wm_maturity_sweep_v1.json"
    rows_path = output_root / "economic_wm_lower_wm_maturity_rows_v1.jsonl"
    markdown_path = output_root / "economic_wm_lower_wm_maturity_sweep_v1.md"
    sweep = build_economic_wm_lower_wm_maturity_sweep_from_paths(
        phase5_prep_path=resolved_phase5,
        lower_wm_preflight_path=lower_preflight,
        canonical_consumption_rows_path=consumption_rows,
        resource_manifest_path=resource_manifest,
        sweep_path=sweep_path,
        maturity_rows_path=rows_path,
        metadata={"source": "sweep_economic_wm_lower_wm_maturity_script"},
    )
    payload = sweep.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "sweep_path": str(sweep_path),
        "maturity_rows_path": str(rows_path),
        "markdown_path": str(markdown_path),
        "phase5_prep_path": str(resolved_phase5),
        "lower_wm_preflight_path": str(lower_preflight),
        "canonical_consumption_rows_path": str(consumption_rows),
        "resource_manifest_path": str(resource_manifest),
    }
    sweep_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMLowerWMMaturitySweep.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_lower_wm_maturity_sweep",
    )
    parser.add_argument("--phase5-prep", default=None)
    parser.add_argument("--lower-wm-preflight", default=None)
    parser.add_argument("--lower-wm-consumption-rows", default=None)
    parser.add_argument("--resource-manifest", default=None)
    parser.add_argument("--no-run-dependencies", action="store_true")
    args = parser.parse_args()
    payload = run_sweep_economic_wm_lower_wm_maturity(
        output_dir=args.output_dir,
        phase5_prep_path=args.phase5_prep,
        lower_wm_preflight_path=args.lower_wm_preflight,
        canonical_consumption_rows_path=args.lower_wm_consumption_rows,
        resource_manifest_path=args.resource_manifest,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] == "ok"
        and payload["ready_for_phase6_contracts"]
        and not payload["ready_for_production"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
