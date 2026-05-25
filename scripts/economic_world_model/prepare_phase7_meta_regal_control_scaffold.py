#!/usr/bin/env python3
"""Materialize Phase 7 Meta-Regal-Node / control WM scaffold artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.economic_world_model.audit_phase35_4_65_local_closure import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE35465_CLOSURE_DIR,
)
from scripts.economic_world_model.audit_phase35_4_65_local_closure import (  # noqa: E402
    run_audit_phase35_4_65_local_closure,
)
from scripts.economic_world_model.prepare_phase65_meta_node_neuralization import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE65_DIR,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase7_meta_regal_control_scaffold,
    load_phase35465_local_closure_audit,
    load_phase65_meta_node_neuralization_report,
    save_phase7_meta_regal_control_scaffold,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase7_meta_regal_control_scaffold"
)


def _paths(output: Path, phase65_dir: Path, closure_dir: Path) -> dict[str, Path]:
    return {
        "phase65_report": phase65_dir / "phase65_meta_node_neuralization_report_v1.json",
        "closure_audit": closure_dir / "phase35_4_65_local_closure_audit_v1.json",
        "report": output / "phase7_meta_regal_control_scaffold_report_v1.json",
        "governance_node_surfaces": output
        / "phase7_governance_node_surfaces_v1.jsonl",
        "composition_modes": output / "phase7_composition_mode_specs_v1.jsonl",
        "conflict_receipts": output
        / "phase7_conflict_override_receipts_v1.jsonl",
        "admissible_regions": output / "phase7_admissible_region_specs_v1.jsonl",
        "control_fields": output / "phase7_control_field_slots_v1.jsonl",
        "training_rows": output / "phase7_training_row_slots_v1.jsonl",
        "promotion_gates": output / "phase7_promotion_gates_v1.jsonl",
        "markdown": output / "phase7_meta_regal_control_scaffold_report_v1.md",
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 7 Meta-Regal-Node / Control WM Scaffold",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local Phase 7 scaffold complete: "
        f"`{str(payload['local_phase7_scaffold_complete']).lower()}`",
        "- Ready for runtime wiring: "
        f"`{str(payload['ready_for_runtime_wiring']).lower()}`",
        f"- Runtime wiring executed: "
        f"`{str(payload['runtime_wiring_executed']).lower()}`",
        f"- Phase 7 authority granted: "
        f"`{str(payload['phase7_authority_granted']).lower()}`",
        f"- Live control authority: "
        f"`{str(payload['live_control_authority']).lower()}`",
        f"- Governance node surfaces: `{payload['governance_node_surface_count']}`",
        f"- Composition modes: `{payload['composition_mode_count']}`",
        f"- Conflict / override receipts: "
        f"`{payload['conflict_override_receipt_count']}`",
        f"- Admissible regions: `{payload['admissible_region_count']}`",
        f"- Control field slots: `{payload['control_field_slot_count']}`",
        f"- Training row slots: `{payload['training_row_slot_count']}`",
        f"- Promotion gates: `{payload['promotion_gate_count']}`",
        "",
        "## Boundary",
        "",
        "These are Stage A typed non-neural governance scaffolds only. They",
        "materialize domain-governance node surfaces, composition modes,",
        "conflict and override receipts, admissible regions, shadow control",
        "field slots, training-row slots, and denied promotion gates. They do",
        "not train, write weights, run providers or hardware, claim Unitree",
        "runtime evidence, mutate reward math, promote outputs, replace lower",
        "WMs, collapse governance into a scalar score, or control live policy.",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(
    *,
    phase65_dir: Path,
    closure_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _paths(Path("."), phase65_dir, closure_dir)
    needed = {
        "phase65_report": paths["phase65_report"],
        "closure_audit": paths["closure_audit"],
    }
    if all(path.exists() for path in needed.values()):
        return needed
    if not run_dependencies_if_missing:
        missing = [str(path) for path in needed.values() if not path.exists()]
        raise FileNotFoundError("Missing Phase 7 scaffold inputs: " + ", ".join(missing))
    run_audit_phase35_4_65_local_closure(
        output_dir=closure_dir,
        phase65_dir=phase65_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in needed.values()):
        missing = [str(path) for path in needed.values() if not path.exists()]
        raise FileNotFoundError(
            "Phase 7 dependency builders did not materialize: " + ", ".join(missing)
        )
    return needed


def run_prepare_phase7_meta_regal_control_scaffold(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase65_dir: str | Path = DEFAULT_PHASE65_DIR,
    closure_dir: str | Path = DEFAULT_PHASE35465_CLOSURE_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase65_path = Path(phase65_dir)
    closure_path = Path(closure_dir)
    input_paths = _resolve_inputs(
        phase65_dir=phase65_path,
        closure_dir=closure_path,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    paths = _paths(output, phase65_path, closure_path)
    refs = {f"{key}_path": str(path) for key, path in paths.items()}
    (
        report,
        surfaces,
        modes,
        conflict_receipts,
        regions,
        control_fields,
        training_rows,
        gates,
    ) = build_phase7_meta_regal_control_scaffold(
        phase65_report=load_phase65_meta_node_neuralization_report(
            input_paths["phase65_report"]
        ),
        closure_audit=load_phase35465_local_closure_audit(
            input_paths["closure_audit"]
        ),
        artifact_refs=refs,
    )
    saved_refs = save_phase7_meta_regal_control_scaffold(
        output,
        report,
        surfaces,
        modes,
        conflict_receipts,
        regions,
        control_fields,
        training_rows,
        gates,
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(paths["markdown"], payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase65-dir", default=str(DEFAULT_PHASE65_DIR))
    parser.add_argument("--closure-dir", default=str(DEFAULT_PHASE35465_CLOSURE_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase7_meta_regal_control_scaffold(
        output_dir=args.output_dir,
        phase65_dir=args.phase65_dir,
        closure_dir=args.closure_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_phase7_scaffold_complete"]
        and payload["ready_for_runtime_wiring"]
        and not payload["runtime_wiring_executed"]
        and not payload["phase7_authority_granted"]
        and not payload["live_control_authority"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["hardware_executed"]
        and not payload["unitree_sim_runtime_executed"]
        and not payload["live_policy_control"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
