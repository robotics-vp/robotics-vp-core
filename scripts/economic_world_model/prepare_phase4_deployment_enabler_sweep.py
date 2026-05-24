#!/usr/bin/env python3
"""Materialize local Phase 4 deployment-enabler contracts and stubs."""

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

from scripts.economic_world_model.prepare_phase35_humanoid_capacity_env_refit import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE35_DIR,
)
from scripts.economic_world_model.prepare_phase35_humanoid_capacity_env_refit import (  # noqa: E402
    run_prepare_phase35_humanoid_capacity_env_refit,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase4_deployment_enabler_sweep,
    load_phase35_humanoid_refit_report,
    save_phase4_deployment_enabler_sweep,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase4_deployment_enabler_sweep"
)


def _phase35_report_path(phase35_dir: Path) -> Path:
    return phase35_dir / "humanoid_phase35_refit_report_v1.json"


def _artifact_refs(output: Path, phase35_path: Path) -> dict[str, str]:
    return {
        "phase35_report_path": str(phase35_path),
        "report_path": str(
            output / "humanoid_phase4_deployment_enabler_sweep_report_v1.json"
        ),
        "contracts_path": str(output / "humanoid_phase4_contract_surfaces_v1.jsonl"),
        "stubs_path": str(output / "humanoid_phase4_stub_surfaces_v1.jsonl"),
        "markdown_path": str(
            output / "humanoid_phase4_deployment_enabler_sweep_report_v1.md"
        ),
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 Deployment-Enabler Sweep Artifacts",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local non-hardware scaffold complete: "
        f"`{str(payload['local_non_hardware_scaffold_complete']).lower()}`",
        f"- Contract surfaces: `{payload['contract_surface_count']}`",
        f"- Stub surfaces: `{payload['stub_surface_count']}`",
        f"- Ready for Phase 6.5 local meta nodes: "
        f"`{str(payload['ready_for_phase65_local_meta_nodes']).lower()}`",
        "",
        "## Boundary",
        "",
        "These artifacts are contract/runbook/interface scaffolds only. They do",
        "not claim live streams, control interfaces, Unitree sim runtime,",
        "hardware execution, provider execution, training, promotion, live",
        "policy control, or reward-math mutation.",
        "",
        "## Remaining blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_phase35_report(
    phase35_dir: Path,
    run_dependencies_if_missing: bool,
) -> Path:
    report_path = _phase35_report_path(phase35_dir)
    if report_path.exists():
        return report_path
    if not run_dependencies_if_missing:
        raise FileNotFoundError(f"Missing Phase 3.5 report: {report_path}")
    run_prepare_phase35_humanoid_capacity_env_refit(output_dir=phase35_dir)
    return report_path


def run_prepare_phase4_deployment_enabler_sweep(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase35_dir: str | Path = DEFAULT_PHASE35_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase35_path = _resolve_phase35_report(
        Path(phase35_dir), run_dependencies_if_missing
    )
    refs = _artifact_refs(output, phase35_path)
    phase35_report = load_phase35_humanoid_refit_report(phase35_path)
    report, contracts, stubs = build_phase4_deployment_enabler_sweep(
        phase35_report,
        artifact_refs=refs,
    )
    saved_refs = save_phase4_deployment_enabler_sweep(output, report, contracts, stubs)
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(Path(refs["markdown_path"]), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase35-dir", default=str(DEFAULT_PHASE35_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase4_deployment_enabler_sweep(
        output_dir=args.output_dir,
        phase35_dir=args.phase35_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_non_hardware_scaffold_complete"]
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
