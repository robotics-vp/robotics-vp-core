#!/usr/bin/env python3
"""Materialize Phase 6.5 local meta-node neuralization artifacts."""

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

from scripts.economic_world_model.audit_phase6_transport_closure import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE6_CLOSURE_DIR,
)
from scripts.economic_world_model.audit_phase6_transport_closure import (  # noqa: E402
    run_audit_phase6_transport_closure,
)
from scripts.economic_world_model.prepare_phase35_humanoid_capacity_env_refit import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE35_DIR,
)
from scripts.economic_world_model.prepare_phase4_deployment_enabler_sweep import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_DIR,
)
from scripts.economic_world_model.prepare_phase4_deployment_enabler_sweep import (  # noqa: E402
    run_prepare_phase4_deployment_enabler_sweep,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase65_meta_node_neuralization,
    load_phase35_humanoid_refit_report,
    load_phase4_deployment_enabler_sweep_report,
    save_phase65_meta_node_neuralization,
)
from src.world_model.transport import load_wm_transport_phase6_closure_audit  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase65_meta_node_neuralization"
)


def _paths(
    output: Path,
    phase35_dir: Path,
    phase4_dir: Path,
    phase6_closure_dir: Path,
) -> dict[str, Path]:
    return {
        "phase35_report": phase35_dir / "humanoid_phase35_refit_report_v1.json",
        "phase4_report": phase4_dir
        / "humanoid_phase4_deployment_enabler_sweep_report_v1.json",
        "phase6_closure_audit": phase6_closure_dir
        / "wm_transport_phase6_closure_audit_v1.json",
        "report": output / "phase65_meta_node_neuralization_report_v1.json",
        "states": output / "meta_node_states_v1.jsonl",
        "trajectories": output / "meta_node_trajectory_receipts_v1.jsonl",
        "interventions": output / "meta_node_intervention_receipts_v1.jsonl",
        "targets": output / "meta_node_counterfactual_targets_v1.jsonl",
        "robustness": output / "meta_node_robustness_reports_v1.jsonl",
        "gates": output / "meta_node_promotion_gates_v1.jsonl",
        "markdown": output / "phase65_meta_node_neuralization_report_v1.md",
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6.5 Meta-Node Neuralization Artifacts",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local meta-node scaffold complete: "
        f"`{str(payload['local_meta_node_scaffold_complete']).lower()}`",
        f"- Node states: `{payload['node_state_count']}`",
        f"- Trajectory receipts: `{payload['trajectory_receipt_count']}`",
        f"- Intervention receipts: `{payload['intervention_receipt_count']}`",
        f"- Counterfactual targets: `{payload['counterfactual_target_count']}`",
        f"- Robustness reports: `{payload['robustness_report_count']}`",
        f"- Promotion gates: `{payload['promotion_gate_count']}`",
        f"- Ready for Phase 7 scaffold: "
        f"`{str(payload['ready_for_phase7_scaffold']).lower()}`",
        f"- Phase 7 authority granted: "
        f"`{str(payload['phase7_authority_granted']).lower()}`",
        "",
        "## Boundary",
        "",
        "These are state, receipt, target-row, robustness, and denied-gate",
        "surfaces only. They do not train meta-node weights, write weights, run",
        "providers or hardware, grant Phase 7 authority, mutate reward math,",
        "promote outputs, or control live policy.",
        "",
        "## Remaining blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(
    *,
    phase35_dir: Path,
    phase4_dir: Path,
    phase6_closure_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _paths(Path("."), phase35_dir, phase4_dir, phase6_closure_dir)
    needed = {
        "phase35_report": paths["phase35_report"],
        "phase4_report": paths["phase4_report"],
        "phase6_closure_audit": paths["phase6_closure_audit"],
    }
    if all(path.exists() for path in needed.values()):
        return needed
    if not run_dependencies_if_missing:
        missing = [str(path) for path in needed.values() if not path.exists()]
        raise FileNotFoundError("Missing Phase 6.5 inputs: " + ", ".join(missing))
    run_prepare_phase4_deployment_enabler_sweep(
        output_dir=phase4_dir,
        phase35_dir=phase35_dir,
        run_dependencies_if_missing=True,
    )
    run_audit_phase6_transport_closure(
        output_dir=phase6_closure_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in needed.values()):
        missing = [str(path) for path in needed.values() if not path.exists()]
        raise FileNotFoundError(
            "Phase 6.5 dependency builders did not materialize: "
            + ", ".join(missing)
        )
    return needed


def run_prepare_phase65_meta_node_neuralization(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase35_dir: str | Path = DEFAULT_PHASE35_DIR,
    phase4_dir: str | Path = DEFAULT_PHASE4_DIR,
    phase6_closure_dir: str | Path = DEFAULT_PHASE6_CLOSURE_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_paths = _resolve_inputs(
        phase35_dir=Path(phase35_dir),
        phase4_dir=Path(phase4_dir),
        phase6_closure_dir=Path(phase6_closure_dir),
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    paths = _paths(output, Path(phase35_dir), Path(phase4_dir), Path(phase6_closure_dir))
    refs = {f"{key}_path": str(path) for key, path in paths.items()}
    report, states, trajectories, interventions, targets, robustness, gates = (
        build_phase65_meta_node_neuralization(
            phase35_report=load_phase35_humanoid_refit_report(
                input_paths["phase35_report"]
            ),
            phase4_report=load_phase4_deployment_enabler_sweep_report(
                input_paths["phase4_report"]
            ),
            phase6_closure_audit=load_wm_transport_phase6_closure_audit(
                input_paths["phase6_closure_audit"]
            ),
            artifact_refs=refs,
        )
    )
    saved_refs = save_phase65_meta_node_neuralization(
        output,
        report,
        states,
        trajectories,
        interventions,
        targets,
        robustness,
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
    parser.add_argument("--phase35-dir", default=str(DEFAULT_PHASE35_DIR))
    parser.add_argument("--phase4-dir", default=str(DEFAULT_PHASE4_DIR))
    parser.add_argument("--phase6-closure-dir", default=str(DEFAULT_PHASE6_CLOSURE_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase65_meta_node_neuralization(
        output_dir=args.output_dir,
        phase35_dir=args.phase35_dir,
        phase4_dir=args.phase4_dir,
        phase6_closure_dir=args.phase6_closure_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_meta_node_scaffold_complete"]
        and payload["ready_for_phase7_scaffold"]
        and not payload["phase7_authority_granted"]
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
