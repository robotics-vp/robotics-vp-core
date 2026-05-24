#!/usr/bin/env python3
"""Audit local structural closure for Phase 3.5, Phase 4, and Phase 6.5."""

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
from scripts.economic_world_model.audit_phase35_bipedal_readiness import (  # noqa: E402
    DEFAULT_BIPEDAL_CHASSIS_DIR,
)
from scripts.economic_world_model.audit_phase35_bipedal_readiness import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE35_BIPEDAL_READINESS_DIR,
)
from scripts.economic_world_model.audit_phase35_bipedal_readiness import (  # noqa: E402
    run_audit_phase35_bipedal_readiness,
)
from scripts.economic_world_model.prepare_phase4_deployment_enabler_sweep import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_DIR,
)
from scripts.economic_world_model.prepare_phase4_downstream_controller_scaffold import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_DOWNSTREAM_CONTROLLER_DIR,
)
from scripts.economic_world_model.prepare_phase4_downstream_controller_scaffold import (  # noqa: E402
    run_prepare_phase4_downstream_controller_scaffold,
)
from scripts.economic_world_model.prepare_phase4_unitree_bringup_readiness import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_UNITREE_BRINGUP_READINESS_DIR,
)
from scripts.economic_world_model.prepare_phase4_unitree_bringup_readiness import (  # noqa: E402
    run_prepare_phase4_unitree_bringup_readiness,
)
from scripts.economic_world_model.prepare_phase4_unitree_local_harnesses import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_UNITREE_LOCAL_HARNESS_DIR,
)
from scripts.economic_world_model.prepare_phase4_unitree_local_harnesses import (  # noqa: E402
    run_prepare_phase4_unitree_local_harnesses,
)
from scripts.economic_world_model.prepare_phase65_meta_node_neuralization import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE65_DIR,
)
from scripts.economic_world_model.prepare_phase65_meta_node_neuralization import (  # noqa: E402
    run_prepare_phase65_meta_node_neuralization,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase35465_local_closure_audit,
    load_phase35_humanoid_refit_report,
    load_phase4_deployment_enabler_sweep_report,
    load_phase4_downstream_controller_scaffold_report,
    load_phase4_unitree_local_harness_report,
    load_phase4_unitree_bringup_readiness_report,
    load_phase65_meta_node_neuralization_report,
    save_phase35465_local_closure_audit,
)
from src.world_model.embodiment_actuation import (  # noqa: E402
    load_phase35_bipedal_readiness_audit,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase35_4_65_local_closure"
)


def _input_paths(
    phase35_dir: Path,
    phase35_bipedal_readiness_dir: Path,
    phase4_dir: Path,
    phase4_downstream_controller_dir: Path,
    phase4_unitree_bringup_readiness_dir: Path,
    phase4_unitree_local_harness_dir: Path,
    phase65_dir: Path,
) -> dict[str, Path]:
    return {
        "phase35_report": phase35_dir / "humanoid_phase35_refit_report_v1.json",
        "phase35_bipedal_readiness_audit": phase35_bipedal_readiness_dir
        / "phase35_bipedal_readiness_audit_v1.json",
        "phase4_report": phase4_dir
        / "humanoid_phase4_deployment_enabler_sweep_report_v1.json",
        "phase4_downstream_controller_report": phase4_downstream_controller_dir
        / "phase4_downstream_controller_scaffold_report_v1.json",
        "phase4_unitree_bringup_readiness_report": phase4_unitree_bringup_readiness_dir
        / "phase4_unitree_bringup_readiness_report_v1.json",
        "phase4_unitree_local_harness_report": phase4_unitree_local_harness_dir
        / "phase4_unitree_local_harness_report_v1.json",
        "phase65_report": phase65_dir / "phase65_meta_node_neuralization_report_v1.json",
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3.5 / 4 / 6.5 Local Closure Audit",
        "",
        f"- Audit: `{payload['audit_id']}`",
        f"- Status: `{payload['status']}`",
        "- All local structures complete: "
        f"`{str(payload['all_local_structures_complete']).lower()}`",
        f"- Phase 3.5 complete: `{str(payload['local_phase35_complete']).lower()}`",
        "- Phase 3.5 bipedal readiness complete: "
        f"`{str(payload['local_phase35_bipedal_readiness_complete']).lower()}`",
        f"- Phase 4 complete: `{str(payload['local_phase4_complete']).lower()}`",
        "- Phase 4 downstream controller complete: "
        f"`{str(payload['local_phase4_downstream_controller_complete']).lower()}`",
        "- Phase 4 Unitree bring-up readiness complete: "
        f"`{str(payload['local_phase4_unitree_bringup_readiness_complete']).lower()}`",
        "- Phase 4 Unitree local harness complete: "
        f"`{str(payload['local_phase4_unitree_local_harness_complete']).lower()}`",
        f"- Phase 6.5 complete: `{str(payload['local_phase65_complete']).lower()}`",
        f"- Ready for Phase 7 scaffold: "
        f"`{str(payload['ready_for_phase7_scaffold']).lower()}`",
        f"- Phase 7 authority granted: "
        f"`{str(payload['phase7_authority_granted']).lower()}`",
        "",
        "## Closed Local Surfaces",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["closed_local_surfaces"])
    lines.extend(["", "## Remaining Evidence Blockers", ""])
    lines.extend(f"- `{item}`" for item in payload["remaining_evidence_blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This audit closes local scaffolding only. It does not claim Unitree",
            "sim runtime, hardware execution, provider execution, GPU training,",
            "weight writes, promotion, live policy control, reward-math mutation,",
            "or Phase 7 authority.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(
    *,
    phase35_dir: Path,
    bipedal_chassis_dir: Path,
    phase35_bipedal_readiness_dir: Path,
    phase4_dir: Path,
    phase4_downstream_controller_dir: Path,
    phase4_unitree_bringup_readiness_dir: Path,
    phase4_unitree_local_harness_dir: Path,
    phase65_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _input_paths(
        phase35_dir,
        phase35_bipedal_readiness_dir,
        phase4_dir,
        phase4_downstream_controller_dir,
        phase4_unitree_bringup_readiness_dir,
        phase4_unitree_local_harness_dir,
        phase65_dir,
    )
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError("Missing local closure inputs: " + ", ".join(missing))
    run_prepare_phase65_meta_node_neuralization(
        output_dir=phase65_dir,
        phase35_dir=phase35_dir,
        phase4_dir=phase4_dir,
        run_dependencies_if_missing=True,
    )
    run_audit_phase35_bipedal_readiness(
        output_dir=phase35_bipedal_readiness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        run_dependencies_if_missing=True,
    )
    run_prepare_phase4_downstream_controller_scaffold(
        output_dir=phase4_downstream_controller_dir,
        phase4_dir=phase4_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        run_dependencies_if_missing=True,
    )
    run_prepare_phase4_unitree_bringup_readiness(
        output_dir=phase4_unitree_bringup_readiness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=phase4_downstream_controller_dir,
        run_dependencies_if_missing=True,
    )
    run_prepare_phase4_unitree_local_harnesses(
        output_dir=phase4_unitree_local_harness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=phase4_downstream_controller_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Local closure dependency builders did not materialize: "
            + ", ".join(missing)
        )
    return paths


def run_audit_phase35_4_65_local_closure(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase35_dir: str | Path = DEFAULT_PHASE35_DIR,
    bipedal_chassis_dir: str | Path = DEFAULT_BIPEDAL_CHASSIS_DIR,
    phase35_bipedal_readiness_dir: str | Path = (
        DEFAULT_PHASE35_BIPEDAL_READINESS_DIR
    ),
    phase4_dir: str | Path = DEFAULT_PHASE4_DIR,
    phase4_downstream_controller_dir: str | Path = (
        DEFAULT_PHASE4_DOWNSTREAM_CONTROLLER_DIR
    ),
    phase4_unitree_bringup_readiness_dir: str | Path = (
        DEFAULT_PHASE4_UNITREE_BRINGUP_READINESS_DIR
    ),
    phase4_unitree_local_harness_dir: str | Path = (
        DEFAULT_PHASE4_UNITREE_LOCAL_HARNESS_DIR
    ),
    phase65_dir: str | Path = DEFAULT_PHASE65_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_paths = _resolve_inputs(
        phase35_dir=Path(phase35_dir),
        bipedal_chassis_dir=Path(bipedal_chassis_dir),
        phase35_bipedal_readiness_dir=Path(phase35_bipedal_readiness_dir),
        phase4_dir=Path(phase4_dir),
        phase4_downstream_controller_dir=Path(phase4_downstream_controller_dir),
        phase4_unitree_bringup_readiness_dir=Path(
            phase4_unitree_bringup_readiness_dir
        ),
        phase4_unitree_local_harness_dir=Path(phase4_unitree_local_harness_dir),
        phase65_dir=Path(phase65_dir),
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    report_path = output / "phase35_4_65_local_closure_audit_v1.json"
    markdown_path = output / "phase35_4_65_local_closure_audit_v1.md"
    refs = {
        "phase35_report_path": str(input_paths["phase35_report"]),
        "phase35_bipedal_readiness_audit_path": str(
            input_paths["phase35_bipedal_readiness_audit"]
        ),
        "phase4_report_path": str(input_paths["phase4_report"]),
        "phase4_downstream_controller_report_path": str(
            input_paths["phase4_downstream_controller_report"]
        ),
        "phase4_unitree_bringup_readiness_report_path": str(
            input_paths["phase4_unitree_bringup_readiness_report"]
        ),
        "phase4_unitree_local_harness_report_path": str(
            input_paths["phase4_unitree_local_harness_report"]
        ),
        "phase65_report_path": str(input_paths["phase65_report"]),
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
    }
    audit = build_phase35465_local_closure_audit(
        phase35_report=load_phase35_humanoid_refit_report(
            input_paths["phase35_report"]
        ),
        phase35_bipedal_readiness_audit=load_phase35_bipedal_readiness_audit(
            input_paths["phase35_bipedal_readiness_audit"]
        ),
        phase4_report=load_phase4_deployment_enabler_sweep_report(
            input_paths["phase4_report"]
        ),
        phase4_downstream_controller_report=(
            load_phase4_downstream_controller_scaffold_report(
                input_paths["phase4_downstream_controller_report"]
            )
        ),
        phase4_unitree_bringup_readiness_report=(
            load_phase4_unitree_bringup_readiness_report(
                input_paths["phase4_unitree_bringup_readiness_report"]
            )
        ),
        phase4_unitree_local_harness_report=(
            load_phase4_unitree_local_harness_report(
                input_paths["phase4_unitree_local_harness_report"]
            )
        ),
        phase65_report=load_phase65_meta_node_neuralization_report(
            input_paths["phase65_report"]
        ),
        artifact_refs=refs,
    )
    save_phase35465_local_closure_audit(report_path, audit)
    payload = audit.to_dict()
    _write_markdown(markdown_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase35-dir", default=str(DEFAULT_PHASE35_DIR))
    parser.add_argument("--bipedal-chassis-dir", default=str(DEFAULT_BIPEDAL_CHASSIS_DIR))
    parser.add_argument(
        "--phase35-bipedal-readiness-dir",
        default=str(DEFAULT_PHASE35_BIPEDAL_READINESS_DIR),
    )
    parser.add_argument("--phase4-dir", default=str(DEFAULT_PHASE4_DIR))
    parser.add_argument(
        "--phase4-downstream-controller-dir",
        default=str(DEFAULT_PHASE4_DOWNSTREAM_CONTROLLER_DIR),
    )
    parser.add_argument(
        "--phase4-unitree-bringup-readiness-dir",
        default=str(DEFAULT_PHASE4_UNITREE_BRINGUP_READINESS_DIR),
    )
    parser.add_argument(
        "--phase4-unitree-local-harness-dir",
        default=str(DEFAULT_PHASE4_UNITREE_LOCAL_HARNESS_DIR),
    )
    parser.add_argument("--phase65-dir", default=str(DEFAULT_PHASE65_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_audit_phase35_4_65_local_closure(
        output_dir=args.output_dir,
        phase35_dir=args.phase35_dir,
        bipedal_chassis_dir=args.bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=args.phase35_bipedal_readiness_dir,
        phase4_dir=args.phase4_dir,
        phase4_downstream_controller_dir=args.phase4_downstream_controller_dir,
        phase4_unitree_bringup_readiness_dir=(
            args.phase4_unitree_bringup_readiness_dir
        ),
        phase4_unitree_local_harness_dir=args.phase4_unitree_local_harness_dir,
        phase65_dir=args.phase65_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["all_local_structures_complete"]
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
