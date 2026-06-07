#!/usr/bin/env python3
"""Compile the local neural trainability audit."""

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

from src.world_model.economic_world_model.neural_trainability_audit import (  # noqa: E402
    NeuralTrainabilityAuditReport,
    NeuralTrainabilityComponent,
    NeuralTrainabilityFollowupRow,
    build_neural_trainability_audit,
    save_neural_trainability_audit,
    validate_neural_trainability_audit,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/economic_world_model/neural_trainability_audit")
DEFAULT_BACKLOG_PATH = Path("scripts/TRAINING_MIGRATION_BACKLOG.json")


def _write_markdown(
    path: Path,
    *,
    report: NeuralTrainabilityAuditReport,
    components: list[NeuralTrainabilityComponent],
    followups: list[NeuralTrainabilityFollowupRow],
    validation: Mapping[str, Any],
) -> None:
    payload = report.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Neural Trainability Audit",
        "",
        f"- Audit ID: `{payload['audit_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Component count: `{payload['component_count']}`",
        f"- Follow-up count: `{payload['followup_count']}`",
        f"- Ready for training count: `{payload['ready_for_training_count']}`",
        f"- Promotion eligible count: `{payload['promotion_eligible_count']}`",
        f"- Training executed: `{str(payload['training_executed']).lower()}`",
        f"- Weights written: `{str(payload['weights_written']).lower()}`",
        f"- Provider executed: `{str(payload['provider_executed']).lower()}`",
        f"- GPU executed: `{str(payload['gpu_executed']).lower()}`",
        f"- Hardware executed: `{str(payload['hardware_executed']).lower()}`",
        f"- Phase 7 authority granted: `{str(payload['phase7_authority_granted']).lower()}`",
        "",
        "## Follow-Up Planes",
        "",
    ]
    for key, value in sorted(payload["plane_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blocker Counts", ""])
    for key, value in sorted(payload["blocker_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Components", ""])
    for component in components:
        lines.extend(
            [
                f"### `{component.component_id}`",
                f"- owner: `{component.owner}`",
                f"- type: `{component.component_type}`",
                f"- roles: `{', '.join(component.surface_roles)}`",
                f"- promotion status: `{component.promotion_status}`",
                f"- missing item count: `{component.missing_item_count}`",
                f"- blockers: `{', '.join(component.blockers)}`",
                "",
            ]
        )
    lines.extend(["## Follow-Up Rows", ""])
    for row in followups:
        lines.extend(
            [
                f"### `{row.missing_item_id}`",
                f"- component: `{row.component_id}`",
                f"- plane: `{row.plane}`",
                f"- blocker: `{row.blocker}`",
                f"- action: {row.action}",
                f"- target: `{row.target}`",
                f"- verify: `{row.verify_receipt}`",
                f"- promotion eligible: `{str(row.promotion_eligible).lower()}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Validation",
            "",
            f"- status: `{validation['status']}`",
            f"- error count: `{validation['error_count']}`",
            f"- warning count: `{validation['warning_count']}`",
            f"- safe for training: `{str(validation['safe_for_training']).lower()}`",
            f"- safe for promotion: `{str(validation['safe_for_promotion']).lower()}`",
            "",
            "## Boundary",
            "",
            "This audit is a non-training local planning receipt. It does not",
            "run providers, run GPU, launch RunPod, operate hardware, train,",
            "write weights, mutate reward/controller math, grant Phase 7",
            "authority, or claim promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compile_neural_trainability_audit(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    training_backlog_path: str | Path = DEFAULT_BACKLOG_PATH,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_refs = {
        "report_path": str(output / "neural_trainability_audit_report_v1.json"),
        "components_path": str(output / "neural_trainability_components_v1.jsonl"),
        "followups_path": str(output / "neural_trainability_followups_v1.jsonl"),
        "markdown_path": str(output / "neural_trainability_audit_v1.md"),
        "validation_path": str(output / "neural_trainability_audit_validation_v1.json"),
    }
    report, components, followups = build_neural_trainability_audit(
        training_backlog_path=training_backlog_path,
        artifact_refs=artifact_refs,
        metadata={"source": "compile_neural_trainability_audit_script"},
    )
    save_neural_trainability_audit(
        report_path=artifact_refs["report_path"],
        report=report,
        components_path=artifact_refs["components_path"],
        components=components,
        followups_path=artifact_refs["followups_path"],
        followups=followups,
    )
    validation = validate_neural_trainability_audit(
        report=report,
        components=components,
        followups=followups,
    )
    validation_payload = {
        **validation,
        "audit_id": report.audit_id,
        "artifact_refs": artifact_refs,
    }
    Path(artifact_refs["validation_path"]).write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(
        Path(artifact_refs["markdown_path"]),
        report=report,
        components=components,
        followups=followups,
        validation=validation_payload,
    )
    payload = {
        **report.to_dict(),
        "components": [component.to_dict() for component in components],
        "followups": [row.to_dict() for row in followups],
        "validation": validation_payload,
    }
    Path(artifact_refs["report_path"]).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--training-backlog", default=str(DEFAULT_BACKLOG_PATH))
    args = parser.parse_args()
    payload = run_compile_neural_trainability_audit(
        output_dir=args.output_dir,
        training_backlog_path=args.training_backlog,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    validation = payload["validation"]
    return (
        0
        if payload["status"] == "ok_neural_trainability_audit_non_training"
        and validation["status"] == "ok"
        and payload["component_count"] > 0
        and payload["followup_count"] > 0
        and payload["ready_for_training_count"] == 0
        and payload["promotion_eligible_count"] == 0
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["gpu_executed"]
        and not payload["hardware_executed"]
        and not payload["phase7_authority_granted"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
