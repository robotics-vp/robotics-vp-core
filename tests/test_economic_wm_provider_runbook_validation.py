from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.validate_economic_wm_provider_runbook import (
    run_validate_economic_wm_provider_runbook,
)
from src.world_model.economic_world_model import (
    EconomicWMEvidenceRequirement,
    EconomicWMTeacherProviderContract,
    build_economic_wm_provider_runbook,
    save_economic_wm_provider_runbook,
    validate_economic_wm_provider_runbook,
    validate_economic_wm_provider_runbook_payload,
)


def _requirement(
    key: str,
    *,
    blocker: str,
    provider_family: str = "provider_family",
    promotion_gate: bool = True,
) -> EconomicWMEvidenceRequirement:
    return EconomicWMEvidenceRequirement(
        requirement_id=f"req_{key}",
        requirement_key=key,
        provider_family=provider_family,
        evidence_kind=f"{key}_evidence",
        current_status="blocked_external_runtime"
        if blocker != "none"
        else "satisfied_local_scaffold",
        required_artifacts=[f"{key}_artifact_v1"],
        local_prep_actions=[f"prepare_{key}"],
        blockers=[] if blocker == "none" else [blocker],
        satisfaction_score=1.0 if blocker == "none" else 0.0,
        promotion_gate=promotion_gate,
    )


def _runbook():
    requirements = [
        _requirement(
            "non_stub_teacher_runtime_invocation",
            blocker="non_stub_teacher_runtime_not_verified",
            provider_family="teacher_runtime",
        ),
        _requirement(
            "provider_runtime_truth_receipts",
            blocker="provider_bringup_not_run",
            provider_family="external_provider_runtime",
        ),
        _requirement(
            "promotion_grade_benchmark_evidence",
            blocker="promotion_grade_benchmark_evidence_missing",
            provider_family="benchmark_evidence",
        ),
        _requirement(
            "gpu_training_runtime_receipt",
            blocker="gpu_training_not_run",
            provider_family="gpu_training_runtime",
        ),
        _requirement(
            "replay_row_linkage_integrity",
            blocker="none",
            provider_family="local_replay_bridge",
            promotion_gate=False,
        ),
    ]
    contract = EconomicWMTeacherProviderContract(
        contract_id="contract_for_validation_test",
        scaffold_id="scaffold",
        allocation_eval_id="allocation_eval",
        corpus_id="corpus",
        readiness_class="scaffold_ready_training_blocked",
        requirements=requirements,
        provider_bringup_ready=False,
        gpu_training_ready=False,
        promotion_eligible=False,
        reward_math_mutation=False,
        authority_class="evidence_contract_only",
        training_blockers=["gpu_training_not_run", "provider_bringup_not_run"],
    )
    return build_economic_wm_provider_runbook(contract=contract)


def test_provider_runbook_validation_accepts_template_only_posture() -> None:
    report = validate_economic_wm_provider_runbook(_runbook())

    assert report.version == "economic_wm_provider_runbook_validation_v1"
    assert report.status == "ok"
    assert report.safe_for_template_storage is True
    assert report.safe_for_launch is False
    assert report.error_count == 0
    assert report.aggregate_counts["template_count"] == 5.0
    assert report.aggregate_counts["runpod_template_count"] == 4.0
    assert report.aggregate_counts["local_template_count"] == 1.0


def test_provider_runbook_validation_rejects_executed_or_unguarded_stubs() -> None:
    runbook = _runbook()
    payload = runbook.to_dict()
    gpu_template = next(
        template
        for template in payload["templates"]
        if template["requirement_key"] == "gpu_training_runtime_receipt"
    )
    gpu_template["manifest_stub"]["status"] = "completed"
    gpu_template["manifest_stub"]["pod_id"] = "pod_fake"
    gpu_template["manifest_stub"]["commands"] = [
        "python3 train_economic_world_model_v0.py"
    ]

    report = validate_economic_wm_provider_runbook_payload(payload)

    assert report.status == "failed"
    assert report.safe_for_template_storage is False
    assert report.safe_for_launch is False
    assert report.error_count >= 3
    assert any("status must remain pending" in error for error in report.errors)
    assert any("lacks guard command" in error for error in report.errors)


def test_validate_provider_runbook_script_roundtrip(tmp_path) -> None:
    runbook = _runbook()
    runbook_path = tmp_path / "runbook.json"
    manifest_dir = tmp_path / "manifest_templates"
    manifest_dir.mkdir()
    save_economic_wm_provider_runbook(runbook_path, runbook)
    for template in runbook.templates:
        manifest_path = manifest_dir / f"{template.template_id}.manifest_template.json"
        manifest_path.write_text(
            json.dumps(
                template.to_manifest_stub(commit_sha="abc123", branch="main"),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    payload = run_validate_economic_wm_provider_runbook(
        output_dir=tmp_path / "validation",
        runbook_path=runbook_path,
        manifest_template_dir=manifest_dir,
        compile_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["safe_for_template_storage"] is True
    assert payload["safe_for_launch"] is False
    assert payload["error_count"] == 0
    assert Path(payload["artifact_refs"]["validation_path"]).exists()
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()
