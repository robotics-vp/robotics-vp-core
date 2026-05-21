from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.compile_economic_wm_provider_runbook import (
    run_compile_economic_wm_provider_runbook,
)
from src.economics.economic_wm_entry import evaluate_economic_wm_entry_preflight
from src.world_model.economic_world_model import (
    build_economic_wm_provider_runbook,
    build_economic_wm_scaffold_report,
    build_economic_wm_shadow_allocation_eval,
    build_economic_wm_teacher_provider_contract,
    build_economic_wm_training_corpus_manifest,
    load_economic_wm_provider_runbook,
    save_economic_wm_teacher_provider_contract,
)


def _scaffold_report():
    scenarios = []
    for idx, ready in enumerate([True, True, False, False, False]):
        scenarios.append(
            {
                "scenario_id": f"scenario_{idx}",
                "passed": True,
                "observed": {
                    "benchmark_ready": ready,
                    "rlds_benchmark_ready": ready,
                    "lerobot_benchmark_ready": ready,
                },
            }
        )
    preflight = evaluate_economic_wm_entry_preflight(
        stage1_sweep_report={
            "status": "ok",
            "scenario_count": 5,
            "admission_count": 5,
            "rlds_episode_count": 5,
            "lerobot_row_count": 5,
            "benchmark_ready_count": 2,
            "shadow_only_count": 3,
            "promotion_eligible": False,
            "failures": [],
            "scenario_reports": scenarios,
        }
    )
    return build_economic_wm_scaffold_report(preflight)


def _admission_row(video_id: str, proposal_id: str, *, ready: bool, calibrated: bool):
    return {
        "video_id": video_id,
        "proposal_id": proposal_id,
        "blocked": False,
        "benchmark_gate": {
            "ready": ready,
            "blocking_preconditions": []
            if ready
            else ["blocked::camera_calibration_missing"],
        },
        "future_training_signals": {
            "benchmark_gate_ready": ready,
            "reconstruction_calibrated": calibrated,
            "reconstruction_real_grounded": ready,
            "reconstruction_training_eligible": ready and calibrated,
            "scene_tracks_backend_real": ready,
            "scene_tracks_non_stub": ready,
            "semantic_grounding_non_heuristic": ready,
            "semantic_memory_grounded": ready,
            "teacher_runtime_contract_complete": True,
            "teacher_runtime_real": False,
            "vision_backbone_real": True,
        },
    }


def _contract():
    scaffold = _scaffold_report()
    admission_rows = [
        _admission_row(
            "calibrated_inline_real", "proposal_0", ready=True, calibrated=True
        ),
        _admission_row(
            "top_level_calibration_real", "proposal_1", ready=True, calibrated=True
        ),
        _admission_row(
            "missing_calibration_real", "proposal_2", ready=False, calibrated=False
        ),
        _admission_row(
            "unknown_artifact_calibrated", "proposal_3", ready=False, calibrated=True
        ),
        _admission_row(
            "passthrough_calibrated", "proposal_4", ready=False, calibrated=True
        ),
    ]
    manifest, rows = build_economic_wm_training_corpus_manifest(
        scaffold_report=scaffold,
        admission_rows=admission_rows,
        rows_path="rows.jsonl",
    )
    allocation_eval = build_economic_wm_shadow_allocation_eval(
        scaffold_report=scaffold,
        corpus_manifest=manifest,
        rows=rows,
    )
    return build_economic_wm_teacher_provider_contract(
        scaffold_report=scaffold,
        allocation_eval=allocation_eval,
        corpus_manifest=manifest,
        rows=rows,
    )


def test_provider_runbook_compiles_manifest_shaped_templates_without_launch() -> None:
    runbook = build_economic_wm_provider_runbook(contract=_contract())

    assert runbook.version == "economic_wm_provider_runbook_v1"
    assert runbook.authority_class == "runbook_template_only"
    assert runbook.launch_allowed is False
    assert runbook.provider_bringup_ready is False
    assert runbook.gpu_training_ready is False
    assert runbook.promotion_eligible is False
    assert runbook.reward_math_mutation is False

    by_key = {template.requirement_key: template for template in runbook.templates}
    assert {
        "non_stub_teacher_runtime_invocation",
        "provider_runtime_truth_receipts",
        "promotion_grade_benchmark_evidence",
        "gpu_training_runtime_receipt",
        "replay_row_linkage_integrity",
    } <= set(by_key)

    gpu_template = by_key["gpu_training_runtime_receipt"]
    assert gpu_template.run_class == "train"
    assert gpu_template.pod_class == "train"
    assert gpu_template.launch_allowed is False
    assert "gpu_training_not_run" in gpu_template.blocked_by
    manifest = gpu_template.to_manifest_stub(commit_sha="abc123", branch="main")
    assert manifest["status"] == "pending"
    assert manifest["pod_id"] is None
    assert manifest["commit_sha"] == "abc123"
    assert manifest["branch"] == "main"
    assert manifest["epistemic_status"] == "proof_of_life"
    assert any("TEMPLATE_ONLY" in command for command in manifest["commands"])

    local_template = by_key["replay_row_linkage_integrity"]
    assert local_template.mode == "local"
    assert local_template.run_class == "loop"
    assert local_template.local_verification_available is True
    assert local_template.launch_allowed is False
    assert runbook.aggregate_counts["template_count"] == 5.0
    assert runbook.aggregate_counts["blocked_template_count"] == 5.0


def test_compile_provider_runbook_script_writes_runbook_and_manifest_templates(
    tmp_path,
) -> None:
    contract = _contract()
    contract_path = tmp_path / "contract.json"
    save_economic_wm_teacher_provider_contract(contract_path, contract)

    payload = run_compile_economic_wm_provider_runbook(
        output_dir=tmp_path / "runbook",
        contract_path=contract_path,
        run_contract_if_missing=False,
    )

    assert payload["authority_class"] == "runbook_template_only"
    assert payload["launch_allowed"] is False
    assert payload["promotion_eligible"] is False
    assert payload["aggregate_counts"]["template_count"] == 5.0
    assert (tmp_path / "runbook" / "economic_wm_provider_runbook_v1.md").exists()
    manifest_paths = payload["artifact_refs"]["manifest_template_paths"]
    assert len(manifest_paths) == 5
    first_manifest = json.loads(Path(manifest_paths[0]).read_text())
    assert first_manifest["status"] == "pending"
    assert first_manifest["pod_id"] is None
    assert first_manifest["task"].startswith("[TEMPLATE ONLY]")

    loaded = load_economic_wm_provider_runbook(payload["artifact_refs"]["runbook_path"])
    assert loaded.runbook_id == payload["runbook_id"]
