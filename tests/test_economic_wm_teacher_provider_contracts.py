from __future__ import annotations

from scripts.economic_world_model.prepare_economic_wm_teacher_provider_contracts import (
    run_prepare_economic_wm_teacher_provider_contracts,
)
from src.economics.economic_wm_entry import evaluate_economic_wm_entry_preflight
from src.world_model.economic_world_model import (
    build_economic_wm_scaffold_report,
    build_economic_wm_shadow_allocation_eval,
    build_economic_wm_teacher_provider_contract,
    build_economic_wm_training_corpus_manifest,
    load_economic_wm_teacher_provider_contract,
    save_economic_wm_scaffold_report,
    save_economic_wm_shadow_allocation_eval,
    save_economic_wm_training_corpus,
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


def _corpus(scaffold):
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
    return build_economic_wm_training_corpus_manifest(
        scaffold_report=scaffold,
        admission_rows=admission_rows,
        rows_path="rows.jsonl",
    )


def _eval_bundle():
    scaffold = _scaffold_report()
    manifest, rows = _corpus(scaffold)
    allocation_eval = build_economic_wm_shadow_allocation_eval(
        scaffold_report=scaffold,
        corpus_manifest=manifest,
        rows=rows,
    )
    return scaffold, manifest, rows, allocation_eval


def test_teacher_provider_contract_names_required_evidence_without_promotion() -> None:
    scaffold, manifest, rows, allocation_eval = _eval_bundle()

    contract = build_economic_wm_teacher_provider_contract(
        scaffold_report=scaffold,
        allocation_eval=allocation_eval,
        corpus_manifest=manifest,
        rows=rows,
    )

    assert contract.version == "economic_wm_teacher_provider_contract_v1"
    assert contract.authority_class == "evidence_contract_only"
    assert contract.provider_bringup_ready is False
    assert contract.gpu_training_ready is False
    assert contract.promotion_eligible is False
    assert contract.reward_math_mutation is False
    keys = {requirement.requirement_key for requirement in contract.requirements}
    assert "non_stub_teacher_runtime_invocation" in keys
    assert "provider_runtime_truth_receipts" in keys
    assert "gpu_training_runtime_receipt" in keys
    assert "replay_row_linkage_integrity" in keys
    teacher_requirement = next(
        requirement
        for requirement in contract.requirements
        if requirement.requirement_key == "non_stub_teacher_runtime_invocation"
    )
    assert teacher_requirement.satisfaction_score == 0.0
    assert "non_stub_teacher_runtime_not_verified" in teacher_requirement.blockers
    replay_requirement = next(
        requirement
        for requirement in contract.requirements
        if requirement.requirement_key == "replay_row_linkage_integrity"
    )
    assert replay_requirement.satisfaction_score == 1.0
    assert replay_requirement.promotion_gate is False
    assert (
        "prepare_non_stub_teacher_runtime_invocation_fixture"
        in contract.recommended_next_actions
    )


def test_prepare_teacher_provider_contract_script_roundtrip(tmp_path) -> None:
    scaffold, manifest, rows, allocation_eval = _eval_bundle()
    scaffold_path = tmp_path / "scaffold.json"
    manifest_path = tmp_path / "manifest.json"
    rows_path = tmp_path / "rows.jsonl"
    eval_path = tmp_path / "allocation_eval.json"
    save_economic_wm_scaffold_report(scaffold_path, scaffold)
    save_economic_wm_training_corpus(
        manifest_path=manifest_path,
        rows_path=rows_path,
        manifest=manifest,
        rows=rows,
    )
    save_economic_wm_shadow_allocation_eval(eval_path, allocation_eval)

    payload = run_prepare_economic_wm_teacher_provider_contracts(
        output_dir=tmp_path / "contracts",
        scaffold_report_path=scaffold_path,
        allocation_eval_path=eval_path,
        corpus_manifest_path=manifest_path,
        rows_path=rows_path,
    )

    assert payload["promotion_eligible"] is False
    assert payload["provider_bringup_ready"] is False
    assert payload["gpu_training_ready"] is False
    assert payload["aggregate_scores"]["teacher_real_fraction"] == 0.0
    loaded = load_economic_wm_teacher_provider_contract(
        payload["artifact_refs"]["contract_path"]
    )
    assert loaded.contract_id == payload["contract_id"]
    assert (
        tmp_path / "contracts" / "economic_wm_teacher_provider_contract_v1.md"
    ).exists()
