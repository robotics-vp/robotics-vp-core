from __future__ import annotations

import json

from scripts.economic_world_model.evaluate_economic_wm_shadow_allocations import (
    run_evaluate_economic_wm_shadow_allocations,
)
from src.economics.economic_wm_entry import evaluate_economic_wm_entry_preflight
from src.world_model.economic_world_model import (
    build_economic_wm_scaffold_report,
    build_economic_wm_shadow_allocation_eval,
    build_economic_wm_training_corpus_manifest,
    load_economic_wm_shadow_allocation_eval,
    save_economic_wm_scaffold_report,
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


def test_shadow_allocation_eval_recommends_allowed_non_gpu_candidate() -> None:
    scaffold = _scaffold_report()
    manifest, rows = _corpus(scaffold)

    eval_report = build_economic_wm_shadow_allocation_eval(
        scaffold_report=scaffold,
        corpus_manifest=manifest,
        rows=rows,
    )

    assert eval_report.version == "economic_wm_shadow_allocation_eval_v1"
    assert eval_report.authority_class == "shadow_eval_only"
    assert eval_report.reward_math_mutation is False
    assert eval_report.promotion_eligible is False
    assert eval_report.ready_for_training is False
    assert eval_report.row_count == 5
    assert eval_report.benchmark_ready_count == 2
    assert eval_report.shadow_only_count == 3
    assert (
        eval_report.recommended_candidate
        == "prepare_teacher_provider_evidence_contracts"
    )
    gpu_candidate = next(
        candidate
        for candidate in eval_report.candidates
        if candidate.label == "run_gpu_training"
    )
    assert gpu_candidate.allowed is False
    assert "gpu_training_not_run" in gpu_candidate.denial_reasons


def test_shadow_allocation_eval_script_roundtrip(tmp_path) -> None:
    scaffold = _scaffold_report()
    manifest, rows = _corpus(scaffold)
    scaffold_path = tmp_path / "scaffold.json"
    manifest_path = tmp_path / "manifest.json"
    rows_path = tmp_path / "rows.jsonl"
    save_economic_wm_scaffold_report(scaffold_path, scaffold)
    save_economic_wm_training_corpus(
        manifest_path=manifest_path,
        rows_path=rows_path,
        manifest=manifest,
        rows=rows,
    )

    payload = run_evaluate_economic_wm_shadow_allocations(
        output_dir=tmp_path / "eval",
        scaffold_report_path=scaffold_path,
        corpus_manifest_path=manifest_path,
        rows_path=rows_path,
    )

    assert payload["promotion_eligible"] is False
    assert payload["reward_math_mutation"] is False
    assert (
        payload["recommended_candidate"]
        == "prepare_teacher_provider_evidence_contracts"
    )
    loaded = load_economic_wm_shadow_allocation_eval(
        payload["artifact_refs"]["eval_path"]
    )
    assert loaded.eval_id == payload["eval_id"]
    assert (tmp_path / "eval" / "economic_wm_shadow_allocation_eval_v1.md").exists()
    assert (
        json.loads(
            (
                tmp_path / "eval" / "economic_wm_shadow_allocation_eval_v1.json"
            ).read_text()
        )["authority_class"]
        == "shadow_eval_only"
    )
