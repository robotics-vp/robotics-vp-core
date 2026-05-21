from __future__ import annotations

import json

from scripts.economic_world_model.materialize_economic_wm_training_rows import (
    run_materialize_economic_wm_training_rows,
)
from src.economics.economic_wm_entry import evaluate_economic_wm_entry_preflight
from src.world_model.economic_world_model import (
    build_economic_wm_training_corpus_manifest,
    build_economic_wm_scaffold_report,
    load_economic_wm_replay_feature_rows,
    load_economic_wm_training_corpus_manifest,
    save_economic_wm_scaffold_report,
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
        "runtime_packet_path": f"artifacts/{video_id}/runtime_packet.json",
        "counterfactual_eval_path": f"artifacts/{video_id}/counterfactual_eval.json",
        "value_target_pack_path": f"artifacts/{video_id}/value_targets.json",
        "value_ledger_receipt_path": f"artifacts/{video_id}/value_ledger.json",
        "governance_trace_path": f"artifacts/{video_id}/governance_trace.json",
        "benchmark_gate_path": f"artifacts/{video_id}/benchmark_gate.json",
        "reconstruction_grounding_report_path": f"artifacts/{video_id}/reconstruction_grounding.json",
        "teacher_contract_path": f"artifacts/{video_id}/teacher_contract.json",
        "teacher_trace_path": f"artifacts/{video_id}/teacher_trace.json",
        "diffusion_provider_truth": "manifest_declared",
        "diffusion_backend_selected": "governed_video_scaffold",
        "routing_score": 1.0 if ready else 0.25,
    }


def _admission_rows():
    return [
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


def test_economic_wm_training_rows_preserve_benchmark_and_shadow_truth() -> None:
    scaffold = _scaffold_report()
    manifest, rows = build_economic_wm_training_corpus_manifest(
        scaffold_report=scaffold,
        admission_rows=_admission_rows(),
        rows_path="rows.jsonl",
        artifact_refs={"admission_log_path": "proposal_admission_v1.jsonl"},
    )

    assert manifest.version == "economic_wm_training_corpus_manifest_v1"
    assert manifest.row_count == 5
    assert manifest.benchmark_ready_count == 2
    assert manifest.shadow_only_count == 3
    assert manifest.ready_for_training is False
    assert manifest.promotion_eligible is False
    assert "gpu_training_not_run" in manifest.training_blockers
    assert rows[0].version == "economic_wm_replay_feature_row_v1"
    assert rows[0].benchmark_ready is True
    assert rows[0].gpu_training_eligible is False
    assert rows[0].feature_vector["replay_export_flow"] == 1.0
    assert rows[2].shadow_only is True
    assert rows[2].target_vector["shadow_gap_weight"] == 1.0
    assert "teacher_runtime_real_missing" in rows[0].denied_promotion_reasons
    assert "blocked::camera_calibration_missing" in rows[2].denied_promotion_reasons


def test_materialize_economic_wm_training_rows_script_roundtrip(tmp_path) -> None:
    scaffold = _scaffold_report()
    scaffold_path = tmp_path / "scaffold.json"
    save_economic_wm_scaffold_report(scaffold_path, scaffold)
    admission_path = tmp_path / "proposal_admission_v1.jsonl"
    admission_path.write_text(
        "\n".join(json.dumps(row) for row in _admission_rows()) + "\n",
        encoding="utf-8",
    )

    payload = run_materialize_economic_wm_training_rows(
        output_dir=tmp_path / "rows",
        scaffold_report_path=scaffold_path,
        admission_log_path=admission_path,
    )

    assert payload["row_count"] == 5
    assert payload["benchmark_ready_count"] == 2
    assert payload["promotion_eligible"] is False
    loaded_manifest = load_economic_wm_training_corpus_manifest(
        payload["artifact_refs"]["manifest_path"]
    )
    loaded_rows = load_economic_wm_replay_feature_rows(
        payload["artifact_refs"]["rows_path"]
    )
    assert loaded_manifest.corpus_id == payload["corpus_id"]
    assert len(loaded_rows) == 5
    assert (
        loaded_rows[0]
        .source_refs["runtime_packet_path"]
        .endswith("runtime_packet.json")
    )
    assert (tmp_path / "rows" / "economic_wm_training_corpus_manifest_v1.md").exists()


def test_economic_wm_training_rows_preserve_native_lower_wm_refs() -> None:
    scaffold = _scaffold_report()
    admission = _admission_row("native_refs", "proposal_0", ready=True, calibrated=True)
    admission.update(
        {
            "canonical_lower_wm_reference_pack_path": "artifacts/native_refs/canonical_lower_wm",
            "perception_grounding_world_state_path": "artifacts/native_refs/perception.json",
            "sim_synth_physics_world_state_path": "artifacts/native_refs/sim.json",
            "embodiment_actuation_world_state_path": "artifacts/native_refs/embodiment.json",
            "canonical_lower_wm_refs": {
                "perception_grounding": {
                    "artifact_path": "artifacts/native_refs/perception.json",
                    "version": "perception_grounding_world_state_v1",
                },
                "sim_synth_physics": {
                    "artifact_path": "artifacts/native_refs/sim.json",
                    "version": "sim_synth_physics_world_state_v1",
                },
                "embodiment_actuation": {
                    "artifact_path": "artifacts/native_refs/embodiment.json",
                    "version": "embodiment_actuation_world_state_v1",
                },
            },
        }
    )

    _, rows = build_economic_wm_training_corpus_manifest(
        scaffold_report=scaffold,
        admission_rows=[admission],
        rows_path="rows.jsonl",
    )

    source_refs = rows[0].source_refs
    assert source_refs["perception_grounding_world_state_path"].endswith(
        "perception.json"
    )
    assert source_refs["sim_synth_physics_world_state_path"].endswith("sim.json")
    assert source_refs["embodiment_actuation_world_state_path"].endswith(
        "embodiment.json"
    )
    assert set(source_refs["canonical_lower_wm_refs"]) == {
        "perception_grounding",
        "sim_synth_physics",
        "embodiment_actuation",
    }
