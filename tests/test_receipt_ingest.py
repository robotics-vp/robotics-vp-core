import json

from scripts.train_semantic_runtime_scorers import _run_training as _run_semantic_runtime_scorer_training
from scripts.train_semantic_runtime_scorers import parse_args as _parse_semantic_runtime_scorer_args
from src.orchestrator.shadow_advisory import build_shadow_advisory_output
from src.orchestrator.semantic_runtime_scorers import (
    SemanticRuntimeCounterfactualScore,
    SemanticRuntimeScoreResult,
)
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.replay.receipt_ingest import (
    build_synthetic_receipt_label_bundle,
    load_receipt_label_bundle,
    write_receipt_label_bundle,
)
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_receipt_ingest_roundtrip_and_shadow_advisory_consumption(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    receipt_dir = tmp_path / "receipt_labels"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=2,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)
    dataset = load_replay_dataset(dataset_dir)
    bundle = build_synthetic_receipt_label_bundle(dataset)
    paths = write_receipt_label_bundle(bundle, receipt_dir)
    restored = load_receipt_label_bundle(receipt_dir)

    assert restored.coverage_summary()["deployment_receipts"] == len(bundle.deployment_receipts)
    assert restored.coverage_summary()["covered_episode_count"] == dataset.manifest.num_episodes
    assert paths["bundle"]

    overlay_path = tmp_path / "epiplexity_overlays.jsonl"
    overlay_path.write_text(
        json.dumps(
            {
                "pack_id": dataset.episodes[0].datapack_summary["datapack_id"],
                "epiplexity_summary": {
                    "canonical_tokens": {
                        "steps_5_bs_4": {
                            "mean": {"delta_epi_vs_baseline": 0.25, "epi_per_flop": 0.4},
                            "confidence": 0.8,
                        }
                    },
                    "_default": {"repr_id": "canonical_tokens", "budget_id": "steps_5_bs_4"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    advisory = build_shadow_advisory_output(
        replay_dataset_dir=str(dataset_dir),
        receipt_label_dir=str(receipt_dir),
        epiplexity_overlay_path=str(overlay_path),
    )
    assert advisory["summary"]["receipt_label_coverage"]["total_labels"] > 0
    assert advisory["episodes"][0]["receipt_feedback"]["deployment_outcome"] is not None
    assert advisory["summary"]["epiplexity_overlay_joins"] >= 1
    assert advisory["episodes"][0]["epiplexity_evidence"]["overlay_joined"] is True
    assert advisory["episodes"][0]["inferential_learnability_contract"]["learnability_class"] == "portable_receipt_backed"
    assert advisory["episodes"][0]["inferential_admission"]["authority_class"] == "work_order"
    assert "execution_preconditions" in advisory["episodes"][0]
    assert advisory["adaptation_budget"]["summary"]["work_orders"] >= 1
    assert advisory["inferential_learnability_summary"]["contract_count"] == advisory["summary"]["episodes"]
    assert advisory["inferential_admission_contract"]["contract_kind"] == "inferential_admission_contract_v1"
    assert advisory["inferential_admission_contract"]["summary"]["decision_count"] == advisory["summary"]["episodes"]
    assert advisory["inferential_work_orders"][0]["metadata"]["contract_kind"] == "inferential_execution_work_order_v1"
    assert advisory["semantic_runtime_scorer_preconditions"]["fallback_active"] is True
    assert advisory["semantic_runtime_scorer_work_orders"][0]["reason"] == "semantic_runtime_scorer_package_missing"


def test_shadow_advisory_threads_semantic_runtime_scores_into_queue_metadata(tmp_path, monkeypatch):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=7,
        episodes=2,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)

    monkeypatch.setattr(
        "src.orchestrator.shadow_advisory._resolve_semantic_runtime_scorer_package",
        lambda **kwargs: (
            object(),
            "memory://semantic_runtime_scorer",
            {
                "contract_type": "runtime_package",
                "benchmark_gate": {"ready": True},
                "execution_preconditions": {"ready": True},
                "promotion_stage": "promoted",
            },
        ),
    )

    def _fake_score(*args, **kwargs):
        return SemanticRuntimeScoreResult(
            score_id="score_test",
            semantic_world_model_id="wm_test",
            meta_route_success_probability=0.82,
            orchestration_route_success_probability=0.77,
            predicted_regret=0.31,
            preferred_authority="dino",
            calibrated_authority="vla",
            chosen_authority_confidence=0.74,
            alternate_authority_confidence=0.81,
            authority_switch_recommended=True,
            counterfactual_scores=[
                SemanticRuntimeCounterfactualScore(
                    counterfactual_id="cf_test",
                    rationale="counterfactual",
                    rescored_value=0.68,
                    baseline_score=0.42,
                    executable=True,
                    candidate={"authority_gt": "vla"},
                )
            ],
            feedback_summary={},
            metadata={},
        )

    monkeypatch.setattr(
        "src.orchestrator.shadow_advisory.score_semantic_runtime_learning_row",
        _fake_score,
    )

    advisory = build_shadow_advisory_output(replay_dataset_dir=str(dataset_dir))

    assert advisory["summary"]["semantic_runtime_scorer_episodes"] == advisory["summary"]["episodes"]
    assert advisory["summary"]["semantic_runtime_scorer_ready"] is True
    assert advisory["summary"]["semantic_runtime_scorer_contract_type"] == "runtime_package"
    assert advisory["summary"]["semantic_runtime_scorer_package_ref"] == "memory://semantic_runtime_scorer"
    episode = advisory["episodes"][0]
    assert episode["semantic_runtime_score"]["meta_route_success_probability"] == 0.82
    assert "runtime_score_candidate" in episode["replay_queue_tags"]
    assert "authority_switch_review" in episode["replay_queue_tags"]
    queue_entry = advisory["live_queue_selection"]["entries"][0]
    assert queue_entry["metadata"]["semantic_runtime_score"]["meta_route_success_probability"] == 0.82
    assert advisory["semantic_runtime_scorer_preconditions"]["ready"] is True
    assert advisory["semantic_runtime_scorer_work_orders"] == []


def test_shadow_advisory_prefers_runtime_package_contract(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    scorer_dir = tmp_path / "semantic_runtime_scorers"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=11,
        episodes=3,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)

    args = _parse_semantic_runtime_scorer_args(
        [
            "--replay-dataset",
            str(dataset_dir),
            "--output-dir",
            str(scorer_dir),
            "--trainer",
            "linear",
            "--skip-regal-runner",
        ]
    )
    _run_semantic_runtime_scorer_training(args, runner=None)

    advisory = build_shadow_advisory_output(
        replay_dataset_dir=str(dataset_dir),
        semantic_runtime_scorer_package_path=str(
            scorer_dir / "semantic_runtime_scorer_runtime_package.json"
        ),
    )

    assert advisory["summary"]["semantic_runtime_scorer_contract_type"] == "runtime_package"
    assert advisory["semantic_runtime_scorer_preconditions"]["contract_type"] == "runtime_package"
    assert advisory["semantic_runtime_scorer_preconditions"]["fallback_active"] is False
