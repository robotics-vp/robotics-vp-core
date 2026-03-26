import json
from pathlib import Path

import numpy as np

from src.evidence.teacher_trace import TeacherStep, TeacherTrace
from src.orchestrator.semantic_runtime_learning import (
    build_meta_transformer_runtime_dataset,
    build_orchestration_runtime_dataset,
    build_semantic_runtime_learning_corpus,
    write_semantic_runtime_learning_corpus,
)
from src.replay.dataset import ReplayDatasetBundle
from src.replay.schema import ReplayDatasetManifest, ReplayEpisodeRecord, ReplayStepRecord, ReplayWindowRecord
from src.utils.config_digest import sha256_json
from src.vla.semantic_evidence import build_vla_semantic_evidence_payload, save_vla_semantic_evidence_npz
from src.world_model.semantic_world_model import (
    SemanticMetaNode,
    SemanticObjectState,
    SemanticRelationState,
    SemanticWorldModelState,
)


def _semantic_world_model() -> SemanticWorldModelState:
    return SemanticWorldModelState(
        world_model_id="wm_runtime",
        episode_id="episode_runtime",
        task_id="drawer_vase_task",
        objective_preset="balanced",
        semantic_tags=["drawer", "vase", "risk:fragility", "affordance:open"],
        objects=[
            SemanticObjectState(
                object_id="object_drawer",
                label="drawer",
                category="container",
                confidence=0.94,
                salience=0.76,
                affordances=["open", "close", "grasp_handle"],
                state_tags=["occluding"],
                risk_tags=[],
            ),
            SemanticObjectState(
                object_id="object_vase",
                label="vase",
                category="fragile_object",
                confidence=0.93,
                salience=0.91,
                affordances=["avoid_contact", "stabilize"],
                state_tags=["fragile"],
                risk_tags=["fragility"],
            ),
        ],
        relations=[
            SemanticRelationState(
                relation_id="rel_runtime",
                subject_id="object_drawer",
                relation_type="near",
                object_id="object_vase",
                confidence=0.71,
            )
        ],
        meta_nodes=[
            SemanticMetaNode(
                node_id="node_risk",
                node_type="risk_triage",
                priority="critical",
                score=0.84,
                rationale="fragile object near manipulated container",
            ),
            SemanticMetaNode(
                node_id="node_refresh",
                node_type="semantic_memory_refresh",
                priority="high",
                score=0.58,
                rationale="refresh object memory before promotion",
            ),
        ],
        capability_scores={
            "risk_reasoning": 0.79,
            "object_memory": 0.82,
            "affordance_grounding": 0.61,
            "fusion_bridge": 0.64,
            "stage2_bridge": 0.48,
            "meta_node_orchestration": 0.77,
        },
        topology={
            "grounded_track_object_count": 2,
            "object_count": 2,
            "relation_count": 1,
            "meta_node_count": 2,
        },
        metadata={"grounded_scene": {"grounding_mode": "scene_tracks"}},
    )


def _teacher_trace() -> TeacherTrace:
    return TeacherTrace.from_components(
        episode_id="episode_runtime",
        teacher_id="openvla",
        modality="vision_language_action",
        instruction="open the drawer carefully without touching the vase",
        steps=[
            TeacherStep(
                step_idx=0,
                instruction="open the drawer carefully without touching the vase",
                confidence=0.73,
                semantic_tags=["drawer", "vase", "risk:fragility"],
                metadata={
                    "object_refs": ["drawer", "vase"],
                    "affordance_hints": ["open"],
                    "risk_hints": ["fragility"],
                },
            )
        ],
        summary={"teacher_confidence_mean": 0.73},
        metadata={
            "semantic_tags": ["drawer", "vase", "risk:fragility"],
            "object_refs": ["drawer", "vase"],
            "affordance_hints": ["open"],
            "risk_hints": ["fragility"],
        },
    )


def _bundle(tmp_path: Path, *, include_teacher_trace: bool = True) -> ReplayDatasetBundle:
    world_model_path = tmp_path / "episode_runtime_semantic_world_model_v1.json"
    world_model_path.write_text(json.dumps(_semantic_world_model().to_dict(), indent=2), encoding="utf-8")

    teacher_trace_path = tmp_path / "episode_runtime_teacher_trace_v1.json"
    if include_teacher_trace:
        teacher_trace_path.write_text(json.dumps(_teacher_trace().to_dict(), indent=2), encoding="utf-8")

    scene_tracks_path = tmp_path / "episode_runtime_scene_tracks_v1.npz"
    np.savez_compressed(
        scene_tracks_path,
        **{
            "scene_tracks_v1/track_ids": np.array(["track_1", "track_2"], dtype="U16"),
            "scene_tracks_v1/track_label_confidence": np.array([0.87, 0.81], dtype=np.float32),
            "scene_tracks_v1/track_motion_score": np.array([0.14, 0.09], dtype=np.float32),
            "scene_tracks_v1/summary_json": np.array(
                [json.dumps({"backend_selected": "real", "training_eligible": True})], dtype="U512"
            ),
        },
    )

    vla_path = tmp_path / "episode_runtime_vla_semantic_evidence_v1.npz"
    vla_payload = build_vla_semantic_evidence_payload(
        scene_tracks={"scene_tracks_v1/track_ids": np.array(["track_1", "track_2"], dtype="U16")},
        vla_payload={
            "vla_available": True,
            "confidence": 0.69,
            "source": "openvla",
            "semantic_tags": ["drawer", "vase"],
            "object_refs": ["drawer", "vase"],
            "affordance_hints": ["open"],
            "risk_hints": ["fragility"],
        },
        teacher_trace_ref=str(teacher_trace_path) if include_teacher_trace else None,
        instruction="open the drawer carefully without touching the vase",
    )
    save_vla_semantic_evidence_npz(vla_path, vla_payload)

    episode_provenance = {
        "semantic_world_model_ref": str(world_model_path),
        "scene_tracks_ref": str(scene_tracks_path),
        "vla_semantic_evidence_ref": str(vla_path),
    }
    if include_teacher_trace:
        episode_provenance["teacher_trace_ref"] = str(teacher_trace_path)

    episode = ReplayEpisodeRecord(
        run_id="run_runtime",
        episode_id="episode_runtime",
        task_id="drawer_vase_task",
        env_id="drawer_vase",
        source_domain="semantic_runtime_test",
        seed=7,
        status="success",
        started_at="2026-03-24T00:00:00Z",
        ended_at="2026-03-24T00:00:04Z",
        total_steps=1,
        total_reward=12.5,
        skill_mode="safety_first",
        condition_vector={"goal": "open_drawer"},
        condition_vector_values=[1.0],
        objective_tensor_summary={"objective_preset": "safety"},
        objective_tensor_ref=None,
        econ_tensor_summary={"wage_parity": 0.98},
        econ_tensor_ref=None,
        pricing_summary={"confidence": 0.82},
        pricing_tick_refs=[],
        constraint_flags=[],
        regal_summary={},
        datapack_summary={},
        ledger_event_ids=[],
        metadata={
            "execution_preconditions": {"ready": True, "readiness_score": 1.0},
            "source_execution_work_order": {"ready": True, "decision": "admit_datapack"},
            "future_training_signals": {
                "semantic_memory_grounded": True,
                "teacher_runtime_live": True,
                "scene_tracks_non_stub": True,
                "promotion_trace_complete": True,
            },
            "semantic_fusion_confidence_mean": 0.74,
            "selection_summary": {
                "selection_policy": "heuristic_plus_learned_helper",
                "selected_ids": ["dp_runtime"],
                "selected_gap_fill_tags": ["drawer"],
                "selection_helper_status": {
                    "status": "available",
                    "promotion_stage": "shadow_candidate",
                    "benchmark_gate_ready": False,
                },
                "selection_meta_choice": {
                    "selected_datapack_id": "dp_runtime",
                    "selection_policy": "heuristic_plus_learned_helper",
                    "candidate_count": 2,
                    "selected_gap_fill_ratio": 0.5,
                    "selected_execution_ready": True,
                    "selected_non_heuristic_grounding": True,
                    "selected_benchmark_eligible": True,
                    "top_score": 2.4,
                    "margin_to_runner_up": 0.7,
                    "selected_quality_score": 0.85,
                },
                "top_candidates": [
                    {
                        "datapack_id": "dp_runtime",
                        "score": 2.4,
                        "selection_features": {
                            "quality_score": 0.85,
                            "execution_ready": 1.0,
                            "semantic_grounding_non_heuristic": 1.0,
                            "benchmark_eligible": 1.0,
                        },
                        "benchmark_support": {
                            "execution_ready": True,
                            "semantic_grounding_non_heuristic": True,
                            "benchmark_eligible": True,
                        },
                    },
                    {
                        "datapack_id": "dp_alt",
                        "score": 1.7,
                    },
                ],
            },
        },
        provenance=episode_provenance,
    )
    step = ReplayStepRecord(
        run_id="run_runtime",
        episode_id="episode_runtime",
        step_idx=0,
        obs={},
        obs_vector=[0.1, 0.2],
        action={"type": "noop"},
        action_vector=[0.0],
        reward=12.5,
        reward_decomposition={},
        done=True,
        task_id="drawer_vase_task",
        env_id="drawer_vase",
        condition_vector={"goal": "open_drawer"},
        condition_vector_values=[1.0],
        skill_mode="safety_first",
        objective_tensor_summary={"objective_preset": "safety"},
        objective_tensor_ref=None,
        econ_tensor_summary={"wage_parity": 0.98},
        econ_tensor_ref=None,
        constraint_flags=[],
        pricing_tick_ref=None,
        ledger_event_ref=None,
        source_domain="semantic_runtime_test",
        seed=7,
        timestamp="2026-03-24T00:00:01Z",
        metadata={},
        provenance={},
    )
    window = ReplayWindowRecord(
        run_id="run_runtime",
        episode_id="episode_runtime",
        window_id="window_0",
        start_step=0,
        end_step=0,
        task_id="drawer_vase_task",
        env_id="drawer_vase",
        source_domain="semantic_runtime_test",
        seed=7,
        timestamp="2026-03-24T00:00:01Z",
        reward_sum=12.5,
        obs_vector_mean=[0.1, 0.2],
        action_vector_mean=[0.0],
        condition_vector={"goal": "open_drawer"},
        condition_vector_values=[1.0],
        skill_mode="safety_first",
        objective_tensor_summary={"objective_preset": "safety"},
        econ_tensor_summary={"wage_parity": 0.98},
        pricing_summary={"confidence": 0.82},
        constraint_flags=[],
        metadata={},
        provenance={},
    )
    manifest = ReplayDatasetManifest(
        schema_version="shadow_replay_dataset_v1",
        run_ids=["run_runtime"],
        source_adapters=["semantic_runtime_test"],
        files={},
        num_episodes=1,
        num_steps=1,
        num_windows=1,
        obs_dim=2,
        action_dim=1,
        condition_dim=1,
        skill_modes=["safety_first"],
        config_digest=sha256_json({"run": "runtime"}),
        dataset_digest=sha256_json({"episode": "runtime"}),
        created_at="2026-03-24T00:00:00Z",
        metadata={},
        artifact_schema_fingerprint={},
        provenance_summary={},
    )
    return ReplayDatasetBundle(
        manifest=manifest,
        episodes=[episode],
        steps=[step],
        windows=[window],
        root_dir=str(tmp_path),
    )


def test_semantic_runtime_learning_corpus_builds_feedback_and_counterfactuals(tmp_path: Path) -> None:
    corpus = build_semantic_runtime_learning_corpus(_bundle(tmp_path))

    assert corpus.summary["row_count"] == 1
    row = corpus.rows[0]
    assert row.semantic_world_model_summary["world_model_id"] == "wm_runtime"
    assert row.vla_summary["vla_available"] is True
    assert row.dino_summary["scene_tracks_available"] is True
    assert row.feedback_summary["annotation_to_world_model"]["openvla_available"] is True
    assert row.meta_transformer_target["objective_preset"] == "safety"
    assert row.orchestration_transformer_target["tool_sequence"]
    assert row.counterfactuals
    assert row.inferential_summary["authority_success_label"] is True
    assert row.inferential_summary["semantic_gain_label"] is True


def test_semantic_runtime_learning_datasets_and_write_paths(tmp_path: Path) -> None:
    corpus = build_semantic_runtime_learning_corpus(_bundle(tmp_path))

    meta_samples = build_meta_transformer_runtime_dataset(corpus.rows)
    orchestration_samples = build_orchestration_runtime_dataset(corpus.rows)
    assert corpus.summary["route_success_count"] == 1
    assert corpus.summary["authority_success_count"] == 1
    assert len(meta_samples) == 1
    assert meta_samples[0].sample_id == corpus.rows[0].sample_id
    assert meta_samples[0].authority_gt in {"dino", "vla"}
    assert meta_samples[0].objective_preset == corpus.rows[0].meta_transformer_target["objective_preset"]
    assert meta_samples[0].chosen_backend == corpus.rows[0].meta_transformer_target["chosen_backend"]
    assert meta_samples[0].task_context["selection_summary"]["selected_ids"] == ["dp_runtime"]
    assert meta_samples[0].task_context["semantic_summary"]["world_model_id"] == "wm_runtime"
    assert len(orchestration_samples) == 1
    assert orchestration_samples[0].target_tool_sequence
    assert orchestration_samples[0].context.semantic_metadata["semantic_world_model_summary"]["world_model_id"] == "wm_runtime"
    assert orchestration_samples[0].context.semantic_metadata["selection_summary"]["selected_ids"] == ["dp_runtime"]
    assert orchestration_samples[0].metadata["selection_summary"]["selection_meta_choice"]["selected_datapack_id"] == "dp_runtime"

    written = write_semantic_runtime_learning_corpus(tmp_path / "exported", corpus)
    assert Path(written["rows_path"]).exists()
    assert Path(written["summary_path"]).exists()


def test_semantic_runtime_learning_corpus_handles_missing_teacher_trace(tmp_path: Path) -> None:
    corpus = build_semantic_runtime_learning_corpus(_bundle(tmp_path, include_teacher_trace=False))

    assert corpus.summary["row_count"] == 1
    row = corpus.rows[0]
    assert row.vla_summary["teacher_trace_available"] is False
    assert row.vla_summary["teacher_confidence_mean"] == 0.0
    assert row.vla_summary["instruction"] == "open the drawer carefully without touching the vase"
