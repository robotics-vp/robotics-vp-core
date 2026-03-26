from pathlib import Path

from src.orchestrator.context import OrchestratorContext
from src.orchestrator.datapack_engine import DatapackSignals
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.meta_transformer import MetaTransformer
from src.orchestrator.orchestration_transformer import OrchestrationTransformer
from src.orchestrator.pipeline_manager import run_pipeline_step_with_causal_order
from src.orchestrator.semantic_runtime_learning import (
    SemanticRuntimeCounterfactual,
    SemanticRuntimeLearningCorpus,
    SemanticRuntimeLearningRow,
)
from src.orchestrator.semantic_runtime_scorer_training import (
    TORCH_AVAILABLE,
    build_semantic_runtime_scorer_training_dataset,
    load_semantic_runtime_scorer_training_dataset,
    save_semantic_runtime_scorer_checkpoint,
    train_semantic_runtime_scorer_net,
    write_semantic_runtime_scorer_training_dataset,
)
from src.orchestrator.semantic_runtime_scorers import (
    _live_lane_summaries,
    load_semantic_runtime_scorer_package,
    score_semantic_runtime_learning_row,
    train_semantic_runtime_scorer_package,
    write_semantic_runtime_scorer_package,
)
from src.world_model.semantic_world_model import (
    SemanticMetaNode,
    SemanticObjectState,
    SemanticRelationState,
    SemanticWorldModelState,
)


def _semantic_world_model() -> SemanticWorldModelState:
    return SemanticWorldModelState(
        world_model_id="wm_scorer",
        episode_id="episode_scorer",
        task_id="drawer_vase_task",
        objective_preset="safety",
        semantic_tags=["drawer", "vase", "risk:fragility"],
        objects=[
            SemanticObjectState(
                object_id="drawer",
                label="drawer",
                category="container",
                confidence=0.94,
                salience=0.76,
                affordances=["open", "close"],
                state_tags=["occluding"],
            ),
            SemanticObjectState(
                object_id="vase",
                label="vase",
                category="fragile_object",
                confidence=0.91,
                salience=0.88,
                affordances=["avoid_contact", "stabilize"],
                state_tags=["fragile"],
                risk_tags=["fragility"],
            ),
        ],
        relations=[
            SemanticRelationState(
                relation_id="near_vase",
                subject_id="drawer",
                relation_type="near",
                object_id="vase",
                confidence=0.7,
            )
        ],
        meta_nodes=[
            SemanticMetaNode(
                node_id="risk_triage",
                node_type="risk_triage",
                priority="critical",
                score=0.84,
                rationale="fragile object in manipulation corridor",
            ),
            SemanticMetaNode(
                node_id="semantic_memory_refresh",
                node_type="semantic_memory_refresh",
                priority="high",
                score=0.58,
                rationale="refresh object memory before promotion",
            ),
        ],
        capability_scores={
            "risk_reasoning": 0.82,
            "object_memory": 0.79,
            "affordance_grounding": 0.68,
            "fusion_bridge": 0.64,
            "stage2_bridge": 0.42,
            "meta_node_orchestration": 0.76,
        },
        topology={
            "grounded_track_object_count": 2,
            "object_count": 2,
            "relation_count": 1,
            "meta_node_count": 2,
        },
    )


def _row(
    sample_id: str,
    *,
    authority: str,
    success: bool,
    quality_score: float,
    readiness_score: float,
    vla_conf: float,
    dino_conf: float,
    objective_preset: str,
    estimated_regret: float,
) -> SemanticRuntimeLearningRow:
    semantic_summary = {
        "present": True,
        "world_model_id": f"wm_{sample_id}",
        "task_id": "drawer_vase_task",
        "object_count": 2,
        "relation_density": 0.5,
        "affordance_density": 1.5,
        "risk_object_fraction": 0.5,
        "fragile_object_fraction": 0.5,
        "priority_high_fraction": 1.0,
        "capability_mean": 0.68,
        "capability_max": 0.84,
        "risk_reasoning": 0.82,
        "object_memory": 0.78 if authority == "dino" else 0.66,
        "affordance_grounding": 0.72 if authority == "vla" else 0.58,
        "fusion_bridge": 0.64,
        "stage2_bridge": 0.41,
        "meta_node_orchestration": 0.76,
        "risk_triage_score": 0.84,
        "recovery_router_score": 0.22,
        "efficiency_router_score": 0.28,
        "semantic_memory_refresh_score": 0.58,
        "grounded_track_object_count": 2,
        "top_meta_nodes": ["risk_triage", "semantic_memory_refresh"],
        "top_object_labels": ["vase", "drawer"],
        "active_capabilities": ["risk_reasoning", "object_memory"],
    }
    vla_summary = {
        "vla_available": True,
        "vla_confidence_mean": vla_conf,
        "teacher_trace_available": True,
        "teacher_confidence_mean": max(vla_conf - 0.05, 0.0),
        "teacher_object_refs": ["drawer", "vase"],
        "teacher_affordance_hints": ["open"],
        "teacher_risk_hints": ["fragility"],
    }
    dino_summary = {
        "dino_proxy_available": True,
        "dino_proxy_confidence_mean": dino_conf,
        "scene_tracks_available": True,
        "scene_track_count": 2,
        "scene_track_label_confidence_mean": dino_conf,
        "map_first_available": True,
        "map_first_confidence_mean": min(dino_conf, 0.8),
        "scene_tracks_backend": "real",
    }
    fusion_summary = {
        "fusion_available": True,
        "semantic_fusion_confidence_mean": max(vla_conf, dino_conf),
        "annotation_agreement_score": 1.0 - abs(vla_conf - dino_conf),
        "source_confidence_gap": abs(vla_conf - dino_conf),
        "fusion_advantage_score": 0.05 if success else 0.0,
    }
    meta_target = {
        "authority_gt": authority,
        "authority_score": max(vla_conf, dino_conf),
        "confidence_vla": vla_conf,
        "confidence_dino": dino_conf,
        "objective_preset": objective_preset,
        "energy_profile_weights": {"SAFE": 1.0} if objective_preset == "safety" else {"BASE": 1.0},
        "data_mix_weights": {"real": 0.6, "synthetic": 0.3, "hybrid": 0.1},
        "chosen_backend": "pybullet",
        "expected_deltas": {
            "expected_delta_mpl": 2.5 if success else 0.5,
            "expected_delta_error": -0.04 if success else 0.08,
            "expected_delta_energy_Wh": 0.3 if success else 1.2,
        },
        "execution_mode": "bounded_execution" if success else "advisory",
        "bounded_actions": ["set_objective_preset", "set_energy_profile", "set_backend"],
        "plan": [{"action": "route_risk_triage"}],
    }
    orchestration_target = {
        "tool_sequence": [
            {"name": "SET_OBJECTIVE_PRESET", "args": {"preset": objective_preset}, "score": 0.9},
            {"name": "SET_BACKEND", "args": {"backend": "pybullet"}, "score": 0.6},
            {"name": "QUERY_DATAPACKS", "args": {"focus": ["risk_triage"]}, "score": 0.5},
        ],
        "chosen_backend": "pybullet",
        "objective_preset": objective_preset,
        "energy_profile_weights": {"SAFE": 1.0} if objective_preset == "safety" else {"BASE": 1.0},
        "data_mix_weights": {"real": 0.6, "synthetic": 0.3, "hybrid": 0.1},
        "execution_mode": "bounded_execution" if success else "advisory",
        "activation_plan": {"mode": "bounded_execution" if success else "advisory"},
    }
    outcome_summary = {
        "success": success,
        "execution_ready": success,
        "work_order_ready": success,
        "readiness_score": readiness_score,
        "semantic_grounded": True,
        "teacher_runtime_live": True,
        "scene_tracks_non_stub": True,
        "promotion_trace_complete": success,
        "reward_signal": 0.75 if success else 0.35,
        "quality_score": quality_score,
        "semantic_fusion_confidence_mean": max(vla_conf, dino_conf),
    }
    feedback_summary = {
        "annotation_to_world_model": {
            "openvla_available": True,
            "teacher_trace_available": True,
            "dino_proxy_available": True,
            "annotation_agreement_score": fusion_summary["annotation_agreement_score"],
            "semantic_grounding_ready": True,
        },
        "world_model_to_transformers": {
            "top_meta_nodes": ["risk_triage"],
            "active_capabilities": ["risk_reasoning", "object_memory"],
            "object_count": 2,
            "affordance_density": 1.5,
        },
        "transformers_to_runtime": {
            "can_execute": success,
            "readiness_score": readiness_score,
            "quality_score": quality_score,
        },
        "runtime_to_world_model": {
            "reward_signal": outcome_summary["reward_signal"],
            "fusion_quality": fusion_summary["semantic_fusion_confidence_mean"],
            "promotion_trace_complete": success,
        },
    }
    inferential_summary = {
        "preferred_authority": authority,
        "chosen_route_score": quality_score,
        "best_counterfactual_score": max(quality_score - estimated_regret, quality_score),
        "estimated_regret": estimated_regret,
        "route_success_label": success,
        "orchestration_route_success_label": success,
        "authority_success_label": success,
        "semantic_gain_label": True,
        "fusion_gain_label": success,
        "feedback_edges": feedback_summary,
    }
    counterfactuals = [
        SemanticRuntimeCounterfactual(
            counterfactual_id=f"{sample_id}_cf_safety",
            lane="meta_transformer",
            candidate={"objective_preset": "safety"},
            predicted_outcome_score=0.84 if objective_preset != "safety" else 0.6,
            predicted_regret=0.08,
            executable=success,
            rationale="counterfactual_objective_preset:safety",
        ),
        SemanticRuntimeCounterfactual(
            counterfactual_id=f"{sample_id}_cf_authority",
            lane="meta_transformer",
            candidate={"authority_gt": "vla" if authority == "dino" else "dino"},
            predicted_outcome_score=0.77 if authority == "dino" else 0.72,
            predicted_regret=0.11,
            executable=True,
            rationale=f"counterfactual_authority:{'vla' if authority == 'dino' else 'dino'}",
        ),
    ]
    return SemanticRuntimeLearningRow(
        sample_id=sample_id,
        run_id="run_scorer",
        episode_id=sample_id,
        task_id="drawer_vase_task",
        env_id="drawer_vase",
        source_domain="semantic_scorer_test",
        semantic_world_model_summary=semantic_summary,
        semantic_tokens=["object:vase", "object:drawer", "meta_node:risk_triage"],
        vla_summary=vla_summary,
        dino_summary=dino_summary,
        fusion_summary=fusion_summary,
        feedback_summary=feedback_summary,
        meta_transformer_target=meta_target,
        orchestration_transformer_target=orchestration_target,
        outcome_summary=outcome_summary,
        inferential_summary=inferential_summary,
        counterfactuals=counterfactuals,
        artifact_refs={},
        metadata={"skill_mode": "safety_first"},
    )


def _corpus() -> SemanticRuntimeLearningCorpus:
    rows = [
        _row(
            "sample_a",
            authority="dino",
            success=True,
            quality_score=0.86,
            readiness_score=0.91,
            vla_conf=0.62,
            dino_conf=0.82,
            objective_preset="safety",
            estimated_regret=0.04,
        ),
        _row(
            "sample_b",
            authority="vla",
            success=True,
            quality_score=0.79,
            readiness_score=0.84,
            vla_conf=0.83,
            dino_conf=0.58,
            objective_preset="throughput",
            estimated_regret=0.07,
        ),
        _row(
            "sample_c",
            authority="dino",
            success=False,
            quality_score=0.33,
            readiness_score=0.22,
            vla_conf=0.41,
            dino_conf=0.49,
            objective_preset="balanced",
            estimated_regret=0.29,
        ),
    ]
    return SemanticRuntimeLearningCorpus(rows=rows, summary={"row_count": len(rows)})


def _econ_signals() -> EconSignals:
    return EconSignals(
        mpl_urgency=0.28,
        error_urgency=0.6,
        energy_urgency=0.12,
        customer_segment="premium_safety",
        task_family="drawer_vase",
    )


def _datapack_signals() -> DatapackSignals:
    return DatapackSignals(
        data_coverage_score=0.46,
        embedding_diversity=0.51,
        vla_annotation_fraction=0.82,
        guidance_annotation_fraction=0.61,
        semantic_tag_diversity=7,
        data_gaps=["frontier_cases"],
    )


def test_semantic_runtime_scorer_package_trains_scores_and_roundtrips(tmp_path: Path) -> None:
    corpus = _corpus()
    package = train_semantic_runtime_scorer_package(corpus)

    assert package.summary["row_count"] == 3
    assert package.summary["counterfactual_count"] == 6
    scored = score_semantic_runtime_learning_row(package, corpus.rows[0])
    assert 0.0 <= scored.meta_route_success_probability <= 1.0
    assert 0.0 <= scored.orchestration_route_success_probability <= 1.0
    assert scored.counterfactual_scores

    package_path = write_semantic_runtime_scorer_package(tmp_path / "semantic_runtime_scorer_package.json", package)
    loaded = load_semantic_runtime_scorer_package(package_path)
    loaded_score = score_semantic_runtime_learning_row(loaded, corpus.rows[0])
    assert loaded_score.calibrated_authority in {"dino", "vla"}


def test_semantic_runtime_heavyweight_training_dataset_and_checkpoint(tmp_path: Path) -> None:
    dataset = build_semantic_runtime_scorer_training_dataset(_corpus())
    assert len(dataset.meta_route_features) == 3
    assert len(dataset.counterfactual_features) == 6

    dataset_path = write_semantic_runtime_scorer_training_dataset(
        tmp_path / "semantic_runtime_scorer_training_dataset.json",
        dataset,
    )
    loaded = load_semantic_runtime_scorer_training_dataset(dataset_path)
    assert loaded.meta_route_feature_names == dataset.meta_route_feature_names

    training = train_semantic_runtime_scorer_net(loaded, epochs=2, hidden_dim=16)
    if TORCH_AVAILABLE:
        assert training["trained"] is True
        checkpoint_path = save_semantic_runtime_scorer_checkpoint(
            tmp_path / "semantic_runtime_scorer_model.pt",
            training,
        )
        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()
    else:
        assert training["trained"] is False


def test_pipeline_manager_emits_semantic_runtime_scoring_for_both_transformers() -> None:
    package = train_semantic_runtime_scorer_package(_corpus())

    class DummyPlan:
        cross_module_constraints = {}
        primitive_updates = {}

        def to_dict(self):
            return {"plan": "semantic"}

    class DummySemanticOrchestrator:
        def build_update_plan(self, econ_signals, datapack_signals, meta_out=None):
            assert meta_out is not None
            return DummyPlan()

        def apply_update_plan(self, semantic_plan):
            return semantic_plan

        def snapshot(self):
            return {"semantic_state": "ok"}

    class DummyEconController:
        def compute_signals(self, datapacks):
            return _econ_signals()

    class DummyDatapackEngine:
        def compute_signals(self, datapacks, econ_signals):
            return _datapack_signals()

    ctx = OrchestratorContext(
        env_name="drawer_vase",
        engine_type="pybullet",
        task_type="drawer_open",
        customer_segment="premium_safety",
        market_region="US",
        objective_vector=[0.6, 0.2, 0.15, 0.8, 0.0],
        wage_human=20.0,
        energy_price_kWh=0.14,
        mean_delta_mpl=4.0,
        mean_delta_error=0.08,
        mean_delta_j=0.1,
        mean_trust=0.72,
        mean_w_econ=0.61,
        profile_summaries={
            "BASE": {"mpl": 60.0, "error": 0.03, "energy_Wh": 14.0, "risk": 0.2},
            "SAFE": {"mpl": 50.0, "error": 0.01, "energy_Wh": 16.0, "risk": 0.1},
        },
        semantic_world_model=_semantic_world_model(),
        semantic_metadata={
            "vla_summary": {
                "vla_available": True,
                "vla_confidence_mean": 0.7,
                "teacher_trace_available": True,
                "teacher_confidence_mean": 0.66,
                "teacher_object_refs": ["drawer", "vase"],
                "teacher_affordance_hints": ["open"],
                "teacher_risk_hints": ["fragility"],
            },
            "dino_summary": {
                "dino_proxy_available": True,
                "dino_proxy_confidence_mean": 0.79,
                "scene_tracks_available": True,
                "scene_track_count": 2,
                "scene_track_label_confidence_mean": 0.78,
                "map_first_available": True,
                "map_first_confidence_mean": 0.71,
                "scene_tracks_backend": "real",
            },
            "fusion_summary": {
                "fusion_available": True,
                "semantic_fusion_confidence_mean": 0.76,
                "annotation_agreement_score": 0.91,
                "source_confidence_gap": 0.09,
                "fusion_advantage_score": 0.03,
            },
            "scene_tracks_non_stub": True,
            "teacher_runtime_live": True,
        },
    )

    result = run_pipeline_step_with_causal_order(
        econ_controller=DummyEconController(),
        datapack_engine=DummyDatapackEngine(),
        semantic_orchestrator=DummySemanticOrchestrator(),
        meta_transformer=MetaTransformer(d_shared=24),
        orchestration_transformer=OrchestrationTransformer(hidden=32),
        semantic_world_model=_semantic_world_model(),
        orchestrator_context=ctx,
        semantic_runtime_scorers=package,
        orchestration_instruction="prioritize safety and fragile-object routing",
    )

    assert result["meta_transformer_execution"]["execution_mode"] == "bounded_execution"
    assert result["orchestration_transformer_execution"]["execution_mode"] == "bounded_execution"
    assert 0.0 <= result["semantic_runtime_scoring"]["meta_route_success_probability"] <= 1.0
    assert 0.0 <= result["semantic_runtime_scoring"]["orchestration_route_success_probability"] <= 1.0
    assert result["semantic_runtime_scoring"]["counterfactual_scores"]


def test_live_lane_summaries_keep_passthrough_scene_tracks_out_of_non_stub_truth() -> None:
    ctx = OrchestratorContext(
        env_name="drawer_vase_env",
        engine_type="pybullet",
        task_type="drawer_vase_task",
        customer_segment="shadow",
        market_region="US",
        objective_vector=[0.6, 0.2, 0.1, 0.1, 0.0],
        wage_human=20.0,
        energy_price_kWh=0.12,
        mean_delta_mpl=0.0,
        mean_delta_error=0.0,
        mean_delta_j=0.0,
        mean_trust=0.7,
        mean_w_econ=0.2,
        profile_summaries={},
        semantic_metadata={
            "scene_tracks_backend": "passthrough",
            "scene_tracks_non_stub": True,
            "teacher_runtime_live": True,
        },
    )

    _, _, _, _, _, orchestration_target = _live_lane_summaries(
        semantic_world_model=_semantic_world_model(),
        orchestrator_context=ctx,
        dino_summary={
            "dino_proxy_available": True,
            "dino_proxy_confidence_mean": 0.74,
            "scene_tracks_available": True,
            "scene_track_count": 2,
            "scene_track_label_confidence_mean": 0.73,
            "map_first_available": False,
            "map_first_confidence_mean": 0.0,
            "scene_tracks_backend": "passthrough",
        },
        vla_summary={
            "vla_available": True,
            "vla_confidence_mean": 0.61,
            "teacher_trace_available": True,
            "teacher_confidence_mean": 0.6,
            "teacher_object_refs": ["drawer"],
            "teacher_affordance_hints": ["open"],
            "teacher_risk_hints": [],
        },
        fusion_summary={
            "fusion_available": True,
            "semantic_fusion_confidence_mean": 0.62,
            "annotation_agreement_score": 0.81,
            "source_confidence_gap": 0.13,
            "fusion_advantage_score": 0.01,
        },
    )

    outcome_summary = orchestration_target["_outcome_summary"]
    assert outcome_summary["teacher_runtime_live"] is True
    assert outcome_summary["scene_tracks_non_stub"] is False
