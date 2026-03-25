from src.orchestrator.context import OrchestratorContext
from src.orchestrator.datapack_engine import DatapackSignals
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.meta_transformer import MetaTransformer
from src.orchestrator.orchestration_transformer import (
    OrchestrationTransformer,
    _encode_ctx,
    propose_orchestrated_plan,
)
from src.orchestrator.pipeline_manager import run_pipeline_step_with_causal_order
from src.orchestrator.semantic_transformer_bridge import ORCHESTRATION_CTX_DIM
from src.world_model.semantic_world_model import (
    SemanticMetaNode,
    SemanticObjectState,
    SemanticRelationState,
    SemanticWorldModelState,
)


def _semantic_world_model() -> SemanticWorldModelState:
    return SemanticWorldModelState(
        world_model_id="wm_test",
        episode_id="episode_test",
        task_id="drawer_vase_task",
        objective_preset="balanced",
        semantic_tags=["drawer", "vase", "risk:fragility", "affordance:open"],
        objects=[
            SemanticObjectState(
                object_id="object_drawer",
                label="drawer",
                category="container",
                confidence=0.95,
                salience=0.8,
                affordances=["open", "close", "grasp_handle"],
                state_tags=["occluding"],
                risk_tags=[],
            ),
            SemanticObjectState(
                object_id="object_vase",
                label="vase",
                category="fragile_object",
                confidence=0.92,
                salience=0.9,
                affordances=["avoid_contact", "stabilize"],
                state_tags=["fragile"],
                risk_tags=["fragility"],
            ),
        ],
        relations=[
            SemanticRelationState(
                relation_id="rel_1",
                subject_id="object_drawer",
                relation_type="near",
                object_id="object_vase",
                confidence=0.7,
            )
        ],
        meta_nodes=[
            SemanticMetaNode(
                node_id="node_risk",
                node_type="risk_triage",
                priority="critical",
                score=0.82,
                rationale="fragile object present near interaction zone",
            ),
            SemanticMetaNode(
                node_id="node_efficiency",
                node_type="efficiency_router",
                priority="medium",
                score=0.41,
                rationale="energy routing remains relevant",
            ),
            SemanticMetaNode(
                node_id="node_refresh",
                node_type="semantic_memory_refresh",
                priority="high",
                score=0.55,
                rationale="refresh object state before promotion",
            ),
        ],
        capability_scores={
            "risk_reasoning": 0.78,
            "object_memory": 0.81,
            "affordance_grounding": 0.67,
            "fusion_bridge": 0.63,
            "stage2_bridge": 0.46,
            "meta_node_orchestration": 0.74,
        },
        topology={
            "grounded_track_object_count": 2,
            "object_count": 2,
            "relation_count": 1,
            "meta_node_count": 3,
        },
        metadata={"grounded_scene": {"grounding_mode": "scene_tracks"}},
    )


def _econ_signals() -> EconSignals:
    signals = EconSignals(
        mpl_urgency=0.25,
        error_urgency=0.62,
        energy_urgency=0.18,
        customer_segment="premium_safety",
        task_family="drawer_vase",
    )
    return signals


def _datapack_signals() -> DatapackSignals:
    return DatapackSignals(
        data_coverage_score=0.42,
        embedding_diversity=0.55,
        vla_annotation_fraction=0.8,
        guidance_annotation_fraction=0.6,
        semantic_tag_diversity=7,
        data_gaps=["frontier_cases"],
    )


def _coverage_feedback_metadata() -> dict:
    return {
        "coverage_feedback_summary": {
            "gap_return_mean": 0.72,
            "process_reward_mean": 0.58,
            "graph_mutation_pressure": 2.0,
            "wm_validation_error_rate": 0.31,
        },
        "wm_validation_summary": {
            "error_rate": 0.31,
            "packet_count": 2,
            "top_targets": ["object_vase", "skill:novel_recovery"],
        },
        "trust_calibration_overlay": {"mean_signal": 0.32},
        "econ_calibration_overlay": {"mean_signal": 0.71},
        "graph_mutation_proposals": [
            {"action": "add_provisional_skill", "target_ref": "skill:novel_recovery"}
        ],
        "semantic_coverage": {
            "coverage_summary": {"total_edges": 10, "missing_edges": 4, "governance_blocked_edges": 1},
            "feedback_summary": {
                "gap_return_mean": 0.72,
                "process_reward_mean": 0.58,
                "graph_mutation_pressure": 2.0,
                "wm_validation_error_rate": 0.31,
            },
        },
    }


def test_meta_transformer_propose_plan_emits_bounded_semantic_execution() -> None:
    transformer = MetaTransformer(d_shared=24)
    output = transformer.propose_plan(
        econ_signals=_econ_signals(),
        datapack_signals=_datapack_signals(),
        semantic_world_model=_semantic_world_model(),
    )

    assert output.objective_preset == "safety"
    assert output.execution_mode == "bounded_execution"
    assert output.execution_preconditions["ready"] is True
    assert output.execution_work_order is not None
    assert output.execution_work_order["ready"] is True
    assert "object:vase" in output.ontology_tokens
    assert any(step["action"] == "route_risk_triage" for step in output.orchestration_plan)


def test_pipeline_manager_meta_transformer_callout_is_live() -> None:
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

    result = run_pipeline_step_with_causal_order(
        econ_controller=DummyEconController(),
        datapack_engine=DummyDatapackEngine(),
        semantic_orchestrator=DummySemanticOrchestrator(),
        meta_transformer=MetaTransformer(d_shared=24),
        semantic_world_model=_semantic_world_model(),
    )

    assert result["meta_transformer_suggestions"]["objective_preset"] == "safety"
    assert result["meta_transformer_execution"]["execution_mode"] == "bounded_execution"
    assert result["meta_transformer_execution"]["execution_work_order"]["ready"] is True


def test_orchestration_transformer_uses_semantic_world_model_for_bounded_plan() -> None:
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
        semantic_metadata={"data_gaps": ["frontier_cases"], **_coverage_feedback_metadata()},
    )
    encoded = _encode_ctx(ctx)
    assert encoded.shape[0] == ORCHESTRATION_CTX_DIM
    assert float(encoded[-8:].sum()) > 0.0

    result = propose_orchestrated_plan(
        model=OrchestrationTransformer(hidden=32),
        ctx=ctx,
        instruction="prioritize safety and fragile-object routing",
        steps=4,
    )

    assert result.objective_preset == "safety"
    assert result.execution_mode == "bounded_execution"
    assert result.activation_work_order is not None
    assert result.activation_work_order["ready"] is True
    assert "SET_OBJECTIVE_PRESET" in [step.tool_call.name for step in result.steps]
    assert result.metadata["semantic_world_model_summary"]["world_model_id"] == "wm_test"
    assert "request_wm_state_validation" in result.activation_plan["bounded_actions"]
    assert "queue_graph_mutation_review" in result.activation_plan["bounded_actions"]


def test_meta_transformer_uses_coverage_feedback_metadata() -> None:
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
        profile_summaries={},
        semantic_world_model=_semantic_world_model(),
        semantic_metadata=_coverage_feedback_metadata(),
    )
    output = MetaTransformer(d_shared=24).propose_plan(
        econ_signals=_econ_signals(),
        datapack_signals=_datapack_signals(),
        orchestrator_context=ctx,
    )
    assert "request_wm_state_validation" in output.bounded_actions
    assert "queue_graph_mutation_review" in output.bounded_actions
    assert output.metadata["coverage_feedback_summary"]["graph_mutation_pressure"] == 2.0
