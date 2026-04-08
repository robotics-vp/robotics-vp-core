from datetime import datetime

from src.orchestrator.pipeline_manager import create_default_pipeline_manager
from src.orchestrator.semantic_orchestrator_v2 import SemanticOrchestratorV2
from src.orchestrator.shell_activation import (
    evaluate_shell_activation_backlog,
    get_shell_activation_assessment,
    normalize_execution_summary,
)
from src.semantic.models import EconSlice, MetaTransformerSlice, SemanticSnapshot
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType
from src.sima2.tags.semantic_tags import SemanticEnrichmentProposal, SupervisionHints
from src.sima2.task_graph_proposals import RefinementType, TaskGraphRefinementProposal


def _ready_execution_summary():
    return {
        "report_count": 1,
        "ready_count": 1,
        "blocked_count": 0,
        "mean_readiness_score": 1.0,
        "blocking_preconditions": {},
        "satisfied_preconditions": {
            "artifact::runtime_packet_ref": 1,
        },
    }


def _build_snapshot():
    now = datetime.utcnow()
    hints = SupervisionHints(
        prioritize_for_training=True,
        priority_level="high",
        suggested_weight_multiplier=1.0,
        suggested_replay_frequency="standard",
        requires_human_review=False,
        safety_critical=True,
        curriculum_stage="mid",
    )
    semantic_tag = SemanticEnrichmentProposal(
        proposal_id="tag_orch",
        timestamp=now.timestamp(),
        video_id="vid1",
        episode_id="ep_orch",
        task="task_orch",
        fragility_tags=[],
        risk_tags=[],
        affordance_tags=[],
        efficiency_tags=[],
        novelty_tags=[],
        intervention_tags=[],
        semantic_conflicts=[],
        coherence_score=0.8,
        supervision_hints=hints,
        confidence=0.9,
        source_proposals=[],
        validation_status="passed",
    )
    return SemanticSnapshot(
        task_id="task_orch",
        ontology_proposals=[
            OntologyUpdateProposal(
                proposal_id="p1",
                proposal_type=ProposalType.ADD_AFFORDANCE,
            )
        ],
        task_refinements=[
            TaskGraphRefinementProposal(
                proposal_id="r1",
                refinement_type=RefinementType.SPLIT_TASK,
            )
        ],
        semantic_tags=[semantic_tag],
        econ_slice=EconSlice(
            task_id="task_orch",
            avg_mpl_units_per_hour=80,
            avg_wage_parity=0.9,
            avg_energy_cost=0.4,
            avg_error_rate=0.05,
            frontier_episodes=["ep_orch"],
        ),
        meta_slice=MetaTransformerSlice(
            task_id="task_orch",
            presets=["balanced"],
            expected_deltas={"mpl": 0.1},
            backends=["pybullet"],
        ),
        timestamp=now.timestamp(),
        metadata={
            "execution_precondition_summary": _ready_execution_summary(),
        },
    )


def test_normalize_execution_summary_handles_single_report_shape():
    normalized = normalize_execution_summary(
        {
            "ready": True,
            "readiness_score": 1.0,
            "blocking_preconditions": [],
            "satisfied_preconditions": ["artifact::runtime_packet_ref"],
        }
    )

    assert normalized["report_count"] == 1
    assert normalized["ready_count"] == 1
    assert normalized["blocked_count"] == 0
    assert normalized["satisfied_preconditions"]["artifact::runtime_packet_ref"] == 1


def test_shell_activation_backlog_activates_current_shells_only():
    payload = evaluate_shell_activation_backlog(
        _ready_execution_summary(),
        module_keys=["semantic_orchestrator_v2", "pipeline_manager"],
        subject_prefix="test",
    )

    semantic_current = get_shell_activation_assessment(
        payload,
        "semantic_orchestrator_preconditioned_routing",
    )
    semantic_future = get_shell_activation_assessment(
        payload,
        "semantic_orchestrator_closed_loop_training",
    )
    pipeline_current = get_shell_activation_assessment(
        payload,
        "pipeline_manager_preconditioned_iteration",
    )

    assert semantic_current is not None
    assert semantic_current["state"] == "activated"
    assert pipeline_current is not None
    assert pipeline_current["state"] == "activated"
    assert semantic_future is not None
    assert semantic_future["state"] == "future_pending"


def test_semantic_orchestrator_emits_preconditioned_routing_plan():
    orchestrator = SemanticOrchestratorV2({"write_to_file": False})
    advisory = orchestrator.propose(_build_snapshot())

    assert advisory.execution_mode == "preconditioned_routing"
    assert advisory.activation_plan["mode"] == "preconditioned_routing"
    assert advisory.activation_work_order is not None
    assert advisory.activation_work_order["ready"] is True
    assert advisory.metadata["shell_activation"]["activated"][0]["activation_id"] == (
        "semantic_orchestrator_preconditioned_routing"
    )


def test_pipeline_manager_preview_emits_stage_activation_plan():
    manager = create_default_pipeline_manager()
    manager.config["execution_precondition_summary"] = _ready_execution_summary()
    manager.config["input_receipt_context"] = {
        "work_orders": [{"receipt_kind": "inferential_execution_work_order_v1"}],
        "canonical_metadata_receipts": [{"receipt_kind": "orchestrator_control_plane_context_v1"}],
    }

    preview = manager.preview_next_iteration()

    assert preview["receipt_kind"] == "pipeline_stage_activation_receipt_v1"
    assert preview["authority_class"] == "remain_advisory"
    assert preview["execution_mode"] == "preconditioned_iteration"
    assert preview["input_receipt_context"]["consumed_receipt_kinds"] == [
        "inferential_execution_work_order_v1",
        "orchestrator_control_plane_context_v1",
    ]
    assert preview["activation_work_order"] is not None
    assert preview["activation_work_order"]["ready"] is True
    assert len(preview["stage_activation_plan"]["stages"]) == 5
