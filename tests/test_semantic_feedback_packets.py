from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalPriority, ProposalType
from src.world_model.fill_outcome_store import FillOutcomeRecord
from src.world_model.semantic_feedback_packets import (
    CoverageOutcomePacket,
    WMValidationPacket,
    compile_semantic_coverage_feedback,
)


def test_compile_semantic_coverage_feedback_emits_overlays_and_mutations() -> None:
    feedback = compile_semantic_coverage_feedback(
        coverage_outcomes=[
            CoverageOutcomePacket(
                edge_key="task:open_drawer -> skill:grasp_handle",
                fill_method="diffusion",
                coverage_delta=0.2,
                process_reward_delta=0.1,
                policy_eval_delta=0.15,
                quality_score=0.7,
                cost_score=0.2,
                backend_health_score=0.8,
            )
        ],
        wm_validation_packets=[
            WMValidationPacket(
                target_ref="skill:novel_recovery",
                validation_kind="novel_skill",
                predicted_value="recover",
                observed_value="novel_recovery",
                error_score=0.8,
                severity="high",
                metadata={"novel_ref": "skill:novel_recovery"},
            )
        ],
        fill_outcome_records=[
            FillOutcomeRecord(
                edge_key="task:open_drawer -> skill:grasp_handle",
                fill_method="diffusion",
                gap_features={"economic_priority": 0.8, "trust_priority": 0.7, "readiness": 0.9},
                pre_evidence_count=0,
                post_evidence_count=1,
                coverage_delta=0.15,
                wall_time_s=2.0,
                quality_score=0.8,
            )
        ],
        process_reward_summaries=[{"phi_star": 0.7, "confidence": 0.9, "phi_star_delta": 0.3}],
        governance_traces=[{"edge_key": "task:blocked -> skill:danger", "outcome": "veto"}],
        stage2_ontology_proposals=[
            OntologyUpdateProposal(
                proposal_id="p1",
                proposal_type=ProposalType.ADD_AFFORDANCE,
                priority=ProposalPriority.HIGH,
                source_primitive_id="prim_1",
                target_object_id="drawer_handle",
                rationale="new affordance observed",
                confidence=0.75,
            )
        ],
        econ_signals={"urgency": 0.6, "w_econ": 0.7},
        trust_state={"calibration_score": 0.5},
    )

    assert feedback.feedback_summary["coverage_outcome_count"] == 1
    assert feedback.feedback_summary["wm_validation_count"] == 1
    assert feedback.feedback_summary["governance_blocked_count"] == 1
    assert feedback.trust_calibration_overlay["mean_signal"] > 0.0
    assert feedback.econ_calibration_overlay["mean_signal"] > 0.0
    assert ("task:blocked", "skill:danger") in feedback.edge_metadata
    assert feedback.edge_metadata[("task:blocked", "skill:danger")]["governance_blocked"] is True
    assert any(item.action == "add_provisional_affordance" for item in feedback.graph_mutation_proposals)
    assert any(item.target_ref == "skill:novel_recovery" for item in feedback.graph_mutation_proposals)
