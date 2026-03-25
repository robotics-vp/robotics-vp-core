import json

from src.world_model.semantic_feedback_packets import GraphMutationProposal, WMValidationPacket
from src.world_model.semantic_wm_correction import SemanticWMCorrectionOverlay
from src.world_model.semantic_wm_refiner import (
    build_semantic_wm_refinement_dataset_from_artifact_dirs,
    build_semantic_wm_refinement_dataset_from_examples,
    generate_graph_mutation_candidates,
    merge_graph_mutation_proposals,
    merge_semantic_wm_correction_overlays,
    shadow_fit_semantic_wm_refiner_package,
)
from src.world_model.semantic_world_model import (
    SemanticMetaNode,
    SemanticObjectState,
    SemanticRelationState,
    SemanticWorldModelState,
)


def _world_model() -> SemanticWorldModelState:
    return SemanticWorldModelState(
        world_model_id="wm_refiner_test",
        episode_id="episode_test",
        task_id="drawer_vase",
        objective_preset="balanced",
        semantic_tags=["drawer", "vase", "risk:fragility"],
        objects=[
            SemanticObjectState(
                object_id="object_drawer",
                label="drawer",
                category="container",
                confidence=0.9,
                salience=0.8,
                affordances=["open", "grasp_handle"],
                track_refs=["track_1"],
            ),
            SemanticObjectState(
                object_id="object_vase",
                label="vase",
                category="fragile_object",
                confidence=0.92,
                salience=0.95,
                affordances=["avoid_contact"],
                risk_tags=["fragility"],
                track_refs=["track_2"],
            ),
        ],
        relations=[
            SemanticRelationState(
                relation_id="rel_drawer_vase",
                subject_id="object_drawer",
                relation_type="near",
                object_id="object_vase",
                confidence=0.7,
            )
        ],
        meta_nodes=[
            SemanticMetaNode(
                node_id="meta_ontology",
                node_type="ontology_router",
                priority="high",
                score=0.72,
                rationale="novel semantics detected",
                target_refs=["skill:novel_recovery"],
            )
        ],
        capability_scores={
            "object_memory": 0.8,
            "affordance_grounding": 0.72,
            "meta_node_orchestration": 0.6,
            "risk_reasoning": 0.68,
        },
        topology={"grounded_track_object_count": 2},
    )


def test_generate_graph_mutation_candidates_uses_packets_and_meta_nodes() -> None:
    candidates = generate_graph_mutation_candidates(
        _world_model(),
        base_proposals=[],
        wm_validation_packets=[
            WMValidationPacket(
                target_ref="skill:novel_insert",
                validation_kind="skill_gap",
                error_score=0.7,
            )
        ],
    )
    assert any(item.target_ref == "skill:novel_insert" for item in candidates)
    assert any(item.target_ref == "skill:novel_recovery" for item in candidates)


def test_build_refinement_dataset_from_examples_collects_all_modalities() -> None:
    dataset = build_semantic_wm_refinement_dataset_from_examples(
        [
            {
                "semantic_world_model": _world_model().to_dict(),
                "correction_overlay": SemanticWMCorrectionOverlay(
                    object_confidence_adjustments={"object_vase": -0.2},
                    relation_confidence_adjustments={"rel_drawer_vase": -0.1},
                    capability_adjustments={"object_memory": -0.1, "risk_reasoning": 0.05},
                    meta_node_pressure=0.6,
                    target_refs=["object_vase", "rel_drawer_vase"],
                    metadata={"packet_count": 1},
                ).to_dict(),
                "feedback_summary": {"gap_return_mean": 0.4, "process_reward_mean": 0.2},
                "wm_validation_packets": [
                    WMValidationPacket(
                        target_ref="object_vase",
                        validation_kind="state_mismatch",
                        error_score=0.8,
                    ).to_dict()
                ],
                "graph_mutation_proposals": [
                    GraphMutationProposal(
                        proposal_id="p1",
                        action="add_provisional_skill",
                        target_ref="skill:novel_recovery",
                        confidence=0.8,
                        rationale="novel recovery",
                    ).to_dict()
                ],
                "graph_mutation_execution": {
                    "records": [{"proposal_id": "p1", "status": "applied"}],
                },
            }
        ]
    )
    assert dataset.object_features
    assert dataset.relation_features
    assert dataset.capability_features
    assert dataset.proposal_features


def test_shadow_fit_refiner_predicts_overlay_and_scored_proposals() -> None:
    package = shadow_fit_semantic_wm_refiner_package(
        _world_model(),
        correction_overlay=SemanticWMCorrectionOverlay(
            object_confidence_adjustments={"object_vase": -0.2},
            relation_confidence_adjustments={"rel_drawer_vase": -0.1},
            capability_adjustments={"object_memory": -0.1},
            meta_node_pressure=0.5,
            target_refs=["object_vase", "rel_drawer_vase"],
            metadata={"packet_count": 1},
        ),
        feedback_summary={"gap_return_mean": 0.3, "process_reward_mean": 0.1},
        wm_validation_packets=[
            WMValidationPacket(
                target_ref="object_vase",
                validation_kind="state_mismatch",
                error_score=0.8,
            )
        ],
        graph_mutation_proposals=[
            GraphMutationProposal(
                proposal_id="p_skill",
                action="add_provisional_skill",
                target_ref="skill:novel_recovery",
                confidence=0.85,
                rationale="novel recovery",
            )
        ],
    )
    assert package is not None
    overlay = package.predict_correction_overlay(
        _world_model(),
        wm_validation_packets=[
            WMValidationPacket(
                target_ref="object_vase",
                validation_kind="state_mismatch",
                error_score=0.8,
            )
        ],
        feedback_summary={"gap_return_mean": 0.3},
    )
    assert overlay.metadata["source"] == "semantic_wm_refiner"
    proposals = package.score_graph_mutation_proposals(
        _world_model(),
        [
            GraphMutationProposal(
                proposal_id="p_skill",
                action="add_provisional_skill",
                target_ref="skill:novel_recovery",
                confidence=0.85,
                rationale="novel recovery",
            )
        ],
        wm_validation_packets=[],
        feedback_summary={"gap_return_mean": 0.3},
    )
    assert proposals


def test_merge_helpers_keep_governed_base_and_learned_additions() -> None:
    merged_overlay = merge_semantic_wm_correction_overlays(
        SemanticWMCorrectionOverlay(
            object_confidence_adjustments={"object_vase": -0.2},
            meta_node_pressure=0.4,
            metadata={"source": "base"},
        ),
        SemanticWMCorrectionOverlay(
            object_confidence_adjustments={"object_vase": -0.1, "object_drawer": -0.05},
            meta_node_pressure=0.6,
            metadata={"source": "semantic_wm_refiner"},
        ),
    )
    assert merged_overlay.meta_node_pressure == 0.6
    assert "object_drawer" in merged_overlay.object_confidence_adjustments

    merged_proposals = merge_graph_mutation_proposals(
        [
            GraphMutationProposal(
                proposal_id="base",
                action="add_provisional_skill",
                target_ref="skill:novel_recovery",
                confidence=0.6,
                rationale="base",
            )
        ],
        [
            GraphMutationProposal(
                proposal_id="learned",
                action="add_provisional_skill",
                target_ref="skill:novel_recovery",
                confidence=0.9,
                rationale="learned",
                metadata={"source": "semantic_wm_refiner"},
            )
        ],
    )
    assert len(merged_proposals) == 1
    assert merged_proposals[0].confidence == 0.9


def test_build_refinement_dataset_from_artifact_dirs(tmp_path) -> None:
    artifact_dir = tmp_path / "coverage_artifact"
    artifact_dir.mkdir()
    (artifact_dir / "input_semantic_world_model.json").write_text(
        json.dumps(_world_model().to_dict()),
        encoding="utf-8",
    )
    (artifact_dir / "semantic_wm_correction_overlay.json").write_text(
        json.dumps(
            SemanticWMCorrectionOverlay(
                object_confidence_adjustments={"object_vase": -0.15},
                meta_node_pressure=0.5,
                metadata={"packet_count": 1},
            ).to_dict()
        ),
        encoding="utf-8",
    )
    (artifact_dir / "coverage_feedback_summary.json").write_text(
        json.dumps({"gap_return_mean": 0.5}),
        encoding="utf-8",
    )
    (artifact_dir / "graph_mutation_proposals.json").write_text(
        json.dumps(
            [
                GraphMutationProposal(
                    proposal_id="p1",
                    action="add_provisional_skill",
                    target_ref="skill:novel_recovery",
                    confidence=0.75,
                    rationale="novel recovery",
                ).to_dict()
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "graph_mutation_execution.json").write_text(
        json.dumps({"records": [{"proposal_id": "p1", "status": "applied"}]}),
        encoding="utf-8",
    )
    dataset = build_semantic_wm_refinement_dataset_from_artifact_dirs([str(artifact_dir)])
    assert dataset.object_features
