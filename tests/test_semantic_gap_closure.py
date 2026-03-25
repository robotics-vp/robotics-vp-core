from src.envs.primitive_inventory import for_env
from src.hrl.skill_graph import SkillGraph
from src.world_model.feedback_topology_adapters import (
    build_feedback_topology_dataset,
    shadow_fit_feedback_adapter_package,
)
from src.world_model.graph_mutation_executor import GovernedGraphMutationExecutor
from src.world_model.semantic_feedback_packets import GraphMutationProposal, WMValidationPacket
from src.world_model.semantic_wm_correction import (
    apply_semantic_wm_correction_overlay,
    compile_semantic_wm_correction_overlay,
)
from src.world_model.semantic_world_model import (
    SemanticMetaNode,
    SemanticObjectState,
    SemanticRelationState,
    SemanticWorldModelState,
)
from src.world_model.semantic_coverage_graph import SemanticCoverageGraph


def _world_model() -> SemanticWorldModelState:
    return SemanticWorldModelState(
        world_model_id="wm_test",
        episode_id="ep",
        task_id="drawer_vase",
        objective_preset="balanced",
        semantic_tags=["drawer", "vase"],
        objects=[
            SemanticObjectState(
                object_id="object_drawer",
                label="drawer",
                category="container",
                confidence=0.9,
                salience=0.8,
                affordances=["open"],
            ),
            SemanticObjectState(
                object_id="object_vase",
                label="vase",
                category="fragile_object",
                confidence=0.92,
                salience=0.9,
                affordances=["avoid_contact"],
                risk_tags=["fragility"],
            ),
        ],
        relations=[
            SemanticRelationState(
                relation_id="rel_1",
                subject_id="object_drawer",
                relation_type="near",
                object_id="object_vase",
                confidence=0.75,
            )
        ],
        meta_nodes=[
            SemanticMetaNode(
                node_id="meta:risk",
                node_type="risk_triage",
                priority="high",
                score=0.7,
                rationale="risk",
            )
        ],
        capability_scores={"object_memory": 0.8, "affordance_grounding": 0.7, "risk_reasoning": 0.6},
        topology={"object_count": 2, "relation_count": 1},
    )


def test_semantic_wm_correction_overlay_applies_validation_pressure() -> None:
    overlay = compile_semantic_wm_correction_overlay(
        _world_model(),
        [
            WMValidationPacket(
                target_ref="object_vase",
                validation_kind="state_mismatch",
                error_score=0.8,
                severity="high",
            )
        ],
    )
    corrected = apply_semantic_wm_correction_overlay(_world_model(), overlay)
    assert corrected is not None
    assert corrected.metadata["semantic_wm_correction_overlay"]["meta_node_pressure"] > 0.0
    vase = next(item for item in corrected.objects if item.object_id == "object_vase")
    assert vase.confidence < 0.92
    assert any(item.node_type == "state_validation_router" for item in corrected.meta_nodes)


def test_graph_mutation_executor_applies_provisional_skill_and_primitive() -> None:
    executor = GovernedGraphMutationExecutor(min_confidence=0.5)
    result = executor.execute(
        SkillGraph.build_from_registry(hrl_skills=True),
        [for_env("workcell")],
        [
            GraphMutationProposal(
                proposal_id="p_skill",
                action="add_provisional_skill",
                target_ref="skill:novel_recovery",
                confidence=0.8,
                rationale="novel runtime recovery observed",
                source_refs=["hrl:retract_safe"],
                metadata={"task_id": "drawer_vase"},
            ),
            GraphMutationProposal(
                proposal_id="p_aff",
                action="add_provisional_affordance",
                target_ref="affordance:precision_place",
                confidence=0.8,
                rationale="new affordance",
            ),
        ],
    )
    assert any(node.skill_family == "runtime" for node in result.skill_graph.nodes)
    assert any("precision_place" == prim.primitive_id for prim in result.env_inventories[0].primitives)
    assert result.metadata["applied_count"] == 2


def test_shadow_fit_feedback_adapter_package_predicts_edge_overlays() -> None:
    graph = SemanticCoverageGraph.build(
        skill_graph=SkillGraph.build_from_registry(hrl_skills=True),
        env_inventories=[for_env("drawer_vase")],
    )
    for idx, edge in enumerate(graph.edges[:6]):
        edge.economic_priority = 0.2 + 0.1 * idx
        edge.trust_priority = 0.3 + 0.05 * idx
        edge.promotion_readiness = 0.4 + 0.05 * idx
        edge.metadata["quality_score"] = 0.5 + 0.05 * idx
        edge.metadata["process_reward_delta"] = 0.2
        edge.metadata["policy_eval_delta"] = 0.1
        edge.metadata["backend_health_score"] = 0.8
    dataset = build_feedback_topology_dataset(graph)
    assert len(dataset.features) >= 6
    package = shadow_fit_feedback_adapter_package(graph, min_samples=4)
    assert package is not None
    predictions = package.predict_edges(graph.edges[:2])
    assert len(predictions) == 2
    assert 0.0 <= predictions[0]["economic_priority"] <= 1.0
