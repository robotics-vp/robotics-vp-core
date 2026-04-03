"""Semantic-input normalization for the sim/synth/physics world model."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..common import mapping


def build_semantic_input_context(
    *,
    coverage_graph: Any,
    semantic_context: Optional[Mapping[str, Any]] = None,
    perception_grounding_state: Any = None,
) -> Dict[str, Any]:
    nodes = list(getattr(coverage_graph, "nodes", []) or [])
    missing_edges = [edge for edge in list(getattr(coverage_graph, "edges", []) or []) if getattr(edge, "is_missing", False)]
    governance_blocked = [
        edge for edge in missing_edges if bool(getattr(edge, "metadata", {}).get("governance_blocked", False))
    ]
    task_nodes = sorted(
        {
            str(getattr(node, "label", ""))
            for node in nodes
            if str(getattr(node, "node_type", "")) == "task"
        }
    )
    context = {
        **mapping(semantic_context),
        "coverage_graph_metadata": mapping(getattr(coverage_graph, "metadata", {})),
        "missing_edge_count": len(missing_edges),
        "governance_blocked_edge_count": len(governance_blocked),
        "task_families": task_nodes,
    }
    if perception_grounding_state is None:
        return context

    scene_graph = getattr(perception_grounding_state, "scene_graph", None)
    evidence_routing = getattr(perception_grounding_state, "evidence_routing", None)
    semantic_bridge_registry = getattr(
        perception_grounding_state,
        "semantic_bridge_registry",
        None,
    )
    sim_bridge = (
        None
        if semantic_bridge_registry is None
        else getattr(semantic_bridge_registry, "sim_synth_bridge", None)
    )
    deployment_surface = getattr(
        perception_grounding_state,
        "deployment_resource_surface",
        None,
    )
    branch_relevance_scores = list(getattr(sim_bridge, "branch_relevance_scores", []) or [])
    preservation_scores = list(getattr(sim_bridge, "object_preservation_scores", []) or [])
    mean_branch_relevance = (
        sum(float(v) for v in branch_relevance_scores) / float(len(branch_relevance_scores))
        if branch_relevance_scores
        else 0.0
    )
    mean_preservation = (
        sum(float(v) for v in preservation_scores) / float(len(preservation_scores))
        if preservation_scores
        else 0.0
    )
    context.update(
        {
            "perception_grounding_state_id": str(
                getattr(perception_grounding_state, "state_id", "")
            ),
            "perception_maturity_stage": str(
                getattr(perception_grounding_state, "maturity_stage", "")
            ),
            "scene_object_count": int(getattr(scene_graph, "object_count", 0) or 0),
            "scene_edge_count": int(getattr(scene_graph, "edge_count", 0) or 0),
            "scene_graph_density": float(getattr(scene_graph, "graph_density", 0.0) or 0.0),
            "perception_fusion_confidence": float(
                getattr(evidence_routing, "fusion_confidence", 0.0) or 0.0
            ),
            "perception_fusion_method": str(
                getattr(evidence_routing, "fusion_method", "")
            ),
            "perception_provider_availability": mapping(
                getattr(evidence_routing, "provider_availability", {})
            ),
            "sim_synth_bridge_ready": bool(sim_bridge is not None),
            "sim_synth_bridge_helper_stage": str(
                getattr(sim_bridge, "helper_promotion_stage", "")
            ),
            "sim_synth_branch_relevance_mean": mean_branch_relevance,
            "sim_synth_object_preservation_mean": mean_preservation,
            "sim_synth_contact_topology_summary": mapping(
                getattr(sim_bridge, "contact_topology_summary", {})
            ),
            "deployment_posture": str(
                getattr(deployment_surface, "deployment_posture", "")
            ),
            "inferential_learnability_summary": {
                "mean_signal_yield_score": min(
                    1.0,
                    0.45 * float(getattr(evidence_routing, "fusion_confidence", 0.0) or 0.0)
                    + 0.35 * mean_branch_relevance
                    + 0.2 * mean_preservation,
                ),
                "mean_inferential_replay_weight": min(
                    1.0,
                    0.4 * mean_preservation
                    + 0.35
                    * min(float(getattr(scene_graph, "object_count", 0) or 0) / 6.0, 1.0)
                    + 0.25
                    * min(float(getattr(scene_graph, "edge_count", 0) or 0) / 8.0, 1.0),
                ),
            },
        }
    )
    return context
