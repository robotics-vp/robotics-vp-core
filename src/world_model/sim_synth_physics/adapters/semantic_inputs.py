"""Semantic-input normalization for the sim/synth/physics world model."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..common import mapping


def build_semantic_input_context(
    *,
    coverage_graph: Any,
    semantic_context: Optional[Mapping[str, Any]] = None,
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
    return {
        **mapping(semantic_context),
        "coverage_graph_metadata": mapping(getattr(coverage_graph, "metadata", {})),
        "missing_edge_count": len(missing_edges),
        "governance_blocked_edge_count": len(governance_blocked),
        "task_families": task_nodes,
    }
