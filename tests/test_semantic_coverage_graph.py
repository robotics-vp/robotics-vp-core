"""Tests for semantic coverage graph substrate (B1).

Validates the SemanticCoverageGraph: node/edge construction, gap ranking,
coverage summary, round-trip serialisation, and builder integration with
SkillGraph + EnvPrimitiveInventory.
"""

from src.world_model.semantic_coverage_graph import (
    CoverageEdge,
    CoverageNode,
    SemanticCoverageGraph,
)


def test_empty_graph_summary():
    g = SemanticCoverageGraph()
    s = g.coverage_summary()
    assert s["total_edges"] == 0
    assert s["coverage_ratio"] == 0.0
    assert s["node_count"] == 0


def test_missing_and_covered_edges():
    g = SemanticCoverageGraph(
        nodes=[
            CoverageNode("a", "skill", "A"),
            CoverageNode("b", "env_primitive", "B"),
            CoverageNode("c", "object_family", "C"),
        ],
        edges=[
            CoverageEdge("a", "b", "requires", evidence_count=0, economic_priority=0.9),
            CoverageEdge("a", "c", "requires", evidence_count=3, economic_priority=0.1),
        ],
    )
    assert len(g.missing_edges) == 1
    assert g.missing_edges[0].source_id == "a"
    assert len(g.covered_edges) == 1


def test_rank_gaps_descending_priority():
    g = SemanticCoverageGraph(
        nodes=[CoverageNode("x", "skill", "X")],
        edges=[
            CoverageEdge("x", "low", "requires", evidence_count=0, economic_priority=0.1),
            CoverageEdge("x", "high", "requires", evidence_count=0, economic_priority=0.9),
            CoverageEdge("x", "covered", "requires", evidence_count=5, economic_priority=1.0),
        ],
    )
    ranked = g.rank_gaps()
    # covered edge should not appear (gap_score == 0 for covered)
    assert len(ranked) == 2
    assert ranked[0].target_id == "high"
    assert ranked[1].target_id == "low"


def test_rank_gaps_with_limit():
    g = SemanticCoverageGraph(
        edges=[
            CoverageEdge("a", f"t{i}", "requires", evidence_count=0, economic_priority=float(i))
            for i in range(10)
        ],
    )
    assert len(g.rank_gaps(limit=3)) == 3


def test_serialisation_round_trip():
    g = SemanticCoverageGraph(
        nodes=[CoverageNode("s", "skill", "Skill", metadata={"key": 42})],
        edges=[CoverageEdge("s", "p", "requires", evidence_count=2, trust_priority=0.5)],
        metadata={"version": 1},
    )
    d = g.to_dict()
    g2 = SemanticCoverageGraph.from_dict(d)
    assert len(g2.nodes) == 1
    assert g2.nodes[0].metadata["key"] == 42
    assert g2.edges[0].trust_priority == 0.5
    assert g2.metadata["version"] == 1


def test_build_from_skill_graph_and_env_inventories():
    """Integration: build() with SkillGraph + EnvPrimitiveInventory."""
    from src.hrl.skill_graph import SkillGraph
    from src.envs.primitive_inventory import for_env

    sg = SkillGraph.build_from_registry(hrl_skills=True)
    inv = for_env("drawer_vase")
    g = SemanticCoverageGraph.build(
        skill_graph=sg,
        env_inventories=[inv],
    )
    assert len(g.nodes) > 0
    assert len(g.edges) > 0
    summary = g.coverage_summary()
    # All edges should be missing (no evidence_counts supplied)
    assert summary["missing_edges"] == summary["total_edges"]
    assert "skill" in summary["node_type_counts"]
    assert "env_primitive" in summary["node_type_counts"]


def test_build_uses_edge_keyed_priority_maps() -> None:
    from types import SimpleNamespace

    skill_graph = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                skill_id="skill:b",
                label="skill b",
                env_primitive_requirements=[],
                object_family_requirements=[],
                risk_families=[],
            )
        ],
        transitions=[
            SimpleNamespace(task_id="a", from_skill="skill:b"),
        ],
    )
    graph = SemanticCoverageGraph.build(
        skill_graph=skill_graph,
        env_inventories=None,
        evidence_counts={("task:a", "skill:b"): 0},
        economic_priorities={("task:a", "skill:b"): 0.9},
        trust_priorities={("task:a", "skill:b"): 0.7},
        readiness_signals={("task:a", "skill:b"): 0.4},
        edge_metadata={("task:a", "skill:b"): {"governance_blocked": True}},
    )
    edge = graph.edges[0]
    assert edge.economic_priority == 0.9
    assert edge.trust_priority == 0.7
    assert edge.promotion_readiness == 0.4
    assert edge.metadata["governance_blocked"] is True
