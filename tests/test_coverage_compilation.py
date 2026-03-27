"""Tests for coverage-gap-driven diffusion prompt generation (C1)
and simulation agenda compilation (C2)."""

from src.world_model.semantic_coverage_graph import (
    CoverageEdge,
    CoverageNode,
    SemanticCoverageGraph,
)
from src.orchestrator.diffusion_requests import build_diffusion_prompt_from_coverage_gaps
from src.orchestrator.semantic_simulation import compile_simulation_agenda


def _make_test_graph() -> SemanticCoverageGraph:
    """Build a small coverage graph with known missing edges."""
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("hrl:grasp_handle", "skill", "Grasp Handle"),
            CoverageNode("prim:locate_handle", "env_primitive", "Locate Handle"),
            CoverageNode("risk:collision", "risk_family", "collision"),
            CoverageNode("obj:vase", "object_family", "vase"),
        ],
        edges=[
            CoverageEdge(
                "hrl:grasp_handle", "prim:locate_handle", "requires",
                evidence_count=0, economic_priority=0.8, trust_priority=0.5,
            ),
            CoverageEdge(
                "hrl:grasp_handle", "risk:collision", "requires",
                evidence_count=0, economic_priority=0.3, trust_priority=0.2,
            ),
            CoverageEdge(
                "hrl:grasp_handle", "obj:vase", "requires",
                evidence_count=5,  # covered
            ),
        ],
    )


# ── C1: Diffusion prompts ────────────────────────────────────────────────


def test_gap_driven_prompts_count():
    g = _make_test_graph()
    prompts = build_diffusion_prompt_from_coverage_gaps(g, limit=10)
    # Only 2 missing edges, so expect 2 prompts
    assert len(prompts) == 2


def test_gap_driven_prompts_ordered_by_priority():
    g = _make_test_graph()
    prompts = build_diffusion_prompt_from_coverage_gaps(g)
    # Higher economic priority first
    assert prompts[0].economic_priority_score >= prompts[1].economic_priority_score


def test_gap_driven_prompt_has_coverage_fields():
    g = _make_test_graph()
    prompts = build_diffusion_prompt_from_coverage_gaps(g, limit=1)
    p = prompts[0]
    assert p.difficulty_hint == "gap_driven"
    assert p.coverage_gap_score > 0
    assert isinstance(p.missing_skill_edges, list)
    assert isinstance(p.missing_env_primitives, list) or isinstance(p.risk_family_targets, list)
    assert "inferential_learnability_contract" in p.routing_context


def test_no_prompts_from_fully_covered_graph():
    g = SemanticCoverageGraph(
        edges=[CoverageEdge("a", "b", "requires", evidence_count=10)],
    )
    prompts = build_diffusion_prompt_from_coverage_gaps(g)
    assert len(prompts) == 0


# ── C2: Simulation agenda ────────────────────────────────────────────────


def test_simulation_agenda_count():
    g = _make_test_graph()
    agenda = compile_simulation_agenda(g, limit=10)
    assert len(agenda) == 2


def test_simulation_agenda_ranked():
    g = _make_test_graph()
    agenda = compile_simulation_agenda(g)
    assert agenda[0]["rank"] == 1
    assert agenda[1]["rank"] == 2
    assert agenda[0]["coverage_gap_score"] >= agenda[1]["coverage_gap_score"]


def test_simulation_agenda_item_fields():
    g = _make_test_graph()
    agenda = compile_simulation_agenda(g, limit=1)
    item = agenda[0]
    assert "skill_edge" in item
    assert "data_collection_intent" in item
    assert item["data_collection_intent"] in ("explore", "exploit", "validate")
    assert "rationale" in item


def test_empty_graph_produces_empty_agenda():
    g = SemanticCoverageGraph()
    agenda = compile_simulation_agenda(g)
    assert len(agenda) == 0
