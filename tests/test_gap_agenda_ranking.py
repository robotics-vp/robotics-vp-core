from pytest import raises

from src.orchestrator.diffusion_requests import build_diffusion_prompt_from_coverage_gaps
from src.orchestrator.semantic_simulation import compile_simulation_agenda
from src.world_model.semantic_coverage_graph import (
    CoverageEdge,
    CoverageNode,
    SemanticCoverageGraph,
)


def _make_test_graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("hrl:grasp_handle", "skill", "Grasp Handle"),
            CoverageNode("prim:locate_handle", "env_primitive", "Locate Handle"),
            CoverageNode("risk:collision", "risk_family", "collision"),
        ],
        edges=[
            CoverageEdge(
                "hrl:grasp_handle",
                "prim:locate_handle",
                "requires",
                evidence_count=0,
                economic_priority=0.8,
                trust_priority=0.5,
                promotion_readiness=0.2,
            ),
            CoverageEdge(
                "hrl:grasp_handle",
                "risk:collision",
                "requires",
                evidence_count=0,
                economic_priority=0.3,
                trust_priority=0.2,
                promotion_readiness=0.9,
            ),
        ],
    )


class PromotedMockRanker:
    benchmark_gate = {"ready": True}

    def rank_edges(self, edges, graph):
        return [(edges[1], 1.0), (edges[0], 0.0)]


def test_compile_simulation_agenda_uses_promoted_gap_ranker() -> None:
    agenda = compile_simulation_agenda(
        _make_test_graph(),
        gap_ranker=PromotedMockRanker(),
        gap_ranker_mode="auto",
    )

    assert agenda[0]["skill_edge"] == "Grasp Handle -> collision"
    assert agenda[0]["ranking_policy"] == "heuristic_plus_learned_gap_ranker"
    assert agenda[0]["metadata"]["agenda_helper_status"]["promotion_stage"] == "promoted"
    assert agenda[0]["metadata"]["score_trace"]["learned_score"] == 1.0


def test_gap_driven_diffusion_prompts_preserve_helper_provenance() -> None:
    prompts = build_diffusion_prompt_from_coverage_gaps(
        _make_test_graph(),
        gap_ranker=PromotedMockRanker(),
        gap_ranker_mode="auto",
    )

    assert prompts[0].routing_context["agenda_ranking_policy"] == "heuristic_plus_learned_gap_ranker"
    assert prompts[0].routing_context["agenda_helper_status"]["promotion_stage"] == "promoted"
    assert prompts[0].governed_hypotheses[0]["metadata"]["agenda_score_trace"]["learned_score"] == 1.0


def test_compile_simulation_agenda_required_mode_demands_ready_ranker() -> None:
    with raises(ValueError, match="required"):
        compile_simulation_agenda(
            _make_test_graph(),
            gap_ranker=None,
            gap_ranker_mode="required",
        )
