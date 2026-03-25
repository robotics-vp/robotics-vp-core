"""Tests for coverage loop orchestrator (Section I)."""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.orchestrator.coverage_loop import (
    CoverageLoopResult,
    FillPathDecision,
    run_coverage_loop,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_rows():
    return [
        {
            "task_id": "open_drawer",
            "env_id": "drawer_vase",
            "semantic_tokens": ["skill:locate_drawer", "skill:grasp_handle"],
            "skill_mode": "efficiency_throughput",
        },
        {
            "task_id": "open_drawer",
            "env_id": "drawer_vase",
            "semantic_tokens": ["skill:grasp_handle"],
            "skill_mode": "safety_constrained",
        },
        {
            "task_id": "wash_dish",
            "env_id": "dishwashing",
            "skill_mode": "hrl_full",
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCoverageLoop:
    def test_basic_loop_runs(self):
        result = run_coverage_loop(_mock_rows())
        assert isinstance(result, CoverageLoopResult)
        assert result.evidence_harvest.rows_processed == 3
        assert result.evidence_harvest.edges_discovered > 0

    def test_coverage_summary_populated(self):
        result = run_coverage_loop(_mock_rows())
        summary = result.coverage_summary
        assert "total_edges" in summary
        assert "covered_edges" in summary
        assert "missing_edges" in summary
        assert "coverage_ratio" in summary

    def test_simulation_agenda_produced(self):
        result = run_coverage_loop(_mock_rows(), sim_agenda_limit=5)
        # Agenda should be a list (may be empty if graph is fully covered)
        assert isinstance(result.simulation_agenda, list)

    def test_diffusion_prompts_produced(self):
        result = run_coverage_loop(_mock_rows(), diffusion_limit=5)
        assert isinstance(result.diffusion_prompts, list)

    def test_fill_decisions_produced(self):
        result = run_coverage_loop(_mock_rows())
        assert isinstance(result.fill_decisions, list)
        for d in result.fill_decisions:
            assert "fill_method" in d
            assert d["fill_method"] in ("real_sim", "diffusion", "synthetic_branch", "blocked")

    def test_write_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_coverage_loop(
                _mock_rows(),
                write_artifacts=True,
                artifact_dir=tmpdir,
            )
            paths = result.write_artifacts(tmpdir)
            for name in ("coverage_graph", "coverage_summary", "simulation_agenda",
                         "diffusion_prompts", "fill_decisions", "evidence_harvest"):
                assert name in paths
                assert os.path.exists(paths[name])
                with open(paths[name]) as f:
                    data = json.load(f)
                assert data is not None

    def test_to_dict_round_trip(self):
        result = run_coverage_loop(_mock_rows())
        d = result.to_dict()
        assert "coverage_graph" in d
        assert "coverage_summary" in d
        assert "evidence_harvest" in d
        assert "simulation_agenda" in d
        assert "fill_decisions" in d

    def test_empty_rows(self):
        result = run_coverage_loop([])
        assert result.evidence_harvest.rows_processed == 0
        # Coverage graph still builds from skill graph/env inventories
        assert result.coverage_summary is not None

    def test_econ_signals_flow_through(self):
        result = run_coverage_loop(
            _mock_rows(),
            econ_signals={"urgency": 0.9, "w_econ": 0.8},
            trust_state={"calibration_score": 0.7},
        )
        harvest = result.evidence_harvest
        # High-urgency signals should produce higher economic priorities
        assert any(v > 0 for v in harvest.economic_priorities.values())

    def test_env_names_filter(self):
        result = run_coverage_loop(
            _mock_rows(),
            env_names=["drawer_vase"],
        )
        assert "drawer_vase" in result.metadata.get("env_names", [])


class TestFillPathDecision:
    def test_fill_path_serialization(self):
        d = FillPathDecision(
            edge_key="a -> b",
            fill_method="diffusion",
            confidence=0.8,
            rationale="test",
            coverage_gap_score=0.5,
            economic_priority=0.7,
            trust_priority=0.6,
            readiness=0.9,
        )
        result = d.to_dict()
        assert result["fill_method"] == "diffusion"
        assert result["confidence"] == 0.8
