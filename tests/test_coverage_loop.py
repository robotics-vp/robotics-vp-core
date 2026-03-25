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

    def test_feedback_signals_flow_into_summary(self):
        result = run_coverage_loop(
            _mock_rows(),
            coverage_outcomes=[
                {
                    "edge_key": "task:open_drawer -> skill:grasp_handle",
                    "fill_method": "diffusion",
                    "coverage_delta": 0.2,
                    "process_reward_delta": 0.15,
                    "policy_eval_delta": 0.1,
                    "quality_score": 0.8,
                    "backend_health_score": 0.7,
                }
            ],
            wm_validation_packets=[
                {
                    "target_ref": "skill:grasp_handle",
                    "validation_kind": "relation_state",
                    "error_score": 0.4,
                }
            ],
            process_reward_summaries=[{"phi_star": 0.8, "confidence": 0.9}],
        )
        assert result.feedback_summary["coverage_outcome_count"] == 1
        assert result.coverage_summary["feedback_loop"]["process_reward_mean"] > 0.0
        assert result.wm_validation_summary["packet_count"] == 1
        assert "trust_calibration_overlay" in result.to_dict()

    def test_governance_blocked_gap_yields_blocked_fill_decision(self):
        baseline = run_coverage_loop(_mock_rows())
        assert baseline.fill_decisions
        blocked_edge = baseline.fill_decisions[0]["edge_key"]

        result = run_coverage_loop(
            _mock_rows(),
            governance_traces=[{"edge_key": blocked_edge, "outcome": "veto"}],
        )
        assert any(item["fill_method"] == "blocked" for item in result.fill_decisions)
        assert result.coverage_summary["governance_blocked_edges"] >= 1

    def test_loop_emits_mutation_execution_and_wm_correction(self):
        result = run_coverage_loop(
            _mock_rows(),
            semantic_world_model={
                "world_model_id": "wm",
                "episode_id": "ep",
                "task_id": "open_drawer",
                "objective_preset": "balanced",
                "semantic_tags": ["drawer"],
                "objects": [{"object_id": "object_drawer", "label": "drawer", "category": "container", "confidence": 0.9, "salience": 0.8}],
                "relations": [],
                "meta_nodes": [],
                "capability_scores": {"object_memory": 0.7},
                "topology": {"object_count": 1},
            },
            wm_validation_packets=[
                {"target_ref": "object_drawer", "validation_kind": "state_mismatch", "error_score": 0.7}
            ],
            stage2_ontology_proposals=[],
        )
        assert "metadata" in result.graph_mutation_execution
        assert result.semantic_wm_correction_overlay["meta_node_pressure"] > 0.0
        assert result.corrected_semantic_world_model is not None
        assert result.semantic_wm_refiner_summary.get("active") is True


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
