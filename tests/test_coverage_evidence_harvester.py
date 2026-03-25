"""Tests for coverage evidence harvester (Section F)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.world_model.coverage_evidence_harvester import (
    EvidenceHarvestResult,
    harvest_evidence_counts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_runtime_rows():
    """Create mock runtime learning rows."""
    return [
        {
            "task_id": "open_drawer",
            "env_id": "drawer_vase",
            "semantic_tokens": ["skill:locate_drawer", "skill:grasp_handle"],
            "skill_mode": "efficiency_throughput",
            "econ_tensor_summary": {"net_value": 0.8},
        },
        {
            "task_id": "open_drawer",
            "env_id": "drawer_vase",
            "semantic_tokens": ["skill:grasp_handle", "skill:open_with_clearance"],
            "skill_mode": "safety_constrained",
        },
        {
            "task_id": "wash_dish",
            "env_id": "dishwashing",
            "semantic_tokens": [],
            "skill_mode": "unknown_mode",
        },
        {
            "task_id": "pick_part",
            "env_id": "workcell",
            "skill_mode": "hrl_full",
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHarvestEvidenceCounts:
    def test_basic_harvest_produces_edges(self):
        rows = _mock_runtime_rows()
        result = harvest_evidence_counts(rows)
        assert isinstance(result, EvidenceHarvestResult)
        assert result.rows_processed == 4
        assert result.edges_discovered > 0
        assert len(result.evidence_counts) > 0

    def test_task_to_env_edges(self):
        rows = _mock_runtime_rows()
        result = harvest_evidence_counts(rows)
        key = ("task:open_drawer", "env:drawer_vase")
        assert key in result.evidence_counts
        assert result.evidence_counts[key] >= 2  # two rows with same task/env

    def test_task_to_skill_edges(self):
        rows = [{"task_id": "open_drawer", "env_id": "drawer_vase", "skill_mode": "efficiency_throughput"}]
        result = harvest_evidence_counts(rows)
        # efficiency_throughput maps to known skills
        found_skill_edges = [k for k in result.evidence_counts if k[0] == "task:open_drawer" and k[1].startswith("skill:")]
        assert len(found_skill_edges) > 0

    def test_task_to_risk_edges(self):
        rows = [{"task_id": "open_drawer", "env_id": "drawer_vase"}]
        result = harvest_evidence_counts(rows)
        risk_edges = [k for k in result.evidence_counts if k[1].startswith("risk:")]
        assert len(risk_edges) > 0  # drawer_vase has collision and fragile_contact risks

    def test_econ_signals_affect_priorities(self):
        rows = [{"task_id": "t1", "env_id": "drawer_vase"}]
        low_econ = harvest_evidence_counts(rows, econ_signals={"urgency": 0.1, "w_econ": 0.1})
        high_econ = harvest_evidence_counts(rows, econ_signals={"urgency": 0.9, "w_econ": 0.9})
        
        key = ("task:t1", "env:drawer_vase")
        assert high_econ.economic_priorities[key] >= low_econ.economic_priorities[key]

    def test_trust_state_affects_priorities(self):
        rows = [{"task_id": "t1", "env_id": "drawer_vase"}]
        low_trust = harvest_evidence_counts(rows, trust_state={"calibration_score": 0.1})
        high_trust = harvest_evidence_counts(rows, trust_state={"calibration_score": 0.9})
        
        key = ("task:t1", "env:drawer_vase")
        assert high_trust.trust_priorities[key] >= low_trust.trust_priorities[key]

    def test_governance_traces_boost_promotion(self):
        rows = [{"task_id": "t1", "env_id": "drawer_vase"}]
        no_gov = harvest_evidence_counts(rows)
        with_gov = harvest_evidence_counts(rows, governance_traces=[{"node_id": "task:t1"}])

        key = ("task:t1", "env:drawer_vase")
        assert with_gov.promotion_readiness.get(key, 0) >= no_gov.promotion_readiness.get(key, 0)

    def test_empty_rows(self):
        result = harvest_evidence_counts([])
        assert result.rows_processed == 0
        assert result.edges_discovered == 0
        assert result.evidence_counts == {}

    def test_rows_without_task_id_skipped(self):
        rows = [{"env_id": "drawer_vase"}, {"task_id": "", "env_id": "drawer_vase"}]
        result = harvest_evidence_counts(rows)
        assert result.rows_processed == 2
        # empty task_id rows should produce no edges
        assert all(k[0] != "task:" for k in result.evidence_counts)

    def test_serialization_round_trip(self):
        rows = _mock_runtime_rows()
        result = harvest_evidence_counts(rows)
        d = result.to_dict()
        restored = EvidenceHarvestResult.from_dict(d)
        assert restored.rows_processed == result.rows_processed
        assert restored.edges_discovered == result.edges_discovered
        assert len(restored.evidence_counts) == len(result.evidence_counts)
