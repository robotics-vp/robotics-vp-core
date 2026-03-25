"""Tests for governance assessment and gate (A3)."""
import pytest

from src.governance.assessment import GovernanceAssessment, GovernanceGate
from src.governance.trace import GovernanceTraceEntry


def _make_trace(outcome: str, node_id: str = "node_a", ts: str = "2026-01-01T00:00:00Z") -> GovernanceTraceEntry:
    return GovernanceTraceEntry.from_components(
        run_id="run_1",
        episode_id="ep_1",
        timestamp=ts,
        node_id=node_id,
        outcome=outcome,
        reason_codes=["test"],
    )


def test_assessment_from_traces():
    traces = [
        _make_trace("approve", "n1", "2026-01-01T00:00:00Z"),
        _make_trace("approve", "n2", "2026-01-01T01:00:00Z"),
        _make_trace("veto", "n3", "2026-01-01T02:00:00Z"),
    ]
    a = GovernanceAssessment.from_traces(traces, expected_node_count=5)
    assert a.traces_present == 3
    assert a.approval_count == 2
    assert a.veto_count == 1
    assert a.coverage_ratio == pytest.approx(3 / 5)
    assert len(a.unique_node_ids) == 3
    assert a.oldest_trace_ts == "2026-01-01T00:00:00Z"
    assert a.newest_trace_ts == "2026-01-01T02:00:00Z"


def test_governance_gate_passes():
    traces = [
        _make_trace("approve", "n1"),
        _make_trace("approve", "n2"),
    ]
    a = GovernanceAssessment.from_traces(traces, expected_node_count=3)
    gate = GovernanceGate(min_coverage_ratio=0.5, max_veto_ratio=0.3)
    check = gate.check(a)
    assert check.satisfied
    assert check.precondition_id == "governance_gate"


def test_governance_gate_fails_on_low_coverage():
    traces = [_make_trace("approve", "n1")]
    a = GovernanceAssessment.from_traces(traces, expected_node_count=10)
    gate = GovernanceGate(min_coverage_ratio=0.5)
    check = gate.check(a)
    assert not check.satisfied


def test_governance_gate_fails_on_high_veto():
    traces = [
        _make_trace("veto", "n1"),
        _make_trace("veto", "n2"),
        _make_trace("approve", "n3"),
    ]
    a = GovernanceAssessment.from_traces(traces, expected_node_count=3)
    gate = GovernanceGate(min_coverage_ratio=0.5, max_veto_ratio=0.3)
    check = gate.check(a)
    assert not check.satisfied  # 2/3 veto ratio > 0.3


def test_empty_traces():
    a = GovernanceAssessment.from_traces([], expected_node_count=1)
    assert a.traces_present == 0
    gate = GovernanceGate()
    check = gate.check(a)
    assert not check.satisfied  # no traces present
