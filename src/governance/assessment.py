"""Governance assessment and gate for precondition checks.

Rolls up ``GovernanceTraceEntry`` instances into:

  1. A ``GovernanceAssessment`` coverage/readiness summary.
  2. A ``GovernanceGate`` that returns a ``PreconditionCheck``.

Purely additive — ``trace.py`` is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from src.evidence.preconditions import PreconditionCheck
from src.governance.trace import GovernanceTraceEntry


@dataclass
class GovernanceAssessment:
    """Aggregate assessment over a set of governance trace entries."""

    traces_present: int = 0
    veto_count: int = 0
    approval_count: int = 0
    reroute_count: int = 0
    advisory_count: int = 0
    coverage_ratio: float = 0.0
    oldest_trace_ts: str = ""
    newest_trace_ts: str = ""
    unique_node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_traces(
        cls,
        traces: Sequence[GovernanceTraceEntry],
        *,
        expected_node_count: int = 1,
    ) -> "GovernanceAssessment":
        """Build an assessment from a collection of trace entries.

        Parameters
        ----------
        traces : sequence of GovernanceTraceEntry
        expected_node_count : int
            How many distinct governance nodes are expected.  Used to
            compute ``coverage_ratio``.
        """
        sorted_traces = sorted(traces, key=lambda t: t.timestamp)
        veto = sum(1 for t in traces if t.outcome == "veto")
        approval = sum(1 for t in traces if t.outcome == "approve")
        reroute = sum(1 for t in traces if t.outcome == "reroute")
        advisory = sum(1 for t in traces if t.outcome == "advisory")
        unique_nodes = sorted({t.node_id for t in traces})
        coverage = len(unique_nodes) / max(expected_node_count, 1)

        return cls(
            traces_present=len(traces),
            veto_count=veto,
            approval_count=approval,
            reroute_count=reroute,
            advisory_count=advisory,
            coverage_ratio=min(coverage, 1.0),
            oldest_trace_ts=sorted_traces[0].timestamp if sorted_traces else "",
            newest_trace_ts=sorted_traces[-1].timestamp if sorted_traces else "",
            unique_node_ids=unique_nodes,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traces_present": self.traces_present,
            "veto_count": self.veto_count,
            "approval_count": self.approval_count,
            "reroute_count": self.reroute_count,
            "advisory_count": self.advisory_count,
            "coverage_ratio": self.coverage_ratio,
            "oldest_trace_ts": self.oldest_trace_ts,
            "newest_trace_ts": self.newest_trace_ts,
            "unique_node_ids": list(self.unique_node_ids),
            "metadata": dict(self.metadata),
        }


class GovernanceGate:
    """Precondition gate based on governance trace coverage.

    Parameters
    ----------
    min_coverage_ratio : float
        Minimum fraction of expected governance nodes that must be
        represented in traces.
    max_veto_ratio : float
        Maximum fraction of traces that can be vetoes before failing.
    """

    def __init__(
        self,
        *,
        min_coverage_ratio: float = 0.5,
        max_veto_ratio: float = 0.3,
    ) -> None:
        self.min_coverage_ratio = min_coverage_ratio
        self.max_veto_ratio = max_veto_ratio

    def check(self, assessment: GovernanceAssessment) -> PreconditionCheck:
        """Return a ``PreconditionCheck`` for the given assessment."""
        total = assessment.traces_present or 1
        veto_ratio = assessment.veto_count / total
        coverage_ok = assessment.coverage_ratio >= self.min_coverage_ratio
        veto_ok = veto_ratio <= self.max_veto_ratio
        passed = coverage_ok and veto_ok and assessment.traces_present > 0

        return PreconditionCheck(
            precondition_id="governance_gate",
            satisfied=passed,
            detail="Governance trace coverage and veto ratio within bounds",
            observed_value=assessment.coverage_ratio * (1.0 - veto_ratio),
            metadata={
                "coverage_ratio": assessment.coverage_ratio,
                "min_coverage_ratio": self.min_coverage_ratio,
                "veto_ratio": veto_ratio,
                "max_veto_ratio": self.max_veto_ratio,
                "traces_present": assessment.traces_present,
                "coverage_ok": coverage_ok,
                "veto_ok": veto_ok,
            },
        )


__all__ = [
    "GovernanceAssessment",
    "GovernanceGate",
]
