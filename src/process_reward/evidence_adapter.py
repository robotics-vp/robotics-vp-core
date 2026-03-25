"""Process-reward ↔ evidence bus adapter.

Bridges the currently disconnected process-reward value signals
(``phi_star``, confidence, hop score, perspective disagreement) into:

  1. An ``EvidenceRecord`` for the evidence bus.
  2. A ``PreconditionCheck`` for the execution-preconditions report.

Purely additive — no existing process-reward or evidence code is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.evidence import EvidenceRecord
from src.evidence.preconditions import PreconditionCheck


@dataclass(frozen=True)
class ProcessRewardSummary:
    """Lightweight snapshot of process-reward outputs for one episode."""

    phi_star: float
    confidence: float
    hop_score: float = 0.0
    perspective_disagreement: float = 0.0
    episode_id: str = ""
    instruction: str = ""
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # frozen dataclass needs object.__setattr__ for defaults
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


def evidence_record_from_process_reward(
    summary: ProcessRewardSummary,
    *,
    timestamp: str = "",
) -> EvidenceRecord:
    """Convert a ``ProcessRewardSummary`` into an ``EvidenceRecord``.

    The record is tagged as ``source="process_reward"`` and
    ``kind="process_reward_fusion"`` so downstream consumers can filter
    for process-reward-specific evidence without confusion.
    """
    return EvidenceRecord.from_components(
        episode_id=summary.episode_id,
        timestamp=timestamp,
        source="process_reward",
        kind="process_reward_fusion",
        confidence=summary.confidence,
        disagreement=summary.perspective_disagreement,
        metrics={
            "phi_star": summary.phi_star,
            "hop_score": summary.hop_score,
            "confidence": summary.confidence,
            "perspective_disagreement": summary.perspective_disagreement,
        },
    )


def precondition_check_from_process_reward(
    summary: ProcessRewardSummary,
    *,
    phi_star_threshold: float = 0.3,
    confidence_threshold: float = 0.4,
) -> PreconditionCheck:
    """Convert a ``ProcessRewardSummary`` into a ``PreconditionCheck``.

    The check passes when **both** ``phi_star ≥ phi_star_threshold`` and
    ``confidence ≥ confidence_threshold`` (i.e. the episode has a meaningful
    process-reward signal with sufficient confidence to act on).
    """
    passed = (
        summary.phi_star >= phi_star_threshold
        and summary.confidence >= confidence_threshold
    )
    return PreconditionCheck(
        precondition_id="process_reward_quality",
        satisfied=passed,
        detail="Process-reward phi_star and confidence meet thresholds",
        observed_value=summary.phi_star * summary.confidence,
        metadata={
            "phi_star": summary.phi_star,
            "confidence": summary.confidence,
            "hop_score": summary.hop_score,
            "phi_star_threshold": phi_star_threshold,
            "confidence_threshold": confidence_threshold,
        },
    )


__all__ = [
    "ProcessRewardSummary",
    "evidence_record_from_process_reward",
    "precondition_check_from_process_reward",
]
