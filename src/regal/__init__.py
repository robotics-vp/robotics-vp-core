"""Meta-regal nodes for Stage-6 deterministic audit gates.

Regal nodes are semantic evaluators that check constraints beyond
simple numeric thresholds. They are deterministic and produce
hashable reports for provenance.
"""
from src.regal.regal_evaluator import (
    RegalNode,
    REGAL_REGISTRY,
    evaluate_regals,
    register_regal,
    SpecGuardianRegal,
    WorldCoherenceRegal,
    RewardIntegrityRegal,
)

__all__ = [
    "RegalNode",
    "REGAL_REGISTRY",
    "evaluate_regals",
    "register_regal",
    "SpecGuardianRegal",
    "WorldCoherenceRegal",
    "RewardIntegrityRegal",
]
