"""Meta-regal nodes for Stage-6 deterministic audit gates.

Regal nodes are semantic evaluators that check constraints beyond
simple numeric thresholds. They are deterministic and produce
hashable reports for provenance.

Includes D4 knob calibration for learned hyperparameters.
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
from src.regal.knob_model import (
    KnobModel,
    HeuristicKnobProvider,
    StubLearnedKnobModel,
    get_knob_model,
)

__all__ = [
    # Regal evaluators
    "RegalNode",
    "REGAL_REGISTRY",
    "evaluate_regals",
    "register_regal",
    "SpecGuardianRegal",
    "WorldCoherenceRegal",
    "RewardIntegrityRegal",
    # Knob calibration (D4)
    "KnobModel",
    "HeuristicKnobProvider",
    "StubLearnedKnobModel",
    "get_knob_model",
]
