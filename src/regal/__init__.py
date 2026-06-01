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
    get_knob_model,
)
from src.regal.base import RegalDecision, RegalReport, RegalNode as MetaRegalNode
from src.regal.objective_integrity import RegalObjectiveIntegrityNode
from src.regal.reward_safety import RegalRewardSafetyNode
from src.regal.econ_consistency import RegalEconConsistencyNode
from src.regal.gen_plausibility import RegalGenPlausibilityNode
from src.regal.data_value import RegalDataValueNode
from src.regal.bio_neuro_anomaly import (
    AnomalySuspicionReceipt,
    GovernanceEscalationEvent,
    build_anomaly_suspicion_receipt,
    build_governance_escalation_event,
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
    "get_knob_model",
    # Lightweight additive regal nodes
    "RegalDecision",
    "RegalReport",
    "MetaRegalNode",
    "RegalObjectiveIntegrityNode",
    "RegalRewardSafetyNode",
    "RegalEconConsistencyNode",
    "RegalGenPlausibilityNode",
    "RegalDataValueNode",
    "AnomalySuspicionReceipt",
    "GovernanceEscalationEvent",
    "build_anomaly_suspicion_receipt",
    "build_governance_escalation_event",
]
