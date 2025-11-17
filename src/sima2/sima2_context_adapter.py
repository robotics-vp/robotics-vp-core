"""
Adapter to convert SIMA-2 rollouts into econ/semantic summaries.
"""
from typing import Dict, Any
from src.orchestrator.training_dataset import EconSemanticDecisionSummary


def sima2_rollout_to_decision_summary(rollout: Dict[str, Any]) -> EconSemanticDecisionSummary:
    """
    Convert SIMA-2 rollout into EconSemanticDecisionSummary placeholders.
    """
    return EconSemanticDecisionSummary(
        chosen_profile="BASE",
        objective_preset="balanced",
        pareto_classification="balanced",
        urgency_level="none",
        recommended_focus="balanced",
        semantic_priority_fraction=0.0,
        data_coverage_score=0.0,
        wage_parity=1.0,
    )
