"""Knob calibration model for D4 - learned hyperparameters with heuristic fallback.

The KnobModel maps RegimeFeatures → KnobPolicyV1 (bounded overrides to PlanPolicyConfig).
Learned outputs are always clamped by hard constraints.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from src.contracts.schemas import (
    RegimeFeaturesV1,
    KnobPolicyV1,
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
)


# =============================================================================
# Hard Constraints (non-negotiable invariants)
# =============================================================================

# Gain multiplier bounds
MIN_GAIN_MULTIPLIER = 0.5
MAX_GAIN_MULTIPLIER = 3.0
MIN_CONSERVATIVE_MULTIPLIER = 0.8
MAX_CONSERVATIVE_MULTIPLIER = 2.0

# Threshold bounds
MIN_SPEC_CONSISTENCY = 0.0
MAX_SPEC_CONSISTENCY = 1.0
MIN_COHERENCE = 0.0
MAX_COHERENCE = 1.0
MIN_HACK_PROB = 0.0
MAX_HACK_PROB = 1.0

# Patience bounds
MIN_PATIENCE = 1
MAX_PATIENCE = 10


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# =============================================================================
# KnobModel Interface
# =============================================================================

class KnobModel(ABC):
    """Interface for knob calibration models.

    Maps regime features to bounded PlanPolicyConfig overrides.
    Implementations can be learned or heuristic-based.
    """

    model_sha: Optional[str] = None  # SHA of the model weights/config

    @abstractmethod
    def predict(
        self,
        features: RegimeFeaturesV1,
        base_config: PlanPolicyConfigV1,
    ) -> KnobPolicyV1:
        """Predict knob overrides from regime features.

        Args:
            features: Current regime state
            base_config: Base policy config to override

        Returns:
            KnobPolicyV1 with bounded overrides
        """
        pass

    def apply_hard_constraints(
        self,
        policy: KnobPolicyV1,
    ) -> KnobPolicyV1:
        """Apply hard constraints to knob policy outputs.

        This is called after prediction to ensure all outputs are bounded.
        """
        clamp_reasons = []

        # Clamp gain multipliers
        if policy.gain_multiplier_override is not None:
            clamped = clamp(
                policy.gain_multiplier_override,
                MIN_GAIN_MULTIPLIER,
                MAX_GAIN_MULTIPLIER,
            )
            if clamped != policy.gain_multiplier_override:
                clamp_reasons.append(
                    f"gain_multiplier: {policy.gain_multiplier_override:.2f} → {clamped:.2f}"
                )
                policy.gain_multiplier_override = clamped

        if policy.conservative_multiplier_override is not None:
            clamped = clamp(
                policy.conservative_multiplier_override,
                MIN_CONSERVATIVE_MULTIPLIER,
                MAX_CONSERVATIVE_MULTIPLIER,
            )
            if clamped != policy.conservative_multiplier_override:
                clamp_reasons.append(
                    f"conservative_multiplier: {policy.conservative_multiplier_override:.2f} → {clamped:.2f}"
                )
                policy.conservative_multiplier_override = clamped

        # Clamp threshold overrides
        if policy.threshold_overrides:
            for key, value in policy.threshold_overrides.items():
                if "spec" in key.lower():
                    clamped = clamp(value, MIN_SPEC_CONSISTENCY, MAX_SPEC_CONSISTENCY)
                elif "coherence" in key.lower():
                    clamped = clamp(value, MIN_COHERENCE, MAX_COHERENCE)
                elif "hack" in key.lower():
                    clamped = clamp(value, MIN_HACK_PROB, MAX_HACK_PROB)
                else:
                    clamped = clamp(value, 0.0, 1.0)

                if clamped != value:
                    clamp_reasons.append(f"{key}: {value:.2f} → {clamped:.2f}")
                    policy.threshold_overrides[key] = clamped

        # Clamp patience
        if policy.patience_override is not None:
            clamped = int(clamp(policy.patience_override, MIN_PATIENCE, MAX_PATIENCE))
            if clamped != policy.patience_override:
                clamp_reasons.append(
                    f"patience: {policy.patience_override} → {clamped}"
                )
                policy.patience_override = clamped

        policy.clamped = len(clamp_reasons) > 0
        policy.clamp_reasons = clamp_reasons
        return policy


# =============================================================================
# Heuristic Fallback Provider
# =============================================================================

class HeuristicKnobProvider(KnobModel):
    """Heuristic-based knob provider (fallback when learned model unavailable).

    Uses simple rule-based logic to adjust knobs based on regime features.
    """

    model_sha = None  # No learned weights

    def predict(
        self,
        features: RegimeFeaturesV1,
        base_config: PlanPolicyConfigV1,
    ) -> KnobPolicyV1:
        """Predict knob overrides using heuristics."""
        policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha=features.sha256(),
        )

        # Rule 1: Reduce gain if audit delta is negative
        if features.audit_delta_success is not None and features.audit_delta_success < -0.1:
            policy.gain_multiplier_override = max(
                base_config.gain_schedule.full_multiplier * 0.8,
                MIN_GAIN_MULTIPLIER,
            )
            policy.conservative_multiplier_override = max(
                base_config.gain_schedule.conservative_multiplier * 0.9,
                MIN_CONSERVATIVE_MULTIPLIER,
            )

        # Rule 2: Increase patience if probe transfer fails repeatedly
        if features.probe_transfer_pass is False:
            current_patience = base_config.gain_schedule.cooldown_steps or 3
            policy.patience_override = min(current_patience + 1, MAX_PATIENCE)

        # Rule 3: Tighten thresholds if regal detected issues
        threshold_overrides = {}

        if features.regal_spec_score is not None and features.regal_spec_score < 0.7:
            # Require higher spec consistency when recent score was low
            threshold_overrides["spec_consistency_min"] = 0.6

        if features.regal_coherence_score is not None and features.regal_coherence_score < 0.7:
            # Require higher coherence when recent score was low
            threshold_overrides["coherence_min"] = 0.6

        if features.regal_hack_prob is not None and features.regal_hack_prob > 0.2:
            # Lower hack tolerance when detected
            threshold_overrides["hack_prob_max"] = 0.2

        if threshold_overrides:
            policy.threshold_overrides = threshold_overrides

        # Rule 4: Adjust task family biases based on coverage
        if features.task_family_weights:
            # Simple balancing: bias toward underweighted families
            avg_weight = sum(features.task_family_weights.values()) / len(features.task_family_weights)
            biases = {}
            for family, weight in features.task_family_weights.items():
                if weight < avg_weight * 0.5:
                    biases[family] = 0.1  # Small upward bias
            if biases:
                policy.task_family_biases = biases

        # Apply hard constraints
        return self.apply_hard_constraints(policy)


# =============================================================================
# Stub Learned Model (placeholder for future implementation)
# =============================================================================

class StubLearnedKnobModel(KnobModel):
    """Stub learned model for testing/integration.

    In production, this would be replaced by a trained model.
    Currently delegates to heuristic fallback with "learned" label.
    """

    def __init__(self, model_sha: str = "stub_model_v1"):
        self.model_sha = model_sha
        self._heuristic = HeuristicKnobProvider()

    def predict(
        self,
        features: RegimeFeaturesV1,
        base_config: PlanPolicyConfigV1,
    ) -> KnobPolicyV1:
        """Predict using stub (delegates to heuristic)."""
        policy = self._heuristic.predict(features, base_config)
        policy.policy_source = "learned"  # Mark as learned for testing
        policy.model_sha = self.model_sha
        return policy


# =============================================================================
# Factory Function
# =============================================================================

def get_knob_model(use_learned: bool = False, model_path: Optional[str] = None) -> KnobModel:
    """Get knob model based on availability.

    Args:
        use_learned: Whether to try loading a learned model
        model_path: Path to learned model (if any)

    Returns:
        KnobModel instance (learned or heuristic fallback)
    """
    if use_learned:
        # In future: load trained model from model_path
        # For now, return stub
        return StubLearnedKnobModel()

    return HeuristicKnobProvider()


__all__ = [
    "KnobModel",
    "HeuristicKnobProvider",
    "StubLearnedKnobModel",
    "get_knob_model",
    # Hard constraints
    "MIN_GAIN_MULTIPLIER",
    "MAX_GAIN_MULTIPLIER",
    "MIN_PATIENCE",
    "MAX_PATIENCE",
]
