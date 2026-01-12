"""Scaffolding hooks for economic and reward integrity policies.

These interfaces allow future integration of complex economic logic and anti-hacking guards
without disrupting the core homeostatic loop.

Includes D4 knob calibration integration for learned/heuristic parameter adaptation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from src.contracts.schemas import (
    PlanGainScheduleV1,
    PlanPolicyConfigV1,
    RegimeFeaturesV1,
    KnobPolicyV1,
    LedgerRegalV1,
)

if TYPE_CHECKING:
    from src.representation.homeostasis import SignalBundle
    from src.valuation.exposure_manifest import ExposureManifestV1
    from src.regal.knob_model import KnobModel
    from src.contracts.schemas import EconTensorV1


class EconPlanPolicyProvider(ABC):
    """Interface for economic policy providers."""
    
    @abstractmethod
    def get_gain_schedule(
        self,
        signal_bundle: "SignalBundle",
        exposure_manifest: Optional["ExposureManifestV1"] = None,
    ) -> Optional[PlanGainScheduleV1]:
        """Get gain schedule override for the current context.
        
        Returns:
            Review-approved gain schedule, or None to use default config.
        """
        pass


class RewardIntegrityGuard(ABC):
    """Interface for reward integrity guards."""
    
    @abstractmethod
    def adjust_gain_schedule(
        self,
        schedule: PlanGainScheduleV1,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> PlanGainScheduleV1:
        """Adjust gain schedule to prevent reward hacking or instability.
        
        Args:
            schedule: Proposed gain schedule
            telemetry: Runtime telemetry
            
        Returns:
            Safe gain schedule
        """
        pass


class DefaultEconPolicyProvider(EconPlanPolicyProvider):
    """Default no-op provider."""
    
    def get_gain_schedule(
        self,
        signal_bundle: "SignalBundle",
        exposure_manifest: Optional["ExposureManifestV1"] = None,
    ) -> Optional[PlanGainScheduleV1]:
        return None


class DefaultRewardIntegrityGuard(RewardIntegrityGuard):
    """Default identity guard."""

    def adjust_gain_schedule(
        self,
        schedule: PlanGainScheduleV1,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> PlanGainScheduleV1:
        return schedule


# =============================================================================
# D4 Knob Calibration Integration
# =============================================================================

def build_regime_features(
    signal_bundle: Optional["SignalBundle"] = None,
    exposure_manifest: Optional["ExposureManifestV1"] = None,
    regal_result: Optional[LedgerRegalV1] = None,
    context: Optional[Dict[str, Any]] = None,
    task_family_weights: Optional[Dict[str, float]] = None,
    econ_tensor: Optional["EconTensorV1"] = None,
) -> RegimeFeaturesV1:
    """Build RegimeFeaturesV1 from available context.

    Args:
        signal_bundle: Current signal bundle
        exposure_manifest: Current exposure manifest
        regal_result: Result from regal evaluation
        context: Additional context (weight_history, etc.)
        task_family_weights: Current task family weights
        econ_tensor: Optional econ tensor for coordinate chart data

    Returns:
        RegimeFeaturesV1 populated from available data
    """
    features = RegimeFeaturesV1()

    # Extract probe transfer info from signal bundle
    if signal_bundle:
        from src.representation.homeostasis import SignalType
        probe_signal = signal_bundle.get_signal(SignalType.DELTA_EPI_PER_FLOP)
        if probe_signal:
            meta = probe_signal.metadata
            features.probe_transfer_pass = meta.get("transfer_pass")
            features.audit_delta_success = meta.get("raw_delta")

    # Extract exposure info
    if exposure_manifest:
        features.current_exposure_count = getattr(exposure_manifest, "total_episodes", None)

    # Extract regal summary scores
    if regal_result and regal_result.reports:
        for report in regal_result.reports:
            if report.regal_id == "spec_guardian":
                features.regal_spec_score = report.spec_consistency_score
            elif report.regal_id == "world_coherence":
                features.regal_coherence_score = report.coherence_score
            elif report.regal_id == "reward_integrity":
                features.regal_hack_prob = report.hack_probability

    # Set task family weights
    if task_family_weights:
        features.task_family_weights = task_family_weights

    # Additional context
    objective_profile: Dict[str, Any] = {}
    if context:
        weight_history = context.get("weight_history", [])
        if weight_history:
            objective_profile["weight_history_len"] = len(weight_history)

    # Extract econ tensor info (if available)
    if econ_tensor is not None:
        objective_profile["econ_tensor_sha"] = econ_tensor.sha256()
        objective_profile["econ_basis_sha"] = econ_tensor.basis_sha
        # Include key econ values for the model
        if econ_tensor.stats:
            objective_profile["econ_norm"] = econ_tensor.stats.get("norm", 0.0)

    if objective_profile:
        features.objective_profile = objective_profile

    return features


class KnobAwareEconPolicyProvider(EconPlanPolicyProvider):
    """Economic policy provider that uses knob model for parameter adaptation.

    Integrates D4 learned/heuristic knob calibration to adjust gain schedules
    and other policy parameters based on regime features.
    """

    def __init__(
        self,
        knob_model: "KnobModel",
        base_config: PlanPolicyConfigV1,
    ):
        """Initialize with knob model and base config.

        Args:
            knob_model: KnobModel instance (learned or heuristic)
            base_config: Base policy config to override
        """
        self._knob_model = knob_model
        self._base_config = base_config
        self._last_knob_policy: Optional[KnobPolicyV1] = None
        self._last_regime_features: Optional[RegimeFeaturesV1] = None

    def get_gain_schedule(
        self,
        signal_bundle: "SignalBundle",
        exposure_manifest: Optional["ExposureManifestV1"] = None,
        regal_result: Optional[LedgerRegalV1] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[PlanGainScheduleV1]:
        """Get gain schedule with knob model overrides.

        Args:
            signal_bundle: Current signal bundle
            exposure_manifest: Current exposure manifest
            regal_result: Result from regal evaluation
            context: Additional context

        Returns:
            Gain schedule with knob overrides applied
        """
        # Build regime features
        features = build_regime_features(
            signal_bundle=signal_bundle,
            exposure_manifest=exposure_manifest,
            regal_result=regal_result,
            context=context,
            task_family_weights=self._base_config.default_weights,
        )
        self._last_regime_features = features

        # Get knob policy from model
        knob_policy = self._knob_model.predict(features, self._base_config)
        self._last_knob_policy = knob_policy

        # Apply overrides to base gain schedule
        base_schedule = self._base_config.gain_schedule
        updates = {}

        if knob_policy.gain_multiplier_override is not None:
            updates["full_multiplier"] = knob_policy.gain_multiplier_override

        if knob_policy.conservative_multiplier_override is not None:
            updates["conservative_multiplier"] = knob_policy.conservative_multiplier_override

        if knob_policy.patience_override is not None:
            updates["cooldown_steps"] = knob_policy.patience_override

        if updates:
            return base_schedule.model_copy(update=updates)

        return None  # No overrides needed

    @property
    def last_knob_policy(self) -> Optional[KnobPolicyV1]:
        """Get the last computed knob policy."""
        return self._last_knob_policy

    @property
    def last_regime_features(self) -> Optional[RegimeFeaturesV1]:
        """Get the last computed regime features."""
        return self._last_regime_features


class KnobAwareRewardIntegrityGuard(RewardIntegrityGuard):
    """Reward integrity guard that respects knob policy constraints.

    Can veto or clip learned knob outputs based on integrity checks.
    """

    def __init__(
        self,
        knob_policy_provider: Optional[KnobAwareEconPolicyProvider] = None,
    ):
        """Initialize with optional knob policy provider.

        Args:
            knob_policy_provider: Provider to get knob policy from
        """
        self._knob_provider = knob_policy_provider
        self._vetoed = False
        self._veto_reason: Optional[str] = None

    def adjust_gain_schedule(
        self,
        schedule: PlanGainScheduleV1,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> PlanGainScheduleV1:
        """Adjust gain schedule with integrity constraints.

        Uses knob policy thresholds if available to enforce bounds.

        Args:
            schedule: Proposed gain schedule
            telemetry: Runtime telemetry including regal results

        Returns:
            Safe gain schedule (possibly clamped)
        """
        self._vetoed = False
        self._veto_reason = None

        # Get knob policy if available
        knob_policy = None
        if self._knob_provider:
            knob_policy = self._knob_provider.last_knob_policy

        # Check telemetry for hack indicators
        if telemetry:
            hack_prob = telemetry.get("hack_probability", 0.0)
            max_hack_prob = 0.3  # Default threshold

            if knob_policy and knob_policy.threshold_overrides:
                max_hack_prob = knob_policy.threshold_overrides.get(
                    "hack_prob_max", max_hack_prob
                )

            if hack_prob > max_hack_prob:
                # Veto aggressive gains
                self._vetoed = True
                self._veto_reason = f"hack_probability {hack_prob:.2f} > {max_hack_prob:.2f}"

                # Clamp to conservative values
                return schedule.model_copy(update={
                    "full_multiplier": min(schedule.full_multiplier, 1.2),
                    "conservative_multiplier": min(schedule.conservative_multiplier, 1.1),
                })

        return schedule

    @property
    def was_vetoed(self) -> bool:
        """Check if last adjustment resulted in a veto."""
        return self._vetoed

    @property
    def veto_reason(self) -> Optional[str]:
        """Get the reason for the last veto."""
        return self._veto_reason


__all__ = [
    "EconPlanPolicyProvider",
    "RewardIntegrityGuard",
    "DefaultEconPolicyProvider",
    "DefaultRewardIntegrityGuard",
    "build_regime_features",
    "KnobAwareEconPolicyProvider",
    "KnobAwareRewardIntegrityGuard",
]
