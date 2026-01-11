"""Tests for D4 knob model clamping behavior.

Verifies that knob model outputs are always bounded by hard constraints,
regardless of learned/heuristic predictions.
"""
import pytest
from typing import Dict, Optional

from src.contracts.schemas import (
    RegimeFeaturesV1,
    KnobPolicyV1,
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
)
from src.regal.knob_model import (
    KnobModel,
    HeuristicKnobProvider,
    StubLearnedKnobModel,
    get_knob_model,
    MIN_GAIN_MULTIPLIER,
    MAX_GAIN_MULTIPLIER,
    MIN_CONSERVATIVE_MULTIPLIER,
    MAX_CONSERVATIVE_MULTIPLIER,
    MIN_PATIENCE,
    MAX_PATIENCE,
    clamp,
)


class TestClampFunction:
    """Tests for the clamp utility function."""

    def test_clamp_within_bounds(self):
        """Test clamp returns value when within bounds."""
        assert clamp(0.5, 0.0, 1.0) == 0.5
        assert clamp(1.5, 1.0, 2.0) == 1.5

    def test_clamp_below_minimum(self):
        """Test clamp returns min when below."""
        assert clamp(-1.0, 0.0, 1.0) == 0.0
        assert clamp(0.3, 0.5, 1.0) == 0.5

    def test_clamp_above_maximum(self):
        """Test clamp returns max when above."""
        assert clamp(2.0, 0.0, 1.0) == 1.0
        assert clamp(5.0, 0.5, 3.0) == 3.0


class TestHardConstraints:
    """Tests for hard constraint constants."""

    def test_gain_multiplier_bounds(self):
        """Test gain multiplier bounds are reasonable."""
        assert MIN_GAIN_MULTIPLIER > 0  # Must be positive
        assert MAX_GAIN_MULTIPLIER > MIN_GAIN_MULTIPLIER
        assert MIN_GAIN_MULTIPLIER == 0.5
        assert MAX_GAIN_MULTIPLIER == 3.0

    def test_conservative_multiplier_bounds(self):
        """Test conservative multiplier bounds are reasonable."""
        assert MIN_CONSERVATIVE_MULTIPLIER > 0
        assert MAX_CONSERVATIVE_MULTIPLIER > MIN_CONSERVATIVE_MULTIPLIER
        assert MIN_CONSERVATIVE_MULTIPLIER == 0.8
        assert MAX_CONSERVATIVE_MULTIPLIER == 2.0

    def test_patience_bounds(self):
        """Test patience bounds are reasonable."""
        assert MIN_PATIENCE >= 1  # At least 1 try
        assert MAX_PATIENCE > MIN_PATIENCE
        assert MIN_PATIENCE == 1
        assert MAX_PATIENCE == 10


class TestKnobModelClamping:
    """Tests for knob model output clamping."""

    def _make_base_config(self) -> PlanPolicyConfigV1:
        """Create base policy config."""
        return PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(
                full_multiplier=1.5,
                conservative_multiplier=1.1,
                cooldown_steps=3,
            ),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )

    def _make_features(
        self,
        audit_delta: Optional[float] = None,
        probe_pass: Optional[bool] = None,
        hack_prob: Optional[float] = None,
    ) -> RegimeFeaturesV1:
        """Create regime features."""
        return RegimeFeaturesV1(
            audit_delta_success=audit_delta,
            probe_transfer_pass=probe_pass,
            regal_hack_prob=hack_prob,
        )

    def test_gain_multiplier_clamped_below(self):
        """Test gain multiplier is clamped when below minimum."""
        # Create a policy with too-low gain multiplier
        policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="test",
            gain_multiplier_override=0.1,  # Below MIN_GAIN_MULTIPLIER (0.5)
        )

        model = HeuristicKnobProvider()
        clamped_policy = model.apply_hard_constraints(policy)

        assert clamped_policy.gain_multiplier_override == MIN_GAIN_MULTIPLIER
        assert clamped_policy.clamped is True
        assert any("gain_multiplier" in r for r in clamped_policy.clamp_reasons)

    def test_gain_multiplier_clamped_above(self):
        """Test gain multiplier is clamped when above maximum."""
        policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="test",
            gain_multiplier_override=5.0,  # Above MAX_GAIN_MULTIPLIER (3.0)
        )

        model = HeuristicKnobProvider()
        clamped_policy = model.apply_hard_constraints(policy)

        assert clamped_policy.gain_multiplier_override == MAX_GAIN_MULTIPLIER
        assert clamped_policy.clamped is True

    def test_conservative_multiplier_clamped(self):
        """Test conservative multiplier is clamped."""
        policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="test",
            conservative_multiplier_override=0.5,  # Below MIN (0.8)
        )

        model = HeuristicKnobProvider()
        clamped_policy = model.apply_hard_constraints(policy)

        assert clamped_policy.conservative_multiplier_override == MIN_CONSERVATIVE_MULTIPLIER
        assert clamped_policy.clamped is True

    def test_patience_clamped_below(self):
        """Test patience is clamped when below minimum."""
        policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="test",
            patience_override=0,  # Below MIN_PATIENCE (1)
        )

        model = HeuristicKnobProvider()
        clamped_policy = model.apply_hard_constraints(policy)

        assert clamped_policy.patience_override == MIN_PATIENCE
        assert clamped_policy.clamped is True

    def test_patience_clamped_above(self):
        """Test patience is clamped when above maximum."""
        policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="test",
            patience_override=100,  # Above MAX_PATIENCE (10)
        )

        model = HeuristicKnobProvider()
        clamped_policy = model.apply_hard_constraints(policy)

        assert clamped_policy.patience_override == MAX_PATIENCE
        assert clamped_policy.clamped is True

    def test_threshold_overrides_clamped(self):
        """Test threshold overrides are clamped to [0, 1]."""
        policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="test",
            threshold_overrides={
                "spec_consistency_min": 1.5,  # Above 1.0
                "coherence_min": -0.5,  # Below 0.0
                "hack_prob_max": 0.5,  # Valid
            },
        )

        model = HeuristicKnobProvider()
        clamped_policy = model.apply_hard_constraints(policy)

        assert clamped_policy.threshold_overrides["spec_consistency_min"] == 1.0
        assert clamped_policy.threshold_overrides["coherence_min"] == 0.0
        assert clamped_policy.threshold_overrides["hack_prob_max"] == 0.5
        assert clamped_policy.clamped is True

    def test_no_clamping_when_valid(self):
        """Test no clamping flag when values are valid."""
        policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="test",
            gain_multiplier_override=1.5,  # Valid
            conservative_multiplier_override=1.2,  # Valid
            patience_override=5,  # Valid
        )

        model = HeuristicKnobProvider()
        clamped_policy = model.apply_hard_constraints(policy)

        assert clamped_policy.clamped is False
        assert len(clamped_policy.clamp_reasons) == 0


class TestHeuristicKnobProvider:
    """Tests for heuristic knob provider rules."""

    def _make_base_config(self) -> PlanPolicyConfigV1:
        """Create base policy config."""
        return PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(
                full_multiplier=1.5,
                conservative_multiplier=1.1,
                cooldown_steps=3,
            ),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )

    def test_reduce_gain_on_negative_delta(self):
        """Test Rule 1: Reduce gain when audit delta is negative."""
        features = RegimeFeaturesV1(
            audit_delta_success=-0.2,  # Negative delta
        )
        config = self._make_base_config()
        provider = HeuristicKnobProvider()

        policy = provider.predict(features, config)

        # Should reduce gain multiplier
        assert policy.gain_multiplier_override is not None
        assert policy.gain_multiplier_override < config.gain_schedule.full_multiplier

    def test_increase_patience_on_transfer_fail(self):
        """Test Rule 2: Increase patience when probe transfer fails."""
        features = RegimeFeaturesV1(
            probe_transfer_pass=False,
        )
        config = self._make_base_config()
        provider = HeuristicKnobProvider()

        policy = provider.predict(features, config)

        # Should increase patience
        assert policy.patience_override is not None
        assert policy.patience_override > config.gain_schedule.cooldown_steps

    def test_tighten_thresholds_on_low_regal_score(self):
        """Test Rule 3: Tighten thresholds when regal detects issues."""
        features = RegimeFeaturesV1(
            regal_spec_score=0.5,  # Low spec score
            regal_coherence_score=0.6,  # Low coherence
            regal_hack_prob=0.3,  # High hack probability
        )
        config = self._make_base_config()
        provider = HeuristicKnobProvider()

        policy = provider.predict(features, config)

        # Should set threshold overrides
        assert policy.threshold_overrides is not None
        assert "spec_consistency_min" in policy.threshold_overrides
        assert "coherence_min" in policy.threshold_overrides
        assert "hack_prob_max" in policy.threshold_overrides

    def test_task_family_biases_on_imbalance(self):
        """Test Rule 4: Bias toward underweighted task families."""
        features = RegimeFeaturesV1(
            task_family_weights={
                "manipulation": 0.9,  # Overweighted
                "navigation": 0.1,  # Underweighted
            },
        )
        config = self._make_base_config()
        provider = HeuristicKnobProvider()

        policy = provider.predict(features, config)

        # Should add upward bias for underweighted family
        if policy.task_family_biases:
            assert "navigation" in policy.task_family_biases


class TestStubLearnedKnobModel:
    """Tests for stub learned model."""

    def test_stub_model_source_is_learned(self):
        """Test stub model reports 'learned' source."""
        features = RegimeFeaturesV1()
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )
        model = StubLearnedKnobModel()

        policy = model.predict(features, config)

        assert policy.policy_source == "learned"
        assert policy.model_sha is not None

    def test_stub_model_has_sha(self):
        """Test stub model has SHA identifier."""
        model = StubLearnedKnobModel()
        assert model.model_sha == "stub_model_v1"

    def test_stub_model_delegates_to_heuristic(self):
        """Test stub model delegates to heuristic (with different label)."""
        features = RegimeFeaturesV1(
            audit_delta_success=-0.2,
        )
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(full_multiplier=1.5),
            default_weights={"manipulation": 0.5},
        )

        stub = StubLearnedKnobModel()
        heuristic = HeuristicKnobProvider()

        stub_policy = stub.predict(features, config)
        heuristic_policy = heuristic.predict(features, config)

        # Should have same overrides (stub delegates to heuristic)
        assert stub_policy.gain_multiplier_override == heuristic_policy.gain_multiplier_override


class TestGetKnobModel:
    """Tests for knob model factory function."""

    def test_get_heuristic_model(self):
        """Test getting heuristic model."""
        model = get_knob_model(use_learned=False)
        assert isinstance(model, HeuristicKnobProvider)

    def test_get_learned_model(self):
        """Test getting learned model (stub)."""
        model = get_knob_model(use_learned=True)
        assert isinstance(model, StubLearnedKnobModel)

    def test_default_is_heuristic(self):
        """Test default model is heuristic."""
        model = get_knob_model()
        assert isinstance(model, HeuristicKnobProvider)


class TestKnobPolicySha:
    """Tests for KnobPolicyV1 SHA computation."""

    def test_knob_policy_sha_deterministic(self):
        """Test knob policy SHA is deterministic."""
        policy1 = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="abc123",
            gain_multiplier_override=1.5,
        )
        policy2 = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="abc123",
            gain_multiplier_override=1.5,
        )

        assert policy1.sha256() == policy2.sha256()

    def test_knob_policy_sha_changes_on_override(self):
        """Test SHA changes when override values change."""
        policy1 = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="abc123",
            gain_multiplier_override=1.5,
        )
        policy2 = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="abc123",
            gain_multiplier_override=2.0,  # Different
        )

        assert policy1.sha256() != policy2.sha256()
