"""Tests for regal evaluators using econ tensor.

Verifies that RewardIntegrityRegal consumes econ tensor and produces
stable flags, with fallback when tensor is absent.
"""
import pytest
from typing import Dict, Any

from src.contracts.schemas import (
    RegalGatesV1,
    RegalReportV1,
    SemanticUpdatePlanV1,
    TaskGraphOp,
    PlanOpType,
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
    EconTensorV1,
)
from src.regal.regal_evaluator import (
    RewardIntegrityRegal,
    SpecGuardianRegal,
    WorldCoherenceRegal,
    evaluate_regals,
)
from src.economics.econ_tensor import econ_to_tensor
from src.economics.econ_basis_registry import get_default_basis


class TestRewardIntegrityWithEconTensor:
    """Tests for RewardIntegrityRegal with econ tensor input."""

    def _make_policy_config(self) -> PlanPolicyConfigV1:
        """Create base policy config."""
        return PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(full_multiplier=1.5),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )

    def test_econ_tensor_consumed(self):
        """Test that econ tensor is consumed when provided."""
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        # Create econ tensor
        econ_data = {
            "mpl_units_per_hour": 10.0,
            "success_rate": 0.9,
            "energy_cost": 2.0,
            "damage_cost": 0.5,
            "reward_scalar_sum": 1.0,
        }
        tensor = econ_to_tensor(econ_data)

        context = {
            "econ_tensor_v1": tensor,
            "econ_basis_sha": get_default_basis().sha256,
        }

        report = regal.evaluate(None, None, config, context)

        assert report.findings.get("econ_tensor_available") is True
        assert report.findings.get("econ_basis_sha") is not None

    def test_fallback_without_tensor(self):
        """Test fallback path when tensor is absent."""
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        report = regal.evaluate(None, None, config, context=None)

        assert report.findings.get("econ_tensor_available") is False
        assert report.passed is True  # Should pass with no violations

    def test_econ_anomaly_detection(self):
        """Test detection of econ anomalies (high reward + high damage + low MPL)."""
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        # Create anomalous econ data
        econ_data = {
            "mpl_units_per_hour": 0.05,  # Very low MPL
            "reward_scalar_sum": 2.0,    # High reward
            "damage_cost": 10.0,         # High damage
        }
        tensor = econ_to_tensor(econ_data)

        context = {
            "econ_tensor_v1": tensor,
            "econ_basis_sha": get_default_basis().sha256,
        }

        report = regal.evaluate(None, None, config, context)

        # Should detect econ anomaly
        assert "econ_anomaly" in report.integrity_flags
        assert report.passed is False

    def test_normal_econ_passes(self):
        """Test normal econ data doesn't trigger false positives."""
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        # Create normal econ data
        econ_data = {
            "mpl_units_per_hour": 15.0,   # Good MPL
            "reward_scalar_sum": 0.8,     # Normal reward
            "damage_cost": 0.5,           # Low damage
        }
        tensor = econ_to_tensor(econ_data)

        context = {
            "econ_tensor_v1": tensor,
            "econ_basis_sha": get_default_basis().sha256,
        }

        report = regal.evaluate(None, None, config, context)

        assert "econ_anomaly" not in report.integrity_flags
        assert report.passed is True

    def test_econ_values_in_findings(self):
        """Test that econ values are included in findings."""
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        econ_data = {
            "mpl_units_per_hour": 10.0,
            "success_rate": 0.85,
        }
        tensor = econ_to_tensor(econ_data)

        context = {
            "econ_tensor_v1": tensor,
            "econ_basis_sha": get_default_basis().sha256,
        }

        report = regal.evaluate(None, None, config, context)

        assert "econ_values" in report.findings
        assert isinstance(report.findings["econ_values"], dict)


class TestRegalStableOutput:
    """Tests for stable regal output with econ tensor."""

    def test_deterministic_with_same_tensor(self):
        """Test regal produces deterministic output with same tensor."""
        regal = RewardIntegrityRegal(seed=42)
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )

        econ_data = {"mpl_units_per_hour": 10.0, "success_rate": 0.9}
        tensor = econ_to_tensor(econ_data)
        context = {
            "econ_tensor_v1": tensor,
            "econ_basis_sha": get_default_basis().sha256,
        }

        report1 = regal.evaluate(None, None, config, context)
        report2 = regal.evaluate(None, None, config, context)

        assert report1.passed == report2.passed
        assert report1.hack_probability == report2.hack_probability

    def test_different_tensor_different_output(self):
        """Test different tensor produces different output."""
        regal = RewardIntegrityRegal(seed=42)
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )

        # Normal tensor
        tensor1 = econ_to_tensor({
            "mpl_units_per_hour": 15.0,
            "reward_scalar_sum": 0.5,
            "damage_cost": 0.1,
        })

        # Anomalous tensor
        tensor2 = econ_to_tensor({
            "mpl_units_per_hour": 0.01,
            "reward_scalar_sum": 5.0,
            "damage_cost": 20.0,
        })

        context1 = {"econ_tensor_v1": tensor1, "econ_basis_sha": "sha1"}
        context2 = {"econ_tensor_v1": tensor2, "econ_basis_sha": "sha2"}

        report1 = regal.evaluate(None, None, config, context1)
        report2 = regal.evaluate(None, None, config, context2)

        # Reports should differ
        assert report1.passed != report2.passed or report1.integrity_flags != report2.integrity_flags


class TestEvaluateRegalsWithEconTensor:
    """Tests for aggregate regal evaluation with econ tensor."""

    def test_all_regals_receive_context(self):
        """Test all regals receive econ tensor context."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian", "world_coherence", "reward_integrity"],
            patience=3,
            determinism_seed=42,
        )

        policy_config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )

        tensor = econ_to_tensor({"mpl_units_per_hour": 10.0})
        context = {
            "econ_tensor_v1": tensor,
            "econ_basis_sha": get_default_basis().sha256,
        }

        result = evaluate_regals(
            config,
            plan=None,
            signals=None,
            policy_config=policy_config,
            context=context,
        )

        assert len(result.reports) == 3
        assert result.all_passed is True

    def test_reward_integrity_uses_tensor_in_aggregate(self):
        """Test reward_integrity regal uses tensor in aggregate evaluation."""
        config = RegalGatesV1(
            enabled_regal_ids=["reward_integrity"],
            patience=3,
            determinism_seed=42,
        )

        policy_config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )

        # Anomalous tensor
        tensor = econ_to_tensor({
            "mpl_units_per_hour": 0.01,
            "reward_scalar_sum": 5.0,
            "damage_cost": 20.0,
        })
        context = {
            "econ_tensor_v1": tensor,
            "econ_basis_sha": get_default_basis().sha256,
        }

        result = evaluate_regals(
            config,
            plan=None,
            signals=None,
            policy_config=policy_config,
            context=context,
        )

        # Should detect anomaly
        reward_report = result.reports[0]
        assert reward_report.regal_id == "reward_integrity"
        assert "econ_anomaly" in reward_report.integrity_flags


class TestRegalEconTensorFallback:
    """Tests for graceful fallback when econ tensor is unavailable."""

    def test_fallback_on_missing_tensor(self):
        """Test regals work when econ tensor is missing."""
        regal = RewardIntegrityRegal(seed=42)
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )

        # No econ tensor in context, stable weight history (no oscillation)
        context = {"weight_history": [1.0, 1.0, 1.0]}

        report = regal.evaluate(None, None, config, context)

        assert report.passed is True  # Should pass
        assert report.findings.get("econ_tensor_available") is False

    def test_fallback_on_empty_context(self):
        """Test regals work with None context."""
        regal = RewardIntegrityRegal(seed=42)
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )

        report = regal.evaluate(None, None, config, context=None)

        assert report.passed is True
        assert report.findings.get("econ_tensor_available") is False

    def test_fallback_preserves_other_checks(self):
        """Test fallback preserves weight oscillation check."""
        regal = RewardIntegrityRegal(seed=42)
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )

        # No tensor, but weight history with oscillations
        context = {
            "weight_history": [1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0],  # High oscillation
        }

        report = regal.evaluate(None, None, config, context)

        assert "oscillation" in report.integrity_flags
        assert report.findings.get("econ_tensor_available") is False
