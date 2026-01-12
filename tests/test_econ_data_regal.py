"""Tests for EconDataRegal sibling node.

Verifies that econ/data is a true sibling regal node:
- Registered in REGAL_REGISTRY
- Produces deterministic RegalReportV1 with stable SHA
- Included in LedgerRegalV1 aggregation when enabled
"""
import pytest
from typing import Dict, Any

from src.contracts.schemas import (
    RegalGatesV1,
    RegalReportV1,
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
    EconTensorV1,
)
from src.regal.regal_evaluator import (
    REGAL_REGISTRY,
    RegalNode,
    EconDataRegal,
    evaluate_regals,
)
from src.economics.econ_tensor import econ_to_tensor
from src.economics.econ_basis_registry import get_default_basis


class TestEconDataRegalRegistry:
    """Tests for EconDataRegal registration and interface."""

    def test_econ_data_in_registry(self):
        """Test that econ_data is registered in REGAL_REGISTRY."""
        assert "econ_data" in REGAL_REGISTRY
        assert REGAL_REGISTRY["econ_data"] is EconDataRegal

    def test_econ_data_is_regal_node(self):
        """Test that EconDataRegal inherits from RegalNode."""
        assert issubclass(EconDataRegal, RegalNode)

    def test_econ_data_has_correct_id(self):
        """Test that EconDataRegal.regal_id is correct."""
        assert EconDataRegal.regal_id == "econ_data"

    def test_econ_data_instantiation(self):
        """Test that EconDataRegal can be instantiated."""
        regal = EconDataRegal(seed=42)
        assert regal.seed == 42
        assert regal.regal_id == "econ_data"


class TestEconDataRegalDeterminism:
    """Tests for EconDataRegal determinism and report SHA stability."""

    def _make_econ_context(self) -> Dict[str, Any]:
        """Create a context with econ tensor."""
        basis = get_default_basis()
        econ_data = {
            "mpl_units_per_hour": 10.0,
            "success_rate": 0.85,
            "energy_cost": 2.0,
        }
        tensor = econ_to_tensor(econ_data, basis=basis.spec, source="synthetic")
        return {
            "econ_tensor_v1": tensor,
            "econ_basis_sha": basis.sha256,
        }

    def test_inputs_sha_deterministic(self):
        """Test that same inputs produce same inputs_sha (deterministic)."""
        regal = EconDataRegal(seed=42)
        context = self._make_econ_context()

        report1 = regal.evaluate(None, None, None, context)
        report2 = regal.evaluate(None, None, None, context)

        # inputs_sha should be deterministic (excludes timestamp)
        assert report1.inputs_sha == report2.inputs_sha
        assert len(report1.inputs_sha) == 64

        # report_sha includes timestamp so may differ, but should be valid
        assert len(report1.report_sha) == 64
        assert len(report2.report_sha) == 64

    def test_inputs_sha_changes_with_tensor(self):
        """Test that different tensor produces different inputs_sha."""
        regal = EconDataRegal(seed=42)
        basis = get_default_basis()

        tensor1 = econ_to_tensor({"mpl_units_per_hour": 10.0}, basis=basis.spec)
        tensor2 = econ_to_tensor({"mpl_units_per_hour": 20.0}, basis=basis.spec)

        context1 = {"econ_tensor_v1": tensor1, "econ_basis_sha": basis.sha256}
        context2 = {"econ_tensor_v1": tensor2, "econ_basis_sha": basis.sha256}

        report1 = regal.evaluate(None, None, None, context1)
        report2 = regal.evaluate(None, None, None, context2)

        # Different inputs should produce different inputs_sha
        assert report1.inputs_sha != report2.inputs_sha

    def test_report_sha_changes_with_seed(self):
        """Test that different seed produces different inputs_sha."""
        context = self._make_econ_context()

        regal1 = EconDataRegal(seed=42)
        regal2 = EconDataRegal(seed=43)

        report1 = regal1.evaluate(None, None, None, context)
        report2 = regal2.evaluate(None, None, None, context)

        assert report1.inputs_sha != report2.inputs_sha


class TestEconDataRegalValidation:
    """Tests for EconDataRegal validation logic."""

    def test_valid_tensor_passes(self):
        """Test that valid econ tensor passes validation."""
        regal = EconDataRegal(seed=42)
        basis = get_default_basis()

        tensor = econ_to_tensor({
            "mpl_units_per_hour": 10.0,
            "success_rate": 0.85,
        }, basis=basis.spec, source="synthetic")

        context = {"econ_tensor_v1": tensor, "econ_basis_sha": basis.sha256}
        report = regal.evaluate(None, None, None, context)

        assert report.passed is True
        assert report.regal_id == "econ_data"
        assert report.findings.get("econ_tensor_available") is True
        assert report.findings.get("basis_verified") is True

    def test_missing_tensor_passes_with_low_confidence(self):
        """Test that missing tensor passes but with low confidence."""
        regal = EconDataRegal(seed=42)
        report = regal.evaluate(None, None, None, context=None)

        assert report.passed is True
        assert report.confidence == 0.5
        assert report.findings.get("econ_tensor_available") is False

    def test_nan_values_fail(self):
        """Test that NaN values cause validation failure."""
        regal = EconDataRegal(seed=42)
        basis = get_default_basis()

        # Create tensor with NaN
        tensor = EconTensorV1(
            basis_id=basis.spec.basis_id,
            basis_sha=basis.sha256,
            x=[float("nan"), 1.0, 2.0] + [0.0] * 7,  # 10 axes
            source="synthetic",
        )

        context = {"econ_tensor_v1": tensor, "econ_basis_sha": basis.sha256}
        report = regal.evaluate(None, None, None, context)

        assert report.passed is False
        assert "nan_values" in report.coherence_tags
        assert report.findings.get("nan_count") == 1

    def test_inf_values_fail(self):
        """Test that Inf values cause validation failure."""
        regal = EconDataRegal(seed=42)
        basis = get_default_basis()

        # Create tensor with Inf
        tensor = EconTensorV1(
            basis_id=basis.spec.basis_id,
            basis_sha=basis.sha256,
            x=[float("inf"), 1.0, 2.0] + [0.0] * 7,
            source="synthetic",
        )

        context = {"econ_tensor_v1": tensor, "econ_basis_sha": basis.sha256}
        report = regal.evaluate(None, None, None, context)

        assert report.passed is False
        assert "inf_values" in report.coherence_tags
        assert report.findings.get("inf_count") == 1

    def test_basis_sha_mismatch_fails(self):
        """Test that basis SHA mismatch causes failure."""
        regal = EconDataRegal(seed=42)
        basis = get_default_basis()

        # Create tensor with wrong basis SHA
        tensor = EconTensorV1(
            basis_id=basis.spec.basis_id,
            basis_sha="wrong_sha" + "0" * 56,  # Wrong SHA
            x=[1.0] * 10,
            source="synthetic",
        )

        context = {"econ_tensor_v1": tensor, "econ_basis_sha": basis.sha256}
        report = regal.evaluate(None, None, None, context)

        assert report.passed is False
        assert "basis_sha_mismatch" in report.coherence_tags


class TestEconDataRegalInAggregation:
    """Tests for EconDataRegal in LedgerRegalV1 aggregation."""

    def test_econ_data_included_in_evaluate_regals(self):
        """Test that econ_data regal is included when enabled."""
        config = RegalGatesV1(
            enabled_regal_ids=["econ_data"],
            patience=3,
            determinism_seed=42,
        )

        policy_config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )

        basis = get_default_basis()
        tensor = econ_to_tensor({"mpl_units_per_hour": 10.0}, basis=basis.spec)
        context = {"econ_tensor_v1": tensor, "econ_basis_sha": basis.sha256}

        result = evaluate_regals(
            config=config,
            plan=None,
            signals=None,
            policy_config=policy_config,
            context=context,
        )

        assert len(result.reports) == 1
        assert result.reports[0].regal_id == "econ_data"
        assert result.all_passed is True

    def test_econ_data_with_other_regals(self):
        """Test econ_data alongside other regals in aggregation."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian", "world_coherence", "reward_integrity", "econ_data"],
            patience=3,
            determinism_seed=42,
        )

        policy_config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )

        basis = get_default_basis()
        tensor = econ_to_tensor({
            "mpl_units_per_hour": 10.0,
            "success_rate": 0.9,
        }, basis=basis.spec)
        context = {"econ_tensor_v1": tensor, "econ_basis_sha": basis.sha256}

        result = evaluate_regals(
            config=config,
            plan=None,
            signals=None,
            policy_config=policy_config,
            context=context,
        )

        # Should have 4 reports (all enabled regals)
        assert len(result.reports) == 4

        regal_ids = [r.regal_id for r in result.reports]
        assert "spec_guardian" in regal_ids
        assert "world_coherence" in regal_ids
        assert "reward_integrity" in regal_ids
        assert "econ_data" in regal_ids

        # All should pass with valid tensor
        assert result.all_passed is True

    def test_econ_data_failure_affects_all_passed(self):
        """Test that econ_data failure affects all_passed in aggregation."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian", "econ_data"],
            patience=3,
            determinism_seed=42,
        )

        policy_config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )

        basis = get_default_basis()
        # Create tensor with NaN to trigger failure
        tensor = EconTensorV1(
            basis_id=basis.spec.basis_id,
            basis_sha=basis.sha256,
            x=[float("nan")] + [0.0] * 9,
            source="synthetic",
        )
        context = {"econ_tensor_v1": tensor, "econ_basis_sha": basis.sha256}

        result = evaluate_regals(
            config=config,
            plan=None,
            signals=None,
            policy_config=policy_config,
            context=context,
        )

        # econ_data should fail, affecting all_passed
        assert len(result.reports) == 2
        econ_report = next(r for r in result.reports if r.regal_id == "econ_data")
        assert econ_report.passed is False
        assert result.all_passed is False

    def test_combined_inputs_sha_includes_econ_data(self):
        """Test that combined_inputs_sha includes econ_data report."""
        config = RegalGatesV1(
            enabled_regal_ids=["econ_data"],
            patience=3,
            determinism_seed=42,
        )

        basis = get_default_basis()
        tensor = econ_to_tensor({"mpl_units_per_hour": 10.0}, basis=basis.spec)
        context = {"econ_tensor_v1": tensor, "econ_basis_sha": basis.sha256}

        result = evaluate_regals(
            config=config,
            plan=None,
            signals=None,
            policy_config=None,
            context=context,
        )

        # combined_inputs_sha should be computed from all reports
        assert len(result.combined_inputs_sha) == 64
        assert result.regal_config_sha == config.sha256()
