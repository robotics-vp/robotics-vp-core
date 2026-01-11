"""Tests for meta-regal gates (Stage-6 deterministic audit nodes)."""
import pytest
from typing import Dict, Any, Optional, List

from src.contracts.schemas import (
    RegalGatesV1,
    RegalReportV1,
    LedgerRegalV1,
    SemanticUpdatePlanV1,
    TaskGraphOp,
    PlanOpType,
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
)
from src.representation.homeostasis import SignalBundle, ControlSignal, SignalType
from src.regal.regal_evaluator import (
    RegalNode,
    REGAL_REGISTRY,
    evaluate_regals,
    register_regal,
    SpecGuardianRegal,
    WorldCoherenceRegal,
    RewardIntegrityRegal,
)


class TestRegalSchemas:
    """Tests for regal schema validation and hashing."""

    def test_regal_gates_config_creation(self):
        """Test RegalGatesV1 creation and defaults."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian", "world_coherence"],
            patience=5,
            penalty_mode="warn",
        )
        assert config.enabled_regal_ids == ["spec_guardian", "world_coherence"]
        assert config.patience == 5
        assert config.penalty_mode == "warn"
        assert config.determinism_seed == 42  # default

    def test_regal_gates_sha_deterministic(self):
        """Test that SHA is deterministic."""
        config1 = RegalGatesV1(enabled_regal_ids=["spec_guardian"])
        config2 = RegalGatesV1(enabled_regal_ids=["spec_guardian"])
        assert config1.sha256() == config2.sha256()

    def test_regal_report_creation(self):
        """Test RegalReportV1 creation and SHA computation."""
        report = RegalReportV1(
            regal_id="spec_guardian",
            inputs_sha="abc123",
            determinism_seed=42,
            passed=True,
            confidence=0.95,
            rationale="All checks passed",
        )
        report.compute_sha()
        assert report.report_sha != ""
        assert len(report.report_sha) == 64  # SHA-256 hex

    def test_regal_report_sha_deterministic(self):
        """Test that report SHA is deterministic given same inputs (including timestamp)."""
        # Use fixed timestamp for determinism
        fixed_ts = "2024-01-01T00:00:00"
        report1 = RegalReportV1(
            regal_id="spec_guardian",
            inputs_sha="abc123",
            determinism_seed=42,
            passed=True,
            confidence=0.95,
            rationale="Test",
            created_at=fixed_ts,
        )
        report1.compute_sha()

        report2 = RegalReportV1(
            regal_id="spec_guardian",
            inputs_sha="abc123",
            determinism_seed=42,
            passed=True,
            confidence=0.95,
            rationale="Test",
            created_at=fixed_ts,
        )
        report2.compute_sha()

        assert report1.report_sha == report2.report_sha

    def test_ledger_regal_creation(self):
        """Test LedgerRegalV1 creation."""
        report = RegalReportV1(
            regal_id="test",
            inputs_sha="test_sha",
            determinism_seed=42,
            passed=True,
        )
        report.compute_sha()

        ledger_regal = LedgerRegalV1(
            regal_config_sha="config_sha",
            reports=[report],
            all_passed=True,
            combined_inputs_sha="combined_sha",
        )
        assert len(ledger_regal.reports) == 1
        assert ledger_regal.all_passed is True


class TestRegalRegistry:
    """Tests for regal node registry."""

    def test_builtin_regals_registered(self):
        """Test that builtin regals are in the registry."""
        assert "spec_guardian" in REGAL_REGISTRY
        assert "world_coherence" in REGAL_REGISTRY
        assert "reward_integrity" in REGAL_REGISTRY

    def test_custom_regal_registration(self):
        """Test custom regal registration."""
        @register_regal("test_custom_regal")
        class TestCustomRegal(RegalNode):
            regal_id = "test_custom_regal"

            def evaluate(self, plan, signals, policy_config, context=None):
                return RegalReportV1(
                    regal_id=self.regal_id,
                    inputs_sha="test",
                    determinism_seed=self.seed,
                    passed=True,
                )

        assert "test_custom_regal" in REGAL_REGISTRY
        assert REGAL_REGISTRY["test_custom_regal"] == TestCustomRegal


class TestSpecGuardianRegal:
    """Tests for SpecGuardian regal node."""

    def _make_policy_config(self) -> PlanPolicyConfigV1:
        return PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(
                max_abs_weight_change=0.5,
            ),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )

    def test_spec_guardian_passes_valid_plan(self):
        """Test SpecGuardian passes valid plan."""
        regal = SpecGuardianRegal(seed=42)
        plan = SemanticUpdatePlanV1(
            plan_id="test",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="manipulation", weight=0.6),
            ],
        )
        config = self._make_policy_config()

        report = regal.evaluate(plan, None, config, None)
        assert report.passed is True
        assert report.regal_id == "spec_guardian"

    def test_spec_guardian_fails_unknown_task_family(self):
        """Test SpecGuardian fails on unknown task family."""
        regal = SpecGuardianRegal(seed=42)
        plan = SemanticUpdatePlanV1(
            plan_id="test",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="unknown_family", weight=0.5),
            ],
        )
        config = self._make_policy_config()

        report = regal.evaluate(plan, None, config, None)
        assert report.passed is False
        assert "unknown_family" in report.rationale or "violations" in str(report.findings)

    def test_spec_guardian_fails_weight_change_exceeds_max(self):
        """Test SpecGuardian fails when weight change exceeds max."""
        regal = SpecGuardianRegal(seed=42)
        plan = SemanticUpdatePlanV1(
            plan_id="test",
            task_graph_changes=[
                # Change from 0.5 to 1.5 = delta of 1.0, exceeds max of 0.5
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="manipulation", weight=1.5),
            ],
        )
        config = self._make_policy_config()

        report = regal.evaluate(plan, None, config, None)
        assert report.passed is False

    def test_spec_guardian_no_plan(self):
        """Test SpecGuardian with no plan."""
        regal = SpecGuardianRegal(seed=42)
        config = self._make_policy_config()

        report = regal.evaluate(None, None, config, None)
        assert report.passed is True
        assert "No plan" in report.rationale


class TestWorldCoherenceRegal:
    """Tests for WorldCoherence regal node."""

    def _make_signals(self, values: Dict[str, float]) -> SignalBundle:
        signals = []
        for name, val in values.items():
            sig_type = SignalType.EPIPLEXITY  # Default
            if "rate" in name.lower():
                sig_type = SignalType.STABILITY
            signals.append(
                ControlSignal(
                    signal_type=sig_type,
                    value=val,
                    target=0.5,
                    metadata={"name": name},
                )
            )
        return SignalBundle(signals=signals, timestamp="test")

    def test_world_coherence_passes_valid_signals(self):
        """Test WorldCoherence passes valid signals."""
        regal = WorldCoherenceRegal(seed=42)
        signals = self._make_signals({"value1": 0.5, "value2": 0.8})

        report = regal.evaluate(None, signals, None, None)
        assert report.passed is True

    def test_world_coherence_fails_nan_value(self):
        """Test WorldCoherence fails on NaN value."""
        regal = WorldCoherenceRegal(seed=42)
        signals = self._make_signals({"value1": float("nan")})

        report = regal.evaluate(None, signals, None, None)
        assert report.passed is False
        assert "NaN" in report.rationale or "violations" in str(report.findings)

    def test_world_coherence_fails_inf_value(self):
        """Test WorldCoherence fails on Inf value."""
        regal = WorldCoherenceRegal(seed=42)
        signals = self._make_signals({"value1": float("inf")})

        report = regal.evaluate(None, signals, None, None)
        assert report.passed is False

    def test_world_coherence_no_signals(self):
        """Test WorldCoherence with no signals."""
        regal = WorldCoherenceRegal(seed=42)

        report = regal.evaluate(None, None, None, None)
        assert report.passed is True


class TestRewardIntegrityRegal:
    """Tests for RewardIntegrity regal node."""

    def _make_policy_config(self) -> PlanPolicyConfigV1:
        return PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(
                full_multiplier=1.5,
            ),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )

    def test_reward_integrity_passes_no_oscillation(self):
        """Test RewardIntegrity passes with no oscillation."""
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        # Steady increase, no oscillation
        context = {"weight_history": [1.0, 1.1, 1.2, 1.3, 1.4]}

        report = regal.evaluate(None, None, config, context)
        assert report.passed is True

    def test_reward_integrity_fails_high_oscillation(self):
        """Test RewardIntegrity fails with high oscillation."""
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        # Alternating up/down = high oscillation
        context = {"weight_history": [1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0]}

        report = regal.evaluate(None, None, config, context)
        assert report.passed is False
        assert "oscillation" in report.rationale.lower() or "oscillation" in str(report.findings).lower()

    def test_reward_integrity_fails_anomalous_gain(self):
        """Test RewardIntegrity fails with anomalous gain request."""
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()
        plan = SemanticUpdatePlanV1(
            plan_id="test",
            task_graph_changes=[
                # 5.0 / 0.5 = 10x, way above full_multiplier * 1.5 = 2.25
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="manipulation", weight=5.0),
            ],
        )

        report = regal.evaluate(plan, None, config, None)
        assert report.passed is False


class TestEvaluateRegals:
    """Tests for the evaluate_regals function."""

    def test_evaluate_regals_all_pass(self):
        """Test evaluate_regals with all passing."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian", "world_coherence"],
            determinism_seed=42,
        )
        policy_config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )
        plan = SemanticUpdatePlanV1(
            plan_id="test",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="manipulation", weight=0.6),
            ],
        )

        result = evaluate_regals(config, plan, None, policy_config, None)

        assert isinstance(result, LedgerRegalV1)
        assert result.all_passed is True
        assert len(result.reports) == 2
        assert result.regal_config_sha == config.sha256()

    def test_evaluate_regals_some_fail(self):
        """Test evaluate_regals with some failing."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian"],
            determinism_seed=42,
        )
        policy_config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5},
        )
        # Unknown task family should fail spec_guardian
        plan = SemanticUpdatePlanV1(
            plan_id="test",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="unknown", weight=0.5),
            ],
        )

        result = evaluate_regals(config, plan, None, policy_config, None)

        assert result.all_passed is False
        assert len(result.reports) == 1
        assert result.reports[0].passed is False

    def test_evaluate_regals_empty_config(self):
        """Test evaluate_regals with no enabled regals."""
        config = RegalGatesV1(enabled_regal_ids=[])

        result = evaluate_regals(config, None, None, None, None)

        assert result.all_passed is True
        assert len(result.reports) == 0

    def test_evaluate_regals_unknown_regal_skipped(self):
        """Test that unknown regal IDs are skipped."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian", "nonexistent_regal"],
        )

        result = evaluate_regals(config, None, None, None, None)

        # Should only have 1 report (spec_guardian), nonexistent skipped
        assert len(result.reports) == 1
        assert result.reports[0].regal_id == "spec_guardian"

    def test_evaluate_regals_deterministic(self):
        """Test that evaluation is deterministic (inputs_sha is stable)."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian", "world_coherence"],
            determinism_seed=42,
        )

        result1 = evaluate_regals(config, None, None, None, None)
        result2 = evaluate_regals(config, None, None, None, None)

        # Combined inputs SHA should be deterministic
        assert result1.combined_inputs_sha == result2.combined_inputs_sha

        # Individual inputs_sha should be deterministic (not report_sha which includes timestamp)
        for r1, r2 in zip(result1.reports, result2.reports):
            assert r1.inputs_sha == r2.inputs_sha
            assert r1.regal_id == r2.regal_id
            assert r1.passed == r2.passed
