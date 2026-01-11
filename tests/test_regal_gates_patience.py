"""Tests for meta-regal gates patience behavior (Stage-6).

Verifies that regal gates respect patience configuration before triggering
actions (noop/warn/clamp).
"""
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
    evaluate_regals,
    SpecGuardianRegal,
)
from src.orchestrator.homeostatic_plan_writer import (
    check_gates,
    build_signal_bundle_for_plan,
)


class TestRegalPatienceBehavior:
    """Tests for regal gate patience tracking."""

    def _make_policy_config_with_regal(
        self,
        enabled_regal_ids: List[str],
        patience: int = 3,
        penalty_mode: str = "noop",
    ) -> PlanPolicyConfigV1:
        """Create policy config with regal gates enabled."""
        return PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(
                max_abs_weight_change=0.5,
            ),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
            regal_gates=RegalGatesV1(
                enabled_regal_ids=enabled_regal_ids,
                patience=patience,
                penalty_mode=penalty_mode,
                determinism_seed=42,
            ),
        )

    def _make_signals_with_probe(
        self,
        stability_pass: bool = True,
        transfer_pass: bool = True,
    ) -> SignalBundle:
        """Create signal bundle with probe signal."""
        return SignalBundle(
            signals=[
                ControlSignal(
                    signal_type=SignalType.DELTA_EPI_PER_FLOP,
                    value=1e-8,
                    target=0.0,
                    metadata={
                        "stability_pass": stability_pass,
                        "transfer_pass": transfer_pass,
                        "sign_consistency": 0.9,
                        "raw_delta": 0.01,
                        "flops_estimate": 1e9,
                    },
                ),
            ],
            timestamp="test",
        )

    def _make_failing_plan(self) -> SemanticUpdatePlanV1:
        """Create a plan that will fail spec_guardian check."""
        return SemanticUpdatePlanV1(
            plan_id="test_failing",
            task_graph_changes=[
                # Unknown task family should fail spec_guardian
                TaskGraphOp(
                    op=PlanOpType.SET_WEIGHT,
                    task_family="unknown_family",
                    weight=0.5,
                ),
            ],
        )

    def test_patience_not_exceeded_no_noop(self):
        """Test that gate doesn't force NOOP before patience exceeded."""
        config = self._make_policy_config_with_regal(
            enabled_regal_ids=["spec_guardian"],
            patience=3,
            penalty_mode="noop",
        )
        signals = self._make_signals_with_probe()
        failing_plan = self._make_failing_plan()

        # First failure (count=1, patience=3)
        gate_status = check_gates(
            signals,
            config,
            previous_regal_fail_count=0,
            plan=failing_plan,
        )

        # Should NOT force NOOP yet (patience not exceeded)
        assert gate_status.regal_failure_count == 1
        assert not gate_status.regal_forced_noop
        assert not gate_status.forced_noop

    def test_patience_exceeded_forces_noop(self):
        """Test that gate forces NOOP after patience exceeded."""
        config = self._make_policy_config_with_regal(
            enabled_regal_ids=["spec_guardian"],
            patience=3,
            penalty_mode="noop",
        )
        signals = self._make_signals_with_probe()
        failing_plan = self._make_failing_plan()

        # Third failure (count=3, patience=3) - should trigger
        gate_status = check_gates(
            signals,
            config,
            previous_regal_fail_count=2,  # Already 2 failures
            plan=failing_plan,
        )

        # Should force NOOP (patience exceeded)
        assert gate_status.regal_failure_count == 3
        assert gate_status.regal_forced_noop
        assert gate_status.forced_noop
        assert "Regal gate failed" in gate_status.reason

    def test_patience_resets_on_success(self):
        """Test that failure count resets when gates pass."""
        config = self._make_policy_config_with_regal(
            enabled_regal_ids=["spec_guardian"],
            patience=3,
            penalty_mode="noop",
        )
        signals = self._make_signals_with_probe()

        # Valid plan that will pass spec_guardian
        passing_plan = SemanticUpdatePlanV1(
            plan_id="test_passing",
            task_graph_changes=[
                TaskGraphOp(
                    op=PlanOpType.SET_WEIGHT,
                    task_family="manipulation",
                    weight=0.6,
                ),
            ],
        )

        # With previous failures, but plan passes
        gate_status = check_gates(
            signals,
            config,
            previous_regal_fail_count=2,
            plan=passing_plan,
        )

        # Failure count should reset
        assert gate_status.regal_failure_count == 0
        assert not gate_status.regal_forced_noop
        assert not gate_status.forced_noop

    def test_warn_mode_no_noop(self):
        """Test that warn mode doesn't force NOOP even after patience exceeded."""
        config = self._make_policy_config_with_regal(
            enabled_regal_ids=["spec_guardian"],
            patience=3,
            penalty_mode="warn",  # Warn mode
        )
        signals = self._make_signals_with_probe()
        failing_plan = self._make_failing_plan()

        # Third failure in warn mode
        gate_status = check_gates(
            signals,
            config,
            previous_regal_fail_count=2,
            plan=failing_plan,
        )

        # Should NOT force NOOP in warn mode
        assert gate_status.regal_failure_count == 3
        # In warn mode, regal_forced_noop tracks the flag but doesn't set forced_noop
        assert not gate_status.forced_noop

    def test_regal_result_attached_to_gate_status(self):
        """Test that regal evaluation result is attached to gate status."""
        config = self._make_policy_config_with_regal(
            enabled_regal_ids=["spec_guardian", "world_coherence"],
            patience=5,
        )
        signals = self._make_signals_with_probe()

        gate_status = check_gates(
            signals,
            config,
            previous_regal_fail_count=0,
            plan=None,  # No plan - should pass
        )

        # Regal result should be attached
        assert gate_status.regal_result is not None
        assert isinstance(gate_status.regal_result, LedgerRegalV1)
        assert len(gate_status.regal_result.reports) == 2


class TestRegalThresholds:
    """Tests for regal threshold configuration."""

    def test_spec_consistency_threshold(self):
        """Test spec consistency threshold in RegalGatesV1."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian"],
            spec_consistency_min=0.8,
        )

        assert config.spec_consistency_min == 0.8

    def test_coherence_threshold(self):
        """Test coherence threshold in RegalGatesV1."""
        config = RegalGatesV1(
            enabled_regal_ids=["world_coherence"],
            coherence_min=0.7,
        )

        assert config.coherence_min == 0.7

    def test_hack_prob_threshold(self):
        """Test hack probability threshold in RegalGatesV1."""
        config = RegalGatesV1(
            enabled_regal_ids=["reward_integrity"],
            hack_prob_max=0.2,
        )

        assert config.hack_prob_max == 0.2

    def test_per_task_family_overrides(self):
        """Test per-task family threshold overrides."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian"],
            per_task_family_overrides={
                "manipulation": {"spec_consistency_min": 0.9},
                "navigation": {"spec_consistency_min": 0.7},
            },
        )

        assert config.per_task_family_overrides is not None
        assert config.per_task_family_overrides["manipulation"]["spec_consistency_min"] == 0.9


class TestRegalReportStructuredOutput:
    """Tests for structured output fields in RegalReportV1."""

    def test_spec_guardian_structured_output(self):
        """Test SpecGuardian produces structured output."""
        regal = SpecGuardianRegal(seed=42)
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
        )
        plan = SemanticUpdatePlanV1(
            plan_id="test",
            task_graph_changes=[
                TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="manipulation", weight=0.6),
            ],
        )

        report = regal.evaluate(plan, None, config, None)

        # Check structured output fields
        assert hasattr(report, "spec_consistency_score")
        assert hasattr(report, "spec_violations")
        assert isinstance(report.spec_consistency_score, float)
        assert isinstance(report.spec_violations, list)
        assert 0.0 <= report.spec_consistency_score <= 1.0

    def test_world_coherence_structured_output(self):
        """Test WorldCoherence produces structured output."""
        from src.regal.regal_evaluator import WorldCoherenceRegal

        regal = WorldCoherenceRegal(seed=42)
        signals = SignalBundle(
            signals=[
                ControlSignal(
                    signal_type=SignalType.STABILITY,
                    value=0.8,
                    target=0.5,
                ),
            ],
            timestamp="test",
        )

        report = regal.evaluate(None, signals, None, None)

        # Check structured output fields
        assert hasattr(report, "coherence_score")
        assert hasattr(report, "coherence_tags")
        assert isinstance(report.coherence_score, float)
        assert isinstance(report.coherence_tags, list)

    def test_reward_integrity_structured_output(self):
        """Test RewardIntegrity produces structured output."""
        from src.regal.regal_evaluator import RewardIntegrityRegal

        regal = RewardIntegrityRegal(seed=42)
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(full_multiplier=1.5),
            default_weights={"manipulation": 0.5},
        )
        context = {"weight_history": [1.0, 1.1, 1.2, 1.3]}

        report = regal.evaluate(None, None, config, context)

        # Check structured output fields
        assert hasattr(report, "hack_probability")
        assert hasattr(report, "integrity_flags")
        assert isinstance(report.hack_probability, float)
        assert isinstance(report.integrity_flags, list)
        assert 0.0 <= report.hack_probability <= 1.0
