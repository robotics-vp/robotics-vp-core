"""Tests for RegalPhaseV1 and phase-aware evaluation, including trajectory audit consumption tests.

Verifies:
- RegalPhaseV1 enum definition and values
- Same phase + same inputs = same report SHA (determinism)
- Different phase + same inputs = different report SHA
- RegalContextV1 strict schema (rejects unknown fields)
- RegalContextV1.sha256() determinism
- TrajectoryAuditV1 consumption by WorldCoherenceRegal (FAIL on physics anomalies)
"""
import pytest

from src.contracts.schemas import (
    RegalPhaseV1,
    RegalContextV1,
    RegalGatesV1,
    RegalReportV1,
    TrajectoryAuditV1,
    SemanticUpdatePlanV1,
    TaskGraphOp,
    PlanOpType,
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
)
from src.regal.regal_evaluator import (
    WorldCoherenceRegal,
    RewardIntegrityRegal,
    SpecGuardianRegal,
    evaluate_regals,
)
from pydantic import ValidationError


class TestRegalPhaseV1:
    """Tests for RegalPhaseV1 enum."""

    def test_phase_enum_values(self):
        """Test RegalPhaseV1 has all expected values."""
        assert RegalPhaseV1.PRE_PLAN.value == "pre_plan"
        assert RegalPhaseV1.POST_PLAN_PRE_APPLY.value == "post_plan_pre_apply"
        assert RegalPhaseV1.POST_APPLY_PRE_TRAIN.value == "post_apply_pre_train"
        assert RegalPhaseV1.DURING_TRAIN.value == "during_train"
        assert RegalPhaseV1.POST_TRAIN_PRE_AUDIT.value == "post_train_pre_audit"
        assert RegalPhaseV1.POST_AUDIT.value == "post_audit"

    def test_phase_count(self):
        """Test all 6 phases exist."""
        assert len(RegalPhaseV1) == 6


class TestPhaseDeterminism:
    """Tests for phase-aware SHA determinism."""

    def test_same_phase_same_sha(self):
        """Same inputs + same phase -> same report sha."""
        regal = WorldCoherenceRegal(seed=42)
        context = RegalContextV1(run_id="test_run", step=1)
        phase = RegalPhaseV1.POST_PLAN_PRE_APPLY

        report1 = regal.evaluate(None, None, None, context, phase)
        report1.compute_sha()
        report2 = regal.evaluate(None, None, None, context, phase)
        report2.compute_sha()

        # inputs_sha should be deterministic
        assert report1.inputs_sha == report2.inputs_sha
        assert report1.phase == report2.phase

    def test_different_phase_different_sha(self):
        """Same inputs, different phase -> different inputs_sha."""
        regal = WorldCoherenceRegal(seed=42)
        context = RegalContextV1(run_id="test_run", step=1)

        report1 = regal.evaluate(None, None, None, context, RegalPhaseV1.POST_PLAN_PRE_APPLY)
        report2 = regal.evaluate(None, None, None, context, RegalPhaseV1.POST_AUDIT)

        # inputs_sha should differ because phase is included
        assert report1.inputs_sha != report2.inputs_sha
        # phase field on report should also differ
        assert report1.phase != report2.phase
        assert report1.phase == RegalPhaseV1.POST_PLAN_PRE_APPLY
        assert report2.phase == RegalPhaseV1.POST_AUDIT


class TestRegalContextV1:
    """Tests for RegalContextV1 typed schema."""

    def test_strict_schema_rejects_unknown(self):
        """Unknown fields rejected by extra='forbid'."""
        with pytest.raises(ValidationError):
            RegalContextV1(run_id="test", unknown_field="bad")

    def test_sha_determinism(self):
        """Same context values -> same sha."""
        ctx1 = RegalContextV1(run_id="test", step=1, plan_sha="abc123")
        ctx2 = RegalContextV1(run_id="test", step=1, plan_sha="abc123")
        assert ctx1.sha256() == ctx2.sha256()

    def test_sha_differs_on_value_change(self):
        """Different context values -> different sha."""
        ctx1 = RegalContextV1(run_id="test", step=1)
        ctx2 = RegalContextV1(run_id="test", step=2)
        assert ctx1.sha256() != ctx2.sha256()

    def test_optional_fields(self):
        """Optional fields can be None."""
        ctx = RegalContextV1(run_id="test")  # Only required field
        assert ctx.step is None
        assert ctx.plan_sha is None
        assert ctx.econ_tensor_sha is None


class TestWorldCoherenceWithTrajectoryAudit:
    """Tests for WorldCoherenceRegal consuming TrajectoryAuditV1."""

    def test_fails_on_velocity_spikes(self):
        """Physics anomaly velocity spikes -> WorldCoherenceRegal must FAIL."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            velocity_spike_count=10,  # >= 5 triggers failure
        )
        regal = WorldCoherenceRegal(seed=42)

        report = regal.evaluate(
            None, None, None, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is False
        assert "velocity_anomaly" in report.coherence_tags
        assert report.findings.get("trajectory_audit_present") is True

    def test_fails_on_penetration(self):
        """High object penetration -> WorldCoherenceRegal must FAIL."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            penetration_max=0.05,  # > 0.01 triggers failure
        )
        regal = WorldCoherenceRegal(seed=42)

        report = regal.evaluate(
            None, None, None, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is False
        assert "physics_violation" in report.coherence_tags

    def test_fails_on_contact_anomalies(self):
        """High contact anomaly count -> WorldCoherenceRegal must FAIL."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            contact_anomaly_count=5,  # >= 3 triggers failure
        )
        regal = WorldCoherenceRegal(seed=42)

        report = regal.evaluate(
            None, None, None, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is False
        assert "contact_anomaly" in report.coherence_tags

    def test_passes_with_normal_audit(self):
        """Normal trajectory audit -> PASS."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            velocity_spike_count=2,  # Below threshold
            contact_anomaly_count=1,  # Below threshold
        )
        regal = WorldCoherenceRegal(seed=42)

        report = regal.evaluate(
            None, None, None, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is True
        assert report.findings.get("trajectory_audit_present") is True

    def test_graceful_fallback_without_audit(self):
        """No trajectory audit -> graceful fallback (PASS)."""
        regal = WorldCoherenceRegal(seed=42)

        report = regal.evaluate(None, None, None, None, RegalPhaseV1.POST_PLAN_PRE_APPLY)

        assert report.passed is True
        assert report.findings.get("trajectory_audit_present") is False


class TestRewardIntegrityWithTrajectoryAudit:
    """Tests for RewardIntegrityRegal consuming TrajectoryAuditV1."""

    def _make_policy_config(self) -> PlanPolicyConfigV1:
        return PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(full_multiplier=1.5),
            default_weights={"manipulation": 0.5},
        )

    def test_flags_extreme_reward_component(self):
        """Extreme reward component -> flag it."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            reward_components={"hack_bonus": 100.0},  # abs > 10 triggers
        )
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        report = regal.evaluate(
            None, None, config, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is False
        assert any("extreme_reward" in flag for flag in report.integrity_flags)

    def test_flags_high_total_return(self):
        """Unusually high total return -> flag it."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            total_return=15.0,  # > 10 triggers
        )
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        report = regal.evaluate(
            None, None, config, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is False
        assert "high_return" in report.integrity_flags

    def test_normal_audit_passes(self):
        """Normal trajectory audit -> PASS."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            total_return=5.0,  # Normal
            reward_components={"base": 2.0, "bonus": 3.0},  # Normal
        )
        regal = RewardIntegrityRegal(seed=42)
        config = self._make_policy_config()

        report = regal.evaluate(
            None, None, config, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is True
        assert report.findings.get("trajectory_audit_present") is True


class TestSpecGuardianWithTrajectoryAudit:
    """Tests for SpecGuardianRegal consuming TrajectoryAuditV1."""

    def test_flags_constraint_violation_event(self):
        """Constraint violation event in trajectory audit -> flag it."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            event_counts={"constraint_violation": 3},
        )
        regal = SpecGuardianRegal(seed=42)

        report = regal.evaluate(
            None, None, None, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is False
        assert any("violation" in v.lower() for v in report.spec_violations)

    def test_flags_violation_event_in_events_list(self):
        """Constraint violation in events list -> flag it."""
        audit = TrajectoryAuditV1(
            episode_id="test_episode",
            num_steps=50,
            events=["collision", "joint_limit_violation"],
        )
        regal = SpecGuardianRegal(seed=42)

        report = regal.evaluate(
            None, None, None, None,
            RegalPhaseV1.POST_PLAN_PRE_APPLY,
            trajectory_audit=audit,
        )

        assert report.passed is False


class TestEvaluateRegalsPhaseAware:
    """Tests for evaluate_regals with explicit phase."""

    def test_reports_include_phase(self):
        """All reports should include the evaluation phase."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian", "world_coherence"],
            determinism_seed=42,
        )

        result = evaluate_regals(
            config=config,
            phase=RegalPhaseV1.POST_AUDIT,
        )

        for report in result.reports:
            assert report.phase == RegalPhaseV1.POST_AUDIT

    def test_phase_affects_inputs_sha(self):
        """Same inputs with different phase -> different combined inputs sha."""
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian"],
            determinism_seed=42,
        )
        context = RegalContextV1(run_id="test")

        result1 = evaluate_regals(
            config=config, phase=RegalPhaseV1.POST_PLAN_PRE_APPLY, context=context
        )
        result2 = evaluate_regals(
            config=config, phase=RegalPhaseV1.POST_AUDIT, context=context
        )

        # Combined inputs sha should differ
        assert result1.combined_inputs_sha != result2.combined_inputs_sha
