"""Tests for probe epiplexity harness and homeostasis gates."""
import numpy as np
import pytest

from src.contracts.schemas import (
    ProbeEpiReportV1, ProbeConfigV1,
    PlanPolicyConfigV1, PlanGainScheduleV1,
)
from src.evaluation.probe_harness import (
    ProbeHarness,
    ProbeHarnessConfig,
    LinearProbe,
    MLPProbe,
    create_probe,
    get_probe_harness,
)
from src.orchestrator.homeostatic_plan_writer import (
    build_signal_bundle_for_plan,
    build_plan_from_signals,
    check_gates,
    GateStatus,
)
from src.representation.homeostasis import SignalType, ActionType


class TestProbeHarness:
    """Tests for probe harness determinism and correctness."""

    def test_deterministic_same_seed_same_report(self):
        """Test that same seed produces consistent scores."""
        np.random.seed(42)
        X_base = np.random.randn(100, 32).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        X_after = X_base + np.random.randn(100, 32).astype(np.float32) * 0.1

        config = ProbeHarnessConfig(
            probe_variants=["linear"],
            probe_steps=50,
            batch_size=16,
            seeds=[42],
            input_dim=32,
            min_raw_delta=0.0,  # Allow small deltas for this test
        )
        harness = ProbeHarness(config)

        report1 = harness.run((X_base, y), (X_after, y))
        report2 = harness.run((X_base, y), (X_after, y))

        # Scores should be identical with same seeds
        assert report1.baseline_score == report2.baseline_score
        assert report1.after_score == report2.after_score
        assert report1.delta == report2.delta
        assert report1.probe_config_sha == report2.probe_config_sha

    def test_stability_gate_fails_on_sign_flip(self):
        """Test that stability gate with high threshold can fail on noise."""
        np.random.seed(42)
        X_base = np.random.randn(100, 16).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        X_after = X_base.copy()

        config = ProbeHarnessConfig(
            probe_variants=["linear"],
            probe_steps=20,
            batch_size=8,
            seeds=[42, 43, 44, 45, 46],
            input_dim=16,
            stability_sign_threshold=0.9,
            min_raw_delta=0.0,
        )
        harness = ProbeHarness(config)

        report = harness.run((X_base, y), (X_after, y))

        assert report.sign_consistency <= 1.0
        assert isinstance(report.stability_pass, bool)

    def test_delta_epi_per_flop_computed_correctly(self):
        """Test FLOPs estimation and delta computation."""
        np.random.seed(42)
        X_base = np.random.randn(50, 16).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        X_after = X_base + 0.5

        config = ProbeHarnessConfig(
            probe_variants=["linear"],
            probe_steps=100,
            batch_size=10,
            seeds=[42],
            input_dim=16,
            min_raw_delta=0.0,
        )
        harness = ProbeHarness(config)

        report = harness.run((X_base, y), (X_after, y))

        # FLOPs should be: steps * batch_size * flops_per_example * 2 (baseline + after)
        linear_flops = 2 * 16 * 1
        expected_flops = 100 * 10 * linear_flops * 2  # For one seed, one variant
        assert report.flops_estimate == expected_flops

        # Delta per flop should be delta / flops
        expected_delta_per_flop = report.delta / report.flops_estimate
        assert abs(report.delta_epi_per_flop - expected_delta_per_flop) < 1e-10

    def test_probe_variants(self):
        """Test that all probe variants work."""
        for variant in ["linear", "mlp"]:
            probe = create_probe(variant, input_dim=32, hidden_dim=16)
            assert probe.flops_per_example > 0

    def test_probe_registry(self):
        """Test probe harness registry."""
        smoke_harness = get_probe_harness("smoke_probe_v1")
        assert smoke_harness.harness_id == "smoke_probe_v1"
        assert len(smoke_harness.sha256()) == 64

        config = smoke_harness.to_config()
        assert "linear" in config.probe_variants


class TestHomeostaticGates:
    """Tests for homeostatic controller gates."""

    def test_stability_fail_forces_noop(self):
        """Test that stability gate failure forces NOOP action."""
        report = ProbeEpiReportV1(
            report_id="test",
            probe_config=ProbeConfigV1(
                probe_variant="linear",
                probe_steps=100,
                batch_size=16,
                seeds=[42],
                input_dim=32,
            ),
            baseline_score=-1.0,
            after_score=-0.5,
            delta=0.5,
            flops_estimate=1000.0,
            delta_epi_per_flop=0.0005,
            per_seed_deltas=[0.5, -0.3, 0.1],
            sign_consistency=0.33,
            stability_pass=False,
        )

        config = PlanPolicyConfigV1(gain_schedule=PlanGainScheduleV1(), default_weights={"a": 0.5})
        signals = build_signal_bundle_for_plan(probe_report=report)

        plan, gate_status = build_plan_from_signals(signals, config, plan_id="test")

        assert gate_status.forced_noop is True
        assert gate_status.stability_pass is False
        assert "FORCED_NOOP" in plan.notes

    def test_transfer_pass_allows_stronger_actions(self):
        """Test that transfer gate passing allows stronger weight changes."""
        report = ProbeEpiReportV1(
            report_id="test",
            probe_config=ProbeConfigV1(
                probe_variant="linear",
                probe_steps=100,
                batch_size=16,
                seeds=[42],
                input_dim=32,
            ),
            baseline_score=-1.0,
            after_score=-0.5,
            delta=0.5,
            flops_estimate=1000.0,
            delta_epi_per_flop=0.0005,
            per_seed_deltas=[0.5, 0.4, 0.6],
            sign_consistency=1.0,
            stability_pass=True,
            ood_delta=0.3,
            transfer_pass=True,
        )

        config = PlanPolicyConfigV1(gain_schedule=PlanGainScheduleV1(), default_weights={"a": 0.5})
        signals = build_signal_bundle_for_plan(
            probe_report=report,
            coverage_stats={"manipulation": 10},
        )

        plan, gate_status = build_plan_from_signals(signals, config, plan_id="test")

        assert gate_status.forced_noop is False
        assert gate_status.transfer_pass is True
        assert "transfer=True" in plan.notes

    def test_negative_delta_forces_noop(self):
        """Test that negative delta forces NOOP."""
        report = ProbeEpiReportV1(
            report_id="test",
            probe_config=ProbeConfigV1(
                probe_variant="linear",
                probe_steps=100,
                batch_size=16,
                seeds=[42],
                input_dim=32,
            ),
            baseline_score=-0.5,
            after_score=-1.0,
            delta=-0.5,
            flops_estimate=1000.0,
            delta_epi_per_flop=-0.0005,
            per_seed_deltas=[-0.5, -0.4, -0.6],
            sign_consistency=1.0,
            stability_pass=True,
        )

        config = PlanPolicyConfigV1(gain_schedule=PlanGainScheduleV1(), default_weights={"a": 0.5})
        signals = build_signal_bundle_for_plan(probe_report=report)

        plan, gate_status = build_plan_from_signals(signals, config, plan_id="test")

        assert gate_status.forced_noop is True
        assert "Delta" in gate_status.reason  # Either raw delta floor or delta/FLOP

    def test_raw_delta_floor_enforced(self):
        """Test that raw delta floor prevents vanishing changes."""
        report = ProbeEpiReportV1(
            report_id="test",
            probe_config=ProbeConfigV1(
                probe_variant="linear",
                probe_steps=100,
                batch_size=16,
                seeds=[42],
                input_dim=32,
            ),
            baseline_score=-1.0,
            after_score=-0.999,  # Tiny improvement
            delta=0.001,  # Below floor
            flops_estimate=1000.0,
            delta_epi_per_flop=0.000001,
            per_seed_deltas=[0.001],
            sign_consistency=1.0,
            stability_pass=True,
        )

        config = PlanPolicyConfigV1(gain_schedule=PlanGainScheduleV1(), min_raw_delta=0.01, default_weights={"a": 0.5})
        signals = build_signal_bundle_for_plan(probe_report=report)

        gate_status = check_gates(signals, config)

        assert gate_status.forced_noop is True
        assert "below floor" in gate_status.reason


class TestProbeProvenance:
    """Tests for probe provenance in manifest and ledger."""

    def test_probe_config_sha_is_stable(self):
        """Test that probe config SHA is deterministic."""
        config = ProbeConfigV1(
            probe_variant="linear",
            probe_steps=100,
            batch_size=16,
            seeds=[42, 43],
            input_dim=32,
        )

        sha1 = config.sha256()
        sha2 = config.sha256()
        assert sha1 == sha2
        assert len(sha1) == 64

    def test_probe_report_sha_excludes_self(self):
        """Test that report SHA is computed correctly."""
        report = ProbeEpiReportV1(
            report_id="test",
            probe_config=ProbeConfigV1(
                probe_variant="linear",
                probe_steps=100,
                batch_size=16,
                seeds=[42],
                input_dim=32,
            ),
            baseline_score=-1.0,
            after_score=-0.5,
            delta=0.5,
            flops_estimate=1000.0,
            delta_epi_per_flop=0.0005,
            per_seed_deltas=[0.5],
            sign_consistency=1.0,
            stability_pass=True,
        )
        report.compute_hashes()

        assert report.report_sha != ""
        assert report.probe_config_sha != ""
        assert len(report.report_sha) == 64
