"""Tests for homeostatic controller."""
import numpy as np
import pytest

from src.representation.homeostasis import (
    SignalType,
    ControlSignal,
    SignalBundle,
    ActionType,
    ActionPlan,
    ControllerConfig,
    HomeostaticController,
    build_signal_bundle_from_leaderboard,
)


class TestControlSignal:
    """Tests for ControlSignal."""

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = ControlSignal(
            signal_type=SignalType.EPIPLEXITY,
            value=0.5,
            target=0.5,
            threshold_low=0.1,
            threshold_high=0.9,
        )
        assert signal.value == 0.5
        assert signal.error == 0.0
        assert signal.is_in_range
        assert signal.status == "ok"

    def test_signal_out_of_range_low(self):
        """Test signal below threshold."""
        signal = ControlSignal(
            signal_type=SignalType.EPIPLEXITY,
            value=0.05,
            target=0.5,
            threshold_low=0.1,
            threshold_high=0.9,
        )
        assert not signal.is_in_range
        assert signal.status == "low"

    def test_signal_out_of_range_high(self):
        """Test signal above threshold."""
        signal = ControlSignal(
            signal_type=SignalType.STABILITY,
            value=1.5,
            target=0.9,
            threshold_low=0.7,
            threshold_high=1.0,
        )
        assert not signal.is_in_range
        assert signal.status == "high"


class TestSignalBundle:
    """Tests for SignalBundle."""

    def test_bundle_creation(self):
        """Test signal bundle creation."""
        signals = [
            ControlSignal(SignalType.EPIPLEXITY, value=0.5),
            ControlSignal(SignalType.STABILITY, value=0.9),
        ]
        bundle = SignalBundle(signals=signals, episode_ids=["ep1", "ep2"])
        assert len(bundle.signals) == 2
        assert len(bundle.episode_ids) == 2

    def test_get_signal(self):
        """Test getting signal by type."""
        signals = [
            ControlSignal(SignalType.EPIPLEXITY, value=0.5),
            ControlSignal(SignalType.STABILITY, value=0.9),
        ]
        bundle = SignalBundle(signals=signals)
        epi = bundle.get_signal(SignalType.EPIPLEXITY)
        assert epi is not None
        assert epi.value == 0.5

        missing = bundle.get_signal(SignalType.DRIFT)
        assert missing is None

    def test_serialization(self):
        """Test to_dict and from_dict."""
        signals = [
            ControlSignal(SignalType.EPIPLEXITY, value=0.5, target=0.5),
        ]
        bundle = SignalBundle(
            signals=signals,
            timestamp="2024-01-01T00:00:00",
            episode_ids=["ep1"],
            metadata={"test": True},
        )
        d = bundle.to_dict()
        assert "signals" in d
        assert d["timestamp"] == "2024-01-01T00:00:00"

        restored = SignalBundle.from_dict(d)
        assert len(restored.signals) == 1
        assert restored.signals[0].value == 0.5
        assert restored.episode_ids == ["ep1"]


class TestHomeostaticController:
    """Tests for HomeostaticController."""

    def test_noop_when_signals_ok(self):
        """Test that controller returns NOOP when all signals are in range."""
        controller = HomeostaticController()
        signals = [
            ControlSignal(
                SignalType.EPIPLEXITY,
                value=0.5,
                threshold_low=0.1,
                threshold_high=0.9,
            ),
            ControlSignal(
                SignalType.STABILITY,
                value=0.9,
                threshold_low=0.7,
                threshold_high=1.0,
            ),
        ]
        bundle = SignalBundle(signals=signals)
        plan = controller.step(bundle)

        assert ActionType.NOOP in plan.actions
        assert plan.priority == 0
        assert "acceptable" in plan.rationale.lower()

    def test_increase_data_on_low_epiplexity(self):
        """Test that INCREASE_DATA is recommended for low epiplexity."""
        controller = HomeostaticController()
        signals = [
            ControlSignal(
                SignalType.EPIPLEXITY,
                value=0.05,  # Very low
                threshold_low=0.1,
                threshold_high=0.9,
            ),
        ]
        bundle = SignalBundle(signals=signals)
        plan = controller.step(bundle)

        assert ActionType.INCREASE_DATA in plan.actions
        assert plan.priority > 0
        assert "epiplexity" in plan.rationale.lower()

    def test_retrain_on_high_epiplexity(self):
        """Test that RETRAIN is recommended for high epiplexity."""
        controller = HomeostaticController()
        signals = [
            ControlSignal(
                SignalType.EPIPLEXITY,
                value=0.95,  # Very high
                threshold_low=0.1,
                threshold_high=0.9,
            ),
        ]
        bundle = SignalBundle(signals=signals)
        plan = controller.step(bundle)

        assert ActionType.RETRAIN in plan.actions
        assert "epiplexity" in plan.rationale.lower()

    def test_realign_on_low_stability(self):
        """Test that REALIGN is recommended for low stability."""
        controller = HomeostaticController()
        signals = [
            ControlSignal(
                SignalType.STABILITY,
                value=0.5,  # Low stability
                threshold_low=0.7,
                threshold_high=1.0,
            ),
        ]
        bundle = SignalBundle(signals=signals)
        plan = controller.step(bundle)

        assert ActionType.REALIGN in plan.actions
        assert "stability" in plan.rationale.lower()

    def test_drift_detection(self):
        """Test drift detection across multiple steps."""
        controller = HomeostaticController()

        # First step with normal epiplexity
        bundle1 = SignalBundle(
            signals=[ControlSignal(SignalType.EPIPLEXITY, value=0.5)]
        )
        plan1 = controller.step(bundle1)

        # Second step with drastically different epiplexity
        bundle2 = SignalBundle(
            signals=[ControlSignal(SignalType.EPIPLEXITY, value=0.9)]
        )
        plan2 = controller.step(bundle2)

        assert ActionType.ALERT in plan2.actions
        assert "drift" in plan2.rationale.lower()

    def test_reset_history(self):
        """Test history reset."""
        controller = HomeostaticController()
        bundle = SignalBundle(signals=[ControlSignal(SignalType.EPIPLEXITY, value=0.5)])
        controller.step(bundle)
        assert len(controller._history) == 1

        controller.reset_history()
        assert len(controller._history) == 0


class TestActionPlan:
    """Tests for ActionPlan."""

    def test_serialization(self):
        """Test to_dict and from_dict."""
        plan = ActionPlan(
            actions=[ActionType.RETRAIN, ActionType.REALIGN],
            priority=5,
            rationale="Test plan",
            parameters={"lr": 0.001},
        )
        d = plan.to_dict()
        assert d["actions"] == ["retrain", "realign"]
        assert d["priority"] == 5

        restored = ActionPlan.from_dict(d)
        assert ActionType.RETRAIN in restored.actions
        assert restored.priority == 5


class TestBuildSignalBundle:
    """Tests for build_signal_bundle_from_leaderboard."""

    def test_build_from_token_only_leaderboard(self):
        """Test building signal bundle from token-only leaderboard output."""
        leaderboard = {
            "vision_rgb": {
                "status": "ok",
                "num_episodes": 10,
                "variance": 0.5,
                "S_T_proxy": 0.5,
            },
            "geometry_bev": {
                "status": "ok",
                "num_episodes": 10,
                "variance": 0.4,
                "S_T_proxy": 0.4,
            },
        }
        bundle = build_signal_bundle_from_leaderboard(
            leaderboard, "dynamic", episode_ids=["ep1", "ep2"]
        )

        assert len(bundle.signals) >= 1
        epi = bundle.get_signal(SignalType.EPIPLEXITY)
        assert epi is not None
        assert 0.4 <= epi.value <= 0.5  # Average of 0.4 and 0.5

        coverage = bundle.get_signal(SignalType.COVERAGE)
        assert coverage is not None
        assert coverage.value == 0.1  # 10/100
