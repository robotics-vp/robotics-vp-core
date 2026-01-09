"""Homeostatic controller for representation stability.

This module provides control signals and a one-step planner that consumes
evaluation metrics (epiplexity, stability, alignment error) and produces
actionable plans for data selection, model tuning, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class SignalType(str, Enum):
    """Types of control signals."""

    EPIPLEXITY = "epiplexity"  # Representation quality signal
    STABILITY = "stability"  # Temporal/seed stability
    ALIGNMENT_ERROR = "alignment_error"  # Cross-space alignment error
    DRIFT = "drift"  # Representation drift from baseline
    COVERAGE = "coverage"  # Data coverage metrics
    DELTA_EPI_PER_FLOP = "delta_epi_per_flop"  # Probe discriminator signal


@dataclass
class ControlSignal:
    """Representation of a single control signal.

    Attributes:
        signal_type: Type of signal (epiplexity, stability, etc.)
        value: Current signal value
        target: Target value (setpoint)
        threshold_low: Lower bound trigger
        threshold_high: Upper bound trigger
        metadata: Additional context
    """

    signal_type: SignalType
    value: float
    target: float = 1.0
    threshold_low: float = 0.0
    threshold_high: float = 1.0
    rising_is_bad: bool = False  # True if higher values indicate problems
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error(self) -> float:
        """Compute error from setpoint."""
        return self.value - self.target

    @property
    def normalized_error(self) -> float:
        """Compute error normalized to threshold range."""
        range_size = self.threshold_high - self.threshold_low
        if range_size < 1e-8:
            return 0.0
        return (self.value - self.target) / range_size

    @property
    def is_in_range(self) -> bool:
        """Check if signal is within acceptable range."""
        return self.threshold_low <= self.value <= self.threshold_high

    @property
    def status(self) -> str:
        """Get signal status."""
        if self.is_in_range:
            return "ok"
        if self.value < self.threshold_low:
            return "low"
        return "high"


@dataclass
class SignalBundle:
    """Bundle of control signals with metadata."""

    signals: List[ControlSignal]
    timestamp: str = ""
    episode_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_signal(self, signal_type: SignalType) -> Optional[ControlSignal]:
        """Get signal by type."""
        for signal in self.signals:
            if signal.signal_type == signal_type:
                return signal
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signals": [
                {
                    "signal_type": s.signal_type.value,
                    "value": s.value,
                    "target": s.target,
                    "threshold_low": s.threshold_low,
                    "threshold_high": s.threshold_high,
                    "rising_is_bad": s.rising_is_bad,
                    "status": s.status,
                    "error": s.error,
                    "metadata": s.metadata,
                }
                for s in self.signals
            ],
            "timestamp": self.timestamp,
            "episode_ids": self.episode_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalBundle":
        """Create from dictionary."""
        signals = []
        for s in data.get("signals", []):
            signals.append(
                ControlSignal(
                    signal_type=SignalType(s["signal_type"]),
                    value=s["value"],
                    target=s.get("target", 1.0),
                    threshold_low=s.get("threshold_low", 0.0),
                    threshold_high=s.get("threshold_high", 1.0),
                    rising_is_bad=s.get("rising_is_bad", False),
                    metadata=s.get("metadata", {}),
                )
            )
        return cls(
            signals=signals,
            timestamp=data.get("timestamp", ""),
            episode_ids=data.get("episode_ids", []),
            metadata=data.get("metadata", {}),
        )


class ActionType(str, Enum):
    """Types of controller actions."""

    NOOP = "noop"  # Do nothing
    INCREASE_DATA = "increase_data"  # Request more data for underrepresented slices
    DECREASE_DATA = "decrease_data"  # Reduce overrepresented slices
    RETRAIN = "retrain"  # Trigger model retraining
    REALIGN = "realign"  # Refit isomorphism adapters
    ALERT = "alert"  # Emit alert for human review


@dataclass
class ActionPlan:
    """Plan describing controller actions.

    Attributes:
        actions: List of action types to take
        priority: Priority level (0-10, higher = more urgent)
        rationale: Explanation for the actions
        parameters: Action-specific parameters
        metadata: Additional context
    """

    actions: List[ActionType]
    priority: int = 0
    rationale: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "actions": [a.value for a in self.actions],
            "priority": self.priority,
            "rationale": self.rationale,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionPlan":
        """Create from dictionary."""
        return cls(
            actions=[ActionType(a) for a in data.get("actions", [])],
            priority=data.get("priority", 0),
            rationale=data.get("rationale", ""),
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ControllerConfig:
    """Configuration for the homeostatic controller."""

    # Epiplexity thresholds
    epiplexity_target: float = 0.5
    epiplexity_threshold_low: float = 0.1
    epiplexity_threshold_high: float = 0.9

    # Stability thresholds
    stability_target: float = 0.9
    stability_threshold_low: float = 0.7
    stability_threshold_high: float = 1.0

    # Alignment error thresholds
    alignment_error_target: float = 0.1
    alignment_error_threshold_low: float = 0.0
    alignment_error_threshold_high: float = 0.5

    # Drift thresholds
    drift_target: float = 0.0
    drift_threshold_low: float = 0.0
    drift_threshold_high: float = 0.3

    # Action thresholds
    priority_alert_threshold: int = 7
    priority_retrain_threshold: int = 5


class HomeostaticController:
    """One-step homeostatic controller for representation stability.

    Consumes signal bundles and produces action plans based on configured thresholds.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        """Initialize controller.

        Args:
            config: Controller configuration (uses defaults if None)
        """
        self.config = config or ControllerConfig()
        self._history: List[SignalBundle] = []

    def step(self, signal_bundle: SignalBundle) -> ActionPlan:
        """Execute one control step.

        Args:
            signal_bundle: Current signal readings

        Returns:
            ActionPlan describing recommended actions
        """
        self._history.append(signal_bundle)

        actions: List[ActionType] = []
        rationale_parts: List[str] = []
        parameters: Dict[str, Any] = {}
        max_priority = 0

        # Check each signal type
        for signal in signal_bundle.signals:
            action, priority, reason = self._check_signal(signal)
            if action != ActionType.NOOP:
                actions.append(action)
                rationale_parts.append(reason)
                max_priority = max(max_priority, priority)

        # Check for drift if we have history
        if len(self._history) >= 2:
            drift_action, drift_priority, drift_reason = self._check_drift()
            if drift_action != ActionType.NOOP:
                actions.append(drift_action)
                rationale_parts.append(drift_reason)
                max_priority = max(max_priority, drift_priority)

        # Deduplicate actions
        actions = list(dict.fromkeys(actions))

        # Build plan
        if not actions:
            return ActionPlan(
                actions=[ActionType.NOOP],
                priority=0,
                rationale="All signals within acceptable ranges",
                metadata={"signal_count": len(signal_bundle.signals)},
            )

        return ActionPlan(
            actions=actions,
            priority=max_priority,
            rationale="; ".join(rationale_parts),
            parameters=parameters,
            metadata={
                "signal_count": len(signal_bundle.signals),
                "history_length": len(self._history),
            },
        )

    def _check_signal(self, signal: ControlSignal) -> tuple[ActionType, int, str]:
        """Check a single signal and return recommended action."""
        if signal.is_in_range:
            return ActionType.NOOP, 0, ""

        # Determine severity
        abs_error = abs(signal.normalized_error)
        if abs_error > 0.8:
            priority = 9
        elif abs_error > 0.5:
            priority = 6
        elif abs_error > 0.2:
            priority = 3
        else:
            priority = 1

        # Determine action based on signal type
        if signal.signal_type == SignalType.EPIPLEXITY:
            if signal.value < signal.threshold_low:
                return (
                    ActionType.INCREASE_DATA,
                    priority,
                    f"Epiplexity too low ({signal.value:.3f} < {signal.threshold_low})",
                )
            return (
                ActionType.RETRAIN,
                priority,
                f"Epiplexity too high ({signal.value:.3f} > {signal.threshold_high})",
            )

        if signal.signal_type == SignalType.STABILITY:
            return (
                ActionType.REALIGN,
                priority,
                f"Stability below threshold ({signal.value:.3f} < {signal.threshold_low})",
            )

        if signal.signal_type == SignalType.ALIGNMENT_ERROR:
            if signal.value > signal.threshold_high:
                return (
                    ActionType.REALIGN,
                    priority,
                    f"Alignment error too high ({signal.value:.3f} > {signal.threshold_high})",
                )
            return ActionType.NOOP, 0, ""

        if signal.signal_type == SignalType.COVERAGE:
            if signal.value < signal.threshold_low:
                return (
                    ActionType.INCREASE_DATA,
                    priority,
                    f"Coverage too low ({signal.value:.3f} < {signal.threshold_low})",
                )
            return ActionType.NOOP, 0, ""

        # Default for unknown signals
        if priority >= self.config.priority_alert_threshold:
            return ActionType.ALERT, priority, f"Signal {signal.signal_type.value} out of range"

        return ActionType.NOOP, 0, ""

    def _check_drift(self) -> tuple[ActionType, int, str]:
        """Check for drift between recent signal bundles."""
        if len(self._history) < 2:
            return ActionType.NOOP, 0, ""

        # Compare last two bundles for epiplexity drift
        prev = self._history[-2]
        curr = self._history[-1]

        prev_epi = prev.get_signal(SignalType.EPIPLEXITY)
        curr_epi = curr.get_signal(SignalType.EPIPLEXITY)

        if prev_epi is None or curr_epi is None:
            return ActionType.NOOP, 0, ""

        drift = abs(curr_epi.value - prev_epi.value)
        if drift > self.config.drift_threshold_high:
            priority = 6 if drift > 0.5 else 3
            return (
                ActionType.ALERT,
                priority,
                f"Epiplexity drift detected ({drift:.3f} > {self.config.drift_threshold_high})",
            )

        return ActionType.NOOP, 0, ""

    def reset_history(self) -> None:
        """Clear controller history."""
        self._history.clear()


def build_signal_bundle_from_leaderboard(
    leaderboard_summaries: Dict[str, Any],
    slice_id: str,
    episode_ids: Optional[List[str]] = None,
) -> SignalBundle:
    """Build signal bundle from leaderboard evaluation results.

    Args:
        leaderboard_summaries: Dict from curated slices leaderboard
        slice_id: Slice identifier (e.g., "occluded", "dynamic", "static")
        episode_ids: Optional list of episode IDs

    Returns:
        SignalBundle containing control signals
    """
    from datetime import datetime

    signals: List[ControlSignal] = []

    # Extract epiplexity signal (average across representations)
    variances = []
    for repr_id, summary in leaderboard_summaries.items():
        if isinstance(summary, dict):
            if "variance" in summary:
                variances.append(summary["variance"])
            elif "S_T_proxy" in summary:
                variances.append(summary["S_T_proxy"])

    if variances:
        avg_variance = float(np.mean(variances))
        signals.append(
            ControlSignal(
                signal_type=SignalType.EPIPLEXITY,
                value=avg_variance,
                target=0.5,
                threshold_low=0.1,
                threshold_high=0.9,
                metadata={"slice_id": slice_id, "num_reprs": len(variances)},
            )
        )

    # Extract stability signal (based on consistency across representations)
    if len(variances) >= 2:
        variance_std = float(np.std(variances))
        stability = max(0.0, 1.0 - variance_std)
        signals.append(
            ControlSignal(
                signal_type=SignalType.STABILITY,
                value=stability,
                target=0.9,
                threshold_low=0.7,
                threshold_high=1.0,
                metadata={"slice_id": slice_id, "variance_std": variance_std},
            )
        )

    # Extract coverage signal (based on number of episodes)
    num_episodes = 0
    for repr_id, summary in leaderboard_summaries.items():
        if isinstance(summary, dict) and "num_episodes" in summary:
            num_episodes = max(num_episodes, summary["num_episodes"])

    if num_episodes > 0:
        # Normalize coverage (assuming 100 episodes is full coverage)
        coverage = min(1.0, num_episodes / 100.0)
        signals.append(
            ControlSignal(
                signal_type=SignalType.COVERAGE,
                value=coverage,
                target=0.8,
                threshold_low=0.3,
                threshold_high=1.0,
                metadata={"slice_id": slice_id, "num_episodes": num_episodes},
            )
        )

    return SignalBundle(
        signals=signals,
        timestamp=datetime.now().isoformat(),
        episode_ids=episode_ids or [],
        metadata={"slice_id": slice_id, "source": "leaderboard"},
    )


__all__ = [
    "SignalType",
    "ControlSignal",
    "SignalBundle",
    "ActionType",
    "ActionPlan",
    "ControllerConfig",
    "HomeostaticController",
    "build_signal_bundle_from_leaderboard",
]
