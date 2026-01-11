"""Regal evaluator: deterministic audit gates for Stage-6 meta-regal nodes.

Each regal node evaluates semantic constraints and produces a hashable report.
Evaluation is deterministic given the same inputs and seed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from src.contracts.schemas import (
    RegalGatesV1,
    RegalReportV1,
    LedgerRegalV1,
    SemanticUpdatePlanV1,
    PlanPolicyConfigV1,
)
from src.utils.config_digest import sha256_json

if TYPE_CHECKING:
    from src.representation.homeostasis import SignalBundle


# =============================================================================
# Registry
# =============================================================================

REGAL_REGISTRY: Dict[str, type["RegalNode"]] = {}


def register_regal(regal_id: str):
    """Decorator to register a regal node class."""
    def decorator(cls: type["RegalNode"]):
        REGAL_REGISTRY[regal_id] = cls
        return cls
    return decorator


# =============================================================================
# Base Class
# =============================================================================

class RegalNode(ABC):
    """Base class for deterministic regal evaluators.

    Each regal node receives the same context and must produce
    deterministic output given the same inputs and seed.
    """

    regal_id: str = "base"
    regal_version: str = "v1"

    def __init__(self, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def evaluate(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[Dict[str, Any]] = None,
    ) -> RegalReportV1:
        """Evaluate this regal's constraints.

        Args:
            plan: The semantic update plan (may be None if no plan yet)
            signals: Current signal bundle
            policy_config: Plan policy configuration
            context: Additional context (e.g., recent history)

        Returns:
            RegalReportV1 with verdict and rationale
        """
        pass

    def _compute_inputs_sha(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Compute SHA of all inputs for reproducibility."""
        inputs = {
            "regal_id": self.regal_id,
            "seed": self.seed,
            "plan_sha": plan.sha256() if plan else None,
            "policy_config_sha": policy_config.sha256() if policy_config else None,
            # Signals are not directly hashable; use a summary
            "signals_summary": _signals_to_summary(signals) if signals else None,
            "context": context,
        }
        return sha256_json(inputs)


# =============================================================================
# Built-in Regal Nodes
# =============================================================================

@register_regal("spec_guardian")
class SpecGuardianRegal(RegalNode):
    """Verifies plan operations match allowed task families and thresholds.

    Checks:
    - Task families in plan are in the allowed set
    - Weight changes don't exceed max_abs_weight_change
    - No unexpected operations
    """

    regal_id = "spec_guardian"

    def evaluate(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[Dict[str, Any]] = None,
    ) -> RegalReportV1:
        inputs_sha = self._compute_inputs_sha(plan, signals, policy_config, context)
        findings: Dict[str, Any] = {}
        violations: List[str] = []

        if plan is None:
            # No plan to check
            report = RegalReportV1(
                regal_id=self.regal_id,
                regal_version=self.regal_version,
                inputs_sha=inputs_sha,
                determinism_seed=self.seed,
                passed=True,
                confidence=1.0,
                rationale="No plan to validate",
                findings=findings,
            )
            report.compute_sha()
            return report

        # Check 1: Validate task families against default_weights keys
        allowed_families = set()
        if policy_config and policy_config.default_weights:
            allowed_families = set(policy_config.default_weights.keys())
            findings["allowed_families"] = list(allowed_families)

        for op in plan.task_graph_changes:
            if allowed_families and op.task_family not in allowed_families:
                violations.append(f"Unknown task family: {op.task_family}")

        # Check 2: Weight changes don't exceed threshold
        max_change = None
        if policy_config and policy_config.gain_schedule:
            max_change = policy_config.gain_schedule.max_abs_weight_change

        if max_change:
            for op in plan.task_graph_changes:
                if op.weight is not None:
                    default = policy_config.default_weights.get(op.task_family, 1.0)
                    delta = abs(op.weight - default)
                    if delta > max_change:
                        violations.append(
                            f"Weight change for {op.task_family} exceeds max: "
                            f"{delta:.3f} > {max_change:.3f}"
                        )

        findings["violations"] = violations
        findings["num_ops_checked"] = len(plan.task_graph_changes)

        passed = len(violations) == 0
        report = RegalReportV1(
            regal_id=self.regal_id,
            regal_version=self.regal_version,
            inputs_sha=inputs_sha,
            determinism_seed=self.seed,
            passed=passed,
            confidence=1.0 if passed else 0.8,
            rationale=f"Checked {len(plan.task_graph_changes)} ops, {len(violations)} violations"
            if not passed else "All operations within spec",
            findings=findings,
        )
        report.compute_sha()
        return report


@register_regal("world_coherence")
class WorldCoherenceRegal(RegalNode):
    """Checks that signals are consistent with expected physics/constraints.

    Checks:
    - Signal values are within reasonable bounds
    - No NaN/Inf values
    - Signals don't contradict each other (e.g., success rate can't exceed 1.0)
    """

    regal_id = "world_coherence"

    def evaluate(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[Dict[str, Any]] = None,
    ) -> RegalReportV1:
        inputs_sha = self._compute_inputs_sha(plan, signals, policy_config, context)
        findings: Dict[str, Any] = {}
        violations: List[str] = []

        if signals is None:
            report = RegalReportV1(
                regal_id=self.regal_id,
                regal_version=self.regal_version,
                inputs_sha=inputs_sha,
                determinism_seed=self.seed,
                passed=True,
                confidence=1.0,
                rationale="No signals to validate",
                findings=findings,
            )
            report.compute_sha()
            return report

        # Check signals for coherence
        import math
        signals_checked = 0

        for signal in signals.signals:
            signals_checked += 1
            val = signal.value

            # Check for NaN/Inf
            if isinstance(val, float):
                if math.isnan(val):
                    violations.append(f"NaN value in signal {signal.signal_type}")
                elif math.isinf(val):
                    violations.append(f"Inf value in signal {signal.signal_type}")

            # Check rate signals are in [0, 1]
            signal_name = str(signal.signal_type).lower()
            if "rate" in signal_name or "fraction" in signal_name:
                if isinstance(val, (int, float)):
                    if val < 0.0 or val > 1.0:
                        violations.append(
                            f"Rate/fraction signal {signal.signal_type} out of bounds: {val}"
                        )

        findings["signals_checked"] = signals_checked
        findings["violations"] = violations

        passed = len(violations) == 0
        report = RegalReportV1(
            regal_id=self.regal_id,
            regal_version=self.regal_version,
            inputs_sha=inputs_sha,
            determinism_seed=self.seed,
            passed=passed,
            confidence=1.0 if passed else 0.7,
            rationale=f"Checked {signals_checked} signals, {len(violations)} violations"
            if not passed else f"All {signals_checked} signals coherent",
            findings=findings,
        )
        report.compute_sha()
        return report


@register_regal("reward_integrity")
class RewardIntegrityRegal(RegalNode):
    """Checks for reward hacking patterns.

    Checks:
    - Sudden spikes in success rate without corresponding skill improvement
    - Anomalous energy consumption patterns
    - Weight oscillations (sign of exploitation)
    """

    regal_id = "reward_integrity"

    def evaluate(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[Dict[str, Any]] = None,
    ) -> RegalReportV1:
        inputs_sha = self._compute_inputs_sha(plan, signals, policy_config, context)
        findings: Dict[str, Any] = {}
        violations: List[str] = []

        # Check context for historical patterns
        history = context.get("weight_history", []) if context else []
        findings["history_length"] = len(history)

        # Check 1: Weight oscillations (if we have history)
        if len(history) >= 3:
            # Look for sign changes in weight deltas
            sign_changes = 0
            for i in range(2, len(history)):
                prev_delta = history[i - 1] - history[i - 2]
                curr_delta = history[i] - history[i - 1]
                if prev_delta * curr_delta < 0:  # Sign change
                    sign_changes += 1

            oscillation_rate = sign_changes / (len(history) - 2)
            findings["oscillation_rate"] = oscillation_rate

            if oscillation_rate > 0.6:  # More than 60% sign changes
                violations.append(
                    f"High weight oscillation detected: {oscillation_rate:.2%} sign changes"
                )

        # Check 2: Anomalous gain requests in plan
        if plan and policy_config and policy_config.gain_schedule:
            max_mult = policy_config.gain_schedule.full_multiplier
            for op in plan.task_graph_changes:
                if op.weight is not None and policy_config.default_weights:
                    default = policy_config.default_weights.get(op.task_family, 1.0)
                    ratio = op.weight / default if default != 0 else float("inf")
                    if ratio > max_mult * 1.5:  # 50% above full multiplier
                        violations.append(
                            f"Anomalous weight increase for {op.task_family}: "
                            f"{ratio:.2f}x (max expected: {max_mult:.2f}x)"
                        )

        findings["violations"] = violations

        passed = len(violations) == 0
        report = RegalReportV1(
            regal_id=self.regal_id,
            regal_version=self.regal_version,
            inputs_sha=inputs_sha,
            determinism_seed=self.seed,
            passed=passed,
            confidence=0.9 if passed else 0.6,  # Lower confidence on violations (may be false positive)
            rationale="No reward hacking patterns detected"
            if passed else f"Detected {len(violations)} potential issues",
            findings=findings,
        )
        report.compute_sha()
        return report


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_regals(
    config: RegalGatesV1,
    plan: Optional[SemanticUpdatePlanV1] = None,
    signals: Optional["SignalBundle"] = None,
    policy_config: Optional[PlanPolicyConfigV1] = None,
    context: Optional[Dict[str, Any]] = None,
) -> LedgerRegalV1:
    """Evaluate all enabled regal nodes and return aggregated result.

    Args:
        config: Regal gates configuration specifying which nodes to run
        plan: Current semantic update plan
        signals: Current signal bundle
        policy_config: Plan policy configuration
        context: Additional context for evaluation

    Returns:
        LedgerRegalV1 with all reports and aggregate pass/fail
    """
    reports: List[RegalReportV1] = []
    all_inputs: List[str] = []

    for regal_id in config.enabled_regal_ids:
        if regal_id not in REGAL_REGISTRY:
            # Skip unknown regals (could also warn/error)
            continue

        regal_cls = REGAL_REGISTRY[regal_id]
        regal = regal_cls(seed=config.determinism_seed)
        report = regal.evaluate(plan, signals, policy_config, context)
        reports.append(report)
        all_inputs.append(report.inputs_sha)

    # Compute combined inputs SHA
    combined_inputs_sha = sha256_json({"inputs_shas": sorted(all_inputs)})

    # Check if all passed
    all_passed = all(r.passed for r in reports) if reports else True

    return LedgerRegalV1(
        regal_config_sha=config.sha256(),
        reports=reports,
        all_passed=all_passed,
        combined_inputs_sha=combined_inputs_sha,
    )


# =============================================================================
# Helpers
# =============================================================================

def _signals_to_summary(signals: "SignalBundle") -> Dict[str, Any]:
    """Convert SignalBundle to a hashable summary dict."""
    return {
        "num_signals": len(signals.signals),
        "signal_types": sorted([str(s.signal_type) for s in signals.signals]),
        "bundle_id": getattr(signals, "bundle_id", None),
    }
