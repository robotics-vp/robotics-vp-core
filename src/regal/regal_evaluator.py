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
    RegalPhaseV1,
    RegalContextV1,
    LedgerRegalV1,
    SemanticUpdatePlanV1,
    PlanPolicyConfigV1,
    TrajectoryAuditV1,
    EconTensorV1,
    SelectionManifestV1,
    OrchestratorStateV1,
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
        context: Optional[RegalContextV1] = None,
        phase: RegalPhaseV1 = RegalPhaseV1.POST_PLAN_PRE_APPLY,
        trajectory_audit: Optional[TrajectoryAuditV1] = None,
        econ_tensor: Optional[EconTensorV1] = None,
        selection_manifest: Optional[SelectionManifestV1] = None,
        orchestrator_state: Optional[OrchestratorStateV1] = None,
    ) -> RegalReportV1:
        """Evaluate this regal's constraints.

        Args:
            plan: The semantic update plan (may be None if no plan yet)
            signals: Current signal bundle
            policy_config: Plan policy configuration
            context: Typed regal context (replaces Dict[str, Any])
            phase: Temporal phase for this evaluation
            trajectory_audit: Episode trajectory audit for substrate grounding
            econ_tensor: Econ tensor for economic metric validation
            selection_manifest: Selection manifest for datapack provenance
            orchestrator_state: Orchestrator state for loop memory/actuation trace

        Returns:
            RegalReportV1 with verdict and rationale
        """
        pass

    def _compute_inputs_sha(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[RegalContextV1],
        phase: RegalPhaseV1,
        trajectory_audit: Optional[TrajectoryAuditV1],
        econ_tensor: Optional[EconTensorV1],
        selection_manifest: Optional[SelectionManifestV1] = None,
        orchestrator_state: Optional[OrchestratorStateV1] = None,
    ) -> str:
        """Compute SHA of all inputs for reproducibility."""
        inputs = {
            "regal_id": self.regal_id,
            "seed": self.seed,
            "phase": phase.value,
            "plan_sha": plan.sha256() if plan else None,
            "policy_config_sha": policy_config.sha256() if policy_config else None,
            # Signals are not directly hashable; use a summary
            "signals_summary": _signals_to_summary(signals) if signals else None,
            "context_sha": context.sha256() if context else None,
            "trajectory_audit_sha": trajectory_audit.sha256() if trajectory_audit else None,
            "econ_tensor_sha": econ_tensor.sha256() if econ_tensor else None,
            "selection_manifest_sha": selection_manifest.sha256() if selection_manifest else None,
            "orchestrator_state_sha": orchestrator_state.sha256() if orchestrator_state else None,
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
    - Quarantined datapacks never selected (using selection_manifest)
    """

    regal_id = "spec_guardian"

    def evaluate(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[RegalContextV1] = None,
        phase: RegalPhaseV1 = RegalPhaseV1.POST_PLAN_PRE_APPLY,
        trajectory_audit: Optional[TrajectoryAuditV1] = None,
        econ_tensor: Optional[EconTensorV1] = None,
        selection_manifest: Optional[SelectionManifestV1] = None,
        orchestrator_state: Optional[OrchestratorStateV1] = None,
    ) -> RegalReportV1:
        inputs_sha = self._compute_inputs_sha(
            plan, signals, policy_config, context, phase, trajectory_audit, econ_tensor,
            selection_manifest, orchestrator_state
        )
        findings: Dict[str, Any] = {}
        violations: List[str] = []

        # NEW: Check quarantine-not-selected invariant
        if selection_manifest is not None:
            findings["selection_manifest_present"] = True
            findings["selected_datapack_count"] = len(selection_manifest.selected_datapack_ids)
            findings["quarantined_count"] = len(selection_manifest.quarantined_datapack_ids)
            
            # CRITICAL: Quarantined IDs must never be selected
            selected_set = set(selection_manifest.selected_datapack_ids)
            quarantined_set = set(selection_manifest.quarantined_datapack_ids)
            forbidden_selected = selected_set & quarantined_set
            
            if forbidden_selected:
                violations.append(
                    f"CRITICAL: Quarantined datapacks selected: {sorted(forbidden_selected)}"
                )
                findings["quarantine_violation"] = True
                findings["forbidden_selected_ids"] = sorted(forbidden_selected)
            else:
                findings["quarantine_violation"] = False
        else:
            findings["selection_manifest_present"] = False

        # Track trajectory audit inspection
        if trajectory_audit is not None:
            findings["trajectory_audit_present"] = True
            # Check for constraint violation events
            if trajectory_audit.event_counts:
                for event, count in trajectory_audit.event_counts.items():
                    if "violation" in event.lower() and count > 0:
                        violations.append(f"Constraint violation event: {event}={count}")
            # Check events list for constraint violations
            if trajectory_audit.events:
                for event in trajectory_audit.events:
                    if "violation" in event.lower():
                        violations.append(f"Constraint violation: {event}")
        else:
            findings["trajectory_audit_present"] = False

        if plan is None:
            # No plan to check
            report = RegalReportV1(
                regal_id=self.regal_id,
                phase=phase,
                regal_version=self.regal_version,
                inputs_sha=inputs_sha,
                determinism_seed=self.seed,
                passed=len(violations) == 0,
                confidence=1.0,
                rationale="No plan to validate" if len(violations) == 0 else f"{len(violations)} trajectory violations",
                spec_consistency_score=1.0 if len(violations) == 0 else 0.5,
                spec_violations=violations,
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

        # Compute spec consistency score (1.0 = perfect, decreases with violations)
        num_ops = len(plan.task_graph_changes)
        spec_consistency_score = 1.0 - (len(violations) / max(num_ops, 1)) if num_ops > 0 else 1.0
        spec_consistency_score = max(0.0, spec_consistency_score)

        passed = len(violations) == 0
        report = RegalReportV1(
            regal_id=self.regal_id,
            phase=phase,
            regal_version=self.regal_version,
            inputs_sha=inputs_sha,
            determinism_seed=self.seed,
            passed=passed,
            confidence=1.0 if passed else 0.8,
            rationale=f"Checked {num_ops} ops, {len(violations)} violations"
            if not passed else "All operations within spec",
            spec_consistency_score=spec_consistency_score,
            spec_violations=violations,
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
        context: Optional[RegalContextV1] = None,
        phase: RegalPhaseV1 = RegalPhaseV1.POST_PLAN_PRE_APPLY,
        trajectory_audit: Optional[TrajectoryAuditV1] = None,
        econ_tensor: Optional[EconTensorV1] = None,
    ) -> RegalReportV1:
        inputs_sha = self._compute_inputs_sha(plan, signals, policy_config, context, phase, trajectory_audit, econ_tensor)
        findings: Dict[str, Any] = {}
        violations: List[str] = []
        coherence_tags: List[str] = []

        # Check trajectory audit for physics anomalies (CRITICAL: can fail solely on this)
        if trajectory_audit is not None:
            findings["trajectory_audit_present"] = True
            
            # Get configurable thresholds from policy_config.regal_gates (or use safe defaults)
            vel_threshold = 5
            pen_threshold = 0.01
            contact_threshold = 3
            if policy_config and policy_config.regal_gates:
                vel_threshold = policy_config.regal_gates.velocity_spike_threshold
                pen_threshold = policy_config.regal_gates.penetration_max_threshold
                contact_threshold = policy_config.regal_gates.contact_anomaly_threshold
            findings["thresholds_used"] = {
                "velocity_spike": vel_threshold,
                "penetration_max": pen_threshold,
                "contact_anomaly": contact_threshold,
            }
            
            # Physics anomaly: velocity spikes
            if trajectory_audit.velocity_spike_count >= vel_threshold:
                violations.append(f"High velocity spikes: {trajectory_audit.velocity_spike_count} >= {vel_threshold}")
                coherence_tags.append("velocity_anomaly")
            # Physics anomaly: object penetration
            if trajectory_audit.penetration_max and trajectory_audit.penetration_max > pen_threshold:
                violations.append(f"Object penetration exceeded threshold: {trajectory_audit.penetration_max:.4f} > {pen_threshold}")
                coherence_tags.append("physics_violation")
            # Physics anomaly: contact anomalies
            if trajectory_audit.contact_anomaly_count >= contact_threshold:
                violations.append(f"Contact anomalies: {trajectory_audit.contact_anomaly_count} >= {contact_threshold}")
                coherence_tags.append("contact_anomaly")
            # State bounds violations
            if trajectory_audit.state_bounds:
                for state_name, bounds in trajectory_audit.state_bounds.items():
                    if len(bounds) == 2 and bounds[1] - bounds[0] > 100.0:
                        violations.append(f"State '{state_name}' has extreme range: [{bounds[0]:.1f}, {bounds[1]:.1f}]")
                        coherence_tags.append("state_bounds_violation")
        else:
            findings["trajectory_audit_present"] = False

        if signals is None and len(violations) == 0:
            report = RegalReportV1(
                regal_id=self.regal_id,
                phase=phase,
                regal_version=self.regal_version,
                inputs_sha=inputs_sha,
                determinism_seed=self.seed,
                passed=True,
                confidence=1.0,
                rationale="No signals to validate",
                coherence_score=1.0,
                coherence_tags=[],
                findings=findings,
            )
            report.compute_sha()
            return report

        # Check signals for coherence
        import math
        signals_checked = 0

        if signals is not None:
            for signal in signals.signals:
                signals_checked += 1
                val = signal.value

                # Check for NaN/Inf
                if isinstance(val, float):
                    if math.isnan(val):
                        violations.append(f"NaN value in signal {signal.signal_type}")
                        coherence_tags.append("nan_value")
                    elif math.isinf(val):
                        violations.append(f"Inf value in signal {signal.signal_type}")
                        coherence_tags.append("inf_value")

                # Check rate signals are in [0, 1]
                signal_name = str(signal.signal_type).lower()
                if "rate" in signal_name or "fraction" in signal_name:
                    if isinstance(val, (int, float)):
                        if val < 0.0 or val > 1.0:
                            violations.append(
                                f"Rate/fraction signal {signal.signal_type} out of bounds: {val}"
                            )
                            coherence_tags.append("bounds_violation")

        findings["signals_checked"] = signals_checked
        findings["violations"] = violations

        # Compute coherence score (1.0 = perfect, decreases with violations)
        total_checks = max(signals_checked, 1) + (1 if trajectory_audit else 0)
        coherence_score = 1.0 - (len(violations) / max(total_checks, 1))
        coherence_score = max(0.0, coherence_score)

        passed = len(violations) == 0
        report = RegalReportV1(
            regal_id=self.regal_id,
            phase=phase,
            regal_version=self.regal_version,
            inputs_sha=inputs_sha,
            determinism_seed=self.seed,
            passed=passed,
            confidence=1.0 if passed else 0.7,
            rationale=f"Checked {signals_checked} signals, {len(violations)} violations"
            if not passed else f"All {signals_checked} signals coherent",
            coherence_score=coherence_score,
            coherence_tags=list(set(coherence_tags)),  # Deduplicate
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
    - Econ tensor anomalies (if available)
    - Orchestrator state oscillation patterns (patience, clamps, knob deltas)
    """

    regal_id = "reward_integrity"

    def evaluate(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[RegalContextV1] = None,
        phase: RegalPhaseV1 = RegalPhaseV1.POST_PLAN_PRE_APPLY,
        trajectory_audit: Optional[TrajectoryAuditV1] = None,
        econ_tensor: Optional[EconTensorV1] = None,
        selection_manifest: Optional[SelectionManifestV1] = None,
        orchestrator_state: Optional[OrchestratorStateV1] = None,
    ) -> RegalReportV1:
        inputs_sha = self._compute_inputs_sha(
            plan, signals, policy_config, context, phase, trajectory_audit, econ_tensor,
            selection_manifest, orchestrator_state
        )
        findings: Dict[str, Any] = {}
        violations: List[str] = []
        integrity_flags: List[str] = []
        hack_indicators = 0

        # NEW: Orchestrator state oscillation detection
        if orchestrator_state is not None:
            findings["orchestrator_state_present"] = True
            findings["orchestrator_step"] = orchestrator_state.step
            findings["total_failures"] = orchestrator_state.total_failures
            findings["total_clamps"] = orchestrator_state.total_clamps
            
            # Check 1a: Rapid failure accumulation (sign of repeated exploitation attempts)
            if orchestrator_state.total_failures >= 5:
                violations.append(
                    f"High failure count in orchestrator: {orchestrator_state.total_failures} failures"
                )
                integrity_flags.append("high_failures")
                hack_indicators += 1
            
            # Check 1b: High clamp rate (sign of system fighting unstable policy)
            if orchestrator_state.total_clamps >= 3:
                violations.append(
                    f"High clamp count in orchestrator: {orchestrator_state.total_clamps} clamps"
                )
                integrity_flags.append("high_clamps")
                hack_indicators += 1
            
            # Check 1c: Analyze knob delta history for oscillation patterns
            if orchestrator_state.knob_deltas:
                knob_values = [kd.new_value for kd in orchestrator_state.knob_deltas]
                if len(knob_values) >= 3:
                    # Detect sign oscillation in knob adjustments
                    sign_changes = 0
                    for i in range(2, len(knob_values)):
                        prev_delta = knob_values[i - 1] - knob_values[i - 2]
                        curr_delta = knob_values[i] - knob_values[i - 1]
                        if prev_delta * curr_delta < 0:  # Sign change
                            sign_changes += 1
                    
                    knob_oscillation_rate = sign_changes / (len(knob_values) - 2)
                    findings["knob_oscillation_rate"] = knob_oscillation_rate
                    
                    if knob_oscillation_rate > 0.5:  # More than 50% sign changes
                        violations.append(
                            f"Knob oscillation detected: {knob_oscillation_rate:.0%} sign changes"
                        )
                        integrity_flags.append("knob_oscillation")
                        hack_indicators += 1
        else:
            findings["orchestrator_state_present"] = False

        # Check context for historical patterns (from notes if available)
        history = context.notes.get("weight_history", []) if context and context.notes else []
        findings["history_length"] = len(history)

        # Check 2: Weight oscillations (if we have history)
        oscillation_rate = 0.0
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
                integrity_flags.append("oscillation")
                hack_indicators += 1

        # Check 2: Trajectory audit reward component analysis
        if trajectory_audit is not None:
            findings["trajectory_audit_present"] = True
            
            # Get configurable thresholds from policy_config.regal_gates (or use safe defaults)
            extreme_reward_threshold = 10.0
            high_return_threshold = 10.0
            if policy_config and policy_config.regal_gates:
                extreme_reward_threshold = policy_config.regal_gates.extreme_reward_component_threshold
                high_return_threshold = policy_config.regal_gates.high_total_return_threshold
            findings["reward_thresholds_used"] = {
                "extreme_reward_component": extreme_reward_threshold,
                "high_total_return": high_return_threshold,
            }
            
            if trajectory_audit.reward_components:
                for name, value in trajectory_audit.reward_components.items():
                    # Flag extreme reward component values
                    if abs(value) > extreme_reward_threshold:
                        violations.append(f"Extreme reward component: {name}={value:.2f} (|val| > {extreme_reward_threshold})")
                        integrity_flags.append(f"extreme_reward_{name}")
                        hack_indicators += 1
            # Reward oscillation indicator (very large spread between min/max reward)
            if trajectory_audit.total_return > high_return_threshold:
                violations.append(f"Unusually high total return: {trajectory_audit.total_return:.2f} > {high_return_threshold}")
                integrity_flags.append("high_return")
                hack_indicators += 1
        else:
            findings["trajectory_audit_present"] = False

        # Check 3: Anomalous gain requests in plan
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
                        integrity_flags.append("anomalous_gain")
                        hack_indicators += 1

        # Check 4: Econ tensor analysis (if available, prefer passed econ_tensor param)
        tensor = econ_tensor
        econ_basis_sha = None
        if context and context.econ_basis_sha:
            econ_basis_sha = context.econ_basis_sha

        if tensor is not None:
            findings["econ_tensor_available"] = True
            findings["econ_basis_sha"] = econ_basis_sha

            # Use econ tensor for more sophisticated checks
            try:
                from src.economics.econ_tensor import tensor_to_econ_dict
                econ_dict = tensor_to_econ_dict(tensor)
                findings["econ_values"] = {k: round(v, 4) for k, v in list(econ_dict.items())[:5]}

                reward = econ_dict.get("reward_scalar_sum", 0.0)
                damage = econ_dict.get("damage_cost", 0.0)
                mpl = econ_dict.get("mpl_units_per_hour", 0.0)

                if reward > 1.0 and mpl < 0.1 and damage > 5.0:
                    violations.append(
                        f"Econ anomaly: high reward ({reward:.2f}) with low MPL ({mpl:.2f}) "
                        f"and high damage ({damage:.2f})"
                    )
                    integrity_flags.append("econ_anomaly")
                    hack_indicators += 1

            except Exception as e:
                findings["econ_tensor_error"] = str(e)
        else:
            findings["econ_tensor_available"] = False

        findings["violations"] = violations

        # Compute hack probability (0.0-1.0)
        hack_probability = min(1.0, (oscillation_rate * 0.5) + (hack_indicators * 0.25))

        passed = len(violations) == 0
        report = RegalReportV1(
            regal_id=self.regal_id,
            phase=phase,
            regal_version=self.regal_version,
            inputs_sha=inputs_sha,
            determinism_seed=self.seed,
            passed=passed,
            confidence=0.9 if passed else 0.6,
            rationale="No reward hacking patterns detected"
            if passed else f"Detected {len(violations)} potential issues",
            hack_probability=hack_probability,
            integrity_flags=list(set(integrity_flags)),
            findings=findings,
        )
        report.compute_sha()
        return report


@register_regal("econ_data")
class EconDataRegal(RegalNode):
    """Validates econ tensor invariants and data allocation constraints.

    True sibling regal node for economic metrics, producing a hashable
    RegalReportV1 that is aggregated into LedgerRegalV1.

    Checks:
    - Econ tensor presence and validity
    - Basis SHA integrity (matches registered basis)
    - Tensor value invariants (no NaN/Inf, reasonable bounds)
    - Axis completeness (all axes have valid values)
    - Exposure/datapack constraints (if available)
    """

    regal_id = "econ_data"

    def evaluate(
        self,
        plan: Optional[SemanticUpdatePlanV1],
        signals: Optional["SignalBundle"],
        policy_config: Optional[PlanPolicyConfigV1],
        context: Optional[RegalContextV1] = None,
        phase: RegalPhaseV1 = RegalPhaseV1.POST_PLAN_PRE_APPLY,
        trajectory_audit: Optional[TrajectoryAuditV1] = None,
        econ_tensor: Optional[EconTensorV1] = None,
    ) -> RegalReportV1:
        inputs_sha = self._compute_inputs_sha(plan, signals, policy_config, context, phase, trajectory_audit, econ_tensor)
        findings: Dict[str, Any] = {}
        violations: List[str] = []
        coherence_tags: List[str] = []

        # Prefer passed econ_tensor param over context lookup
        tensor = econ_tensor
        econ_basis_sha = context.econ_basis_sha if context else None

        if tensor is None:
            # No econ tensor provided - pass with warning
            findings["econ_tensor_available"] = False
            findings["reason"] = "No econ tensor in context"

            report = RegalReportV1(
                regal_id=self.regal_id,
                phase=phase,
                regal_version=self.regal_version,
                inputs_sha=inputs_sha,
                determinism_seed=self.seed,
                passed=True,
                confidence=0.5,  # Lower confidence without econ data
                rationale="No econ tensor provided; skipping econ validation",
                spec_consistency_score=1.0,  # Not applicable
                coherence_score=1.0,  # Not applicable
                findings=findings,
            )
            report.compute_sha()
            return report

        findings["econ_tensor_available"] = True
        findings["basis_id"] = tensor.basis_id
        findings["basis_sha"] = tensor.basis_sha[:16]
        findings["tensor_sha"] = tensor.sha256()[:16]
        findings["num_axes"] = len(tensor.x)

        # Check 1: Basis SHA integrity
        try:
            from src.economics.econ_basis_registry import get_basis
            registered_basis = get_basis(tensor.basis_id)
            if registered_basis is not None:
                if registered_basis.sha256 != tensor.basis_sha:
                    violations.append(
                        f"Basis SHA mismatch: tensor has {tensor.basis_sha[:16]}... "
                        f"but registry has {registered_basis.sha256[:16]}..."
                    )
                    coherence_tags.append("basis_sha_mismatch")
                else:
                    findings["basis_verified"] = True
            else:
                findings["basis_verified"] = False
                findings["basis_warning"] = f"Basis '{tensor.basis_id}' not in registry"
        except Exception as e:
            findings["basis_check_error"] = str(e)

        # Check 2: Tensor value invariants
        import math
        nan_count = 0
        inf_count = 0
        for i, val in enumerate(tensor.x):
            if math.isnan(val):
                nan_count += 1
            elif math.isinf(val):
                inf_count += 1

        if nan_count > 0:
            violations.append(f"Tensor contains {nan_count} NaN values")
            coherence_tags.append("nan_values")
        if inf_count > 0:
            violations.append(f"Tensor contains {inf_count} Inf values")
            coherence_tags.append("inf_values")

        findings["nan_count"] = nan_count
        findings["inf_count"] = inf_count

        # Check 3: Reasonable bounds (heuristic)
        if tensor.stats:
            norm = tensor.stats.get("norm", 0.0)
            findings["tensor_norm"] = norm
            if norm > 1000.0:
                violations.append(f"Tensor norm unusually large: {norm:.2f}")
                coherence_tags.append("large_norm")

        # Check 4: Axis completeness (if mask present)
        if tensor.mask is not None:
            missing_count = sum(1 for m in tensor.mask if not m)
            findings["masked_axes"] = missing_count
            if missing_count > len(tensor.x) // 2:
                violations.append(f"More than half of axes masked: {missing_count}/{len(tensor.x)}")
                coherence_tags.append("high_missing")

        # Check 5: Trajectory audit correlation (econ tensor must match trajectory reality)
        if trajectory_audit is not None:
            findings["trajectory_audit_present"] = True
            # Sanity: if tensor shows high success but trajectory shows many anomalies, flag
            try:
                from src.economics.econ_tensor import tensor_to_econ_dict
                econ_dict = tensor_to_econ_dict(tensor)
                success = econ_dict.get("success_rate", 0.0)
                anomaly_count = trajectory_audit.velocity_spike_count + trajectory_audit.contact_anomaly_count
                if success > 0.9 and anomaly_count >= 5:
                    violations.append(
                        f"Econ-trajectory mismatch: success={success:.2f} but {anomaly_count} anomalies"
                    )
                    coherence_tags.append("econ_trajectory_mismatch")
            except Exception:
                pass
        else:
            findings["trajectory_audit_present"] = False

        # Compute scores
        num_checks = 6  # basis, nan, inf, bounds, completeness, trajectory
        num_failures = len(violations)
        econ_consistency_score = 1.0 - (num_failures / num_checks)
        econ_consistency_score = max(0.0, econ_consistency_score)

        findings["violations"] = violations
        findings["econ_consistency_score"] = econ_consistency_score

        passed = len(violations) == 0
        report = RegalReportV1(
            regal_id=self.regal_id,
            phase=phase,
            regal_version=self.regal_version,
            inputs_sha=inputs_sha,
            determinism_seed=self.seed,
            passed=passed,
            confidence=0.9 if passed else 0.7,
            rationale="Econ tensor validation passed"
            if passed else f"Detected {len(violations)} econ violations",
            spec_consistency_score=econ_consistency_score,
            coherence_score=econ_consistency_score,
            coherence_tags=list(set(coherence_tags)),
            findings=findings,
        )
        report.compute_sha()
        return report


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_regals(
    config: RegalGatesV1,
    phase: RegalPhaseV1 = RegalPhaseV1.POST_PLAN_PRE_APPLY,
    plan: Optional[SemanticUpdatePlanV1] = None,
    signals: Optional["SignalBundle"] = None,
    policy_config: Optional[PlanPolicyConfigV1] = None,
    context: Optional[RegalContextV1] = None,
    trajectory_audit: Optional[TrajectoryAuditV1] = None,
    econ_tensor: Optional[EconTensorV1] = None,
    selection_manifest: Optional[SelectionManifestV1] = None,
    orchestrator_state: Optional[OrchestratorStateV1] = None,
) -> LedgerRegalV1:
    """Evaluate all enabled regal nodes and return aggregated result.

    Args:
        config: Regal gates configuration specifying which nodes to run
        phase: Temporal phase for this evaluation
        plan: Current semantic update plan
        signals: Current signal bundle
        policy_config: Plan policy configuration
        context: Typed regal context
        trajectory_audit: Episode trajectory audit for substrate grounding
        econ_tensor: Econ tensor for economic metric validation
        selection_manifest: Selection manifest for datapack provenance
        orchestrator_state: Orchestrator state for loop memory/actuation trace

    Returns:
        LedgerRegalV1 with all reports and aggregate pass/fail

    Raises:
        NotImplementedError: If phase is not yet wired into the runner
    """
    # Phase validation: only shipped phases are allowed
    SHIPPED_PHASES = {RegalPhaseV1.POST_PLAN_PRE_APPLY, RegalPhaseV1.POST_AUDIT}
    if phase not in SHIPPED_PHASES:
        raise NotImplementedError(
            f"Phase {phase.value!r} is not yet wired into the runner. "
            f"Shipped phases: {[p.value for p in SHIPPED_PHASES]}. "
            "See REGAL_WIRING_SPEC.md for planned phases."
        )

    reports: List[RegalReportV1] = []
    all_inputs: List[str] = []

    for regal_id in config.enabled_regal_ids:
        if regal_id not in REGAL_REGISTRY:
            # Skip unknown regals (could also warn/error)
            continue

        regal_cls = REGAL_REGISTRY[regal_id]
        regal = regal_cls(seed=config.determinism_seed)
        report = regal.evaluate(
            plan, signals, policy_config, context, phase, trajectory_audit, econ_tensor,
            selection_manifest, orchestrator_state
        )
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


def write_ledger_regal(ledger_regal: LedgerRegalV1, output_dir: Path) -> str:
    """Write LedgerRegalV1 to file and return SHA.
    
    Args:
        ledger_regal: The regal ledger to persist
        output_dir: Directory to write ledger_regal.json
        
    Returns:
        SHA-256 of the written file (for manifest inclusion)
    """
    from src.utils.config_digest import sha256_file
    import json
    
    output_path = Path(output_dir) / "ledger_regal.json"
    
    # Serialize with deterministic ordering
    data = ledger_regal.model_dump(mode="json")
    
    # Sort reports by (phase, regal_id) for deterministic ordering
    if data.get("reports"):
        data["reports"] = sorted(
            data["reports"],
            key=lambda r: (r.get("phase", ""), r.get("regal_id", ""))
        )
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    
    return sha256_file(str(output_path))
