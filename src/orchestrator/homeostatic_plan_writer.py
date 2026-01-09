"""Homeostatic plan writer with gate-aware logic.

Generates SemanticUpdatePlanV1 deterministically from SignalBundle.
Maps ActionPlan → plan operations for hot-reload.
Respects stability and transfer gates from probe discriminator.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from src.contracts.schemas import (
    SemanticUpdatePlanV1,
    TaskGraphOp,
    DatapackSelectionConfig,
    PlanOpType,
    ProbeEpiReportV1,
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
    LedgerPlanPolicyV1,
)
from src.representation.homeostasis import (
    SignalBundle,
    ActionPlan,
    ActionType,
    SignalType,
    ControlSignal,
)
from src.utils.config_digest import sha256_json
from src.orchestrator.policy_hooks import (
    EconPlanPolicyProvider,
    RewardIntegrityGuard,
    DefaultEconPolicyProvider,
    DefaultRewardIntegrityGuard,
)


def _default_policy_config() -> PlanPolicyConfigV1:
    """Create default policy config with hardcoded values (migrated from v0)."""
    return PlanPolicyConfigV1(
        gain_schedule=PlanGainScheduleV1(
            conservative_multiplier=1.1,
            full_multiplier=1.5,
            max_abs_weight_change=0.5,
            min_weight_clamp=0.1,
            max_weight_clamp=2.0,
        ),
        default_weights={"manipulation": 0.5, "navigation": 0.5},
    )


@dataclass
class GateStatus:
    """Status of stability and transfer gates."""

    stability_pass: bool = True
    transfer_pass: bool = False
    delta_epi_per_flop: Optional[float] = None
    raw_delta: Optional[float] = None  # Raw delta before normalization
    flops_estimate: Optional[float] = None
    sign_consistency: Optional[float] = None
    ood_delta: Optional[float] = None
    forced_noop: bool = False
    reason: str = ""

    # Transfer patience tracking
    transfer_failure_count: int = 0
    transfer_patience_exceeded: bool = False

    # Exposure normalization
    delta_per_exposure: Optional[float] = None
    exposure_count: Optional[int] = None

    # Policy Ledger (for audit/provenance)
    ledger_policy: Optional[LedgerPlanPolicyV1] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stability_pass": self.stability_pass,
            "transfer_pass": self.transfer_pass,
            "delta_epi_per_flop": self.delta_epi_per_flop,
            "raw_delta": self.raw_delta,
            "flops_estimate": self.flops_estimate,
            "sign_consistency": self.sign_consistency,
            "ood_delta": self.ood_delta,
            "forced_noop": self.forced_noop,
            "reason": self.reason,
            "transfer_failure_count": self.transfer_failure_count,
            "transfer_patience_exceeded": self.transfer_patience_exceeded,
            "delta_per_exposure": self.delta_per_exposure,
            "exposure_count": self.exposure_count,
            "ledger_policy": self.ledger_policy.model_dump() if self.ledger_policy else None,
        }


def build_signal_bundle_for_plan(
    audit_deltas: Optional[Dict[str, float]] = None,
    epiplexity_metrics: Optional[Dict[str, Any]] = None,
    alignment_errors: Optional[Dict[str, float]] = None,
    coverage_stats: Optional[Dict[str, int]] = None,
    probe_report: Optional[ProbeEpiReportV1] = None,
) -> SignalBundle:
    """Build signal bundle from various metric sources.

    Args:
        audit_deltas: Deltas from audit (e.g., delta_success, delta_mpl)
        epiplexity_metrics: Epiplexity/curated slice outputs
        alignment_errors: Cycle errors from isomorphism adapters
        coverage_stats: Episodes per slice/task
        probe_report: Optional probe epiplexity report

    Returns:
        SignalBundle for controller consumption
    """
    signals: List[ControlSignal] = []

    # Epiplexity from audit or token-only
    if epiplexity_metrics:
        epi_value = epiplexity_metrics.get("mean_variance", 0.5)
        signals.append(
            ControlSignal(
                signal_type=SignalType.EPIPLEXITY,
                value=float(epi_value),
                target=0.5,
                threshold_low=0.1,
                threshold_high=0.9,
                metadata={"source": "epiplexity_metrics"},
            )
        )

    # Stability from audit delta stability
    if audit_deltas:
        # Use negative of absolute delta as stability proxy
        delta_success = abs(audit_deltas.get("delta_success", 0) or 0)
        stability = max(0.0, 1.0 - delta_success * 5)  # Scale to [0,1]
        signals.append(
            ControlSignal(
                signal_type=SignalType.STABILITY,
                value=stability,
                target=0.9,
                threshold_low=0.7,
                threshold_high=1.0,
                metadata={"source": "audit_deltas"},
            )
        )

    # Alignment error from isomorphisms
    if alignment_errors:
        avg_error = sum(alignment_errors.values()) / max(1, len(alignment_errors))
        signals.append(
            ControlSignal(
                signal_type=SignalType.ALIGNMENT_ERROR,
                value=float(avg_error),
                target=0.1,
                threshold_low=0.0,
                threshold_high=0.5,
                rising_is_bad=True,
                metadata={"source": "alignment_errors"},
            )
        )

    # Coverage from episode counts
    if coverage_stats:
        total = sum(coverage_stats.values())
        coverage = min(1.0, total / 100.0)  # Normalize to 100 episodes
        signals.append(
            ControlSignal(
                signal_type=SignalType.COVERAGE,
                value=coverage,
                target=0.8,
                threshold_low=0.3,
                threshold_high=1.0,
                metadata={"source": "coverage_stats", "total": total},
            )
        )

    # Delta-epi-per-flop from probe report
    if probe_report:
        signals.append(
            ControlSignal(
                signal_type=SignalType.DELTA_EPI_PER_FLOP,
                value=probe_report.delta_epi_per_flop,
                target=1e-9,  # Any positive delta is good
                threshold_low=0.0,
                threshold_high=1.0,
                metadata={
                    "source": "probe_report",
                    "stability_pass": probe_report.stability_pass,
                    "transfer_pass": probe_report.transfer_pass,
                    "sign_consistency": probe_report.sign_consistency,
                    "ood_delta": probe_report.ood_delta,
                    "report_sha": probe_report.report_sha,
                    "raw_delta": probe_report.delta,
                    "flops_estimate": probe_report.flops_estimate,
                },
            )
        )

    return SignalBundle(
        signals=signals,
        timestamp=datetime.now().isoformat(),
        metadata={"generator": "build_signal_bundle_for_plan"},
    )


def check_gates(
    signal_bundle: SignalBundle,
    config: PlanPolicyConfigV1,
    exposure_count: Optional[int] = None,
    previous_transfer_fail_count: int = 0,
) -> GateStatus:
    """Check stability and transfer gates from probe signal.

    Args:
        signal_bundle: Bundle containing probe signal
        config: Plan config with thresholds
        exposure_count: Optional exposure count for normalization
        previous_transfer_fail_count: Previous consecutive transfer fail count

    Returns:
        GateStatus with pass/fail and reason
    """
    probe_signal = signal_bundle.get_signal(SignalType.DELTA_EPI_PER_FLOP)

    if probe_signal is None:
        return GateStatus(reason="No probe signal")

    meta = probe_signal.metadata
    delta_epi = probe_signal.value
    stability_pass = meta.get("stability_pass", True)
    transfer_pass = meta.get("transfer_pass", False)
    sign_consistency = meta.get("sign_consistency")
    ood_delta = meta.get("ood_delta")
    raw_delta = meta.get("raw_delta", delta_epi * meta.get("flops_estimate", 1.0))
    flops_estimate = meta.get("flops_estimate")

    # Compute delta per exposure if available
    delta_per_exposure = None
    if exposure_count and exposure_count > 0 and raw_delta:
        delta_per_exposure = raw_delta / exposure_count

    # Check stability gate
    if not stability_pass:
        return GateStatus(
            stability_pass=False,
            transfer_pass=transfer_pass,
            delta_epi_per_flop=delta_epi,
            raw_delta=raw_delta,
            flops_estimate=flops_estimate,
            sign_consistency=sign_consistency,
            ood_delta=ood_delta,
            forced_noop=True,
            reason=f"Stability gate failed: sign_consistency={sign_consistency:.2f}",
            delta_per_exposure=delta_per_exposure,
            exposure_count=exposure_count,
            transfer_failure_count=previous_transfer_fail_count,
        )

    # Check raw delta floor (prevent vanishing changes)
    if raw_delta is not None and abs(raw_delta) < config.min_raw_delta:
        return GateStatus(
            stability_pass=True,
            transfer_pass=transfer_pass,
            delta_epi_per_flop=delta_epi,
            raw_delta=raw_delta,
            flops_estimate=flops_estimate,
            sign_consistency=sign_consistency,
            ood_delta=ood_delta,
            forced_noop=True,
            reason=f"Raw delta below floor: |{raw_delta:.4f}| < {config.min_raw_delta}",
            delta_per_exposure=delta_per_exposure,
            exposure_count=exposure_count,
            transfer_failure_count=previous_transfer_fail_count,
        )

    # Check normalized delta threshold
    if delta_epi <= config.delta_epi_per_flop_threshold:
        return GateStatus(
            stability_pass=True,
            transfer_pass=transfer_pass,
            delta_epi_per_flop=delta_epi,
            raw_delta=raw_delta,
            flops_estimate=flops_estimate,
            sign_consistency=sign_consistency,
            ood_delta=ood_delta,
            forced_noop=True,
            reason=f"Delta/FLOP too low: {delta_epi:.2e}",
            delta_per_exposure=delta_per_exposure,
            exposure_count=exposure_count,
            transfer_failure_count=previous_transfer_fail_count,
        )

    # Track transfer patience
    transfer_failure_count = previous_transfer_fail_count
    
    if not transfer_pass:
        transfer_failure_count += 1
        if transfer_failure_count > config.max_transfer_failures:
            return GateStatus(
                stability_pass=True,
                transfer_pass=False,
                delta_epi_per_flop=delta_epi,
                raw_delta=raw_delta,
                flops_estimate=flops_estimate,
                sign_consistency=sign_consistency,
                ood_delta=ood_delta,
                forced_noop=True,
                reason=f"Transfer patience exceeded: {transfer_failure_count} failures",
                transfer_failure_count=transfer_failure_count,
                transfer_patience_exceeded=True,
                delta_per_exposure=delta_per_exposure,
                exposure_count=exposure_count,
            )
    else:
        # Reset transfer failure count on success
        transfer_failure_count = 0

    return GateStatus(
        stability_pass=True,
        transfer_pass=transfer_pass,
        delta_epi_per_flop=delta_epi,
        raw_delta=raw_delta,
        flops_estimate=flops_estimate,
        sign_consistency=sign_consistency,
        ood_delta=ood_delta,
        forced_noop=False,
        reason="Gates passed",
        transfer_failure_count=transfer_failure_count,
        delta_per_exposure=delta_per_exposure,
        exposure_count=exposure_count,
    )


def map_action_to_plan_ops(
    action_plan: ActionPlan,
    config: PlanPolicyConfigV1,
    gate_status: GateStatus,
    current_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[TaskGraphOp], float, bool, Dict[str, Any]]:
    """Map ActionPlan actions to TaskGraphOp operations with gain scheduling.
    
    Includes normalization and clamping tracking.

    Args:
        action_plan: ActionPlan from controller
        config: Generation config (includes gain schedule)
        gate_status: Gate check results
        current_weights: Current task weights

    Returns:
        Tuple of (ops, applied_multiplier, was_clamped, metadata)
    """
    
    # Helper for deterministic hashing (platform-stable)
    def _deterministic_weights(w: Dict[str, float]) -> Dict[str, float]:
        # Sort keys and round to 8 decimals to prevent float jitter
        return {k: round(val, 8) for k, val in sorted(w.items())}

    weights = dict(current_weights or config.default_weights)
    pre_weights_sha = sha256_json(_deterministic_weights(weights))
    ops: List[TaskGraphOp] = []
    
    schedule = config.gain_schedule
    applied_multiplier = 1.0
    was_clamped = False
    clamp_reasons = set()

    # If stability gate forces NOOP, return default/current weights only
    if gate_status.forced_noop:
        for task, weight in weights.items():
            ops.append(TaskGraphOp(
                op=PlanOpType.SET_WEIGHT,
                task_family=task,
                weight=weight,
            ))
        return ops, 1.0, False, {
            "pre_weights_sha": pre_weights_sha,
            "post_weights_sha": pre_weights_sha,
            "renormalized": False,
            "clamp_reasons": ["forced_noop"]
        }

    # Determine applied gain multiplier
    if gate_status.transfer_pass:
        applied_multiplier = schedule.full_multiplier
    else:
        applied_multiplier = schedule.conservative_multiplier

    increase_factor = applied_multiplier
    
    # Apply actions to weights
    for action in action_plan.actions:
        if action == ActionType.NOOP:
            continue

        target_new_weight = None
        task_list = list(weights.keys())

        if action in (ActionType.INCREASE_DATA, ActionType.DECREASE_DATA):
            is_increase = action == ActionType.INCREASE_DATA
            factor = increase_factor if is_increase else (1.0 / increase_factor if increase_factor != 0 else 0.0)
            
            for task in task_list:
                current_w = weights[task]
                proposed = current_w * factor
                
                # Apply delta clamp
                delta = proposed - current_w
                if schedule.max_abs_weight_change is not None:
                    if abs(delta) > schedule.max_abs_weight_change:
                        delta = schedule.max_abs_weight_change if delta > 0 else -schedule.max_abs_weight_change
                        was_clamped = True
                        clamp_reasons.add("max_abs_weight_change")
                
                new_weight = current_w + delta
                
                # Apply min/max caps
                if schedule.min_weight_clamp is not None:
                    if new_weight < schedule.min_weight_clamp:
                        new_weight = schedule.min_weight_clamp
                        was_clamped = True
                        clamp_reasons.add("min_weight_clamp")
                        
                if schedule.max_weight_clamp is not None:
                    if new_weight > schedule.max_weight_clamp:
                        new_weight = schedule.max_weight_clamp
                        was_clamped = True
                        clamp_reasons.add("max_weight_clamp")

                weights[task] = new_weight

    # Renormalize
    renormalized = False
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6 and total_weight > 0:
        norm_factor = 1.0 / total_weight
        for task in weights:
            weights[task] *= norm_factor
        renormalized = True
        
    # Generate ops
    for task, weight in weights.items():
        ops.append(TaskGraphOp(
            op=PlanOpType.SET_WEIGHT,
            task_family=task,
            weight=weight,
        ))
        
    post_weights_sha = sha256_json(_deterministic_weights(weights))
        
    metadata = {
        "pre_weights_sha": pre_weights_sha,
        "post_weights_sha": post_weights_sha,
        "renormalized": renormalized,
        "clamp_reasons": list(clamp_reasons),
    }

    return ops, applied_multiplier, was_clamped, metadata



def build_plan_from_signals(
    signal_bundle: SignalBundle,
    config: Optional[PlanPolicyConfigV1] = None,
    plan_id: Optional[str] = None,
    source_commit: Optional[str] = None,
    probe_report: Optional[ProbeEpiReportV1] = None,
    exposure_count: Optional[int] = None,
    econ_policy_provider: Optional[EconPlanPolicyProvider] = None,
    integrity_guard: Optional[RewardIntegrityGuard] = None,
    previous_transfer_fail_count: int = 0,
    steps_since_last_change: Optional[int] = None,
) -> Tuple[SemanticUpdatePlanV1, GateStatus]:
    """Build SemanticUpdatePlanV1 deterministically from signals.

    Same signals + same config → same plan sha.
    Respects stability and transfer gates.

    Args:
        signal_bundle: Input signals
        config: Generation config
        plan_id: Optional plan ID (generated if not provided)
        source_commit: Optional git commit
        probe_report: Optional probe report (adds to signal bundle)
        exposure_count: Optional exposure count for normalization
        econ_policy_provider: Hook for economic policy
        integrity_guard: Hook for reward integrity
        previous_transfer_fail_count: Previous failure count (state)

    Returns:
        Tuple of (SemanticUpdatePlanV1, GateStatus)
    """
    from src.representation.homeostasis import HomeostaticController

    config = config or _default_policy_config()
    plan_id = plan_id or f"auto_{str(uuid.uuid4())[:8]}"

    # Add probe signal if report provided
    if probe_report:
        signal_bundle.signals.append(
            ControlSignal(
                signal_type=SignalType.DELTA_EPI_PER_FLOP,
                value=probe_report.delta_epi_per_flop,
                target=1e-9,
                threshold_low=0.0,
                threshold_high=1.0,
                metadata={
                    "source": "probe_report",
                    "stability_pass": probe_report.stability_pass,
                    "transfer_pass": probe_report.transfer_pass,
                    "sign_consistency": probe_report.sign_consistency,
                    "ood_delta": probe_report.ood_delta,
                    "report_sha": probe_report.report_sha,
                    "raw_delta": probe_report.delta,
                    "flops_estimate": probe_report.flops_estimate,
                },
            )
        )

    # 1. Apply Economic Policy Hook
    gain_schedule_override = None
    if econ_policy_provider:
        gain_schedule_override = econ_policy_provider.get_gain_schedule(signal_bundle)
        if gain_schedule_override:
            # Create a shallow copy with new gain schedule
            config = config.model_copy(update={"gain_schedule": gain_schedule_override})

    # 2. Check Gates
    gate_status = check_gates(signal_bundle, config, exposure_count, previous_transfer_fail_count)
    
    # Check cooldown logic
    if steps_since_last_change is not None and config.min_apply_interval_steps > 0:
        if steps_since_last_change < config.min_apply_interval_steps:
             if not gate_status.forced_noop:
                 gate_status.forced_noop = True
                 gate_status.reason = f"Cooldown: {steps_since_last_change} < {config.min_apply_interval_steps}"

    # 3. Generate Action Plan
    controller = HomeostaticController()
    action_plan = controller.step(signal_bundle)

    # Override to NOOP if gates force it
    if gate_status.forced_noop:
        action_plan = ActionPlan(
            actions=[ActionType.NOOP],
            priority=0,
            rationale=f"Gate forced NOOP: {gate_status.reason}",
        )

    # 4. Apply Integrity Guard Hook and Map Ops
    if integrity_guard:
        # Integrity guard might adjust schedule based on telemetry or proposed schedule
        safe_schedule = integrity_guard.adjust_gain_schedule(config.gain_schedule)
        config = config.model_copy(update={"gain_schedule": safe_schedule})

    ops, applied_multiplier, was_clamped, normalization_meta = map_action_to_plan_ops(
        action_plan, config, gate_status
    )

    # If no ops, add default weights (but only if not forced NOOP?)
    # If forced NOOP, we still return empty ops?
    # Old logic: if forced NOOP, map_action_to_plan_ops returns SET_WEIGHT ops for defaults?
    # Wait, check map_action_to_plan_ops logic in Step 733:
    # "If stability gate forces NOOP, return default weights only" (returns SET_WEIGHT ops).
    # So ops will not be empty in forced NOOP case.
    # But if ops IS empty (e.g. NOOP action plan and no defaults needed?), then we append defaults.
    if not ops:
        for task, weight in config.default_weights.items():
            ops.append(TaskGraphOp(
                op=PlanOpType.SET_WEIGHT,
                task_family=task,
                weight=weight,
            ))

    # Calculate plan SHA
    config_dump = config.model_dump()
    plan_sha = sha256_json({
        "ops": [op.model_dump() for op in ops],
        "signals": [asdict(s) for s in signal_bundle.signals],
        "config": config_dump,
    })

    # Create LedgerPlanPolicy record
    ledger_policy = LedgerPlanPolicyV1(
         policy_config_sha=sha256_json(config_dump),
         gain_schedule_sha=sha256_json(config.gain_schedule.model_dump()),
         applied_multiplier=float(applied_multiplier),
         gain_schedule_source="econ_override" if gain_schedule_override else "default",
         clamped=was_clamped,
         transfer_failure_count=gate_status.transfer_failure_count,
         pre_weights_sha=normalization_meta.get("pre_weights_sha"),
         post_weights_sha=normalization_meta.get("post_weights_sha"),
         renormalized=normalization_meta.get("renormalized", False),
         clamp_reasons=normalization_meta.get("clamp_reasons", []),
    )
    
    # Attach to GateStatus
    gate_status.ledger_policy = ledger_policy

    # Build notes
    notes = f"Priority: {action_plan.priority}. Actions: {[a.value for a in action_plan.actions]}."
    if applied_multiplier is not None:
         notes += f" Multiplier: {applied_multiplier:.2f}."
    if gate_status.delta_epi_per_flop is not None:
        notes += f" ΔEpi/FLOP: {gate_status.delta_epi_per_flop:.2e}."
    notes += f" Gates: stability={gate_status.stability_pass}, transfer={gate_status.transfer_pass}."
    if gate_status.forced_noop:
        notes += f" FORCED_NOOP: {gate_status.reason}"
    if was_clamped:
        notes += f" [CLAMPED: {','.join(normalization_meta.get('clamp_reasons', []))}]"
    if normalization_meta.get("renormalized"):
        notes += " [Normalized]"

    # Build plan
    plan = SemanticUpdatePlanV1(
        plan_id=plan_id,
        source_commit=source_commit,
        task_graph_changes=ops,
        notes=notes,
        created_at=datetime.now().isoformat(),
        plan_sha=plan_sha,
        rationales=[gate_status.reason] if gate_status.reason else [],
    )

    return plan, gate_status


def write_plan(path: str, plan: SemanticUpdatePlanV1) -> str:
    """Write plan to JSON file.

    Args:
        path: Output path
        plan: Plan to write

    Returns:
        SHA-256 of written plan
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(plan.model_dump(mode="json"), f, indent=2)
    return plan.sha256()


__all__ = [
    "PlanFromSignalsConfig",
    "GateStatus",
    "build_signal_bundle_for_plan",
    "check_gates",
    "map_action_to_plan_ops",
    "build_plan_from_signals",
    "write_plan",
]
