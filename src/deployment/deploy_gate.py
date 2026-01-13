"""Deployment gating module for regal-aware deploy decisions.

Provides typed deploy decision interface based on regal outputs.
Actual deployment logic is external; this module provides the decision API.

Phase 6: Deploy gate is fully causal - decision is deterministic given inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.schemas import (
    LedgerRegalV1,
    RegalReportV1,
    DeployGateInputsV1,
    DeployGateDecisionV1,
)
from src.utils.config_digest import sha256_json


@dataclass
class DeployGateDecision:
    """Typed deploy gate decision based on regal outputs.
    
    This is the canonical output of deploy gating logic.
    External deployment systems consume this decision.
    """
    
    # Core decision
    allow_deploy: bool = True
    block_reason: Optional[str] = None
    
    # Regal provenance
    regal_all_passed: bool = True
    regal_report_sha: Optional[str] = None
    regal_degraded: bool = False  # True if regal was missing/incomplete
    
    # Audit delta requirements
    audit_regression_detected: bool = False
    audit_delta_success: Optional[float] = None
    
    # Decision metadata
    decision_sha: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Detailed breakdown
    failed_regals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def compute_sha(self) -> str:
        """Compute deterministic SHA of this decision."""
        data = {
            "allow_deploy": self.allow_deploy,
            "block_reason": self.block_reason,
            "regal_all_passed": self.regal_all_passed,
            "regal_report_sha": self.regal_report_sha,
            "regal_degraded": self.regal_degraded,
            "audit_regression_detected": self.audit_regression_detected,
            "audit_delta_success": self.audit_delta_success,
            "failed_regals": sorted(self.failed_regals),
        }
        self.decision_sha = sha256_json(data)
        return self.decision_sha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "allow_deploy": self.allow_deploy,
            "block_reason": self.block_reason,
            "regal_all_passed": self.regal_all_passed,
            "regal_report_sha": self.regal_report_sha,
            "regal_degraded": self.regal_degraded,
            "audit_regression_detected": self.audit_regression_detected,
            "audit_delta_success": self.audit_delta_success,
            "decision_sha": self.decision_sha,
            "created_at": self.created_at,
            "failed_regals": self.failed_regals,
            "warnings": self.warnings,
        }


def check_deploy_gate(
    regal_result: Optional[LedgerRegalV1] = None,
    audit_delta_success: Optional[float] = None,
    min_audit_delta: float = -0.05,  # Max regression allowed
    require_regal: bool = True,
) -> DeployGateDecision:
    """Check deploy gate based on regal and audit results.
    
    Args:
        regal_result: LedgerRegalV1 from regal evaluation
        audit_delta_success: Delta success rate from audit (negative = regression)
        min_audit_delta: Minimum allowed audit delta (default -5%)
        require_regal: If True, missing regal blocks deploy
        
    Returns:
        DeployGateDecision with allow/block and reason
    """
    decision = DeployGateDecision()
    
    # Check regal result
    if regal_result is None:
        decision.regal_degraded = True
        if require_regal:
            decision.allow_deploy = False
            decision.block_reason = "Regal evaluation missing (required)"
            decision.warnings.append("regal_missing")
    else:
        decision.regal_all_passed = regal_result.all_passed
        decision.regal_report_sha = sha256_json([r.report_sha for r in regal_result.reports])
        
        if not regal_result.all_passed:
            decision.allow_deploy = False
            decision.failed_regals = [
                r.regal_id for r in regal_result.reports if not r.passed
            ]
            decision.block_reason = f"Regal gate failed: {', '.join(decision.failed_regals)}"
    
    # Check audit delta
    if audit_delta_success is not None:
        decision.audit_delta_success = audit_delta_success
        if audit_delta_success < min_audit_delta:
            decision.audit_regression_detected = True
            if decision.allow_deploy:  # Only override if not already blocked
                decision.allow_deploy = False
                decision.block_reason = f"Audit regression: {audit_delta_success:.2%} < {min_audit_delta:.2%}"
    
    # Compute decision SHA
    decision.compute_sha()
    
    return decision


# =============================================================================
# Phase 6: Typed Deploy Gate (fully causal)
# =============================================================================


def create_deploy_gate_inputs(
    regal_result: Optional[LedgerRegalV1] = None,
    audit_delta_success: Optional[float] = None,
    audit_delta_error: Optional[float] = None,
    audit_delta_mpl: Optional[float] = None,
    run_manifest_sha: Optional[str] = None,
    ledger_record_sha: Optional[str] = None,
    trajectory_audit_sha: Optional[str] = None,
    econ_tensor_sha: Optional[str] = None,
    deploy_threshold_success: float = 0.0,
    deploy_threshold_mpl: float = 0.0,
) -> DeployGateInputsV1:
    """Create typed deploy gate inputs for causal replay.

    Args:
        regal_result: LedgerRegalV1 from regal evaluation
        audit_delta_success: Delta success rate from audit
        audit_delta_error: Delta error rate from audit
        audit_delta_mpl: Delta MPL from audit
        run_manifest_sha: SHA of run manifest
        ledger_record_sha: SHA of ledger record
        trajectory_audit_sha: SHA of trajectory audit
        econ_tensor_sha: SHA of econ tensor
        deploy_threshold_success: Minimum delta_success to allow
        deploy_threshold_mpl: Minimum delta_mpl to allow

    Returns:
        DeployGateInputsV1 with all inputs for deterministic decision
    """
    regal_all_passed = True
    regal_degraded = regal_result is None
    regal_report_sha = None

    if regal_result is not None:
        regal_all_passed = regal_result.all_passed
        regal_report_sha = sha256_json([r.report_sha for r in regal_result.reports])

    return DeployGateInputsV1(
        audit_delta_success=audit_delta_success,
        audit_delta_error=audit_delta_error,
        audit_delta_mpl=audit_delta_mpl,
        regal_all_passed=regal_all_passed,
        regal_degraded=regal_degraded,
        regal_report_sha=regal_report_sha,
        run_manifest_sha=run_manifest_sha,
        ledger_record_sha=ledger_record_sha,
        deploy_threshold_success=deploy_threshold_success,
        deploy_threshold_mpl=deploy_threshold_mpl,
        trajectory_audit_sha=trajectory_audit_sha,
        econ_tensor_sha=econ_tensor_sha,
    )


def compute_deploy_decision(
    inputs: DeployGateInputsV1,
    require_regal: bool = True,
) -> DeployGateDecisionV1:
    """Compute deploy decision deterministically from inputs.

    This is the canonical deploy gate: same inputs -> same decision.

    Args:
        inputs: Typed deploy gate inputs
        require_regal: If True, missing regal blocks deploy

    Returns:
        DeployGateDecisionV1 with decision and full provenance
    """
    inputs_sha = inputs.sha256()
    checks_performed: List[Dict[str, Any]] = []
    allow_deploy = True
    reason_parts: List[str] = []

    # Check 1: Regal degraded (missing)
    if inputs.regal_degraded:
        checks_performed.append({
            "check": "regal_not_degraded",
            "passed": not require_regal,
            "detail": "Regal evaluation missing",
        })
        if require_regal:
            allow_deploy = False
            reason_parts.append("regal_missing")

    # Check 2: Regal all passed
    if not inputs.regal_degraded and not inputs.regal_all_passed:
        checks_performed.append({
            "check": "regal_all_passed",
            "passed": False,
            "detail": "Regal gate failed",
        })
        allow_deploy = False
        reason_parts.append("regal_failed")
    elif not inputs.regal_degraded:
        checks_performed.append({
            "check": "regal_all_passed",
            "passed": True,
            "detail": "Regal gate passed",
        })

    # Check 3: Audit delta success threshold
    if inputs.audit_delta_success is not None:
        passed = inputs.audit_delta_success >= inputs.deploy_threshold_success
        checks_performed.append({
            "check": "audit_delta_success_threshold",
            "passed": passed,
            "detail": f"{inputs.audit_delta_success:.4f} >= {inputs.deploy_threshold_success}",
        })
        if not passed:
            allow_deploy = False
            reason_parts.append(f"audit_regression_{inputs.audit_delta_success:.2%}")

    # Check 4: Audit delta MPL threshold
    if inputs.audit_delta_mpl is not None:
        passed = inputs.audit_delta_mpl >= inputs.deploy_threshold_mpl
        checks_performed.append({
            "check": "audit_delta_mpl_threshold",
            "passed": passed,
            "detail": f"{inputs.audit_delta_mpl:.4f} >= {inputs.deploy_threshold_mpl}",
        })
        if not passed:
            allow_deploy = False
            reason_parts.append(f"mpl_regression_{inputs.audit_delta_mpl:.4f}")

    # Check 5: Verification all passed (CRITICAL for FULL runs)
    # This is the key causal link: verification failures → deploy blocked
    if inputs.is_full_regality_run:
        if not inputs.verification_all_passed:
            checks_performed.append({
                "check": "verification_required_for_full",
                "passed": False,
                "detail": f"Verification failed: {inputs.verification_blocking_failures} blocking checks failed",
                "blocking_check_ids": inputs.verification_blocking_check_ids,
            })
            allow_deploy = False
            blocking_ids = ",".join(inputs.verification_blocking_check_ids[:3])
            if len(inputs.verification_blocking_check_ids) > 3:
                blocking_ids += "..."
            reason_parts.append(f"verification_failed:{blocking_ids}")
        else:
            checks_performed.append({
                "check": "verification_required_for_full",
                "passed": True,
                "detail": "Verification passed",
            })

    reason = ", ".join(reason_parts) if reason_parts else "all_checks_passed"

    return DeployGateDecisionV1(
        allow_deploy=allow_deploy,
        reason=reason,
        inputs_sha=inputs_sha,
        inputs=inputs,
        checks_performed=checks_performed,
    )


def write_deploy_gate_inputs(path: str, inputs: DeployGateInputsV1) -> str:
    """Write deploy gate inputs to JSON file.

    Args:
        path: Output path
        inputs: Deploy gate inputs to write

    Returns:
        SHA-256 of written file content
    """
    from src.utils.config_digest import sha256_file
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(inputs.model_dump(mode="json"), f, indent=2)
    return sha256_file(str(output_path))


def write_deploy_gate_decision(path: str, decision: DeployGateDecisionV1) -> str:
    """Write deploy gate decision to JSON file.

    Args:
        path: Output path
        decision: Deploy gate decision to write

    Returns:
        SHA-256 of written file content
    """
    from src.utils.config_digest import sha256_file
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(decision.model_dump(mode="json"), f, indent=2)
    return sha256_file(str(output_path))


__all__ = [
    # Legacy
    "DeployGateDecision",
    "check_deploy_gate",
    # Phase 6: Typed deploy gate
    "create_deploy_gate_inputs",
    "compute_deploy_decision",
    "write_deploy_gate_inputs",
    "write_deploy_gate_decision",
]
