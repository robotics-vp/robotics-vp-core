"""Deployment gating module for regal-aware deploy decisions.

P2 stub: Provides typed deploy decision interface based on regal outputs.
Actual deployment logic is external; this module provides the decision API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.contracts.schemas import (
    LedgerRegalV1,
    RegalReportV1,
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


__all__ = [
    "DeployGateDecision",
    "check_deploy_gate",
]
