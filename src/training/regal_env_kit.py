"""RegalEnvKit - Protocol for regality-compliant environments.

Prevents "cute demo envs" from polluting the training graph by requiring:
- Explicit REGALITY_LEVEL declaration
- Required hooks for FULL regality (reward breakdown, trajectory audit)
- CI enforcement via test_env_regality_compliance.py

Usage:
    class MyEnv(RegalEnvKit):
        REGALITY_LEVEL = "FULL"
        env_type = "workcell"
        task_family = "dishwashing"
        ...
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable
from enum import Enum

from src.contracts.schemas import RewardBreakdownV1, TrajectoryAuditV1


class RegalityLevel(str, Enum):
    """Regality compliance level for environments."""
    
    FULL = "FULL"     # All hooks required, Stage6 eligible
    BASIC = "BASIC"   # Limited hooks, excluded from Stage6 by default


@runtime_checkable
class RegalEnvKit(Protocol):
    """Protocol for regality-compliant environments.
    
    FULL regality environments must implement all hooks.
    BASIC regality environments must include a reason string.
    
    CI rule enforces:
    - Any env under src/envs/* must declare REGALITY_LEVEL
    - FULL: must implement compute_reward_breakdown + build_trajectory_audit
    - BASIC: must include REGALITY_BASIC_REASON
    """
    
    # Required declarations
    REGALITY_LEVEL: Literal["FULL", "BASIC"]
    env_type: str
    task_family: str
    
    # Safety invariants (list of invariant names this env respects)
    safety_invariants: List[str]
    
    def compute_reward_breakdown(
        self,
        reward: float,
        info: Dict[str, Any],
    ) -> RewardBreakdownV1:
        """Produce structured reward breakdown.
        
        All envs with FULL regality must decompose rewards into
        standardized components for reward integrity verification.
        
        Args:
            reward: Total reward value
            info: Step info dict with component values
            
        Returns:
            RewardBreakdownV1 with all components
        """
        ...
    
    def build_trajectory_audit(
        self,
        episode_id: str,
        trajectory: List[Dict[str, Any]],
    ) -> TrajectoryAuditV1:
        """Produce trajectory audit from episode data.
        
        Summarizes episode telemetry for:
        - Spec violation detection
        - Physics coherence checks
        - Reward-work correlation analysis
        
        Args:
            episode_id: Unique episode identifier
            trajectory: List of step dicts
            
        Returns:
            TrajectoryAuditV1 with episode summary
        """
        ...


def validate_env_regality(env: Any) -> Dict[str, Any]:
    """Validate an environment's regality compliance.
    
    Args:
        env: Environment instance to validate
        
    Returns:
        Dict with validation results:
        - level: "FULL", "BASIC", or "NONE"
        - valid: bool
        - missing: list of missing requirements
        - reason: str if BASIC
    """
    result = {
        "level": "NONE",
        "valid": False,
        "missing": [],
        "reason": None,
    }
    
    # Check for REGALITY_LEVEL declaration
    level = getattr(env, "REGALITY_LEVEL", None)
    if level is None:
        env_class = getattr(env, "__class__", None)
        if env_class:
            level = getattr(env_class, "REGALITY_LEVEL", None)
    
    if level is None:
        result["missing"].append("REGALITY_LEVEL")
        return result
    
    result["level"] = level
    
    # Check required attributes
    required_attrs = ["env_type", "task_family"]
    for attr in required_attrs:
        if not hasattr(env, attr) and not hasattr(env.__class__, attr):
            result["missing"].append(attr)
    
    # Check safety_invariants
    if not hasattr(env, "safety_invariants") and not hasattr(env.__class__, "safety_invariants"):
        result["missing"].append("safety_invariants")
    
    if level == "FULL":
        # Check for required methods
        if not callable(getattr(env, "compute_reward_breakdown", None)):
            result["missing"].append("compute_reward_breakdown")
        if not callable(getattr(env, "build_trajectory_audit", None)):
            result["missing"].append("build_trajectory_audit")
    
    elif level == "BASIC":
        # Must have reason string
        reason = getattr(env, "REGALITY_BASIC_REASON", None)
        if reason is None:
            env_class = getattr(env, "__class__", None)
            if env_class:
                reason = getattr(env_class, "REGALITY_BASIC_REASON", None)
        
        if reason is None:
            result["missing"].append("REGALITY_BASIC_REASON")
        else:
            result["reason"] = reason
    
    result["valid"] = len(result["missing"]) == 0
    
    return result


def is_stage6_eligible(env: Any) -> bool:
    """Check if an environment is eligible for Stage6 training.
    
    Only FULL regality envs are Stage6 eligible.
    
    Args:
        env: Environment to check
        
    Returns:
        True if Stage6 eligible
    """
    validation = validate_env_regality(env)
    return validation["level"] == "FULL" and validation["valid"]


# Workcell is paramount - default env_type for Stage6
DEFAULT_ENV_TYPE = "workcell"


__all__ = [
    "RegalityLevel",
    "RegalEnvKit",
    "validate_env_regality",
    "is_stage6_eligible",
    "DEFAULT_ENV_TYPE",
]
