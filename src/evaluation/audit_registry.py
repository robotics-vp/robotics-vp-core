"""Immutable audit suite registry.

Suite definitions are frozen by sha - changing them yields a new sha
that must be recorded in the ledger for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class AuditScenarioDefinition:
    """Immutable scenario definition."""

    scenario_id: str
    task_name: str
    task_family: Optional[str] = None
    num_episodes: int = 1
    env_config: Optional[tuple] = None  # Frozen tuple of (k,v) pairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "num_episodes": self.num_episodes,
            "env_config": dict(self.env_config) if self.env_config else None,
        }


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for which metrics to compute."""

    compute_success_rate: bool = True
    compute_error: bool = True
    compute_return: bool = True
    compute_energy: bool = True
    compute_mpl: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compute_success_rate": self.compute_success_rate,
            "compute_error": self.compute_error,
            "compute_return": self.compute_return,
            "compute_energy": self.compute_energy,
            "compute_mpl": self.compute_mpl,
        }


@dataclass(frozen=True)
class Tolerances:
    """Success/failure tolerances for audit metrics."""

    success_rate_min: float = 0.0
    error_max: float = 1.0
    regression_threshold: float = 0.05  # Max allowed regression

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success_rate_min": self.success_rate_min,
            "error_max": self.error_max,
            "regression_threshold": self.regression_threshold,
        }


@dataclass(frozen=True)
class AuditSuiteDefinition:
    """Immutable audit suite definition.

    Once registered, a suite definition is frozen. Any change yields
    a new sha that must be recorded in provenance.
    """

    suite_id: str
    schema_version: str = "v1"
    scenarios: tuple = ()  # Tuple of AuditScenarioDefinition
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)
    tolerances: Tolerances = field(default_factory=Tolerances)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "schema_version": self.schema_version,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "metrics_config": self.metrics_config.to_dict(),
            "tolerances": self.tolerances.to_dict(),
            "description": self.description,
        }

    def sha256(self) -> str:
        """Compute immutable SHA-256 of the suite definition."""
        return sha256_json(self.to_dict())


# =============================================================================
# Immutable Audit Registry
# =============================================================================

# Default smoke test suite
_SMOKE_AUDIT_SUITE = AuditSuiteDefinition(
    suite_id="smoke_audit_v1",
    schema_version="v1",
    scenarios=(
        AuditScenarioDefinition("balanced_01", "drawer_vase", "manipulation", 3),
        AuditScenarioDefinition("occluded_01", "drawer_vase_occluded", "manipulation", 3),
    ),
    metrics_config=MetricsConfig(),
    tolerances=Tolerances(success_rate_min=0.5, regression_threshold=0.1),
    description="Minimal smoke test audit suite",
)

# Standard regression suite
_REGRESSION_SUITE = AuditSuiteDefinition(
    suite_id="regression_v1",
    schema_version="v1",
    scenarios=(
        AuditScenarioDefinition("balanced_01", "drawer_vase", "manipulation", 5),
        AuditScenarioDefinition("occluded_01", "drawer_vase_occluded", "manipulation", 5),
        AuditScenarioDefinition("dynamic_01", "drawer_vase_dynamic", "manipulation", 5),
        AuditScenarioDefinition("static_01", "pick_place", "manipulation", 5),
    ),
    metrics_config=MetricsConfig(),
    tolerances=Tolerances(success_rate_min=0.6, regression_threshold=0.05),
    description="Standard regression test suite",
)


# Registry is immutable - add new suites but don't modify existing
AUDIT_REGISTRY: Dict[str, AuditSuiteDefinition] = {
    "smoke_audit_v1": _SMOKE_AUDIT_SUITE,
    "regression_v1": _REGRESSION_SUITE,
}


def get_suite(suite_id: str) -> AuditSuiteDefinition:
    """Get audit suite definition by ID.

    Args:
        suite_id: Suite identifier

    Returns:
        AuditSuiteDefinition

    Raises:
        KeyError: If suite_id not in registry
    """
    if suite_id not in AUDIT_REGISTRY:
        raise KeyError(
            f"Unknown audit suite: {suite_id}. "
            f"Available: {list(AUDIT_REGISTRY.keys())}"
        )
    return AUDIT_REGISTRY[suite_id]


def list_suites() -> List[str]:
    """List available suite IDs."""
    return list(AUDIT_REGISTRY.keys())


def get_suite_sha(suite_id: str) -> str:
    """Get SHA-256 of a registered suite."""
    return get_suite(suite_id).sha256()


def register_suite(suite: AuditSuiteDefinition) -> str:
    """Register a new suite definition.

    Args:
        suite: Suite definition to register

    Returns:
        SHA-256 of the registered suite

    Raises:
        ValueError: If suite_id already exists with different sha
    """
    if suite.suite_id in AUDIT_REGISTRY:
        existing = AUDIT_REGISTRY[suite.suite_id]
        if existing.sha256() != suite.sha256():
            raise ValueError(
                f"Suite {suite.suite_id} already registered with different sha. "
                f"Existing: {existing.sha256()[:16]}, New: {suite.sha256()[:16]}"
            )
        return existing.sha256()

    AUDIT_REGISTRY[suite.suite_id] = suite
    return suite.sha256()


__all__ = [
    "AuditScenarioDefinition",
    "MetricsConfig",
    "Tolerances",
    "AuditSuiteDefinition",
    "AUDIT_REGISTRY",
    "get_suite",
    "list_suites",
    "get_suite_sha",
    "register_suite",
]
