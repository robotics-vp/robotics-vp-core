"""Econ Basis Registry - Immutable basis definitions for econ tensors.

The basis registry stores frozen basis definitions that define the canonical
axis ordering for econ tensors. Treat basis as an API - once registered,
the axis ordering is immutable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.contracts.schemas import EconBasisSpecV1


# =============================================================================
# Registry
# =============================================================================

@dataclass(frozen=True)
class EconBasisDefinition:
    """Frozen wrapper for EconBasisSpecV1."""
    basis_id: str
    spec: EconBasisSpecV1

    @property
    def sha256(self) -> str:
        return self.spec.sha256()


# Global registry
_ECON_BASIS_REGISTRY: Dict[str, EconBasisDefinition] = {}


def register_basis(spec: EconBasisSpecV1) -> EconBasisDefinition:
    """Register a basis definition.

    Once registered, the basis is immutable. Attempting to register
    a different spec with the same basis_id will raise an error.
    """
    if spec.basis_id in _ECON_BASIS_REGISTRY:
        existing = _ECON_BASIS_REGISTRY[spec.basis_id]
        if existing.sha256 != spec.sha256():
            raise ValueError(
                f"Basis '{spec.basis_id}' already registered with different spec. "
                f"Existing SHA: {existing.sha256[:16]}..., New SHA: {spec.sha256()[:16]}..."
            )
        return existing

    definition = EconBasisDefinition(basis_id=spec.basis_id, spec=spec)
    _ECON_BASIS_REGISTRY[spec.basis_id] = definition
    return definition


def get_basis(basis_id: str) -> Optional[EconBasisDefinition]:
    """Get a registered basis definition by ID."""
    return _ECON_BASIS_REGISTRY.get(basis_id)


def get_basis_sha(basis_id: str) -> Optional[str]:
    """Get the SHA of a registered basis."""
    defn = get_basis(basis_id)
    return defn.sha256 if defn else None


def list_bases() -> Dict[str, str]:
    """List all registered bases with their SHAs."""
    return {k: v.sha256 for k, v in _ECON_BASIS_REGISTRY.items()}


# =============================================================================
# Default Basis: econ_basis_v1
# =============================================================================

# Canonical axis ordering matching EconVector fields
ECON_BASIS_V1_AXES = [
    "mpl_units_per_hour",   # Marginal product of labor (units/hour)
    "wage_parity",          # Wage parity ratio (dimensionless)
    "energy_cost",          # Energy cost (currency units)
    "damage_cost",          # Damage/collision cost (currency units)
    "novelty_delta",        # Novelty improvement delta (dimensionless)
    "reward_scalar_sum",    # Sum of scalar rewards (dimensionless)
    "mobility_penalty",     # Mobility penalty (dimensionless)
    "throughput",           # Units processed per hour
    "error_rate",           # Error rate (0-1)
    "success_rate",         # Success rate (0-1)
]

ECON_BASIS_V1_UNITS = {
    "mpl_units_per_hour": "units/hour",
    "wage_parity": "ratio",
    "energy_cost": "currency",
    "damage_cost": "currency",
    "novelty_delta": "delta",
    "reward_scalar_sum": "scalar",
    "mobility_penalty": "penalty",
    "throughput": "units/hour",
    "error_rate": "fraction",
    "success_rate": "fraction",
}

# Default normalization scales (1.0 = no scaling)
ECON_BASIS_V1_SCALES = {
    "mpl_units_per_hour": 1.0,
    "wage_parity": 1.0,
    "energy_cost": 1.0,
    "damage_cost": 1.0,
    "novelty_delta": 1.0,
    "reward_scalar_sum": 1.0,
    "mobility_penalty": 1.0,
    "throughput": 1.0,
    "error_rate": 1.0,
    "success_rate": 1.0,
}

# Create and register the default basis
ECON_BASIS_V1 = EconBasisSpecV1(
    schema_version="v1",
    basis_id="econ_basis_v1",
    axes=ECON_BASIS_V1_AXES,
    units=ECON_BASIS_V1_UNITS,
    scales=ECON_BASIS_V1_SCALES,
    missing_policy="zero_fill",
)

# Register on module load
_DEFAULT_BASIS = register_basis(ECON_BASIS_V1)


def get_default_basis() -> EconBasisDefinition:
    """Get the default econ basis (econ_basis_v1)."""
    return _DEFAULT_BASIS


__all__ = [
    "EconBasisDefinition",
    "register_basis",
    "get_basis",
    "get_basis_sha",
    "list_bases",
    "get_default_basis",
    "ECON_BASIS_V1",
    "ECON_BASIS_V1_AXES",
]
