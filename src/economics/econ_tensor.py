"""Econ Tensor Conversion Utilities.

Provides deterministic conversion between econ dicts/vectors and EconTensorV1.
All conversions are stable and hashable for provenance.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.contracts.schemas import (
    EconBasisSpecV1,
    EconTensorV1,
    RegimeFeaturesV1,
)
from src.economics.econ_basis_registry import get_default_basis, get_basis

if TYPE_CHECKING:
    from src.ontology.models import EconVector


# =============================================================================
# Core Conversion Functions
# =============================================================================

def econ_to_tensor(
    econ_data: Dict[str, float],
    basis: Optional[EconBasisSpecV1] = None,
    regime_features: Optional[RegimeFeaturesV1] = None,
    source: str = "episode_metrics",
) -> EconTensorV1:
    """Convert an econ dict/vector to EconTensorV1.

    Args:
        econ_data: Dict mapping axis names to values
        basis: Basis spec to use (defaults to econ_basis_v1)
        regime_features: Optional regime features to link
        source: Source of the data

    Returns:
        EconTensorV1 with values ordered by basis axes
    """
    if basis is None:
        basis = get_default_basis().spec

    x: List[float] = []
    mask: Optional[List[bool]] = None

    if basis.missing_policy == "mask":
        mask = []

    for axis in basis.axes:
        if axis in econ_data:
            value = float(econ_data[axis])
            # Handle NaN/Inf by replacing with 0.0
            if math.isnan(value) or math.isinf(value):
                value = 0.0
            x.append(value)
            if mask is not None:
                mask.append(True)
        else:
            # Missing axis
            if basis.missing_policy == "zero_fill":
                x.append(0.0)
            else:  # mask
                x.append(0.0)  # Placeholder
                if mask is not None:
                    mask.append(False)

    # Compute stats for debugging
    non_zero = [v for v in x if v != 0.0]
    stats = {
        "norm": math.sqrt(sum(v * v for v in x)),
        "min": min(x) if x else 0.0,
        "max": max(x) if x else 0.0,
        "nnz": len(non_zero),
    }

    return EconTensorV1(
        basis_id=basis.basis_id,
        basis_sha=basis.sha256(),
        x=x,
        mask=mask,
        source=source,
        regime_features_sha=regime_features.sha256() if regime_features else None,
        stats=stats,
    )


def econ_vector_to_tensor(
    econ_vector: "EconVector",
    basis: Optional[EconBasisSpecV1] = None,
    regime_features: Optional[RegimeFeaturesV1] = None,
) -> EconTensorV1:
    """Convert an EconVector dataclass to EconTensorV1.

    Args:
        econ_vector: EconVector from ontology.models
        basis: Basis spec to use (defaults to econ_basis_v1)
        regime_features: Optional regime features to link

    Returns:
        EconTensorV1 with values ordered by basis axes
    """
    # Convert EconVector fields to dict
    econ_data = {
        "mpl_units_per_hour": econ_vector.mpl_units_per_hour,
        "wage_parity": econ_vector.wage_parity,
        "energy_cost": econ_vector.energy_cost,
        "damage_cost": econ_vector.damage_cost,
        "novelty_delta": econ_vector.novelty_delta,
        "reward_scalar_sum": econ_vector.reward_scalar_sum,
        "mobility_penalty": econ_vector.mobility_penalty,
    }

    return econ_to_tensor(
        econ_data,
        basis=basis,
        regime_features=regime_features,
        source="econ_vector",
    )


def tensor_to_econ_dict(
    tensor: EconTensorV1,
    basis: Optional[EconBasisSpecV1] = None,
) -> Dict[str, float]:
    """Convert EconTensorV1 back to a dict.

    Note: This is primarily for debugging/best-effort. The tensor is the
    canonical representation.

    Args:
        tensor: The econ tensor
        basis: Basis spec (if None, fetched from registry by basis_id)

    Returns:
        Dict mapping axis names to values
    """
    if basis is None:
        basis_defn = get_basis(tensor.basis_id)
        if basis_defn is None:
            raise ValueError(f"Unknown basis_id: {tensor.basis_id}")
        basis = basis_defn.spec

    if len(tensor.x) != len(basis.axes):
        raise ValueError(
            f"Tensor length {len(tensor.x)} doesn't match basis axes {len(basis.axes)}"
        )

    result: Dict[str, float] = {}
    for i, axis in enumerate(basis.axes):
        # If mask exists and indicates missing, skip
        if tensor.mask is not None and not tensor.mask[i]:
            continue
        result[axis] = tensor.x[i]

    return result


def hash_econ_tensor(tensor: EconTensorV1) -> str:
    """Compute stable SHA-256 hash of an econ tensor."""
    return tensor.sha256()


# =============================================================================
# Synthetic Tensor Generation (for testing/smoke)
# =============================================================================

def create_synthetic_econ_tensor(
    basis: Optional[EconBasisSpecV1] = None,
    seed: int = 42,
) -> EconTensorV1:
    """Create a synthetic econ tensor for testing.

    Args:
        basis: Basis spec to use
        seed: Random seed for reproducibility

    Returns:
        Synthetic EconTensorV1
    """
    import random
    rng = random.Random(seed)

    if basis is None:
        basis = get_default_basis().spec

    # Generate synthetic values
    econ_data: Dict[str, float] = {}
    for axis in basis.axes:
        if "rate" in axis or "fraction" in axis or "parity" in axis:
            # Bounded [0, 1]
            econ_data[axis] = rng.uniform(0.0, 1.0)
        elif "cost" in axis or "penalty" in axis:
            # Positive costs
            econ_data[axis] = rng.uniform(0.0, 10.0)
        elif "throughput" in axis or "mpl" in axis:
            # Throughput-like metrics
            econ_data[axis] = rng.uniform(0.0, 100.0)
        else:
            # General scalars
            econ_data[axis] = rng.uniform(-1.0, 1.0)

    return econ_to_tensor(
        econ_data,
        basis=basis,
        source="synthetic",
    )


# =============================================================================
# Integration Helpers
# =============================================================================

def extract_key_econ_values(tensor: EconTensorV1, basis: Optional[EconBasisSpecV1] = None) -> Dict[str, float]:
    """Extract key econ values for summary/debugging.

    Returns a small dict with the most important values.
    """
    econ_dict = tensor_to_econ_dict(tensor, basis)

    # Return subset of key metrics
    key_axes = ["mpl_units_per_hour", "success_rate", "energy_cost", "damage_cost"]
    return {k: v for k, v in econ_dict.items() if k in key_axes}


def compute_tensor_summary(tensor: EconTensorV1, basis: Optional[EconBasisSpecV1] = None) -> Dict[str, float]:
    """Compute a summary dict for ledger provenance.

    Returns norm and a couple key values.
    """
    summary: Dict[str, float] = {}

    # Include stats if present
    if tensor.stats:
        summary["norm"] = tensor.stats.get("norm", 0.0)

    # Include first few axis values
    key_values = extract_key_econ_values(tensor, basis)
    for k, v in list(key_values.items())[:3]:
        summary[k] = v

    return summary


__all__ = [
    "econ_to_tensor",
    "econ_vector_to_tensor",
    "tensor_to_econ_dict",
    "hash_econ_tensor",
    "create_synthetic_econ_tensor",
    "extract_key_econ_values",
    "compute_tensor_summary",
]
