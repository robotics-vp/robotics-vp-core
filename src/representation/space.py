"""Representation space contracts and utilities.

This module defines the RepresentationSpace contract for composable/comparable
representations across env/task/channel sets with explicit isomorphism/alignment
mechanisms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class RepresentationPayload:
    """Payload containing encoded representation with metadata."""

    features: np.ndarray  # Shape: (dim,) or (seq_len, dim)
    dim: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "v1"

    def pooled(self) -> np.ndarray:
        """Return pooled representation (single vector)."""
        if self.features.ndim == 1:
            return self.features
        return self.features.mean(axis=0)


@dataclass
class InvariantReport:
    """Report of representation invariants/stability metrics."""

    norm_mean: float = 0.0
    norm_std: float = 0.0
    variance: float = 0.0
    entropy_proxy: float = 0.0  # Approximated via feature variance
    stability_score: float = 1.0  # 1.0 = stable, 0.0 = drifted
    metadata: Dict[str, Any] = field(default_factory=dict)


class RepresentationSpace(ABC):
    """Abstract base class for representation spaces.

    Defines a contract for:
    - encode: episode/artifacts -> representation payload
    - project: source payload -> target space payload
    - distance: compute distance between payloads
    - invariants: compute stability/validity metrics
    """

    def __init__(self, name: str, dim: int):
        """Initialize representation space.

        Args:
            name: Unique identifier for this representation space
            dim: Dimensionality of the representation
        """
        self.name = name
        self.dim = dim

    @abstractmethod
    def encode(
        self,
        episode_or_artifacts: Union[Dict[str, Any], Any],
    ) -> RepresentationPayload:
        """Encode episode or artifacts into this representation space.

        Args:
            episode_or_artifacts: Episode dict or artifact data

        Returns:
            RepresentationPayload with encoded features
        """
        ...

    def project(
        self,
        payload: RepresentationPayload,
        target_space: "RepresentationSpace",
        adapter: Optional["IsomorphismAdapter"] = None,
    ) -> RepresentationPayload:
        """Project payload from this space to target space.

        Args:
            payload: Payload in this space
            target_space: Target representation space
            adapter: Optional isomorphism adapter for the projection

        Returns:
            Projected payload in target space
        """
        if adapter is None:
            # Default: re-encode in target space (requires original data)
            raise ValueError(
                "Projection requires an adapter or the original data. "
                "Use target_space.encode() for re-encoding."
            )
        return adapter.transform(payload)

    def distance(
        self,
        a: RepresentationPayload,
        b: RepresentationPayload,
        metric: str = "cosine",
    ) -> float:
        """Compute distance between two payloads in this space.

        Args:
            a: First payload
            b: Second payload
            metric: Distance metric ('cosine', 'euclidean', 'l2')

        Returns:
            Distance value (0 = identical for cosine, larger = more different)
        """
        a_vec = a.pooled()
        b_vec = b.pooled()

        if metric == "cosine":
            a_norm = np.linalg.norm(a_vec)
            b_norm = np.linalg.norm(b_vec)
            if a_norm < 1e-8 or b_norm < 1e-8:
                return 1.0  # Max distance for zero vectors
            return 1.0 - float(np.dot(a_vec, b_vec) / (a_norm * b_norm))
        elif metric in ("euclidean", "l2"):
            return float(np.linalg.norm(a_vec - b_vec))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def invariants(self, payload: RepresentationPayload) -> InvariantReport:
        """Compute invariant/stability metrics for a payload.

        Args:
            payload: Representation payload

        Returns:
            InvariantReport with computed metrics
        """
        features = payload.features
        if features.ndim == 1:
            features = features.reshape(1, -1)

        norms = np.linalg.norm(features, axis=-1)
        variance = float(np.var(features))

        return InvariantReport(
            norm_mean=float(np.mean(norms)),
            norm_std=float(np.std(norms)),
            variance=variance,
            entropy_proxy=min(1.0, variance),  # Simplified entropy proxy
            stability_score=1.0,  # Default, updated by drift detection
            metadata={"dim": payload.dim, "num_samples": features.shape[0]},
        )

    def batch_invariants(
        self,
        payloads: List[RepresentationPayload],
    ) -> InvariantReport:
        """Compute aggregate invariants over a batch of payloads.

        Args:
            payloads: List of representation payloads

        Returns:
            Aggregated InvariantReport
        """
        if not payloads:
            return InvariantReport()

        all_features = np.stack([p.pooled() for p in payloads])
        norms = np.linalg.norm(all_features, axis=-1)
        variance = float(np.var(all_features))

        return InvariantReport(
            norm_mean=float(np.mean(norms)),
            norm_std=float(np.std(norms)),
            variance=variance,
            entropy_proxy=min(1.0, variance),
            stability_score=1.0,
            metadata={
                "dim": payloads[0].dim,
                "num_samples": len(payloads),
            },
        )


class IsomorphismAdapter(ABC):
    """Abstract base class for representation space adapters.

    Adapters provide invertible-ish / topology-preserving mappings
    between representation spaces.
    """

    @abstractmethod
    def fit(
        self,
        source_payloads: List[RepresentationPayload],
        target_payloads: List[RepresentationPayload],
    ) -> "IsomorphismAdapter":
        """Fit the adapter on paired source/target payloads.

        Args:
            source_payloads: Payloads from source space
            target_payloads: Payloads from target space

        Returns:
            Self for chaining
        """
        ...

    @abstractmethod
    def transform(self, payload: RepresentationPayload) -> RepresentationPayload:
        """Transform payload from source to target space.

        Args:
            payload: Payload in source space

        Returns:
            Transformed payload in target space
        """
        ...

    @abstractmethod
    def inverse_transform(
        self,
        payload: RepresentationPayload,
    ) -> RepresentationPayload:
        """Transform payload from target back to source space.

        Args:
            payload: Payload in target space

        Returns:
            Transformed payload in source space
        """
        ...

    @abstractmethod
    def export(self) -> Dict[str, Any]:
        """Export adapter parameters for serialization.

        Returns:
            Dict with adapter parameters
        """
        ...

    @classmethod
    @abstractmethod
    def from_export(cls, data: Dict[str, Any]) -> "IsomorphismAdapter":
        """Load adapter from exported parameters.

        Args:
            data: Dict with adapter parameters

        Returns:
            Loaded adapter instance
        """
        ...


__all__ = [
    "RepresentationPayload",
    "InvariantReport",
    "RepresentationSpace",
    "IsomorphismAdapter",
]
