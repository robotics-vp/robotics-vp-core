"""Isomorphism adapters for representation space alignment.

This module provides adapters that align representations across different
spaces using invertible-ish / topology-preserving mappings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.representation.space import (
    RepresentationPayload,
    InvariantReport,
    IsomorphismAdapter,
)


@dataclass
class AlignmentReport:
    """Report from isomorphism adapter alignment."""

    alignment_error: float = 0.0  # Mean alignment error
    cycle_error: float = 0.0  # A->B->A reconstruction error
    rank: int = 0  # Effective rank of the mapping
    condition_number: float = 1.0  # Conditioning of the transform
    per_sample_errors: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LinearAlign(IsomorphismAdapter):
    """Linear alignment adapter using Procrustes / whitening + rotation.

    Fits an orthogonal transformation (rotation + scaling) to align
    source representations to target representations.

    Properties:
    - Trainable on portable artifacts (no raw frames required)
    - Exports a versioned mapping
    - Supports cycle-consistency checks (A->B->A error)
    """

    def __init__(
        self,
        source_name: str = "source",
        target_name: str = "target",
        whiten: bool = True,
        regularization: float = 1e-6,
    ):
        """Initialize linear alignment adapter.

        Args:
            source_name: Name of source representation space
            target_name: Name of target representation space
            whiten: Whether to apply whitening before rotation
            regularization: Regularization for covariance inversion
        """
        self.source_name = source_name
        self.target_name = target_name
        self.whiten = whiten
        self.regularization = regularization

        # Fitted parameters (initialized as identity)
        self._rotation: Optional[np.ndarray] = None  # (dim, dim) orthogonal
        self._source_mean: Optional[np.ndarray] = None  # (dim,)
        self._target_mean: Optional[np.ndarray] = None  # (dim,)
        self._source_scale: Optional[np.ndarray] = None  # (dim,) or scalar
        self._target_scale: Optional[np.ndarray] = None  # (dim,) or scalar
        self._dim: int = 0
        self._fitted: bool = False
        self._version: str = "linear_align::v1"

    def fit(
        self,
        source_payloads: List[RepresentationPayload],
        target_payloads: List[RepresentationPayload],
    ) -> "LinearAlign":
        """Fit orthogonal Procrustes alignment.

        Uses SVD to find the optimal rotation matrix that minimizes
        ||target - source @ R||^2

        Args:
            source_payloads: Payloads from source space
            target_payloads: Payloads from target space

        Returns:
            Self for chaining
        """
        if len(source_payloads) != len(target_payloads):
            raise ValueError("Source and target payloads must have same length")
        if len(source_payloads) == 0:
            raise ValueError("Need at least one sample to fit")

        # Stack pooled features
        source = np.stack([p.pooled() for p in source_payloads])
        target = np.stack([p.pooled() for p in target_payloads])

        n_samples, self._dim = source.shape

        # Center the data
        self._source_mean = source.mean(axis=0)
        self._target_mean = target.mean(axis=0)
        source_centered = source - self._source_mean
        target_centered = target - self._target_mean

        # Optional whitening (normalize variance)
        if self.whiten:
            source_var = np.var(source_centered, axis=0) + self.regularization
            target_var = np.var(target_centered, axis=0) + self.regularization
            self._source_scale = np.sqrt(source_var)
            self._target_scale = np.sqrt(target_var)
            source_centered = source_centered / self._source_scale
            target_centered = target_centered / self._target_scale
        else:
            self._source_scale = np.ones(self._dim)
            self._target_scale = np.ones(self._dim)

        # Compute optimal rotation via Orthogonal Procrustes
        # Using scipy.linalg.orthogonal_procrustes for robust implementation
        from scipy.linalg import orthogonal_procrustes
        R, scale = orthogonal_procrustes(source_centered, target_centered)
        self._rotation = R

        self._fitted = True
        return self

    def transform(self, payload: RepresentationPayload) -> RepresentationPayload:
        """Transform payload from source to target space.

        Args:
            payload: Payload in source space

        Returns:
            Transformed payload in target space
        """
        if not self._fitted:
            raise RuntimeError("Adapter not fitted. Call fit() first.")

        features = payload.pooled()
        if features.shape[-1] != self._dim:
            raise ValueError(f"Expected dim {self._dim}, got {features.shape[-1]}")

        # Apply transform: center, scale, rotate, unscale, uncenter
        centered = features - self._source_mean
        if self.whiten:
            centered = centered / self._source_scale
        rotated = centered @ self._rotation
        if self.whiten:
            rotated = rotated * self._target_scale
        transformed = rotated + self._target_mean

        return RepresentationPayload(
            features=transformed,
            dim=int(transformed.shape[-1]),
            metadata={
                **payload.metadata,
                "adapter": self._version,
                "source": self.source_name,
                "target": self.target_name,
            },
            version=payload.version,
        )

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
        if not self._fitted:
            raise RuntimeError("Adapter not fitted. Call fit() first.")

        features = payload.pooled()
        if features.shape[-1] != self._dim:
            raise ValueError(f"Expected dim {self._dim}, got {features.shape[-1]}")

        # Inverse transform: center, scale, inverse rotate, unscale, uncenter
        centered = features - self._target_mean
        if self.whiten:
            centered = centered / self._target_scale
        rotated = centered @ self._rotation.T  # Inverse of orthogonal is transpose
        if self.whiten:
            rotated = rotated * self._source_scale
        transformed = rotated + self._source_mean

        return RepresentationPayload(
            features=transformed,
            dim=int(transformed.shape[-1]),
            metadata={
                **payload.metadata,
                "adapter": self._version,
                "source": self.target_name,
                "target": self.source_name,
                "inverse": True,
            },
            version=payload.version,
        )

    def cycle_error(self, source_payloads: List[RepresentationPayload]) -> float:
        """Compute A->B->A cycle consistency error.

        Args:
            source_payloads: Payloads in source space

        Returns:
            Mean reconstruction error (L2 distance after round-trip)
        """
        if not self._fitted:
            raise RuntimeError("Adapter not fitted. Call fit() first.")

        errors = []
        for payload in source_payloads:
            original = payload.pooled()
            transformed = self.transform(payload)
            reconstructed = self.inverse_transform(transformed).pooled()
            error = float(np.linalg.norm(original - reconstructed))
            errors.append(error)

        return float(np.mean(errors)) if errors else 0.0

    def alignment_report(
        self,
        source_payloads: List[RepresentationPayload],
        target_payloads: List[RepresentationPayload],
    ) -> AlignmentReport:
        """Generate comprehensive alignment report.

        Args:
            source_payloads: Payloads from source space
            target_payloads: Ground truth payloads from target space

        Returns:
            AlignmentReport with alignment and cycle errors
        """
        if not self._fitted:
            raise RuntimeError("Adapter not fitted. Call fit() first.")

        # Compute per-sample alignment errors
        errors = []
        for source, target in zip(source_payloads, target_payloads):
            predicted = self.transform(source).pooled()
            actual = target.pooled()
            error = float(np.linalg.norm(predicted - actual))
            errors.append(error)

        # Compute cycle consistency error
        cycle = self.cycle_error(source_payloads)

        # Compute condition number of the rotation matrix
        cond = float(np.linalg.cond(self._rotation)) if self._rotation is not None else 1.0

        return AlignmentReport(
            alignment_error=float(np.mean(errors)) if errors else 0.0,
            cycle_error=cycle,
            rank=int(np.linalg.matrix_rank(self._rotation)) if self._rotation is not None else 0,
            condition_number=cond,
            per_sample_errors=errors,
            metadata={
                "source": self.source_name,
                "target": self.target_name,
                "dim": self._dim,
                "whiten": self.whiten,
                "num_samples": len(source_payloads),
            },
        )

    def export(self) -> Dict[str, Any]:
        """Export adapter parameters for serialization."""
        if not self._fitted:
            raise RuntimeError("Adapter not fitted. Call fit() first.")

        return {
            "version": self._version,
            "source_name": self.source_name,
            "target_name": self.target_name,
            "whiten": self.whiten,
            "regularization": self.regularization,
            "dim": self._dim,
            "rotation": self._rotation.tolist(),
            "source_mean": self._source_mean.tolist(),
            "target_mean": self._target_mean.tolist(),
            "source_scale": self._source_scale.tolist(),
            "target_scale": self._target_scale.tolist(),
        }

    @classmethod
    def from_export(cls, data: Dict[str, Any]) -> "LinearAlign":
        """Load adapter from exported parameters."""
        adapter = cls(
            source_name=data["source_name"],
            target_name=data["target_name"],
            whiten=data.get("whiten", True),
            regularization=data.get("regularization", 1e-6),
        )
        adapter._version = data["version"]
        adapter._dim = data["dim"]
        adapter._rotation = np.array(data["rotation"])
        adapter._source_mean = np.array(data["source_mean"])
        adapter._target_mean = np.array(data["target_mean"])
        adapter._source_scale = np.array(data["source_scale"])
        adapter._target_scale = np.array(data["target_scale"])
        adapter._fitted = True
        return adapter


__all__ = [
    "AlignmentReport",
    "LinearAlign",
]
