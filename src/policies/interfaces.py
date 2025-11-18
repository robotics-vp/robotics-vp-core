"""
Policy interface definitions for Phase G.

All policies are deterministic, JSON-safe, and advisory-only. Inputs/outputs are
lightweight dict-like payloads to make swapping heuristic vs neural backends
transparent to callers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence


class DataValuationPolicy(Protocol):
    def build_features(
        self,
        datapack: Any,
        econ_slice: Optional[Dict[str, Any]] = None,
        semantic_tags: Optional[Sequence[Any]] = None,
        recap_scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build JSON-safe features from datapack metadata and optional context.
        """

    def score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return {"valuation_score": float, "metadata": {...}} without side effects.
        """


class PricingPolicy(Protocol):
    def build_features(
        self,
        task_econ: Dict[str, Any],
        datapack_value: Optional[float] = None,
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract pricing-relevant features (unit costs, spreads, rebates)."""

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return {"unit_price": float, "robot_hour_price": float, "metadata": {...}}.
        """


class SafetyRiskPolicy(Protocol):
    def build_features(self, events: Sequence[Any]) -> Dict[str, Any]:
        """Collect safety/damage evidence from episode events."""

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return {"risk_level": str, "damage_estimate": float, "metadata": {...}}.
        """


class EnergyCostPolicy(Protocol):
    def build_features(self, events: Sequence[Any]) -> Dict[str, Any]:
        """Extract energy-related signals from events or summaries."""

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Return {"energy_cost": float, "metadata": {...}} (JSON-safe)."""


class EpisodeQualityPolicy(Protocol):
    def build_features(
        self,
        rewards: Sequence[float],
        reward_components: Sequence[Dict[str, Any]],
        collisions: Sequence[Any],
        recap_scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct metrics used to judge recap goodness / anomalies."""

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return {"quality_score": float, "anomaly_score": float, "metadata": {...}}.
        """


class OrchestratorPolicy(Protocol):
    def advise(self, snapshot: Any) -> Any:
        """Return an OrchestratorAdvisory from a SemanticSnapshot."""


class SamplerWeightPolicy(Protocol):
    def build_features(self, descriptors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return per-descriptor features that will map to deterministic weights."""

    def evaluate(self, features: List[Dict[str, Any]], strategy: str) -> Dict[str, float]:
        """Return mapping of pack_id/episode_id -> weight."""


class MetaAdvisorPolicy(Protocol):
    def build_features(self, meta_slice: Any) -> Dict[str, Any]:
        """Collect contextual tensors/dicts for the meta-transformer abstraction."""

    def evaluate(self, features: Dict[str, Any]) -> Any:
        """Return MetaTransformerOutputs-equivalent structure."""


class VisionEncoderPolicy(Protocol):
    def encode(self, frame: Any) -> Any:
        """Return a deterministic VisionLatent for the given VisionFrame."""

    def batch_encode(self, frames: Sequence[Any]) -> List[Any]:
        """Vectorized encoding helper with deterministic ordering."""
