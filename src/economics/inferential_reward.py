"""Additive inferential reward compilation for successor training/budget paths.

This module does not modify the stable Phase B reward path. It compiles a
bounded inferential view over value, signal yield, and risk so successor
controllers can reason about adaptation spend without rewriting legacy math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp_unit(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


@dataclass(frozen=True)
class InferentialSignalYield:
    """Bounded decomposition of signal-yield support."""

    frontier_term: float
    epiplexity_term: float
    transfer_term: float
    quality_factor: float
    score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "frontier_term": float(self.frontier_term),
            "epiplexity_term": float(self.epiplexity_term),
            "transfer_term": float(self.transfer_term),
            "quality_factor": float(self.quality_factor),
            "score": float(self.score),
        }


@dataclass(frozen=True)
class InferentialRewardBreakdown:
    """Compiled successor inferential reward terms."""

    base_value_gain: float
    adaptation_gain: float
    data_value_support: float
    signal_yield: InferentialSignalYield
    compute_cost: float
    risk_cost: float
    uncertainty_penalty: float
    governance_penalty: float
    expected_gain: float
    expected_cost: float
    expected_risk: float
    net_benefit: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_value_gain": float(self.base_value_gain),
            "adaptation_gain": float(self.adaptation_gain),
            "data_value_support": float(self.data_value_support),
            "signal_yield": self.signal_yield.to_dict(),
            "compute_cost": float(self.compute_cost),
            "risk_cost": float(self.risk_cost),
            "uncertainty_penalty": float(self.uncertainty_penalty),
            "governance_penalty": float(self.governance_penalty),
            "expected_gain": float(self.expected_gain),
            "expected_cost": float(self.expected_cost),
            "expected_risk": float(self.expected_risk),
            "net_benefit": float(self.net_benefit),
        }


def compile_signal_yield(
    *,
    frontier_gain: float = 0.0,
    epiplexity_delta: float = 0.0,
    epiplexity_confidence: float = 0.0,
    transfer_score: float = 0.0,
    data_quality: Optional[float] = None,
    provenance_quality: Optional[float] = None,
) -> InferentialSignalYield:
    """Compile a bounded signal-yield score from learnability and data value."""
    frontier_term = max(0.0, _safe_float(frontier_gain))
    epiplexity_term = max(0.0, _safe_float(epiplexity_delta)) * _clamp_unit(epiplexity_confidence)
    transfer_term = max(0.0, _safe_float(transfer_score))

    quality_inputs = []
    if data_quality is not None:
        quality_inputs.append(_clamp_unit(data_quality))
    if provenance_quality is not None:
        quality_inputs.append(_clamp_unit(provenance_quality))
    quality_factor = max(0.25, sum(quality_inputs) / len(quality_inputs)) if quality_inputs else 1.0

    score = quality_factor * (epiplexity_term + 0.5 * frontier_term + 0.25 * transfer_term)
    return InferentialSignalYield(
        frontier_term=float(frontier_term),
        epiplexity_term=float(epiplexity_term),
        transfer_term=float(transfer_term),
        quality_factor=float(quality_factor),
        score=float(score),
    )


def compute_inferential_replay_weight(
    *,
    signal_yield_score: float,
    trust_score: float = 0.5,
) -> float:
    """Translate signal yield into a bounded replay weight."""
    return max(0.0, _safe_float(signal_yield_score)) * max(0.25, _clamp_unit(trust_score, default=0.5))


def compile_inferential_reward(
    *,
    expected_value_gain: float,
    expected_adaptation_benefit: float,
    learned_data_value: float,
    compute_cost: float,
    risk_cost: float,
    uncertainty: float,
    ood_score: float,
    data_quality: float,
    provenance_quality: float,
    frontier_gain: float = 0.0,
    epiplexity_delta: float = 0.0,
    epiplexity_confidence: float = 0.0,
    transfer_score: float = 0.0,
    governance_penalty: float = 0.0,
    signal_yield_override: Optional[float] = None,
) -> InferentialRewardBreakdown:
    """Compile successor inferential reward terms without touching legacy reward math."""
    signal_yield = compile_signal_yield(
        frontier_gain=frontier_gain,
        epiplexity_delta=epiplexity_delta,
        epiplexity_confidence=epiplexity_confidence,
        transfer_score=transfer_score,
        data_quality=data_quality,
        provenance_quality=provenance_quality,
    )
    if signal_yield_override is not None:
        signal_yield = InferentialSignalYield(
            frontier_term=signal_yield.frontier_term,
            epiplexity_term=signal_yield.epiplexity_term,
            transfer_term=signal_yield.transfer_term,
            quality_factor=signal_yield.quality_factor,
            score=float(max(0.0, _safe_float(signal_yield_override))),
        )

    base_value_gain = _safe_float(expected_value_gain)
    adaptation_gain = _safe_float(expected_adaptation_benefit)
    data_value_support = 0.25 * max(0.0, _safe_float(learned_data_value))
    compute_term = max(0.0, _safe_float(compute_cost))
    risk_term = max(0.0, _safe_float(risk_cost))
    uncertainty_penalty = 0.5 * max(0.0, _safe_float(uncertainty)) + 0.5 * max(0.0, _safe_float(ood_score))
    governance_term = max(0.0, _safe_float(governance_penalty))

    expected_gain = base_value_gain + adaptation_gain + data_value_support + signal_yield.score
    expected_cost = compute_term
    expected_risk = risk_term + uncertainty_penalty + governance_term
    net_benefit = expected_gain - expected_cost - expected_risk

    return InferentialRewardBreakdown(
        base_value_gain=float(base_value_gain),
        adaptation_gain=float(adaptation_gain),
        data_value_support=float(data_value_support),
        signal_yield=signal_yield,
        compute_cost=float(expected_cost),
        risk_cost=float(risk_term),
        uncertainty_penalty=float(uncertainty_penalty),
        governance_penalty=float(governance_term),
        expected_gain=float(expected_gain),
        expected_cost=float(expected_cost),
        expected_risk=float(expected_risk),
        net_benefit=float(net_benefit),
    )


__all__ = [
    "InferentialSignalYield",
    "InferentialRewardBreakdown",
    "compile_signal_yield",
    "compute_inferential_replay_weight",
    "compile_inferential_reward",
]
