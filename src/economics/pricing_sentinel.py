"""Deterministic shadow pricing sentinel for episode and window pricing ticks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from src.economics.econ_tensor import EconTensor
from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class PricingPolicy:
    """Explicit, auditable policy parameters for shadow pricing."""

    policy_id: str = "shadow_pricing_default_v1"
    base_task_hour_usd: float = 28.0
    econ_price_weight: float = 1.0
    frontier_gain_weight: float = 6.5
    constraint_penalty_weight: float = 4.5
    hard_violation_penalty: float = 8.0
    soft_violation_penalty: float = 2.5
    uncertainty_penalty_weight: float = 10.0
    uncertainty_discount_multiplier: float = 1.0
    data_share_credit_weight: float = 5.0
    max_credit_fraction: float = 0.35
    min_net_customer_rate: float = 8.0
    max_net_customer_rate: float = 120.0
    confidence_floor: float = 0.05
    confidence_high_threshold: float = 0.75
    confidence_medium_threshold: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "base_task_hour_usd": float(self.base_task_hour_usd),
            "econ_price_weight": float(self.econ_price_weight),
            "frontier_gain_weight": float(self.frontier_gain_weight),
            "constraint_penalty_weight": float(self.constraint_penalty_weight),
            "hard_violation_penalty": float(self.hard_violation_penalty),
            "soft_violation_penalty": float(self.soft_violation_penalty),
            "uncertainty_penalty_weight": float(self.uncertainty_penalty_weight),
            "uncertainty_discount_multiplier": float(self.uncertainty_discount_multiplier),
            "data_share_credit_weight": float(self.data_share_credit_weight),
            "max_credit_fraction": float(self.max_credit_fraction),
            "min_net_customer_rate": float(self.min_net_customer_rate),
            "max_net_customer_rate": float(self.max_net_customer_rate),
            "confidence_floor": float(self.confidence_floor),
            "confidence_high_threshold": float(self.confidence_high_threshold),
            "confidence_medium_threshold": float(self.confidence_medium_threshold),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PricingPolicy":
        return cls(
            policy_id=str(payload.get("policy_id", cls.policy_id)),
            base_task_hour_usd=float(payload.get("base_task_hour_usd", cls.base_task_hour_usd)),
            econ_price_weight=float(payload.get("econ_price_weight", cls.econ_price_weight)),
            frontier_gain_weight=float(payload.get("frontier_gain_weight", cls.frontier_gain_weight)),
            constraint_penalty_weight=float(payload.get("constraint_penalty_weight", cls.constraint_penalty_weight)),
            hard_violation_penalty=float(payload.get("hard_violation_penalty", cls.hard_violation_penalty)),
            soft_violation_penalty=float(payload.get("soft_violation_penalty", cls.soft_violation_penalty)),
            uncertainty_penalty_weight=float(payload.get("uncertainty_penalty_weight", cls.uncertainty_penalty_weight)),
            uncertainty_discount_multiplier=float(
                payload.get("uncertainty_discount_multiplier", cls.uncertainty_discount_multiplier)
            ),
            data_share_credit_weight=float(payload.get("data_share_credit_weight", cls.data_share_credit_weight)),
            max_credit_fraction=float(payload.get("max_credit_fraction", cls.max_credit_fraction)),
            min_net_customer_rate=float(payload.get("min_net_customer_rate", cls.min_net_customer_rate)),
            max_net_customer_rate=float(payload.get("max_net_customer_rate", cls.max_net_customer_rate)),
            confidence_floor=float(payload.get("confidence_floor", cls.confidence_floor)),
            confidence_high_threshold=float(
                payload.get("confidence_high_threshold", cls.confidence_high_threshold)
            ),
            confidence_medium_threshold=float(
                payload.get("confidence_medium_threshold", cls.confidence_medium_threshold)
            ),
        )


@dataclass(frozen=True)
class PricingTickInput:
    """Single tick input for episode-level or window-level shadow pricing."""

    run_id: str
    episode_id: str
    objective_profile_id: str
    source_domain: str
    timestamp: str
    mode: str
    econ_tensor: Mapping[str, Any] | EconTensor
    uncertainty: float = 0.0
    constraint_flags: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    trust_score: float = 1.0
    data_share_credit_override: Optional[float] = None
    tick_id: Optional[str] = None
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "objective_profile_id": self.objective_profile_id,
            "source_domain": self.source_domain,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "econ_tensor": _econ_axes(self.econ_tensor),
            "uncertainty": float(self.uncertainty),
            "constraint_flags": [dict(flag) for flag in self.constraint_flags],
            "trust_score": float(self.trust_score),
            "data_share_credit_override": self.data_share_credit_override,
            "tick_id": self.tick_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PricingTick:
    """Auditable pricing tick emitted by the shadow pricing sentinel."""

    tick_id: str
    run_id: str
    episode_id: str
    objective_profile_id: str
    source_domain: str
    timestamp: str
    mode: str
    task_hour_price_tick: float
    constraint_adjustment: float
    uncertainty_adjustment: float
    data_share_credit: float
    net_customer_rate: float
    confidence: float
    trust_annotation: str
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_id": self.tick_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "objective_profile_id": self.objective_profile_id,
            "source_domain": self.source_domain,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "task_hour_price_tick": float(self.task_hour_price_tick),
            "constraint_adjustment": float(self.constraint_adjustment),
            "uncertainty_adjustment": float(self.uncertainty_adjustment),
            "data_share_credit": float(self.data_share_credit),
            "net_customer_rate": float(self.net_customer_rate),
            "confidence": float(self.confidence),
            "trust_annotation": self.trust_annotation,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "metadata": dict(self.metadata),
        }


class PricingSentinel:
    """Emit deterministic pricing ticks from EconTensor and shadow evidence."""

    def __init__(self, policy: Optional[PricingPolicy] = None) -> None:
        self.policy = policy or PricingPolicy()

    @classmethod
    def from_path(cls, path: str | Path) -> "PricingSentinel":
        raw = Path(path).read_text()
        payload = json.loads(raw) if Path(path).suffix.lower() == ".json" else yaml.safe_load(raw)
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"Pricing policy at {path} must be a mapping")
        return cls(PricingPolicy.from_mapping(payload))

    def emit_tick(self, tick_input: PricingTickInput) -> PricingTick:
        econ_axes = _econ_axes(tick_input.econ_tensor)
        hard_flags = sum(1 for flag in tick_input.constraint_flags if str(flag.get("severity", "hard")) == "hard")
        soft_flags = sum(1 for flag in tick_input.constraint_flags if str(flag.get("severity", "")) == "soft")

        task_hour_price_tick = (
            self.policy.base_task_hour_usd
            + self.policy.econ_price_weight * max(0.0, econ_axes.get("price_tick", 0.0))
            + self.policy.frontier_gain_weight * max(0.0, econ_axes.get("marginal_frontier_gain", 0.0))
        )
        constraint_adjustment = -(
            self.policy.constraint_penalty_weight * max(0.0, econ_axes.get("constraint_penalty", 0.0))
            + self.policy.hard_violation_penalty * hard_flags
            + self.policy.soft_violation_penalty * soft_flags
        )
        uncertainty_adjustment = -(
            self.policy.uncertainty_penalty_weight * max(0.0, tick_input.uncertainty)
            + self.policy.uncertainty_discount_multiplier
            * max(0.0, econ_axes.get("uncertainty_discount", 0.0))
            * max(1.0, task_hour_price_tick)
        )

        confidence = max(
            self.policy.confidence_floor,
            min(
                1.0,
                float(tick_input.trust_score)
                * (1.0 - min(1.0, float(tick_input.uncertainty)))
                * (1.0 - min(0.85, 0.22 * hard_flags + 0.08 * soft_flags)),
            ),
        )
        derived_credit = (
            self.policy.data_share_credit_weight
            * max(0.0, econ_axes.get("marginal_frontier_gain", 0.0))
            * confidence
        )
        max_credit = max(0.0, task_hour_price_tick * self.policy.max_credit_fraction)
        requested_credit = tick_input.data_share_credit_override
        data_share_credit = min(max_credit, derived_credit if requested_credit is None else float(requested_credit))
        net_customer_rate = max(
            self.policy.min_net_customer_rate,
            min(
                self.policy.max_net_customer_rate,
                task_hour_price_tick + constraint_adjustment + uncertainty_adjustment - data_share_credit,
            ),
        )

        trust_annotation = "low"
        if confidence >= self.policy.confidence_high_threshold:
            trust_annotation = "high"
        elif confidence >= self.policy.confidence_medium_threshold:
            trust_annotation = "medium"

        tick_id = tick_input.tick_id or _deterministic_tick_id(
            {
                "run_id": tick_input.run_id,
                "episode_id": tick_input.episode_id,
                "mode": tick_input.mode,
                "start_step": tick_input.start_step,
                "end_step": tick_input.end_step,
                "timestamp": tick_input.timestamp,
                "objective_profile_id": tick_input.objective_profile_id,
            }
        )
        metadata = dict(tick_input.metadata)
        metadata.update(
            {
                "policy_id": self.policy.policy_id,
                "policy_hash": sha256_json(self.policy.to_dict()),
                "econ_axes": econ_axes,
                "hard_flag_count": hard_flags,
                "soft_flag_count": soft_flags,
            }
        )
        return PricingTick(
            tick_id=tick_id,
            run_id=tick_input.run_id,
            episode_id=tick_input.episode_id,
            objective_profile_id=tick_input.objective_profile_id,
            source_domain=tick_input.source_domain,
            timestamp=tick_input.timestamp,
            mode=tick_input.mode,
            task_hour_price_tick=float(task_hour_price_tick),
            constraint_adjustment=float(constraint_adjustment),
            uncertainty_adjustment=float(uncertainty_adjustment),
            data_share_credit=float(data_share_credit),
            net_customer_rate=float(net_customer_rate),
            confidence=float(confidence),
            trust_annotation=trust_annotation,
            start_step=tick_input.start_step,
            end_step=tick_input.end_step,
            metadata=metadata,
        )

    def emit_window_ticks(self, tick_inputs: Sequence[PricingTickInput]) -> list[PricingTick]:
        return [self.emit_tick(tick_input) for tick_input in tick_inputs]


def _deterministic_tick_id(payload: Mapping[str, Any]) -> str:
    return f"tick_{sha256_json(dict(payload))[:16]}"


def _econ_axes(econ_tensor: Mapping[str, Any] | EconTensor) -> Dict[str, float]:
    if isinstance(econ_tensor, EconTensor):
        return {
            axis: float(econ_tensor.values[index])
            for index, axis in enumerate(econ_tensor.schema.axes)
        }
    axes = econ_tensor.get("axes")
    values = econ_tensor.get("values")
    if isinstance(axes, Sequence) and isinstance(values, Sequence):
        return {str(axis): float(values[index]) for index, axis in enumerate(axes)}
    return {
        str(key): float(value)
        for key, value in dict(econ_tensor).items()
        if isinstance(value, (int, float))
    }
