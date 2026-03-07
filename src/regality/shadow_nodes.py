"""Deterministic shadow governance nodes for the economic control plane."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Protocol, Sequence

from src.utils.config_digest import sha256_json


class ShadowRegalStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class ShadowRegalContext:
    """Stable input bundle shared across shadow regal nodes."""

    run_id: str
    episode_id: str
    source_domain: str
    objective_tensor: Mapping[str, Any]
    objective_profile: Mapping[str, Any]
    compile_artifact: Mapping[str, Any]
    constraint_set: Mapping[str, Any]
    constraint_flags: Sequence[Mapping[str, Any]]
    econ_tensor: Mapping[str, Any]
    pricing_ticks: Sequence[Mapping[str, Any]]
    datapack_credit_update: Mapping[str, Any]
    episode_metrics: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    evidence_pointers: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "source_domain": self.source_domain,
            "objective_tensor": dict(self.objective_tensor),
            "objective_profile": dict(self.objective_profile),
            "compile_artifact": dict(self.compile_artifact),
            "constraint_set": dict(self.constraint_set),
            "constraint_flags": [dict(flag) for flag in self.constraint_flags],
            "econ_tensor": dict(self.econ_tensor),
            "pricing_ticks": [dict(tick) for tick in self.pricing_ticks],
            "datapack_credit_update": dict(self.datapack_credit_update),
            "episode_metrics": dict(self.episode_metrics),
            "provenance": dict(self.provenance),
            "evidence_pointers": dict(self.evidence_pointers),
        }


@dataclass(frozen=True)
class ShadowRegalDecision:
    """Typed output for a single shadow regal node."""

    node_id: str
    status: ShadowRegalStatus
    score: float
    reasons: list[str]
    evidence_pointers: Dict[str, str]
    recommended_action: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def decision_hash(self) -> str:
        return sha256_json(self._base_dict())

    def _base_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "score": float(self.score),
            "reasons": list(self.reasons),
            "evidence_pointers": dict(self.evidence_pointers),
            "recommended_action": self.recommended_action,
            "details": dict(self.details),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._base_dict()
        payload["decision_hash"] = self.decision_hash
        return payload


class ShadowRegalNode(Protocol):
    node_id: str

    def evaluate(self, context: ShadowRegalContext) -> ShadowRegalDecision:
        ...


@dataclass(frozen=True)
class ObjectiveIntegrityRegal:
    """Detect missing runtime contract fields and premature scalarization leakage."""

    node_id: str = "objective_integrity_regal"

    def evaluate(self, context: ShadowRegalContext) -> ShadowRegalDecision:
        objective_context = dict(context.objective_tensor.get("context", {}) or {})
        required_fields = (
            "task_id",
            "episode_id",
            "env_id",
            "world_id",
            "robot_id",
            "source_domain",
            "seed",
            "run_id",
            "timestamp",
            "schema_version_hash",
        )
        missing = [field for field in required_fields if field not in objective_context]
        profile_payload = dict(context.objective_profile.get("profile", context.objective_profile) or {})
        weight_axes = set((profile_payload.get("weights", {}) or {}).keys())
        objective_axes = set((context.objective_tensor.get("schema", {}) or {}).get("axes", []))
        compile_boundary = str(context.compile_artifact.get("scalarization_boundary", ""))
        scalarized_upstream = bool(context.compile_artifact.get("scalarized_upstream", False))
        profile_id_match = str(context.compile_artifact.get("objective_profile_id", "")) == str(profile_payload.get("profile_id", ""))

        reasons: list[str] = []
        status = ShadowRegalStatus.PASS
        score = 1.0
        if missing:
            status = ShadowRegalStatus.FAIL
            reasons.append("missing_objective_context_fields")
            score = 0.0
        if scalarized_upstream or compile_boundary != "contract_boundary":
            status = ShadowRegalStatus.FAIL
            reasons.append("premature_scalarization_detected")
            score = 0.0
        if weight_axes and not weight_axes.issubset(objective_axes):
            status = ShadowRegalStatus.FAIL
            reasons.append("profile_axis_mismatch")
            score = 0.0
        if not profile_id_match:
            status = ShadowRegalStatus.FAIL
            reasons.append("profile_compile_mismatch")
            score = 0.0
        if status == ShadowRegalStatus.PASS and not context.objective_tensor.get("provenance"):
            status = ShadowRegalStatus.WARN
            reasons.append("missing_objective_provenance")
            score = 0.7
        if not reasons:
            reasons.append("objective_contract_intact")
        return ShadowRegalDecision(
            node_id=self.node_id,
            status=status,
            score=score,
            reasons=reasons,
            evidence_pointers=dict(context.evidence_pointers),
            recommended_action="proceed_shadow" if status == ShadowRegalStatus.PASS else "repair_objective_contract",
            details={
                "missing_fields": missing,
                "compile_boundary": compile_boundary,
                "profile_id_match": profile_id_match,
            },
        )


@dataclass(frozen=True)
class PlausibilityRegal:
    """Check for malformed provenance and impossible metric combinations."""

    node_id: str = "plausibility_regal"

    def evaluate(self, context: ShadowRegalContext) -> ShadowRegalDecision:
        metrics = dict(context.episode_metrics or {})
        throughput = _metric(metrics, "throughput_units_per_hour")
        error = _metric(metrics, "error_rate")
        safety = _metric(metrics, "safety_score")
        duration_s = _metric(metrics, "duration_s")
        reasons: list[str] = []
        status = ShadowRegalStatus.PASS
        score = 1.0

        if throughput < 0.0 or error < 0.0 or error > 1.0 or safety < 0.0 or safety > 1.0:
            reasons.append("metric_range_impossible")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if throughput > 0.0 and duration_s <= 0.0:
            reasons.append("duration_missing_or_invalid")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if safety > 0.92 and error > 0.30:
            reasons.append("safety_error_conflict")
            status = ShadowRegalStatus.WARN if status != ShadowRegalStatus.FAIL else status
            score = min(score, 0.45)
        if not context.provenance:
            reasons.append("missing_provenance_bundle")
            status = ShadowRegalStatus.WARN if status != ShadowRegalStatus.FAIL else status
            score = min(score, 0.55)
        malformed_constraints = [
            axis
            for axis, spec in dict(context.constraint_set.get("hard_bounds", {}) or {}).items()
            if not isinstance(spec, Mapping) or not set(spec.keys()).intersection({"min", "max"})
        ]
        if malformed_constraints:
            reasons.append("malformed_constraints")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if not reasons:
            reasons.append("plausibility_ok")
        return ShadowRegalDecision(
            node_id=self.node_id,
            status=status,
            score=score,
            reasons=reasons,
            evidence_pointers=dict(context.evidence_pointers),
            recommended_action="inspect_metrics_and_provenance" if status != ShadowRegalStatus.PASS else "proceed_shadow",
            details={
                "throughput_units_per_hour": throughput,
                "error_rate": error,
                "safety_score": safety,
                "duration_s": duration_s,
                "malformed_constraints": malformed_constraints,
            },
        )


@dataclass(frozen=True)
class RewardSafetyRegal:
    """Detect obvious reward/accounting conflicts and reward hacking patterns."""

    node_id: str = "reward_safety_regal"
    exploit_ratio_threshold: float = 1.75

    def evaluate(self, context: ShadowRegalContext) -> ShadowRegalDecision:
        scalar_reward = float(context.compile_artifact.get("scalar_reward", 0.0))
        value_earned = _axis(context.econ_tensor, "value_earned")
        hard_flags = sum(1 for flag in context.constraint_flags if str(flag.get("severity", "hard")) == "hard")
        safety = _metric(context.episode_metrics, "safety_score")
        ratio = scalar_reward / max(0.1, value_earned) if scalar_reward > 0.0 else 0.0
        reasons: list[str] = []
        status = ShadowRegalStatus.PASS
        score = 1.0

        if scalar_reward > 0.0 and value_earned <= 0.0 and hard_flags > 0:
            reasons.append("positive_reward_with_negative_value_story")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if ratio >= self.exploit_ratio_threshold and (hard_flags > 0 or safety < 0.65):
            reasons.append("reward_econ_ratio_suspicious")
            status = ShadowRegalStatus.WARN if status != ShadowRegalStatus.FAIL else status
            score = min(score, 0.4)
        if not reasons:
            reasons.append("reward_story_consistent")
        return ShadowRegalDecision(
            node_id=self.node_id,
            status=status,
            score=score,
            reasons=reasons,
            evidence_pointers=dict(context.evidence_pointers),
            recommended_action="review_reward_accounting" if status != ShadowRegalStatus.PASS else "proceed_shadow",
            details={
                "scalar_reward": scalar_reward,
                "value_earned": value_earned,
                "reward_to_value_ratio": ratio,
                "hard_flag_count": hard_flags,
                "safety_score": safety,
            },
        )


@dataclass(frozen=True)
class PricingTruthRegal:
    """Verify pricing outputs remain consistent with econ inputs and trust."""

    node_id: str = "pricing_truth_regal"

    def evaluate(self, context: ShadowRegalContext) -> ShadowRegalDecision:
        if not context.pricing_ticks:
            return ShadowRegalDecision(
                node_id=self.node_id,
                status=ShadowRegalStatus.FAIL,
                score=0.0,
                reasons=["pricing_ticks_missing"],
                evidence_pointers=dict(context.evidence_pointers),
                recommended_action="recompute_pricing_ticks",
                details={},
            )

        tick = dict(context.pricing_ticks[0])
        net = float(tick.get("net_customer_rate", 0.0))
        gross = float(tick.get("task_hour_price_tick", 0.0))
        constraint_adjustment = float(tick.get("constraint_adjustment", 0.0))
        uncertainty_adjustment = float(tick.get("uncertainty_adjustment", 0.0))
        confidence = float(tick.get("confidence", 0.0))
        uncertainty = float(tick.get("metadata", {}).get("uncertainty", context.episode_metrics.get("uncertainty", 0.0)))
        hard_flags = sum(1 for flag in context.constraint_flags if str(flag.get("severity", "hard")) == "hard")
        marginal_gain = _axis(context.econ_tensor, "marginal_frontier_gain")
        reasons: list[str] = []
        status = ShadowRegalStatus.PASS
        score = 1.0

        if net - gross > 1e-6:
            reasons.append("net_rate_exceeds_gross_tick")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if constraint_adjustment > 0.0 or uncertainty_adjustment > 0.0:
            reasons.append("pricing_adjustment_sign_error")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if confidence > 0.85 and (uncertainty > 0.50 or hard_flags > 0):
            reasons.append("confidence_too_optimistic")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if float(tick.get("data_share_credit", 0.0)) > 0.0 and marginal_gain <= 0.0:
            reasons.append("credit_without_frontier_gain")
            status = ShadowRegalStatus.WARN if status != ShadowRegalStatus.FAIL else status
            score = min(score, 0.45)
        if not reasons:
            reasons.append("pricing_truth_ok")
        return ShadowRegalDecision(
            node_id=self.node_id,
            status=status,
            score=score,
            reasons=reasons,
            evidence_pointers=dict(context.evidence_pointers),
            recommended_action="discount_or_suppress_pricing" if status != ShadowRegalStatus.PASS else "publish_shadow_tick",
            details={
                "net_customer_rate": net,
                "task_hour_price_tick": gross,
                "constraint_adjustment": constraint_adjustment,
                "uncertainty_adjustment": uncertainty_adjustment,
                "confidence": confidence,
                "uncertainty": uncertainty,
            },
        )


@dataclass(frozen=True)
class DataValueRegal:
    """Verify datapack credit claims remain justified by available evidence."""

    node_id: str = "data_value_regal"

    def evaluate(self, context: ShadowRegalContext) -> ShadowRegalDecision:
        datapack_update = dict(context.datapack_credit_update or {})
        claimed_credit = float(datapack_update.get("data_share_credit", 0.0))
        marginal_gain = float(datapack_update.get("marginal_frontier_gain", _axis(context.econ_tensor, "marginal_frontier_gain")))
        quality_score = float(datapack_update.get("quality_score", context.episode_metrics.get("quality_score", 0.0)))
        confidence = 0.0
        if context.pricing_ticks:
            confidence = float(dict(context.pricing_ticks[0]).get("confidence", 0.0))
        justified_credit = max(0.0, marginal_gain) * max(0.1, min(1.0, quality_score)) * max(0.1, confidence)
        reasons: list[str] = []
        status = ShadowRegalStatus.PASS
        score = 1.0

        if claimed_credit > 0.0 and marginal_gain <= 0.0:
            reasons.append("credit_claim_without_gain")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if claimed_credit > justified_credit * 1.25 and justified_credit > 0.0:
            reasons.append("credit_claim_exceeds_evidence")
            status = ShadowRegalStatus.FAIL
            score = 0.0
        if claimed_credit > 0.0 and quality_score < 0.55:
            reasons.append("credit_claim_low_quality")
            status = ShadowRegalStatus.WARN if status != ShadowRegalStatus.FAIL else status
            score = min(score, 0.45)
        if not reasons:
            reasons.append("data_value_claim_justified")
        return ShadowRegalDecision(
            node_id=self.node_id,
            status=status,
            score=score,
            reasons=reasons,
            evidence_pointers=dict(context.evidence_pointers),
            recommended_action="review_datapack_credit" if status != ShadowRegalStatus.PASS else "keep_shadow_credit",
            details={
                "claimed_credit": claimed_credit,
                "justified_credit": justified_credit,
                "marginal_frontier_gain": marginal_gain,
                "quality_score": quality_score,
                "pricing_confidence": confidence,
            },
        )


def default_shadow_nodes() -> list[ShadowRegalNode]:
    return [
        ObjectiveIntegrityRegal(),
        PlausibilityRegal(),
        RewardSafetyRegal(),
        PricingTruthRegal(),
        DataValueRegal(),
    ]


def _metric(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return default


def _axis(tensor: Mapping[str, Any], axis: str, default: float = 0.0) -> float:
    axis_payload = tensor.get("axes")
    if isinstance(axis_payload, Mapping) and axis in axis_payload:
        return float(axis_payload[axis])
    values = tensor.get("values")
    if isinstance(axis_payload, Sequence) and isinstance(values, Sequence):
        for index, axis_name in enumerate(axis_payload):
            if str(axis_name) == axis:
                try:
                    return float(values[index])
                except Exception:
                    return default
    summary_axes = tensor.get("summary", {}).get("axes") if isinstance(tensor.get("summary"), Mapping) else None
    if isinstance(summary_axes, Mapping) and axis in summary_axes:
        return float(summary_axes[axis])
    try:
        return float(tensor.get(axis, default))
    except Exception:
        return default
