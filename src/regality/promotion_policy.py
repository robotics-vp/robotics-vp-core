"""Config-driven maturity registry for staged regal authority promotion."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from src.learning.calibration import CalibrationSummary
from src.utils.config_digest import sha256_json


class RegalMaturityStage(str, Enum):
    """Authority stages for regal nodes."""

    COMPARE_ONLY = "compare_only"
    ADVISORY = "advisory"
    BUDGET_GATE = "budget_gate"
    NARROW_HARD_GATE = "narrow_hard_gate"


_STAGE_ORDER = {
    RegalMaturityStage.COMPARE_ONLY: 0,
    RegalMaturityStage.ADVISORY: 1,
    RegalMaturityStage.BUDGET_GATE: 2,
    RegalMaturityStage.NARROW_HARD_GATE: 3,
}


@dataclass(frozen=True)
class PromotionCriteria:
    """Quantitative evidence required for promotion or demotion decisions."""

    min_replay_coverage: float = 0.0
    min_downstream_label_count: int = 0
    min_deployment_receipt_count: int = 0
    max_calibration_error: float = 1.0
    min_baseline_agreement: float = 0.0
    min_monotonicity: float = 0.0
    min_sign_consistency: float = 0.0
    max_false_positive_rate: float = 1.0
    max_false_negative_rate: float = 1.0
    max_drift_score: float = 1.0
    min_residual_gain: float = -1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PromotionCriteria":
        source = dict(payload or {})
        return cls(
            min_replay_coverage=float(source.get("min_replay_coverage", 0.0)),
            min_downstream_label_count=int(source.get("min_downstream_label_count", 0)),
            min_deployment_receipt_count=int(source.get("min_deployment_receipt_count", 0)),
            max_calibration_error=float(source.get("max_calibration_error", 1.0)),
            min_baseline_agreement=float(source.get("min_baseline_agreement", 0.0)),
            min_monotonicity=float(source.get("min_monotonicity", 0.0)),
            min_sign_consistency=float(source.get("min_sign_consistency", 0.0)),
            max_false_positive_rate=float(source.get("max_false_positive_rate", 1.0)),
            max_false_negative_rate=float(source.get("max_false_negative_rate", 1.0)),
            max_drift_score=float(source.get("max_drift_score", 1.0)),
            min_residual_gain=float(source.get("min_residual_gain", -1.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_replay_coverage": float(self.min_replay_coverage),
            "min_downstream_label_count": int(self.min_downstream_label_count),
            "min_deployment_receipt_count": int(self.min_deployment_receipt_count),
            "max_calibration_error": float(self.max_calibration_error),
            "min_baseline_agreement": float(self.min_baseline_agreement),
            "min_monotonicity": float(self.min_monotonicity),
            "min_sign_consistency": float(self.min_sign_consistency),
            "max_false_positive_rate": float(self.max_false_positive_rate),
            "max_false_negative_rate": float(self.max_false_negative_rate),
            "max_drift_score": float(self.max_drift_score),
            "min_residual_gain": float(self.min_residual_gain),
        }


@dataclass(frozen=True)
class PromotionMetrics:
    """Observed quality metrics for one regal node or learned advisor."""

    replay_coverage: float
    downstream_label_count: int
    deployment_receipt_count: int
    calibration_error: float
    baseline_agreement: float
    monotonicity: float
    sign_consistency: float
    false_positive_rate: float
    false_negative_rate: float
    drift_score: float
    residual_gain: float = 0.0
    calibration_summary: Optional[CalibrationSummary] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "replay_coverage": float(self.replay_coverage),
            "downstream_label_count": int(self.downstream_label_count),
            "deployment_receipt_count": int(self.deployment_receipt_count),
            "calibration_error": float(self.calibration_error),
            "baseline_agreement": float(self.baseline_agreement),
            "monotonicity": float(self.monotonicity),
            "sign_consistency": float(self.sign_consistency),
            "false_positive_rate": float(self.false_positive_rate),
            "false_negative_rate": float(self.false_negative_rate),
            "drift_score": float(self.drift_score),
            "residual_gain": float(self.residual_gain),
            "metadata": dict(self.metadata),
        }
        if self.calibration_summary is not None:
            payload["calibration_summary"] = self.calibration_summary.to_dict()
        return payload


@dataclass(frozen=True)
class PromotionDecision:
    """Recommendation for one node's maturity stage."""

    node_id: str
    current_stage: RegalMaturityStage
    recommended_stage: RegalMaturityStage
    outcome: str
    reasons: list[str]
    evidence_pointers: Dict[str, str]
    metrics_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "current_stage": self.current_stage.value,
            "recommended_stage": self.recommended_stage.value,
            "outcome": self.outcome,
            "reasons": list(self.reasons),
            "evidence_pointers": dict(self.evidence_pointers),
            "metrics_summary": dict(self.metrics_summary),
        }


@dataclass(frozen=True)
class RegalPromotionPolicyEntry:
    """Per-node maturity configuration."""

    node_id: str
    current_stage: RegalMaturityStage
    allowed_actions: Dict[str, list[str]]
    promotion_criteria: PromotionCriteria
    demotion_criteria: PromotionCriteria
    calibration_requirements: Dict[str, Any] = field(default_factory=dict)
    minimum_data_requirements: Dict[str, Any] = field(default_factory=dict)
    last_evaluation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "current_stage": self.current_stage.value,
            "allowed_actions": {stage: list(actions) for stage, actions in self.allowed_actions.items()},
            "promotion_criteria": self.promotion_criteria.to_dict(),
            "demotion_criteria": self.demotion_criteria.to_dict(),
            "calibration_requirements": dict(self.calibration_requirements),
            "minimum_data_requirements": dict(self.minimum_data_requirements),
            "last_evaluation_metadata": dict(self.last_evaluation_metadata),
        }


def _stage_from_value(value: Any) -> RegalMaturityStage:
    return value if isinstance(value, RegalMaturityStage) else RegalMaturityStage(str(value))


def _sorted_actions(actions: Iterable[str]) -> list[str]:
    return sorted({str(action) for action in actions})


def _evaluate_promotion(criteria: PromotionCriteria, metrics: PromotionMetrics) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if metrics.replay_coverage < criteria.min_replay_coverage:
        failures.append("replay_coverage_below_threshold")
    if metrics.downstream_label_count < criteria.min_downstream_label_count:
        failures.append("insufficient_downstream_labels")
    if metrics.deployment_receipt_count < criteria.min_deployment_receipt_count:
        failures.append("insufficient_deployment_receipts")
    if metrics.calibration_error > criteria.max_calibration_error:
        failures.append("calibration_error_too_high")
    if metrics.baseline_agreement < criteria.min_baseline_agreement:
        failures.append("baseline_agreement_too_low")
    if metrics.monotonicity < criteria.min_monotonicity:
        failures.append("monotonicity_too_low")
    if metrics.sign_consistency < criteria.min_sign_consistency:
        failures.append("sign_consistency_too_low")
    if metrics.false_positive_rate > criteria.max_false_positive_rate:
        failures.append("false_positive_rate_too_high")
    if metrics.false_negative_rate > criteria.max_false_negative_rate:
        failures.append("false_negative_rate_too_high")
    if metrics.drift_score > criteria.max_drift_score:
        failures.append("drift_score_too_high")
    if metrics.residual_gain < criteria.min_residual_gain:
        failures.append("residual_gain_too_low")
    return not failures, failures


def _next_stage(stage: RegalMaturityStage) -> RegalMaturityStage:
    if stage == RegalMaturityStage.COMPARE_ONLY:
        return RegalMaturityStage.ADVISORY
    if stage == RegalMaturityStage.ADVISORY:
        return RegalMaturityStage.BUDGET_GATE
    return RegalMaturityStage.NARROW_HARD_GATE


def _previous_stage(stage: RegalMaturityStage) -> RegalMaturityStage:
    if stage == RegalMaturityStage.NARROW_HARD_GATE:
        return RegalMaturityStage.BUDGET_GATE
    if stage == RegalMaturityStage.BUDGET_GATE:
        return RegalMaturityStage.ADVISORY
    return RegalMaturityStage.COMPARE_ONLY


@dataclass(frozen=True)
class RegalPromotionPolicy:
    """Registry of maturity state and evidence thresholds for regal authority."""

    schema_version: str
    policy_name: str
    nodes: Dict[str, RegalPromotionPolicyEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def config_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_name": self.policy_name,
            "nodes": {node_id: entry.to_dict() for node_id, entry in sorted(self.nodes.items())},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RegalPromotionPolicy":
        nodes: Dict[str, RegalPromotionPolicyEntry] = {}
        for node_id, raw_entry in dict(payload.get("nodes", {}) or {}).items():
            entry = dict(raw_entry or {})
            allowed_actions = {
                str(stage): _sorted_actions(actions)
                for stage, actions in dict(entry.get("allowed_actions", {}) or {}).items()
            }
            nodes[str(node_id)] = RegalPromotionPolicyEntry(
                node_id=str(node_id),
                current_stage=_stage_from_value(entry.get("current_stage", RegalMaturityStage.COMPARE_ONLY.value)),
                allowed_actions=allowed_actions,
                promotion_criteria=PromotionCriteria.from_mapping(entry.get("promotion_criteria")),
                demotion_criteria=PromotionCriteria.from_mapping(entry.get("demotion_criteria")),
                calibration_requirements=dict(entry.get("calibration_requirements", {}) or {}),
                minimum_data_requirements=dict(entry.get("minimum_data_requirements", {}) or {}),
                last_evaluation_metadata=dict(entry.get("last_evaluation_metadata", {}) or {}),
            )
        return cls(
            schema_version=str(payload.get("schema_version", "regal_promotion_policy_v1")),
            policy_name=str(payload.get("policy_name", "promotion_default")),
            nodes=nodes,
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "RegalPromotionPolicy":
        raw = Path(path).read_text(encoding="utf-8")
        payload = json.loads(raw) if Path(path).suffix.lower() == ".json" else yaml.safe_load(raw)
        return cls.from_mapping(payload or {})

    def node_stage(self, node_id: str) -> RegalMaturityStage:
        return self.nodes[node_id].current_stage

    def stage_allows(self, node_id: str, action: str) -> bool:
        entry = self.nodes[node_id]
        allowed = entry.allowed_actions.get(entry.current_stage.value, [])
        return str(action) in allowed

    def gate_eligible(self, node_id: str) -> bool:
        return _STAGE_ORDER[self.node_stage(node_id)] >= _STAGE_ORDER[RegalMaturityStage.BUDGET_GATE]

    def evaluate_node(
        self,
        node_id: str,
        metrics: PromotionMetrics,
        *,
        evidence_pointers: Mapping[str, str] | None = None,
    ) -> PromotionDecision:
        entry = self.nodes[node_id]
        current_stage = entry.current_stage
        promote_ok, promotion_failures = _evaluate_promotion(entry.promotion_criteria, metrics)
        demote_ok, demotion_failures = _evaluate_promotion(entry.demotion_criteria, metrics)

        if not demote_ok and current_stage != RegalMaturityStage.COMPARE_ONLY:
            reasons = ["demotion_criteria_breached", *sorted(set(demotion_failures))]
            return PromotionDecision(
                node_id=node_id,
                current_stage=current_stage,
                recommended_stage=_previous_stage(current_stage),
                outcome="recommend_demote",
                reasons=reasons,
                evidence_pointers=dict(evidence_pointers or {}),
                metrics_summary=metrics.to_dict(),
            )

        if promote_ok and current_stage != RegalMaturityStage.NARROW_HARD_GATE:
            return PromotionDecision(
                node_id=node_id,
                current_stage=current_stage,
                recommended_stage=_next_stage(current_stage),
                outcome="recommend_promote",
                reasons=["promotion_criteria_satisfied"],
                evidence_pointers=dict(evidence_pointers or {}),
                metrics_summary=metrics.to_dict(),
            )

        hold_reasons = ["promotion_hold"]
        hold_reasons.extend(sorted(set(promotion_failures)))
        return PromotionDecision(
            node_id=node_id,
            current_stage=current_stage,
            recommended_stage=current_stage,
            outcome="recommend_hold",
            reasons=hold_reasons,
            evidence_pointers=dict(evidence_pointers or {}),
            metrics_summary=metrics.to_dict(),
        )


def load_regal_promotion_policy(path: str | Path) -> RegalPromotionPolicy:
    """Load a promotion policy from YAML or JSON."""

    return RegalPromotionPolicy.from_path(path)
