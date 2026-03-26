"""Bounded learned-plus-heuristic routing for coverage-loop fill decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Sequence

from src.world_model.fill_path_policy import (
    FILL_METHODS,
    load_fill_path_helper_predictions,
)
from src.world_model.fill_path_runtime import resolve_fill_path_helper


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in scores.values())
    if total <= 1e-9:
        uniform = 1.0 / float(len(FILL_METHODS))
        return {method: uniform for method in FILL_METHODS}
    return {
        method: max(0.0, float(scores.get(method, 0.0))) / total
        for method in FILL_METHODS
    }


@dataclass(frozen=True)
class FillPathRoutingDecision:
    edge_key: str
    fill_method: str
    confidence: float
    rationale: str
    coverage_gap_score: float
    economic_priority: float
    trust_priority: float
    readiness: float
    routing_policy: str
    heuristic_fill_method: str | None = None
    learned_fill_method: str | None = None
    helper_status: Dict[str, Any] | None = None
    score_trace: Dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "edge_key": self.edge_key,
            "fill_method": self.fill_method,
            "confidence": float(self.confidence),
            "rationale": self.rationale,
            "coverage_gap_score": float(self.coverage_gap_score),
            "economic_priority": float(self.economic_priority),
            "trust_priority": float(self.trust_priority),
            "readiness": float(self.readiness),
            "routing_policy": self.routing_policy,
            "heuristic_fill_method": self.heuristic_fill_method,
            "learned_fill_method": self.learned_fill_method,
            "helper_status": dict(self.helper_status or {}),
            "score_trace": dict(self.score_trace or {}),
        }
        return payload


def _gap_from_ranked_item(item: Any) -> Any:
    return getattr(item, "gap", item)


def _ranking_trace(item: Any) -> dict[str, Any]:
    if hasattr(item, "ranking_score"):
        return {
            "agenda_ranking_score": _safe_float(getattr(item, "ranking_score", 0.0)),
            "agenda_heuristic_score": _safe_float(getattr(item, "heuristic_score", 0.0)),
            "agenda_heuristic_score_norm": _safe_float(
                getattr(item, "heuristic_score_norm", 0.0)
            ),
            "agenda_learned_score": _safe_float(getattr(item, "learned_score", 0.0)),
            "agenda_learned_score_norm": _safe_float(
                getattr(item, "learned_score_norm", 0.0)
            ),
            "agenda_ranking_policy": str(getattr(item, "ranking_policy", "heuristic_only")),
            "agenda_helper_status": dict(getattr(item, "helper_status", {}) or {}),
        }
    return {}


def _heuristic_decision(
    gap: Any,
    *,
    trust_threshold: float = 0.3,
    readiness_threshold: float = 0.5,
) -> tuple[str, float, str, dict[str, float]]:
    econ = _safe_float(getattr(gap, "economic_priority", 0.0))
    trust = _safe_float(getattr(gap, "trust_priority", 0.0))
    readiness = _safe_float(getattr(gap, "promotion_readiness", 0.0))
    metadata = dict(getattr(gap, "metadata", {}) or {})

    if bool(metadata.get("governance_blocked", False)):
        fill_method = "blocked"
        confidence = 0.95
        rationale = "Governance trace blocked this edge; keep as meta-node review target"
    elif readiness < readiness_threshold:
        fill_method = "blocked"
        confidence = 0.3
        rationale = f"Readiness {readiness:.2f} < {readiness_threshold}: prerequisites not met"
    elif trust < trust_threshold:
        fill_method = "real_sim"
        confidence = 0.7
        rationale = f"Trust {trust:.2f} < {trust_threshold}: real sim preferred for high-fidelity evidence"
    elif econ > 0.7:
        fill_method = "diffusion"
        confidence = 0.8
        rationale = f"Economic priority {econ:.2f} > 0.7: diffusion for fast gap filling"
    elif econ > 0.3:
        fill_method = "synthetic_branch"
        confidence = 0.6
        rationale = f"Moderate economic priority {econ:.2f}: synthetic branching"
    else:
        fill_method = "diffusion"
        confidence = 0.5
        rationale = f"Low economic priority {econ:.2f}: diffusion with low urgency"

    scores = {method: 0.05 for method in FILL_METHODS}
    scores[fill_method] = 1.0
    scores["real_sim"] += _clip01((trust_threshold - trust) / max(trust_threshold, 1e-6)) * 0.6
    scores["diffusion"] += _clip01(econ) * 0.45
    scores["synthetic_branch"] += _clip01(econ * max(readiness, 0.1)) * 0.35
    scores["blocked"] += _clip01((readiness_threshold - readiness) / max(readiness_threshold, 1e-6)) * 0.75
    return fill_method, confidence, rationale, _normalize_scores(scores)


def _helper_weight(helper_status: Mapping[str, Any], gap: Any) -> float:
    if str(helper_status.get("promotion_stage", "")) == "heuristic_fallback":
        return 0.0
    base = 0.7 if bool(helper_status.get("benchmark_gate_ready", False)) else 0.25
    readiness = _safe_float(getattr(gap, "promotion_readiness", 0.0))
    trust = _safe_float(getattr(gap, "trust_priority", 0.0))
    evidence = _safe_float(getattr(gap, "evidence_count", 0.0))
    metadata = dict(getattr(gap, "metadata", {}) or {})
    bonus = (0.15 * _clip01(readiness)) + (0.1 * _clip01(trust))
    if evidence <= 0.0:
        bonus += 0.05
    if bool(metadata.get("governance_blocked", False)):
        bonus = 0.0
    return _clip01(base + bonus)


def route_fill_paths(
    ranked_gaps: Sequence[Any],
    coverage_graph: Any,
    *,
    fill_path_policy: Any = None,
    fill_path_policy_mode: Literal["disabled", "auto", "required"] = "auto",
) -> tuple[list[FillPathRoutingDecision], Dict[str, Any]]:
    helper, helper_status = resolve_fill_path_helper(fill_path_policy, mode=fill_path_policy_mode)
    gaps = [_gap_from_ranked_item(item) for item in ranked_gaps]
    if helper is None:
        decisions = []
        for item, gap in zip(ranked_gaps, gaps):
            fill_method, confidence, rationale, heuristic_scores = _heuristic_decision(gap)
            edge_key = f"{gap.source_id} -> {gap.target_id}"
            decisions.append(
                FillPathRoutingDecision(
                    edge_key=edge_key,
                    fill_method=fill_method,
                    confidence=confidence,
                    rationale=rationale,
                    coverage_gap_score=_safe_float(
                        getattr(item, "ranking_score", None)
                        if hasattr(item, "ranking_score")
                        else gap.gap_score() if callable(getattr(gap, "gap_score", None)) else 0.0
                    ),
                    economic_priority=_safe_float(getattr(gap, "economic_priority", 0.0)),
                    trust_priority=_safe_float(getattr(gap, "trust_priority", 0.0)),
                    readiness=_safe_float(getattr(gap, "promotion_readiness", 0.0)),
                    routing_policy="heuristic_only",
                    heuristic_fill_method=fill_method,
                    learned_fill_method=None,
                    helper_status=dict(helper_status),
                    score_trace={
                        "heuristic_scores": heuristic_scores,
                        "learned_scores": {method: 0.0 for method in FILL_METHODS},
                        "blended_scores": heuristic_scores,
                        "helper_weight": 0.0,
                        **_ranking_trace(item),
                    },
                )
            )
        return decisions, dict(helper_status)

    try:
        learned_predictions = load_fill_path_helper_predictions(helper, gaps, coverage_graph)
    except Exception as exc:
        fallback_status = {
            **dict(helper_status),
            "status": "inference_failed",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
            "error": str(exc),
        }
        decisions = []
        for item, gap in zip(ranked_gaps, gaps):
            fill_method, confidence, rationale, heuristic_scores = _heuristic_decision(gap)
            edge_key = f"{gap.source_id} -> {gap.target_id}"
            decisions.append(
                FillPathRoutingDecision(
                    edge_key=edge_key,
                    fill_method=fill_method,
                    confidence=confidence,
                    rationale=rationale,
                    coverage_gap_score=_safe_float(
                        getattr(item, "ranking_score", None)
                        if hasattr(item, "ranking_score")
                        else gap.gap_score() if callable(getattr(gap, "gap_score", None)) else 0.0
                    ),
                    economic_priority=_safe_float(getattr(gap, "economic_priority", 0.0)),
                    trust_priority=_safe_float(getattr(gap, "trust_priority", 0.0)),
                    readiness=_safe_float(getattr(gap, "promotion_readiness", 0.0)),
                    routing_policy="heuristic_only",
                    heuristic_fill_method=fill_method,
                    learned_fill_method=None,
                    helper_status=dict(fallback_status),
                    score_trace={
                        "heuristic_scores": heuristic_scores,
                        "learned_scores": {method: 0.0 for method in FILL_METHODS},
                        "blended_scores": heuristic_scores,
                        "helper_weight": 0.0,
                        **_ranking_trace(item),
                    },
                )
            )
        return decisions, fallback_status
    decisions: list[FillPathRoutingDecision] = []
    effective_helper_status = dict(helper_status)
    for item, gap, learned in zip(ranked_gaps, gaps, learned_predictions):
        heuristic_fill_method, heuristic_confidence, heuristic_rationale, heuristic_scores = _heuristic_decision(gap)
        learned_scores = _normalize_scores(dict(learned.get("method_probabilities", {}) or {}))
        helper_weight = _helper_weight(helper_status, gap)
        blended_scores = {
            method: ((1.0 - helper_weight) * heuristic_scores[method]) + (helper_weight * learned_scores[method])
            for method in FILL_METHODS
        }
        metadata = dict(getattr(gap, "metadata", {}) or {})
        if bool(metadata.get("governance_blocked", False)) or _safe_float(getattr(gap, "promotion_readiness", 0.0)) < 0.5:
            fill_method = heuristic_fill_method
            confidence = heuristic_confidence
            rationale = heuristic_rationale
            blended_scores = dict(heuristic_scores)
            helper_weight = 0.0
            routing_policy = "heuristic_hard_gate"
        else:
            fill_method = max(blended_scores, key=blended_scores.get)
            confidence = float(blended_scores[fill_method])
            routing_policy = "heuristic_plus_learned_fill_path_policy"
            rationale = (
                f"Blended fill routing picked {fill_method}: heuristic={heuristic_fill_method}, "
                f"learned={str(learned.get('fill_method', ''))}, helper_weight={helper_weight:.2f}"
            )
        edge_key = f"{gap.source_id} -> {gap.target_id}"
        decisions.append(
            FillPathRoutingDecision(
                edge_key=edge_key,
                fill_method=fill_method,
                confidence=confidence,
                rationale=rationale,
                coverage_gap_score=_safe_float(
                    getattr(item, "ranking_score", None)
                    if hasattr(item, "ranking_score")
                    else gap.gap_score() if callable(getattr(gap, "gap_score", None)) else 0.0
                ),
                economic_priority=_safe_float(getattr(gap, "economic_priority", 0.0)),
                trust_priority=_safe_float(getattr(gap, "trust_priority", 0.0)),
                readiness=_safe_float(getattr(gap, "promotion_readiness", 0.0)),
                routing_policy=routing_policy,
                heuristic_fill_method=heuristic_fill_method,
                learned_fill_method=str(learned.get("fill_method", "")),
                helper_status={**dict(helper_status), "helper_weight": helper_weight},
                score_trace={
                    "heuristic_scores": heuristic_scores,
                    "learned_scores": learned_scores,
                    "blended_scores": blended_scores,
                    "heuristic_confidence": heuristic_confidence,
                    "learned_confidence": _safe_float(learned.get("confidence", 0.0)),
                    "helper_weight": helper_weight,
                    **_ranking_trace(item),
                },
            )
        )
    effective_helper_status["used"] = True
    return decisions, effective_helper_status


__all__ = ["FillPathRoutingDecision", "route_fill_paths"]
