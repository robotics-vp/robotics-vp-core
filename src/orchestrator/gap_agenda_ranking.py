"""Bounded learned-plus-heuristic ranking for simulation and diffusion agendas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence

from src.world_model.gap_ranker_runtime import resolve_gap_ranker_helper


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalized(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) <= 1e-9:
        return [0.5 for _ in values]
    return [_clip01((value - lo) / float(hi - lo)) for value in values]


@dataclass(frozen=True)
class RankedAgendaGap:
    gap: Any
    ranking_score: float
    heuristic_score: float
    heuristic_score_norm: float
    learned_score: float
    learned_score_norm: float
    ranking_policy: str
    helper_status: Dict[str, Any]


def rank_gaps_for_agenda(
    coverage_graph: Any,
    *,
    economic_weight: float = 1.0,
    trust_weight: float = 1.0,
    readiness_weight: float = 1.0,
    limit: int = 10,
    gap_ranker: Any = None,
    gap_ranker_mode: Literal["disabled", "auto", "required"] = "auto",
) -> list[RankedAgendaGap]:
    gaps = [edge for edge in list(getattr(coverage_graph, "edges", []) or []) if getattr(edge, "is_missing", False)]
    if not gaps:
        return []

    heuristic_scores = [
        float(
            edge.gap_score(
                economic_weight=economic_weight,
                trust_weight=trust_weight,
                readiness_weight=readiness_weight,
            )
        )
        for edge in gaps
    ]
    heuristic_norm = _normalized(heuristic_scores)

    helper, helper_status = resolve_gap_ranker_helper(gap_ranker, mode=gap_ranker_mode)
    if helper is None:
        ranked = [
            RankedAgendaGap(
                gap=edge,
                ranking_score=heuristic_score,
                heuristic_score=heuristic_score,
                heuristic_score_norm=heuristic_score_norm,
                learned_score=0.0,
                learned_score_norm=0.0,
                ranking_policy="heuristic_only",
                helper_status=dict(helper_status),
            )
            for edge, heuristic_score, heuristic_score_norm in zip(gaps, heuristic_scores, heuristic_norm)
        ]
        ranked.sort(key=lambda item: item.ranking_score, reverse=True)
        return ranked[:limit]

    try:
        learned_pairs = helper.rank_edges(gaps, coverage_graph)
        learned_map = {id(edge): float(score) for edge, score in learned_pairs}
        learned_scores = [float(learned_map.get(id(edge), 0.0)) for edge in gaps]
    except Exception as exc:
        fallback_status = {
            **dict(helper_status),
            "status": "inference_failed",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
            "error": str(exc),
        }
        ranked = [
            RankedAgendaGap(
                gap=edge,
                ranking_score=heuristic_score,
                heuristic_score=heuristic_score,
                heuristic_score_norm=heuristic_score_norm,
                learned_score=0.0,
                learned_score_norm=0.0,
                ranking_policy="heuristic_only",
                helper_status=fallback_status,
            )
            for edge, heuristic_score, heuristic_score_norm in zip(gaps, heuristic_scores, heuristic_norm)
        ]
        ranked.sort(key=lambda item: item.ranking_score, reverse=True)
        return ranked[:limit]

    learned_norm = _normalized(learned_scores)
    benchmark_gate_ready = bool(helper_status.get("benchmark_gate_ready", False))
    helper_weight = 0.7 if benchmark_gate_ready else 0.25
    ranked = [
        RankedAgendaGap(
            gap=edge,
            ranking_score=((1.0 - helper_weight) * heuristic_score_norm) + (helper_weight * learned_score_norm),
            heuristic_score=heuristic_score,
            heuristic_score_norm=heuristic_score_norm,
            learned_score=learned_score,
            learned_score_norm=learned_score_norm,
            ranking_policy="heuristic_plus_learned_gap_ranker",
            helper_status={**dict(helper_status), "helper_weight": helper_weight},
        )
        for edge, heuristic_score, heuristic_score_norm, learned_score, learned_score_norm in zip(
            gaps,
            heuristic_scores,
            heuristic_norm,
            learned_scores,
            learned_norm,
        )
    ]
    ranked.sort(key=lambda item: item.ranking_score, reverse=True)
    return ranked[:limit]


__all__ = ["RankedAgendaGap", "rank_gaps_for_agenda"]
