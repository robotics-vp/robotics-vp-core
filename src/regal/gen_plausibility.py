"""Geometry plausibility regal gate for generation promotion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.regal.base import RegalDecision, RegalNode, RegalReport


@dataclass
class PlausibilityThresholds:
    min_map_first_quality: float = 0.35
    max_semantic_disagreement: float = 0.6
    min_vla_coverage: float = 0.25
    min_plausibility_score: float = 0.45


class RegalGenPlausibilityNode(RegalNode):
    """Block physically-valid but geometrically implausible generation outputs."""

    node_id = "regal_gen_plausibility"

    def __init__(self, thresholds: PlausibilityThresholds | None = None) -> None:
        self.thresholds = thresholds or PlausibilityThresholds()

    def evaluate(self, context: Mapping[str, object]) -> RegalReport:
        map_first_quality = _as_float(context.get("map_first_quality_score"), 0.0)
        semantic_disagreement = _as_float(
            context.get("semantic_disagreement_vla_vs_map"), 1.0
        )
        vla_coverage = _as_float(context.get("vla_evidence_coverage"), 0.0)

        plausibility = (
            0.45 * map_first_quality
            + 0.35 * (1.0 - min(1.0, semantic_disagreement))
            + 0.20 * vla_coverage
        )

        reasons = []
        if map_first_quality < self.thresholds.min_map_first_quality:
            reasons.append("map_first_quality_low")
        if semantic_disagreement > self.thresholds.max_semantic_disagreement:
            reasons.append("semantic_disagreement_high")
        if vla_coverage < self.thresholds.min_vla_coverage:
            reasons.append("vla_coverage_low")
        if plausibility < self.thresholds.min_plausibility_score:
            reasons.append("plausibility_score_low")

        if reasons:
            return RegalReport(
                node_id=self.node_id,
                decision=RegalDecision.BLOCK,
                reason_codes=reasons,
                details={
                    "map_first_quality_score": map_first_quality,
                    "semantic_disagreement_vla_vs_map": semantic_disagreement,
                    "vla_evidence_coverage": vla_coverage,
                    "plausibility_score": plausibility,
                },
                recommended_action="tighten_constraints_and_regenerate",
                confidence=0.9,
            )

        return RegalReport(
            node_id=self.node_id,
            decision=RegalDecision.ALLOW,
            reason_codes=["plausibility_ok"],
            details={
                "map_first_quality_score": map_first_quality,
                "semantic_disagreement_vla_vs_map": semantic_disagreement,
                "vla_evidence_coverage": vla_coverage,
                "plausibility_score": plausibility,
            },
            confidence=0.85,
        )


def _as_float(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        return default
    except (TypeError, ValueError):
        return default
