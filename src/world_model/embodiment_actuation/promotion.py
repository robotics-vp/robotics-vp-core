"""Promotion posture for bounded Embodiment / Actuation learned seams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .common import clip01, mapping, strings

VALID_POSTURES = {"disabled", "auto", "required"}


@dataclass(frozen=True)
class EmbodimentSeamResolution:
    seam_id: str
    posture: str = "disabled"
    promotion_stage: str = "heuristic_fallback"
    can_execute: bool = False
    should_emit_receipt: bool = True
    blocked_reasons: list[str] = field(default_factory=list)
    benchmark_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_seam_resolution_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seam_id": self.seam_id,
            "posture": self.posture,
            "promotion_stage": self.promotion_stage,
            "can_execute": bool(self.can_execute),
            "should_emit_receipt": bool(self.should_emit_receipt),
            "blocked_reasons": strings(self.blocked_reasons),
            "benchmark_score": clip01(self.benchmark_score),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


def resolve_embodiment_seam(
    seam_id: str,
    *,
    posture: str = "disabled",
    benchmark_signals: Mapping[str, Any] | None = None,
    provider_available: bool = False,
    required_score: float = 0.75,
) -> EmbodimentSeamResolution:
    selected_posture = posture if posture in VALID_POSTURES else "disabled"
    signals = mapping(benchmark_signals)
    score = clip01(signals.get("score", signals.get("benchmark_score", 0.0)))
    has_benchmark = bool(signals.get("benchmark_ready", False))
    blocked: list[str] = []

    if selected_posture == "disabled":
        blocked.append("posture_disabled")
    if not provider_available:
        blocked.append("provider_unavailable")
    if not has_benchmark:
        blocked.append("benchmark_not_ready")
    if score < required_score:
        blocked.append("benchmark_score_below_threshold")

    can_execute = selected_posture in {"auto", "required"} and not blocked
    stage = "promoted" if can_execute else "heuristic_fallback"
    if selected_posture == "required" and not can_execute:
        stage = "required_blocked"

    return EmbodimentSeamResolution(
        seam_id=str(seam_id),
        posture=selected_posture,
        promotion_stage=stage,
        can_execute=can_execute,
        blocked_reasons=blocked,
        benchmark_score=score,
        metadata={"required_score": float(required_score)},
    )
