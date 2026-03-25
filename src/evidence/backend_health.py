"""Backend health metadata and degradation predicates.

Addresses the SceneTracks stub-dominance fragility by providing explicit
per-episode metadata about which perception/teacher backends were real vs.
stub, plus a degradation-aware ``PreconditionCheck``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from src.evidence.preconditions import PreconditionCheck


@dataclass
class BackendHealthReport:
    """Per-episode summary of perception/teacher backend status."""

    episode_id: str = ""
    scene_tracks_mode: str = "stub"  # stub | passthrough | real
    vla_mode: str = "stub"
    teacher_mode: str = "stub"
    map_first_mode: str = "stub"
    degradation_flags: List[str] = field(default_factory=list)
    evidence_density_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    _MODE_WEIGHTS = {"real": 1.0, "passthrough": 0.3, "stub": 0.0}

    def __post_init__(self) -> None:
        # Auto-compute degradation flags if not already set
        if not self.degradation_flags:
            flags: List[str] = []
            if self.scene_tracks_mode == "stub":
                flags.append("scene_tracks_stub")
            if self.vla_mode == "stub":
                flags.append("vla_stub")
            if self.teacher_mode == "stub":
                flags.append("teacher_stub")
            if self.map_first_mode == "stub":
                flags.append("map_first_stub")
            self.degradation_flags = flags

        # Auto-compute evidence density if not already set
        if self.evidence_density_score == 0.0:
            weights = self._MODE_WEIGHTS
            self.evidence_density_score = (
                weights.get(self.scene_tracks_mode, 0.0)
                + weights.get(self.vla_mode, 0.0)
                + weights.get(self.teacher_mode, 0.0)
                + weights.get(self.map_first_mode, 0.0)
            ) / 4.0

    @property
    def is_fully_real(self) -> bool:
        return not self.degradation_flags

    @property
    def is_fully_stub(self) -> bool:
        return self.evidence_density_score == 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scene_tracks_mode": self.scene_tracks_mode,
            "vla_mode": self.vla_mode,
            "teacher_mode": self.teacher_mode,
            "map_first_mode": self.map_first_mode,
            "degradation_flags": list(self.degradation_flags),
            "evidence_density_score": self.evidence_density_score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackendHealthReport":
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            scene_tracks_mode=str(payload.get("scene_tracks_mode", "stub")),
            vla_mode=str(payload.get("vla_mode", "stub")),
            teacher_mode=str(payload.get("teacher_mode", "stub")),
            map_first_mode=str(payload.get("map_first_mode", "stub")),
            degradation_flags=list(payload.get("degradation_flags", [])),
            evidence_density_score=float(payload.get("evidence_density_score", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


def check_backend_health(
    report: BackendHealthReport,
    *,
    min_density: float = 0.25,
    max_stub_count: int = 3,
) -> PreconditionCheck:
    """Return a ``PreconditionCheck`` based on backend health.

    Parameters
    ----------
    report : BackendHealthReport
    min_density : float
        Minimum evidence-density score to pass.
    max_stub_count : int
        Maximum number of degradation (stub) flags allowed.
    """
    stub_count = len(report.degradation_flags)
    density_ok = report.evidence_density_score >= min_density
    stub_ok = stub_count <= max_stub_count
    passed = density_ok and stub_ok

    return PreconditionCheck(
        precondition_id="backend_health",
        satisfied=passed,
        detail="Backend perception/teacher modes meet evidence-density thresholds",
        observed_value=report.evidence_density_score,
        metadata={
            "evidence_density_score": report.evidence_density_score,
            "min_density": min_density,
            "degradation_flags": list(report.degradation_flags),
            "stub_count": stub_count,
            "max_stub_count": max_stub_count,
            "scene_tracks_mode": report.scene_tracks_mode,
            "vla_mode": report.vla_mode,
            "teacher_mode": report.teacher_mode,
            "map_first_mode": report.map_first_mode,
        },
    )


__all__ = [
    "BackendHealthReport",
    "check_backend_health",
]
