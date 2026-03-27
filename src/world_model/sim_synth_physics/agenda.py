"""Canonical simulation-agenda contracts for the sim/synth/physics WM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .common import clip01, mapping, strings


@dataclass(frozen=True)
class SimulationJobSpec:
    """One typed simulation or branch-generation job inside the WM agenda."""

    job_id: str
    rank: int
    task_family: str
    env_backend: str
    skill_edge: str
    risk_family: str
    object_family: str
    objective_preset: str
    data_collection_intent: str
    coverage_gap_score: float
    economic_priority: float
    trust_priority: float
    readiness: float
    ranking_policy: str
    rationale: str
    coverage_targets: Dict[str, Any] = field(default_factory=dict)
    expected_receipts: list[str] = field(default_factory=list)
    inferential_learnability_contract: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "sim_synth_physics_job_spec_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "rank": int(self.rank),
            "task_family": self.task_family,
            "env_backend": self.env_backend,
            "skill_edge": self.skill_edge,
            "risk_family": self.risk_family,
            "object_family": self.object_family,
            "objective_preset": self.objective_preset,
            "data_collection_intent": self.data_collection_intent,
            "coverage_gap_score": float(self.coverage_gap_score),
            "economic_priority": clip01(self.economic_priority),
            "trust_priority": clip01(self.trust_priority),
            "readiness": clip01(self.readiness),
            "ranking_policy": self.ranking_policy,
            "rationale": self.rationale,
            "coverage_targets": mapping(self.coverage_targets),
            "expected_receipts": strings(self.expected_receipts),
            "inferential_learnability_contract": mapping(
                self.inferential_learnability_contract
            ),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SimulationAgenda:
    """Ranked simulation-agenda owned by the sim/synth/physics WM."""

    agenda_id: str
    coverage_window_ref: str
    jobs: list[SimulationJobSpec] = field(default_factory=list)
    ranking_policy: str = "heuristic_only"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "simulation_agenda_v2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agenda_id": self.agenda_id,
            "coverage_window_ref": self.coverage_window_ref,
            "jobs": [job.to_dict() for job in self.jobs],
            "ranking_policy": self.ranking_policy,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }

    def to_legacy_items(self) -> list[Dict[str, Any]]:
        return [job.to_dict() for job in self.jobs]
