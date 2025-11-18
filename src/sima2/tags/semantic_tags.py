"""
Dataclasses for Stage 2.4 semantic enrichment tags.

All tag types are JSON-safe, deterministic, and advisory-only. They are used by
the SemanticTagPropagator to generate enrichment proposals without mutating
economics, rewards, or task graph state.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Tuple

_FRAGILITY_LEVELS = {"low", "medium", "high", "critical"}
_RISK_SEVERITY = {"low", "medium", "high", "critical"}
_NOVELTY_TYPES = {
    "state_coverage",
    "action_diversity",
    "failure_mode",
    "edge_case",
}
_REPLAY_FREQUENCY = {"standard", "frequent", "rare"}
_CURRICULUM_STAGES = {"early", "mid", "late", "advanced"}
_PRIORITY_LEVELS = {"low", "medium", "high", "critical"}


def _as_json_dict(data):
    """Helper to convert dataclass to a plain dict."""
    return asdict(data)


@dataclass
class FragilityTag:
    """Tags fragile objects in an episode."""

    object_name: str
    fragility_level: str  # "low" | "medium" | "high" | "critical"
    damage_cost_usd: float
    contact_frames: List[int] = field(default_factory=list)
    justification: str = ""

    def __post_init__(self) -> None:
        if self.fragility_level not in _FRAGILITY_LEVELS:
            raise ValueError(f"Invalid fragility_level: {self.fragility_level}")
        if self.damage_cost_usd < 0.0:
            raise ValueError("damage_cost_usd must be non-negative")
        if not self.contact_frames:
            self.contact_frames = [0]

    def to_dict(self) -> dict:
        return _as_json_dict(self)


@dataclass
class RiskTag:
    """Tags safety risks in an episode."""

    risk_type: str  # "collision" | "tip_over" | "entanglement" | "human_proximity"
    severity: str  # "low" | "medium" | "high" | "critical"
    affected_frames: List[int]
    mitigation_hints: List[str] = field(default_factory=list)
    justification: str = ""

    def __post_init__(self) -> None:
        if self.severity not in _RISK_SEVERITY:
            raise ValueError(f"Invalid severity: {self.severity}")
        if not self.affected_frames:
            self.affected_frames = [0]

    def to_dict(self) -> dict:
        return _as_json_dict(self)


@dataclass
class AffordanceTag:
    """Tags demonstrated affordances."""

    affordance_name: str
    object_name: str
    demonstrated: bool
    alternative_affordances: List[str] = field(default_factory=list)
    justification: str = ""

    def to_dict(self) -> dict:
        return _as_json_dict(self)


@dataclass
class EfficiencyTag:
    """Tags execution efficiency metrics."""

    metric: str  # "time" | "energy" | "precision" | "success_rate"
    score: float  # 0.0 = worst, 1.0 = optimal
    benchmark: str
    improvement_hints: List[str] = field(default_factory=list)
    justification: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("EfficiencyTag.score must be in [0.0, 1.0]")

    def to_dict(self) -> dict:
        return _as_json_dict(self)


@dataclass
class NoveltyTag:
    """Tags novelty and expected marginal productivity gains."""

    novelty_type: str  # "state_coverage" | "action_diversity" | "failure_mode" | "edge_case"
    novelty_score: float  # 0.0 = redundant, 1.0 = maximally novel
    comparison_basis: str
    expected_mpl_gain: float
    justification: str = ""

    def __post_init__(self) -> None:
        if self.novelty_type not in _NOVELTY_TYPES:
            raise ValueError(f"Invalid novelty_type: {self.novelty_type}")
        if not 0.0 <= self.novelty_score <= 1.0:
            raise ValueError("NoveltyTag.novelty_score must be in [0.0, 1.0]")
        if self.expected_mpl_gain < 0.0:
            raise ValueError("NoveltyTag.expected_mpl_gain must be non-negative")

    def to_dict(self) -> dict:
        return _as_json_dict(self)


@dataclass
class InterventionTag:
    """Tags human interventions or corrections."""

    intervention_type: str  # "human_correction" | "failure_recovery" | "safety_override"
    frame_range: Tuple[int, int]
    trigger: str
    learning_opportunity: str
    justification: str = ""

    def __post_init__(self) -> None:
        if len(self.frame_range) != 2:
            raise ValueError("InterventionTag.frame_range must have two elements")
        start, end = self.frame_range
        if end < start:
            self.frame_range = (start, start)

    def to_dict(self) -> dict:
        data = _as_json_dict(self)
        data["frame_range"] = list(self.frame_range)
        return data


@dataclass
class SemanticConflict:
    """Represents conflicting tags within an enrichment."""

    conflict_type: str  # "affordance_mismatch" | "task_order_violation" | ...
    conflicting_tags: List[str]
    severity: str  # "low" | "medium" | "high"
    resolution_hint: str
    justification: str = ""

    def __post_init__(self) -> None:
        if self.severity not in {"low", "medium", "high"}:
            raise ValueError(f"Invalid conflict severity: {self.severity}")

    def to_dict(self) -> dict:
        return _as_json_dict(self)


@dataclass
class SupervisionHints:
    """Advisory supervision hints for the orchestrator."""

    prioritize_for_training: bool
    priority_level: str  # "low" | "medium" | "high" | "critical"

    suggested_weight_multiplier: float
    suggested_replay_frequency: str  # "standard" | "frequent" | "rare"

    requires_human_review: bool
    safety_critical: bool

    curriculum_stage: str  # "early" | "mid" | "late" | "advanced"
    prerequisite_tags: List[str] = field(default_factory=list)

    justification: str = ""

    def __post_init__(self) -> None:
        if self.priority_level not in _PRIORITY_LEVELS:
            raise ValueError(f"Invalid priority_level: {self.priority_level}")
        if self.suggested_replay_frequency not in _REPLAY_FREQUENCY:
            raise ValueError(
                f"Invalid suggested_replay_frequency: {self.suggested_replay_frequency}"
            )
        if self.curriculum_stage not in _CURRICULUM_STAGES:
            raise ValueError(f"Invalid curriculum_stage: {self.curriculum_stage}")
        if self.suggested_weight_multiplier < 0.0:
            raise ValueError("suggested_weight_multiplier must be non-negative")

    def to_dict(self) -> dict:
        return _as_json_dict(self)


@dataclass
class SemanticEnrichmentProposal:
    """Complete semantic enrichment (advisory-only) for a datapack episode."""

    proposal_id: str
    timestamp: float
    video_id: str
    episode_id: str
    task: str

    fragility_tags: List[FragilityTag] = field(default_factory=list)
    risk_tags: List[RiskTag] = field(default_factory=list)
    affordance_tags: List[AffordanceTag] = field(default_factory=list)
    efficiency_tags: List[EfficiencyTag] = field(default_factory=list)
    novelty_tags: List[NoveltyTag] = field(default_factory=list)
    intervention_tags: List[InterventionTag] = field(default_factory=list)

    semantic_conflicts: List[SemanticConflict] = field(default_factory=list)
    coherence_score: float = 0.0

    supervision_hints: SupervisionHints = None  # type: ignore

    confidence: float = 0.0
    source_proposals: List[str] = field(default_factory=list)
    justification: str = ""

    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.coherence_score <= 1.0:
            raise ValueError("coherence_score must be in [0.0, 1.0]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        if self.validation_status not in {"pending", "passed", "failed"}:
            raise ValueError("validation_status must be pending|passed|failed")
        if self.supervision_hints is None:
            raise ValueError("supervision_hints is required")

    def to_jsonl_enrichment(self) -> dict:
        """Convert to merge-ready JSONL enrichment record."""
        return {
            "episode_id": self.episode_id,
            "enrichment": {
                "fragility_tags": [t.to_dict() for t in self.fragility_tags],
                "risk_tags": [t.to_dict() for t in self.risk_tags],
                "affordance_tags": [t.to_dict() for t in self.affordance_tags],
                "efficiency_tags": [t.to_dict() for t in self.efficiency_tags],
                "novelty_tags": [t.to_dict() for t in self.novelty_tags],
                "intervention_tags": [t.to_dict() for t in self.intervention_tags],
                "semantic_conflicts": [c.to_dict() for c in self.semantic_conflicts],
                "coherence_score": self.coherence_score,
                "supervision_hints": self.supervision_hints.to_dict(),
                "confidence": self.confidence,
                "source_proposals": self.source_proposals,
                "validation_status": self.validation_status,
            },
        }

    def to_dict(self) -> dict:
        """Full dictionary (includes justification + validation errors)."""
        base = self.to_jsonl_enrichment()
        base["enrichment"]["justification"] = self.justification
        base["enrichment"]["validation_errors"] = self.validation_errors
        base["metadata"] = {
            "proposal_id": self.proposal_id,
            "timestamp": self.timestamp,
            "video_id": self.video_id,
            "task": self.task,
        }
        return base
