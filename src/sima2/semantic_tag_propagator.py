"""
Stage 2.4 SemanticTagPropagator

Generates advisory-only semantic enrichments for Stage 1 datapacks using
validated Stage 2.2 ontology proposals, Stage 2.3 task graph proposals, and
Stage 2.1 economics outputs. Deterministic and JSON-safe by construction.
"""

import copy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.sima2.config import get_segmentation_config
from src.sima2.tags import (
    AffordanceTag,
    EfficiencyTag,
    FragilityTag,
    InterventionTag,
    NoveltyTag,
    RiskTag,
    SegmentBoundaryTag,
    SemanticConflict,
    SemanticEnrichmentProposal,
    SubtaskTag,
    SupervisionHints,
    MobilityRiskTag,
    ContactQualityTag,
    PrecisionToleranceTag,
    RecoveryPatternTag,
)
from src.sima2.tags.ood_recovery_tags import detect_ood_from_segment, detect_recovery_from_segments

FORBIDDEN_FIELDS = {
    "rewards",
    "mpl_value",
    "wage_parity",
    "sampling_weight",
    "task_order",
    "affordance_definitions",
}


def _get_status(obj: Any) -> str:
    """Best-effort extraction of validation_status."""
    if isinstance(obj, dict):
        return obj.get("validation_status", "passed")
    return getattr(obj, "validation_status", "passed")


def _get_proposal_id(obj: Any) -> str:
    if isinstance(obj, dict):
        return obj.get("proposal_id", "")
    return getattr(obj, "proposal_id", "")


def _deepcopy_list(items: Sequence[Any]) -> List[Any]:
    """Copy inputs defensively to guarantee no mutation."""
    return copy.deepcopy(list(items))


def _extract_objects(datapack: Dict[str, Any]) -> List[str]:
    """Safely extract objects present in a datapack."""
    metadata = datapack.get("metadata", {}) or {}
    objects = metadata.get("objects_present", [])
    if objects:
        return list(objects)

    # Fallback: infer from task name (split on delimiters)
    fallback = datapack.get("task", "") or ""
    if fallback:
        return [part for part in fallback.replace("-", "_").split("_") if part]
    return []


class OntologyMatcher:
    """Generates affordance and fragility tags from validated ontology proposals."""

    def __init__(self, ontology_proposals: Sequence[Any]):
        self.proposals = [
            copy.deepcopy(p) for p in ontology_proposals if _get_status(p) == "passed"
        ]

    def match_affordances(
        self, datapack: Dict[str, Any]
    ) -> Tuple[List[AffordanceTag], List[str]]:
        tags: List[AffordanceTag] = []
        used_ids: List[str] = []
        objects = _extract_objects(datapack)

        for obj in sorted(objects):
            for prop in self.proposals:
                for aff in self._extract_affordances(prop):
                    if aff.get("object") != obj:
                        continue
                    tags.append(
                        AffordanceTag(
                            affordance_name=aff["name"],
                            object_name=obj,
                            demonstrated=self._check_demonstrated(datapack),
                            alternative_affordances=self._find_alternatives(prop, obj, aff["name"]),
                            justification=f"From ontology proposal {_get_proposal_id(prop)}",
                        )
                    )
                    used_ids.append(_get_proposal_id(prop))

        tags.sort(key=lambda t: (t.object_name, t.affordance_name))
        return tags, sorted(set(used_ids))

    def match_fragilities(
        self, datapack: Dict[str, Any]
    ) -> Tuple[List[FragilityTag], List[str]]:
        tags: List[FragilityTag] = []
        used_ids: List[str] = []
        objects = _extract_objects(datapack)

        for obj in sorted(objects):
            for prop in self.proposals:
                for frag in self._extract_fragilities(prop):
                    if frag.get("object") != obj:
                        continue
                    tags.append(
                        FragilityTag(
                            object_name=obj,
                            fragility_level=frag.get("level", "medium"),
                            damage_cost_usd=float(frag.get("damage_cost", frag.get("damage_cost_usd", 0.0))),
                            contact_frames=self._find_contact_frames(datapack),
                            justification=f"From ontology proposal {_get_proposal_id(prop)}",
                        )
                    )
                    used_ids.append(_get_proposal_id(prop))

        tags.sort(key=lambda t: (t.object_name, t.contact_frames[0] if t.contact_frames else 0))
        return tags, sorted(set(used_ids))

    def _extract_affordances(self, proposal: Any) -> List[Dict[str, Any]]:
        if isinstance(proposal, dict):
            return proposal.get("new_affordances", [])
        return getattr(proposal, "new_affordances", [])

    def _extract_fragilities(self, proposal: Any) -> List[Dict[str, Any]]:
        if isinstance(proposal, dict):
            return proposal.get("fragility_updates", [])
        return getattr(proposal, "fragility_updates", [])

    def _check_demonstrated(self, datapack: Dict[str, Any]) -> bool:
        metadata = datapack.get("metadata", {}) or {}
        return bool(metadata.get("success"))

    def _find_alternatives(
        self, proposal: Any, obj: str, affordance_name: str
    ) -> List[str]:
        """List other affordances for the same object from the proposal."""
        alternatives: List[str] = []
        for alt in self._extract_affordances(proposal):
            if alt.get("object") == obj and alt.get("name") != affordance_name:
                alternatives.append(alt.get("name", ""))
        return sorted({alt for alt in alternatives if alt})

    def _find_contact_frames(self, datapack: Dict[str, Any]) -> List[int]:
        frames = datapack.get("frames", []) or []
        if not frames:
            return [0]
        mid = len(frames) // 2
        return [mid]


class TaskGraphMatcher:
    """Generates risk and efficiency tags from validated task graph proposals."""

    def __init__(self, task_graph_proposals: Sequence[Any]):
        self.proposals = [
            copy.deepcopy(p) for p in task_graph_proposals if _get_status(p) == "passed"
        ]

    def match_risks(
        self, datapack: Dict[str, Any]
    ) -> Tuple[List[RiskTag], List[str]]:
        tags: List[RiskTag] = []
        used_ids: List[str] = []
        task = datapack.get("task")

        for prop in self.proposals:
            for risk in self._extract_risks(prop):
                if risk.get("task") != task:
                    continue
                tags.append(
                    RiskTag(
                        risk_type=risk["risk_type"],
                        severity=risk.get("severity", "medium"),
                        affected_frames=self._find_affected_frames(datapack),
                        mitigation_hints=risk.get("mitigation_hints", []),
                        justification=f"From task graph proposal {_get_proposal_id(prop)}",
                    )
                )
                used_ids.append(_get_proposal_id(prop))

        tags.sort(key=lambda t: (t.affected_frames[0] if t.affected_frames else 0, t.risk_type))
        return tags, sorted(set(used_ids))

    def match_efficiencies(
        self, datapack: Dict[str, Any]
    ) -> Tuple[List[EfficiencyTag], List[str]]:
        tags: List[EfficiencyTag] = []
        used_ids: List[str] = []
        task = datapack.get("task")

        for prop in self.proposals:
            for eff in self._extract_efficiencies(prop):
                if eff.get("task") != task:
                    continue
                score = self._compute_efficiency_score(datapack, eff)
                tags.append(
                    EfficiencyTag(
                        metric=eff["metric"],
                        score=score,
                        benchmark=eff.get("benchmark", f"avg_{task}_{eff['metric']}"),
                        improvement_hints=[hint for hint in eff.get("improvement_hints", [eff.get("suggestion", "")]) if hint],
                        justification=f"From task graph proposal {_get_proposal_id(prop)}",
                    )
                )
                used_ids.append(_get_proposal_id(prop))

        tags.sort(key=lambda t: (t.metric, -t.score))
        return tags, sorted(set(used_ids))

    def _extract_risks(self, proposal: Any) -> List[Dict[str, Any]]:
        if isinstance(proposal, dict):
            return proposal.get("risk_annotations", [])
        return getattr(proposal, "risk_annotations", [])

    def _extract_efficiencies(self, proposal: Any) -> List[Dict[str, Any]]:
        if isinstance(proposal, dict):
            return proposal.get("efficiency_hints", [])
        return getattr(proposal, "efficiency_hints", [])

    def _find_affected_frames(self, datapack: Dict[str, Any]) -> List[int]:
        frames = datapack.get("frames", []) or []
        if not frames:
            return [0]
        mid = len(frames) // 2
        return [mid]

    def _compute_efficiency_score(
        self, datapack: Dict[str, Any], eff_hint: Dict[str, Any]
    ) -> float:
        metric = eff_hint.get("metric")
        metadata = datapack.get("metadata", {}) or {}

        if metric == "time":
            duration = float(metadata.get("duration_sec", 0.0)) or 0.0
            benchmark = float(eff_hint.get("benchmark_time", 8.0))
            if duration <= 0:
                return 0.5
            return max(0.0, min(1.0, benchmark / duration))

        if metric == "energy":
            energy_used = float(metadata.get("energy_used", 0.0)) or 0.0
            benchmark = float(eff_hint.get("benchmark_energy", energy_used + 1.0))
            if benchmark <= 0:
                return 0.5
            return max(0.0, min(1.0, benchmark / max(energy_used, 1e-6)))

        # Default neutral score
        return 0.5


class EconomicsMatcher:
    """Generates novelty tags from economics outputs (required input)."""

    def __init__(self, economics_outputs: Dict[str, Dict[str, Any]]):
        self.economics = copy.deepcopy(economics_outputs)

    def validate_economics_present(self, episode_id: str) -> bool:
        return episode_id in self.economics

    def match_novelty(self, datapack: Dict[str, Any]) -> List[NoveltyTag]:
        episode_id = datapack["episode_id"]
        econ_data = self.economics.get(episode_id)
        if not econ_data:
            raise ValueError(f"Missing economics data for episode {episode_id}")

        novelty_score = float(econ_data.get("novelty_score", 0.0))
        expected_mpl_gain = float(econ_data.get("expected_mpl_gain", 0.0))
        tier = int(econ_data.get("tier", 0))
        novelty_types = {
            0: "state_coverage",
            1: "action_diversity",
            2: "edge_case",
        }
        novelty_type = novelty_types.get(tier, "state_coverage")

        tag = NoveltyTag(
            novelty_type=novelty_type,
            novelty_score=max(0.0, min(1.0, novelty_score)),
            comparison_basis=f"all_{datapack.get('task', 'task')}_episodes",
            expected_mpl_gain=max(0.0, expected_mpl_gain),
            justification=f"Tier {tier} novelty from economics module",
        )
        return [tag]


class CoherenceChecker:
    """Detects semantic conflicts and computes coherence scores."""

    def detect_conflicts(
        self,
        fragility_tags: List[FragilityTag],
        risk_tags: List[RiskTag],
        affordance_tags: List[AffordanceTag],
        efficiency_tags: List[EfficiencyTag],
        novelty_tags: List[NoveltyTag],
        intervention_tags: List[InterventionTag],
        supervision_hints: SupervisionHints,
    ) -> List[SemanticConflict]:
        conflicts: List[SemanticConflict] = []

        # High fragility but low/absent risk tags
        high_fragility = any(t.fragility_level in {"high", "critical"} for t in fragility_tags)
        low_risk = not risk_tags or all(t.severity in {"low", "medium"} for t in risk_tags)
        if high_fragility and low_risk:
            conflicts.append(
                SemanticConflict(
                    conflict_type="fragility_risk_mismatch",
                    conflicting_tags=["fragility_tags", "risk_tags"],
                    severity="medium",
                    resolution_hint="Add risk annotations for fragile objects",
                    justification="Fragility present without matching risk tags",
                )
            )

        # Efficiency very high but novelty also high (unusual combination)
        high_eff = any(t.score > 0.8 for t in efficiency_tags)
        high_novel = any(t.novelty_score > 0.7 for t in novelty_tags)
        if high_eff and high_novel:
            conflicts.append(
                SemanticConflict(
                    conflict_type="efficiency_vs_novelty",
                    conflicting_tags=["efficiency:high", "novelty:high"],
                    severity="low",
                    resolution_hint="Verify expert demonstration or mislabeled novelty",
                    justification="High efficiency on highly novel data",
                )
            )

        # Safety-critical hints but missing risk tags
        if supervision_hints.safety_critical and not risk_tags:
            conflicts.append(
                SemanticConflict(
                    conflict_type="safety_risk_missing",
                    conflicting_tags=["supervision:safety_critical", "risk_tags:empty"],
                    severity="medium",
                    resolution_hint="Provide risk tags for safety-critical episodes",
                    justification="Safety-critical flag requires explicit risks",
                )
            )

        # Human intervention present should reduce coherence slightly
        if intervention_tags:
            conflicts.append(
                SemanticConflict(
                    conflict_type="human_intervention",
                    conflicting_tags=["intervention_tags"],
                    severity="low",
                    resolution_hint="Review intervention segments for consistency",
                    justification="Human intervention indicates potential inconsistency",
                )
            )

        return conflicts

    def compute_coherence_score(self, conflicts: List[SemanticConflict]) -> float:
        if not conflicts:
            return 1.0
        severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.5}
        total_penalty = sum(severity_weights.get(c.severity, 0.3) for c in conflicts)
        return max(0.0, 1.0 - min(1.0, total_penalty))


class SupervisionHintsGenerator:
    """Derives orchestration hints from tags and conflicts."""

    def generate(
        self,
        task: str,
        fragility_tags: List[FragilityTag],
        risk_tags: List[RiskTag],
        novelty_tags: List[NoveltyTag],
        conflicts: List[SemanticConflict],
        coherence_score: float,
    ) -> SupervisionHints:
        priority_level = self._compute_priority_level(novelty_tags, risk_tags, conflicts)
        weight_multiplier = self._compute_weight_multiplier(priority_level)
        replay_frequency = self._compute_replay_frequency(priority_level)
        safety_critical = self._check_safety_critical(fragility_tags, risk_tags)
        requires_human_review = self._check_human_review(conflicts, safety_critical)
        curriculum_stage = self._assign_curriculum_stage(fragility_tags, risk_tags)
        prerequisite_tags = self._identify_prerequisites(task, curriculum_stage, fragility_tags)
        justification = self._generate_justification(
            priority_level, novelty_tags, safety_critical, curriculum_stage, coherence_score
        )

        return SupervisionHints(
            prioritize_for_training=priority_level in {"high", "critical"},
            priority_level=priority_level,
            suggested_weight_multiplier=weight_multiplier,
            suggested_replay_frequency=replay_frequency,
            requires_human_review=requires_human_review,
            safety_critical=safety_critical,
            curriculum_stage=curriculum_stage,
            prerequisite_tags=prerequisite_tags,
            justification=justification,
        )

    def _compute_priority_level(
        self, novelty_tags: List[NoveltyTag], risk_tags: List[RiskTag], conflicts: List[SemanticConflict]
    ) -> str:
        max_novelty = max((t.novelty_score for t in novelty_tags), default=0.0)
        has_high_risk = any(t.severity in {"high", "critical"} for t in risk_tags)
        has_conflict = bool(conflicts)

        if max_novelty > 0.7 or has_high_risk:
            return "high"
        if max_novelty > 0.4 or has_conflict:
            return "medium"
        return "low"

    def _compute_weight_multiplier(self, priority_level: str) -> float:
        return {"low": 1.0, "medium": 1.5, "high": 2.0, "critical": 3.0}[priority_level]

    def _compute_replay_frequency(self, priority_level: str) -> str:
        if priority_level in {"high", "critical"}:
            return "frequent"
        if priority_level == "low":
            return "rare"
        return "standard"

    def _check_safety_critical(
        self, fragility_tags: List[FragilityTag], risk_tags: List[RiskTag]
    ) -> bool:
        has_high_fragility = any(t.fragility_level in {"high", "critical"} for t in fragility_tags)
        has_high_risk = any(t.severity in {"high", "critical"} for t in risk_tags)
        return has_high_fragility or has_high_risk

    def _check_human_review(
        self, conflicts: List[SemanticConflict], safety_critical: bool
    ) -> bool:
        has_high_conflict = any(c.severity == "high" for c in conflicts)
        return has_high_conflict or (safety_critical and bool(conflicts))

    def _assign_curriculum_stage(
        self, fragility_tags: List[FragilityTag], risk_tags: List[RiskTag]
    ) -> str:
        has_high_fragility = any(t.fragility_level in {"high", "critical"} for t in fragility_tags)
        has_high_risk = any(t.severity in {"high", "critical"} for t in risk_tags)
        if has_high_fragility or has_high_risk:
            return "advanced"
        if risk_tags:
            return "mid"
        return "early"

    def _identify_prerequisites(
        self, task: str, curriculum_stage: str, fragility_tags: List[FragilityTag]
    ) -> List[str]:
        prereqs: List[str] = []
        if curriculum_stage == "advanced":
            prereqs.append(f"basic_{task}")
            if fragility_tags:
                prereqs.append("fragile_object_awareness")
        return prereqs

    def _generate_justification(
        self,
        priority_level: str,
        novelty_tags: List[NoveltyTag],
        safety_critical: bool,
        curriculum_stage: str,
        coherence_score: float,
    ) -> str:
        reasons: List[str] = []
        if priority_level in {"high", "critical"}:
            reasons.append(f"{priority_level} priority")
        if novelty_tags and novelty_tags[0].novelty_score > 0.7:
            reasons.append(f"{novelty_tags[0].novelty_type} novelty")
        if safety_critical:
            reasons.append("safety-critical scenario")
        if curriculum_stage == "advanced":
            reasons.append("advanced curriculum")
        if coherence_score < 0.8:
            reasons.append(f"coherence={coherence_score:.2f}")
        return " + ".join(reasons) if reasons else "standard training episode"


class SegmentationBuilder:
    """Builds deterministic segment boundaries and subtask tags."""

    def __init__(self, segmentation_config: Optional[Dict[str, Any]] = None):
        cfg = get_segmentation_config(segmentation_config)
        self.min_segment_length = int(cfg.get("min_segment_length", 1))
        self.max_idle_gap = int(cfg.get("max_idle_gap", 0))
        self.max_segment_length = int(cfg.get("max_segment_length", 1_000_000))
        self.risk_jump_delta = float(cfg.get("risk_jump_delta", 0))
        self.allow_risk_jumps = bool(cfg.get("allow_risk_jumps", True))

    def build(self, datapack: Dict[str, Any]) -> Tuple[List[SegmentBoundaryTag], List[SubtaskTag]]:
        episode_id = datapack.get("episode_id", "")
        primitives = self._extract_primitives(datapack)
        if not primitives:
            primitives = [{"timestep": 0, "object": "unknown", "risk": "low"}]
        primitives = sorted(primitives, key=lambda p: p.get("timestep", 0))

        segment_boundary_tags: List[SegmentBoundaryTag] = []
        subtask_tags: List[SubtaskTag] = []

        current_segment = 0
        current_label = self._label_for_primitive(primitives[0])
        start_ts = primitives[0].get("timestep", 0)
        current_risk = self._risk_level(str(primitives[0].get("risk", "low")))
        segment_id = self._segment_id(episode_id, current_segment)
        segment_boundary_tags.append(
            SegmentBoundaryTag(
                episode_id=episode_id,
                segment_id=segment_id,
                timestep=start_ts,
                reason="start",
                subtask_label=current_label,
            )
        )

        last_ts = start_ts
        for idx in range(1, len(primitives)):
            prim = primitives[idx]
            ts = prim.get("timestep", last_ts + 1)
            label = self._label_for_primitive(prim)
            risk_val = self._risk_level(str(prim.get("risk", "low")))
            object_change = label != current_label
            status = str(prim.get("status", "")).lower()
            gap = ts - last_ts
            duration = max(last_ts, ts) - start_ts

            risk_jump = self.allow_risk_jumps and abs(risk_val - current_risk) >= self.risk_jump_delta
            long_gap = gap > self.max_idle_gap if self.max_idle_gap >= 0 else False
            exceeded_length = duration >= self.max_segment_length
            ready_to_split = duration >= self.min_segment_length

            if (ready_to_split and (object_change or risk_jump or long_gap)) or exceeded_length:
                segment_boundary_tags.append(
                    SegmentBoundaryTag(
                        episode_id=episode_id,
                        segment_id=segment_id,
                        timestep=max(last_ts, ts),
                        reason="end",
                        subtask_label=current_label,
                    )
                )
                subtask_tags.append(
                    SubtaskTag(
                        episode_id=episode_id,
                        segment_id=segment_id,
                        subtask_label=current_label or "unknown_subtask",
                        parent_segment_id=None,
                    )
                )
                current_segment += 1
                segment_id = self._segment_id(episode_id, current_segment)
                segment_boundary_tags.append(
                    SegmentBoundaryTag(
                        episode_id=episode_id,
                        segment_id=segment_id,
                        timestep=ts,
                        reason="start",
                        subtask_label=label,
                    )
                )
                current_label = label
                start_ts = ts
                duration = 0

            if status == "failure":
                segment_boundary_tags.append(
                    SegmentBoundaryTag(
                        episode_id=episode_id,
                        segment_id=segment_id,
                        timestep=ts,
                        reason="failure",
                        subtask_label=current_label,
                    )
                )
            if status == "recovery":
                segment_boundary_tags.append(
                    SegmentBoundaryTag(
                        episode_id=episode_id,
                        segment_id=segment_id,
                        timestep=ts,
                        reason="recovery",
                        subtask_label=current_label,
                    )
                )
            last_ts = ts
            current_risk = risk_val

        segment_boundary_tags.append(
            SegmentBoundaryTag(
                episode_id=episode_id,
                segment_id=segment_id,
                timestep=last_ts if primitives else 0,
                reason="end",
                subtask_label=current_label,
            )
        )
        subtask_tags.append(
            SubtaskTag(
                episode_id=episode_id,
                segment_id=segment_id,
                subtask_label=current_label or "unknown_subtask",
                parent_segment_id=None,
            )
        )

        segment_boundary_tags = sorted(
            segment_boundary_tags, key=lambda t: (t.timestep, t.segment_id, t.reason)
        )
        subtask_tags = sorted(subtask_tags, key=lambda t: (t.segment_id, t.subtask_label))
        return segment_boundary_tags, subtask_tags

    def _extract_primitives(self, datapack: Dict[str, Any]) -> List[Dict[str, Any]]:
        primitives = datapack.get("primitives") or datapack.get("primitive_events")
        if isinstance(primitives, list):
            return [p for p in primitives if isinstance(p, dict)]
        metadata = datapack.get("metadata", {}) or {}
        timeline = metadata.get("semantic_primitives") or metadata.get("primitive_events")
        if isinstance(timeline, list):
            return [p for p in timeline if isinstance(p, dict)]
        return []

    def _risk_level(self, risk: str) -> int:
        levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return levels.get(risk.lower(), 0)

    def _label_for_primitive(self, prim: Dict[str, Any]) -> str:
        obj = str(prim.get("object") or prim.get("focus") or prim.get("object_focus") or "")
        action = str(prim.get("action") or prim.get("primitive") or "")
        label_parts: List[str] = []
        if obj:
            label_parts.append(obj)
        if action:
            label_parts.append(action)
        label = "_".join([p for p in label_parts if p])
        if "drawer" in obj:
            return "drawer_interaction" if not action else f"drawer_{action}"
        if "vase" in obj:
            return "vase_handling" if not action else f"vase_{action}"
        if label:
            return label
        return "generic_segment"

    def _segment_id(self, episode_id: str, idx: int) -> str:
        return f"{episode_id}_seg{idx}"


class SemanticTagPropagator:
    """Generates semantic enrichments for datapacks (advisory-only)."""

    def __init__(self, trust_matrix: Optional[Dict[str, Any]] = None, enable_ood_recovery_tags: bool = False) -> None:
        self.coherence_checker = CoherenceChecker()
        self.hints_generator = SupervisionHintsGenerator()
        self.segmentation_config = get_segmentation_config({})
        self.segmentation_builder = SegmentationBuilder(self.segmentation_config)
        self.trust_matrix = trust_matrix or {}
        self.enable_ood_recovery_tags = bool(enable_ood_recovery_tags)

    def generate_proposals(
        self,
        datapacks: Sequence[Dict[str, Any]],
        ontology_proposals: Sequence[Any],
        task_graph_proposals: Sequence[Any],
        economics_outputs: Dict[str, Dict[str, Any]],
    ) -> List[SemanticEnrichmentProposal]:
        sorted_datapacks = sorted(_deepcopy_list(datapacks), key=lambda d: d.get("episode_id", ""))
        ontology_matcher = OntologyMatcher(ontology_proposals)
        task_graph_matcher = TaskGraphMatcher(task_graph_proposals)
        economics_matcher = EconomicsMatcher(economics_outputs)
        segmentation_builder = self.segmentation_builder

        proposals: List[SemanticEnrichmentProposal] = []
        for datapack in sorted_datapacks:
            proposal = self.generate_single_proposal(
                datapack, ontology_matcher, task_graph_matcher, economics_matcher
            )
            proposals.append(proposal)

        # Deterministic ordering by episode_id then timestamp
        proposals.sort(key=lambda p: (p.episode_id, p.timestamp))
        return proposals

    def generate_single_proposal(
        self,
        datapack: Dict[str, Any],
        ontology_matcher: OntologyMatcher,
        task_graph_matcher: TaskGraphMatcher,
        economics_matcher: EconomicsMatcher,
    ) -> SemanticEnrichmentProposal:
        datapack_copy = copy.deepcopy(datapack)
        episode_id = datapack_copy["episode_id"]
        video_id = datapack_copy.get("video_id", "unknown")
        task = datapack_copy.get("task", "unknown")

        if not economics_matcher.validate_economics_present(episode_id):
            return self._create_failed_proposal(
                episode_id, video_id, task, f"Missing economics data for episode {episode_id}"
            )

        try:
            fragility_tags, ont_frag_ids = ontology_matcher.match_fragilities(datapack_copy)
            affordance_tags, ont_aff_ids = ontology_matcher.match_affordances(datapack_copy)
            risk_tags, task_risk_ids = task_graph_matcher.match_risks(datapack_copy)
            efficiency_tags, task_eff_ids = task_graph_matcher.match_efficiencies(datapack_copy)
            novelty_tags = economics_matcher.match_novelty(datapack_copy)
            intervention_tags = self._detect_interventions(datapack_copy)
            segment_boundary_tags, subtask_tags = self.segmentation_builder.build(datapack_copy)
            mobility_risk_tags, contact_quality_tags, precision_tolerance_tags, recovery_pattern_tags = self._infer_mobility_tags(datapack_copy, segment_boundary_tags)
        except Exception as exc:  # pragma: no cover - defensive
            return self._create_failed_proposal(
                episode_id, video_id, task, f"Tag generation failed: {exc}"
            )

        fragility_tags.sort(key=lambda t: (t.object_name, t.contact_frames[0]))
        risk_tags.sort(key=lambda t: (t.affected_frames[0], t.risk_type))
        affordance_tags.sort(key=lambda t: (t.object_name, t.affordance_name))
        efficiency_tags.sort(key=lambda t: (t.metric, -t.score))
        novelty_tags.sort(key=lambda t: (-t.novelty_score, t.novelty_type))
        intervention_tags.sort(key=lambda t: t.frame_range[0])
        segment_boundary_tags.sort(key=lambda t: (t.timestep, t.segment_id, t.reason))
        subtask_tags.sort(key=lambda t: (t.segment_id, t.subtask_label))
        mobility_risk_tags.sort(key=lambda t: t.level)
        contact_quality_tags.sort(key=lambda t: t.quality)
        precision_tolerance_tags.sort(key=lambda t: (t.grade, t.target_mm))
        recovery_pattern_tags.sort(key=lambda t: t.recovery_rate)
        ood_tags_payload: List[Dict[str, Any]] = []
        recovery_tags_payload: List[Dict[str, Any]] = []
        if self.enable_ood_recovery_tags:
            segments_for_detection = self._segments_for_detection(segment_boundary_tags, subtask_tags, datapack_copy)
            detection_cfg = {
                "trust_matrix": self.trust_matrix,
                "visual_threshold": self.segmentation_config.get("visual_threshold", 0.7),
                "mean_durations": self.segmentation_config.get("mean_durations", {}),
                "training_stats": self.segmentation_config.get("training_stats", {}),
            }
            for seg in segments_for_detection:
                ood_tag = detect_ood_from_segment(seg, config=detection_cfg)
                if ood_tag:
                    wrapped = self._wrap_trust_tag("OODTag", ood_tag.to_dict())
                    if wrapped:
                        ood_tags_payload.append(wrapped)
            recovery_cfg = {
                "trust_matrix": self.trust_matrix,
                "base_power_per_timestep": detection_cfg.get("base_power_per_timestep", 0.5),
            }
            for rec_tag in detect_recovery_from_segments(segments_for_detection, config=recovery_cfg):
                wrapped = self._wrap_trust_tag("RecoveryTag", rec_tag.to_dict())
                if wrapped:
                    recovery_tags_payload.append(wrapped)

        preliminary_hints = self.hints_generator.generate(
            task, fragility_tags, risk_tags, novelty_tags, [], 1.0
        )
        conflicts = self.coherence_checker.detect_conflicts(
            fragility_tags,
            risk_tags,
            affordance_tags,
            efficiency_tags,
            novelty_tags,
            intervention_tags,
            preliminary_hints,
        )
        coherence_score = self.coherence_checker.compute_coherence_score(conflicts)
        supervision_hints = self.hints_generator.generate(
            task, fragility_tags, risk_tags, novelty_tags, conflicts, coherence_score
        )

        source_proposals = sorted(set(ont_frag_ids + ont_aff_ids + task_risk_ids + task_eff_ids))
        confidence = self._compute_confidence(
            ontology_matcher, task_graph_matcher, coherence_score, datapack_copy
        )

        proposal_id = self._generate_proposal_id(
            datapack_copy, source_proposals, fragility_tags, risk_tags, novelty_tags
        )
        enrichment_dict = {
            "fragility_tags": [t.to_dict() for t in fragility_tags],
            "risk_tags": [t.to_dict() for t in risk_tags],
            "affordance_tags": [t.to_dict() for t in affordance_tags],
            "efficiency_tags": [t.to_dict() for t in efficiency_tags],
            "novelty_tags": [t.to_dict() for t in novelty_tags],
            "intervention_tags": [t.to_dict() for t in intervention_tags],
            "segment_boundary_tags": [t.to_dict() for t in segment_boundary_tags],
            "subtask_tags": [t.to_dict() for t in subtask_tags],
            "mobility_risk_tags": [t.to_dict() for t in mobility_risk_tags],
            "contact_quality_tags": [t.to_dict() for t in contact_quality_tags],
            "precision_tolerance_tags": [t.to_dict() for t in precision_tolerance_tags],
            "recovery_pattern_tags": [t.to_dict() for t in recovery_pattern_tags],
            "semantic_conflicts": [c.to_dict() for c in conflicts],
            "coherence_score": coherence_score,
            "supervision_hints": supervision_hints.to_dict(),
            "confidence": confidence,
            "source_proposals": source_proposals,
            "validation_status": "pending",
        }
        validation_errors = self._validate_enrichment_schema(enrichment_dict)
        validation_status = "passed" if not validation_errors else "failed"

        justification = self._generate_justification(task, fragility_tags, novelty_tags, supervision_hints)

        return SemanticEnrichmentProposal(
            proposal_id=proposal_id,
            timestamp=datapack_copy.get("metadata", {}).get("timestamp", 0.0),
            video_id=video_id,
            episode_id=episode_id,
            task=task,
            fragility_tags=fragility_tags,
            risk_tags=risk_tags,
            affordance_tags=affordance_tags,
            efficiency_tags=efficiency_tags,
            novelty_tags=novelty_tags,
            intervention_tags=intervention_tags,
            segment_boundary_tags=segment_boundary_tags,
            subtask_tags=subtask_tags,
            mobility_risk_tags=mobility_risk_tags,
            contact_quality_tags=contact_quality_tags,
            precision_tolerance_tags=precision_tolerance_tags,
            recovery_pattern_tags=recovery_pattern_tags,
            ood_tags=ood_tags_payload,
            recovery_tags=recovery_tags_payload,
            semantic_conflicts=conflicts,
            coherence_score=coherence_score,
            supervision_hints=supervision_hints,
            confidence=confidence,
            source_proposals=source_proposals,
            justification=justification,
            validation_status=validation_status,
            validation_errors=validation_errors,
        )

    def _segments_for_detection(
        self,
        boundaries: List[SegmentBoundaryTag],
        subtask_tags: List[SubtaskTag],
        datapack: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Reconstruct lightweight segments from boundary tags for tag detection."""
        subtask_lookup = {st.segment_id: st.subtask_label for st in subtask_tags}
        segments: Dict[str, Dict[str, Any]] = {}
        for b in boundaries:
            seg = segments.setdefault(
                b.segment_id,
                {"start": b.timestep, "end": b.timestep, "failure": False, "recovery": False, "label": b.subtask_label},
            )
            if b.reason == "start":
                seg["start"] = min(seg["start"], b.timestep)
            seg["end"] = max(seg["end"], b.timestep)
            if b.reason == "failure":
                seg["failure"] = True
            if b.reason == "recovery":
                seg["recovery"] = True
            if not seg.get("label"):
                seg["label"] = subtask_lookup.get(b.segment_id)

        objects_present = (datapack.get("metadata", {}) or {}).get("objects_present", [])
        seg_list: List[Dict[str, Any]] = []
        for idx, (seg_id, seg_meta) in enumerate(sorted(segments.items(), key=lambda item: item[0])):
            start = int(seg_meta.get("start", 0))
            end = int(seg_meta.get("end", start))
            duration = max(0, end - start)
            outcome = "success"
            if seg_meta.get("failure") and seg_meta.get("recovery"):
                outcome = "recovered"
            elif seg_meta.get("failure"):
                outcome = "failure"
            elif seg_meta.get("recovery"):
                outcome = "recovery"
            seg_metadata = {
                "objects_present": list(objects_present),
                "segment_index": idx,
                "recovery_observed": bool(seg_meta.get("recovery")),
                "failure_observed": bool(seg_meta.get("failure")),
                "duration": duration,
                "label": seg_meta.get("label") or subtask_lookup.get(seg_id) or "segment",
            }
            seg_metadata.update(datapack.get("metadata", {}) or {})
            seg_list.append(
                {
                    "segment_id": seg_id,
                    "episode_id": datapack.get("episode_id"),
                    "label": seg_meta.get("label") or subtask_lookup.get(seg_id) or "segment",
                    "start_t": start,
                    "end_t": end,
                    "outcome": outcome,
                    "metadata": seg_metadata,
                }
            )
        return seg_list

    def _wrap_trust_tag(self, tag_type: str, tag_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply trust-tier semantics for tag inclusion."""
        trust_score = float(self.trust_matrix.get(tag_type, {}).get("trust_score", 1.0))
        if trust_score <= 0.5:
            return None
        payload = dict(tag_data)
        payload.setdefault("tag_type", tag_type)
        payload["trust_score"] = trust_score
        if trust_score > 0.8:
            payload["trust_tier"] = "Tier1_Trusted"
            payload["confidence"] = 0.9
        else:
            payload["trust_tier"] = "Tier2_Provisional"
            payload["confidence"] = 0.6
            payload["warning"] = "Tag has provisional trust; verify manually"
        return payload

    def aggregate_for_task(
        self, enrichments: List[SemanticEnrichmentProposal], task: Optional[str] = None
    ) -> Optional[SemanticEnrichmentProposal]:
        """Aggregate enrichments across videos for the same task."""
        if not enrichments:
            return None
        task_name = task or enrichments[0].task
        task_enrichments = [e for e in enrichments if e.task == task_name]
        if not task_enrichments:
            return None

        # Union tags without duplicates
        def _unique(tags: Iterable[Any], key_fn):
            seen = set()
            unique_tags = []
            for tag in tags:
                key = key_fn(tag)
                if key in seen:
                    continue
                seen.add(key)
                unique_tags.append(tag)
            return unique_tags

        fragility = _unique(
            sum([e.fragility_tags for e in task_enrichments], []),
            lambda t: (t.object_name, t.fragility_level),
        )
        risk = _unique(
            sum([e.risk_tags for e in task_enrichments], []), lambda t: (t.risk_type, t.severity)
        )
        affordance = _unique(
            sum([e.affordance_tags for e in task_enrichments], []),
            lambda t: (t.object_name, t.affordance_name),
        )
        efficiency = _unique(
            sum([e.efficiency_tags for e in task_enrichments], []),
            lambda t: (t.metric, round(t.score, 3)),
        )
        novelty = _unique(
            sum([e.novelty_tags for e in task_enrichments], []),
            lambda t: (t.novelty_type, round(t.novelty_score, 3)),
        )
        intervention = _unique(
            sum([e.intervention_tags for e in task_enrichments], []),
            lambda t: t.frame_range,
        )

        # Coherence: conservative minimum
        coherence_score = min(e.coherence_score for e in task_enrichments)
        conflicts = self.coherence_checker.detect_conflicts(
            fragility, risk, affordance, efficiency, novelty, intervention, task_enrichments[0].supervision_hints
        )
        coherence_score = self.coherence_checker.compute_coherence_score(conflicts)
        supervision_hints = self.hints_generator.generate(
            task_name, fragility, risk, novelty, conflicts, coherence_score
        )

        source_proposals = sorted(
            set(sum([e.source_proposals for e in task_enrichments], []))
        )

        return SemanticEnrichmentProposal(
            proposal_id=f"aggregated_{task_name}_{len(task_enrichments)}videos",
            timestamp=task_enrichments[0].timestamp,
            video_id=task_enrichments[0].video_id,
            episode_id=f"{task_name}_aggregate",
            task=task_name,
            fragility_tags=fragility,
            risk_tags=risk,
            affordance_tags=affordance,
            efficiency_tags=efficiency,
            novelty_tags=novelty,
            intervention_tags=intervention,
            semantic_conflicts=conflicts,
            coherence_score=coherence_score,
            supervision_hints=supervision_hints,
            confidence=min(e.confidence for e in task_enrichments),
            source_proposals=source_proposals,
            justification=f"Aggregated from {len(task_enrichments)} enrichments",
            validation_status="passed",
            validation_errors=[],
        )

    def _detect_interventions(self, datapack: Dict[str, Any]) -> List[InterventionTag]:
        tags: List[InterventionTag] = []
        metadata = datapack.get("metadata", {}) or {}
        if metadata.get("human_intervention"):
            frame_range = metadata.get("intervention_frames")
            if frame_range and isinstance(frame_range, (list, tuple)) and len(frame_range) == 2:
                frame_tuple = (int(frame_range[0]), int(frame_range[1]))
            else:
                frame_tuple = (0, len(datapack.get("frames", []) or []))
            tags.append(
                InterventionTag(
                    intervention_type="human_correction",
                    frame_range=frame_tuple,
                    trigger=metadata.get("intervention_trigger", "unknown"),
                    learning_opportunity="Human intervention indicates recovery opportunity",
                    justification="human_intervention flag present in metadata",
                )
            )
        return tags

    def _compute_confidence(
        self,
        ontology_matcher: OntologyMatcher,
        task_graph_matcher: TaskGraphMatcher,
        coherence_score: float,
        datapack: Dict[str, Any],
    ) -> float:
        confidence = 0.8
        if not ontology_matcher.proposals:
            confidence -= 0.2
        if not task_graph_matcher.proposals:
            confidence -= 0.1
        if not datapack.get("metadata", {}).get("objects_present"):
            confidence -= 0.1
        confidence *= coherence_score
        return max(0.0, min(1.0, confidence))

    def _generate_proposal_id(
        self,
        datapack: Dict[str, Any],
        source_proposals: List[str],
        fragility_tags: List[FragilityTag],
        risk_tags: List[RiskTag],
        novelty_tags: List[NoveltyTag],
    ) -> str:
        id_payload = {
            "episode_id": datapack.get("episode_id"),
            "task": datapack.get("task"),
            "source_proposals": sorted(source_proposals),
            "fragility": [t.to_dict() for t in fragility_tags],
            "risk": [t.to_dict() for t in risk_tags],
            "novelty": [t.to_dict() for t in novelty_tags],
        }
        payload_str = json.dumps(id_payload, sort_keys=True)
        digest = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:12]
        return f"enrich_{datapack.get('episode_id')}_{digest}"

    def _validate_enrichment_schema(self, enrichment: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        required = {"coherence_score", "confidence", "validation_status"}
        missing = required - set(enrichment.keys())
        if missing:
            errors.append(f"Missing required fields: {sorted(missing)}")
        coherence = enrichment.get("coherence_score", 0.0)
        confidence = enrichment.get("confidence", 0.0)
        if not 0.0 <= coherence <= 1.0:
            errors.append("coherence_score must be in [0.0, 1.0]")
        if not 0.0 <= confidence <= 1.0:
            errors.append("confidence must be in [0.0, 1.0]")
        forbidden_found = FORBIDDEN_FIELDS.intersection(enrichment.keys())
        if forbidden_found:
            errors.append(f"Forbidden fields in enrichment: {sorted(forbidden_found)}")

        def _check_nested(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                bad = FORBIDDEN_FIELDS.intersection(obj.keys())
                if bad:
                    errors.append(f"Forbidden fields at {path or 'root'}: {sorted(bad)}")
                for k, v in obj.items():
                    _check_nested(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, (list, tuple)):
                for idx, v in enumerate(obj):
                    _check_nested(v, f"{path}[{idx}]")

        _check_nested(enrichment)
        return errors

    def _generate_justification(
        self,
        task: str,
        fragility_tags: List[FragilityTag],
        novelty_tags: List[NoveltyTag],
        hints: SupervisionHints,
    ) -> str:
        parts = [f"Task: {task}"]
        if fragility_tags:
            parts.append(f"{len(fragility_tags)} fragile object(s)")
        high_novelty = [t for t in novelty_tags if t.novelty_score > 0.5]
        if high_novelty:
            parts.append(f"{high_novelty[0].novelty_type} novelty")
        parts.append(f"priority={hints.priority_level}")
        return ", ".join(parts)

    def _infer_mobility_tags(
        self, datapack: Dict[str, Any], segment_boundaries: List[SegmentBoundaryTag]
    ) -> Tuple[List[MobilityRiskTag], List[ContactQualityTag], List[PrecisionToleranceTag], List[RecoveryPatternTag]]:
        md = datapack.get("metadata", {}) or {}
        mobility_meta = md.get("mobility", {})
        stability_history = mobility_meta.get("stability_margins", [])
        slip_history = mobility_meta.get("slip_rates", [])
        precision_target = float(mobility_meta.get("target_precision_mm", 5.0))
        achieved_precision = float(mobility_meta.get("achieved_precision_mm", precision_target))
        recovery_flags = mobility_meta.get("recovery_flags", [])

        stability_mean = float(sum(stability_history) / len(stability_history)) if stability_history else 1.0
        stability_min = float(min(stability_history)) if stability_history else 1.0
        level = "LOW"
        if stability_min < 0.3:
            level = "HIGH"
        elif stability_min < 0.6:
            level = "MEDIUM"
        mobility_risk_tags = [
            MobilityRiskTag(
                level=level,
                reason="stability_margin",
                stability_margin_mean=stability_mean,
                stability_margin_min=stability_min,
            )
        ]

        slip_freq = float(sum(1 for s in slip_history if s > 0.1) / max(len(slip_history), 1)) if slip_history else 0.0
        grasp_fail_rate = float(mobility_meta.get("grasp_fail_rate", 0.0))
        quality = "GOOD"
        if slip_freq > 0.3 or grasp_fail_rate > 0.2:
            quality = "POOR"
        elif slip_freq > 0.1:
            quality = "UNCERTAIN"
        contact_quality_tags = [
            ContactQualityTag(
                quality=quality,
                slip_frequency=slip_freq,
                grasp_fail_rate=grasp_fail_rate,
            )
        ]

        grade = "FAILED"
        if achieved_precision <= precision_target:
            grade = "GRADE_5" if precision_target <= 5.0 else "GRADE_3"
        elif achieved_precision <= precision_target * 1.5:
            grade = "GRADE_3"
        precision_tolerance_tags = [
            PrecisionToleranceTag(
                target_mm=precision_target,
                achieved_mm=achieved_precision,
                grade=grade,
                economic_importance=float(mobility_meta.get("economic_importance", 1.0)),
            )
        ]

        recovery_events = [b for b in segment_boundaries if b.reason in {"recovery", "failure"}]
        seg_ids = {b.segment_id for b in segment_boundaries} or {""}
        recovery_rate = float(len(recovery_events) / max(len(seg_ids), 1))
        catastrophic_recovery_rate = float(sum(1 for b in recovery_events if b.reason == "failure") / max(len(seg_ids), 1))
        recovery_pattern_tags = [
            RecoveryPatternTag(
                recovery_rate=recovery_rate,
                catastrophic_recovery_rate=catastrophic_recovery_rate,
                mean_recovery_time_steps=float(sum(b.timestep for b in recovery_events) / max(len(recovery_events), 1))
                if recovery_events
                else 0.0,
            )
        ]

        return mobility_risk_tags, contact_quality_tags, precision_tolerance_tags, recovery_pattern_tags

    def _create_failed_proposal(
        self, episode_id: str, video_id: str, task: str, error: str
    ) -> SemanticEnrichmentProposal:
        default_hints = SupervisionHints(
            prioritize_for_training=False,
            priority_level="low",
            suggested_weight_multiplier=1.0,
            suggested_replay_frequency="standard",
            requires_human_review=False,
            safety_critical=False,
            curriculum_stage="early",
            prerequisite_tags=[],
            justification="Validation failed",
        )
        return SemanticEnrichmentProposal(
            proposal_id=f"enrich_{episode_id}_failed",
            timestamp=0.0,
            video_id=video_id,
            episode_id=episode_id,
            task=task,
            fragility_tags=[],
            risk_tags=[],
            affordance_tags=[],
            efficiency_tags=[],
            novelty_tags=[],
            intervention_tags=[],
            semantic_conflicts=[],
            coherence_score=0.0,
            supervision_hints=default_hints,
            confidence=0.0,
            source_proposals=[],
            justification=error,
            validation_status="failed",
            validation_errors=[error],
        )
