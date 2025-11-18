# Stage 2.4: SemanticTagPropagator - Codex Implementation Guide

**Version**: 1.0
**Last Updated**: 2025-11-17

---

## Overview

This guide provides **concrete implementation instructions** for building the SemanticTagPropagator. Follow this step-by-step to ensure specification compliance.

**Prerequisites**:
- Stage 2.2 OntologyUpdateEngine implemented and tested
- Stage 2.3 TaskGraphRefiner implemented and tested
- Economics module (Stage 2.1) producing novelty scores and MPL gains
- Stage 1 datapacks in JSONL format

---

## File Structure

```
economics/
├── semantic_tag_propagator.py       # Main implementation
├── tag_types.py                     # Tag dataclass definitions
├── coherence_checker.py             # Semantic conflict detection
├── supervision_hints_generator.py   # Orchestrator hint generation
└── tag_matchers/
    ├── ontology_matcher.py          # Match ontology → tags
    ├── task_graph_matcher.py        # Match task graph → tags
    └── economics_matcher.py         # Match economics → tags

tests/
└── smoke_tests/
    └── test_semantic_tag_propagator.py  # Full smoke test suite

scripts/
└── smoke_test_tag_propagator.py     # Standalone smoke test runner
```

---

## Step 1: Implement Tag Dataclasses

**File**: `economics/tag_types.py`

```python
"""
Tag type definitions for semantic enrichments.
All tags must be JSON-serializable.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


@dataclass
class FragilityTag:
    """Tags fragile objects in episode"""
    object_name: str
    fragility_level: str  # "low" | "medium" | "high" | "critical"
    damage_cost_usd: float
    contact_frames: List[int]
    justification: str

    def __post_init__(self):
        assert self.fragility_level in ['low', 'medium', 'high', 'critical']
        assert self.damage_cost_usd >= 0.0
        assert len(self.contact_frames) > 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskTag:
    """Tags safety risks in episode"""
    risk_type: str  # "collision" | "tip_over" | "entanglement" | "human_proximity"
    severity: str   # "low" | "medium" | "high" | "critical"
    affected_frames: List[int]
    mitigation_hints: List[str]
    justification: str

    def __post_init__(self):
        assert self.severity in ['low', 'medium', 'high', 'critical']
        assert len(self.affected_frames) > 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AffordanceTag:
    """Tags demonstrated affordances"""
    affordance_name: str
    object_name: str
    demonstrated: bool
    alternative_affordances: List[str]
    justification: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EfficiencyTag:
    """Tags execution efficiency metrics"""
    metric: str  # "time" | "energy" | "precision" | "success_rate"
    score: float  # 0.0 = worst, 1.0 = optimal
    benchmark: str
    improvement_hints: List[str]
    justification: str

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NoveltyTag:
    """Tags novelty and expected value"""
    novelty_type: str  # "state_coverage" | "action_diversity" | "failure_mode" | "edge_case"
    novelty_score: float  # 0.0 = redundant, 1.0 = maximally novel
    comparison_basis: str
    expected_mpl_gain: float
    justification: str

    def __post_init__(self):
        assert 0.0 <= self.novelty_score <= 1.0
        assert self.expected_mpl_gain >= 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InterventionTag:
    """Tags human interventions or corrections"""
    intervention_type: str  # "human_correction" | "failure_recovery" | "safety_override"
    frame_range: Tuple[int, int]
    trigger: str
    learning_opportunity: str
    justification: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d['frame_range'] = list(self.frame_range)  # Tuple → list for JSON
        return d


@dataclass
class SemanticConflict:
    """Represents conflicting tags within enrichment"""
    conflict_type: str
    conflicting_tags: List[str]
    severity: str  # "low" | "medium" | "high"
    resolution_hint: str
    justification: str

    def __post_init__(self):
        assert self.severity in ['low', 'medium', 'high']

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SupervisionHints:
    """Orchestrator supervision hints"""
    prioritize_for_training: bool
    priority_level: str  # "low" | "medium" | "high" | "critical"

    suggested_weight_multiplier: float
    suggested_replay_frequency: str  # "standard" | "frequent" | "rare"

    requires_human_review: bool
    safety_critical: bool

    curriculum_stage: str  # "early" | "mid" | "late" | "advanced"
    prerequisite_tags: List[str]

    justification: str

    def __post_init__(self):
        assert self.priority_level in ['low', 'medium', 'high', 'critical']
        assert self.suggested_replay_frequency in ['standard', 'frequent', 'rare']
        assert self.curriculum_stage in ['early', 'mid', 'late', 'advanced']
        assert self.suggested_weight_multiplier >= 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SemanticEnrichmentProposal:
    """Complete semantic enrichment for a datapack episode"""
    proposal_id: str
    timestamp: float

    # Target datapack
    video_id: str
    episode_id: str
    task: str

    # Semantic tags
    fragility_tags: List[FragilityTag]
    risk_tags: List[RiskTag]
    affordance_tags: List[AffordanceTag]
    efficiency_tags: List[EfficiencyTag]
    novelty_tags: List[NoveltyTag]
    intervention_tags: List[InterventionTag]

    # Coherence
    semantic_conflicts: List[SemanticConflict]
    coherence_score: float

    # Orchestrator hints
    supervision_hints: SupervisionHints

    # Metadata
    confidence: float
    source_proposals: List[str]
    justification: str

    # Validation
    validation_status: str
    validation_errors: List[str]

    def __post_init__(self):
        assert 0.0 <= self.coherence_score <= 1.0
        assert 0.0 <= self.confidence <= 1.0
        assert self.validation_status in ['pending', 'passed', 'failed']

    def to_jsonl_enrichment(self) -> dict:
        """Convert to JSONL format for merging with datapacks"""
        return {
            'episode_id': self.episode_id,
            'enrichment': {
                'fragility_tags': [t.to_dict() for t in self.fragility_tags],
                'risk_tags': [t.to_dict() for t in self.risk_tags],
                'affordance_tags': [t.to_dict() for t in self.affordance_tags],
                'efficiency_tags': [t.to_dict() for t in self.efficiency_tags],
                'novelty_tags': [t.to_dict() for t in self.novelty_tags],
                'intervention_tags': [t.to_dict() for t in self.intervention_tags],
                'semantic_conflicts': [c.to_dict() for c in self.semantic_conflicts],
                'coherence_score': self.coherence_score,
                'supervision_hints': self.supervision_hints.to_dict(),
                'confidence': self.confidence,
                'source_proposals': self.source_proposals,
                'validation_status': self.validation_status
            }
        }
```

---

## Step 2: Implement Tag Matchers

### 2.1 Ontology Matcher

**File**: `economics/tag_matchers/ontology_matcher.py`

```python
"""
Matches ontology proposals to datapacks → generates affordance/fragility tags.
"""

from typing import List, Dict, Any
from economics.tag_types import AffordanceTag, FragilityTag


class OntologyMatcher:
    """Generates affordance and fragility tags from ontology proposals"""

    def __init__(self, ontology_proposals: List[Dict[str, Any]]):
        # Filter to validated proposals only
        self.proposals = [p for p in ontology_proposals
                         if p.get('validation_status') == 'passed']

    def match_affordances(self, datapack: Dict[str, Any]) -> List[AffordanceTag]:
        """Generate affordance tags for datapack"""
        tags = []
        objects = datapack.get('metadata', {}).get('objects_present', [])

        for obj in objects:
            # Find relevant proposals
            for prop in self.proposals:
                for affordance in prop.get('new_affordances', []):
                    if affordance['object'] == obj:
                        tags.append(AffordanceTag(
                            affordance_name=affordance['name'],
                            object_name=obj,
                            demonstrated=self._check_demonstrated(datapack, affordance),
                            alternative_affordances=self._find_alternatives(obj, affordance['name']),
                            justification=f"From ontology proposal {prop['proposal_id']}"
                        ))

        # Sort for determinism
        tags.sort(key=lambda t: (t.object_name, t.affordance_name))
        return tags

    def match_fragilities(self, datapack: Dict[str, Any]) -> List[FragilityTag]:
        """Generate fragility tags for datapack"""
        tags = []
        objects = datapack.get('metadata', {}).get('objects_present', [])

        for obj in objects:
            # Find relevant proposals
            for prop in self.proposals:
                for fragility in prop.get('fragility_updates', []):
                    if fragility['object'] == obj:
                        tags.append(FragilityTag(
                            object_name=obj,
                            fragility_level=fragility['level'],
                            damage_cost_usd=fragility['damage_cost'],
                            contact_frames=self._find_contact_frames(datapack, obj),
                            justification=f"From ontology proposal {prop['proposal_id']}"
                        ))

        # Sort for determinism
        tags.sort(key=lambda t: (t.object_name, t.contact_frames[0] if t.contact_frames else 0))
        return tags

    def get_source_proposal_ids(self) -> List[str]:
        """Return IDs of used proposals"""
        return [p['proposal_id'] for p in self.proposals]

    def _check_demonstrated(self, datapack: Dict[str, Any], affordance: dict) -> bool:
        """Check if affordance was demonstrated in episode"""
        # TODO: Implement based on datapack action sequence
        # For now, assume demonstrated if episode succeeded
        return datapack.get('metadata', {}).get('success', False)

    def _find_alternatives(self, obj: str, affordance: str) -> List[str]:
        """Find alternative affordances for object"""
        # TODO: Query ontology for alternatives
        return []

    def _find_contact_frames(self, datapack: Dict[str, Any], obj: str) -> List[int]:
        """Find frames where object was contacted"""
        # TODO: Implement based on datapack action sequence or vision
        # For now, return middle frames as placeholder
        num_frames = len(datapack.get('frames', []))
        if num_frames > 0:
            return [num_frames // 2]
        return [0]
```

### 2.2 Task Graph Matcher

**File**: `economics/tag_matchers/task_graph_matcher.py`

```python
"""
Matches task graph proposals to datapacks → generates risk/efficiency tags.
"""

from typing import List, Dict, Any
from economics.tag_types import RiskTag, EfficiencyTag


class TaskGraphMatcher:
    """Generates risk and efficiency tags from task graph proposals"""

    def __init__(self, task_graph_proposals: List[Dict[str, Any]]):
        # Filter to validated proposals only
        self.proposals = [p for p in task_graph_proposals
                         if p.get('validation_status') == 'passed']

    def match_risks(self, datapack: Dict[str, Any]) -> List[RiskTag]:
        """Generate risk tags for datapack"""
        tags = []
        task = datapack.get('task')

        for prop in self.proposals:
            for risk in prop.get('risk_annotations', []):
                if risk['task'] == task:
                    tags.append(RiskTag(
                        risk_type=risk['risk_type'],
                        severity=risk.get('severity', 'medium'),
                        affected_frames=self._find_affected_frames(datapack, risk),
                        mitigation_hints=risk.get('mitigation_hints', []),
                        justification=f"From task graph proposal {prop['proposal_id']}"
                    ))

        # Sort for determinism
        tags.sort(key=lambda t: (t.affected_frames[0] if t.affected_frames else 0, t.risk_type))
        return tags

    def match_efficiencies(self, datapack: Dict[str, Any]) -> List[EfficiencyTag]:
        """Generate efficiency tags for datapack"""
        tags = []
        task = datapack.get('task')

        for prop in self.proposals:
            for eff in prop.get('efficiency_hints', []):
                if eff['task'] == task:
                    tags.append(EfficiencyTag(
                        metric=eff['metric'],
                        score=self._compute_efficiency_score(datapack, eff),
                        benchmark=eff.get('benchmark', f"average_{task}_{eff['metric']}"),
                        improvement_hints=[eff.get('suggestion', '')],
                        justification=f"From task graph proposal {prop['proposal_id']}"
                    ))

        # Sort for determinism
        tags.sort(key=lambda t: (t.metric, -t.score))
        return tags

    def get_source_proposal_ids(self) -> List[str]:
        """Return IDs of used proposals"""
        return [p['proposal_id'] for p in self.proposals]

    def _find_affected_frames(self, datapack: Dict[str, Any], risk: dict) -> List[int]:
        """Find frames affected by risk"""
        # TODO: Implement based on risk conditions
        # For now, return middle frames
        num_frames = len(datapack.get('frames', []))
        if num_frames > 0:
            return [num_frames // 2]
        return [0]

    def _compute_efficiency_score(self, datapack: Dict[str, Any], eff_hint: dict) -> float:
        """Compute efficiency score for metric"""
        metric = eff_hint['metric']

        if metric == 'time':
            # Compare duration to benchmark
            duration = datapack.get('metadata', {}).get('duration_sec', 10.0)
            benchmark_time = 8.0  # TODO: Get from task graph or config
            score = min(1.0, benchmark_time / duration) if duration > 0 else 0.5
            return score

        # Default
        return 0.5
```

### 2.3 Economics Matcher

**File**: `economics/tag_matchers/economics_matcher.py`

```python
"""
Matches economics outputs to datapacks → generates novelty tags.
"""

from typing import List, Dict, Any, Optional
from economics.tag_types import NoveltyTag


class EconomicsMatcher:
    """Generates novelty tags from economics outputs"""

    def __init__(self, economics_outputs: Dict[str, Dict[str, Any]]):
        self.economics = economics_outputs

    def match_novelty(self, datapack: Dict[str, Any]) -> List[NoveltyTag]:
        """Generate novelty tags for datapack"""
        episode_id = datapack['episode_id']
        econ_data = self.economics.get(episode_id)

        if not econ_data:
            # Missing economics data → hard fail
            raise ValueError(f"Missing economics data for episode {episode_id}")

        tags = []

        novelty_score = econ_data['novelty_score']
        expected_mpl_gain = econ_data['expected_mpl_gain']

        # Determine novelty type based on tier
        tier = econ_data.get('tier', 0)
        novelty_types = {
            0: 'state_coverage',  # Redundant → standard coverage
            1: 'action_diversity',  # Context-novel → diverse actions
            2: 'edge_case'  # Causal-novel → edge case
        }
        novelty_type = novelty_types.get(tier, 'state_coverage')

        tags.append(NoveltyTag(
            novelty_type=novelty_type,
            novelty_score=novelty_score,
            comparison_basis=f"all_{datapack.get('task', 'task')}_episodes",
            expected_mpl_gain=expected_mpl_gain,
            justification=f"Tier {tier} novelty from economics module"
        ))

        return tags

    def validate_economics_present(self, episode_id: str) -> bool:
        """Check if economics data exists for episode"""
        return episode_id in self.economics
```

---

## Step 3: Implement Coherence Checker

**File**: `economics/coherence_checker.py`

```python
"""
Detects semantic conflicts and computes coherence scores.
"""

from typing import List
from economics.tag_types import (
    FragilityTag, RiskTag, AffordanceTag, EfficiencyTag,
    NoveltyTag, InterventionTag, SemanticConflict, SupervisionHints
)


class CoherenceChecker:
    """Detects semantic conflicts between tags"""

    def detect_conflicts(
        self,
        fragility_tags: List[FragilityTag],
        risk_tags: List[RiskTag],
        affordance_tags: List[AffordanceTag],
        efficiency_tags: List[EfficiencyTag],
        novelty_tags: List[NoveltyTag],
        intervention_tags: List[InterventionTag],
        supervision_hints: SupervisionHints
    ) -> List[SemanticConflict]:
        """Detect all semantic conflicts"""
        conflicts = []

        # Conflict 1: High fragility + Low risk
        conflicts.extend(self._check_fragility_risk_mismatch(fragility_tags, risk_tags))

        # Conflict 2: Safety-critical + No risk tags
        conflicts.extend(self._check_safety_risk_missing(supervision_hints, risk_tags))

        # Conflict 3: High efficiency + High novelty (unusual but not wrong)
        conflicts.extend(self._check_efficiency_novelty_tension(efficiency_tags, novelty_tags))

        return conflicts

    def compute_coherence_score(self, conflicts: List[SemanticConflict]) -> float:
        """Compute coherence score (0.0 = high conflict, 1.0 = fully coherent)"""
        if not conflicts:
            return 1.0

        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.5}
        total_penalty = sum(severity_weights.get(c.severity, 0.3) for c in conflicts)

        # Normalize (cap at 1.0 total penalty)
        coherence = max(0.0, 1.0 - min(1.0, total_penalty))
        return coherence

    def _check_fragility_risk_mismatch(
        self,
        fragility_tags: List[FragilityTag],
        risk_tags: List[RiskTag]
    ) -> List[SemanticConflict]:
        """Detect high fragility + low risk contradiction"""
        conflicts = []

        high_fragility_objects = {t.object_name for t in fragility_tags
                                  if t.fragility_level in ['high', 'critical']}
        low_risk_tags = [t for t in risk_tags if t.severity == 'low']

        if high_fragility_objects and low_risk_tags:
            conflicts.append(SemanticConflict(
                conflict_type="risk_fragility_mismatch",
                conflicting_tags=[f"fragility:{obj}" for obj in high_fragility_objects] +
                                [f"risk:{t.risk_type}" for t in low_risk_tags],
                severity="medium",
                resolution_hint="High fragility should correlate with higher risk severity",
                justification="Fragile objects present but risk rated as low"
            ))

        return conflicts

    def _check_safety_risk_missing(
        self,
        supervision_hints: SupervisionHints,
        risk_tags: List[RiskTag]
    ) -> List[SemanticConflict]:
        """Detect safety-critical flag + no risk tags"""
        conflicts = []

        if supervision_hints.safety_critical and not risk_tags:
            conflicts.append(SemanticConflict(
                conflict_type="safety_risk_missing",
                conflicting_tags=["supervision:safety_critical", "risk_tags:empty"],
                severity="medium",
                resolution_hint="Safety-critical episodes should have risk tags",
                justification="Marked safety-critical but no risk tags provided"
            ))

        return conflicts

    def _check_efficiency_novelty_tension(
        self,
        efficiency_tags: List[EfficiencyTag],
        novelty_tags: List[NoveltyTag]
    ) -> List[SemanticConflict]:
        """Detect high efficiency + high novelty (unusual but possible)"""
        conflicts = []

        high_efficiency = any(t.score > 0.9 for t in efficiency_tags)
        high_novelty = any(t.novelty_score > 0.8 for t in novelty_tags)

        if high_efficiency and high_novelty:
            conflicts.append(SemanticConflict(
                conflict_type="efficiency_novelty_tension",
                conflicting_tags=["efficiency:high", "novelty:high"],
                severity="low",
                resolution_hint="High efficiency on novel data is unusual but possible (expert demo)",
                justification="Novel scenario executed with high efficiency"
            ))

        return conflicts
```

---

## Step 4: Implement Supervision Hints Generator

**File**: `economics/supervision_hints_generator.py`

```python
"""
Generates orchestrator supervision hints from tags.
"""

from typing import List
from economics.tag_types import (
    FragilityTag, RiskTag, NoveltyTag, SemanticConflict, SupervisionHints
)


class SupervisionHintsGenerator:
    """Generates orchestrator supervision hints"""

    def generate(
        self,
        task: str,
        fragility_tags: List[FragilityTag],
        risk_tags: List[RiskTag],
        novelty_tags: List[NoveltyTag],
        semantic_conflicts: List[SemanticConflict],
        coherence_score: float
    ) -> SupervisionHints:
        """Generate supervision hints from tags"""

        # Priority level
        priority_level = self._compute_priority_level(novelty_tags, risk_tags, semantic_conflicts)

        # Weight multiplier
        weight_multiplier = self._compute_weight_multiplier(priority_level, novelty_tags)

        # Replay frequency
        replay_frequency = self._compute_replay_frequency(priority_level, novelty_tags)

        # Safety flags
        safety_critical = self._check_safety_critical(fragility_tags, risk_tags)
        requires_human_review = self._check_human_review(semantic_conflicts, safety_critical)

        # Curriculum stage
        curriculum_stage = self._assign_curriculum_stage(task, fragility_tags, risk_tags)

        # Prerequisites
        prerequisite_tags = self._identify_prerequisites(task, curriculum_stage, fragility_tags)

        # Justification
        justification = self._generate_justification(
            priority_level, novelty_tags, safety_critical, curriculum_stage
        )

        return SupervisionHints(
            prioritize_for_training=(priority_level in ['high', 'critical']),
            priority_level=priority_level,
            suggested_weight_multiplier=weight_multiplier,
            suggested_replay_frequency=replay_frequency,
            requires_human_review=requires_human_review,
            safety_critical=safety_critical,
            curriculum_stage=curriculum_stage,
            prerequisite_tags=prerequisite_tags,
            justification=justification
        )

    def _compute_priority_level(
        self,
        novelty_tags: List[NoveltyTag],
        risk_tags: List[RiskTag],
        conflicts: List[SemanticConflict]
    ) -> str:
        """Compute priority level from tags"""
        # High priority if:
        # - High novelty (edge case, failure mode)
        # - High risk (critical severity)
        # - Semantic conflicts present

        max_novelty = max([t.novelty_score for t in novelty_tags], default=0.0)
        has_high_risk = any(t.severity in ['high', 'critical'] for t in risk_tags)
        has_conflicts = len(conflicts) > 0

        if max_novelty > 0.7 or has_high_risk:
            return 'high'
        elif max_novelty > 0.4 or has_conflicts:
            return 'medium'
        else:
            return 'low'

    def _compute_weight_multiplier(self, priority_level: str, novelty_tags: List[NoveltyTag]) -> float:
        """Compute weight multiplier for training"""
        base_weights = {
            'low': 1.0,
            'medium': 1.5,
            'high': 2.0,
            'critical': 3.0
        }
        return base_weights[priority_level]

    def _compute_replay_frequency(self, priority_level: str, novelty_tags: List[NoveltyTag]) -> str:
        """Compute replay frequency"""
        if priority_level in ['high', 'critical']:
            return 'frequent'
        elif priority_level == 'low':
            return 'rare'
        else:
            return 'standard'

    def _check_safety_critical(self, fragility_tags: List[FragilityTag], risk_tags: List[RiskTag]) -> bool:
        """Check if episode is safety-critical"""
        has_high_fragility = any(t.fragility_level in ['high', 'critical'] for t in fragility_tags)
        has_high_risk = any(t.severity in ['high', 'critical'] for t in risk_tags)
        return has_high_fragility or has_high_risk

    def _check_human_review(self, conflicts: List[SemanticConflict], safety_critical: bool) -> bool:
        """Check if human review required"""
        has_high_conflict = any(c.severity == 'high' for c in conflicts)
        return has_high_conflict or (safety_critical and len(conflicts) > 0)

    def _assign_curriculum_stage(
        self,
        task: str,
        fragility_tags: List[FragilityTag],
        risk_tags: List[RiskTag]
    ) -> str:
        """Assign curriculum stage based on complexity"""
        # Advanced if fragile objects or high risks present
        has_fragility = len(fragility_tags) > 0
        has_high_risk = any(t.severity in ['high', 'critical'] for t in risk_tags)

        if has_fragility or has_high_risk:
            return 'advanced'
        elif len(risk_tags) > 0:
            return 'mid'
        else:
            return 'early'

    def _identify_prerequisites(
        self,
        task: str,
        curriculum_stage: str,
        fragility_tags: List[FragilityTag]
    ) -> List[str]:
        """Identify prerequisite tags for curriculum"""
        prereqs = []

        if curriculum_stage == 'advanced':
            # Require basic task mastery
            prereqs.append(f"basic_{task}")

            # If fragile objects present, require fragility awareness
            if fragility_tags:
                prereqs.append("fragile_object_awareness")

        return prereqs

    def _generate_justification(
        self,
        priority_level: str,
        novelty_tags: List[NoveltyTag],
        safety_critical: bool,
        curriculum_stage: str
    ) -> str:
        """Generate human-readable justification"""
        reasons = []

        if priority_level in ['high', 'critical']:
            reasons.append(f"{priority_level} priority")

        if novelty_tags and novelty_tags[0].novelty_score > 0.7:
            reasons.append(f"{novelty_tags[0].novelty_type} novelty")

        if safety_critical:
            reasons.append("safety-critical scenario")

        if curriculum_stage == 'advanced':
            reasons.append("advanced curriculum stage")

        return " + ".join(reasons) if reasons else "standard training episode"
```

---

## Step 5: Implement Main Propagator

**File**: `economics/semantic_tag_propagator.py`

```python
"""
SemanticTagPropagator: Enriches datapacks with semantic tags.
"""

import hashlib
import json
from typing import List, Dict, Any
from economics.tag_types import SemanticEnrichmentProposal, InterventionTag
from economics.tag_matchers.ontology_matcher import OntologyMatcher
from economics.tag_matchers.task_graph_matcher import TaskGraphMatcher
from economics.tag_matchers.economics_matcher import EconomicsMatcher
from economics.coherence_checker import CoherenceChecker
from economics.supervision_hints_generator import SupervisionHintsGenerator


class SemanticTagPropagator:
    """Generates semantic enrichments for datapacks"""

    def __init__(self):
        self.coherence_checker = CoherenceChecker()
        self.hints_generator = SupervisionHintsGenerator()

    def generate_proposals(
        self,
        datapacks: List[Dict[str, Any]],
        ontology_proposals: List[Dict[str, Any]],
        task_graph_proposals: List[Dict[str, Any]],
        economics_outputs: Dict[str, Dict[str, Any]]
    ) -> List[SemanticEnrichmentProposal]:
        """Generate semantic enrichment proposals for all datapacks"""

        # Sort datapacks for determinism
        sorted_datapacks = sorted(datapacks, key=lambda d: d['episode_id'])

        # Initialize matchers
        ontology_matcher = OntologyMatcher(ontology_proposals)
        task_graph_matcher = TaskGraphMatcher(task_graph_proposals)
        economics_matcher = EconomicsMatcher(economics_outputs)

        proposals = []
        for datapack in sorted_datapacks:
            proposal = self.generate_single_proposal(
                datapack,
                ontology_matcher,
                task_graph_matcher,
                economics_matcher
            )
            proposals.append(proposal)

        return proposals

    def generate_single_proposal(
        self,
        datapack: Dict[str, Any],
        ontology_matcher: OntologyMatcher,
        task_graph_matcher: TaskGraphMatcher,
        economics_matcher: EconomicsMatcher
    ) -> SemanticEnrichmentProposal:
        """Generate enrichment proposal for single datapack"""

        episode_id = datapack['episode_id']
        video_id = datapack.get('video_id', 'unknown')
        task = datapack.get('task', 'unknown')

        # Validate economics present (required)
        if not economics_matcher.validate_economics_present(episode_id):
            return self._create_failed_proposal(
                episode_id, video_id, task,
                error=f"Missing economics data for episode {episode_id}"
            )

        # Generate tags from each source
        try:
            fragility_tags = ontology_matcher.match_fragilities(datapack)
            affordance_tags = ontology_matcher.match_affordances(datapack)
            risk_tags = task_graph_matcher.match_risks(datapack)
            efficiency_tags = task_graph_matcher.match_efficiencies(datapack)
            novelty_tags = economics_matcher.match_novelty(datapack)
            intervention_tags = self._detect_interventions(datapack)

            # Sort tags for determinism
            fragility_tags.sort(key=lambda t: (t.object_name, t.contact_frames[0] if t.contact_frames else 0))
            risk_tags.sort(key=lambda t: (t.affected_frames[0] if t.affected_frames else 0, t.risk_type))
            affordance_tags.sort(key=lambda t: (t.object_name, t.affordance_name))
            efficiency_tags.sort(key=lambda t: (t.metric, -t.score))

        except Exception as e:
            return self._create_failed_proposal(
                episode_id, video_id, task,
                error=f"Tag generation failed: {str(e)}"
            )

        # Generate supervision hints (preliminary, before coherence check)
        supervision_hints = self.hints_generator.generate(
            task, fragility_tags, risk_tags, novelty_tags, [], 1.0  # No conflicts yet
        )

        # Check coherence
        conflicts = self.coherence_checker.detect_conflicts(
            fragility_tags, risk_tags, affordance_tags, efficiency_tags,
            novelty_tags, intervention_tags, supervision_hints
        )
        coherence_score = self.coherence_checker.compute_coherence_score(conflicts)

        # Regenerate supervision hints with conflicts
        supervision_hints = self.hints_generator.generate(
            task, fragility_tags, risk_tags, novelty_tags, conflicts, coherence_score
        )

        # Source proposal IDs
        source_proposals = (
            ontology_matcher.get_source_proposal_ids() +
            task_graph_matcher.get_source_proposal_ids()
        )

        # Compute confidence
        confidence = self._compute_confidence(
            ontology_matcher, task_graph_matcher, coherence_score, datapack
        )

        # Generate deterministic proposal ID
        proposal_id = self._generate_proposal_id(datapack, source_proposals)

        # Validation
        validation_errors = self._validate_proposal(
            fragility_tags, risk_tags, affordance_tags, efficiency_tags,
            novelty_tags, intervention_tags, conflicts, supervision_hints
        )
        validation_status = "passed" if not validation_errors else "failed"

        # Justification
        justification = self._generate_justification(
            task, fragility_tags, novelty_tags, supervision_hints
        )

        return SemanticEnrichmentProposal(
            proposal_id=proposal_id,
            timestamp=datapack.get('metadata', {}).get('timestamp', 0.0),
            video_id=video_id,
            episode_id=episode_id,
            task=task,
            fragility_tags=fragility_tags,
            risk_tags=risk_tags,
            affordance_tags=affordance_tags,
            efficiency_tags=efficiency_tags,
            novelty_tags=novelty_tags,
            intervention_tags=intervention_tags,
            semantic_conflicts=conflicts,
            coherence_score=coherence_score,
            supervision_hints=supervision_hints,
            confidence=confidence,
            source_proposals=source_proposals,
            justification=justification,
            validation_status=validation_status,
            validation_errors=validation_errors
        )

    def _detect_interventions(self, datapack: Dict[str, Any]) -> List[InterventionTag]:
        """Detect human interventions in episode"""
        tags = []

        # Check metadata for intervention flag
        metadata = datapack.get('metadata', {})
        if metadata.get('human_intervention'):
            tags.append(InterventionTag(
                intervention_type='human_correction',
                frame_range=(0, len(datapack.get('frames', []))),
                trigger=metadata.get('intervention_trigger', 'unknown'),
                learning_opportunity='Human demonstrated correct approach after failure',
                justification='Human intervention flag present in metadata'
            ))

        return tags

    def _compute_confidence(
        self,
        ontology_matcher: OntologyMatcher,
        task_graph_matcher: TaskGraphMatcher,
        coherence_score: float,
        datapack: Dict[str, Any]
    ) -> float:
        """Compute overall confidence in enrichment"""
        base_confidence = 0.8

        # Reduce if missing ontology
        if not ontology_matcher.proposals:
            base_confidence -= 0.2

        # Reduce if missing task graph
        if not task_graph_matcher.proposals:
            base_confidence -= 0.1

        # Reduce if partial datapack
        if not datapack.get('metadata', {}).get('objects_present'):
            base_confidence -= 0.1

        # Factor in coherence
        confidence = base_confidence * coherence_score

        return max(0.0, min(1.0, confidence))

    def _generate_proposal_id(self, datapack: Dict[str, Any], source_proposals: List[str]) -> str:
        """Generate deterministic proposal ID from inputs"""
        # Hash episode_id + source_proposals
        hash_input = f"{datapack['episode_id']}_{'_'.join(sorted(source_proposals))}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"enrich_{datapack['episode_id']}_{hash_digest}"

    def _validate_proposal(self, *args) -> List[str]:
        """Validate proposal against schema"""
        errors = []
        # TODO: Implement schema validation
        # For now, basic checks only
        return errors

    def _generate_justification(
        self,
        task: str,
        fragility_tags: List,
        novelty_tags: List,
        hints: Any
    ) -> str:
        """Generate human-readable justification"""
        parts = [f"Task: {task}"]

        if fragility_tags:
            parts.append(f"{len(fragility_tags)} fragile object(s)")

        if novelty_tags and novelty_tags[0].novelty_score > 0.5:
            parts.append(f"{novelty_tags[0].novelty_type} scenario")

        return ", ".join(parts)

    def _create_failed_proposal(
        self,
        episode_id: str,
        video_id: str,
        task: str,
        error: str
    ) -> SemanticEnrichmentProposal:
        """Create failed proposal for invalid inputs"""
        from economics.tag_types import SupervisionHints

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
            supervision_hints=SupervisionHints(
                prioritize_for_training=False,
                priority_level='low',
                suggested_weight_multiplier=1.0,
                suggested_replay_frequency='standard',
                requires_human_review=False,
                safety_critical=False,
                curriculum_stage='early',
                prerequisite_tags=[],
                justification='Validation failed'
            ),
            confidence=0.0,
            source_proposals=[],
            justification=error,
            validation_status='failed',
            validation_errors=[error]
        )
```

---

## Step 6: Create Test Fixtures

**File**: `tests/fixtures/test_datapacks.jsonl`

```jsonl
{"video_id": "drawer_001", "episode_id": "ep_001", "task": "open_drawer", "frames": [0, 1, 2], "actions": [], "metadata": {"success": true, "objects_present": ["drawer", "handle"], "timestamp": 1700000000.0}}
{"video_id": "drawer_002", "episode_id": "ep_002", "task": "open_drawer", "frames": [0, 1, 2], "actions": [], "metadata": {"success": true, "objects_present": ["drawer", "handle", "vase_inside"], "timestamp": 1700000001.0}}
```

*(Continue with other fixtures as defined in SMOKE_TESTS.md)*

---

## Step 7: Run Implementation

```bash
# 1. Implement all files above

# 2. Run smoke tests
python -m pytest tests/smoke_tests/test_semantic_tag_propagator.py -v

# 3. Run standalone smoke test
python scripts/smoke_test_tag_propagator.py

# 4. Generate sample enrichments
python -m economics.semantic_tag_propagator \
  --datapacks tests/fixtures/test_datapacks.jsonl \
  --ontology tests/fixtures/test_ontology_proposals.json \
  --task-graph tests/fixtures/test_task_graph_proposals.json \
  --economics tests/fixtures/test_economics_outputs.json \
  --output enrichments.jsonl

# 5. Verify JSONL output
cat enrichments.jsonl | jq .
```

---

## Step 8: Integration Commands

After implementation passes smoke tests:

```bash
# Generate enrichments for real datapacks
python -m economics.semantic_tag_propagator \
  --datapacks data/datapacks/drawer_datapacks.jsonl \
  --ontology logs/ontology_proposals_validated.json \
  --task-graph logs/task_graph_proposals_validated.json \
  --economics logs/economics_outputs.json \
  --output data/enrichments/drawer_enrichments.jsonl

# Merge enrichments with datapacks (Stage 3 orchestrator)
python -m orchestrator.merge_enrichments \
  --datapacks data/datapacks/drawer_datapacks.jsonl \
  --enrichments data/enrichments/drawer_enrichments.jsonl \
  --output data/merged/drawer_merged.jsonl

# Verify merged format
head -1 data/merged/drawer_merged.jsonl | jq .
```

---

## Expected Directory Structure After Implementation

```
economics/
├── semantic_tag_propagator.py        ✅ Main implementation
├── tag_types.py                      ✅ Dataclasses
├── coherence_checker.py              ✅ Conflict detection
├── supervision_hints_generator.py    ✅ Hint generation
└── tag_matchers/
    ├── __init__.py
    ├── ontology_matcher.py           ✅ Ontology → tags
    ├── task_graph_matcher.py         ✅ Task graph → tags
    └── economics_matcher.py          ✅ Economics → tags

tests/
├── fixtures/
│   ├── test_datapacks.jsonl
│   ├── test_ontology_proposals.json
│   ├── test_task_graph_proposals.json
│   └── test_economics_outputs.json
└── smoke_tests/
    └── test_semantic_tag_propagator.py  ✅ Full test suite

scripts/
└── smoke_test_tag_propagator.py      ✅ Standalone runner
```

---

**End of Codex Implementation Guide**
