# Stage 2.2: OntologyUpdateEngine — Full Specification

**Status**: Ready for Codex implementation
**Date**: 2025-11-17
**Prerequisites**: Stage 2.1 (SemanticPrimitiveExtractor) complete

---

## 1. Overview

### Purpose
The **OntologyUpdateEngine** is a Stage 2 module that:
1. Consumes `SemanticPrimitive` objects from Stage 2.1
2. Produces `OntologyUpdateProposal` objects (advisory-only, no mutations)
3. Respects constraints from EconomicController, DatapackEngine, TaskGraph, and SemanticOrchestrator
4. Provides schema-driven, JSON-safe proposals for downstream consumption

### Key Design Principles
- **Advisory-only**: Does NOT mutate ontology directly
- **Schema-driven**: All proposals use typed dataclasses
- **JSON-safe**: All outputs serializable for storage/logging
- **Constraint-respecting**: Cannot originate econ/data-valuation logic
- **Deterministic**: Given same inputs, produces same outputs (for testing)

---

## 2. Constraint Sources (Dependency Mapping)

### Upstream Dependencies (Constraints FROM)

#### 2.1 EconomicController
**File**: `src/orchestrator/economic_controller.py`

**Constraints**:
- Cannot propose changes to reward math
- Cannot propose changes to wage parity logic
- Cannot propose changes to MPL/error/energy attribution
- Cannot set `price_per_unit`, `damage_cost`, `energy_price_kWh`

**Allowed Consumption**:
- `EconSignals.error_urgency` → May elevate fragility/risk awareness
- `EconSignals.energy_urgency` → May suggest energy-efficient affordance priorities
- `EconSignals.damage_cost_total` → May influence fragility thresholds

#### 2.2 DatapackEngine
**File**: `src/orchestrator/datapack_engine.py`

**Constraints**:
- Cannot propose changes to data valuation logic
- Cannot propose changes to tier classification
- Cannot propose changes to novelty scoring

**Allowed Consumption**:
- `DatapackSignals.tier2_fraction` → May suggest focus on frontier skills
- `DatapackSignals.data_coverage_score` → May propose new object categories if gaps detected
- `DatapackSignals.semantic_tag_diversity` → May unify/standardize tags

#### 2.3 TaskGraph
**File**: `src/orchestrator/task_graph.py`

**Constraints**:
- Cannot delete existing task nodes
- Cannot modify task dependencies directly
- Cannot change task execution order

**Allowed Consumption**:
- `TaskNode.affordances` → May propose new affordances for discovered skills
- `TaskNode.objects_involved` → May propose new object relationships
- `TaskNode.metadata['semantic_priority']` → May suggest priority-aware skill gating

#### 2.4 SemanticOrchestrator
**File**: `src/orchestrator/semantic_orchestrator.py`

**Constraints**:
- Cannot bypass semantic orchestrator to mutate ontology
- Cannot directly set cross-module constraints

**Allowed Consumption**:
- `SemanticUpdatePlan.ontology_changes` → May align proposals with planned changes
- `SemanticOrchestrator._current_constraints` → May respect existing cross-module constraints

### Downstream Dependencies (Proposals TO)

#### 2.5 SemanticOrchestratorV2
**Consumption Interface** (to be added in Stage 2.3):
- `apply_ontology_proposals(List[OntologyUpdateProposal])` → Advisory application
- `validate_proposals(List[OntologyUpdateProposal])` → Validate before application
- `merge_proposals(List[OntologyUpdateProposal])` → Conflict resolution

#### 2.6 SIMA-2 Integration
**Consumption Path**:
- Proposals with `source="sima2"` feed back to SIMA primitive vocabulary
- Skill gating proposals filter SIMA rollout selection

#### 2.7 VLA / Diffusion / RL
**Consumption Path** (via SemanticOrchestrator):
- Affordance proposals → VLA action space constraints
- Fragility proposals → Diffusion safety margins
- Energy heuristics → RL exploration bias

---

## 3. OntologyUpdateProposal Schema

### 3.1 Core Proposal Dataclass

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from enum import Enum


class ProposalType(Enum):
    """Types of ontology update proposals."""
    ADD_AFFORDANCE = "add_affordance"
    ADJUST_RISK = "adjust_risk"
    INFER_FRAGILITY = "infer_fragility"
    ADD_OBJECT_CATEGORY = "add_object_category"
    ADD_SEMANTIC_TAG = "add_semantic_tag"
    ADD_SKILL_GATE = "add_skill_gate"
    ADD_SAFETY_CONSTRAINT = "add_safety_constraint"
    ADD_ENERGY_HEURISTIC = "add_energy_heuristic"
    UPDATE_OBJECT_RELATIONSHIP = "update_object_relationship"


class ProposalPriority(Enum):
    """Priority levels for proposal application."""
    CRITICAL = "critical"  # Safety-critical (fragility, collision risk)
    HIGH = "high"  # Economic urgency-driven
    MEDIUM = "medium"  # Quality-of-life improvements
    LOW = "low"  # Nice-to-have, no urgency


@dataclass
class OntologyUpdateProposal:
    """
    Schema for a single ontology update proposal.

    Advisory-only; does not mutate ontology directly.
    SemanticOrchestrator decides whether/how to apply.
    """
    # Identification
    proposal_id: str  # Unique ID (e.g., "prop_123abc")
    proposal_type: ProposalType
    priority: ProposalPriority = ProposalPriority.MEDIUM

    # Source tracking
    source_primitive_id: str = ""  # Which SemanticPrimitive triggered this
    source: str = "sima2"  # "sima2", "rule_based", "heuristic"

    # Target specification
    target_object_id: Optional[str] = None  # If affecting specific object
    target_skill_id: Optional[int] = None  # If affecting specific skill
    target_affordance_type: Optional[str] = None  # If affecting affordance

    # Proposal content (type-specific)
    proposed_changes: Dict[str, Any] = field(default_factory=dict)
    # Examples:
    #   ADD_AFFORDANCE: {"affordance_type": "graspable", "confidence": 0.8, ...}
    #   ADJUST_RISK: {"old_risk": 0.1, "new_risk": 0.6, "reason": "fragile nearby"}
    #   ADD_SKILL_GATE: {"gated_skill_id": 2, "precondition": "fragility_check"}

    # Justification
    rationale: str = ""  # Human-readable explanation
    confidence: float = 1.0  # Confidence in proposal (0-1)

    # Constraint compliance
    respects_econ_constraints: bool = True
    respects_datapack_constraints: bool = True
    respects_task_graph: bool = True

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.value,
            "priority": self.priority.value,
            "source_primitive_id": self.source_primitive_id,
            "source": self.source,
            "target_object_id": self.target_object_id,
            "target_skill_id": self.target_skill_id,
            "target_affordance_type": self.target_affordance_type,
            "proposed_changes": self.proposed_changes,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "respects_econ_constraints": self.respects_econ_constraints,
            "respects_datapack_constraints": self.respects_datapack_constraints,
            "respects_task_graph": self.respects_task_graph,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OntologyUpdateProposal":
        """Create from dictionary."""
        return cls(
            proposal_id=d["proposal_id"],
            proposal_type=ProposalType(d["proposal_type"]),
            priority=ProposalPriority(d.get("priority", "medium")),
            source_primitive_id=d.get("source_primitive_id", ""),
            source=d.get("source", "sima2"),
            target_object_id=d.get("target_object_id"),
            target_skill_id=d.get("target_skill_id"),
            target_affordance_type=d.get("target_affordance_type"),
            proposed_changes=d.get("proposed_changes", {}),
            rationale=d.get("rationale", ""),
            confidence=d.get("confidence", 1.0),
            respects_econ_constraints=d.get("respects_econ_constraints", True),
            respects_datapack_constraints=d.get("respects_datapack_constraints", True),
            respects_task_graph=d.get("respects_task_graph", True),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )
```

### 3.2 Proposal Type Specifications

#### ADD_AFFORDANCE
```python
# proposed_changes format:
{
    "affordance_type": "graspable",  # AffordanceType enum value
    "confidence": 0.85,
    "constraints": {"max_force": 10.0},
    "activation_skill_ids": [1],  # Which skills activate this
    "energy_cost_estimate": 0.02,
    "risk_level": 0.7,
}
```

#### ADJUST_RISK
```python
# proposed_changes format:
{
    "old_risk_level": 0.1,  # Current risk
    "new_risk_level": 0.6,  # Proposed risk
    "adjustment_factor": 6.0,  # Multiplier
    "trigger": "fragile_nearby",  # Why risk increased
}
```

#### INFER_FRAGILITY
```python
# proposed_changes format:
{
    "object_id": "vase_01",
    "inferred_fragility": 0.9,  # 0-1 scale
    "evidence": ["collision_sensitive", "high_damage_cost"],
    "damage_cost_estimate": 100.0,  # USD
}
```

#### ADD_OBJECT_CATEGORY
```python
# proposed_changes format:
{
    "category": "fragile",  # ObjectCategory enum value
    "objects": ["vase_01", "glass_plate"],
    "category_properties": {
        "requires_gentle_grasp": True,
        "max_grip_force": 5.0,
    },
}
```

#### ADD_SEMANTIC_TAG
```python
# proposed_changes format:
{
    "tag": "fragile_glassware",
    "applies_to_objects": ["vase_01"],
    "applies_to_skills": [1, 2, 6],  # GRASP, PULL, PLACE
    "propagate_to_subtasks": True,
}
```

#### ADD_SKILL_GATE
```python
# proposed_changes format:
{
    "gated_skill_id": 2,  # PULL skill
    "preconditions": ["fragility_check_passed", "vase_position_known"],
    "safety_threshold": 0.8,  # Minimum safety score to proceed
    "fallback_skill_id": 0,  # APPROACH (if gate fails)
}
```

#### ADD_SAFETY_CONSTRAINT
```python
# proposed_changes format:
{
    "constraint_type": "collision_avoidance",
    "priority": "critical",
    "objects": ["vase_01"],
    "min_clearance_m": 0.05,
    "applies_to_skills": [2, 5],  # PULL, MOVE
}
```

#### ADD_ENERGY_HEURISTIC
```python
# proposed_changes format:
{
    "heuristic_type": "prefer_efficient_path",
    "skill_id": 5,  # MOVE skill
    "energy_multiplier": 0.8,  # Reduce energy cost by 20%
    "conditions": ["short_reach", "low_load"],
}
```

#### UPDATE_OBJECT_RELATIONSHIP
```python
# proposed_changes format:
{
    "object_a": "vase_01",
    "object_b": "table_01",
    "relationship_type": "supported_by",
    "confidence": 0.95,
}
```

---

## 4. OntologyUpdateEngine Module

### 4.1 File Structure
```
src/sima2/
├── semantic_primitive_extractor.py  # Stage 2.1 (existing)
└── ontology_update_engine.py        # Stage 2.2 (NEW)
```

### 4.2 Module Implementation

```python
"""
OntologyUpdateEngine: Advisory ontology proposal generator.

IMPORTANT:
- Does NOT mutate ontology directly
- Consumes SemanticPrimitives from Stage 2.1
- Produces OntologyUpdateProposals for SemanticOrchestrator
- Respects econ/datapack/task-graph constraints
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.sima2.ontology_proposals import (
    OntologyUpdateProposal,
    ProposalType,
    ProposalPriority,
)
from src.orchestrator.ontology import (
    EnvironmentOntology,
    AffordanceType,
    ObjectCategory,
)
from src.orchestrator.task_graph import TaskGraph
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals


class OntologyUpdateEngine:
    """
    Generates ontology update proposals from semantic primitives.

    Advisory-only; does not mutate ontology state.
    """

    def __init__(
        self,
        ontology: EnvironmentOntology,
        task_graph: Optional[TaskGraph] = None,
        econ_signals: Optional[EconSignals] = None,
        datapack_signals: Optional[DatapackSignals] = None,
    ):
        """
        Initialize OntologyUpdateEngine.

        Args:
            ontology: Current ontology (read-only)
            task_graph: Current task graph (read-only)
            econ_signals: Economic constraints (optional)
            datapack_signals: Data constraints (optional)
        """
        self.ontology = ontology
        self.task_graph = task_graph
        self.econ_signals = econ_signals or EconSignals()
        self.datapack_signals = datapack_signals or DatapackSignals()

        # Proposal generation state (for determinism)
        self._proposal_counter = 0

    def generate_proposals(
        self, primitives: List[SemanticPrimitive]
    ) -> List[OntologyUpdateProposal]:
        """
        Generate ontology update proposals from semantic primitives.

        Args:
            primitives: List of SemanticPrimitives from Stage 2.1

        Returns:
            List of OntologyUpdateProposals (advisory-only)
        """
        proposals = []

        for prim in primitives:
            # 1. Affordance proposals
            proposals.extend(self._propose_affordances(prim))

            # 2. Risk adjustment proposals
            proposals.extend(self._propose_risk_adjustments(prim))

            # 3. Fragility inference proposals
            proposals.extend(self._propose_fragility_inference(prim))

            # 4. Skill gating proposals
            proposals.extend(self._propose_skill_gates(prim))

            # 5. Energy heuristic proposals
            proposals.extend(self._propose_energy_heuristics(prim))

            # 6. Semantic tag proposals
            proposals.extend(self._propose_semantic_tags(prim))

        return proposals

    def _make_proposal_id(self) -> str:
        """Generate unique proposal ID."""
        self._proposal_counter += 1
        return f"prop_{self._proposal_counter:06d}_{uuid.uuid4().hex[:6]}"

    def _propose_affordances(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose new affordances based on primitive actions.

        Rules:
        - "grasp" tags → graspable affordance
        - "lift" tags → liftable affordance
        - "pull" tags → pullable affordance
        - etc.
        """
        proposals = []

        # Map tags to affordances
        tag_to_affordance = {
            "grasp": AffordanceType.GRASPABLE,
            "lift": AffordanceType.LIFTABLE,
            "pull": AffordanceType.PULLABLE,
            "push": AffordanceType.PUSHABLE,
            "place": AffordanceType.PLACEABLE,
            "open": AffordanceType.OPENABLE,
        }

        for tag, aff_type in tag_to_affordance.items():
            if tag in prim.tags:
                # Check if already exists in ontology
                # (In production, query ontology for object; here we use heuristics)

                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.ADD_AFFORDANCE,
                    priority=ProposalPriority.MEDIUM,
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    target_affordance_type=aff_type.value,
                    proposed_changes={
                        "affordance_type": aff_type.value,
                        "confidence": prim.success_rate,
                        "activation_skill_ids": [],  # Inferred from task_graph
                        "energy_cost_estimate": prim.energy_intensity,
                        "risk_level": self._risk_level_to_float(prim.risk_level),
                    },
                    rationale=f"Primitive '{prim.primitive_id}' demonstrated '{tag}' action",
                    confidence=prim.success_rate,
                    tags=prim.tags,
                )

                proposals.append(proposal)

        return proposals

    def _propose_risk_adjustments(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose risk level adjustments based on primitive risk.

        Rules:
        - High risk primitives → elevate affordance risk
        - Econ error_urgency > 0.5 → further risk elevation
        """
        proposals = []

        if prim.risk_level in {"medium", "high"}:
            # Get risk multiplier from econ signals
            risk_multiplier = 1.0
            if self.econ_signals.error_urgency > 0.5:
                risk_multiplier = 1.5
            if self.econ_signals.error_urgency > 0.7:
                risk_multiplier = 2.0

            old_risk = self._risk_level_to_float(prim.risk_level)
            new_risk = min(1.0, old_risk * risk_multiplier)

            if new_risk > old_risk:
                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.ADJUST_RISK,
                    priority=(
                        ProposalPriority.CRITICAL
                        if self.econ_signals.error_urgency > 0.7
                        else ProposalPriority.HIGH
                    ),
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    proposed_changes={
                        "old_risk_level": old_risk,
                        "new_risk_level": new_risk,
                        "adjustment_factor": risk_multiplier,
                        "trigger": f"error_urgency={self.econ_signals.error_urgency:.2f}",
                    },
                    rationale=(
                        f"Primitive risk '{prim.risk_level}' + "
                        f"error urgency {self.econ_signals.error_urgency:.2f} "
                        f"→ elevate risk to {new_risk:.2f}"
                    ),
                    confidence=0.8,
                    tags=prim.tags + ["risk_adjustment"],
                )

                proposals.append(proposal)

        return proposals

    def _propose_fragility_inference(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Infer object fragility from primitive tags.

        Rules:
        - "fragile" tag → high fragility (0.9)
        - "glass" tag → high fragility (0.85)
        - High damage cost from econ → medium fragility (0.6)
        """
        proposals = []

        fragility_rules = {
            "fragile": 0.9,
            "glass": 0.85,
            "delicate": 0.8,
            "vase": 0.7,
        }

        for tag, fragility in fragility_rules.items():
            if tag in prim.tags:
                # Adjust based on econ damage costs
                if self.econ_signals.damage_cost_total > 50.0:
                    fragility = min(1.0, fragility * 1.1)

                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.INFER_FRAGILITY,
                    priority=ProposalPriority.CRITICAL,
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    proposed_changes={
                        "inferred_fragility": fragility,
                        "evidence": [tag, f"task_type={prim.task_type}"],
                        "damage_cost_estimate": self.econ_signals.damage_cost_per_error,
                    },
                    rationale=f"Tag '{tag}' indicates fragility {fragility:.2f}",
                    confidence=0.85,
                    tags=prim.tags + ["fragility_inference"],
                )

                proposals.append(proposal)

        return proposals

    def _propose_skill_gates(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose skill gating based on risk/safety.

        Rules:
        - High-risk primitives → gate with safety check
        - "fragile" tags → gate pull/move/lift skills
        """
        proposals = []

        # If primitive is high-risk OR involves fragile objects
        if prim.risk_level == "high" or "fragile" in prim.tags:
            # Gate high-energy or high-force skills
            gated_skill_ids = [2, 5, 7]  # PULL, MOVE, PUSH (heuristic)

            for skill_id in gated_skill_ids:
                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.ADD_SKILL_GATE,
                    priority=ProposalPriority.HIGH,
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    target_skill_id=skill_id,
                    proposed_changes={
                        "gated_skill_id": skill_id,
                        "preconditions": [
                            "fragility_check_passed",
                            "collision_avoidance_active",
                        ],
                        "safety_threshold": 0.8,
                        "fallback_skill_id": 0,  # APPROACH
                    },
                    rationale=(
                        f"Skill {skill_id} involves high-risk action "
                        f"near fragile object (risk={prim.risk_level})"
                    ),
                    confidence=0.9,
                    tags=prim.tags + ["skill_gating", "safety"],
                )

                proposals.append(proposal)

        return proposals

    def _propose_energy_heuristics(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose energy efficiency heuristics.

        Rules:
        - Low energy_intensity primitives → prefer_efficient_path heuristic
        - High energy_urgency → boost energy-saving heuristics
        """
        proposals = []

        # If primitive is energy-efficient AND energy urgency is high
        if (
            prim.energy_intensity < 0.5
            and self.econ_signals.energy_urgency > 0.3
        ):
            proposal = OntologyUpdateProposal(
                proposal_id=self._make_proposal_id(),
                proposal_type=ProposalType.ADD_ENERGY_HEURISTIC,
                priority=ProposalPriority.MEDIUM,
                source_primitive_id=prim.primitive_id,
                source=prim.source,
                proposed_changes={
                    "heuristic_type": "prefer_efficient_path",
                    "energy_multiplier": 0.8,  # 20% energy reduction
                    "conditions": [
                        "short_reach",
                        f"energy_intensity<{prim.energy_intensity}",
                    ],
                },
                rationale=(
                    f"Primitive demonstrates low energy use ({prim.energy_intensity:.2f}); "
                    f"energy urgency={self.econ_signals.energy_urgency:.2f}"
                ),
                confidence=0.7,
                tags=prim.tags + ["energy_optimization"],
            )

            proposals.append(proposal)

        return proposals

    def _propose_semantic_tags(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose semantic tag additions for consistency.

        Rules:
        - Unify tags across primitives (e.g., "fragile" vs "delicate")
        - Propagate safety tags to related skills
        """
        proposals = []

        # Tag normalization rules
        tag_unification = {
            ("fragile", "delicate"): "fragile_glassware",
            ("drawer", "open"): "drawer_manipulation",
        }

        for tag_set, unified_tag in tag_unification.items():
            if all(t in prim.tags for t in tag_set):
                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.ADD_SEMANTIC_TAG,
                    priority=ProposalPriority.LOW,
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    proposed_changes={
                        "tag": unified_tag,
                        "applies_to_objects": [],  # Inferred from ontology
                        "applies_to_skills": [],  # Inferred from task_graph
                        "propagate_to_subtasks": True,
                    },
                    rationale=f"Unify tags {tag_set} → '{unified_tag}'",
                    confidence=0.6,
                    tags=[unified_tag],
                )

                proposals.append(proposal)

        return proposals

    @staticmethod
    def _risk_level_to_float(risk_level: str) -> float:
        """Convert categorical risk to float."""
        mapping = {"low": 0.1, "medium": 0.5, "high": 0.9}
        return mapping.get(risk_level, 0.5)

    def validate_proposals(
        self, proposals: List[OntologyUpdateProposal]
    ) -> List[OntologyUpdateProposal]:
        """
        Validate proposals against constraints.

        Filters out proposals that violate econ/datapack/task-graph constraints.

        Args:
            proposals: List of proposals to validate

        Returns:
            List of valid proposals
        """
        valid = []

        for prop in proposals:
            # Check econ constraints
            if not self._check_econ_constraints(prop):
                prop.respects_econ_constraints = False
                continue

            # Check datapack constraints
            if not self._check_datapack_constraints(prop):
                prop.respects_datapack_constraints = False
                continue

            # Check task graph constraints
            if not self._check_task_graph_constraints(prop):
                prop.respects_task_graph = False
                continue

            valid.append(prop)

        return valid

    def _check_econ_constraints(self, prop: OntologyUpdateProposal) -> bool:
        """
        Check if proposal respects economic constraints.

        Forbidden:
        - Modifying price_per_unit
        - Modifying damage_cost
        - Modifying reward weights
        """
        # Proposals cannot set economic parameters
        forbidden_keys = {
            "price_per_unit",
            "damage_cost",
            "wage_parity",
            "alpha",
            "beta",
            "gamma",
        }

        if any(k in prop.proposed_changes for k in forbidden_keys):
            return False

        return True

    def _check_datapack_constraints(self, prop: OntologyUpdateProposal) -> bool:
        """
        Check if proposal respects datapack constraints.

        Forbidden:
        - Modifying tier classification
        - Modifying novelty scoring
        """
        forbidden_keys = {"tier", "novelty_score", "data_premium"}

        if any(k in prop.proposed_changes for k in forbidden_keys):
            return False

        return True

    def _check_task_graph_constraints(self, prop: OntologyUpdateProposal) -> bool:
        """
        Check if proposal respects task graph constraints.

        Forbidden:
        - Deleting tasks
        - Modifying task dependencies directly
        """
        if prop.proposal_type == ProposalType.ADD_SKILL_GATE:
            # Skill gates are allowed (they add preconditions, not modify)
            return True

        # Other proposals should not reference task deletion
        if "delete_task" in prop.proposed_changes:
            return False

        return True
```

---

## 5. Causality & Dependency Constraints

### 5.1 Dependency Diagram

```
┌─────────────────────────────────────────────────────────┐
│               UPSTREAM (Constraint Sources)             │
├─────────────────────────────────────────────────────────┤
│  EconomicController  │  DatapackEngine  │  TaskGraph   │
│  (econ physics)      │  (data physics)  │  (task DAG)  │
└──────────────┬─────────────────┬──────────────┬─────────┘
               │                 │              │
               ▼                 ▼              ▼
         ┌──────────────────────────────────────────┐
         │   Stage 2.1: SemanticPrimitiveExtractor  │
         │   (SIMA-2 rollouts → SemanticPrimitive)  │
         └──────────────────┬───────────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────────┐
         │   Stage 2.2: OntologyUpdateEngine        │
         │   (SemanticPrimitive → Proposal)         │
         └──────────────────┬───────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              DOWNSTREAM (Proposal Consumers)            │
├─────────────────────────────────────────────────────────┤
│  SemanticOrchestratorV2  →  Apply proposals (advisory) │
│  TaskGraphRefiner        →  Update task structure      │
│  SIMA-2 Bridge           →  Filter primitives          │
│  VLA/Diffusion/RL        →  Constraint propagation     │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Causality Rules

**OntologyUpdateEngine**:
1. **Cannot originate** economic parameters (price, wage, damage cost)
2. **Cannot originate** data valuation logic (tier, novelty, premium)
3. **Cannot mutate** ontology directly (proposals only)
4. **Cannot modify** reward math or RL training loops
5. **Cannot delete** existing task graph nodes

**Can consume**:
1. `EconSignals.error_urgency` → elevate risk proposals
2. `DatapackSignals.tier2_fraction` → suggest frontier focus
3. `TaskNode.affordances` → propose new affordances
4. `SemanticPrimitive.risk_level` → propose risk adjustments

---

## 6. Stage 2 Pipeline Contract

### 6.1 Input
- **Source**: Stage 2.1 `SemanticPrimitiveExtractor.extract_primitives_from_rollout()`
- **Type**: `List[SemanticPrimitive]`
- **Format**: Dataclass with `primitive_id`, `task_type`, `tags`, `risk_level`, etc.

### 6.2 Processing
- **Module**: `OntologyUpdateEngine.generate_proposals(primitives)`
- **Constraints**: Respect econ/datapack/task-graph boundaries
- **Validation**: `OntologyUpdateEngine.validate_proposals()`

### 6.3 Output
- **Type**: `List[OntologyUpdateProposal]`
- **Format**: JSON-safe dataclass with `proposal_id`, `proposal_type`, `proposed_changes`

### 6.4 Storage
- **Location**: `results/stage2/ontology_proposals/`
- **Format**: JSONL (one proposal per line)
- **Schema**: `{"proposal_id": "...", "proposal_type": "...", ...}`

### 6.5 Downstream Consumers
- **SemanticOrchestratorV2**: `apply_ontology_proposals(proposals)` (Stage 2.3)
- **TaskGraphRefiner**: Uses proposals to suggest task splits/merges (Stage 2.3)
- **SIMA-2 Bridge**: Filters rollouts based on skill gates
- **VLA/Diffusion**: Receives affordance/fragility constraints via orchestrator

---

## 7. Smoke Test Specification

### 7.1 File
`scripts/smoke_test_ontology_update_engine.py`

### 7.2 Test Cases

```python
#!/usr/bin/env python3
"""
Smoke test for Stage 2.2 OntologyUpdateEngine.

Validates:
1. Proposal generation from SemanticPrimitives
2. JSON-safety of all proposals
3. Constraint compliance (econ/datapack/task-graph)
4. Determinism (same inputs → same outputs)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.ontology_proposals import ProposalType, ProposalPriority
from src.orchestrator.ontology import build_drawer_vase_ontology
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals


def _make_test_primitives():
    """Create test SemanticPrimitives."""
    return [
        SemanticPrimitive(
            primitive_id="prim_001",
            task_type="open_drawer",
            tags=["drawer", "grasp", "pull"],
            risk_level="medium",
            energy_intensity=0.3,
            success_rate=0.9,
            avg_steps=5.0,
            source="sima2",
        ),
        SemanticPrimitive(
            primitive_id="prim_002",
            task_type="move_vase",
            tags=["vase", "fragile", "lift", "place"],
            risk_level="high",
            energy_intensity=0.15,
            success_rate=0.85,
            avg_steps=8.0,
            source="sima2",
        ),
        SemanticPrimitive(
            primitive_id="prim_003",
            task_type="push_box",
            tags=["box", "push", "forceful"],
            risk_level="low",
            energy_intensity=1.2,
            success_rate=0.95,
            avg_steps=3.0,
            source="sima2",
        ),
    ]


def main():
    print("[smoke_test_ontology_update_engine] Starting tests...")

    # Setup
    ontology = build_drawer_vase_ontology()
    econ_signals = EconSignals(
        error_urgency=0.6,  # High error urgency
        energy_urgency=0.3,
        damage_cost_total=75.0,
    )
    datapack_signals = DatapackSignals(tier2_fraction=0.08)

    engine = OntologyUpdateEngine(
        ontology=ontology,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
    )

    primitives = _make_test_primitives()

    # Test 1: Generate proposals
    proposals = engine.generate_proposals(primitives)
    assert proposals, "Expected proposals to be generated"
    print(f"[TEST 1 PASS] Generated {len(proposals)} proposals")

    # Test 2: JSON-safety
    for prop in proposals:
        prop_dict = prop.to_dict()
        assert isinstance(prop_dict, dict), "Proposal to_dict() should return dict"
        json_str = json.dumps(prop_dict)
        assert json_str, "Proposal should be JSON-serializable"
    print(f"[TEST 2 PASS] All {len(proposals)} proposals are JSON-safe")

    # Test 3: Required fields
    for prop in proposals:
        assert prop.proposal_id, "proposal_id required"
        assert prop.proposal_type, "proposal_type required"
        assert prop.priority, "priority required"
        assert prop.source_primitive_id, "source_primitive_id required"
        assert isinstance(prop.proposed_changes, dict), "proposed_changes must be dict"
        assert prop.rationale, "rationale required"
    print(f"[TEST 3 PASS] All proposals have required fields")

    # Test 4: Constraint compliance
    valid_proposals = engine.validate_proposals(proposals)
    assert len(valid_proposals) <= len(proposals), "Validation should not add proposals"
    for prop in valid_proposals:
        assert prop.respects_econ_constraints, "Must respect econ constraints"
        assert prop.respects_datapack_constraints, "Must respect datapack constraints"
        assert prop.respects_task_graph, "Must respect task graph"
    print(f"[TEST 4 PASS] {len(valid_proposals)}/{len(proposals)} proposals valid")

    # Test 5: Proposal type coverage
    proposal_types = {prop.proposal_type for prop in proposals}
    expected_types = {
        ProposalType.ADD_AFFORDANCE,
        ProposalType.ADJUST_RISK,
        ProposalType.INFER_FRAGILITY,
        ProposalType.ADD_SKILL_GATE,
    }
    assert proposal_types & expected_types, f"Expected proposal types in {expected_types}"
    print(f"[TEST 5 PASS] Proposal types: {[pt.value for pt in proposal_types]}")

    # Test 6: Fragility inference for high-risk primitive
    fragility_proposals = [
        p for p in proposals if p.proposal_type == ProposalType.INFER_FRAGILITY
    ]
    assert fragility_proposals, "Expected fragility proposals for 'fragile' primitive"
    for prop in fragility_proposals:
        assert "inferred_fragility" in prop.proposed_changes
        assert 0.0 <= prop.proposed_changes["inferred_fragility"] <= 1.0
    print(f"[TEST 6 PASS] Fragility inference working ({len(fragility_proposals)} proposals)")

    # Test 7: Risk adjustment for high error urgency
    risk_adjust_proposals = [
        p for p in proposals if p.proposal_type == ProposalType.ADJUST_RISK
    ]
    # With error_urgency=0.6, should elevate risk for medium/high primitives
    assert risk_adjust_proposals, "Expected risk adjustments for high error urgency"
    for prop in risk_adjust_proposals:
        assert prop.proposed_changes["new_risk_level"] > prop.proposed_changes["old_risk_level"]
    print(f"[TEST 7 PASS] Risk adjustment working ({len(risk_adjust_proposals)} proposals)")

    # Test 8: Skill gating for fragile objects
    skill_gate_proposals = [
        p for p in proposals if p.proposal_type == ProposalType.ADD_SKILL_GATE
    ]
    # "fragile" primitive should trigger skill gates
    assert skill_gate_proposals, "Expected skill gates for fragile objects"
    for prop in skill_gate_proposals:
        assert "gated_skill_id" in prop.proposed_changes
        assert "preconditions" in prop.proposed_changes
        assert "safety_threshold" in prop.proposed_changes
    print(f"[TEST 8 PASS] Skill gating working ({len(skill_gate_proposals)} proposals)")

    # Test 9: Priority assignment
    critical_proposals = [p for p in proposals if p.priority == ProposalPriority.CRITICAL]
    # Fragility inference + high error urgency → CRITICAL priority
    assert critical_proposals, "Expected CRITICAL priority for fragility/high-urgency"
    print(f"[TEST 9 PASS] Priority assignment working ({len(critical_proposals)} CRITICAL)")

    # Test 10: Determinism
    proposals_2 = engine.generate_proposals(primitives)
    assert len(proposals_2) == len(proposals), "Determinism check: same input → same count"
    # Note: proposal_ids will differ (UUID), but types/counts should match
    types_1 = sorted([p.proposal_type.value for p in proposals])
    types_2 = sorted([p.proposal_type.value for p in proposals_2])
    assert types_1 == types_2, "Determinism check: same proposal types"
    print(f"[TEST 10 PASS] Determinism validated")

    print("[smoke_test_ontology_update_engine] All tests passed!")


if __name__ == "__main__":
    main()
```

### 7.3 Integration with run_all_smokes.py

Add to `SMOKES` list:
```python
["python3", "scripts/smoke_test_ontology_update_engine.py"],
```

---

## 8. Codex Implementation Instructions

### 8.1 Files to Create

#### File 1: `src/sima2/ontology_proposals.py`
**Purpose**: Proposal schema dataclasses

**Classes**:
- `ProposalType(Enum)`: 9 proposal types
- `ProposalPriority(Enum)`: CRITICAL, HIGH, MEDIUM, LOW
- `OntologyUpdateProposal(dataclass)`: Main proposal schema

**Methods**:
- `to_dict() -> Dict[str, Any]`: JSON serialization
- `from_dict(d: Dict[str, Any]) -> OntologyUpdateProposal`: Deserialization

**Validation**:
- All fields JSON-safe (no numpy, no torch tensors)
- `proposed_changes` must be `Dict[str, Any]`
- `confidence` in [0.0, 1.0]

#### File 2: `src/sima2/ontology_update_engine.py`
**Purpose**: Proposal generation engine

**Class**: `OntologyUpdateEngine`

**Constructor**:
```python
def __init__(
    self,
    ontology: EnvironmentOntology,
    task_graph: Optional[TaskGraph] = None,
    econ_signals: Optional[EconSignals] = None,
    datapack_signals: Optional[DatapackSignals] = None,
)
```

**Methods**:
1. `generate_proposals(primitives: List[SemanticPrimitive]) -> List[OntologyUpdateProposal]`
2. `validate_proposals(proposals: List[OntologyUpdateProposal]) -> List[OntologyUpdateProposal]`
3. `_propose_affordances(prim: SemanticPrimitive) -> List[OntologyUpdateProposal]`
4. `_propose_risk_adjustments(prim: SemanticPrimitive) -> List[OntologyUpdateProposal]`
5. `_propose_fragility_inference(prim: SemanticPrimitive) -> List[OntologyUpdateProposal]`
6. `_propose_skill_gates(prim: SemanticPrimitive) -> List[OntologyUpdateProposal]`
7. `_propose_energy_heuristics(prim: SemanticPrimitive) -> List[OntologyUpdateProposal]`
8. `_propose_semantic_tags(prim: SemanticPrimitive) -> List[OntologyUpdateProposal]`
9. `_check_econ_constraints(prop: OntologyUpdateProposal) -> bool`
10. `_check_datapack_constraints(prop: OntologyUpdateProposal) -> bool`
11. `_check_task_graph_constraints(prop: OntologyUpdateProposal) -> bool`
12. `_make_proposal_id() -> str`
13. `_risk_level_to_float(risk_level: str) -> float`

**Validation Logic**:
- No mutation of ontology state
- Read-only access to task_graph
- Constraint checks before returning proposals

#### File 3: `scripts/smoke_test_ontology_update_engine.py`
**Purpose**: Smoke test validation

**Test Coverage**:
1. Proposal generation
2. JSON-safety
3. Required fields
4. Constraint compliance
5. Proposal type coverage
6. Fragility inference
7. Risk adjustment
8. Skill gating
9. Priority assignment
10. Determinism

**Exit Code**: 0 if pass, 1 if fail

### 8.2 Import Statements

```python
# ontology_proposals.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from enum import Enum

# ontology_update_engine.py
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.sima2.ontology_proposals import (
    OntologyUpdateProposal,
    ProposalType,
    ProposalPriority,
)
from src.orchestrator.ontology import (
    EnvironmentOntology,
    AffordanceType,
    ObjectCategory,
)
from src.orchestrator.task_graph import TaskGraph
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals
```

### 8.3 Strict Requirements

1. **No mutations**: OntologyUpdateEngine must NOT call `ontology.add_object()` or similar
2. **Advisory-only**: All outputs are proposals, not changes
3. **JSON-safe**: Every proposal must serialize to JSON without errors
4. **Deterministic**: Same primitives + same signals → same proposal types/counts (UUIDs may differ)
5. **Constraint validation**: Forbidden keys must be rejected

### 8.4 Missing Helpers

None — all dependencies exist in current codebase.

### 8.5 Testing Checklist

Before committing:
- [ ] Run `python3 scripts/smoke_test_ontology_update_engine.py`
- [ ] Verify JSON serialization: `json.dumps(prop.to_dict())`
- [ ] Check constraint violations trigger validation failures
- [ ] Verify no ontology mutations in `generate_proposals()`
- [ ] Confirm proposals reference correct primitive_id

---

## 9. Stage 2 Pipeline Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                   Stage 2: Semantic Layer                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Stage 2.1: SemanticPrimitiveExtractor                 │   │
│  │  Input:  SIMA-2 rollouts (stubbed)                     │   │
│  │  Output: SemanticPrimitive objects                     │   │
│  │  Status: ✅ COMPLETE                                   │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                       │                                        │
│                       ▼                                        │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Stage 2.2: OntologyUpdateEngine                       │   │
│  │  Input:  SemanticPrimitive[]                           │   │
│  │  Output: OntologyUpdateProposal[]                      │   │
│  │  Storage: results/stage2/ontology_proposals/*.jsonl    │   │
│  │  Status: 🔄 THIS STAGE                                 │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                       │                                        │
│                       ▼                                        │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Stage 2.3: TaskGraphRefiner (NEXT)                    │   │
│  │  Input:  OntologyUpdateProposal[]                      │   │
│  │  Output: TaskGraphUpdate[]                             │   │
│  │  Status: ⏸️  PENDING                                    │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                       │                                        │
│                       ▼                                        │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Stage 2.4: SemanticTagPropagator (NEXT)               │   │
│  │  Input:  OntologyUpdateProposal[]                      │   │
│  │  Output: Unified semantic tags across VLA/SIMA/etc.    │   │
│  │  Status: ⏸️  PENDING                                    │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 10. SemanticOrchestratorV2 Proposal Consumption Interface (Preview for Stage 2.3)

```python
class SemanticOrchestratorV2(SemanticOrchestrator):
    """
    Extended SemanticOrchestrator with ontology proposal consumption.

    To be implemented in Stage 2.3.
    """

    def apply_ontology_proposals(
        self,
        proposals: List[OntologyUpdateProposal],
        apply_mode: Literal["advisory", "immediate"] = "advisory",
    ) -> Dict[str, Any]:
        """
        Apply ontology update proposals (advisory-only by default).

        Args:
            proposals: List of OntologyUpdateProposals from Stage 2.2
            apply_mode: "advisory" (log only) or "immediate" (mutate ontology)

        Returns:
            Application report with accepted/rejected/deferred proposals
        """
        report = {
            "accepted": [],
            "rejected": [],
            "deferred": [],
            "conflicts": [],
        }

        # Validate all proposals first
        valid_proposals = self._validate_proposal_batch(proposals)

        # Detect conflicts (e.g., multiple risk adjustments for same object)
        conflicts = self._detect_conflicts(valid_proposals)
        if conflicts:
            report["conflicts"] = conflicts
            # Resolve via priority + confidence
            resolved = self._resolve_conflicts(conflicts)
            valid_proposals = resolved

        # Apply proposals based on mode
        for prop in valid_proposals:
            if apply_mode == "advisory":
                # Log only, no mutation
                report["accepted"].append(prop.proposal_id)
            elif apply_mode == "immediate":
                # Mutate ontology in-place
                success = self._apply_single_proposal(prop)
                if success:
                    report["accepted"].append(prop.proposal_id)
                else:
                    report["rejected"].append(prop.proposal_id)

        return report

    def _apply_single_proposal(self, prop: OntologyUpdateProposal) -> bool:
        """Apply a single proposal to the ontology (Stage 2.3)."""
        # Implementation in Stage 2.3
        pass
```

---

## 11. Acceptance Criteria

### Stage 2.2 is complete when:

1. ✅ `src/sima2/ontology_proposals.py` exists with all 9 ProposalTypes
2. ✅ `src/sima2/ontology_update_engine.py` exists with OntologyUpdateEngine class
3. ✅ `scripts/smoke_test_ontology_update_engine.py` passes all 10 tests
4. ✅ `run_all_smokes.py` includes new smoke test
5. ✅ All proposals are JSON-safe (can serialize/deserialize)
6. ✅ No ontology mutations occur in OntologyUpdateEngine
7. ✅ Constraint validation rejects forbidden proposals
8. ✅ Determinism: same primitives → same proposal types/counts
9. ✅ Storage: JSONL output format tested
10. ✅ Documentation: This spec committed as STAGE_2_2_*.md

---

## 12. Next Steps (Stage 2.3 Preview)

After Stage 2.2 is complete:

**Stage 2.3: TaskGraphRefiner**
- Consumes `OntologyUpdateProposal[]`
- Produces `TaskGraphUpdate[]` (split/merge/reorder tasks)
- Advisory-only, no direct task graph mutation
- Smoke test: validate task graph updates respect dependencies

**Stage 2.4: SemanticTagPropagator**
- Consumes `OntologyUpdateProposal[]`
- Unifies semantic tags across VLA/SIMA/diffusion/RL
- Updates cross-module vocabularies
- Smoke test: tag consistency across modules

**User Prompt for Continuation**:
> "Ready for Stage 2.3 (TaskGraphRefiner)?"

---

**End of Stage 2.2 Specification**
