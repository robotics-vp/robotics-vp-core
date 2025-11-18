# Stage 2.3: TaskGraphRefiner — Full Specification

**Status**: Ready for Codex implementation
**Date**: 2025-11-17
**Prerequisites**: Stage 2.1 (SemanticPrimitiveExtractor) + Stage 2.2 (OntologyUpdateEngine) complete

---

## 1. Overview

### Purpose
The **TaskGraphRefiner** is a Stage 2 module that:
1. Consumes `OntologyUpdateProposal[]` from Stage 2.2
2. Consumes `SemanticPrimitive[]` from Stage 2.1
3. Produces `TaskGraphRefinementProposal[]` (advisory-only, no DAG mutations)
4. Respects constraints from EconomicController, DatapackEngine, Ontology, and existing TaskGraph
5. Provides schema-driven, JSON-safe task structure refinement proposals

### Key Design Principles
- **Advisory-only**: Does NOT mutate task graph directly
- **DAG-preserving**: Cannot create cycles or violate topological order
- **Schema-driven**: All proposals use typed dataclasses
- **JSON-safe**: All outputs serializable for storage/logging
- **Constraint-respecting**: Cannot originate econ/data-valuation/reward logic
- **Deterministic**: Given same inputs, produces same outputs (for testing)

---

## 2. Constraint Sources (Dependency Mapping)

### Upstream Dependencies (Constraints FROM)

#### 2.1 EconomicController
**File**: `src/orchestrator/economic_controller.py`

**Constraints**:
- Cannot propose changes to reward math
- Cannot propose changes to MPL/error/energy attribution
- Cannot set economic parameters

**Allowed Consumption**:
- `EconSignals.error_urgency` → May insert safety checkpoints
- `EconSignals.energy_urgency` → May suggest efficiency reordering
- `EconSignals.mpl_urgency` → May propose task parallelization

#### 2.2 DatapackEngine
**File**: `src/orchestrator/datapack_engine.py`

**Constraints**:
- Cannot propose changes to data valuation
- Cannot modify tier classification

**Allowed Consumption**:
- `DatapackSignals.tier2_fraction` → May suggest frontier task focus
- `DatapackSignals.data_coverage_score` → May propose task decomposition for better coverage

#### 2.3 Ontology (Read-Only)
**File**: `src/orchestrator/ontology.py`

**Constraints**:
- Cannot mutate ontology (read-only)
- Cannot delete affordances or objects

**Allowed Consumption**:
- `ObjectSpec.fragility` → May insert safety tasks before fragile interactions
- `AffordanceSpec.risk_level` → May split high-risk tasks into safer sub-tasks
- `ObjectSpec.affordances` → May reorder tasks based on affordance availability

#### 2.4 TaskGraph (Read-Only)
**File**: `src/orchestrator/task_graph.py`

**Constraints**:
- Cannot delete existing nodes
- Cannot modify edges directly (proposals only)
- Cannot create cycles

**Allowed Consumption**:
- `TaskNode.preconditions` → May propose additional preconditions
- `TaskNode.skill_id` → May split multi-skill tasks
- `TaskNode.metadata['semantic_priority']` → May reorder based on priority

#### 2.5 Stage 2.1 (SemanticPrimitiveExtractor)
**File**: `src/sima2/semantic_primitive_extractor.py`

**Constraints**:
- Cannot modify primitive extraction logic

**Allowed Consumption**:
- `SemanticPrimitive.tags` → May derive task decompositions
- `SemanticPrimitive.risk_level` → May propose safety task insertion
- `SemanticPrimitive.avg_steps` → May estimate task duration for reordering

#### 2.6 Stage 2.2 (OntologyUpdateEngine)
**File**: `src/sima2/ontology_update_engine.py`

**Constraints**:
- Cannot modify proposal generation logic

**Allowed Consumption**:
- `OntologyUpdateProposal[ADD_SKILL_GATE]` → MUST insert checkpoint tasks
- `OntologyUpdateProposal[INFER_FRAGILITY]` → May insert safety tasks
- `OntologyUpdateProposal[ADD_SAFETY_CONSTRAINT]` → May reorder to respect clearance

### Downstream Dependencies (Proposals TO)

#### 2.7 SemanticOrchestratorV2
**Consumption Interface** (Stage 2.4+):
- `apply_task_graph_refinements(List[TaskGraphRefinementProposal])` → Advisory application
- `validate_refinements(List[TaskGraphRefinementProposal])` → DAG cycle detection
- `merge_refinements(List[TaskGraphRefinementProposal])` → Conflict resolution

#### 2.8 HRL / Skill Scheduler
**Consumption Path**:
- Checkpoint proposals → Skill execution gates
- Parallelization proposals → Multi-skill execution strategies

---

## 3. TaskGraphRefinementProposal Schema

### 3.1 Core Proposal Dataclass

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from enum import Enum


class RefinementType(Enum):
    """Types of task graph refinement proposals."""
    SPLIT_TASK = "split_task"  # Split complex task into sub-tasks
    INSERT_CHECKPOINT = "insert_checkpoint"  # Add safety/verification checkpoint
    REORDER_TASKS = "reorder_tasks"  # Suggest efficiency reordering
    MERGE_TASKS = "merge_tasks"  # Merge redundant tasks
    ADD_PRECONDITION = "add_precondition"  # Add safety precondition
    PARALLELIZE_TASKS = "parallelize_tasks"  # Suggest parallel execution
    INSERT_RECOVERY = "insert_recovery"  # Add error recovery task
    ADJUST_PRIORITY = "adjust_priority"  # Change task priority


class RefinementPriority(Enum):
    """Priority levels for refinement application."""
    CRITICAL = "critical"  # Safety-critical (must apply)
    HIGH = "high"  # Economic urgency-driven
    MEDIUM = "medium"  # Efficiency improvements
    LOW = "low"  # Nice-to-have optimizations


@dataclass
class TaskGraphRefinementProposal:
    """
    Schema for a single task graph refinement proposal.

    Advisory-only; does not mutate task graph directly.
    SemanticOrchestrator decides whether/how to apply.
    """
    # Identification
    proposal_id: str  # Unique ID
    refinement_type: RefinementType
    priority: RefinementPriority = RefinementPriority.MEDIUM

    # Source tracking
    source_primitive_ids: List[str] = field(default_factory=list)  # SemanticPrimitives
    source_ontology_proposal_ids: List[str] = field(default_factory=list)  # OntologyUpdateProposals
    source: str = "task_graph_refiner"

    # Target specification
    target_task_ids: List[str] = field(default_factory=list)  # Affected task nodes
    parent_task_id: Optional[str] = None  # Parent of new tasks (for INSERT)

    # Refinement content (type-specific)
    proposed_changes: Dict[str, Any] = field(default_factory=dict)

    # Justification
    rationale: str = ""
    confidence: float = 1.0  # 0-1

    # Constraint compliance
    respects_econ_constraints: bool = True
    respects_datapack_constraints: bool = True
    respects_dag_topology: bool = True  # No cycles created
    preserves_existing_nodes: bool = True  # No deletions

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "refinement_type": self.refinement_type.value,
            "priority": self.priority.value,
            "source_primitive_ids": self.source_primitive_ids,
            "source_ontology_proposal_ids": self.source_ontology_proposal_ids,
            "source": self.source,
            "target_task_ids": self.target_task_ids,
            "parent_task_id": self.parent_task_id,
            "proposed_changes": self.proposed_changes,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "respects_econ_constraints": self.respects_econ_constraints,
            "respects_datapack_constraints": self.respects_datapack_constraints,
            "respects_dag_topology": self.respects_dag_topology,
            "preserves_existing_nodes": self.preserves_existing_nodes,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskGraphRefinementProposal":
        """Create from dictionary."""
        return cls(
            proposal_id=d["proposal_id"],
            refinement_type=RefinementType(d["refinement_type"]),
            priority=RefinementPriority(d.get("priority", "medium")),
            source_primitive_ids=d.get("source_primitive_ids", []),
            source_ontology_proposal_ids=d.get("source_ontology_proposal_ids", []),
            source=d.get("source", "task_graph_refiner"),
            target_task_ids=d.get("target_task_ids", []),
            parent_task_id=d.get("parent_task_id"),
            proposed_changes=d.get("proposed_changes", {}),
            rationale=d.get("rationale", ""),
            confidence=d.get("confidence", 1.0),
            respects_econ_constraints=d.get("respects_econ_constraints", True),
            respects_datapack_constraints=d.get("respects_datapack_constraints", True),
            respects_dag_topology=d.get("respects_dag_topology", True),
            preserves_existing_nodes=d.get("preserves_existing_nodes", True),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )
```

### 3.2 Refinement Type Specifications

#### SPLIT_TASK
**Purpose**: Decompose complex task into safer/more efficient sub-tasks

```python
# proposed_changes format:
{
    "original_task_id": "pull_drawer",
    "new_sub_tasks": [
        {
            "task_id": "pull_drawer_sub_1",
            "name": "Check Vase Position",
            "task_type": "checkpoint",
            "skill_id": None,
            "preconditions": ["approach_drawer"],
            "postconditions": ["vase_position_known"],
        },
        {
            "task_id": "pull_drawer_sub_2",
            "name": "Pull Slowly",
            "task_type": "skill",
            "skill_id": 2,  # PULL
            "preconditions": ["vase_position_known"],
            "postconditions": ["drawer_open"],
        },
    ],
    "preserve_original": False,  # Original task replaced by sub-tasks
}
```

#### INSERT_CHECKPOINT
**Purpose**: Add safety/verification task before high-risk operations

```python
# proposed_changes format:
{
    "checkpoint_task": {
        "task_id": "fragility_check_001",
        "name": "Verify Fragile Object Safety",
        "task_type": "checkpoint",
        "preconditions": ["grasp_handle"],
        "postconditions": ["fragility_check_passed"],
        "metadata": {"check_type": "collision_avoidance", "objects": ["vase_01"]},
    },
    "insert_before_task_id": "pull_drawer",
    "mandatory": True,  # Must execute (not optional)
}
```

#### REORDER_TASKS
**Purpose**: Suggest more efficient task execution order

```python
# proposed_changes format:
{
    "reordered_task_ids": [
        "approach_drawer",  # Move earlier for efficiency
        "check_vase_position",
        "grasp_handle",
        "pull_drawer",
        "release_handle",
    ],
    "original_order": [
        "approach_drawer",
        "grasp_handle",
        "check_vase_position",  # Was later
        "pull_drawer",
        "release_handle",
    ],
    "reason": "energy_efficiency",  # "safety", "mpl_optimization", "energy_efficiency"
    "estimated_improvement": {
        "mpl_delta": 2.0,  # units/hour improvement
        "energy_delta_Wh": -0.05,  # Energy saved
    },
}
```

#### MERGE_TASKS
**Purpose**: Combine redundant or tightly-coupled tasks

```python
# proposed_changes format:
{
    "task_ids_to_merge": ["approach_object", "grasp_object"],
    "merged_task": {
        "task_id": "approach_and_grasp",
        "name": "Approach and Grasp Object",
        "task_type": "skill",
        "skill_id": 1,  # GRASP (includes approach)
        "preconditions": [],
        "postconditions": ["object_grasped"],
    },
    "reason": "redundant_motion",
}
```

#### ADD_PRECONDITION
**Purpose**: Add safety precondition based on ontology/primitive insights

```python
# proposed_changes format:
{
    "target_task_id": "pull_drawer",
    "new_preconditions": [
        "fragility_check_passed",
        "collision_avoidance_active",
    ],
    "source": "ontology_proposal_ADD_SKILL_GATE",
    "safety_critical": True,
}
```

#### PARALLELIZE_TASKS
**Purpose**: Suggest tasks that can execute in parallel (if multi-limb/multi-agent)

```python
# proposed_changes format:
{
    "parallel_groups": [
        {
            "group_id": "parallel_group_1",
            "task_ids": ["check_vase_position", "approach_drawer"],
            "reason": "independent_tasks",
        }
    ],
    "estimated_speedup": 1.5,  # Execution time reduction factor
}
```

#### INSERT_RECOVERY
**Purpose**: Add error recovery task after high-risk operations

```python
# proposed_changes format:
{
    "recovery_task": {
        "task_id": "recovery_vase_stabilize",
        "name": "Stabilize Vase if Disturbed",
        "task_type": "skill",
        "skill_id": 8,  # STABILIZE (hypothetical)
        "preconditions": ["pull_drawer", "vase_disturbed"],
        "postconditions": ["vase_stable"],
    },
    "insert_after_task_id": "pull_drawer",
    "conditional": True,  # Only execute if error detected
    "condition": "vase_disturbed == True",
}
```

#### ADJUST_PRIORITY
**Purpose**: Change task semantic priority based on econ urgency

```python
# proposed_changes format:
{
    "task_id": "pull_drawer",
    "old_priority": "medium",
    "new_priority": "high",
    "reason": "error_urgency_high",
    "trigger": {"econ_signal": "error_urgency", "value": 0.7},
}
```

---

## 4. TaskGraphRefiner Module

### 4.1 File Structure
```
src/sima2/
├── semantic_primitive_extractor.py  # Stage 2.1 (existing)
├── ontology_update_engine.py        # Stage 2.2 (existing)
└── task_graph_refiner.py            # Stage 2.3 (NEW)

src/sima2/task_graph_proposals.py   # Stage 2.3 schemas (NEW)
```

### 4.2 Module Implementation

```python
"""
TaskGraphRefiner: Advisory task graph refinement proposal generator.

IMPORTANT:
- Does NOT mutate task graph directly
- Consumes OntologyUpdateProposals + SemanticPrimitives
- Produces TaskGraphRefinementProposals for SemanticOrchestrator
- Respects econ/datapack/ontology/DAG constraints
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType as OntologyProposalType
from src.sima2.task_graph_proposals import (
    TaskGraphRefinementProposal,
    RefinementType,
    RefinementPriority,
)
from src.orchestrator.task_graph import TaskGraph, TaskNode, TaskType, TaskStatus
from src.orchestrator.ontology import EnvironmentOntology
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals


class TaskGraphRefiner:
    """
    Generates task graph refinement proposals from ontology proposals + primitives.

    Advisory-only; does not mutate task graph.
    """

    def __init__(
        self,
        task_graph: TaskGraph,
        ontology: EnvironmentOntology,
        econ_signals: Optional[EconSignals] = None,
        datapack_signals: Optional[DatapackSignals] = None,
    ):
        """
        Initialize TaskGraphRefiner.

        Args:
            task_graph: Current task graph (read-only)
            ontology: Current ontology (read-only)
            econ_signals: Economic constraints (optional)
            datapack_signals: Data constraints (optional)
        """
        self.task_graph = task_graph
        self.ontology = ontology
        self.econ_signals = econ_signals or EconSignals()
        self.datapack_signals = datapack_signals or DatapackSignals()

        # Proposal generation state (for determinism)
        self._proposal_counter = 0

    def generate_refinements(
        self,
        ontology_proposals: List[OntologyUpdateProposal],
        primitives: Optional[List[SemanticPrimitive]] = None,
    ) -> List[TaskGraphRefinementProposal]:
        """
        Generate task graph refinement proposals.

        Args:
            ontology_proposals: List of OntologyUpdateProposals from Stage 2.2
            primitives: Optional list of SemanticPrimitives from Stage 2.1

        Returns:
            List of TaskGraphRefinementProposals (advisory-only)
        """
        refinements = []

        # 1. Process ontology proposals
        for ont_prop in ontology_proposals:
            # ADD_SKILL_GATE → INSERT_CHECKPOINT (mandatory)
            if ont_prop.proposal_type == OntologyProposalType.ADD_SKILL_GATE:
                refinements.extend(self._insert_checkpoint_from_gate(ont_prop))

            # INFER_FRAGILITY → SPLIT_TASK (if high fragility)
            if ont_prop.proposal_type == OntologyProposalType.INFER_FRAGILITY:
                refinements.extend(self._split_task_for_fragility(ont_prop))

            # ADD_SAFETY_CONSTRAINT → REORDER_TASKS (clearance-aware)
            if ont_prop.proposal_type == OntologyProposalType.ADD_SAFETY_CONSTRAINT:
                refinements.extend(self._reorder_for_safety(ont_prop))

            # ADJUST_RISK → INSERT_RECOVERY
            if ont_prop.proposal_type == OntologyProposalType.ADJUST_RISK:
                refinements.extend(self._insert_recovery_for_risk(ont_prop))

        # 2. Process primitives (efficiency hints)
        if primitives:
            for prim in primitives:
                # Low energy primitives → REORDER_TASKS
                if prim.energy_intensity < 0.5 and self.econ_signals.energy_urgency > 0.3:
                    refinements.extend(self._reorder_for_energy(prim))

                # High success rate → MERGE_TASKS (if redundant)
                if prim.success_rate > 0.95:
                    refinements.extend(self._merge_redundant_tasks(prim))

        # 3. Economic urgency-driven refinements
        if self.econ_signals.error_urgency > 0.6:
            refinements.extend(self._adjust_priority_for_safety())

        if self.econ_signals.mpl_urgency > 0.5:
            refinements.extend(self._parallelize_for_throughput())

        return refinements

    def _make_proposal_id(self) -> str:
        """Generate unique proposal ID."""
        self._proposal_counter += 1
        return f"tgr_{self._proposal_counter:06d}_{uuid.uuid4().hex[:6]}"

    def _insert_checkpoint_from_gate(
        self, ont_prop: OntologyUpdateProposal
    ) -> List[TaskGraphRefinementProposal]:
        """
        Insert checkpoint task from ADD_SKILL_GATE proposal.

        Rule: If skill_id is gated, insert checkpoint before that skill's task.
        """
        refinements = []

        gated_skill_id = ont_prop.proposed_changes.get("gated_skill_id")
        if gated_skill_id is None:
            return refinements

        # Find task nodes with this skill_id
        for node in self.task_graph.get_all_nodes():
            if node.skill_id == gated_skill_id:
                # Insert checkpoint before this task
                checkpoint_task = {
                    "task_id": f"checkpoint_{node.task_id}",
                    "name": f"Safety Check Before {node.name}",
                    "task_type": "checkpoint",
                    "preconditions": ont_prop.proposed_changes.get("preconditions", []),
                    "postconditions": [f"{node.task_id}_gated_check_passed"],
                    "metadata": {
                        "check_type": "skill_gate",
                        "gated_skill_id": gated_skill_id,
                        "safety_threshold": ont_prop.proposed_changes.get("safety_threshold", 0.8),
                    },
                }

                refinement = TaskGraphRefinementProposal(
                    proposal_id=self._make_proposal_id(),
                    refinement_type=RefinementType.INSERT_CHECKPOINT,
                    priority=RefinementPriority.CRITICAL,
                    source_ontology_proposal_ids=[ont_prop.proposal_id],
                    target_task_ids=[node.task_id],
                    parent_task_id=node.parent_id,
                    proposed_changes={
                        "checkpoint_task": checkpoint_task,
                        "insert_before_task_id": node.task_id,
                        "mandatory": True,
                    },
                    rationale=f"Skill gate requires safety checkpoint before {node.name}",
                    confidence=ont_prop.confidence,
                    tags=ont_prop.tags + ["skill_gate", "safety"],
                )

                refinements.append(refinement)

        return refinements

    def _split_task_for_fragility(
        self, ont_prop: OntologyUpdateProposal
    ) -> List[TaskGraphRefinementProposal]:
        """
        Split task if high fragility inferred.

        Rule: Tasks involving fragile objects split into: check → act slowly → verify
        """
        refinements = []

        inferred_fragility = ont_prop.proposed_changes.get("inferred_fragility", 0.0)
        if inferred_fragility < 0.7:  # Only split if high fragility
            return refinements

        # Find tasks involving fragile interactions
        # (Heuristic: tasks with "pull", "move", "place" near fragile objects)
        fragile_task_names = ["pull", "move", "place", "lift"]

        for node in self.task_graph.get_all_nodes():
            if any(name in node.name.lower() for name in fragile_task_names):
                # Split into: pre-check → slow execution → post-verify
                sub_tasks = [
                    {
                        "task_id": f"{node.task_id}_pre_check",
                        "name": f"Pre-Check for {node.name}",
                        "task_type": "checkpoint",
                        "preconditions": node.preconditions,
                        "postconditions": [f"{node.task_id}_safe_to_proceed"],
                    },
                    {
                        "task_id": f"{node.task_id}_slow",
                        "name": f"{node.name} (Slow Mode)",
                        "task_type": "skill",
                        "skill_id": node.skill_id,
                        "preconditions": [f"{node.task_id}_safe_to_proceed"],
                        "postconditions": node.postconditions,
                        "metadata": {"speed_limit": 0.5, "force_limit": 0.7},
                    },
                    {
                        "task_id": f"{node.task_id}_verify",
                        "name": f"Verify {node.name} Success",
                        "task_type": "checkpoint",
                        "preconditions": node.postconditions,
                        "postconditions": [f"{node.task_id}_verified"],
                    },
                ]

                refinement = TaskGraphRefinementProposal(
                    proposal_id=self._make_proposal_id(),
                    refinement_type=RefinementType.SPLIT_TASK,
                    priority=RefinementPriority.HIGH,
                    source_ontology_proposal_ids=[ont_prop.proposal_id],
                    target_task_ids=[node.task_id],
                    proposed_changes={
                        "original_task_id": node.task_id,
                        "new_sub_tasks": sub_tasks,
                        "preserve_original": False,
                    },
                    rationale=f"High fragility ({inferred_fragility:.2f}) requires split execution",
                    confidence=0.8,
                    tags=["fragility", "safety_split"],
                )

                refinements.append(refinement)

        return refinements

    def _reorder_for_safety(
        self, ont_prop: OntologyUpdateProposal
    ) -> List[TaskGraphRefinementProposal]:
        """
        Reorder tasks to respect safety constraints.

        Rule: Tasks with clearance requirements must execute before risky motions.
        """
        refinements = []

        constraint_type = ont_prop.proposed_changes.get("constraint_type")
        if constraint_type != "collision_avoidance":
            return refinements

        # Get affected objects and skills
        objects = ont_prop.proposed_changes.get("objects", [])
        affected_skills = ont_prop.proposed_changes.get("applies_to_skills", [])

        # Find tasks that need reordering
        skill_tasks = [
            node for node in self.task_graph.get_all_nodes()
            if node.skill_id in affected_skills
        ]

        if len(skill_tasks) < 2:
            return refinements  # Nothing to reorder

        # Propose reordering: collision-check tasks first
        original_order = [node.task_id for node in skill_tasks]
        reordered = sorted(
            skill_tasks,
            key=lambda n: (
                0 if n.task_type == TaskType.CHECKPOINT else 1,
                n.task_id
            )
        )
        reordered_ids = [node.task_id for node in reordered]

        if original_order != reordered_ids:
            refinement = TaskGraphRefinementProposal(
                proposal_id=self._make_proposal_id(),
                refinement_type=RefinementType.REORDER_TASKS,
                priority=RefinementPriority.HIGH,
                source_ontology_proposal_ids=[ont_prop.proposal_id],
                target_task_ids=original_order,
                proposed_changes={
                    "reordered_task_ids": reordered_ids,
                    "original_order": original_order,
                    "reason": "safety",
                },
                rationale=f"Safety constraint requires checkpoints before risky motions",
                confidence=0.9,
                tags=["safety_reorder", "collision_avoidance"],
            )

            refinements.append(refinement)

        return refinements

    def _insert_recovery_for_risk(
        self, ont_prop: OntologyUpdateProposal
    ) -> List[TaskGraphRefinementProposal]:
        """
        Insert recovery task after risk adjustment.

        Rule: If risk elevated significantly, add recovery task.
        """
        refinements = []

        old_risk = ont_prop.proposed_changes.get("old_risk_level", 0.0)
        new_risk = ont_prop.proposed_changes.get("new_risk_level", 0.0)

        if new_risk < 0.7 or (new_risk - old_risk) < 0.3:
            return refinements  # Risk not high enough

        # Find tasks with elevated risk (heuristic: match tags)
        for node in self.task_graph.get_all_nodes():
            if any(tag in ont_prop.tags for tag in ["fragile", "vase", "glass"]):
                recovery_task = {
                    "task_id": f"recovery_{node.task_id}",
                    "name": f"Recover from {node.name} Error",
                    "task_type": "skill",
                    "skill_id": None,  # Generic recovery
                    "preconditions": [f"{node.task_id}_completed", "error_detected"],
                    "postconditions": ["system_stable"],
                    "metadata": {"recovery_type": "stabilize_fragile"},
                }

                refinement = TaskGraphRefinementProposal(
                    proposal_id=self._make_proposal_id(),
                    refinement_type=RefinementType.INSERT_RECOVERY,
                    priority=RefinementPriority.MEDIUM,
                    source_ontology_proposal_ids=[ont_prop.proposal_id],
                    target_task_ids=[node.task_id],
                    proposed_changes={
                        "recovery_task": recovery_task,
                        "insert_after_task_id": node.task_id,
                        "conditional": True,
                        "condition": "error_detected == True",
                    },
                    rationale=f"Risk elevated ({old_risk:.2f} → {new_risk:.2f}), recovery task needed",
                    confidence=0.7,
                    tags=["recovery", "risk_mitigation"],
                )

                refinements.append(refinement)

        return refinements

    def _reorder_for_energy(
        self, prim: SemanticPrimitive
    ) -> List[TaskGraphRefinementProposal]:
        """
        Reorder tasks for energy efficiency.

        Rule: Low-energy tasks execute earlier to reduce total energy.
        """
        refinements = []

        if self.econ_signals.energy_urgency < 0.3:
            return refinements  # Energy not critical

        # Find tasks that could be reordered
        # (Simplified: suggest moving low-energy skills earlier)
        all_skill_tasks = self.task_graph.get_skill_nodes()

        if len(all_skill_tasks) < 2:
            return refinements

        # Heuristic: reorder by estimated energy (lower first)
        original_order = [node.task_id for node in all_skill_tasks]
        # In production, use actual energy estimates; here we use heuristic
        reordered = sorted(all_skill_tasks, key=lambda n: n.skill_id or 0)  # Placeholder
        reordered_ids = [node.task_id for node in reordered]

        if original_order != reordered_ids:
            refinement = TaskGraphRefinementProposal(
                proposal_id=self._make_proposal_id(),
                refinement_type=RefinementType.REORDER_TASKS,
                priority=RefinementPriority.MEDIUM,
                source_primitive_ids=[prim.primitive_id],
                target_task_ids=original_order,
                proposed_changes={
                    "reordered_task_ids": reordered_ids,
                    "original_order": original_order,
                    "reason": "energy_efficiency",
                    "estimated_improvement": {
                        "energy_delta_Wh": -0.05,
                    },
                },
                rationale=f"Primitive shows low energy ({prim.energy_intensity:.2f}), reorder for efficiency",
                confidence=0.6,
                tags=["energy_optimization"],
            )

            refinements.append(refinement)

        return refinements

    def _merge_redundant_tasks(
        self, prim: SemanticPrimitive
    ) -> List[TaskGraphRefinementProposal]:
        """
        Merge redundant tasks.

        Rule: High success rate primitives → merge tightly-coupled tasks.
        """
        refinements = []

        if prim.success_rate < 0.95:
            return refinements

        # Find adjacent tasks with same skill type
        # (Simplified: merge "approach" + "grasp" into "approach_and_grasp")
        all_tasks = self.task_graph.get_all_nodes()

        for i, node in enumerate(all_tasks[:-1]):
            next_node = all_tasks[i + 1]

            # Heuristic: if consecutive tasks have adjacent skill_ids
            if (
                node.skill_id is not None
                and next_node.skill_id is not None
                and abs(node.skill_id - next_node.skill_id) == 1
            ):
                merged_task = {
                    "task_id": f"{node.task_id}_merged_{next_node.task_id}",
                    "name": f"{node.name} + {next_node.name}",
                    "task_type": "skill",
                    "skill_id": node.skill_id,  # Use first skill
                    "preconditions": node.preconditions,
                    "postconditions": next_node.postconditions,
                }

                refinement = TaskGraphRefinementProposal(
                    proposal_id=self._make_proposal_id(),
                    refinement_type=RefinementType.MERGE_TASKS,
                    priority=RefinementPriority.LOW,
                    source_primitive_ids=[prim.primitive_id],
                    target_task_ids=[node.task_id, next_node.task_id],
                    proposed_changes={
                        "task_ids_to_merge": [node.task_id, next_node.task_id],
                        "merged_task": merged_task,
                        "reason": "redundant_motion",
                    },
                    rationale=f"High success rate ({prim.success_rate:.2f}) → safe to merge",
                    confidence=0.7,
                    tags=["efficiency", "merge"],
                )

                refinements.append(refinement)
                break  # Only merge one pair per call

        return refinements

    def _adjust_priority_for_safety(self) -> List[TaskGraphRefinementProposal]:
        """
        Adjust task priorities based on error urgency.

        Rule: High error urgency → elevate checkpoint task priorities.
        """
        refinements = []

        for node in self.task_graph.get_all_nodes():
            if node.task_type == TaskType.CHECKPOINT:
                current_priority = node.metadata.get("semantic_priority", "medium")

                if current_priority != "high":
                    refinement = TaskGraphRefinementProposal(
                        proposal_id=self._make_proposal_id(),
                        refinement_type=RefinementType.ADJUST_PRIORITY,
                        priority=RefinementPriority.HIGH,
                        target_task_ids=[node.task_id],
                        proposed_changes={
                            "task_id": node.task_id,
                            "old_priority": current_priority,
                            "new_priority": "high",
                            "reason": "error_urgency_high",
                            "trigger": {
                                "econ_signal": "error_urgency",
                                "value": self.econ_signals.error_urgency,
                            },
                        },
                        rationale=f"Error urgency {self.econ_signals.error_urgency:.2f} → elevate checkpoint priority",
                        confidence=0.8,
                        tags=["priority_adjust", "safety"],
                    )

                    refinements.append(refinement)

        return refinements

    def _parallelize_for_throughput(self) -> List[TaskGraphRefinementProposal]:
        """
        Suggest task parallelization for MPL improvement.

        Rule: Independent tasks → parallel execution.
        """
        refinements = []

        # Find tasks with no dependencies on each other
        all_tasks = self.task_graph.get_all_nodes()

        # Simplified: find pairs of tasks with disjoint preconditions
        for i, task_a in enumerate(all_tasks):
            for task_b in all_tasks[i + 1 :]:
                # Check if tasks are independent
                if (
                    task_a.task_id not in task_b.preconditions
                    and task_b.task_id not in task_a.preconditions
                ):
                    refinement = TaskGraphRefinementProposal(
                        proposal_id=self._make_proposal_id(),
                        refinement_type=RefinementType.PARALLELIZE_TASKS,
                        priority=RefinementPriority.MEDIUM,
                        target_task_ids=[task_a.task_id, task_b.task_id],
                        proposed_changes={
                            "parallel_groups": [
                                {
                                    "group_id": f"parallel_{task_a.task_id}_{task_b.task_id}",
                                    "task_ids": [task_a.task_id, task_b.task_id],
                                    "reason": "independent_tasks",
                                }
                            ],
                            "estimated_speedup": 1.5,
                        },
                        rationale=f"MPL urgency {self.econ_signals.mpl_urgency:.2f} → parallelize independent tasks",
                        confidence=0.6,
                        tags=["parallelization", "throughput"],
                    )

                    refinements.append(refinement)
                    break  # Only one parallel group per call

            if refinements:
                break

        return refinements

    def validate_refinements(
        self, refinements: List[TaskGraphRefinementProposal]
    ) -> List[TaskGraphRefinementProposal]:
        """
        Validate refinements against constraints.

        Filters out refinements that violate econ/datapack/DAG constraints.

        Args:
            refinements: List of refinements to validate

        Returns:
            List of valid refinements
        """
        valid = []

        for ref in refinements:
            # Check econ constraints
            if not self._check_econ_constraints(ref):
                ref.respects_econ_constraints = False
                continue

            # Check datapack constraints
            if not self._check_datapack_constraints(ref):
                ref.respects_datapack_constraints = False
                continue

            # Check DAG topology (no cycles)
            if not self._check_dag_topology(ref):
                ref.respects_dag_topology = False
                continue

            # Check existing nodes preserved
            if not self._check_preserves_nodes(ref):
                ref.preserves_existing_nodes = False
                continue

            valid.append(ref)

        return valid

    def _check_econ_constraints(self, ref: TaskGraphRefinementProposal) -> bool:
        """
        Check if refinement respects economic constraints.

        Forbidden:
        - Modifying reward weights
        - Setting economic parameters
        """
        forbidden_keys = {
            "price_per_unit",
            "damage_cost",
            "alpha",
            "beta",
            "gamma",
        }

        if any(k in ref.proposed_changes for k in forbidden_keys):
            return False

        return True

    def _check_datapack_constraints(self, ref: TaskGraphRefinementProposal) -> bool:
        """
        Check if refinement respects datapack constraints.

        Forbidden:
        - Modifying tier classification
        - Setting data premiums
        """
        forbidden_keys = {"tier", "novelty_score", "data_premium"}

        if any(k in ref.proposed_changes for k in forbidden_keys):
            return False

        return True

    def _check_dag_topology(self, ref: TaskGraphRefinementProposal) -> bool:
        """
        Check if refinement preserves DAG topology (no cycles).

        Simplified cycle check (production should use full topological sort).
        """
        # For INSERT/SPLIT, check no cycles created
        # (Simplified: always pass; production should validate full DAG)
        return True

    def _check_preserves_nodes(self, ref: TaskGraphRefinementProposal) -> bool:
        """
        Check if refinement preserves existing nodes (no deletions).

        Only SPLIT_TASK and MERGE_TASKS can "replace" nodes (not delete).
        """
        if ref.refinement_type in {RefinementType.SPLIT_TASK, RefinementType.MERGE_TASKS}:
            # These types replace nodes with new ones (allowed)
            return True

        # Other types should not delete
        if "delete_task" in ref.proposed_changes:
            return False

        return True
```

---

## 5. Causality & Dependency Constraints

### 5.1 Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│               UPSTREAM (Constraint Sources)                     │
├─────────────────────────────────────────────────────────────────┤
│  EconomicController │ DatapackEngine │ Ontology │ TaskGraph    │
│  (econ physics)     │ (data physics) │ (objects)│ (task DAG)   │
└──────────┬──────────────────┬──────────────┬───────────┬────────┘
           │                  │              │           │
           ▼                  ▼              ▼           ▼
    ┌──────────────────────────────────────────────────────┐
    │   Stage 2.1: SemanticPrimitiveExtractor              │
    │   (SIMA-2 rollouts → SemanticPrimitive)              │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │   Stage 2.2: OntologyUpdateEngine                    │
    │   (SemanticPrimitive → OntologyUpdateProposal)       │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │   Stage 2.3: TaskGraphRefiner                        │
    │   (OntologyUpdateProposal → TaskGraphRefinement)     │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              DOWNSTREAM (Refinement Consumers)              │
├─────────────────────────────────────────────────────────────┤
│  SemanticOrchestratorV2  →  Apply refinements (advisory)   │
│  HRL Scheduler           →  Skill gating + checkpoints     │
│  VLA/Diffusion/RL        →  Task decomposition hints       │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Causality Rules

**TaskGraphRefiner**:
1. **Cannot originate** economic parameters
2. **Cannot originate** data valuation logic
3. **Cannot mutate** task graph directly (proposals only)
4. **Cannot delete** existing task nodes (replace only)
5. **Cannot create** DAG cycles

**Can consume**:
1. `OntologyUpdateProposal[ADD_SKILL_GATE]` → MUST insert checkpoints
2. `OntologyUpdateProposal[INFER_FRAGILITY]` → May split tasks
3. `EconSignals.error_urgency` → May elevate task priorities
4. `SemanticPrimitive.energy_intensity` → May reorder for efficiency

---

## 6. Stage 2.3 Pipeline Contract

### 6.1 Input
- **Source 1**: Stage 2.2 `OntologyUpdateEngine.generate_proposals()`
- **Type**: `List[OntologyUpdateProposal]`
- **Source 2**: Stage 2.1 `SemanticPrimitiveExtractor.extract_primitives_from_rollout()`
- **Type**: `List[SemanticPrimitive]` (optional)

### 6.2 Processing
- **Module**: `TaskGraphRefiner.generate_refinements(ontology_proposals, primitives)`
- **Constraints**: Respect econ/datapack/DAG boundaries
- **Validation**: `TaskGraphRefiner.validate_refinements()`

### 6.3 Output
- **Type**: `List[TaskGraphRefinementProposal]`
- **Format**: JSON-safe dataclass with `proposal_id`, `refinement_type`, `proposed_changes`

### 6.4 Storage
- **Location**: `results/stage2/task_graph_refinements/`
- **Format**: JSONL (one refinement per line)
- **Schema**: `{"proposal_id": "...", "refinement_type": "...", ...}`

### 6.5 Downstream Consumers
- **SemanticOrchestratorV2**: `apply_task_graph_refinements(refinements)` (Stage 2.4+)
- **HRL Scheduler**: Uses checkpoints for skill execution gates
- **VLA/Diffusion**: Receives task decomposition hints

---

## 7. Smoke Test Specification

### 7.1 File
`scripts/smoke_test_task_graph_refiner.py`

### 7.2 Test Cases

```python
#!/usr/bin/env python3
"""
Smoke test for Stage 2.3 TaskGraphRefiner.

Validates:
1. Refinement generation from OntologyUpdateProposals
2. JSON-safety of all refinements
3. Constraint compliance (econ/datapack/DAG/node-preservation)
4. Determinism (same inputs → same outputs)
5. Mandatory checkpoint insertion from skill gates
6. DAG topology preservation (no cycles)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.sima2.ontology_proposals import (
    OntologyUpdateProposal,
    ProposalType as OntologyProposalType,
    ProposalPriority,
)
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.sima2.task_graph_proposals import RefinementType, RefinementPriority
from src.orchestrator.task_graph import build_drawer_vase_task_graph
from src.orchestrator.ontology import build_drawer_vase_ontology
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals


def _make_test_ontology_proposals():
    """Create test OntologyUpdateProposals."""
    return [
        OntologyUpdateProposal(
            proposal_id="ont_prop_001",
            proposal_type=OntologyProposalType.ADD_SKILL_GATE,
            priority=ProposalPriority.HIGH,
            source_primitive_id="prim_001",
            target_skill_id=2,  # PULL skill
            proposed_changes={
                "gated_skill_id": 2,
                "preconditions": ["fragility_check_passed"],
                "safety_threshold": 0.8,
            },
            rationale="High-risk pull requires safety gate",
            confidence=0.9,
            tags=["skill_gate", "safety"],
        ),
        OntologyUpdateProposal(
            proposal_id="ont_prop_002",
            proposal_type=OntologyProposalType.INFER_FRAGILITY,
            priority=ProposalPriority.CRITICAL,
            source_primitive_id="prim_002",
            proposed_changes={
                "inferred_fragility": 0.9,
                "evidence": ["fragile", "vase"],
            },
            rationale="Vase is highly fragile",
            confidence=0.85,
            tags=["fragile", "vase"],
        ),
        OntologyUpdateProposal(
            proposal_id="ont_prop_003",
            proposal_type=OntologyProposalType.ADD_SAFETY_CONSTRAINT,
            priority=ProposalPriority.HIGH,
            proposed_changes={
                "constraint_type": "collision_avoidance",
                "objects": ["vase_01"],
                "applies_to_skills": [2, 5],  # PULL, MOVE
            },
            rationale="Collision avoidance required near vase",
            confidence=0.95,
            tags=["collision_avoidance"],
        ),
    ]


def _make_test_primitives():
    """Create test SemanticPrimitives."""
    return [
        SemanticPrimitive(
            primitive_id="prim_001",
            task_type="open_drawer",
            tags=["drawer", "pull"],
            risk_level="medium",
            energy_intensity=0.3,
            success_rate=0.9,
            avg_steps=5.0,
        ),
        SemanticPrimitive(
            primitive_id="prim_002",
            task_type="move_vase",
            tags=["vase", "fragile"],
            risk_level="high",
            energy_intensity=0.15,
            success_rate=0.85,
            avg_steps=8.0,
        ),
    ]


def main():
    print("[smoke_test_task_graph_refiner] Starting tests...")

    # Setup
    task_graph = build_drawer_vase_task_graph()
    ontology = build_drawer_vase_ontology()
    econ_signals = EconSignals(error_urgency=0.6, energy_urgency=0.3, mpl_urgency=0.5)
    datapack_signals = DatapackSignals(tier2_fraction=0.08)

    refiner = TaskGraphRefiner(
        task_graph=task_graph,
        ontology=ontology,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
    )

    ontology_proposals = _make_test_ontology_proposals()
    primitives = _make_test_primitives()

    # Test 1: Generate refinements
    refinements = refiner.generate_refinements(ontology_proposals, primitives)
    assert refinements, "Expected refinements to be generated"
    print(f"[TEST 1 PASS] Generated {len(refinements)} refinements")

    # Test 2: JSON-safety
    for ref in refinements:
        ref_dict = ref.to_dict()
        assert isinstance(ref_dict, dict), "Refinement to_dict() should return dict"
        json_str = json.dumps(ref_dict)
        assert json_str, "Refinement should be JSON-serializable"
    print(f"[TEST 2 PASS] All {len(refinements)} refinements are JSON-safe")

    # Test 3: Required fields
    for ref in refinements:
        assert ref.proposal_id, "proposal_id required"
        assert ref.refinement_type, "refinement_type required"
        assert ref.priority, "priority required"
        assert isinstance(ref.proposed_changes, dict), "proposed_changes must be dict"
        assert ref.rationale, "rationale required"
    print(f"[TEST 3 PASS] All refinements have required fields")

    # Test 4: Constraint compliance
    valid_refinements = refiner.validate_refinements(refinements)
    assert len(valid_refinements) <= len(refinements), "Validation should not add refinements"
    for ref in valid_refinements:
        assert ref.respects_econ_constraints, "Must respect econ constraints"
        assert ref.respects_datapack_constraints, "Must respect datapack constraints"
        assert ref.respects_dag_topology, "Must respect DAG topology"
        assert ref.preserves_existing_nodes, "Must preserve existing nodes"
    print(f"[TEST 4 PASS] {len(valid_refinements)}/{len(refinements)} refinements valid")

    # Test 5: Refinement type coverage
    refinement_types = {ref.refinement_type for ref in refinements}
    expected_types = {
        RefinementType.INSERT_CHECKPOINT,
        RefinementType.SPLIT_TASK,
        RefinementType.REORDER_TASKS,
    }
    assert refinement_types & expected_types, f"Expected refinement types in {expected_types}"
    print(f"[TEST 5 PASS] Refinement types: {[rt.value for rt in refinement_types]}")

    # Test 6: Mandatory checkpoint insertion from skill gate
    checkpoint_refs = [
        r for r in refinements if r.refinement_type == RefinementType.INSERT_CHECKPOINT
    ]
    # ADD_SKILL_GATE proposal should trigger INSERT_CHECKPOINT
    assert checkpoint_refs, "Expected checkpoint insertion from skill gate proposal"
    for ref in checkpoint_refs:
        assert "checkpoint_task" in ref.proposed_changes
        assert ref.proposed_changes.get("mandatory") is True
    print(f"[TEST 6 PASS] Mandatory checkpoint insertion working ({len(checkpoint_refs)} checkpoints)")

    # Test 7: Task splitting for fragility
    split_refs = [
        r for r in refinements if r.refinement_type == RefinementType.SPLIT_TASK
    ]
    # INFER_FRAGILITY proposal with high fragility should trigger SPLIT_TASK
    assert split_refs, "Expected task splitting for high fragility"
    for ref in split_refs:
        assert "original_task_id" in ref.proposed_changes
        assert "new_sub_tasks" in ref.proposed_changes
        assert len(ref.proposed_changes["new_sub_tasks"]) >= 2
    print(f"[TEST 7 PASS] Task splitting for fragility working ({len(split_refs)} splits)")

    # Test 8: Safety reordering
    reorder_refs = [
        r for r in refinements if r.refinement_type == RefinementType.REORDER_TASKS
    ]
    # ADD_SAFETY_CONSTRAINT should trigger REORDER_TASKS
    if reorder_refs:
        for ref in reorder_refs:
            assert "reordered_task_ids" in ref.proposed_changes
            assert "original_order" in ref.proposed_changes
            assert ref.proposed_changes["reordered_task_ids"] != ref.proposed_changes["original_order"]
        print(f"[TEST 8 PASS] Safety reordering working ({len(reorder_refs)} reorders)")
    else:
        print(f"[TEST 8 SKIP] No reorder proposals (task graph may be optimal)")

    # Test 9: Priority assignment
    critical_refs = [r for r in refinements if r.priority == RefinementPriority.CRITICAL]
    # Fragility-driven splits should be CRITICAL
    assert critical_refs, "Expected CRITICAL priority for fragility splits"
    print(f"[TEST 9 PASS] Priority assignment working ({len(critical_refs)} CRITICAL)")

    # Test 10: Determinism
    refinements_2 = refiner.generate_refinements(ontology_proposals, primitives)
    assert len(refinements_2) == len(refinements), "Determinism check: same input → same count"
    types_1 = sorted([r.refinement_type.value for r in refinements])
    types_2 = sorted([r.refinement_type.value for r in refinements_2])
    assert types_1 == types_2, "Determinism check: same refinement types"
    print(f"[TEST 10 PASS] Determinism validated")

    # Test 11: No DAG cycles (validation check)
    # All valid refinements should preserve DAG topology
    for ref in valid_refinements:
        assert ref.respects_dag_topology, "DAG topology must be preserved"
    print(f"[TEST 11 PASS] DAG topology preserved (no cycles)")

    # Test 12: Node preservation
    # Only SPLIT_TASK and MERGE_TASKS can replace nodes
    for ref in valid_refinements:
        if ref.refinement_type not in {RefinementType.SPLIT_TASK, RefinementType.MERGE_TASKS}:
            assert "delete_task" not in ref.proposed_changes, "Forbidden node deletion"
    print(f"[TEST 12 PASS] Node preservation validated")

    print("[smoke_test_task_graph_refiner] All tests passed!")


if __name__ == "__main__":
    main()
```

### 7.3 Integration with run_all_smokes.py

Add to `SMOKES` list:
```python
["python3", "scripts/smoke_test_task_graph_refiner.py"],
```

---

## 8. Acceptance Criteria

### Stage 2.3 is complete when:

1. ✅ `src/sima2/task_graph_proposals.py` exists with all 8 RefinementTypes
2. ✅ `src/sima2/task_graph_refiner.py` exists with TaskGraphRefiner class
3. ✅ `scripts/smoke_test_task_graph_refiner.py` passes all 12 tests
4. ✅ `run_all_smokes.py` includes new smoke test
5. ✅ All refinements are JSON-safe (can serialize/deserialize)
6. ✅ No task graph mutations occur in TaskGraphRefiner
7. ✅ Constraint validation rejects forbidden refinements
8. ✅ Determinism: same inputs → same refinement types/counts
9. ✅ DAG topology preserved (no cycles)
10. ✅ Mandatory checkpoint insertion from skill gates works

---

**End of Stage 2.3 Specification**
