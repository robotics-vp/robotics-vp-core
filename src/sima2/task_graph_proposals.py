"""
Schemas for task graph refinement proposals (Stage 2.3).

These are advisory-only and JSON-safe; they do not mutate the task graph.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple


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

    def validate(self) -> Tuple[bool, List[str]]:
        """Light validation of required fields and forbidden keys."""
        errors: List[str] = []
        if not self.proposal_id:
            errors.append("proposal_id is required")
        if not isinstance(self.refinement_type, RefinementType):
            errors.append("refinement_type must be RefinementType")
        if not isinstance(self.priority, RefinementPriority):
            errors.append("priority must be RefinementPriority")
        if not isinstance(self.proposed_changes, dict):
            errors.append("proposed_changes must be a dict")
        if not self.rationale:
            errors.append("rationale is required")

        forbidden_keys = {
            "price_per_unit",
            "damage_cost",
            "tier",
            "data_premium",
            "w_econ",
            "sampling_weight",
            "reward_vector",
            "objective_weights",
        }
        if any(k in self.proposed_changes for k in forbidden_keys):
            errors.append("proposed_changes contains forbidden economic/datapack fields")

        if self.refinement_type not in {RefinementType.SPLIT_TASK, RefinementType.MERGE_TASKS}:
            if "delete_task" in self.proposed_changes:
                errors.append("delete_task not allowed for this refinement type")

        return (len(errors) == 0, errors)
