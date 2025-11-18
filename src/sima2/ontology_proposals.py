"""
Schema definitions for ontology update proposals (advisory-only).

These proposals are JSON-safe and deterministic; they do not mutate ontology
state. SemanticOrchestrator decides whether/how to apply them.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


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
