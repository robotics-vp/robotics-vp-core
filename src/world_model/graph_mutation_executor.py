"""Governed executor for runtime graph mutation proposals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.envs.primitive_inventory import EnvPrimitive, EnvPrimitiveInventory
from src.hrl.skill_graph import SkillGraph, SkillNode, SkillTransitionEdge
from src.world_model.semantic_feedback_packets import GraphMutationProposal


def _normalize_id(value: str) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return normalized


def _coerce_proposals(values: Optional[Iterable[Any]]) -> List[GraphMutationProposal]:
    proposals: List[GraphMutationProposal] = []
    for item in values or []:
        if isinstance(item, GraphMutationProposal):
            proposals.append(item)
        elif isinstance(item, Mapping):
            proposals.append(
                GraphMutationProposal(
                    proposal_id=str(item.get("proposal_id", "")),
                    action=str(item.get("action", "")),
                    target_ref=str(item.get("target_ref", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    rationale=str(item.get("rationale", "")),
                    source_refs=[str(ref) for ref in item.get("source_refs", []) or []],
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            )
    return proposals


@dataclass(frozen=True)
class GraphMutationExecutionRecord:
    proposal_id: str
    action: str
    target_ref: str
    status: str
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "action": self.action,
            "target_ref": self.target_ref,
            "status": self.status,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }


@dataclass
class GraphMutationExecutionResult:
    skill_graph: SkillGraph
    env_inventories: List[EnvPrimitiveInventory]
    records: List[GraphMutationExecutionRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_graph": self.skill_graph.to_dict(),
            "env_inventories": [item.to_dict() for item in self.env_inventories],
            "records": [item.to_dict() for item in self.records],
            "metadata": dict(self.metadata),
        }


class GovernedGraphMutationExecutor:
    """Applies bounded runtime graph mutations under explicit governance."""

    def __init__(self, *, min_confidence: float = 0.55) -> None:
        self.min_confidence = float(min_confidence)

    def execute(
        self,
        skill_graph: SkillGraph,
        env_inventories: Sequence[EnvPrimitiveInventory],
        proposals: Optional[Iterable[Any]],
        *,
        governance_traces: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> GraphMutationExecutionResult:
        graph = SkillGraph.from_dict(skill_graph.to_dict())
        inventories = [EnvPrimitiveInventory.from_dict(item.to_dict()) for item in env_inventories]
        records: List[GraphMutationExecutionRecord] = []
        blocked_targets = {
            str(trace.get("target_ref", "")).strip()
            for trace in (governance_traces or [])
            if str(trace.get("outcome", trace.get("status", ""))).lower() in {"veto", "blocked", "deny"}
        }
        for proposal in _coerce_proposals(proposals):
            if proposal.target_ref in blocked_targets:
                records.append(
                    GraphMutationExecutionRecord(
                        proposal_id=proposal.proposal_id,
                        action=proposal.action,
                        target_ref=proposal.target_ref,
                        status="blocked",
                        rationale="governance_veto",
                    )
                )
                continue
            if float(proposal.confidence) < self.min_confidence:
                records.append(
                    GraphMutationExecutionRecord(
                        proposal_id=proposal.proposal_id,
                        action=proposal.action,
                        target_ref=proposal.target_ref,
                        status="deferred",
                        rationale="confidence_below_threshold",
                    )
                )
                continue
            if proposal.action == "add_provisional_skill":
                skill_id = proposal.target_ref
                if not skill_id.startswith(("hrl:", "sima:", "vla:", "stage2:", "runtime:")):
                    skill_id = f"runtime:{_normalize_id(skill_id.replace('skill:', ''))}"
                if not any(node.skill_id == skill_id for node in graph.nodes):
                    graph.nodes.append(
                        SkillNode(
                            skill_id=skill_id,
                            skill_family="runtime",
                            label=proposal.target_ref,
                            description=proposal.rationale,
                            metadata={"provisional": True, **dict(proposal.metadata or {})},
                        )
                    )
                task_id = str(proposal.metadata.get("task_id", "runtime_discovered"))
                for source_ref in proposal.source_refs:
                    if source_ref.startswith(("hrl:", "sima:", "vla:", "stage2:", "runtime:")):
                        graph.transitions.append(
                            SkillTransitionEdge(
                                from_skill=source_ref,
                                to_skill=skill_id,
                                task_id=task_id,
                                metadata={"provisional": True, "proposal_id": proposal.proposal_id},
                            )
                        )
                records.append(
                    GraphMutationExecutionRecord(
                        proposal_id=proposal.proposal_id,
                        action=proposal.action,
                        target_ref=skill_id,
                        status="applied",
                        rationale=proposal.rationale,
                    )
                )
                continue
            if proposal.action == "add_provisional_affordance":
                primitive_id = _normalize_id(proposal.target_ref.replace("prim:", "").replace("affordance:", ""))
                if not primitive_id:
                    primitive_id = f"runtime_affordance_{proposal.proposal_id}"
                for inventory in inventories:
                    if not inventory.has_primitive(primitive_id):
                        inventory.primitives.append(
                            EnvPrimitive(
                                primitive_id=primitive_id,
                                category=str(proposal.metadata.get("category", "manipulation")),
                                label=proposal.target_ref,
                                description=proposal.rationale,
                                metadata={"provisional": True, **dict(proposal.metadata or {})},
                            )
                        )
                records.append(
                    GraphMutationExecutionRecord(
                        proposal_id=proposal.proposal_id,
                        action=proposal.action,
                        target_ref=primitive_id,
                        status="applied",
                        rationale=proposal.rationale,
                    )
                )
                continue
            if proposal.action == "add_object_family":
                object_family = _normalize_id(proposal.target_ref.replace("obj:", ""))
                for inventory in inventories:
                    if object_family and object_family not in inventory.object_families:
                        inventory.object_families.append(object_family)
                records.append(
                    GraphMutationExecutionRecord(
                        proposal_id=proposal.proposal_id,
                        action=proposal.action,
                        target_ref=object_family,
                        status="applied",
                        rationale=proposal.rationale,
                    )
                )
                continue
            if proposal.action == "update_relationship":
                task_id = str(proposal.metadata.get("task_id", "runtime_discovered"))
                source_skill = next(
                    (ref for ref in proposal.source_refs if ref.startswith(("hrl:", "sima:", "vla:", "stage2:", "runtime:"))),
                    "",
                )
                target_skill = proposal.target_ref if proposal.target_ref.startswith(("hrl:", "sima:", "vla:", "stage2:", "runtime:")) else ""
                if source_skill and target_skill:
                    graph.transitions.append(
                        SkillTransitionEdge(
                            from_skill=source_skill,
                            to_skill=target_skill,
                            task_id=task_id,
                            metadata={"provisional": True, "proposal_id": proposal.proposal_id},
                        )
                    )
                    status = "applied"
                else:
                    status = "deferred"
                records.append(
                    GraphMutationExecutionRecord(
                        proposal_id=proposal.proposal_id,
                        action=proposal.action,
                        target_ref=proposal.target_ref,
                        status=status,
                        rationale=proposal.rationale,
                    )
                )
                continue
            records.append(
                GraphMutationExecutionRecord(
                    proposal_id=proposal.proposal_id,
                    action=proposal.action,
                    target_ref=proposal.target_ref,
                    status="deferred",
                    rationale="manual_review_required",
                )
            )
        return GraphMutationExecutionResult(
            skill_graph=graph,
            env_inventories=inventories,
            records=records,
            metadata={
                "applied_count": sum(1 for item in records if item.status == "applied"),
                "blocked_count": sum(1 for item in records if item.status == "blocked"),
                "deferred_count": sum(1 for item in records if item.status == "deferred"),
            },
        )


__all__ = [
    "GovernedGraphMutationExecutor",
    "GraphMutationExecutionRecord",
    "GraphMutationExecutionResult",
]
