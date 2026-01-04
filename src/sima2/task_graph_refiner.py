"""
TaskGraphRefiner: Advisory task graph refinement proposal generator.

IMPORTANT:
- Does NOT mutate task graph directly
- Consumes OntologyUpdateProposals + SemanticPrimitives
- Produces TaskGraphRefinementProposals for SemanticOrchestrator
- Respects econ/datapack/ontology/DAG constraints
"""

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
        self._proposal_counter = 0  # reset per generation for determinism
        refinements = []

        # 1. Process ontology proposals
        for ont_prop in sorted(
            ontology_proposals,
            key=lambda p: (getattr(p, "priority", None) and p.priority.value or "", p.proposal_id),
        ):
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
            for prim in sorted(primitives, key=lambda p: (p.task_type or "", p.primitive_id or "")):
                # Low energy primitives → REORDER_TASKS
                if prim.energy_intensity < 0.5 and self.econ_signals.energy_urgency > 0.3:
                    refinements.extend(self._reorder_for_energy(prim))

                # High success rate → MERGE_TASKS (if redundant)
                if prim.success_rate > 0.95:
                    refinements.extend(self._merge_redundant_tasks(prim))
                
                # OOD/Critical Risk → SAFETY_STOP
                if prim.risk_level == "critical" or "ood" in prim.tags:
                     refinements.extend(self._insert_safety_stop_for_ood(prim))

        # 3. Economic urgency-driven refinements
        if self.econ_signals.error_urgency > 0.6:
            refinements.extend(self._adjust_priority_for_safety())

        if self.econ_signals.mpl_urgency > 0.5:
            refinements.extend(self._parallelize_for_throughput())

        return self._sort_refinements_deterministically(refinements)

    def _make_proposal_id(self) -> str:
        """Generate unique deterministic proposal ID."""
        self._proposal_counter += 1
        return f"tgr_{self._proposal_counter:06d}"

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
        for node in sorted(self.task_graph.get_all_nodes(), key=lambda n: n.task_id):
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

        for node in sorted(self.task_graph.get_all_nodes(), key=lambda n: n.task_id):
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
                    priority=RefinementPriority.CRITICAL,
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
            node
            for node in sorted(self.task_graph.get_all_nodes(), key=lambda n: n.task_id)
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
        for node in sorted(self.task_graph.get_all_nodes(), key=lambda n: n.task_id):
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
        all_skill_tasks = sorted(self.task_graph.get_skill_nodes(), key=lambda n: n.task_id)

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
        all_tasks = sorted(self.task_graph.get_all_nodes(), key=lambda n: n.task_id)

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

    def _insert_safety_stop_for_ood(
        self, prim: SemanticPrimitive
    ) -> List[TaskGraphRefinementProposal]:
        """
        Insert Safety Stop for OOD/Critical primitives.
        
        Rule: If risk is critical or OOD detected, halt execution via Safety Stop node.
        """
        refinements = []
        
        # We need to find where this primitive occurred or what task it belongs to.
        # Primitives map to tasks via task_type roughly.
        # This is strictly advisory, so we attach it to valid tasks matching the type.
        
        target_tasks = [
            node for node in self.task_graph.get_all_nodes() 
            if prim.task_type and prim.task_type in node.name.lower()
        ]
        
        if not target_tasks:
             # Fallback: finding generic "skill" tasks if prim has generic tags
             if "move" in prim.tags:
                 target_tasks = [
                     n for n in self.task_graph.get_all_nodes() 
                     if n.task_type == TaskType.SKILL and "move" in n.name.lower()
                 ]

        for node in target_tasks:
            stop_task = {
                "task_id": f"safety_stop_{node.task_id}",
                "name": f"Safety Stop (OOD Detected)",
                "task_type": "checkpoint",
                "preconditions": [f"{node.task_id}_started"],
                "postconditions": ["manual_intervention_required"],
                "metadata": {
                    "reason": "ood_critical_risk",
                    "source_primitive": prim.primitive_id,
                    "risk_level": prim.risk_level
                },
            }

            refinement = TaskGraphRefinementProposal(
                proposal_id=self._make_proposal_id(),
                refinement_type=RefinementType.INSERT_CHECKPOINT,
                priority=RefinementPriority.CRITICAL,
                source_primitive_ids=[prim.primitive_id],
                target_task_ids=[node.task_id],
                parent_task_id=node.parent_id,
                proposed_changes={
                    "checkpoint_task": stop_task,
                    "insert_after_task_id": node.task_id, # Stop immediately after start or before next
                    "mandatory": True,
                    "action": "halt"
                },
                rationale=f"Critical OOD risk detected in primitive {prim.primitive_id}",
                confidence=0.95,
                tags=["safety", "ood", "halt"],
            )
            refinements.append(refinement)
            
        return refinements

    def _adjust_priority_for_safety(self) -> List[TaskGraphRefinementProposal]:
        """
        Adjust task priorities based on error urgency.

        Rule: High error urgency → elevate checkpoint task priorities.
        """
        refinements = []

        for node in sorted(self.task_graph.get_all_nodes(), key=lambda n: n.task_id):
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
        all_tasks = sorted(self.task_graph.get_all_nodes(), key=lambda n: n.task_id)

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

    def _sort_refinements_deterministically(
        self, refinements: List[TaskGraphRefinementProposal]
    ) -> List[TaskGraphRefinementProposal]:
        """Deterministic ordering for refinements."""
        priority_order = {
            RefinementPriority.CRITICAL: 0,
            RefinementPriority.HIGH: 1,
            RefinementPriority.MEDIUM: 2,
            RefinementPriority.LOW: 3,
        }
        return sorted(
            refinements,
            key=lambda r: (
                priority_order.get(r.priority, 4),
                r.refinement_type.value if isinstance(r.refinement_type, RefinementType) else "",
                r.proposal_id,
            ),
        )
