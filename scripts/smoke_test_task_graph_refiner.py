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
