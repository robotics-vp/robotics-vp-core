#!/usr/bin/env python3
"""
Smoke test for Stage 2.2 OntologyUpdateEngine.

Validates:
1. Proposal generation from SemanticPrimitives
2. JSON-safety of all proposals
3. Constraint compliance (econ/datapack/task-graph)
4. Determinism (same inputs → same outputs)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.datapack_engine import DatapackSignals
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.ontology import build_drawer_vase_ontology
from src.sima2.ontology_proposals import ProposalPriority, ProposalType
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.semantic_primitive_extractor import SemanticPrimitive


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
