#!/usr/bin/env python3
"""
End-to-end smoke for Stage 2 pipeline (SemanticPrimitiveExtractor → OntologyUpdateEngine → TaskGraphRefiner).

Validates determinism, validation APIs, constraint safety, and coverage of key proposal/refinement types.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.datapack_engine import DatapackSignals
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.ontology import build_drawer_vase_ontology
from src.orchestrator.task_graph import build_drawer_vase_task_graph
from src.sima2.ontology_proposals import ProposalPriority, ProposalType
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.semantic_primitive_extractor import extract_primitives_from_rollout
from src.sima2.task_graph_proposals import RefinementType
from src.sima2.task_graph_refiner import TaskGraphRefiner


def _make_rollouts():
    """Synthetic fragile drawer/vase rollouts."""
    return [
        {
            "task_type": "open_drawer",
            "events": [
                {"action": "grasp", "object": "drawer_handle", "tags": ["drawer", "grasp"], "success": True},
                {"action": "pull", "object": "drawer", "tags": ["drawer", "open"], "energy_intensity": 0.2},
            ],
            "metrics": {"steps": 2, "success": True, "energy_used": 0.4},
        },
        {
            "task_type": "move_fragile_vase",
            "events": [
                {"action": "lift", "object": "vase", "tags": ["vase", "fragile"], "success_rate": 0.85},
                {"action": "place", "object": "table", "tags": ["table", "place"], "energy_intensity": 0.1},
            ],
            "metrics": {"steps": 3, "success": True},
        },
    ]


def _run_stage2_pipeline():
    """Run primitives → ontology proposals → task graph refinements."""
    rollouts = _make_rollouts()
    primitives = []
    for ro in rollouts:
        primitives.extend(extract_primitives_from_rollout(ro))
    assert primitives, "Expected primitives from rollouts"
    for prim in primitives:
        ok, errors = prim.validate()
        assert ok, f"Primitive validation failed: {errors}"

    ontology = build_drawer_vase_ontology()
    task_graph = build_drawer_vase_task_graph()
    econ_signals = EconSignals(error_urgency=0.6, energy_urgency=0.3, mpl_urgency=0.5, damage_cost_total=75.0)
    datapack_signals = DatapackSignals(tier2_fraction=0.08)

    oe = OntologyUpdateEngine(
        ontology=ontology,
        task_graph=task_graph,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
    )
    proposals = oe.generate_proposals(primitives)
    assert proposals, "Expected ontology proposals"
    for prop in proposals:
        ok, errors = prop.validate()
        assert ok, f"Ontology proposal validation failed: {errors}"

    refiner = TaskGraphRefiner(
        task_graph=task_graph,
        ontology=ontology,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
    )
    refinements = refiner.generate_refinements(proposals, primitives)
    assert refinements, "Expected task graph refinements"
    for ref in refinements:
        ok, errors = ref.validate()
        assert ok, f"Refinement validation failed: {errors}"

    return primitives, proposals, refinements


def _assert_constraint_safety(items, forbidden_keys):
    """Ensure proposed_changes do not contain forbidden keys."""
    for item in items:
        if hasattr(item, "proposed_changes"):
            assert not any(k in item.proposed_changes for k in forbidden_keys), f"Forbidden key in {item}"


def main():
    print("[smoke_test_stage2_e2e_pipeline] Starting Stage 2 E2E smoke...")
    primitives_1, proposals_1, refinements_1 = _run_stage2_pipeline()
    primitives_2, proposals_2, refinements_2 = _run_stage2_pipeline()

    # Determinism across full pipeline
    assert json.dumps([p.__dict__ for p in primitives_1]) == json.dumps(
        [p.__dict__ for p in primitives_2]
    ), "Primitive outputs are not deterministic"
    assert json.dumps([p.to_dict() for p in proposals_1]) == json.dumps(
        [p.to_dict() for p in proposals_2]
    ), "Ontology proposals are not deterministic"
    assert json.dumps([r.to_dict() for r in refinements_1]) == json.dumps(
        [r.to_dict() for r in refinements_2]
    ), "Task graph refinements are not deterministic"
    print("[TEST 1 PASS] Determinism validated across pipeline")

    # Coverage: ontology proposals
    proposal_types = {p.proposal_type for p in proposals_1}
    required_proposal_types = {
        ProposalType.ADD_AFFORDANCE,
        ProposalType.INFER_FRAGILITY,
        ProposalType.ADD_SAFETY_CONSTRAINT,
        ProposalType.ADD_SKILL_GATE,
    }
    assert required_proposal_types.issubset(proposal_types), "Missing required ontology proposal types"
    print(f"[TEST 2 PASS] Ontology proposal coverage: {[pt.value for pt in proposal_types]}")

    # Coverage: refinements
    refinement_types = {r.refinement_type for r in refinements_1}
    assert RefinementType.INSERT_CHECKPOINT in refinement_types, "Expected checkpoint refinements"
    assert RefinementType.SPLIT_TASK in refinement_types, "Expected fragility-driven splits"
    print(f"[TEST 3 PASS] Refinement coverage: {[rt.value for rt in refinement_types]}")

    # Constraint safety
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
    _assert_constraint_safety(proposals_1, forbidden_keys)
    _assert_constraint_safety(refinements_1, forbidden_keys)
    for ref in refinements_1:
        assert "delete_task" not in ref.proposed_changes, "Refinement must not delete nodes"
    print("[TEST 4 PASS] Constraint safety validated")

    print("[smoke_test_stage2_e2e_pipeline] All tests passed!")


if __name__ == "__main__":
    main()
