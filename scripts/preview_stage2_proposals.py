#!/usr/bin/env python3
"""
Developer preview for Stage 2 proposals/refinements.

Runs Stage 2 pipeline on a synthetic fragile drawer/vase scenario and prints concise summaries.
Also writes JSON dumps to results/stage2_preview/ for inspection.
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.datapack_engine import DatapackSignals
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.ontology import build_drawer_vase_ontology
from src.orchestrator.task_graph import build_drawer_vase_task_graph
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.semantic_primitive_extractor import extract_primitives_from_rollout
from src.sima2.task_graph_refiner import TaskGraphRefiner


def _make_rollouts():
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


def _run_pipeline():
    rollouts = _make_rollouts()
    primitives = []
    for ro in rollouts:
        primitives.extend(extract_primitives_from_rollout(ro))

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

    refiner = TaskGraphRefiner(
        task_graph=task_graph,
        ontology=ontology,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
    )
    refinements = refiner.generate_refinements(proposals, primitives)

    return primitives, proposals, refinements


def _print_summary(primitives, proposals, refinements):
    print("=== Stage 2 Preview ===")
    print(f"Semantic Primitives ({len(primitives)}):")
    for prim in primitives:
        print(f"  - {prim.primitive_id}: task={prim.task_type}, risk={prim.risk_level}, tags={prim.tags}")

    grouped_props = defaultdict(list)
    for prop in proposals:
        grouped_props[prop.proposal_type.value].append(prop)
    print(f"\nOntology Proposals ({len(proposals)}):")
    for ptype, props in grouped_props.items():
        print(f"  - {ptype} ({len(props)}):")
        for prop in props:
            key_fields = prop.proposed_changes
            print(f"      id={prop.proposal_id}, priority={prop.priority.value}, changes={key_fields}")

    grouped_refs = defaultdict(list)
    for ref in refinements:
        grouped_refs[ref.refinement_type.value].append(ref)
    print(f"\nTask Graph Refinements ({len(refinements)}):")
    for rtype, refs in grouped_refs.items():
        print(f"  - {rtype} ({len(refs)}):")
        for ref in refs:
            print(f"      id={ref.proposal_id}, priority={ref.priority.value}, targets={ref.target_task_ids}")


def _write_outputs(primitives, proposals, refinements):
    out_dir = os.path.join("results", "stage2_preview")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage2_primitives.json"), "w") as f:
        json.dump([p.__dict__ for p in primitives], f, indent=2)
    with open(os.path.join(out_dir, "stage2_ontology_proposals.json"), "w") as f:
        json.dump([p.to_dict() for p in proposals], f, indent=2)
    with open(os.path.join(out_dir, "stage2_task_graph_refinements.json"), "w") as f:
        json.dump([r.to_dict() for r in refinements], f, indent=2)
    print(f"\nWrote preview JSON to {out_dir}/")


def main():
    primitives, proposals, refinements = _run_pipeline()
    _print_summary(primitives, proposals, refinements)
    _write_outputs(primitives, proposals, refinements)


if __name__ == "__main__":
    main()
