#!/usr/bin/env python3
"""
Smoke test for SIMA-2 segmentation/ontology contract.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.sima2.client import Sima2Client
from src.sima2.segmentation_engine import SegmentationEngine, Segment
from src.sima2.semantic_primitive_extractor import SemanticPrimitiveExtractor, SemanticPrimitive
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.sima2.semantic_tag_propagator import SemanticTagPropagator
from src.orchestrator.task_graph import TaskGraph, TaskNode, TaskType
from src.orchestrator.ontology import EnvironmentOntology
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals


def _seed_rollouts(client: Sima2Client):
    base = client.run_episode({"task_id": "drawer_vase", "episode_index": 0})
    # Inject a failure + recovery to exercise recovery tags.
    base["primitives"][1]["status"] = "failure"
    base["primitives"].append(
        {
            "timestep": base["primitives"][-1]["timestep"] + 3,
            "object": "drawer",
            "action": "recover_pull",
            "risk": "low",
            "status": "recovery",
        }
    )
    alt = client.run_episode({"task_id": "dish_place", "episode_index": 1})
    return [base, alt]


def _primitive_models(records):
    return [
        SemanticPrimitive(
            primitive_id=p.get("primitive_id", "prim"),
            task_type=p.get("task_type", p.get("task", "unknown")),
            tags=p.get("tags", []),
            risk_level=p.get("risk_level", "low"),
            energy_intensity=float(p.get("energy_intensity", 0.0)),
            success_rate=float(p.get("success_rate", 1.0)),
            avg_steps=float(p.get("avg_steps", 1.0)),
            source=p.get("source", "sima2"),
            provenance=p.get("provenance", {}),
        )
        for p in records
    ]


def main():
    client = Sima2Client(task_id="drawer_vase")
    rollouts = _seed_rollouts(client)

    seg_engine = SegmentationEngine(
        segmentation_config={"min_segment_length": 1, "max_idle_gap": 2, "max_segment_length": 64, "risk_jump_delta": 1, "allow_risk_jumps": True}
    )
    assert seg_engine.builder.min_segment_length == seg_engine.segmentation_config.get("min_segment_length")

    segmented_rollouts = []
    segment_index = {}
    boundary_index = {}
    for rollout in rollouts:
        result = seg_engine.segment_rollout(rollout)
        segmented_rollouts.append(result["rollout"])
        segment_index[result["rollout"]["episode_id"]] = result["segments"]
        boundary_index[result["rollout"]["episode_id"]] = result["segment_boundaries"]
        assert result["segments"], "Expected segments to be generated"

    primitive_extractor = SemanticPrimitiveExtractor()
    primitives = primitive_extractor.extract(segmented_rollouts)
    primitive_ids = {p["primitive_id"] for p in primitives}
    assert primitive_ids, "Expected primitives from segmentation events"

    for seg_list in segment_index.values():
        assert all(isinstance(seg, Segment) for seg in seg_list)
        for seg in seg_list:
            assert seg.segment_id in primitive_ids, "Segment missing primitive coverage"

    primitive_models = _primitive_models(primitives)
    ontology = EnvironmentOntology(ontology_id="sima2_contract", name="sima2_contract")
    task_graph = TaskGraph(TaskNode(task_id="root", name="root", task_type=TaskType.ROOT))
    econ_signals = EconSignals()
    datapack_signals = DatapackSignals()

    ontology_engine = OntologyUpdateEngine(ontology=ontology, task_graph=task_graph, econ_signals=econ_signals, datapack_signals=datapack_signals)
    ontology_props = ontology_engine.generate_proposals(primitive_models)
    for prop in ontology_props:
        assert prop.source_primitive_id in primitive_ids, "Ontology proposal references unknown primitive"

    refiner = TaskGraphRefiner(task_graph=task_graph, ontology=ontology, econ_signals=econ_signals, datapack_signals=datapack_signals)
    task_graph_props = refiner.generate_refinements(ontology_props, primitives=primitive_models)
    ontology_prop_ids = {p.proposal_id for p in ontology_props}
    for prop in task_graph_props:
        for src_id in getattr(prop, "source_ontology_proposal_ids", []) or []:
            assert src_id in ontology_prop_ids, "Task graph proposal references unknown ontology proposal"

    propagator = SemanticTagPropagator()
    econ_outputs = {p["episode_id"]: {"novelty_delta": 0.0} for p in primitives}
    enrichment = propagator.generate_proposals(segmented_rollouts, ontology_props, task_graph_props, econ_outputs)

    valid_episode_ids = set(segment_index.keys())
    valid_segment_ids = {seg.segment_id for segs in segment_index.values() for seg in segs}
    for proposal in enrichment:
        assert proposal.episode_id in valid_episode_ids, "Enrichment references unknown episode"
        for b in proposal.segment_boundary_tags:
            assert b.segment_id in valid_segment_ids, "Boundary references unknown segment"
        for rec in proposal.recovery_pattern_tags:
            segref = getattr(rec, "segment_id", None)
            if segref:
                assert segref in valid_segment_ids, "Recovery tag references unknown segment"

    print("[smoke_test_sima2_contract] PASS")


if __name__ == "__main__":
    main()
