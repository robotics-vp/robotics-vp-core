#!/usr/bin/env python3
"""
End-to-end Stage 2 pipeline over SIMA-2 rollouts (stubbed, deterministic).

Generates primitives, ontology proposals, task graph refinements, and semantic tags.
Outputs are JSONL; ontology is not mutated.
"""
import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.sima2.semantic_primitive_extractor import SemanticPrimitiveExtractor, SemanticPrimitive
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.sima2.semantic_tag_propagator import SemanticTagPropagator
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType, ProposalPriority
from src.sima2.task_graph_proposals import TaskGraphRefinementProposal, RefinementType, RefinementPriority
from src.sima2.segmentation_engine import SegmentationEngine
from src.orchestrator.ontology import EnvironmentOntology
from src.orchestrator.task_graph import TaskGraph, TaskNode, TaskType
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals


def _load_rollouts(path: Path):
    rollouts = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rollouts.append(json.loads(line))
    return rollouts


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            obj = rec.to_dict() if hasattr(rec, "to_dict") else rec
            f.write(json.dumps(obj, sort_keys=True))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run Stage2 SIMA-2 pipeline on rollouts.")
    parser.add_argument("--rollouts-path", type=str, required=True)
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--output-dir", type=str, default="results/stage2")
    args = parser.parse_args()

    rollouts = _load_rollouts(Path(args.rollouts_path))

    seg_engine = SegmentationEngine()
    segmented_rollouts = []
    segmented_boundaries = []
    seg_segments = []
    for rollout in rollouts:
        seg_out = seg_engine.segment_rollout(rollout)
        segmented_rollouts.append(seg_out["rollout"])
        segmented_boundaries.extend([getattr(b, "to_dict", lambda: b)() for b in seg_out["segment_boundaries"]])
        seg_segments.extend([s.to_dict() if hasattr(s, "to_dict") else s for s in seg_out["segments"]])

    primitive_extractor = SemanticPrimitiveExtractor()
    primitives = primitive_extractor.extract(segmented_rollouts)

    ontology = EnvironmentOntology(ontology_id="sima2_stub", name="sima2_stub")
    task_graph = TaskGraph(TaskNode(task_id="root", name="root", task_type=TaskType.ROOT))
    econ_signals = EconSignals()
    datapack_signals = DatapackSignals()

    primitive_models = [
        SemanticPrimitive(
            primitive_id=p.get("primitive_id", f"{p.get('task_type')}_prim"),
            task_type=p.get("task_type", p.get("task", "unknown")),
            tags=p.get("tags", []),
            risk_level=p.get("risk_level", "low"),
            energy_intensity=float(p.get("energy_intensity", 0.0)),
            success_rate=float(p.get("success_rate", 1.0)),
            avg_steps=float(p.get("avg_steps", 1.0)),
            source="sima2",
        )
        for p in primitives
    ]

    ontology_engine = OntologyUpdateEngine(ontology=ontology, task_graph=task_graph, econ_signals=econ_signals, datapack_signals=datapack_signals)
    ontology_proposals = ontology_engine.generate_proposals(primitive_models)
    if not ontology_proposals:
        ontology_proposals = [
            OntologyUpdateProposal(
                proposal_id=f"sima2_prop_{idx}",
                proposal_type=ProposalType.ADD_AFFORDANCE,
                priority=ProposalPriority.MEDIUM,
                source_primitive_id=prim.primitive_id,
                source="sima2_stub",
                proposed_changes={"tags": prim.tags, "risk_level": prim.risk_level},
                rationale="stubbed_sima2",
            )
            for idx, prim in enumerate(primitive_models)
        ]

    task_graph_refiner = TaskGraphRefiner(task_graph=task_graph, ontology=ontology, econ_signals=econ_signals, datapack_signals=datapack_signals)
    task_graph_proposals = task_graph_refiner.generate_refinements(ontology_proposals, primitives=primitive_models)
    if not task_graph_proposals:
        task_graph_proposals = [
            TaskGraphRefinementProposal(
                proposal_id=f"sima2_refine_{idx}",
                refinement_type=RefinementType.SPLIT_TASK,
                priority=RefinementPriority.MEDIUM,
                target_task_ids=[prim.primitive_id],
                metadata={"tags": prim.tags},
            )
            for idx, prim in enumerate(primitive_models)
        ]

    propagator = SemanticTagPropagator()
    econ_outputs = {p.get("episode_id"): {"novelty_delta": 0.0} for p in primitives}
    tags = propagator.generate_proposals(segmented_rollouts, ontology_proposals, task_graph_proposals, econ_outputs)

    out_dir = Path(args.output_dir)
    _write_jsonl(out_dir / "sima2_primitives.jsonl", primitives)
    _write_jsonl(out_dir / "sima2_ontology_proposals.jsonl", ontology_proposals)
    _write_jsonl(out_dir / "sima2_task_refinements.jsonl", task_graph_proposals)
    _write_jsonl(out_dir / "sima2_semantic_tags.jsonl", tags)
    _write_jsonl(out_dir / "sima2_segments.jsonl", seg_segments)
    _write_jsonl(out_dir / "sima2_segment_boundaries.jsonl", segmented_boundaries)

    print(f"[run_stage2_sima2_pipeline] primitives={len(primitives)}, ontology_proposals={len(ontology_proposals)}, task_refinements={len(task_graph_proposals)}, tags={len(tags)}")


if __name__ == "__main__":
    main()
