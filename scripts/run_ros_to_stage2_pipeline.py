#!/usr/bin/env python3
"""
ROS → Stage 2 pipeline runner.

Steps:
1) Ingest ROS JSON/bag via ROSBridgeIngestor → RawRollout
2) Segment with SegmentationEngine (heuristic)
3) Extract primitives → Ontology/TaskGraph proposals
4) Propagate semantic tags (OOD/Recovery flag-gated)
5) Compute EconCorrelator trust matrix from tags + econ stubs

Outputs JSONL artifacts under results/ros_stage2 by default.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.econ_correlator_impl import EconCorrelator
from src.ingestion.ros_bridge import ROSBridgeIngestor
from src.ingestion.rollout_types import RawRollout
from src.orchestrator.datapack_engine import DatapackSignals
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.ontology import EnvironmentOntology
from src.orchestrator.task_graph import TaskGraph, TaskNode, TaskType
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalPriority, ProposalType
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.segmentation_engine import SegmentationEngine
from src.sima2.semantic_primitive_extractor import SemanticPrimitiveExtractor, SemanticPrimitive
from src.sima2.semantic_tag_propagator import SemanticTagPropagator
from src.sima2.task_graph_proposals import RefinementPriority, RefinementType, TaskGraphRefinementProposal
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.utils.json_safe import to_json_safe


def _write_jsonl(path: Path, records: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            obj = rec.to_dict() if hasattr(rec, "to_dict") else rec
            f.write(json.dumps(to_json_safe(obj), sort_keys=True))
            f.write("\n")


def _primitives_from_proprio(rollout: RawRollout) -> List[Dict[str, Any]]:
    primitives: List[Dict[str, Any]] = []
    for pf in rollout.proprio_frames:
        vel_mag = sum(abs(v) for v in (pf.joint_velocities or []))
        contact_force = max((abs(c) for c in (pf.contact_sensors or [])), default=0.0)
        primitives.append(
            {
                "timestep": int(pf.timestep),
                "gripper_width": float((pf.joint_positions or [0.05])[0]) if pf.joint_positions else 0.05,
                "ee_velocity": float(vel_mag),
                "contact": bool(contact_force > 0.0),
                "force": float(contact_force),
            }
        )
    return primitives


def _econ_from_rollout(rollout: RawRollout) -> Dict[str, float]:
    energy_wh = sum(pf.energy_estimate_Wh for pf in rollout.proprio_frames)
    damage = sum(abs(c) for pf in rollout.proprio_frames for c in (pf.contact_sensors or []))
    mpl = float(len(rollout.action_frames))
    return {"energy_wh": float(energy_wh), "damage": float(damage), "mpl": mpl, "success": True}


def run_pipeline(log_path: Path, output_dir: Path, inject_anomaly: bool = False) -> Dict[str, Any]:
    ingestor = ROSBridgeIngestor(output_root=str(output_dir / "ingestion"))
    rollout = ingestor.ingest(str(log_path), task_id="real_ros", backend_id="ros_bridge")
    rollout_dict = rollout.to_dict()
    rollout_dict.setdefault("task", rollout.task_id)
    rollout_dict.setdefault("task_type", rollout.task_id)
    rollout_dict.setdefault("source", "ros_bridge")
    rollout_dict.setdefault("primitives", _primitives_from_proprio(rollout))

    seg_engine = SegmentationEngine(segmentation_config={"use_heuristic_segmenter": True, "temporal_decay_window": 1})
    seg_out = seg_engine.segment_rollout(rollout_dict)
    segmented_rollout = seg_out["rollout"]
    segments = segmented_rollout.get("segments", [])

    if inject_anomaly and segments:
        for idx, seg in enumerate(segments):
            md = dict(seg.get("metadata", {}) or {})
            md.setdefault("embedding_distance", 1.2)
            md.setdefault("force", 15.0)
            md.setdefault("joint_velocity", [2.0])
            md.setdefault("duration", int(md.get("duration", seg.get("end_t", 1) - seg.get("start_t", 0))))
            if idx == 0:
                md.setdefault("failure_observed", True)
            seg["metadata"] = md
        if len(segments) == 1:
            seg = segments[0]
            segments.append(
                {
                    "segment_id": f"{rollout.episode_id}_recovery",
                    "episode_id": rollout.episode_id,
                    "label": "recovery",
                    "start_t": int(seg.get("end_t", seg.get("start_t", 0) + 1)),
                    "end_t": int(seg.get("end_t", seg.get("start_t", 0) + 1)) + 1,
                    "outcome": "recovered",
                    "confidence": 0.9,
                    "metadata": {"recovery_observed": True, "duration": 1},
                }
            )
        segmented_rollout["segments"] = segments
        seg_out["segments"] = segments

    primitive_extractor = SemanticPrimitiveExtractor()
    primitives = primitive_extractor.extract([segmented_rollout])

    ontology = EnvironmentOntology(ontology_id="ros_stage2", name="ros_stage2")
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
            source="ros_bridge",
        )
        for p in primitives
    ]

    ontology_engine = OntologyUpdateEngine(ontology=ontology, task_graph=task_graph, econ_signals=econ_signals, datapack_signals=datapack_signals)
    ontology_proposals = ontology_engine.generate_proposals(primitive_models)
    if not ontology_proposals:
        ontology_proposals = [
            OntologyUpdateProposal(
                proposal_id=f"ros_prop_{idx}",
                proposal_type=ProposalType.ADD_AFFORDANCE,
                priority=ProposalPriority.MEDIUM,
                source_primitive_id=prim.primitive_id,
                source="ros_stage2",
                proposed_changes={"tags": prim.tags, "risk_level": prim.risk_level},
                rationale="ros_stage2_stub",
            )
            for idx, prim in enumerate(primitive_models)
        ]

    task_graph_refiner = TaskGraphRefiner(task_graph=task_graph, ontology=ontology, econ_signals=econ_signals, datapack_signals=datapack_signals)
    task_graph_proposals = task_graph_refiner.generate_refinements(ontology_proposals, primitives=primitive_models)
    if not task_graph_proposals:
        task_graph_proposals = [
            TaskGraphRefinementProposal(
                proposal_id=f"ros_refine_{idx}",
                refinement_type=RefinementType.SPLIT_TASK,
                priority=RefinementPriority.MEDIUM,
                target_task_ids=[prim.primitive_id],
                metadata={"tags": prim.tags},
            )
            for idx, prim in enumerate(primitive_models)
        ]

    econ_vector = _econ_from_rollout(rollout)
    econ_outputs = {rollout.episode_id: econ_vector}
    propagator = SemanticTagPropagator(enable_ood_recovery_tags=True)
    tags = propagator.generate_proposals([segmented_rollout], ontology_proposals, task_graph_proposals, econ_outputs)

    if inject_anomaly:
        for idx, tag in enumerate(tags):
            if hasattr(tag, "ood_tags"):
                if not getattr(tag, "ood_tags"):
                    tag.ood_tags.append({"tag_type": "OODTag", "severity": 1.0, "source": "synthetic", "details": {"anomaly": True}})
            elif isinstance(tag, dict):
                tag.setdefault("ood_tags", [{"tag_type": "OODTag", "severity": 1.0, "source": "synthetic"}])
            if hasattr(tag, "recovery_tags"):
                if not getattr(tag, "recovery_tags"):
                    tag.recovery_tags.append(
                        {
                            "tag_type": "RecoveryTag",
                            "value_add": "medium",
                            "correction_type": "synthetic_recovery",
                            "cost_wh": econ_vector.get("energy_wh", 0.0),
                        }
                    )
            elif isinstance(tag, dict):
                tag.setdefault(
                    "recovery_tags",
                    [
                        {
                            "tag_type": "RecoveryTag",
                            "value_add": "medium",
                            "correction_type": "synthetic_recovery",
                            "cost_wh": econ_vector.get("energy_wh", 0.0),
                        }
                    ],
                )

    datapacks = [
        {
            "segments": segments,
            "econ_vector": {
                "damage": econ_vector.get("damage", 0.0),
                "mpl": econ_vector.get("mpl", 0.0),
                "energy_wh": econ_vector.get("energy_wh", 0.0),
                "success": True,
            },
        }
    ]
    trust_matrix = EconCorrelator(config={"min_samples_for_trust": 1}).compute_correlations(datapacks)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "primitives.jsonl", primitives)
    _write_jsonl(output_dir / "ontology_proposals.jsonl", [p.to_dict() if hasattr(p, "to_dict") else p for p in ontology_proposals])
    _write_jsonl(output_dir / "semantic_tags.jsonl", [t.to_dict() if hasattr(t, "to_dict") else t for t in tags])
    _write_jsonl(output_dir / "segments.jsonl", segments)
    with (output_dir / "trust_matrix.json").open("w") as f:
        json.dump(to_json_safe(trust_matrix), f, sort_keys=True, indent=2)
    with (output_dir / "econ_vector.json").open("w") as f:
        json.dump(to_json_safe(econ_vector), f, sort_keys=True, indent=2)

    return {
        "rollout": segmented_rollout,
        "primitives": primitives,
        "ontology_proposals": ontology_proposals,
        "task_graph_proposals": task_graph_proposals,
        "semantic_tags": tags,
        "trust_matrix": trust_matrix,
        "econ_vector": econ_vector,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ROS → Stage 2 pipeline.")
    parser.add_argument("--log-path", type=str, required=True, help="Path to ROS JSON/bag export.")
    parser.add_argument("--output-dir", type=str, default="results/ros_stage2", help="Output directory for artifacts.")
    parser.add_argument("--inject-anomaly", action="store_true", help="Inject synthetic anomaly for OOD/Recovery tagging.")
    args = parser.parse_args()

    outputs = run_pipeline(Path(args.log_path), Path(args.output_dir), inject_anomaly=bool(args.inject_anomaly))
    print(
        f"[run_ros_to_stage2_pipeline] rollout={outputs['rollout'].get('episode_id')} "
        f"primitives={len(outputs['primitives'])} tags={len(outputs['semantic_tags'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
