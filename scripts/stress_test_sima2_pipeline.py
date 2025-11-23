#!/usr/bin/env python3
"""
Synthetic stress test for SIMA-2 pipeline with optional policy dataset export.

Implements SIMA2_SCALING_AND_STRESS_TESTS.md:
- Deterministic 10k rollout benchmark (batched)
- OOM-safe (streaming JSONL writes)
- Ontology write-amplification guardrails
- Tag frequency analytics and proposal explosion detection
- Performance counters (throughput/latency)
"""
import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.analytics.econ_correlator import load_trust_matrix
from src.sima2.client import Sima2Client
from src.sima2.config import load_sima2_config, extract_provenance
from src.sima2.segmentation_engine import SegmentationEngine
from src.sima2.semantic_primitive_extractor import SemanticPrimitiveExtractor, SemanticPrimitive
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.sima2.semantic_tag_propagator import SemanticTagPropagator
from src.sima2.ontology_proposals import OntologyUpdateProposal
from src.sima2.task_graph_proposals import TaskGraphRefinementProposal
from src.ontology.sima2_adapters import datapack_from_sima2_rollout
from src.ontology.models import Task, Robot, Datapack, Episode, EconVector, EpisodeEvent
from src.ontology.store import OntologyStore
from src.orchestrator.ontology import EnvironmentOntology
from src.orchestrator.task_graph import TaskGraph, TaskNode, TaskType
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals
from scripts.build_policy_datasets import build_datasets


def _write_jsonl(path: Path, records: List[Any]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in records:
            obj = rec.to_dict() if hasattr(rec, "to_dict") else rec
            f.write(json.dumps(obj, sort_keys=True))
            f.write("\n")


def _task_sequence(distribution: Dict[str, float], total: int) -> List[str]:
    tasks: List[str] = []
    normalized = {k: float(v) for k, v in distribution.items()}
    total_weight = sum(normalized.values()) or 1.0
    normalized = {k: v / total_weight for k, v in normalized.items()}
    for task, weight in normalized.items():
        count = int(round(weight * total))
        tasks.extend([task] * max(count, 1))
    # Deterministic padding if rounding left gaps
    while len(tasks) < total:
        for task in sorted(normalized.keys()):
            tasks.append(task)
            if len(tasks) >= total:
                break
    return tasks[:total]


def _episode_events(episode_id: str, rollout: Dict[str, Any]) -> List[EpisodeEvent]:
    events: List[EpisodeEvent] = []
    timestamp = datetime.utcnow()
    for ev in rollout.get("events", []):
        events.append(
            EpisodeEvent(
                episode_id=episode_id,
                timestep=int(ev.get("timestep", 0)),
                event_type=str(ev.get("event_type", "step")),
                timestamp=timestamp,
                reward_scalar=float(ev.get("reward_scalar", 0.0)),
                reward_components=ev.get("reward_components", {}),
                state_summary=ev.get("state", ev.get("payload", {})),
                metadata=ev.get("metadata", {}),
            )
        )
    return events


def _econ_vector(episode_id: str, task_id: str, primitives: List[Dict[str, Any]]) -> EconVector:
    energy_cost = float(len(primitives)) * 0.1
    damage_cost = -0.1 * float(sum(1 for p in primitives if p.get("risk_level") == "high"))
    mpl = float(len(primitives)) or 1.0
    return EconVector(
        episode_id=episode_id,
        mpl_units_per_hour=mpl,
        wage_parity=0.5,
        energy_cost=energy_cost,
        damage_cost=damage_cost,
        novelty_delta=0.0,
        reward_scalar_sum=mpl,
    )


def _noise_rollout(task_id: str, idx: int) -> Dict[str, Any]:
    """Generate a lightweight noise rollout for robustness testing."""
    return {
        "episode_id": f"{task_id}_ep_{idx}",
        "task": task_id,
        "primitives": [
            {"timestep": 0, "object": "noise", "action": "noop", "risk": "low", "contact": False},
            {"timestep": 1, "object": "noise", "action": "wiggle", "risk": "medium", "contact": bool(idx % 2)},
        ],
        "events": [{"timestep": 1, "event_type": "primitive", "payload": {"action": "wiggle"}}],
        "metadata": {"objects_present": ["noise"], "debug_mode": False},
    }


def _template_for_task(idx: int) -> str:
    """Deterministic template schedule to cover success/failure/recovery."""
    if idx % 10 == 0:
        return "failure"
    if idx % 7 == 0:
        return "recovery"
    if idx % 13 == 0:
        return "mixed"
    return "success"


def _maybe_gc(batch_idx: int) -> None:
    if batch_idx % 2 == 0:
        gc.collect()


def _p99(latencies: Sequence[float]) -> float:
    if not latencies:
        return 0.0
    vals = sorted(latencies)
    k = int(0.99 * (len(vals) - 1))
    return float(vals[k])


def _guardrails(
    ontology_props: Sequence[Any],
    task_graph_props: Sequence[Any],
    rollouts_in_batch: int,
    max_write_multiplier: float,
) -> None:
    if rollouts_in_batch <= 0:
        return
    if len(ontology_props) > max_write_multiplier * rollouts_in_batch:
        raise AssertionError(
            f"Ontology write amplification too high: {len(ontology_props)} props for {rollouts_in_batch} rollouts"
        )
    if len(task_graph_props) > max_write_multiplier * rollouts_in_batch:
        raise AssertionError(
            f"Task graph refinement explosion: {len(task_graph_props)} props for {rollouts_in_batch} rollouts"
        )


def main():
    parser = argparse.ArgumentParser(description="Stress test SIMA-2 pipeline")
    parser.add_argument("--num-rollouts", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--task-distribution", type=str, default="dataset_stress_mix_v1")
    parser.add_argument("--output-dir", type=str, default="results/sima2_stress")
    parser.add_argument("--ontology-root", type=str, default="results/sima2_stress/ontology_store")
    parser.add_argument("--emit-policy-datasets", action="store_true")
    parser.add_argument("--policy-dataset-dir", type=str, default="results/policy_datasets_sima2_stress")
    parser.add_argument("--max-write-multiplier", type=float, default=50.0)
    args = parser.parse_args()

    cfg = load_sima2_config()
    dataset_presets = {
        "dataset_stress_mix_v1": {
            "distribution": {"drawer_open": 0.4, "dish_place": 0.4, "noise": 0.2},
            "params": {"failure_rate": 0.3, "recovery_rate": 0.1, "ood_rate": 0.05},
        },
        "dataset_edge_cases_v1": {"distribution": {"noise": 1.0}},
    }
    preset = dataset_presets.get(args.task_distribution) or cfg.get("task_distribution", {}).get(args.task_distribution, {})
    if not preset:
        preset = {"distribution": {"drawer_open": 0.5, "dish_place": 0.5}}
    tasks = _task_sequence(preset.get("distribution", preset), args.num_rollouts)
    rng = random.Random(0)
    rng.shuffle(tasks)

    client = Sima2Client(task_id=tasks[0] if tasks else "drawer_open")
    seg_engine = SegmentationEngine(segmentation_config=cfg)
    primitive_extractor = SemanticPrimitiveExtractor()

    ontology = EnvironmentOntology(ontology_id="sima2_stress", name="sima2_stress")
    task_graph = TaskGraph(TaskNode(task_id="root", name="root", task_type=TaskType.ROOT))
    econ_signals = EconSignals()
    datapack_signals = DatapackSignals()
    ontology_engine = OntologyUpdateEngine(ontology=ontology, task_graph=task_graph, econ_signals=econ_signals, datapack_signals=datapack_signals)
    refiner = TaskGraphRefiner(task_graph=task_graph, ontology=ontology, econ_signals=econ_signals, datapack_signals=datapack_signals)
    propagator = SemanticTagPropagator()

    store = OntologyStore(root_dir=args.ontology_root)
    store.upsert_robot(Robot(robot_id="sima2_stub_robot", name="sima2_stub_robot"))
    trust_matrix = load_trust_matrix()

    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    primitives_path = result_dir / "primitives.jsonl"
    ontology_path = result_dir / "ontology_proposals.jsonl"
    task_graph_path = result_dir / "task_graph_proposals.jsonl"
    tags_path = result_dir / "semantic_tags.jsonl"

    total_rollouts = 0
    total_primitives = 0
    total_ontology_props = 0
    total_task_graph_props = 0
    total_tags = 0
    tag_frequency: Dict[str, int] = {}
    per_rollout_latencies: List[float] = []
    start_time = time.perf_counter()

    for batch_idx in range(0, len(tasks), args.batch_size):
        batch_tasks = tasks[batch_idx : batch_idx + args.batch_size]
        batch_primitives: List[Dict[str, Any]] = []
        batch_primitive_models: List[SemanticPrimitive] = []
        batch_segmented_rollouts: List[Dict[str, Any]] = []
        batch_latency: List[float] = []

        for local_idx, task_id in enumerate(batch_tasks):
            idx = batch_idx + local_idx
            store.upsert_task(Task(task_id=task_id, name=task_id, environment_id="sima2"))
            template = _template_for_task(idx)
            t0 = time.perf_counter()
            if task_id in {"noise", "random_noise"}:
                rollout = _noise_rollout(task_id, idx)
            else:
                rollout = client.run_episode({"task_id": task_id, "episode_index": idx, "template": template})
            seg_out = seg_engine.segment_rollout(rollout)
            seg_rollout = seg_out["rollout"]
            batch_segmented_rollouts.append(seg_rollout)

            prims = primitive_extractor.extract([seg_rollout])
            batch_primitives.extend(prims)
            batch_primitive_models.extend(
                [
                    SemanticPrimitive(
                        primitive_id=p.get("primitive_id", f"{task_id}_prim_{idx}"),
                        task_type=p.get("task_type", task_id),
                        tags=p.get("tags", []),
                        risk_level=p.get("risk_level", "low"),
                        energy_intensity=float(p.get("energy_intensity", 0.0)),
                        success_rate=float(p.get("success_rate", 1.0)),
                        avg_steps=float(p.get("avg_steps", 1.0)),
                        source=p.get("source", "sima2"),
                        provenance=p.get("provenance", {}),
                    )
                    for p in prims
                ]
            )

            dp = datapack_from_sima2_rollout(seg_rollout, task_id)
            if isinstance(dp, Datapack):
                store.append_datapacks([dp])

            prov = extract_provenance(seg_rollout, cfg)
            ep = Episode(
                episode_id=seg_rollout.get("episode_id", f"{task_id}_ep_{idx}"),
                task_id=task_id,
                robot_id="sima2_stub_robot",
                datapack_id=getattr(dp, "datapack_id", None) if isinstance(dp, Datapack) else None,
                status="success",
                metadata={"segmented": True, "sima2_provenance": prov},
                sima2_backend_id=prov.get("sima2_backend_id"),
                sima2_model_version=prov.get("sima2_model_version"),
                sima2_task_spec=prov.get("sima2_task_spec", {}),
            )
            store.upsert_episode(ep)
            store.append_events(_episode_events(ep.episode_id, seg_rollout))
            econ_vec = _econ_vector(ep.episode_id, task_id, prims)
            store.upsert_econ_vector(econ_vec)
            batch_latency.append(time.perf_counter() - t0)

        ontology_proposals = ontology_engine.generate_proposals(batch_primitive_models)
        task_graph_proposals = refiner.generate_refinements(ontology_proposals, primitives=batch_primitive_models)
        econ_outputs = {
            r.get("episode_id"): {"novelty_delta": 0.0, "trust_matrix": trust_matrix} for r in batch_segmented_rollouts
        }
        tag_enrichments = propagator.generate_proposals(
            batch_segmented_rollouts, ontology_proposals, task_graph_proposals, econ_outputs
        )

        _guardrails(ontology_proposals, task_graph_proposals, len(batch_tasks), args.max_write_multiplier)

        _write_jsonl(primitives_path, batch_primitives)
        _write_jsonl(ontology_path, ontology_proposals)
        _write_jsonl(task_graph_path, task_graph_proposals)
        _write_jsonl(tags_path, tag_enrichments)

        total_rollouts += len(batch_tasks)
        total_primitives += len(batch_primitives)
        total_ontology_props += len(ontology_proposals)
        total_task_graph_props += len(task_graph_proposals)
        total_tags += len(tag_enrichments)
        per_rollout_latencies.extend(batch_latency)

        for proposal in tag_enrichments:
            try:
                # Count tag types directly
                tag_frequency["risk_tags"] = tag_frequency.get("risk_tags", 0) + len(proposal.risk_tags)
                tag_frequency["recovery_tags"] = tag_frequency.get("recovery_tags", 0) + len(
                    getattr(proposal, "recovery_tags", [])
                )
                tag_frequency["ood_tags"] = tag_frequency.get("ood_tags", 0) + len(getattr(proposal, "ood_tags", []))
            except Exception:
                continue

        _maybe_gc(batch_idx // args.batch_size)

    if args.emit_policy_datasets:
        out_dir = Path(args.policy_dataset_dir)
        build_datasets(store, out_dir)
        shards = list(out_dir.glob("*.jsonl"))
        assert shards, "Expected at least one policy dataset shard"
        print(f"[stress_test_sima2_pipeline] policy datasets written to {out_dir}")

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    throughput = total_rollouts / elapsed
    p99_latency = _p99(per_rollout_latencies)
    rss_mb = 0.0
    try:
        import resource  # Unix only

        rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        rss_mb = 0.0

    metrics = {
        "rollouts": total_rollouts,
        "primitives": total_primitives,
        "ontology_proposals": total_ontology_props,
        "task_graph_proposals": total_task_graph_props,
        "tag_enrichments": total_tags,
        "throughput_rps": throughput,
        "latency_p99_s": p99_latency,
        "rss_mb": rss_mb,
        "tag_frequency": tag_frequency,
        "params": {
            "num_rollouts": args.num_rollouts,
            "batch_size": args.batch_size,
            "task_distribution": args.task_distribution,
        },
    }

    # Basic performance assertions per spec (soft thresholds for smoke)
    if args.num_rollouts >= 1000:
        assert throughput > 10.0, f"Throughput too low: {throughput:.2f} rps"
    assert args.max_write_multiplier >= 1.0

    metrics_path = result_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(
        f"[stress_test_sima2_pipeline] rollouts={total_rollouts} primitives={total_primitives} "
        f"ontology_proposals={total_ontology_props} task_graph_proposals={total_task_graph_props} "
        f"tags={total_tags} throughput={throughput:.2f}rps p99={p99_latency:.4f}s rss_mb={rss_mb:.2f}"
    )


if __name__ == "__main__":
    main()
