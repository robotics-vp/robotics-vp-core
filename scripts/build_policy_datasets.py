#!/usr/bin/env python3
"""
Build read-only feature/target JSONL datasets for each Phase G policy.

Uses current heuristic policies to emit datasets under results/policy_datasets/.
No behavior changes or training occur; this is purely logging.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(repo_root))

from src.analytics.econ_reports import compute_task_econ_summary, compute_datapack_mix_summary  # noqa: E402
from src.ontology.store import OntologyStore  # noqa: E402
from src.policies.registry import build_all_policies  # noqa: E402
from src.semantic.aggregator import SemanticAggregator  # noqa: E402
from src.utils.json_safe import to_json_safe  # noqa: E402
from src.utils.logging_schema import POLICY_LOG_FIELDS  # noqa: E402
from src.vision.interfaces import VisionFrame  # noqa: E402

DEFAULT_TIMESTAMP = datetime.utcfromtimestamp(0).isoformat()


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(to_json_safe(rec), sort_keys=True))
            f.write("\n")


def _record(
    policy: str,
    features: Any,
    target: Any,
    task_id: str = "",
    episode_id: str = "",
    datapack_id: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        POLICY_LOG_FIELDS["policy_name"]: policy,
        POLICY_LOG_FIELDS["input_features"]: to_json_safe(features),
        POLICY_LOG_FIELDS["output"]: to_json_safe(target),
        POLICY_LOG_FIELDS["meta"]: to_json_safe(meta or {}),
        POLICY_LOG_FIELDS["task_id"]: task_id,
        POLICY_LOG_FIELDS["episode_id"]: episode_id,
        POLICY_LOG_FIELDS["datapack_id"]: datapack_id,
        POLICY_LOG_FIELDS["timestamp"]: DEFAULT_TIMESTAMP,
    }


def build_datasets(store: OntologyStore, out_dir: Path) -> None:
    policies = build_all_policies()
    aggregator = SemanticAggregator(store, policies=policies)
    tasks = sorted(store.list_tasks(), key=lambda t: getattr(t, "task_id", ""))
    datapacks = sorted(store.list_datapacks(), key=lambda dp: getattr(dp, "pack_id", getattr(dp, "datapack_id", "")))
    episodes = sorted(store.list_episodes(), key=lambda ep: getattr(ep, "episode_id", ""))
    econ_vectors = {ev.episode_id: ev for ev in sorted(store.list_econ_vectors(), key=lambda ev: ev.episode_id)}

    econ_summary = {t.task_id: compute_task_econ_summary(store, t.task_id) for t in tasks}
    datapack_values: Dict[str, float] = {}
    datapack_task: Dict[str, str] = {}
    data_val_records: List[Dict[str, Any]] = []
    auditor_records: List[Dict[str, Any]] = []
    for dp in datapacks:
        dp_id = getattr(dp, "pack_id", getattr(dp, "datapack_id", ""))
        task_id = getattr(dp, "task_id", "")
        datapack_task[dp_id] = task_id
        feat = policies.data_valuation.build_features(dp, econ_slice=econ_summary.get(task_id))
        scored = policies.data_valuation.score(feat)
        datapack_values[dp_id] = float(scored.get("valuation_score", 0.0))
        data_val_records.append(_record("data_valuation", feat, scored, task_id=task_id, episode_id=dp_id, datapack_id=dp_id))
        semantic_tags = []
        if isinstance(getattr(dp, "tags", {}), dict):
            for v in dp.tags.values():
                if isinstance(v, list):
                    semantic_tags.extend(v)
        auditor_features = policies.datapack_auditor.build_features(
            datapack=dp,
            semantic_tags=semantic_tags,
            econ_slice={"expected_mpl_gain": econ_summary.get(dp.task_id, {}).get("mpl", {}).get("mean", 0.0), "novelty_score": getattr(dp, "novelty_score", 0.0)},
        )
        auditor_eval = policies.datapack_auditor.evaluate(auditor_features)
        auditor_records.append(
            _record(
                "datapack_auditor",
                auditor_features,
                auditor_eval,
                task_id=getattr(dp, "task_id", ""),
                datapack_id=getattr(dp, "datapack_id", getattr(dp, "pack_id", "")),
                meta={"policy_backend": auditor_eval.get("metadata", {}).get("auditor_backend", "heuristic_v1")},
            )
        )
    _write_jsonl(out_dir / "data_valuation.jsonl", data_val_records)
    _write_jsonl(out_dir / "datapack_auditor.jsonl", auditor_records)

    pricing_records: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = task.task_id
        summary = econ_summary.get(task_id, {})
        dp_mix = compute_datapack_mix_summary(store, task_id)
        # Average datapack value for this task_id if available
        vals = [v for pid, v in datapack_values.items() if datapack_task.get(pid) == task_id]
        avg_val = sum(vals) / len(vals) if vals else 0.0
        features = policies.pricing.build_features(task_econ=summary, datapack_value=avg_val, semantic_context={"datapack_mix": dp_mix})
        target = policies.pricing.evaluate(features)
        pricing_records.append(_record("pricing", features, target, task_id=task_id))
    _write_jsonl(out_dir / "pricing.jsonl", pricing_records)

    safety_records: List[Dict[str, Any]] = []
    energy_records: List[Dict[str, Any]] = []
    quality_records: List[Dict[str, Any]] = []
    sampler_records: List[Dict[str, Any]] = []

    descriptors: List[Dict[str, Any]] = []
    for ep in episodes:
        events = store.get_events(ep.episode_id)
        reward_components = [getattr(ev, "reward_components", {}) for ev in events]
        rewards = [getattr(ev, "reward_scalar", 0.0) for ev in events]
        collisions = [rc for rc in reward_components if rc.get("collision_penalty", 0.0)]

        safety_feat = policies.safety_risk.build_features(events)
        safety_target = policies.safety_risk.evaluate(safety_feat)
        safety_records.append(_record("safety_risk", safety_feat, safety_target, task_id=ep.task_id, episode_id=ep.episode_id))

        energy_feat = policies.energy_cost.build_features(events)
        energy_target = policies.energy_cost.evaluate(energy_feat)
        energy_records.append(_record("energy_cost", energy_feat, energy_target, task_id=ep.task_id, episode_id=ep.episode_id))

        recap_context = {}
        econ_vec = econ_vectors.get(ep.episode_id)
        if econ_vec:
            recap_context["recap_goodness_score"] = getattr(econ_vec, "novelty_delta", 0.0)
        eq_feat = policies.episode_quality.build_features(rewards, reward_components, collisions, recap_context)
        eq_target = policies.episode_quality.evaluate(eq_feat)
        quality_records.append(_record("episode_quality", eq_feat, eq_target, task_id=ep.task_id, episode_id=ep.episode_id))

        descriptors.append(
            {
                "descriptor": {
                    "pack_id": ep.episode_id,
                    "tier": getattr(ep, "metadata", {}).get("tier", 1) if hasattr(ep, "metadata") else 1,
                    "trust_score": getattr(econ_vec, "wage_parity", 0.5) if econ_vec else 0.5,
                    "sampling_weight": getattr(econ_vec, "mpl_units_per_hour", 1.0) if econ_vec else 1.0,
                },
                "frontier_score": getattr(econ_vec, "novelty_delta", 0.0) if econ_vec else 0.0,
                "econ_urgency_score": getattr(econ_vec, "energy_cost", 0.0) if econ_vec else 0.0,
                "recap_weight_multiplier": 1.0,
            }
        )

    _write_jsonl(out_dir / "safety_risk.jsonl", safety_records)
    _write_jsonl(out_dir / "energy_cost.jsonl", energy_records)
    _write_jsonl(out_dir / "episode_quality.jsonl", quality_records)

    if descriptors:
        sampler_feats = policies.sampler_weights.build_features(descriptors)
        sampler_weights = policies.sampler_weights.evaluate(sampler_feats, strategy="balanced")
        for feat in sampler_feats:
            pid = feat.get("descriptor", {}).get("pack_id") or feat.get("descriptor", {}).get("episode_id")
            sampler_records.append(
                _record(
                    "sampler_weights",
                    feat,
                    {"weight": sampler_weights.get(pid, 0.0)},
                    episode_id=str(pid),
                    datapack_id=str(pid),
                )
            )
    _write_jsonl(out_dir / "sampler_weights.jsonl", sampler_records)

    orchestrator_records: List[Dict[str, Any]] = []
    meta_records: List[Dict[str, Any]] = []
    for task in tasks:
        snapshot = aggregator.build_snapshot(
            task_id=task.task_id,
            stage2_ontology_proposals=[],
            stage2_task_refinements=[],
            stage2_tags=[],
            meta_outputs=None,
            recap_scores={},
        )
        advisory = policies.orchestrator.advise(snapshot)
        orchestrator_records.append(_record("orchestrator", snapshot.to_dict(), advisory.to_json(), task_id=task.task_id))

        meta_feats = policies.meta_advisor.build_features(snapshot.meta_slice)
        meta_target = policies.meta_advisor.evaluate(meta_feats)
        meta_records.append(_record("meta_advisor", meta_feats, meta_target, task_id=task.task_id))

    _write_jsonl(out_dir / "orchestrator.jsonl", orchestrator_records)
    _write_jsonl(out_dir / "meta_advisor.jsonl", meta_records)

    vision_records: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks[:3] or [None]):
        ep_ref = episodes[idx] if idx < len(episodes) else None
        ep_id = ep_ref.episode_id if ep_ref else f"vision_ep_{idx}"
        frame = VisionFrame(
            backend="policy_dataset",
            task_id=task.task_id if task else getattr(ep_ref, "task_id", "unknown_task"),
            episode_id=ep_id,
            timestep=idx,
            rgb_path=f"/tmp/{ep_id}.png",
            camera_name="cam0",
            metadata={"source": "policy_dataset"},
        )
        latent = policies.vision_encoder.encode(frame)
        vision_records.append(_record("vision_encoder", frame.to_dict(), latent.to_dict(), task_id=getattr(task, "task_id", ""), episode_id=ep_id))
    _write_jsonl(out_dir / "vision_encoder.jsonl", vision_records)


def main():
    parser = argparse.ArgumentParser(description="Build per-policy datasets (heuristic targets).")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--output-dir", type=str, default="results/policy_datasets")
    args = parser.parse_args()

    store = OntologyStore(root_dir=args.ontology_root)
    out_dir = Path(args.output_dir)
    build_datasets(store, out_dir)
    print(f"[build_policy_datasets] Wrote datasets to {out_dir}")


if __name__ == "__main__":
    main()
