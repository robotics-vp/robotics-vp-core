#!/usr/bin/env python3
"""
Micro-smokes for heuristic policy implementations.

Validates deterministic, JSON-safe outputs and required keys.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.policies.registry import build_all_policies  # noqa: E402
from src.semantic.models import EconSlice, MetaTransformerSlice, SemanticSnapshot  # noqa: E402
from src.vision.interfaces import VisionFrame  # noqa: E402


@dataclass
class DummyEvent:
    episode_id: str
    timestep: int
    reward_components: dict


def assert_json_safe(payload):
    import json

    json.dumps(payload)


def main():
    bundle = build_all_policies()
    # Data valuation
    dp = {"pack_id": "dp_test", "quality_score": 0.7, "novelty_score": 0.1, "semantic_tags": ["tag_a"], "attribution": {"trust_score": 0.8, "w_econ": 0.9}}
    fv = bundle.data_valuation.build_features(dp, econ_slice={"avg_mpl": 1.0})
    dv1 = bundle.data_valuation.score(fv)
    dv2 = bundle.data_valuation.score(fv)
    assert dv1["valuation_score"] == dv2["valuation_score"]
    assert_json_safe(dv1)

    # Pricing
    task_econ = {"task": {"human_mpl_units_per_hour": 10.0, "human_wage_per_hour": 20.0}, "mpl": {"mean": 5.0}, "wage_parity": {"mean": 1.0}}
    pf = bundle.pricing.build_features(task_econ, datapack_value=0.7)
    price = bundle.pricing.evaluate(pf)
    assert "unit_price" in price and "robot_hour_price" in price
    assert_json_safe(price)

    # Safety + energy
    events = [
        DummyEvent("ep", 0, {"collision_penalty": -0.2, "energy_penalty": -0.1}),
        DummyEvent("ep", 1, {"collision_penalty": 0.0, "energy_penalty": -0.2}),
    ]
    safety_feats = bundle.safety_risk.build_features(events)
    safety = bundle.safety_risk.evaluate(safety_feats)
    energy_feats = bundle.energy_cost.build_features(events)
    energy = bundle.energy_cost.evaluate(energy_feats)
    assert safety["damage_estimate"] == sum([-0.2, 0.0])
    assert energy["energy_cost"] == sum([-0.1, -0.2])
    assert_json_safe(safety)
    assert_json_safe(energy)

    # Episode quality
    eq_feats = bundle.episode_quality.build_features([1.0, 2.0, 3.0], [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}], [1, 2], {"recap_goodness_score": 0.5})
    eq = bundle.episode_quality.evaluate(eq_feats)
    assert "quality_score" in eq and "anomaly_score" in eq
    assert_json_safe(eq)

    # Sampler weights
    descriptors = [
        {"descriptor": {"pack_id": "dp_a", "trust_score": 0.5, "sampling_weight": 1.0}, "frontier_score": 0.2, "econ_urgency_score": 0.1, "recap_weight_multiplier": 1.0},
        {"descriptor": {"pack_id": "dp_b", "trust_score": 0.8, "sampling_weight": 1.3}, "frontier_score": 0.3, "econ_urgency_score": 0.2, "recap_weight_multiplier": 1.1},
    ]
    sw_feats = bundle.sampler_weights.build_features(descriptors)
    weights = bundle.sampler_weights.evaluate(sw_feats, strategy="balanced")
    assert set(weights.keys()) == {"dp_a", "dp_b"}

    # Orchestrator + meta advisor
    econ_slice = EconSlice(task_id="task", avg_mpl_units_per_hour=1.0, avg_wage_parity=1.0, avg_energy_cost=0.1, avg_error_rate=0.0)
    meta_slice = MetaTransformerSlice(task_id="task")
    snapshot = SemanticSnapshot(task_id="task", ontology_proposals=[], task_refinements=[], semantic_tags=[], econ_slice=econ_slice, meta_slice=meta_slice, metadata={})
    advisory = bundle.orchestrator.advise(snapshot)
    assert getattr(advisory, "task_id", "") == "task"

    meta_out = bundle.meta_advisor.evaluate(bundle.meta_advisor.build_features(meta_slice))
    assert hasattr(meta_out, "objective_preset")

    # Vision encoder
    frame = VisionFrame(backend="sim", task_id="task", episode_id="ep", timestep=0, rgb_path="path")
    latent = bundle.vision_encoder.encode(frame)
    latents = bundle.vision_encoder.batch_encode([frame, frame])
    assert latent.latent == latents[0].latent == latents[1].latent
    assert_json_safe(latent.to_dict())

    print("[smoke_test_policy_heuristics] OK")


if __name__ == "__main__":
    main()
