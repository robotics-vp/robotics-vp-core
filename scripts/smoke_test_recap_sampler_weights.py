#!/usr/bin/env python3
"""
Smoke test for recap-weighted sampler adjustments.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.rl.episode_sampling import DataPackRLSampler, _balanced_weight


def _make_descriptors():
    descs = []
    for i in range(3):
        descs.append(
            {
                "pack_id": f"pack_{i}",
                "env_name": "drawer",
                "task_type": "drawer",
                "engine_type": "pybullet",
                "backend": "pybullet",
                "objective_vector": [1, 1, 1, 1, 0],
                "objective_preset": "balanced",
                "tier": 1,
                "trust_score": 0.5,
                "sampling_weight": 1.0,
                "semantic_tags": [],
            }
        )
    return descs


def main():
    descs = _make_descriptors()
    sampler_base = DataPackRLSampler(existing_descriptors=descs, use_recap_weights=False)
    weights_base = [_balanced_weight(ep) for ep in sampler_base._episodes]
    assert max(weights_base) - min(weights_base) < 0.2

    recap_scores = {"pack_0": 2.0, "pack_1": 0.0, "pack_2": -2.0}
    sampler_recap = DataPackRLSampler(existing_descriptors=descs, use_recap_weights=True, recap_scores=recap_scores)
    weights_recap = [_balanced_weight(ep) for ep in sampler_recap._episodes]
    ordered = sorted(zip([ep["descriptor"]["pack_id"] for ep in sampler_recap._episodes], weights_recap), key=lambda kv: -kv[1])
    assert ordered[0][0] == "pack_0" and ordered[-1][0] == "pack_2"

    batch_recap = sampler_recap.sample_batch(batch_size=3, seed=0, strategy="balanced")
    ids_recap = [b["pack_id"] for b in batch_recap]
    assert ids_recap.count("pack_0") >= 1
    print("[smoke_test_recap_sampler_weights] All tests passed.")


if __name__ == "__main__":
    main()
