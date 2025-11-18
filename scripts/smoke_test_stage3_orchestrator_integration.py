#!/usr/bin/env python3
"""
Smoke test for sampler/curriculum advisory integration.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.rl.episode_sampling import DataPackRLSampler
from src.rl.curriculum import DataPackCurriculum
from src.orchestrator.semantic_orchestrator_v2 import OrchestratorAdvisory


def make_sampler(advisory=None):
    descriptors = []
    enrichments = []
    for i in range(4):
        descriptors.append({
            "pack_id": f"dp{i}",
            "env_name": "drawer_vase",
            "task_type": "drawer_vase",
            "engine_type": "pybullet",
            "backend": "pybullet",
            "objective_vector": [1,1,1,1,0],
            "objective_preset": "balanced",
            "tier": i % 3,
            "trust_score": 0.5 + 0.1*i,
            "sampling_weight": 1.0 + 0.1*i,
            "semantic_tags": ["safety"] if i==0 else [],
        })
        enrichments.append({
            "episode_id": f"dp{i}",
            "enrichment": {
                "novelty_tags": [{"novelty_score": 0.2*i, "expected_mpl_gain": 0.5*i}],
                "coherence_score": 0.6,
            }
        })
    sampler = DataPackRLSampler(datapacks=None, enrichments=enrichments, existing_descriptors=descriptors, advisory=advisory)
    return sampler


def main():
    advisory = OrchestratorAdvisory(
        task_id="task_adv",
        focus_objective_presets=["balanced"],
        sampler_strategy_overrides={"econ_urgency":0.5,"frontier_prioritized":0.4,"balanced":0.1},
        datapack_priority_tags=["safety"],
        safety_emphasis=0.8,
    )
    sampler = make_sampler(advisory=advisory)
    batch = sampler.sample_batch(batch_size=3, seed=1)
    strategies = {b["sampling_metadata"]["strategy"] for b in batch if "sampling_metadata" in b}
    assert strategies  # strategies present
    sampler_no_adv = make_sampler()
    batch_default = sampler_no_adv.sample_batch(batch_size=3, seed=1)
    assert batch != batch_default, "Advisory overrides should affect sampling choice deterministically"

    curriculum = DataPackCurriculum(sampler=sampler, total_steps=10, config={"base_seed":0}, advisory=advisory)
    batch_curr = curriculum.sample_batch(step=6, batch_size=3)
    strat_counts = {}
    for b in batch_curr:
        strat = b.get("sampling_metadata", {}).get("strategy")
        strat_counts[strat] = strat_counts.get(strat,0)+1
    assert strat_counts.get("econ_urgency",0) >= 1
    print("[smoke_test_stage3_orchestrator_integration] All tests passed.")


if __name__ == "__main__":
    main()
