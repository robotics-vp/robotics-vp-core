#!/usr/bin/env python3
"""
Smoke test for centralized skill_mode resolver.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.rl.skill_mode_resolver import SkillModeResolver
from src.rl.episode_sampling import DataPackRLSampler
from src.rl.curriculum import DataPackCurriculum


def make_descriptor(pack_id: str, tags=None, trust: float = 0.9):
    tags = tags or []
    return {
        "pack_id": pack_id,
        "env_name": "drawer_vase",
        "task_type": "drawer_vase",
        "engine_type": "pybullet",
        "backend": "pybullet",
        "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
        "objective_preset": "balanced",
        "tier": 1,
        "trust_score": trust,
        "sampling_weight": 1.0,
        "delta_mpl": 0.1,
        "delta_J": 0.1,
        "semantic_tags": tags,
    }


def main():
    resolver = SkillModeResolver()
    assert resolver.resolve(tags={"frontier": 1.0}, trust_matrix=None, curriculum_phase="frontier", advisory={"strategy": "frontier_prioritized"}) == "frontier_exploration"
    assert resolver.resolve(tags=None, trust_matrix={"a": 0.95}, curriculum_phase="skill_building", advisory={"strategy": "balanced"}) == "efficiency_throughput"
    assert resolver.resolve(tags={"recovery_zone": 1.0}, trust_matrix={"a": 0.2}, curriculum_phase="warmup", advisory={}) == "recovery_heavy"
    assert resolver.resolve(tags=None, trust_matrix=None, curriculum_phase=None, advisory=None, use_condition_vector=False) == resolver.default_mode
    # Determinism
    assert resolver.resolve(tags={"frontier": 1.0}, trust_matrix=None, curriculum_phase="frontier", advisory={"strategy": "frontier_prioritized"}) == "frontier_exploration"

    descriptors = [make_descriptor("dp_frontier", tags=["frontier"]), make_descriptor("dp_safe", tags=["fragile"], trust=0.4)]
    sampler = DataPackRLSampler(existing_descriptors=descriptors, use_condition_vector=True)
    batch = sampler.sample_batch(batch_size=2, seed=0, strategy="balanced")
    skill_modes = {item["pack_id"]: item.get("sampling_metadata", {}).get("skill_mode") for item in batch}
    assert skill_modes["dp_frontier"] == "frontier_exploration"
    assert skill_modes["dp_safe"] in {"safety_critical", "recovery_heavy", "efficiency_throughput"}

    curriculum = DataPackCurriculum(sampler=sampler, total_steps=10, config={"base_seed": 0, "use_condition_vector": True})
    cur_batch = curriculum.sample_batch(step=5, batch_size=1)
    meta = cur_batch[0].get("sampling_metadata", {})
    assert meta.get("skill_mode"), "Curriculum should attach skill_mode via resolver"
    assert cur_batch[0].get("condition_metadata"), "Condition metadata should be present when condition vectors enabled"

    print("[smoke_test_skill_mode_resolver] All checks passed.")


if __name__ == "__main__":
    main()
