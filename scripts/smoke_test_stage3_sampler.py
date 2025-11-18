#!/usr/bin/env python3
"""
Smoke test for Stage 3.1 DataPackRLSampler.

Builds a small pool of fake datapacks + Stage 2 enrichments and verifies:
- balanced / frontier_prioritized / econ_urgency strategies run
- determinism (same seed → same ordering)
- priority weighting favors frontier/urgent datapacks without changing objectives
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.rl.episode_sampling import DataPackRLSampler


def _make_descriptor(
    pack_id: str,
    tier: int,
    trust: float,
    delta_mpl: float,
    delta_J: float,
    weight: float,
) -> dict:
    """Create a minimal descriptor for testing."""
    return {
        "pack_id": pack_id,
        "env_name": "drawer_vase",
        "task_type": "drawer_vase",
        "engine_type": "pybullet",
        "backend": "pybullet",
        "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
        "objective_preset": "balanced",
        "tier": tier,
        "trust_score": trust,
        "sampling_weight": weight,
        "delta_mpl": delta_mpl,
        "delta_J": delta_J,
        "semantic_tags": ["test"],
    }


def _make_enrichment(episode_id: str, novelty: float, expected_gain: float, priority: str = "medium") -> dict:
    return {
        "episode_id": episode_id,
        "enrichment": {
            "novelty_tags": [
                {"novelty_score": novelty, "expected_mpl_gain": expected_gain, "comparison_basis": "smoke"},
            ],
            "coherence_score": 0.5,
            "supervision_hints": {
                "priority_level": priority,
                "suggested_weight_multiplier": 1.0,
                "requires_human_review": False,
                "safety_critical": priority in {"high", "critical"},
                "curriculum_stage": "mid",
                "prioritize_for_training": priority in {"high", "critical"},
                "suggested_replay_frequency": "standard",
                "prerequisite_tags": [],
            },
        },
    }


def build_fixture():
    descriptors = [
        _make_descriptor("ep_tier0_low", 0, 0.4, 0.1, 0.05, 0.6),
        _make_descriptor("ep_tier1_mid", 1, 0.65, 0.4, 0.2, 1.1),
        _make_descriptor("ep_tier1_frontier", 1, 0.8, 1.2, 0.6, 1.4),
        _make_descriptor("ep_tier2_frontier", 2, 0.9, 1.8, 1.1, 2.0),
        _make_descriptor("ep_tier2_safety", 2, 0.7, 0.6, 0.3, 1.5),
        _make_descriptor("ep_tier1_econ_hot", 1, 0.75, 0.5, 0.4, 1.2),
    ]

    enrichments = [
        _make_enrichment("ep_tier0_low", novelty=0.1, expected_gain=0.0),
        _make_enrichment("ep_tier1_mid", novelty=0.25, expected_gain=0.3),
        _make_enrichment("ep_tier1_frontier", novelty=0.7, expected_gain=1.2, priority="high"),
        _make_enrichment("ep_tier2_frontier", novelty=0.9, expected_gain=1.8, priority="critical"),
        _make_enrichment("ep_tier2_safety", novelty=0.5, expected_gain=0.9, priority="high"),
        _make_enrichment("ep_tier1_econ_hot", novelty=0.6, expected_gain=1.5, priority="critical"),
    ]
    return descriptors, enrichments


def assert_deterministic(sampler: DataPackRLSampler):
    batch_1 = sampler.sample_batch(batch_size=4, seed=123, strategy="frontier_prioritized")
    batch_2 = sampler.sample_batch(batch_size=4, seed=123, strategy="frontier_prioritized")
    assert batch_1 == batch_2, "Sampler must be deterministic for same seed/strategy"


def assert_frontier_bias(sampler: DataPackRLSampler):
    batch = sampler.sample_batch(batch_size=4, seed=7, strategy="frontier_prioritized")
    pack_ids = [b["pack_id"] for b in batch]
    assert "ep_tier2_frontier" in pack_ids, "Frontier sample should include top ΔMPL/ΔJ datapack"
    assert "ep_tier1_frontier" in pack_ids or "ep_tier1_econ_hot" in pack_ids, "Frontier sampler should elevate frontier tier1 datapacks"


def assert_econ_bias(sampler: DataPackRLSampler):
    batch = sampler.sample_batch(batch_size=4, seed=9, strategy="econ_urgency")
    pack_ids = [b["pack_id"] for b in batch]
    assert "ep_tier1_econ_hot" in pack_ids, "Econ urgency sampler should prioritize econ_hot datapack"


def assert_balanced_mix(sampler: DataPackRLSampler):
    batch = sampler.sample_batch(batch_size=6, seed=21, strategy="balanced")
    tiers = [b["tier"] for b in batch]
    assert any(t == 0 for t in tiers), "Balanced sampler should not drop tier 0 coverage"
    assert any(t == 2 for t in tiers), "Balanced sampler should include tier 2 coverage"


def main():
    descriptors, enrichments = build_fixture()
    sampler = DataPackRLSampler(
        datapacks=None,
        enrichments=enrichments,
        existing_descriptors=descriptors,
    )

    print("[Stage3 Sampler] Pool summary:", sampler.pool_summary())
    assert_balanced_mix(sampler)
    print("[PASS] Balanced strategy covers multiple tiers.")
    assert_frontier_bias(sampler)
    print("[PASS] Frontier strategy favors frontier datapacks.")
    assert_econ_bias(sampler)
    print("[PASS] Econ urgency strategy favors urgent datapacks.")
    assert_deterministic(sampler)
    print("[PASS] Determinism check (same seed → same ordering).")
    print("[smoke_test_stage3_sampler] All tests passed.")


if __name__ == "__main__":
    main()
