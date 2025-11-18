#!/usr/bin/env python3
"""
Smoke test for Stage 3.2 DataPackCurriculum.

Validates:
- Phase boundaries -> expected phase names
- Strategy selection per phase (balanced/mixed/frontier/econ)
- Critical-priority episodes are included during fine-tuning
- Determinism for fixed seed + config
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.rl.episode_sampling import DataPackRLSampler
from src.rl.curriculum import DataPackCurriculum


def _make_descriptor(
    pack_id: str,
    tier: int,
    trust: float,
    delta_mpl: float,
    delta_J: float,
    weight: float,
) -> dict:
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


def assert_phase_boundaries(curriculum: DataPackCurriculum):
    expected = {
        0: "warmup",
        100: "warmup",
        149: "warmup",
        150: "skill_building",
        300: "skill_building",
        499: "skill_building",
        500: "frontier",
        799: "frontier",
        800: "fine_tuning",
        950: "fine_tuning",
        999: "fine_tuning",
    }
    for step, phase in expected.items():
        assert curriculum.get_phase(step) == phase, f"Phase mismatch at step {step}"


def _strategy_counts(batch):
    counts = {}
    for item in batch:
        strat = item.get("sampling_metadata", {}).get("strategy")
        counts[strat] = counts.get(strat, 0) + 1
    return counts


def assert_phase_behavior(curriculum: DataPackCurriculum):
    # Warmup: only balanced
    warmup_batch = curriculum.sample_batch(step=42, batch_size=5)
    warmup_strats = set(_strategy_counts(warmup_batch).keys())
    assert warmup_strats == {"balanced"}, f"Warmup strategies unexpected: {warmup_strats}"

    # Skill building: must include some frontier prioritization
    skill_batch = curriculum.sample_batch(step=200, batch_size=5)
    skill_counts = _strategy_counts(skill_batch)
    assert skill_counts.get("frontier_prioritized", 0) > 0, "Skill building should include frontier samples"
    assert skill_counts.get("balanced", 0) > 0, "Skill building should retain balanced samples"

    # Frontier: majority frontier_prioritized with econ support
    frontier_batch = curriculum.sample_batch(step=650, batch_size=6)
    frontier_counts = _strategy_counts(frontier_batch)
    assert frontier_counts.get("frontier_prioritized", 0) >= 3, "Frontier phase should emphasize frontier prioritization"
    assert frontier_counts.get("econ_urgency", 0) >= 1, "Frontier phase should include econ urgency support"

    # Fine tuning: econ only and must include critical episodes
    finetune_batch = curriculum.sample_batch(step=900, batch_size=4)
    finetune_strats = set(_strategy_counts(finetune_batch).keys())
    assert finetune_strats == {"econ_urgency"}, f"Fine tuning should use only econ_urgency: {finetune_strats}"
    critical_ids = {"ep_tier2_frontier", "ep_tier1_econ_hot"}
    finetune_ids = {b["pack_id"] for b in finetune_batch}
    assert finetune_ids & critical_ids, "Fine tuning should include at least one critical-priority episode"


def assert_determinism(curriculum_factory):
    steps = [42, 200, 650, 900]
    first = []
    second = []
    cur1 = curriculum_factory()
    cur2 = curriculum_factory()
    for s in steps:
        first.append(cur1.sample_batch(step=s, batch_size=5))
        second.append(cur2.sample_batch(step=s, batch_size=5))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True), "Curriculum sampling should be deterministic"


def main():
    descriptors, enrichments = build_fixture()
    sampler = DataPackRLSampler(
        datapacks=None,
        enrichments=enrichments,
        existing_descriptors=descriptors,
    )
    curriculum = DataPackCurriculum(sampler=sampler, total_steps=1000, config={"base_seed": 123})

    print("[Stage3 Curriculum] Pool summary:", sampler.pool_summary())
    assert_phase_boundaries(curriculum)
    print("[PASS] Phase boundaries map to expected phases.")
    assert_phase_behavior(curriculum)
    print("[PASS] Phase strategies behave as expected.")

    def factory():
        s = DataPackRLSampler(None, enrichments=enrichments, existing_descriptors=descriptors)
        return DataPackCurriculum(s, total_steps=1000, config={"base_seed": 123})

    assert_determinism(factory)
    print("[PASS] Determinism check passed.")
    print("[smoke_test_stage3_curriculum] All tests passed.")


if __name__ == "__main__":
    main()
