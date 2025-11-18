#!/usr/bin/env python3
"""
Smoke test for Stage 3 training integration (flag-gated).

Runs a short SAC training loop with datapack curriculum enabled and verifies:
- No crashes
- Phases progress according to total_steps
- Rewards/losses remain finite
"""

import csv
import os
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from train_sac import train_sac
from src.rl.episode_sampling import DataPackRLSampler
from src.rl.curriculum import DataPackCurriculum


def _make_descriptor(pack_id, tier, trust, delta_mpl, delta_J, weight):
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
        "semantic_tags": ["smoke"],
    }


def _make_enrichment(episode_id, novelty, gain, priority="medium"):
    return {
        "episode_id": episode_id,
        "enrichment": {
            "novelty_tags": [{"novelty_score": novelty, "expected_mpl_gain": gain, "comparison_basis": "smoke"}],
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


def build_sampler_curriculum():
    descriptors = [
        _make_descriptor("ep_warmup", 0, 0.5, 0.1, 0.05, 0.8),
        _make_descriptor("ep_skill", 1, 0.7, 0.6, 0.3, 1.2),
        _make_descriptor("ep_frontier", 2, 0.8, 1.4, 0.9, 1.8),
        _make_descriptor("ep_critical", 2, 0.9, 1.6, 1.1, 2.0),
    ]
    enrichments = [
        _make_enrichment("ep_warmup", novelty=0.1, gain=0.0),
        _make_enrichment("ep_skill", novelty=0.4, gain=0.8, priority="high"),
        _make_enrichment("ep_frontier", novelty=0.8, gain=1.5, priority="high"),
        _make_enrichment("ep_critical", novelty=0.9, gain=1.8, priority="critical"),
    ]
    sampler = DataPackRLSampler(datapacks=None, enrichments=enrichments, existing_descriptors=descriptors)
    curriculum = DataPackCurriculum(sampler=sampler, total_steps=8, config={"base_seed": 0})
    return sampler, curriculum


def read_log(log_path):
    rows = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    sampler, curriculum = build_sampler_curriculum()
    log_path = "logs/sac_stage3_curriculum_smoke.csv"
    checkpoint_path = "checkpoints/sac_stage3_curriculum_smoke.pt"

    # Clean old artifacts for deterministic smoke logging
    for path in [log_path, checkpoint_path]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    train_sac(
        episodes=8,
        seed=0,
        econ_preset="toy",
        use_datapack_curriculum=True,
        sampler_mode=None,
        curriculum_total_steps=8,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        sampler=sampler,
        curriculum=curriculum,
    )

    rows = read_log(log_path)
    assert len(rows) >= 8, "Expected log rows for each episode"
    phases = [r.get("curriculum_phase") for r in rows[:8]]
    assert "warmup" in phases and "skill_building" in phases and "frontier" in phases and "fine_tuning" in phases, \
        f"Unexpected phases observed: {set(phases)}"
    # Losses/rewards finite
    for r in rows:
        reward = float(r.get("episode_reward", 0.0))
        critic_loss = float(r.get("critic_loss", 0.0))
        actor_loss = float(r.get("actor_loss", 0.0))
        assert reward == reward, "NaN reward detected"
        assert critic_loss == critic_loss, "NaN critic loss detected"
        assert actor_loss == actor_loss, "NaN actor loss detected"
    print("[smoke_test_stage3_training_integration] All tests passed.")


if __name__ == "__main__":
    main()
