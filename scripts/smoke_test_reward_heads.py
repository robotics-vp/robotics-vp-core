#!/usr/bin/env python3
"""
Smoke test for multi_reward_heads (no training hookup).
"""
from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams
from src.rl.multi_reward_heads.mpl_head import MPLRewardHead
from src.rl.multi_reward_heads.energy_head import EnergyRewardHead
from src.rl.multi_reward_heads.damage_head import DamageRewardHead
from src.rl.multi_reward_heads.novelty_head import NoveltyRewardHead
from src.rl.multi_reward_heads.wage_parity_head import WageParityRewardHead


def main():
    summary = EpisodeInfoSummary(
        termination_reason="success",
        mpl_episode=120.0,
        ep_episode=0.0,
        error_rate_episode=0.05,
        throughput_units_per_hour=120.0,
        energy_Wh=4.0,
        energy_Wh_per_unit=0.033,
        energy_Wh_per_hour=40.0,
        limb_energy_Wh={},
        skill_energy_Wh={},
        energy_per_limb={},
        energy_per_skill={},
        energy_per_joint={},
        energy_per_effector={},
        coordination_metrics={},
        profit=0.0,
        episode_id="",
        media_refs={},
        wage_parity=1.2,
    )
    econ = EconParams(
        price_per_unit=0.3,
        damage_cost=1.0,
        energy_Wh_per_attempt=0.05,
        time_step_s=60.0,
        base_rate=2.0,
        p_min=0.02,
        k_err=0.12,
        q_speed=1.2,
        q_care=1.5,
        care_cost=0.25,
        max_steps=240,
        max_catastrophic_errors=3,
        max_error_rate_sla=0.12,
        min_steps_for_sla=5,
        zero_throughput_patience=10,
        preset="toy",
    )
    heads = [
        ("mpl", MPLRewardHead()),
        ("energy", EnergyRewardHead()),
        ("damage", DamageRewardHead()),
        ("novelty", NoveltyRewardHead()),
        ("wage_parity", WageParityRewardHead()),
    ]
    print("Reward head outputs:")
    for name, head in heads:
        print(f"{name}: {head.compute(summary, econ)}")


if __name__ == "__main__":
    main()
