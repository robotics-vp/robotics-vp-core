#!/usr/bin/env python3
"""
Smoke test for reward_builder (no training behavior change).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.valuation.reward_builder import build_reward_terms, combine_reward, default_objective_vector
from src.config.econ_params import EconParams
from src.envs.dishwashing_env import EpisodeInfoSummary


def main():
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
    summary = EpisodeInfoSummary(
        termination_reason="max_steps",
        mpl_episode=100.0,
        ep_episode=10.0,
        error_rate_episode=0.05,
        throughput_units_per_hour=100.0,
        energy_Wh=5.0,
        energy_Wh_per_unit=0.05,
        energy_Wh_per_hour=50.0,
        limb_energy_Wh={},
        skill_energy_Wh={},
        energy_per_limb={},
        energy_per_skill={},
        energy_per_joint={},
        energy_per_effector={},
        coordination_metrics={},
        profit=0.0,
        episode_id="test_episode",
        media_refs={},
        wage_parity=None,
    )
    terms = build_reward_terms(summary, econ)
    obj = default_objective_vector()
    total = combine_reward(obj, terms)
    print("Reward terms:", terms)
    print("Objective vector:", obj)
    print("Combined reward:", total)


if __name__ == "__main__":
    main()
