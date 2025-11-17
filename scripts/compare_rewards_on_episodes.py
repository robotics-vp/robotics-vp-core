#!/usr/bin/env python3
"""
Offline comparison of legacy vs objective rewards on EpisodeInfoSummary logs.
"""
import argparse
import json
import numpy as np

from src.valuation.reward_builder import build_reward_terms, combine_reward, default_objective_vector
from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams


def load_episode(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Try to extract metrics
    metrics = data.get("episode_metrics", data)
    econ = data.get("econ_params", {})
    econ_params = EconParams(
        price_per_unit=econ.get("price_per_unit", 0.3),
        damage_cost=econ.get("damage_cost", 1.0),
        energy_Wh_per_attempt=econ.get("energy_Wh_per_attempt", 0.05),
        time_step_s=econ.get("time_step_s", 60.0),
        base_rate=econ.get("base_rate", 2.0),
        p_min=econ.get("p_min", 0.02),
        k_err=econ.get("k_err", 0.12),
        q_speed=econ.get("q_speed", 1.2),
        q_care=econ.get("q_care", 1.5),
        care_cost=econ.get("care_cost", 0.25),
        max_steps=econ.get("max_steps", 240),
        max_catastrophic_errors=econ.get("max_catastrophic_errors", 3),
        max_error_rate_sla=econ.get("max_error_rate_sla", 0.12),
        min_steps_for_sla=econ.get("min_steps_for_sla", 5),
        zero_throughput_patience=econ.get("zero_throughput_patience", 10),
        preset=econ.get("preset", "toy"),
    )
    summary = EpisodeInfoSummary(**metrics)
    return summary, econ_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_files", nargs="+", help="JSON files with EpisodeInfoSummary data")
    args = parser.parse_args()

    legacy_rewards = []
    objective_rewards = []
    obj_vec = default_objective_vector()
    for path in args.episode_files:
        summary, econ_params = load_episode(path)
        # Legacy approximation: use mpl - error - energy_Wh
        legacy = summary.mpl_episode - summary.error_rate_episode - summary.energy_Wh
        terms = build_reward_terms(summary, econ_params)
        obj_reward = combine_reward(obj_vec, terms)
        legacy_rewards.append(legacy)
        objective_rewards.append(obj_reward)

    legacy_arr = np.array(legacy_rewards)
    obj_arr = np.array(objective_rewards)
    corr = np.corrcoef(legacy_arr, obj_arr)[0, 1] if len(legacy_arr) > 1 else 0.0
    print(f"Episodes compared: {len(legacy_rewards)}")
    print(f"Legacy mean={legacy_arr.mean():.3f} std={legacy_arr.std():.3f}")
    print(f"Objective mean={obj_arr.mean():.3f} std={obj_arr.std():.3f}")
    print(f"Correlation: {corr:.3f}")
    for pct in [0, 25, 50, 75, 100]:
        lp = np.percentile(legacy_arr, pct)
        op = np.percentile(obj_arr, pct)
        print(f"P{pct}: legacy={lp:.3f} objective={op:.3f}")


if __name__ == "__main__":
    main()
