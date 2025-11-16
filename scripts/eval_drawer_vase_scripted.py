#!/usr/bin/env python3
"""
Evaluate Scripted Policy on Drawer+Vase Environment

Runs the scripted baseline policy for N episodes and logs:
- Success rate
- Vase collision rate
- Clearance distribution
- Energy usage
- EpisodeInfoSummary
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from src.envs.drawer_vase_physics_env import (
    DrawerVasePhysicsEnv,
    DrawerVaseConfig,
    summarize_drawer_vase_episode
)
from policies.scripted.drawer_open_avoid_vase import DrawerOpenAvoidVasePolicy
from src.config.internal_profile import get_internal_experiment_profile
from src.config.econ_params import load_econ_params
from src.valuation.datapacks import build_datapack_from_episode


def evaluate_scripted_policy(n_episodes=20, render=False, datapack_path=None):
    """
    Evaluate scripted policy on drawer+vase task.

    Args:
        n_episodes: Number of episodes to run
        render: Whether to render (requires PyBullet GUI)

    Returns:
        results: Dictionary with evaluation metrics
    """
    print("=" * 70)
    print("EVALUATING SCRIPTED DRAWER+VASE POLICY")
    print("=" * 70)
    print(f"Episodes: {n_episodes}")
    print()

    # Create environment
    config = DrawerVaseConfig()
    env = DrawerVasePhysicsEnv(
        config=config,
        obs_mode='state',
        render_mode='human' if render else None
    )
    profile = get_internal_experiment_profile("dishwashing")
    econ_params = load_econ_params(profile, preset=profile.get("econ_preset", "toy"))

    # Create policy
    policy = DrawerOpenAvoidVasePolicy()

    # Track metrics
    successes = 0
    vase_collisions = 0
    clearances = []
    energy_usages = []
    episode_lengths = []
    termination_reasons = {}
    all_summaries = []

    datapacks = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        policy.reset()

        info_history = []
        done = False

        while not done:
            action = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            info_history.append(info)

        # Summarize episode
        summary = summarize_drawer_vase_episode(info_history)
        all_summaries.append(summary)
        if datapack_path:
            datapacks.append(build_datapack_from_episode(
                summary, econ_params,
                condition_profile={"task": "drawer_vase", "tags": []},
                agent_profile={"policy": "scripted"},
                brick_id=None
            ))

        # Track metrics
        if info.get('success', False):
            successes += 1

        if not info.get('vase_intact', True):
            vase_collisions += 1

        clearances.append(info.get('min_clearance', 0.0))
        energy_usages.append(info.get('energy_Wh', 0.0))
        episode_lengths.append(info.get('steps', 0))

        reason = info.get('terminated_reason', 'unknown')
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

        print(f"Episode {ep+1}/{n_episodes}: "
              f"{reason}, drawer={info.get('drawer_fraction', 0):.2f}, "
              f"clearance={info.get('min_clearance', 0):.4f}, "
              f"vase_intact={info.get('vase_intact', True)}")

    env.close()

    # Compute statistics
    success_rate = successes / n_episodes
    collision_rate = vase_collisions / n_episodes
    mean_clearance = np.mean(clearances)
    std_clearance = np.std(clearances)
    min_clearance = np.min(clearances)
    mean_energy = np.mean(energy_usages)
    mean_length = np.mean(episode_lengths)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Success rate: {success_rate:.2%} ({successes}/{n_episodes})")
    print(f"Vase collision rate: {collision_rate:.2%} ({vase_collisions}/{n_episodes})")
    print(f"Clearance: mean={mean_clearance:.4f}, std={std_clearance:.4f}, min={min_clearance:.4f}")
    print(f"Energy usage: mean={mean_energy:.4f} Wh")
    print(f"Episode length: mean={mean_length:.1f} steps")
    print(f"\nTermination reasons:")
    for reason, count in sorted(termination_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/n_episodes:.1%})")

    # Aggregate EpisodeInfoSummary
    print("\nEpisodeInfoSummary (aggregate):")
    mpl_episodes = [s.mpl_episode for s in all_summaries]
    ep_episodes = [s.ep_episode for s in all_summaries]
    error_rates = [s.error_rate_episode for s in all_summaries]
    profits = [s.profit for s in all_summaries]

    print(f"  MPL (units/hr): mean={np.mean(mpl_episodes):.4f}, std={np.std(mpl_episodes):.4f}")
    print(f"  EP (units/Wh): mean={np.mean(ep_episodes):.4f}, std={np.std(ep_episodes):.4f}")
    print(f"  Error rate: mean={np.mean(error_rates):.4f}, std={np.std(error_rates):.4f}")
    print(f"  Profit: mean={np.mean(profits):.4f}, std={np.std(profits):.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'n_episodes': n_episodes,
        'success_rate': float(success_rate),
        'collision_rate': float(collision_rate),
        'clearance_mean': float(mean_clearance),
        'clearance_std': float(std_clearance),
        'clearance_min': float(min_clearance),
        'clearance_all': [float(c) for c in clearances],
        'energy_mean': float(mean_energy),
        'episode_length_mean': float(mean_length),
        'termination_reasons': termination_reasons,
        'episode_summaries': [
            {
                'termination_reason': s.termination_reason,
                'mpl_episode': float(s.mpl_episode),
                'ep_episode': float(s.ep_episode),
                'error_rate_episode': float(s.error_rate_episode),
                'energy_Wh': float(s.energy_Wh),
                'profit': float(s.profit),
            }
            for s in all_summaries
        ]
    }

    with open('results/drawer_vase_scripted_eval.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to results/drawer_vase_scripted_eval.json")

    if datapack_path:
        os.makedirs(os.path.dirname(datapack_path), exist_ok=True)
        with open(datapack_path, 'w') as f:
            json.dump(datapacks, f, indent=2)
        print(f"Saved datapacks to {datapack_path}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--emit-datapacks', type=str, default=None,
                        help='Path to write datapacks JSON')
    args = parser.parse_args()

    evaluate_scripted_policy(n_episodes=args.episodes, render=args.render, datapack_path=args.emit_datapacks)
