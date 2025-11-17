#!/usr/bin/env python3
"""
Fei-Fei Benchmark Evaluation Script.

Comprehensive evaluation of drawer+vase task under various conditions.

Evaluates 4 stack configurations:
1. Flat RL (no hierarchy, no affordance)
2. HRL only (hierarchical skills)
3. HRL + Affordance (vision heads)
4. HRL + Affordance + VLA + SIMA (full stack)

With perturbations:
- Vase location offset
- Lighting noise (vision mode)
- Occlusion
- Drawer friction variance

Metrics:
- Success rate
- Vase collision rate
- Mean clearance
- Sample efficiency
- Economic profit (Phase B integration)
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.envs.drawer_vase_physics_env import (
    DrawerVasePhysicsEnv,
    DrawerVaseConfig,
    summarize_drawer_vase_episode
)
from src.hrl.skills import SkillID, SkillParams
from src.hrl.low_level_policy import ScriptedSkillPolicy
from src.hrl.high_level_controller import (
    ScriptedHighLevelController,
    HierarchicalAgent
)
from src.hrl.skill_termination import SkillTerminationDetector
from src.sima.co_agent import SIMACoAgent
from src.config.internal_profile import get_internal_experiment_profile
from src.config.econ_params import load_econ_params
from src.valuation.datapacks import build_datapack_from_episode, build_datapack_meta_from_episode
from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import ObjectiveProfile


def create_perturbed_config(
    vase_offset=(0, 0, 0),
    drawer_friction_mult=1.0,
    lighting_noise=0.0,
    occlusion_prob=0.0
):
    """
    Create DrawerVaseConfig with perturbations.

    Args:
        vase_offset: (x, y, z) offset from default vase position
        drawer_friction_mult: Multiplier for drawer friction
        lighting_noise: Std of lighting noise (for vision mode)
        occlusion_prob: Probability of occlusion

    Returns:
        config: DrawerVaseConfig
    """
    config = DrawerVaseConfig()

    # Vase position perturbation
    config.vase_position = (
        0.3 + vase_offset[0],
        0.0 + vase_offset[1],
        0.8 + vase_offset[2]
    )

    # Drawer friction perturbation
    config.drawer_friction = 0.3 * drawer_friction_mult

    # Store for later use
    config.lighting_noise = lighting_noise
    config.occlusion_prob = occlusion_prob

    return config


class FlatRLPolicy:
    """
    Flat RL policy (no hierarchy).

    Uses simple scripted behavior without skill decomposition.
    """

    def __init__(self):
        self.step_count = 0
        self.phase = "approach"

    def reset(self):
        self.step_count = 0
        self.phase = "approach"

    def act(self, obs):
        """Generate action without hierarchical structure."""
        ee_pos = obs[0:3]
        drawer_frac = obs[6]
        vase_pos = obs[7:10]

        self.step_count += 1

        # Simple state machine (no skills)
        if self.phase == "approach":
            handle_pos = np.array([0.0, -0.42, 0.65])
            dist = np.linalg.norm(ee_pos - handle_pos)

            if dist < 0.05:
                self.phase = "pull"

            direction = handle_pos - ee_pos
            if np.linalg.norm(direction) > 0.01:
                action = direction / np.linalg.norm(direction) * 0.8
            else:
                action = np.zeros(3)

        elif self.phase == "pull":
            action = np.array([0.0, -0.6, 0.0])

            if drawer_frac >= 0.9:
                self.phase = "done"

        else:
            action = np.zeros(3)

        # Basic vase avoidance
        ee_to_vase = ee_pos - vase_pos
        dist_to_vase = np.linalg.norm(ee_to_vase)

        if dist_to_vase < 0.15:
            repulsion = ee_to_vase / (dist_to_vase + 1e-6) * 0.3
            action = action + repulsion

        return np.clip(action, -1, 1).astype(np.float32)


def evaluate_flat_rl(env, n_episodes=50):
    """
    Evaluate flat RL policy.

    Args:
        env: Environment
        n_episodes: Number of episodes

    Returns:
        metrics: Evaluation metrics
    """
    print("Evaluating Flat RL (no hierarchy)...")

    policy = FlatRLPolicy()
    successes = 0
    vase_collisions = 0
    clearances = []
    steps_list = []
    energy_list = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        policy.reset()

        info_history = []
        done = False
        step = 0

        while not done and step < 300:
            action = policy.act(obs)
            obs, _, done, _, info = env.step(action)
            info_history.append(info)
            step += 1

            if info.get('success', False):
                successes += 1
                break

            if not info.get('vase_intact', True):
                vase_collisions += 1
                break

        clearances.append(info.get('min_clearance', 0))
        steps_list.append(step)
        energy_list.append(info.get('energy_Wh', 0))

    return {
        'success_rate': successes / n_episodes,
        'collision_rate': vase_collisions / n_episodes,
        'mean_clearance': np.mean(clearances),
        'std_clearance': np.std(clearances),
        'mean_steps': np.mean(steps_list),
        'mean_energy': np.mean(energy_list),
    }


def evaluate_hrl_only(env, n_episodes=50):
    """
    Evaluate HRL with scripted skills (no vision heads).

    Args:
        env: Environment
        n_episodes: Number of episodes

    Returns:
        metrics: Evaluation metrics
    """
    print("Evaluating HRL Only...")

    pi_h = ScriptedHighLevelController()
    pi_l = ScriptedSkillPolicy()
    termination_detector = SkillTerminationDetector()

    agent = HierarchicalAgent(
        pi_h, pi_l, termination_detector, use_scripted_hl=True
    )

    successes = 0
    vase_collisions = 0
    clearances = []
    steps_list = []
    energy_list = []
    skill_sequences = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()

        done = False
        step = 0
        episode_skills = []

        while not done and step < 300:
            action, skill_info = agent.act(obs, info)

            if skill_info['task_done']:
                break

            if skill_info['skill_id'] not in episode_skills:
                episode_skills.append(skill_info['skill_id'])

            obs, _, env_done, _, info = env.step(action)
            step += 1

            if info.get('success', False):
                successes += 1
                done = True

            if not info.get('vase_intact', True):
                vase_collisions += 1
                done = True

            if env_done:
                done = True

        clearances.append(info.get('min_clearance', 0))
        steps_list.append(step)
        energy_list.append(info.get('energy_Wh', 0))
        skill_sequences.append(episode_skills)

    return {
        'success_rate': successes / n_episodes,
        'collision_rate': vase_collisions / n_episodes,
        'mean_clearance': np.mean(clearances),
        'std_clearance': np.std(clearances),
        'mean_steps': np.mean(steps_list),
        'mean_energy': np.mean(energy_list),
        'mean_skills': np.mean([len(s) for s in skill_sequences]),
    }


def evaluate_hrl_with_affordance(env, n_episodes=50):
    """
    Evaluate HRL with vision affordance heads.

    Args:
        env: Environment
        n_episodes: Number of episodes

    Returns:
        metrics: Evaluation metrics
    """
    print("Evaluating HRL + Affordance...")

    # For now, use same as HRL-only since vision heads need training
    # In full implementation, this would use trained vision heads

    pi_h = ScriptedHighLevelController()
    pi_l = ScriptedSkillPolicy()
    termination_detector = SkillTerminationDetector()

    agent = HierarchicalAgent(
        pi_h, pi_l, termination_detector, use_scripted_hl=True
    )

    successes = 0
    vase_collisions = 0
    clearances = []
    steps_list = []
    energy_list = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()

        done = False
        step = 0

        # Simulate affordance-guided behavior
        # Increase clearance threshold based on "affordance"
        affordance_clearance = 0.18  # Higher due to risk awareness

        while not done and step < 300:
            action, skill_info = agent.act(obs, info)

            if skill_info['task_done']:
                break

            # Affordance-based action modification
            min_clearance = obs[11]
            if min_clearance < affordance_clearance:
                # Apply stronger vase avoidance
                vase_pos = obs[7:10]
                ee_pos = obs[0:3]
                ee_to_vase = ee_pos - vase_pos
                dist = np.linalg.norm(ee_to_vase)

                if dist < affordance_clearance:
                    repulsion = ee_to_vase / (dist + 1e-6) * 0.4
                    action = action + repulsion
                    action = np.clip(action, -1, 1)

            obs, _, env_done, _, info = env.step(action)
            step += 1

            if info.get('success', False):
                successes += 1
                done = True

            if not info.get('vase_intact', True):
                vase_collisions += 1
                done = True

            if env_done:
                done = True

        clearances.append(info.get('min_clearance', 0))
        steps_list.append(step)
        energy_list.append(info.get('energy_Wh', 0))

    return {
        'success_rate': successes / n_episodes,
        'collision_rate': vase_collisions / n_episodes,
        'mean_clearance': np.mean(clearances),
        'std_clearance': np.std(clearances),
        'mean_steps': np.mean(steps_list),
        'mean_energy': np.mean(energy_list),
    }


def evaluate_full_stack(env, n_episodes=50):
    """
    Evaluate full stack: HRL + Affordance + VLA + SIMA.

    Args:
        env: Environment
        n_episodes: Number of episodes

    Returns:
        metrics: Evaluation metrics
    """
    print("Evaluating Full Stack (HRL + Affordance + VLA + SIMA)...")

    # Use SIMA co-agent for full stack evaluation
    co_agent = SIMACoAgent()

    successes = 0
    vase_collisions = 0
    clearances = []
    steps_list = []
    energy_list = []
    narration_counts = []

    instructions = [
        "open the drawer without hitting the vase",
        "carefully open the top drawer while avoiding fragile objects",
        "grasp handle and pull drawer open safely",
    ]

    for ep in range(n_episodes):
        instruction = instructions[ep % len(instructions)]
        trajectory = co_agent.generate_full_demonstration(env, instruction)

        if trajectory.success:
            successes += 1

        # Get final info
        if len(trajectory.infos) > 0:
            final_info = trajectory.infos[-1]
            if not final_info.get('vase_intact', True):
                vase_collisions += 1

            clearances.append(final_info.get('min_clearance', 0))
            energy_list.append(final_info.get('energy_Wh', 0))

        steps_list.append(trajectory.total_steps)
        narration_counts.append(len(trajectory.narrations))

    return {
        'success_rate': successes / n_episodes,
        'collision_rate': vase_collisions / n_episodes,
        'mean_clearance': np.mean(clearances) if clearances else 0,
        'std_clearance': np.std(clearances) if clearances else 0,
        'mean_steps': np.mean(steps_list),
        'mean_energy': np.mean(energy_list) if energy_list else 0,
        'mean_narrations': np.mean(narration_counts),
    }


def run_perturbation_experiments(
    n_episodes=20,
    vase_offsets=None,
    friction_mults=None,
    save_path=None
):
    """
    Run experiments with perturbations.

    Args:
        n_episodes: Episodes per configuration
        vase_offsets: List of (x, y, z) offsets
        friction_mults: List of friction multipliers
        save_path: Optional path to save results

    Returns:
        results: Dict with all experiment results
    """
    print("=" * 70)
    print("FEI-FEI BENCHMARK - PERTURBATION EXPERIMENTS")
    print("=" * 70)

    if vase_offsets is None:
        vase_offsets = [
            (0.0, 0.0, 0.0),   # Default
            (0.1, 0.0, 0.0),   # Shifted right
            (-0.1, 0.0, 0.0),  # Shifted left
            (0.0, 0.05, 0.0),  # Shifted forward
        ]

    if friction_mults is None:
        friction_mults = [1.0, 1.5, 2.0]

    results = defaultdict(dict)

    # Test each perturbation
    for vase_offset in vase_offsets:
        for friction_mult in friction_mults:
            config_name = f"vase_{vase_offset}_friction_{friction_mult}"
            print(f"\n--- Configuration: {config_name} ---")

            config = create_perturbed_config(
                vase_offset=vase_offset,
                drawer_friction_mult=friction_mult
            )
            env = DrawerVasePhysicsEnv(config=config, obs_mode='state')

            # Evaluate each stack
            flat_results = evaluate_flat_rl(env, n_episodes)
            hrl_results = evaluate_hrl_only(env, n_episodes)
            affordance_results = evaluate_hrl_with_affordance(env, n_episodes)
            full_results = evaluate_full_stack(env, n_episodes)

            results[config_name] = {
                'flat_rl': flat_results,
                'hrl_only': hrl_results,
                'hrl_affordance': affordance_results,
                'full_stack': full_results,
            }

            env.close()

            # Print summary
            print(f"  Flat RL:        {flat_results['success_rate']:.2%} success")
            print(f"  HRL Only:       {hrl_results['success_rate']:.2%} success")
            print(f"  HRL+Affordance: {affordance_results['success_rate']:.2%} success")
            print(f"  Full Stack:     {full_results['success_rate']:.2%} success")

    # Save results
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(dict(results), f, indent=2)
        print(f"\nSaved results to {save_path}")

    return results


def compute_economic_performance(results):
    """
    Compute economic performance metrics.

    Uses Phase B EconParams integration.

    Args:
        results: Experiment results

    Returns:
        econ_metrics: Economic performance by stack
    """
    from src.config.econ_params import load_econ_params
    from src.config.internal_profile import get_internal_experiment_profile

    profile = get_internal_experiment_profile("default")
    econ_params = load_econ_params(profile, preset="drawer_vase")

    econ_metrics = {}

    for config_name, stack_results in results.items():
        econ_metrics[config_name] = {}

        for stack_name, metrics in stack_results.items():
            # Compute profit per episode
            success_rate = metrics['success_rate']
            collision_rate = metrics['collision_rate']
            energy = metrics.get('mean_energy', 0)

            # Revenue: successful drawer opens
            revenue = success_rate * econ_params.value_per_successful_drawer_open

            # Costs: vase breaks + energy
            vase_cost = collision_rate * econ_params.vase_break_cost
            energy_cost = energy * econ_params.electricity_price_kWh / 1000  # Wh to kWh

            profit = revenue - vase_cost - energy_cost

            econ_metrics[config_name][stack_name] = {
                'revenue': revenue,
                'vase_cost': vase_cost,
                'energy_cost': energy_cost,
                'profit': profit,
            }

    return econ_metrics


def generate_benchmark_report(results, econ_metrics, save_path=None):
    """
    Generate comprehensive benchmark report.

    Args:
        results: Experiment results
        econ_metrics: Economic metrics
        save_path: Optional path to save report
    """
    print("\n" + "=" * 70)
    print("FEI-FEI BENCHMARK REPORT")
    print("=" * 70)

    # Aggregate across configurations
    stack_aggregates = defaultdict(lambda: defaultdict(list))

    for config_name, stack_results in results.items():
        for stack_name, metrics in stack_results.items():
            for metric_name, value in metrics.items():
                stack_aggregates[stack_name][metric_name].append(value)

    # Print summary table
    print("\nAGGREGATED RESULTS (mean across configurations):")
    print("-" * 70)
    print(f"{'Stack':<20} {'Success':>10} {'Collision':>10} {'Clearance':>10} {'Profit':>10}")
    print("-" * 70)

    for stack_name in ['flat_rl', 'hrl_only', 'hrl_affordance', 'full_stack']:
        agg = stack_aggregates[stack_name]

        mean_success = np.mean(agg['success_rate'])
        mean_collision = np.mean(agg['collision_rate'])
        mean_clearance = np.mean(agg['mean_clearance'])

        # Economic profit
        profits = []
        for config_name in results.keys():
            if stack_name in econ_metrics[config_name]:
                profits.append(econ_metrics[config_name][stack_name]['profit'])
        mean_profit = np.mean(profits) if profits else 0

        display_name = {
            'flat_rl': 'Flat RL',
            'hrl_only': 'HRL Only',
            'hrl_affordance': 'HRL + Affordance',
            'full_stack': 'Full Stack'
        }.get(stack_name, stack_name)

        print(f"{display_name:<20} {mean_success:>10.2%} {mean_collision:>10.2%} "
              f"{mean_clearance:>10.4f} ${mean_profit:>9.2f}")

    print("-" * 70)

    # Best stack
    best_stack = max(
        stack_aggregates.keys(),
        key=lambda s: np.mean(stack_aggregates[s]['success_rate'])
    )
    print(f"\nBest Overall: {best_stack} "
          f"({np.mean(stack_aggregates[best_stack]['success_rate']):.2%} success)")

    # Robustness analysis
    print("\nROBUSTNESS ANALYSIS (std across perturbations):")
    for stack_name in ['flat_rl', 'hrl_only', 'hrl_affordance', 'full_stack']:
        std_success = np.std(stack_aggregates[stack_name]['success_rate'])
        print(f"  {stack_name}: success_std = {std_success:.4f}")

    # Save report
    if save_path:
        report = {
            'aggregated': {
                stack: {k: float(np.mean(v)) for k, v in metrics.items()}
                for stack, metrics in stack_aggregates.items()
            },
            'economic': econ_metrics,
            'best_stack': best_stack,
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {save_path}")


def collect_datapacks(env, econ_params, n_episodes=5, use_repo=True):
    """Run scripted HRL rollout and export datapacks."""
    pi_h = ScriptedHighLevelController()
    pi_l = ScriptedSkillPolicy()
    termination_detector = SkillTerminationDetector()
    agent = HierarchicalAgent(pi_h, pi_l, termination_detector, use_scripted_hl=True)

    datapacks = []
    datapack_metas = []

    # Compute baseline from first episode
    baseline_mpl = None
    baseline_error = None
    baseline_ep = None

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()
        info_history = []
        done = False
        while not done:
            action, skill_info = agent.act(obs, info)
            obs, _, env_done, _, info = env.step(action)
            info_history.append(info)
            if env_done or info.get("success", False) or not info.get("vase_intact", True):
                done = True
        summary = summarize_drawer_vase_episode(info_history)

        # Set baseline from first episode
        if ep == 0:
            baseline_mpl = summary.mpl_episode
            baseline_error = summary.error_rate_episode
            baseline_ep = summary.ep_episode

        # Build unified DataPackMeta
        if use_repo:
            # Build ObjectiveProfile for econ profile logging
            obj_profile = ObjectiveProfile(
                objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],  # balanced
                wage_human=18.0,
                energy_price_kWh=0.12,
                market_region="US",
                task_family="drawer_vase",
                customer_segment="balanced",
                baseline_mpl_human=20.0,
                baseline_error_human=0.05,
                env_name="drawer_vase",
                engine_type="pybullet",
                task_type="fragility",
                econ_profile_deltas=None,
                econ_params_effective={
                    "base_rate": float(econ_params.base_rate),
                    "damage_cost": float(econ_params.damage_cost),
                    "care_cost": float(econ_params.care_cost),
                    "energy_Wh_per_attempt": float(econ_params.energy_Wh_per_attempt),
                    "max_steps": int(econ_params.max_steps),
                },
                reward_weights=None,
            )

            dp_meta = build_datapack_meta_from_episode(
                summary, econ_params,
                condition_profile={"task": "drawer_vase", "tags": ["feifei_benchmark"], "engine_type": "pybullet"},
                agent_profile={"policy": "hrl_scripted", "version": "v1"},
                brick_id=f"feifei_benchmark_{ep:04d}",
                env_type="drawer_vase",
                baseline_mpl=baseline_mpl,
                baseline_error=baseline_error,
                baseline_ep=baseline_ep,
                objective_profile=obj_profile,
            )
            datapack_metas.append(dp_meta)

        # Legacy format for backwards compatibility
        datapacks.append(build_datapack_from_episode(
            summary,
            econ_params,
            condition_profile={"task": "drawer_vase", "tags": ["feifei_benchmark"]},
            agent_profile={"policy": "hrl_scripted"},
            brick_id=None,
            env_type="drawer_vase",
        ))

    # Write to DataPackRepo if enabled
    if use_repo and datapack_metas:
        repo_dir = "data/datapacks/phase_c"
        repo = DataPackRepo(base_dir=repo_dir)
        repo.append_batch(datapack_metas)
        stats = repo.get_statistics("drawer_vase")
        print(f"\nSaved {len(datapack_metas)} datapacks to DataPackRepo ({repo_dir})")
        print(f"  Total in repo: {stats['total']}")
        print(f"  Positive: {stats['positive']}, Negative: {stats['negative']}")

    return datapacks


def main():
    parser = argparse.ArgumentParser(description='Fei-Fei Benchmark Evaluation')
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Episodes per configuration'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='results/feifei_benchmark',
        help='Directory to save results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick evaluation with fewer configurations'
    )
    parser.add_argument(
        '--emit-datapacks',
        type=str,
        default=None,
        help='Path to write datapacks (Phase C)'
    )
    parser.add_argument(
        '--datapack-episodes',
        type=int,
        default=5,
        help='Number of episodes for datapack export'
    )
    parser.add_argument(
        '--no-repo',
        action='store_true',
        help='Skip writing to DataPackRepo'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FEI-FEI BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Episodes per config: {args.episodes}")
    print()

    # Define perturbations
    if args.quick:
        vase_offsets = [(0.0, 0.0, 0.0)]
        friction_mults = [1.0]
    else:
        vase_offsets = [
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (-0.1, 0.0, 0.0),
        ]
        friction_mults = [1.0, 1.5]

    # Run experiments
    results_path = os.path.join(args.save_dir, 'perturbation_results.json')
    results = run_perturbation_experiments(
        n_episodes=args.episodes,
        vase_offsets=vase_offsets,
        friction_mults=friction_mults,
        save_path=results_path
    )

    # Compute economic performance
    econ_metrics = compute_economic_performance(results)

    # Generate report
    report_path = os.path.join(args.save_dir, 'benchmark_report.json')
    generate_benchmark_report(results, econ_metrics, report_path)

    # Optional datapack export (Phase C → valuation)
    if args.emit_datapacks or not args.no_repo:
        profile = get_internal_experiment_profile("dishwashing")
        econ_params = load_econ_params(profile, preset=profile.get("econ_preset", "toy"))
        config = DrawerVaseConfig()
        env = DrawerVasePhysicsEnv(
            config=config,
            obs_mode='state',
            render_mode=None
        )
        datapacks = collect_datapacks(
            env, econ_params,
            n_episodes=args.datapack_episodes,
            use_repo=not args.no_repo
        )

        # Legacy JSON output if requested
        if args.emit_datapacks:
            os.makedirs(os.path.dirname(args.emit_datapacks), exist_ok=True)
            with open(args.emit_datapacks, 'w') as f:
                json.dump(datapacks, f, indent=2)
            print(f"Wrote legacy datapacks to {args.emit_datapacks}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
