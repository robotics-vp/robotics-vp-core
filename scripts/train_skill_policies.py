#!/usr/bin/env python3
"""
Train Low-Level Skill Policies for HRL.

Trains individual skill policies (π_L) for each skill in the drawer+vase task.
Uses PPO with shaped rewards specific to each skill.

Usage:
    python scripts/train_skill_policies.py --skill all --steps 100000
    python scripts/train_skill_policies.py --skill GRASP_HANDLE --steps 50000
"""

import os
import sys
import argparse
import json
import time
import numpy as np

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.envs.drawer_vase_physics_env import DrawerVasePhysicsEnv, DrawerVaseConfig
from src.hrl.skills import SkillID, SkillParams
from src.hrl.low_level_policy import LowLevelSkillPolicy, ScriptedSkillPolicy
from src.hrl.skill_termination import SkillTerminationDetector, SkillRewardShaper


def train_single_skill(
    skill_id,
    total_steps=100000,
    steps_per_update=2048,
    lr=3e-4,
    device='cpu',
    save_dir='checkpoints/hrl/skills'
):
    """
    Train a single skill policy.

    Args:
        skill_id: SkillID to train
        total_steps: Total training steps
        steps_per_update: Steps per PPO update
        lr: Learning rate
        device: Training device
        save_dir: Directory to save checkpoints

    Returns:
        metrics: Training metrics
    """
    print("=" * 70)
    print(f"TRAINING SKILL: {SkillID.name(skill_id)}")
    print("=" * 70)

    # Create environment
    config = DrawerVaseConfig()
    env = DrawerVasePhysicsEnv(config=config, obs_mode='state')

    # Create policy
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Using scripted policy for demonstration.")
        policy = ScriptedSkillPolicy()
        print(f"Scripted policy created for {SkillID.name(skill_id)}")
        env.close()
        return {'skill_id': skill_id, 'success': True, 'method': 'scripted'}

    policy = LowLevelSkillPolicy(
        obs_dim=13,
        action_dim=3,
        num_skills=SkillID.NUM_SKILLS
    ).to(device)

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create trainer
    from src.hrl.hrl_trainer import SkillTrainer

    trainer = SkillTrainer(
        env=env,
        skill_id=skill_id,
        policy=policy,
        lr=lr,
        device=device
    )

    # Train
    start_time = time.time()
    metrics = trainer.train(
        total_steps=total_steps,
        steps_per_update=steps_per_update,
        log_interval=10
    )
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Final Success Rate: {metrics['success_rate']:.2%}")
    print(f"Final Mean Reward: {metrics['mean_episode_reward']:.4f}")

    # Save policy
    os.makedirs(save_dir, exist_ok=True)
    skill_name = SkillID.name(skill_id).lower()
    save_path = os.path.join(save_dir, f"{skill_name}.pt")
    trainer.save(save_path)

    # Save metrics
    metrics['skill_id'] = skill_id
    metrics['skill_name'] = SkillID.name(skill_id)
    metrics['training_time'] = training_time
    metrics['total_steps'] = total_steps

    metrics_path = os.path.join(save_dir, f"{skill_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    env.close()
    return metrics


def train_all_skills(
    total_steps_per_skill=100000,
    steps_per_update=2048,
    lr=3e-4,
    device='cpu',
    save_dir='checkpoints/hrl/skills'
):
    """
    Train all skill policies.

    Args:
        total_steps_per_skill: Steps for each skill
        steps_per_update: Steps per PPO update
        lr: Learning rate
        device: Training device
        save_dir: Save directory

    Returns:
        all_metrics: List of metrics for each skill
    """
    print("=" * 70)
    print("TRAINING ALL SKILL POLICIES")
    print("=" * 70)

    all_metrics = []

    for skill_id in SkillID.all_ids():
        print(f"\n[{skill_id + 1}/{SkillID.NUM_SKILLS}] Training {SkillID.name(skill_id)}")

        metrics = train_single_skill(
            skill_id=skill_id,
            total_steps=total_steps_per_skill,
            steps_per_update=steps_per_update,
            lr=lr,
            device=device,
            save_dir=save_dir
        )

        all_metrics.append(metrics)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for metrics in all_metrics:
        skill_name = metrics.get('skill_name', f"Skill {metrics['skill_id']}")
        success_rate = metrics.get('success_rate', 0)
        print(f"{skill_name}: {success_rate:.2%} success")

    # Save combined metrics
    os.makedirs(save_dir, exist_ok=True)
    combined_path = os.path.join(save_dir, "all_skills_metrics.json")
    with open(combined_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved combined metrics to {combined_path}")

    return all_metrics


def evaluate_scripted_baseline(n_episodes=50):
    """
    Evaluate scripted skill policies as baseline.

    Args:
        n_episodes: Number of episodes to evaluate
    """
    print("=" * 70)
    print("EVALUATING SCRIPTED SKILL BASELINE")
    print("=" * 70)

    config = DrawerVaseConfig()
    env = DrawerVasePhysicsEnv(config=config, obs_mode='state')

    policy = ScriptedSkillPolicy()
    termination_detector = SkillTerminationDetector()
    reward_shaper = SkillRewardShaper()

    # Test each skill
    skill_results = {}

    for skill_id in SkillID.all_ids():
        print(f"\nTesting {SkillID.name(skill_id)}...")

        successes = 0
        total_rewards = []
        episode_lengths = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            skill_params = SkillParams.default_for_skill(skill_id)

            total_reward = 0
            step = 0
            done = False

            while not done and step < skill_params.timeout_steps:
                action = policy.act(obs, skill_id, skill_params)
                next_obs, _, env_done, _, info = env.step(action)

                # Shaped reward
                reward, _ = reward_shaper.compute_reward(
                    skill_id, obs, next_obs, action, info, skill_params, step
                )
                total_reward += reward

                # Check skill termination
                skill_done, skill_success, reason = termination_detector.is_done(
                    skill_id, next_obs, info, step, skill_params.timeout_steps
                )

                obs = next_obs
                step += 1

                if skill_done or env_done:
                    if skill_success:
                        successes += 1
                    done = True

            total_rewards.append(total_reward)
            episode_lengths.append(step)

        success_rate = successes / n_episodes
        mean_reward = np.mean(total_rewards)
        mean_length = np.mean(episode_lengths)

        skill_results[SkillID.name(skill_id)] = {
            'success_rate': success_rate,
            'mean_reward': mean_reward,
            'mean_length': mean_length,
        }

        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Mean Reward: {mean_reward:.4f}")
        print(f"  Mean Length: {mean_length:.1f} steps")

    env.close()

    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = 'results/scripted_skill_baseline.json'
    with open(results_path, 'w') as f:
        json.dump(skill_results, f, indent=2)
    print(f"\nSaved baseline results to {results_path}")

    return skill_results


def main():
    parser = argparse.ArgumentParser(description='Train HRL Skill Policies')
    parser.add_argument(
        '--skill',
        type=str,
        default='all',
        help='Skill to train (LOCATE_DRAWER, GRASP_HANDLE, etc.) or "all" or "baseline"'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100000,
        help='Total training steps per skill'
    )
    parser.add_argument(
        '--update-steps',
        type=int,
        default=2048,
        help='Steps per PPO update'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Training device (cpu or cuda)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='checkpoints/hrl/skills',
        help='Directory to save checkpoints'
    )

    args = parser.parse_args()

    if args.skill.lower() == 'baseline':
        evaluate_scripted_baseline()
    elif args.skill.lower() == 'all':
        train_all_skills(
            total_steps_per_skill=args.steps,
            steps_per_update=args.update_steps,
            lr=args.lr,
            device=args.device,
            save_dir=args.save_dir
        )
    else:
        # Train specific skill
        skill_name = args.skill.upper()
        skill_id = getattr(SkillID, skill_name, None)

        if skill_id is None:
            print(f"Unknown skill: {args.skill}")
            print(f"Available skills: {[SkillID.name(i) for i in SkillID.all_ids()]}")
            sys.exit(1)

        train_single_skill(
            skill_id=skill_id,
            total_steps=args.steps,
            steps_per_update=args.update_steps,
            lr=args.lr,
            device=args.device,
            save_dir=args.save_dir
        )


if __name__ == '__main__':
    main()
