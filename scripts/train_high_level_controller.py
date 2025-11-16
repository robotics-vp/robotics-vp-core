#!/usr/bin/env python3
"""
Train High-Level Controller (π_H) for HRL.

Trains the skill selection policy that chooses which skills to execute
and their parameters.

Usage:
    python scripts/train_high_level_controller.py --episodes 1000
    python scripts/train_high_level_controller.py --use-scripted-ll
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
from src.hrl.high_level_controller import (
    HighLevelController,
    ScriptedHighLevelController,
    HierarchicalAgent
)
from src.hrl.skill_termination import SkillTerminationDetector


def train_high_level_controller(
    n_episodes=1000,
    use_scripted_ll=True,
    lr=3e-4,
    gamma=0.99,
    device='cpu',
    save_path='checkpoints/hrl/high_level_controller.pt',
    skill_checkpoints_dir='checkpoints/hrl/skills'
):
    """
    Train high-level controller.

    Args:
        n_episodes: Number of training episodes
        use_scripted_ll: Use scripted low-level policies
        lr: Learning rate
        gamma: Discount factor
        device: Training device
        save_path: Path to save controller
        skill_checkpoints_dir: Directory with trained skill policies

    Returns:
        metrics: Training metrics
    """
    print("=" * 70)
    print("TRAINING HIGH-LEVEL CONTROLLER")
    print("=" * 70)

    # Create environment
    config = DrawerVaseConfig()
    env = DrawerVasePhysicsEnv(config=config, obs_mode='state')

    # Create low-level policy
    if use_scripted_ll:
        print("Using scripted low-level policies")
        pi_l = ScriptedSkillPolicy()
    else:
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Falling back to scripted policies.")
            pi_l = ScriptedSkillPolicy()
        else:
            print("Loading trained low-level policies...")
            # Load shared policy (trained on all skills)
            pi_l = LowLevelSkillPolicy().to(device)
            # In production, you might load skill-specific weights or
            # a shared policy trained on all skills

    # Create high-level controller
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Using scripted high-level controller.")
        pi_h = ScriptedHighLevelController()
        use_learned_hl = False
    else:
        pi_h = HighLevelController(
            obs_dim=13,
            num_skills=SkillID.NUM_SKILLS,
            use_vision=False
        ).to(device)
        use_learned_hl = True
        print(f"High-level controller parameters: {sum(p.numel() for p in pi_h.parameters()):,}")

    # Training loop
    if use_learned_hl:
        from src.hrl.hrl_trainer import HighLevelTrainer

        trainer = HighLevelTrainer(
            env=env,
            pi_h=pi_h,
            pi_l=pi_l,
            lr=lr,
            gamma=gamma,
            device=device
        )

        start_time = time.time()
        metrics = trainer.train(n_episodes=n_episodes, log_interval=50)
        training_time = time.time() - start_time

        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Final Success Rate: {metrics['success_rate']:.2%}")
        print(f"Mean Episode Length: {metrics['mean_episode_length']:.1f} skills")

        # Save controller
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pi_h.save(save_path)
        print(f"Saved controller to {save_path}")

        # Save metrics
        metrics['training_time'] = training_time
        metrics['n_episodes'] = n_episodes
        metrics['use_scripted_ll'] = use_scripted_ll

        metrics_path = save_path.replace('.pt', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    else:
        # Use scripted controller for baseline
        print("Using scripted high-level controller...")
        metrics = evaluate_scripted_hrl(env, pi_l, n_episodes)

    env.close()
    return metrics


def evaluate_scripted_hrl(env, pi_l, n_episodes=100):
    """
    Evaluate scripted HRL baseline.

    Args:
        env: Environment
        pi_l: Low-level policy
        n_episodes: Number of episodes

    Returns:
        metrics: Evaluation metrics
    """
    print("=" * 70)
    print("EVALUATING SCRIPTED HRL BASELINE")
    print("=" * 70)

    pi_h = ScriptedHighLevelController()
    termination_detector = SkillTerminationDetector()

    successes = 0
    vase_collisions = 0
    episode_lengths = []
    skill_sequences = []
    clearances = []
    energy_usages = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        pi_h.reset()

        episode_skills = []
        total_energy = 0

        # Execute skill sequence
        while True:
            skill_id, skill_params, seq_done = pi_h.select_skill(obs, done_previous=True)

            if skill_id == -1 or seq_done:
                break

            episode_skills.append(SkillID.name(skill_id))

            # Execute skill
            step_in_skill = 0
            while True:
                action = pi_l.act(obs, skill_id, skill_params)
                obs, _, env_done, _, info = env.step(action)

                total_energy = info.get('energy_Wh', total_energy)

                skill_done, skill_success, reason = termination_detector.is_done(
                    skill_id, obs, info, step_in_skill, skill_params.timeout_steps
                )

                step_in_skill += 1

                if skill_done or env_done:
                    break

            # Check task completion
            if info.get('success', False):
                successes += 1
                break

            if not info.get('vase_intact', True):
                vase_collisions += 1
                break

            if env_done:
                break

        skill_sequences.append(episode_skills)
        episode_lengths.append(len(episode_skills))
        clearances.append(info.get('min_clearance', 0))
        energy_usages.append(total_energy)

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: "
                  f"success={info.get('success', False)}, "
                  f"skills={len(episode_skills)}")

    success_rate = successes / n_episodes
    collision_rate = vase_collisions / n_episodes
    mean_length = np.mean(episode_lengths)
    mean_clearance = np.mean(clearances)
    mean_energy = np.mean(energy_usages)

    print(f"\n" + "=" * 70)
    print("SCRIPTED HRL BASELINE RESULTS")
    print("=" * 70)
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Vase Collision Rate: {collision_rate:.2%}")
    print(f"Mean Skills per Episode: {mean_length:.1f}")
    print(f"Mean Clearance: {mean_clearance:.4f}")
    print(f"Mean Energy: {mean_energy:.6f} Wh")

    metrics = {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'mean_episode_length': mean_length,
        'mean_clearance': mean_clearance,
        'mean_energy': mean_energy,
    }

    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = 'results/scripted_hrl_baseline.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return metrics


def evaluate_hierarchical_agent(
    n_episodes=50,
    use_learned_hl=False,
    use_learned_ll=False,
    device='cpu'
):
    """
    Evaluate complete hierarchical agent.

    Args:
        n_episodes: Number of episodes
        use_learned_hl: Use learned high-level controller
        use_learned_ll: Use learned low-level policies
        device: Device

    Returns:
        metrics: Evaluation metrics
    """
    print("=" * 70)
    print("EVALUATING HIERARCHICAL AGENT")
    print("=" * 70)

    config = DrawerVaseConfig()
    env = DrawerVasePhysicsEnv(config=config, obs_mode='state')

    # Load policies
    if use_learned_hl and TORCH_AVAILABLE:
        pi_h = HighLevelController.load('checkpoints/hrl/high_level_controller.pt', device)
    else:
        pi_h = ScriptedHighLevelController()

    if use_learned_ll and TORCH_AVAILABLE:
        pi_l = LowLevelSkillPolicy.load('checkpoints/hrl/skills/shared_policy.pt', device)
    else:
        pi_l = ScriptedSkillPolicy()

    termination_detector = SkillTerminationDetector()

    # Create hierarchical agent
    agent = HierarchicalAgent(
        high_level_controller=pi_h,
        low_level_policy=pi_l,
        termination_detector=termination_detector,
        use_scripted_hl=not use_learned_hl
    )

    # Evaluate
    successes = 0
    vase_collisions = 0
    total_steps_list = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()

        done = False
        total_steps = 0

        while not done and total_steps < 500:
            action, skill_info = agent.act(obs, info)

            if skill_info['task_done']:
                break

            obs, _, env_done, _, info = env.step(action)
            total_steps += 1

            if info.get('success', False):
                successes += 1
                done = True

            if not info.get('vase_intact', True):
                vase_collisions += 1
                done = True

            if env_done:
                done = True

        total_steps_list.append(total_steps)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: "
                  f"success={info.get('success', False)}, "
                  f"steps={total_steps}")

    env.close()

    success_rate = successes / n_episodes
    collision_rate = vase_collisions / n_episodes
    mean_steps = np.mean(total_steps_list)

    print(f"\nSuccess Rate: {success_rate:.2%}")
    print(f"Collision Rate: {collision_rate:.2%}")
    print(f"Mean Steps: {mean_steps:.1f}")

    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'mean_steps': mean_steps,
    }


def main():
    parser = argparse.ArgumentParser(description='Train High-Level Controller')
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--use-scripted-ll',
        action='store_true',
        default=True,
        help='Use scripted low-level policies'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Training device'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='checkpoints/hrl/high_level_controller.pt',
        help='Path to save controller'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate, no training'
    )

    args = parser.parse_args()

    if args.eval_only:
        evaluate_hierarchical_agent(n_episodes=50)
    else:
        train_high_level_controller(
            n_episodes=args.episodes,
            use_scripted_ll=args.use_scripted_ll,
            lr=args.lr,
            gamma=args.gamma,
            device=args.device,
            save_path=args.save_path
        )


if __name__ == '__main__':
    main()
