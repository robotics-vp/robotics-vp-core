#!/usr/bin/env python3
"""
Evaluate Offline-Trained Policy in Live Environment

Runs the trained offline policy in the actual physics environment to measure:
- MPL (dishes/hr)
- Error rate
- Wage parity

This is the proper validation: does training with high-value bricks improve real performance?

Usage:
    python scripts/eval_offline_policy.py --model checkpoints/offline_baseline_actor.pt --episodes 20
    python scripts/eval_offline_policy.py --model checkpoints/offline_augmented_actor.pt --episodes 20
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path


def load_actor(model_path, latent_dim=128, action_dim=2, hidden_dim=256, device='cpu'):
    """Load trained actor from checkpoint."""
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self, latent_dim=128, action_dim=2, hidden_dim=256):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            mean = self.mean(x)
            log_std = self.log_std(x).clamp(-20, 2)
            return mean, log_std

        def sample(self, state):
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.sigmoid(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob, mean

    actor = Actor(latent_dim, action_dim, hidden_dim).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    print(f"Loaded actor from {model_path}")
    if 'config' in checkpoint:
        print(f"  Training config: {checkpoint['config']}")

    return actor, checkpoint


def load_aligned_encoder(config_path='configs/dishwashing_physics_aligned.yaml', device='cpu'):
    """Load pretrained aligned encoder (z_V)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.encoders.student_video_encoder import AlignedVideoEncoder

    encoder_cfg = config['encoder']
    aligned_cfg = encoder_cfg.get('aligned', {})

    encoder = AlignedVideoEncoder(
        latent_dim=encoder_cfg['latent_dim'],
        arch=aligned_cfg.get('arch', 'simple2dcnn'),
        input_channels=aligned_cfg.get('input_channels', 3),
    ).to(device)

    checkpoint_path = encoder_cfg.get('checkpoint', 'checkpoints/student_video_aligned.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'student_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['student_state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"Loaded encoder from {checkpoint_path}")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder, config


def create_physics_env(config):
    """Create physics environment from config."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.envs.physics.dishwashing_physics_env import DishwashingPhysicsEnv

    env_config = config['env']

    env = DishwashingPhysicsEnv(
        frames=env_config.get('frames', 4),
        image_size=tuple(env_config.get('image_size', [64, 64])),
        max_steps=env_config.get('max_steps', 60),
        headless=env_config.get('headless', True),
        randomize_dishes=env_config.get('randomize_dishes', True),
        camera_jitter=env_config.get('camera_jitter', 0.02),
        lighting_variation=env_config.get('lighting_variation', 0.1),
        slip_probability=env_config.get('slip_probability', 0.015),
        gripper_failure_rate=env_config.get('gripper_failure_rate', 0.008),
    )

    return env


def compute_mpl(completed, episode_length, seconds_per_step=6.0):
    """Compute MPL (units/hour)."""
    time_hours = (episode_length * seconds_per_step) / 3600
    return completed / time_hours if time_hours > 0 else 0


def evaluate_policy(
    model_path,
    config_path='configs/dishwashing_physics_aligned.yaml',
    n_episodes=20,
    device=None,
):
    """
    Evaluate policy in live environment.

    Returns:
        Dict with MPL, error rate, wage parity statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("ONLINE POLICY EVALUATION")
    print("="*60)

    # Load components
    actor, ckpt = load_actor(model_path, device=device)
    encoder, config = load_aligned_encoder(config_path, device)
    env = create_physics_env(config)

    # Economic parameters
    econ = config['economics']
    price_per_unit = econ['price_per_unit']
    human_wage = econ['human_wage']
    damage_cost = econ['damage_cost']

    print(f"\nEvaluating for {n_episodes} episodes...")

    # Track metrics
    mpls = []
    error_rates = []
    wage_parities = []
    completed_all = []
    errors_all = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        step_count = 0

        while not done:
            # Encode observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                z_v = encoder(obs_tensor)

            # Get action from policy
            with torch.no_grad():
                action, _, _ = actor.sample(z_v)
            action = action.cpu().numpy().squeeze()

            # Step environment
            obs, info, done = env.step(action)
            step_count += 1

        # Compute episode metrics
        completed = info.get('completed', 0)
        attempts = info.get('attempts', completed)
        errors = info.get('errors', 0)

        mpl = compute_mpl(completed, step_count)
        error_rate = errors / attempts if attempts > 0 else 0
        robot_wage = price_per_unit * mpl - damage_cost * errors
        wage_parity = robot_wage / human_wage

        mpls.append(mpl)
        error_rates.append(error_rate)
        wage_parities.append(wage_parity)
        completed_all.append(completed)
        errors_all.append(errors)

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: "
                  f"MPL={mpl:.1f}, Error={error_rate:.3f}, WagePar={wage_parity:.3f}")

    # Compute statistics
    results = {
        'model_path': model_path,
        'n_episodes': n_episodes,
        'mpl_mean': np.mean(mpls),
        'mpl_std': np.std(mpls),
        'error_rate_mean': np.mean(error_rates),
        'error_rate_std': np.std(error_rates),
        'wage_parity_mean': np.mean(wage_parities),
        'wage_parity_std': np.std(wage_parities),
        'total_completed': sum(completed_all),
        'total_errors': sum(errors_all),
    }

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print()
    print(f"MPL:         {results['mpl_mean']:.2f} ± {results['mpl_std']:.2f} dishes/hr")
    print(f"Error Rate:  {results['error_rate_mean']:.4f} ± {results['error_rate_std']:.4f}")
    print(f"Wage Parity: {results['wage_parity_mean']:.4f} ± {results['wage_parity_std']:.4f}")
    print()
    print(f"Total completed: {results['total_completed']}")
    print(f"Total errors:    {results['total_errors']}")
    print("="*60)

    return results


def compare_policies(baseline_path, augmented_path, n_episodes=20, device=None):
    """
    Compare baseline vs augmented policy.

    This is the key experiment: do high-value bricks improve policy?
    """
    print("\n" + "="*70)
    print("BASELINE vs AUGMENTED POLICY COMPARISON")
    print("="*70)

    # Evaluate both
    print("\n--- BASELINE (Real-Only Training) ---")
    baseline_results = evaluate_policy(baseline_path, n_episodes=n_episodes, device=device)

    print("\n--- AUGMENTED (Real + High-Value Bricks) ---")
    augmented_results = evaluate_policy(augmented_path, n_episodes=n_episodes, device=device)

    # Compute improvements
    mpl_improvement = augmented_results['mpl_mean'] - baseline_results['mpl_mean']
    error_improvement = baseline_results['error_rate_mean'] - augmented_results['error_rate_mean']
    wage_improvement = augmented_results['wage_parity_mean'] - baseline_results['wage_parity_mean']

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<20} {'Baseline':<15} {'Augmented':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'MPL (dishes/hr)':<20} "
          f"{baseline_results['mpl_mean']:<15.2f} "
          f"{augmented_results['mpl_mean']:<15.2f} "
          f"{mpl_improvement:+15.2f}")
    print(f"{'Error Rate':<20} "
          f"{baseline_results['error_rate_mean']:<15.4f} "
          f"{augmented_results['error_rate_mean']:<15.4f} "
          f"{error_improvement:+15.4f}")
    print(f"{'Wage Parity':<20} "
          f"{baseline_results['wage_parity_mean']:<15.4f} "
          f"{augmented_results['wage_parity_mean']:<15.4f} "
          f"{wage_improvement:+15.4f}")
    print("="*70)

    if mpl_improvement > 0 or error_improvement > 0:
        print("\n✅ AUGMENTATION SHOWS VALUE!")
        print(f"   MPL improved by {mpl_improvement:+.2f} dishes/hr")
        print(f"   Error rate reduced by {error_improvement:+.4f}")
    else:
        print("\n⚠️  No clear improvement. Debug world model / brick selection.")

    return baseline_results, augmented_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate offline-trained policy online')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained actor checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dishwashing_physics_aligned.yaml',
        help='Path to env/encoder config'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--compare',
        type=str,
        default=None,
        help='Path to second model for comparison (e.g., augmented vs baseline)'
    )

    args = parser.parse_args()

    if args.compare:
        compare_policies(args.model, args.compare, n_episodes=args.episodes)
    else:
        evaluate_policy(args.model, config_path=args.config, n_episodes=args.episodes)
