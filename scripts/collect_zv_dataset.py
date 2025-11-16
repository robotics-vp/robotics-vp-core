#!/usr/bin/env python3
"""
Collect z_V Dataset from Physics Environment (Phase B.1)

Runs trained SAC policy in physics environment and records:
- z_V sequences (T x 128 latent vectors)
- Actions, rewards, MPL, error flags
- Episode-level metrics for data valuation

Usage:
    python scripts/collect_zv_dataset.py --policy checkpoints/sac_aligned_actor.pt --episodes 100
    python scripts/collect_zv_dataset.py --episodes 50  # Random policy
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path


def load_sac_policy(policy_path, latent_dim=128, action_dim=2, hidden_dim=256, device='cpu'):
    """Load trained SAC actor policy."""
    from src.agents.sac_agent import Actor

    actor = Actor(latent_dim, action_dim, hidden_dim).to(device)

    if os.path.exists(policy_path):
        checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
        if 'actor_state_dict' in checkpoint:
            actor.load_state_dict(checkpoint['actor_state_dict'])
        else:
            actor.load_state_dict(checkpoint)
        print(f"Loaded policy from {policy_path}")
        actor.eval()
    else:
        print(f"Warning: Policy not found at {policy_path}, using random policy")
        actor = None

    return actor


def load_aligned_encoder(config_path, device='cpu'):
    """Load pretrained aligned encoder (z_V)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    from src.encoders.student_video_encoder import AlignedVideoEncoder

    encoder_cfg = config['encoder']
    aligned_cfg = encoder_cfg.get('aligned', {})

    encoder = AlignedVideoEncoder(
        latent_dim=encoder_cfg['latent_dim'],
        arch=aligned_cfg.get('arch', 'simple2dcnn'),
        input_channels=aligned_cfg.get('input_channels', 3),
        projection_dim=aligned_cfg.get('projection_dim', None),
        alignment_type=aligned_cfg.get('alignment_type', 'mse'),
        temperature=aligned_cfg.get('temperature', 0.1),
    ).to(device)

    # Load pretrained checkpoint
    checkpoint_path = encoder_cfg.get('checkpoint', 'checkpoints/student_video_aligned.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'student_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['student_state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"Loaded aligned encoder from {checkpoint_path}")
    else:
        print(f"Warning: Encoder checkpoint not found at {checkpoint_path}")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder, config


def create_physics_env(config):
    """Create physics environment from config."""
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
        max_speed_multiplier=env_config.get('max_speed_multiplier', 2.0),
        max_acceleration=env_config.get('max_acceleration', 1.0)
    )

    return env


def compute_economic_metrics(info, config, episode_length=60):
    """Compute economic metrics from episode info."""
    econ = config['economics']

    price_per_unit = econ['price_per_unit']
    human_mp = econ['human_mp']
    human_wage = econ['human_wage']
    damage_cost = econ['damage_cost']

    # MPL calculation
    completed = info.get('completed', 0)
    # Compute time from episode length (each step = ~6 seconds real time)
    # 60 steps = 6 minutes = 0.1 hours
    time_seconds = info.get('time_seconds', episode_length * 6.0)  # 6 sec/step
    time_hours = time_seconds / 3600
    mpl = completed / time_hours if time_hours > 0 else 0

    # Error rate
    attempts = info.get('attempts', completed)
    errors = info.get('errors', 0)
    error_rate = errors / attempts if attempts > 0 else 0

    # Robot implied wage
    robot_wage = price_per_unit * mpl - damage_cost * errors
    wage_parity = robot_wage / human_wage if human_wage > 0 else 0

    return {
        'mpl': mpl,
        'error_rate': error_rate,
        'robot_wage': robot_wage,
        'wage_parity': wage_parity,
        'completed': completed,
        'attempts': attempts,
        'errors': errors,
        'time_hours': time_hours,
    }


def collect_zv_rollouts(
    config_path='configs/dishwashing_physics_aligned.yaml',
    policy_path=None,
    n_episodes=100,
    output_dir='data',
    device=None
):
    """
    Collect z_V rollouts from physics environment.

    Args:
        config_path: Path to aligned encoder config
        policy_path: Path to trained SAC policy (None for random)
        n_episodes: Number of episodes to collect
        output_dir: Directory to save rollouts
        device: Torch device

    Returns:
        Path to saved dataset
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Collecting z_V rollouts on device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load encoder and config
    encoder, config = load_aligned_encoder(config_path, device)
    latent_dim = config['encoder']['latent_dim']

    # Load policy (or use random)
    actor = None
    if policy_path:
        actor = load_sac_policy(
            policy_path,
            latent_dim=latent_dim,
            action_dim=2,
            hidden_dim=config['sac'].get('hidden_dim', 256),
            device=device
        )

    # Create environment
    env = create_physics_env(config)

    print(f"\nCollecting {n_episodes} episodes...")
    print(f"  Encoder: AlignedVideoEncoder (latent_dim={latent_dim})")
    print(f"  Policy: {'Trained SAC' if actor else 'Random'}")
    print()

    # Storage for rollouts
    all_rollouts = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False

        # Episode storage
        episode_data = {
            'z_sequence': [],      # List of z_V vectors
            'actions': [],         # Actions taken
            'rewards': [],         # Step rewards
            'dones': [],           # Done flags
            'infos': [],           # Step infos
        }

        step_idx = 0
        while not done:
            # Encode current observation to z_V
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # (1, T, C, H, W)
            with torch.no_grad():
                z_v = encoder(obs_tensor).cpu().numpy().squeeze()  # (latent_dim,)

            episode_data['z_sequence'].append(z_v)

            # Select action
            if actor is not None:
                z_tensor = torch.FloatTensor(z_v).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _ = actor.sample(z_tensor)
                action = action.cpu().numpy().squeeze()
            else:
                # Random action
                action = np.random.rand(2)  # [speed, care]

            # Step environment
            obs, info, done = env.step(action)

            # Compute step reward (simplified)
            step_reward = -0.01  # Small penalty per step
            if info.get('completed', 0) > 0:
                step_reward += 1.0  # Bonus for completion

            episode_data['actions'].append(action)
            episode_data['rewards'].append(step_reward)
            episode_data['dones'].append(done)
            episode_data['infos'].append(info)

            step_idx += 1

        # Final z_V after episode ends
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            z_v_final = encoder(obs_tensor).cpu().numpy().squeeze()
        episode_data['z_sequence'].append(z_v_final)

        # Convert to numpy arrays
        episode_data['z_sequence'] = np.array(episode_data['z_sequence'])  # (T+1, latent_dim)
        episode_data['actions'] = np.array(episode_data['actions'])         # (T, action_dim)
        episode_data['rewards'] = np.array(episode_data['rewards'])         # (T,)
        episode_data['dones'] = np.array(episode_data['dones'])             # (T,)

        # Compute episode-level economic metrics
        final_info = episode_data['infos'][-1] if episode_data['infos'] else {}
        econ_metrics = compute_economic_metrics(final_info, config, episode_length=step_idx)

        episode_data['episode_metrics'] = {
            'episode': ep,
            'length': step_idx,
            'total_reward': float(np.sum(episode_data['rewards'])),
            **econ_metrics
        }

        # Remove infos (not easily serializable)
        del episode_data['infos']

        all_rollouts.append(episode_data)

        # Progress
        if (ep + 1) % 10 == 0:
            print(f"  Progress: {ep + 1}/{n_episodes} episodes")
            print(f"    Last episode: MPL={econ_metrics['mpl']:.1f}, "
                  f"Error={econ_metrics['error_rate']:.3f}, "
                  f"WagePar={econ_metrics['wage_parity']:.3f}")

    # Save dataset
    output_path = os.path.join(output_dir, 'physics_zv_rollouts.npz')

    # Prepare for saving
    save_data = {
        'n_episodes': n_episodes,
        'latent_dim': latent_dim,
        'config_path': config_path,
        'policy_path': policy_path or 'random',
    }

    # Save episode data separately (variable length)
    for i, rollout in enumerate(all_rollouts):
        save_data[f'ep_{i}_z_sequence'] = rollout['z_sequence']
        save_data[f'ep_{i}_actions'] = rollout['actions']
        save_data[f'ep_{i}_rewards'] = rollout['rewards']
        save_data[f'ep_{i}_dones'] = rollout['dones']
        # Save metrics as string for easy access
        for k, v in rollout['episode_metrics'].items():
            save_data[f'ep_{i}_metric_{k}'] = v

    np.savez_compressed(output_path, **save_data)

    print(f"\n✅ Saved {n_episodes} rollouts to {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Summary statistics
    print("\nDataset Summary:")
    all_mpls = [r['episode_metrics']['mpl'] for r in all_rollouts]
    all_errors = [r['episode_metrics']['error_rate'] for r in all_rollouts]
    all_wages = [r['episode_metrics']['wage_parity'] for r in all_rollouts]

    print(f"  MPL: {np.mean(all_mpls):.2f} ± {np.std(all_mpls):.2f} dishes/hr")
    print(f"  Error Rate: {np.mean(all_errors):.3f} ± {np.std(all_errors):.3f}")
    print(f"  Wage Parity: {np.mean(all_wages):.3f} ± {np.std(all_wages):.3f}")

    # Save summary CSV
    summary_path = os.path.join(output_dir, 'zv_rollouts_summary.csv')
    import csv
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rollouts[0]['episode_metrics'].keys()))
        writer.writeheader()
        for rollout in all_rollouts:
            writer.writerow(rollout['episode_metrics'])

    print(f"   Summary CSV saved to {summary_path}")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect z_V rollouts from physics environment')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dishwashing_physics_aligned.yaml',
        help='Path to aligned encoder config'
    )
    parser.add_argument(
        '--policy',
        type=str,
        default=None,
        help='Path to trained SAC policy (None for random)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes to collect'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Directory to save rollouts'
    )

    args = parser.parse_args()

    print("="*60)
    print("Phase B.1: Collecting z_V Dataset")
    print("="*60)

    output_path = collect_zv_rollouts(
        config_path=args.config,
        policy_path=args.policy,
        n_episodes=args.episodes,
        output_dir=args.output_dir,
    )

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print(f"1. Train latent dynamics model:")
    print(f"   python scripts/train_latent_diffusion.py \\")
    print(f"     --dataset {output_path} \\")
    print(f"     --epochs 100")
    print()
    print("2. Sample synthetic z_V trajectories:")
    print(f"   python scripts/sample_zv_rollouts.py \\")
    print(f"     --model checkpoints/latent_diffusion_zv.pt \\")
    print(f"     --samples 50")
    print("="*60)
