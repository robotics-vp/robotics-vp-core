#!/usr/bin/env python3
"""
Sample Synthetic z_V Rollouts from Latent Dynamics Model (Phase B.3)

Uses trained latent dynamics model to generate synthetic z_V trajectories:
- Samples random initial z_0 from dataset distribution
- Rolls out dynamics model with sampled/random actions
- Computes ΔMPL/novelty valuation on synthetic episodes

Usage:
    python scripts/sample_zv_rollouts.py --model checkpoints/latent_diffusion_zv.pt --samples 50
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path


def load_latent_dynamics_model(model_path, device='cpu'):
    """Load trained latent dynamics model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_type = checkpoint['model_type']
    latent_dim = checkpoint['latent_dim']
    action_dim = checkpoint['action_dim']
    hidden_dim = checkpoint['hidden_dim']

    if model_type == 'mlp':
        from scripts.train_latent_diffusion import LatentDynamicsModel
        model = LatentDynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    elif model_type == 'transformer':
        from scripts.train_latent_diffusion import TemporalTransformer
        model = TemporalTransformer(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded {model_type} model from {model_path}")
    print(f"  Latent dim: {latent_dim}, Action dim: {action_dim}")
    # Handle different checkpoint formats
    if 'final_mse' in checkpoint:
        print(f"  Final MSE: {checkpoint['final_mse']:.6f}")
    elif 'final_recon_loss' in checkpoint:
        print(f"  Final Recon Loss: {checkpoint['final_recon_loss']:.6f}")
    if 'final_trust_mean' in checkpoint:
        print(f"  Final Trust: {checkpoint['final_trust_mean']:.6f}")

    return model, checkpoint


def load_initial_distribution(dataset_path):
    """Load initial z_V distribution from real rollouts."""
    data = np.load(dataset_path, allow_pickle=True)

    n_episodes = int(data['n_episodes'])
    latent_dim = int(data['latent_dim'])

    # Collect all initial z_0 states
    initial_z = []
    all_actions = []

    for ep in range(n_episodes):
        z_seq = data[f'ep_{ep}_z_sequence']
        actions = data[f'ep_{ep}_actions']

        initial_z.append(z_seq[0])  # First z_V in episode
        all_actions.append(actions)

    initial_z = np.array(initial_z)  # (n_episodes, latent_dim)
    all_actions = np.concatenate(all_actions, axis=0)  # (total_steps, action_dim)

    # Compute statistics for sampling
    z_mean = initial_z.mean(axis=0)
    z_std = initial_z.std(axis=0)

    a_mean = all_actions.mean(axis=0)
    a_std = all_actions.std(axis=0)

    print(f"Initial z_V distribution: mean={z_mean.mean():.4f}, std={z_std.mean():.4f}")
    print(f"Action distribution: mean={a_mean}, std={a_std}")

    return {
        'initial_z': initial_z,
        'z_mean': z_mean,
        'z_std': z_std,
        'a_mean': a_mean,
        'a_std': a_std,
        'latent_dim': latent_dim,
        'action_dim': all_actions.shape[1],
    }


def sample_synthetic_rollouts(
    model,
    initial_dist,
    n_samples=50,
    max_steps=60,
    action_strategy='random',
    device='cpu',
):
    """
    Generate synthetic z_V rollouts using latent dynamics model.

    Args:
        model: Trained latent dynamics model
        initial_dist: Distribution info from real data
        n_samples: Number of synthetic episodes
        max_steps: Max steps per episode
        action_strategy: 'random', 'dataset', 'noise'
        device: Torch device

    Returns:
        List of synthetic rollouts
    """
    model.eval()

    synthetic_rollouts = []

    for ep in range(n_samples):
        # Sample initial z_0
        if action_strategy == 'dataset':
            # Sample from actual initial states
            idx = np.random.randint(0, len(initial_dist['initial_z']))
            z_current = initial_dist['initial_z'][idx]
        else:
            # Sample from Gaussian fitted to initial states
            z_current = np.random.randn(initial_dist['latent_dim']) * initial_dist['z_std'] + initial_dist['z_mean']

        z_current = torch.FloatTensor(z_current).unsqueeze(0).to(device)  # (1, latent_dim)

        # Roll out dynamics
        z_sequence = [z_current.cpu().numpy().squeeze()]
        actions = []

        for step in range(max_steps):
            # Sample action
            if action_strategy == 'random':
                # Uniform random action in [0, 1]
                action = np.random.rand(initial_dist['action_dim'])
            elif action_strategy == 'dataset':
                # Sample from dataset action distribution
                action = np.random.randn(initial_dist['action_dim']) * initial_dist['a_std'] + initial_dist['a_mean']
                action = np.clip(action, 0, 1)  # Clip to valid range
            else:  # 'noise'
                # Small perturbations
                action = 0.5 + 0.2 * np.random.randn(initial_dist['action_dim'])
                action = np.clip(action, 0, 1)

            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)

            # Predict next state
            with torch.no_grad():
                if hasattr(model, 'sample_next'):
                    z_next = model.sample_next(z_current, action_tensor)
                else:
                    z_next, _ = model(z_current, action_tensor)

            # Store
            z_sequence.append(z_next.cpu().numpy().squeeze())
            actions.append(action)

            # Update current state
            z_current = z_next

        # Store rollout
        rollout = {
            'z_sequence': np.array(z_sequence),  # (T+1, latent_dim)
            'actions': np.array(actions),        # (T, action_dim)
            'episode': ep,
            'length': max_steps,
        }

        synthetic_rollouts.append(rollout)

        if (ep + 1) % 10 == 0:
            print(f"  Generated {ep + 1}/{n_samples} synthetic episodes")

    return synthetic_rollouts


def compute_novelty_scores(synthetic_rollouts, real_rollouts_path):
    """
    Compute novelty scores for synthetic rollouts vs real data.

    Uses embedding distance in z_V space.
    """
    # Load real z_V sequences
    real_data = np.load(real_rollouts_path, allow_pickle=True)
    n_real = int(real_data['n_episodes'])

    # Collect all real z_V vectors
    real_z = []
    for ep in range(n_real):
        z_seq = real_data[f'ep_{ep}_z_sequence']
        real_z.extend(z_seq)
    real_z = np.array(real_z)

    # Compute mean z_V for each real episode
    real_episode_means = []
    for ep in range(n_real):
        z_seq = real_data[f'ep_{ep}_z_sequence']
        real_episode_means.append(z_seq.mean(axis=0))
    real_episode_means = np.array(real_episode_means)

    # Compute novelty for each synthetic rollout
    novelty_scores = []

    for rollout in synthetic_rollouts:
        z_seq = rollout['z_sequence']

        # Episode-level: mean z_V distance to nearest real episode
        z_mean = z_seq.mean(axis=0)
        distances = np.linalg.norm(real_episode_means - z_mean, axis=1)
        min_dist = distances.min()

        novelty_scores.append(min_dist)

    return novelty_scores


def estimate_synthetic_dmpl(synthetic_rollouts, real_rollouts_path):
    """
    Estimate ΔMPL for synthetic rollouts based on z_V dynamics.

    This is a proxy estimate based on:
    - Trajectory smoothness (low variance = higher MPL potential)
    - Distance from successful episodes in real data
    """
    # Load real MPL data
    real_data = np.load(real_rollouts_path, allow_pickle=True)
    n_real = int(real_data['n_episodes'])

    real_mpls = []
    real_z_means = []

    for ep in range(n_real):
        mpl = float(real_data[f'ep_{ep}_metric_mpl'])
        z_seq = real_data[f'ep_{ep}_z_sequence']
        z_mean = z_seq.mean(axis=0)

        real_mpls.append(mpl)
        real_z_means.append(z_mean)

    real_mpls = np.array(real_mpls)
    real_z_means = np.array(real_z_means)

    # Estimate ΔMPL for synthetic rollouts
    dmpl_estimates = []

    for rollout in synthetic_rollouts:
        z_seq = rollout['z_sequence']
        z_mean = z_seq.mean(axis=0)

        # Find nearest real episode in z_V space
        distances = np.linalg.norm(real_z_means - z_mean, axis=1)
        nearest_idx = distances.argmin()
        nearest_mpl = real_mpls[nearest_idx]

        # Trajectory smoothness (lower variance = more stable)
        z_variance = z_seq.var(axis=0).mean()

        # Estimate MPL based on nearest real + smoothness bonus
        smoothness_factor = 1.0 / (1.0 + z_variance)
        estimated_mpl = nearest_mpl * smoothness_factor

        # ΔMPL is difference from mean real MPL
        dmpl = estimated_mpl - real_mpls.mean()

        dmpl_estimates.append(dmpl)

    return dmpl_estimates


def save_synthetic_rollouts(synthetic_rollouts, novelty_scores, dmpl_estimates, output_path):
    """Save synthetic rollouts with valuation metrics."""
    save_data = {
        'n_episodes': len(synthetic_rollouts),
        'latent_dim': synthetic_rollouts[0]['z_sequence'].shape[1],
    }

    for i, rollout in enumerate(synthetic_rollouts):
        save_data[f'ep_{i}_z_sequence'] = rollout['z_sequence']
        save_data[f'ep_{i}_actions'] = rollout['actions']
        save_data[f'ep_{i}_episode'] = i
        save_data[f'ep_{i}_length'] = rollout['length']
        save_data[f'ep_{i}_novelty'] = novelty_scores[i]
        save_data[f'ep_{i}_dmpl_estimate'] = dmpl_estimates[i]

    np.savez_compressed(output_path, **save_data)
    print(f"✅ Saved {len(synthetic_rollouts)} synthetic rollouts to {output_path}")

    # Summary CSV
    summary_path = output_path.replace('.npz', '_summary.csv')
    import csv
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'length', 'novelty', 'dmpl_estimate'])
        writer.writeheader()
        for i, rollout in enumerate(synthetic_rollouts):
            writer.writerow({
                'episode': i,
                'length': rollout['length'],
                'novelty': novelty_scores[i],
                'dmpl_estimate': dmpl_estimates[i],
            })
    print(f"   Summary saved to {summary_path}")


def sample_zv_rollouts_main(
    model_path='checkpoints/latent_diffusion_zv.pt',
    dataset_path='data/physics_zv_rollouts.npz',
    n_samples=50,
    max_steps=60,
    action_strategy='random',
    output_dir='data',
    device=None,
):
    """
    Main function to sample synthetic z_V rollouts.

    Args:
        model_path: Path to trained latent dynamics model
        dataset_path: Path to real z_V rollouts
        n_samples: Number of synthetic episodes
        max_steps: Max steps per episode
        action_strategy: How to sample actions
        output_dir: Directory to save results
        device: Torch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Sampling on device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, checkpoint = load_latent_dynamics_model(model_path, device)

    # Load initial distribution
    initial_dist = load_initial_distribution(dataset_path)

    # Generate synthetic rollouts
    print(f"\nGenerating {n_samples} synthetic rollouts...")
    print(f"  Max steps: {max_steps}")
    print(f"  Action strategy: {action_strategy}")

    synthetic_rollouts = sample_synthetic_rollouts(
        model=model,
        initial_dist=initial_dist,
        n_samples=n_samples,
        max_steps=max_steps,
        action_strategy=action_strategy,
        device=device,
    )

    # Compute valuation metrics
    print("\nComputing valuation metrics...")
    novelty_scores = compute_novelty_scores(synthetic_rollouts, dataset_path)
    dmpl_estimates = estimate_synthetic_dmpl(synthetic_rollouts, dataset_path)

    # Summary
    print(f"\nSynthetic Rollouts Summary:")
    print(f"  Novelty: {np.mean(novelty_scores):.4f} ± {np.std(novelty_scores):.4f}")
    print(f"  ΔMPL Estimate: {np.mean(dmpl_estimates):.2f} ± {np.std(dmpl_estimates):.2f}")

    # Save results
    output_path = os.path.join(output_dir, 'synthetic_zv_rollouts.npz')
    save_synthetic_rollouts(synthetic_rollouts, novelty_scores, dmpl_estimates, output_path)

    return synthetic_rollouts, novelty_scores, dmpl_estimates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample synthetic z_V rollouts')
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/latent_diffusion_zv.pt',
        help='Path to trained latent dynamics model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/physics_zv_rollouts.npz',
        help='Path to real z_V rollouts'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Number of synthetic episodes to generate'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=60,
        help='Max steps per episode'
    )
    parser.add_argument(
        '--action-strategy',
        type=str,
        default='random',
        choices=['random', 'dataset', 'noise'],
        help='How to sample actions'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Directory to save results'
    )

    args = parser.parse_args()

    print("="*60)
    print("Phase B.3: Sampling Synthetic z_V Rollouts")
    print("="*60)

    synthetic_rollouts, novelty_scores, dmpl_estimates = sample_zv_rollouts_main(
        model_path=args.model,
        dataset_path=args.dataset,
        n_samples=args.samples,
        max_steps=args.max_steps,
        action_strategy=args.action_strategy,
        output_dir=args.output_dir,
    )

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Cluster z_V trajectories into capability packs:")
    print("   Use k-means or HDBSCAN on episode-level z_V means")
    print()
    print("2. Compare synthetic vs real data in policy training:")
    print("   - Train policy on real + synthetic data")
    print("   - Measure performance improvement")
    print()
    print("3. Data brick taxonomy:")
    print("   - Group episodes by (novelty, ΔMPL) clusters")
    print("   - Assign economic value to each cluster")
    print("="*60)
