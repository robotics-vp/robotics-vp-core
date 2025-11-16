#!/usr/bin/env python3
"""
Validate ΔMPL and novelty estimation on calibrated physics environment.

This script:
- Collects N episodes in the environment
- Extracts latent embeddings from encoder
- Computes novelty scores (k-NN distance in latent space)
- Computes ΔMPL estimates (deviation from running mean)
- Generates diagnostic plots and summary

Usage:
    python scripts/validate_dmpl_novelty.py --config configs/dishwashing_physics_fast.yaml
    python scripts/validate_dmpl_novelty.py --config configs/dishwashing_feasible.yaml --episodes 100
"""

import numpy as np
import torch
import yaml
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def load_env_and_encoder(config_path, checkpoint_path=None):
    """Load environment and encoder from config"""
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load environment
    env_config = config['env']
    env_type = env_config.get('type', 'dishwashing')

    print(f"Creating environment: {env_type}")

    if env_type == 'dishwashing_physics':
        from src.envs.physics.dishwashing_physics_env import DishwashingPhysicsEnv
        env = DishwashingPhysicsEnv(
            frames=env_config.get('frames', 8),
            image_size=tuple(env_config.get('image_size', [64, 64])),
            max_steps=env_config.get('max_steps', 60),
            headless=env_config.get('headless', True),
            camera_config=env_config.get('physics', {}).get('camera', None),
            # Phase A parameters
            randomize_dishes=env_config.get('randomize_dishes', True),
            camera_jitter=env_config.get('camera_jitter', 0.02),
            lighting_variation=env_config.get('lighting_variation', 0.1),
            slip_probability=env_config.get('slip_probability', 0.015),
            gripper_failure_rate=env_config.get('gripper_failure_rate', 0.008),
            max_speed_multiplier=env_config.get('max_speed_multiplier', 2.0),
            max_acceleration=env_config.get('max_acceleration', 1.0)
        )
    elif env_type == 'dishwashing':
        from src.envs.dishwashing_env import DishwashingEnv
        env = DishwashingEnv()
    else:
        raise ValueError(f"Unknown env type: {env_type}")

    # Load encoder
    encoder_config = config['encoder']
    encoder_type = encoder_config['type']
    latent_dim = encoder_config['latent_dim']

    print(f"Creating encoder: {encoder_type} (latent_dim={latent_dim})")

    if encoder_type == 'video':
        from src.encoders.video_encoder import VideoEncoder
        video_config = encoder_config.get('video', {})
        input_channels = video_config.get('input_channels', 3)
        encoder = VideoEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim
        )
    elif encoder_type == 'mlp':
        from src.encoders.mlp_encoder import MLPEncoder
        mlp_config = encoder_config.get('mlp', {})
        state_dim = env.observation_space.shape[0]
        encoder = MLPEncoder(
            input_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=mlp_config.get('hidden_dim', 256)
        )
    elif encoder_type == 'aligned':
        # Phase A.5: Aligned video encoder (distillation-based)
        from src.encoders.student_video_encoder import AlignedVideoEncoder
        aligned_config = encoder_config.get('aligned', {})
        encoder = AlignedVideoEncoder(
            latent_dim=latent_dim,
            arch=aligned_config.get('arch', 'simple2dcnn'),
            input_channels=aligned_config.get('input_channels', 3),
            projection_dim=aligned_config.get('projection_dim', None),
            alignment_type=aligned_config.get('alignment_type', 'mse'),
            temperature=aligned_config.get('temperature', 0.1),
        )
    elif encoder_type == 'teacher':
        # Teacher adapter (for testing pretrained backbone directly)
        from src.encoders.teacher_adapter import TeacherAdapter
        teacher_config = encoder_config.get('teacher', {})
        encoder = TeacherAdapter(
            teacher_type=teacher_config.get('teacher_type', 'r3d'),
            latent_dim=latent_dim,
            freeze=teacher_config.get('freeze', True),
            input_channels=teacher_config.get('input_channels', 3),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose 'video', 'mlp', 'aligned', or 'teacher'")

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading encoder checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Handle different checkpoint formats
        if 'student_state_dict' in checkpoint:
            # Aligned encoder checkpoint
            encoder.load_state_dict(checkpoint['student_state_dict'])
            print("✅ Loaded trained aligned encoder")
        elif 'encoder_state_dict' in checkpoint:
            # Standard encoder checkpoint
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("✅ Loaded trained encoder")
        else:
            # Try direct state dict
            encoder.load_state_dict(checkpoint)
            print("✅ Loaded encoder weights")
    else:
        print("⚠️  Using untrained encoder (for degeneracy check only)")

    encoder.eval()

    return env, encoder, config


def collect_episodes(env, encoder, n_episodes=50):
    """Collect episodes and extract latents, metrics"""
    episodes = []

    print(f"\nCollecting {n_episodes} episodes...")

    with torch.no_grad():
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            step_count = 0

            # Run episode with random actions
            while not done:
                # Generate random action [speed, care] in [0, 1]^2
                action = np.random.rand(2)
                obs, info, done = env.step(action)
                step_count += 1

            # Extract metrics from info
            attempts = info.get('attempts', 0)
            errors = info.get('errors', 0)
            completed = info.get('completed', 0)

            # Compute MPL (marginal product per hour)
            # Assuming 60 steps at ~0.1s per step = 6s per episode
            time_hours = step_count * 0.1 / 3600.0  # Convert to hours
            mpl = completed / time_hours if time_hours > 0 else 0.0

            # Get final latent from observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            latent = encoder(obs_tensor).squeeze(0).cpu().numpy()

            episodes.append({
                'latent': latent,
                'mpl': mpl,
                'attempts': attempts,
                'errors': errors,
                'completed': completed,
                'error_rate': errors / attempts if attempts > 0 else 0
            })

            if (ep + 1) % 10 == 0:
                print(f"  Progress: {ep + 1}/{n_episodes} episodes")

    print(f"✅ Collected {n_episodes} episodes\n")

    return episodes


def compute_novelty(episodes, k=5):
    """Compute novelty scores using k-NN distance in latent space"""
    latents = np.array([ep['latent'] for ep in episodes])

    # Fit k-NN
    k_neighbors = min(k, len(latents) - 1)
    if k_neighbors < 1:
        print("⚠️  Not enough episodes for k-NN, using zeros")
        return np.zeros(len(episodes))

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto').fit(latents)
    distances, indices = nbrs.kneighbors(latents)

    # Novelty = mean distance to k nearest neighbors (excluding self at index 0)
    novelty_scores = distances[:, 1:].mean(axis=1)

    return novelty_scores


def compute_dmpl(episodes):
    """Compute ΔMPL estimates (deviation from running mean)"""
    mpls = np.array([ep['mpl'] for ep in episodes])

    # ΔMPL = deviation from running mean
    dmpl_estimates = []
    running_mean = mpls[0] if len(mpls) > 0 else 0

    for i, mpl in enumerate(mpls):
        if i > 0:
            dmpl = mpl - running_mean
            running_mean = (running_mean * i + mpl) / (i + 1)
        else:
            dmpl = 0
        dmpl_estimates.append(dmpl)

    return np.array(dmpl_estimates)


def validate_dmpl_novelty(config_path, n_episodes=50, output_dir='reports', checkpoint_path=None):
    """Run ΔMPL / novelty validation"""
    os.makedirs(output_dir, exist_ok=True)

    # Load environment and encoder
    env, encoder, config = load_env_and_encoder(config_path, checkpoint_path)

    # Collect episodes
    episodes = collect_episodes(env, encoder, n_episodes)

    # Compute novelty scores
    print("Computing novelty scores...")
    novelty_scores = compute_novelty(episodes, k=5)

    # Compute ΔMPL estimates
    print("Computing ΔMPL estimates...")
    dmpl_estimates = compute_dmpl(episodes)

    # Extract MPL values
    mpls = np.array([ep['mpl'] for ep in episodes])
    attempts = np.array([ep['attempts'] for ep in episodes])
    error_rates = np.array([ep['error_rate'] for ep in episodes])

    # Generate plots
    print("Generating diagnostic plots...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Plot 1: Novelty distribution
    axes[0, 0].hist(novelty_scores, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].set_xlabel('Novelty Score', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Novelty Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(novelty_scores.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].legend()

    # Plot 2: ΔMPL vs Novelty
    axes[0, 1].scatter(novelty_scores, dmpl_estimates, alpha=0.6, s=40, color='darkgreen')
    axes[0, 1].set_xlabel('Novelty Score', fontsize=11)
    axes[0, 1].set_ylabel('ΔMPL (dishes/hr)', fontsize=11)
    axes[0, 1].set_title('ΔMPL vs Novelty', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Compute correlation (handling constant arrays)
    if novelty_scores.std() > 1e-10 and dmpl_estimates.std() > 1e-10:
        corr_novelty_dmpl = np.corrcoef(novelty_scores, dmpl_estimates)[0, 1]
    else:
        corr_novelty_dmpl = 0.0

    axes[0, 1].text(0.05, 0.95, f'ρ = {corr_novelty_dmpl:.3f}',
                     transform=axes[0, 1].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 3: ΔMPL vs MPL
    axes[1, 0].scatter(mpls, dmpl_estimates, alpha=0.6, s=40, color='darkorange')
    axes[1, 0].set_xlabel('MPL (dishes/hr)', fontsize=11)
    axes[1, 0].set_ylabel('ΔMPL (dishes/hr)', fontsize=11)
    axes[1, 0].set_title('ΔMPL vs True MPL', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Compute correlation
    if mpls.std() > 1e-10 and dmpl_estimates.std() > 1e-10:
        corr_mpl_dmpl = np.corrcoef(mpls, dmpl_estimates)[0, 1]
    else:
        corr_mpl_dmpl = 0.0

    axes[1, 0].text(0.05, 0.95, f'ρ = {corr_mpl_dmpl:.3f}',
                     transform=axes[1, 0].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 4: MPL distribution
    axes[1, 1].hist(mpls, bins=20, alpha=0.7, edgecolor='black', color='coral')
    axes[1, 1].set_xlabel('MPL (dishes/hr)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('MPL Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(mpls.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dmpl_novelty_validation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved plot to {plot_path}")

    # Write summary
    summary_path = os.path.join(output_dir, 'dmpl_novelty_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ΔMPL / Novelty Validation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Configuration: {config_path}\n")
        f.write(f"Episodes collected: {n_episodes}\n")
        if checkpoint_path:
            f.write(f"Encoder checkpoint: {checkpoint_path}\n")
        else:
            f.write(f"Encoder: Untrained (for degeneracy check)\n")
        f.write("\n")

        f.write("Novelty Statistics:\n")
        f.write(f"  Mean:  {novelty_scores.mean():>10.4f}\n")
        f.write(f"  Std:   {novelty_scores.std():>10.4f}\n")
        f.write(f"  Range: [{novelty_scores.min():.4f}, {novelty_scores.max():.4f}]\n\n")

        f.write("ΔMPL Statistics:\n")
        f.write(f"  Mean:  {dmpl_estimates.mean():>10.2f} dishes/hr\n")
        f.write(f"  Std:   {dmpl_estimates.std():>10.2f} dishes/hr\n")
        f.write(f"  Range: [{dmpl_estimates.min():.2f}, {dmpl_estimates.max():.2f}]\n\n")

        f.write("MPL Statistics:\n")
        f.write(f"  Mean:  {mpls.mean():>10.2f} dishes/hr\n")
        f.write(f"  Std:   {mpls.std():>10.2f} dishes/hr\n")
        f.write(f"  Range: [{mpls.min():.2f}, {mpls.max():.2f}]\n\n")

        f.write("Episode Metrics:\n")
        f.write(f"  Avg Attempts:    {attempts.mean():>6.2f}\n")
        f.write(f"  Avg Error Rate:  {error_rates.mean():>6.3f}\n\n")

        f.write("Correlations:\n")
        f.write(f"  Novelty ↔ ΔMPL:  {corr_novelty_dmpl:>7.3f}\n")
        f.write(f"  MPL ↔ ΔMPL:      {corr_mpl_dmpl:>7.3f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Degeneracy Checks\n")
        f.write("=" * 70 + "\n\n")

        # Check 1: Novelty diversity
        novelty_unique = len(np.unique(novelty_scores.round(6)))
        f.write(f"1. Novelty Diversity:\n")
        f.write(f"   Unique values: {novelty_unique}/{n_episodes}\n")

        if novelty_scores.std() < 1e-6:
            f.write("   ⚠️  WARNING: Novelty scores are collapsed (std < 1e-6)\n")
            f.write("       Latent space may be degenerate or encoder not training\n")
        elif novelty_unique < n_episodes * 0.3:
            f.write("   ⚠️  WARNING: Low novelty diversity (<30% unique)\n")
            f.write("       Latent representations may be too similar\n")
        else:
            f.write("   ✅ Novelty scores show healthy spread\n")

        f.write("\n")

        # Check 2: Correlation strength
        f.write(f"2. Novelty-ΔMPL Correlation:\n")
        if abs(corr_novelty_dmpl) < 0.05:
            f.write("   ⚠️  WARNING: Very weak correlation (|ρ| < 0.05)\n")
            f.write("       Novelty may not be predictive of ΔMPL\n")
        elif abs(corr_novelty_dmpl) < 0.15:
            f.write("   ✓  Weak correlation (|ρ| < 0.15)\n")
            f.write("      Expected for early training / untrained encoder\n")
        else:
            f.write("   ✅ Moderate to strong correlation detected\n")

        f.write("\n")

        # Check 3: MPL variance
        f.write(f"3. MPL Variance:\n")
        if mpls.std() < 1.0:
            f.write("   ⚠️  WARNING: MPL variance very low (std < 1.0)\n")
            f.write("       Environment may not be exploring diverse states\n")
        else:
            f.write("   ✅ MPL shows healthy variance\n")

        f.write("\n")

        # Check 4: Latent collapse
        latent_dim = episodes[0]['latent'].shape[0]
        latents = np.array([ep['latent'] for ep in episodes])
        latent_std_mean = latents.std(axis=0).mean()

        f.write(f"4. Latent Space Collapse:\n")
        f.write(f"   Latent dimension: {latent_dim}\n")
        f.write(f"   Mean std across dims: {latent_std_mean:.4f}\n")

        if latent_std_mean < 1e-4:
            f.write("   ⚠️  WARNING: Latents collapsed (mean std < 1e-4)\n")
            f.write("       Encoder producing near-constant embeddings\n")
        else:
            f.write("   ✅ Latent space shows variation\n")

        f.write("\n" + "=" * 70 + "\n\n")

        f.write("Interpretation:\n")
        f.write("-" * 70 + "\n")
        f.write("- Novelty diversity checks if latent space is degenerate\n")
        f.write("- Weak Novelty-ΔMPL correlation is expected for untrained encoders\n")
        f.write("- Strong correlation emerges after encoder training (Phase A.5)\n")
        f.write("- These metrics validate that environment + encoder infrastructure\n")
        f.write("  are ready for aligned visual backbone training\n")

    print(f"✅ Saved summary to {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    with open(summary_path, 'r') as f:
        print(f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate ΔMPL and novelty estimation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dishwashing_physics_fast.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of episodes to collect'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports',
        help='Output directory'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to encoder checkpoint (optional)'
    )
    args = parser.parse_args()

    validate_dmpl_novelty(args.config, args.episodes, args.output, args.checkpoint)
