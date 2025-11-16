#!/usr/bin/env python3
"""
Filter Synthetic z_V Episodes by Distribution Match

Drop synthetic episodes whose latent statistics are too far from real distribution.
This is a quick bandaid to stop training on OOD garbage while we build trust_net.

Usage:
    python scripts/filter_synthetic_zv.py
"""

import numpy as np
from pathlib import Path


def compute_episode_stats(z_sequence):
    """Compute summary statistics for a z_V episode."""
    return {
        'mean': z_sequence.mean(),
        'std': z_sequence.std(),
        'min': z_sequence.min(),
        'max': z_sequence.max(),
        'per_dim_mean': z_sequence.mean(axis=0),
        'per_dim_std': z_sequence.std(axis=0),
    }


def filter_synthetic_by_stats(
    real_path='data/physics_zv_rollouts.npz',
    synthetic_path='data/synthetic_zv_rollouts.npz',
    output_path='data/synthetic_zv_filtered.npz',
    std_tolerance=1.5,  # Accept if std within [real/tol, real*tol]
    mean_tolerance=3.0,  # Accept if mean within ±tol*real_std
):
    """
    Filter synthetic episodes to only keep those with similar z_V distribution.

    Args:
        std_tolerance: Multiplicative tolerance for std
        mean_tolerance: How many real_stds away the mean can be
    """
    print("=" * 60)
    print("Z_V DISTRIBUTION-BASED SYNTHETIC FILTER")
    print("=" * 60)

    # Load real data and compute reference stats
    print(f"\nLoading real data from {real_path}...")
    real_data = np.load(real_path, allow_pickle=True)
    n_real = int(real_data['n_episodes'])

    real_zs = []
    for ep in range(n_real):
        z_seq = real_data[f'ep_{ep}_z_sequence']
        real_zs.append(z_seq)
    real_zs = np.concatenate(real_zs, axis=0)

    real_mean = real_zs.mean()
    real_std = real_zs.std()
    real_per_dim_std = real_zs.std(axis=0).mean()

    print(f"Real z_V reference stats:")
    print(f"  Global mean: {real_mean:.6f}")
    print(f"  Global std:  {real_std:.6f}")
    print(f"  Per-dim std: {real_per_dim_std:.6f}")

    # Define acceptance bounds
    std_lower = real_std / std_tolerance
    std_upper = real_std * std_tolerance
    mean_lower = real_mean - mean_tolerance * real_std
    mean_upper = real_mean + mean_tolerance * real_std

    print(f"\nAcceptance bounds (std_tol={std_tolerance}, mean_tol={mean_tolerance}):")
    print(f"  Std range: [{std_lower:.6f}, {std_upper:.6f}]")
    print(f"  Mean range: [{mean_lower:.6f}, {mean_upper:.6f}]")

    # Load synthetic data and filter
    print(f"\nLoading synthetic data from {synthetic_path}...")
    syn_data = np.load(synthetic_path, allow_pickle=True)
    n_syn = int(syn_data['n_episodes'])

    kept_episodes = []
    dropped_episodes = []
    kept_indices = []

    for ep in range(n_syn):
        z_seq = syn_data[f'ep_{ep}_z_sequence']
        ep_mean = z_seq.mean()
        ep_std = z_seq.std()

        # Check if within bounds
        mean_ok = mean_lower <= ep_mean <= mean_upper
        std_ok = std_lower <= ep_std <= std_upper

        if mean_ok and std_ok:
            kept_episodes.append(ep)
            kept_indices.append(ep)
        else:
            dropped_episodes.append({
                'ep': ep,
                'mean': ep_mean,
                'std': ep_std,
                'mean_ok': mean_ok,
                'std_ok': std_ok,
            })

    print(f"\nFiltering results:")
    print(f"  Total synthetic episodes: {n_syn}")
    print(f"  Kept: {len(kept_episodes)} ({100*len(kept_episodes)/n_syn:.1f}%)")
    print(f"  Dropped: {len(dropped_episodes)} ({100*len(dropped_episodes)/n_syn:.1f}%)")

    if dropped_episodes:
        print(f"\nDropped episodes breakdown:")
        mean_issues = sum(1 for d in dropped_episodes if not d['mean_ok'])
        std_issues = sum(1 for d in dropped_episodes if not d['std_ok'])
        both_issues = sum(1 for d in dropped_episodes if not d['mean_ok'] and not d['std_ok'])
        print(f"  Mean out of range: {mean_issues}")
        print(f"  Std out of range: {std_issues}")
        print(f"  Both out of range: {both_issues}")

        # Show worst offenders
        print(f"\nWorst std offenders (top 5):")
        by_std = sorted(dropped_episodes, key=lambda x: abs(x['std'] - real_std), reverse=True)
        for d in by_std[:5]:
            print(f"  Ep {d['ep']}: mean={d['mean']:.4f}, std={d['std']:.4f} (vs real std={real_std:.4f})")

    # Save filtered dataset
    if len(kept_episodes) == 0:
        print("\n⚠️  WARNING: No episodes passed filter! Loosening bounds...")
        # Fallback: keep top 50% by closeness to real std
        syn_stats = []
        for ep in range(n_syn):
            z_seq = syn_data[f'ep_{ep}_z_sequence']
            syn_stats.append((ep, abs(z_seq.std() - real_std)))
        syn_stats.sort(key=lambda x: x[1])
        kept_indices = [s[0] for s in syn_stats[:n_syn//2]]
        print(f"  Fallback: keeping {len(kept_indices)} episodes closest to real std")

    # Build filtered npz
    filtered_data = {
        'n_episodes': len(kept_indices),
        'original_indices': np.array(kept_indices),
    }

    kept_stds = []
    kept_means = []

    for new_idx, old_idx in enumerate(kept_indices):
        z_seq = syn_data[f'ep_{old_idx}_z_sequence']
        actions = syn_data[f'ep_{old_idx}_actions']

        filtered_data[f'ep_{new_idx}_z_sequence'] = z_seq
        filtered_data[f'ep_{new_idx}_actions'] = actions

        # Copy metrics if available
        if f'ep_{old_idx}_metrics' in syn_data:
            filtered_data[f'ep_{new_idx}_metrics'] = syn_data[f'ep_{old_idx}_metrics']

        kept_stds.append(z_seq.std())
        kept_means.append(z_seq.mean())

    # Save
    np.savez(output_path, **filtered_data)
    print(f"\nSaved filtered dataset to {output_path}")
    print(f"  Episodes: {len(kept_indices)}")
    print(f"  Filtered mean of means: {np.mean(kept_means):.6f} (real: {real_mean:.6f})")
    print(f"  Filtered mean of stds: {np.mean(kept_stds):.6f} (real: {real_std:.6f})")

    # Final distribution check
    filtered_zs = []
    for new_idx in range(len(kept_indices)):
        filtered_zs.append(filtered_data[f'ep_{new_idx}_z_sequence'])
    filtered_zs = np.concatenate(filtered_zs, axis=0)

    print(f"\nFinal filtered z_V stats:")
    print(f"  Global mean: {filtered_zs.mean():.6f} (real: {real_mean:.6f})")
    print(f"  Global std:  {filtered_zs.std():.6f} (real: {real_std:.6f})")

    improvement = abs(filtered_zs.std() - real_std) / abs(np.concatenate([syn_data[f'ep_{i}_z_sequence'] for i in range(n_syn)], axis=0).std() - real_std)
    print(f"  Std mismatch improved by: {(1-improvement)*100:.1f}%")

    return kept_indices


if __name__ == '__main__':
    filter_synthetic_by_stats()
