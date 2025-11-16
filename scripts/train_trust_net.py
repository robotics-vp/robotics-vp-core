#!/usr/bin/env python3
"""
Train Trust Network and Score Episodes

Trains a classifier to distinguish real vs synthetic z_V episodes,
then scores all episodes with trust weights for use in offline RL.

Usage:
    python scripts/train_trust_net.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from src.valuation.trust_net import train_trust_net, score_episodes_with_trust


def main():
    # Paths
    real_path = 'data/physics_zv_rollouts.npz'
    synthetic_path = 'data/synthetic_zv_rollouts.npz'
    checkpoint_path = 'checkpoints/trust_net.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Train trust_net
    print("\n" + "=" * 60)
    print("STEP 1: TRAIN TRUST_NET")
    print("=" * 60)

    model, metrics = train_trust_net(
        real_path=real_path,
        synthetic_path=synthetic_path,
        n_epochs=100,
        batch_size=32,
        lr=1e-3,
        device=device,
    )

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'X_mean': metrics['X_mean'],
        'X_std': metrics['X_std'],
    }, checkpoint_path)
    print(f"\nSaved trust_net to {checkpoint_path}")

    # Step 2: Score all episodes
    print("\n" + "=" * 60)
    print("STEP 2: SCORE ALL EPISODES")
    print("=" * 60)

    # Score real episodes
    print("\nScoring real episodes...")
    real_trust = score_episodes_with_trust(
        model, real_path, metrics['X_mean'], metrics['X_std'], device
    )
    print(f"  Real episodes: {len(real_trust)}")
    print(f"  Trust mean: {real_trust.mean():.4f}")
    print(f"  Trust std:  {real_trust.std():.4f}")
    print(f"  Trust min:  {real_trust.min():.4f}")
    print(f"  Trust max:  {real_trust.max():.4f}")

    # Score synthetic episodes
    print("\nScoring synthetic episodes...")
    syn_trust = score_episodes_with_trust(
        model, synthetic_path, metrics['X_mean'], metrics['X_std'], device
    )
    print(f"  Synthetic episodes: {len(syn_trust)}")
    print(f"  Trust mean: {syn_trust.mean():.4f}")
    print(f"  Trust std:  {syn_trust.std():.4f}")
    print(f"  Trust min:  {syn_trust.min():.4f}")
    print(f"  Trust max:  {syn_trust.max():.4f}")

    # Step 3: Save augmented datasets with trust scores
    print("\n" + "=" * 60)
    print("STEP 3: SAVE TRUST-AUGMENTED DATASETS")
    print("=" * 60)

    # Augment real data
    real_data = np.load(real_path, allow_pickle=True)
    real_augmented = dict(real_data.items())
    real_augmented['trust_scores'] = real_trust
    real_out_path = 'data/physics_zv_rollouts_trust.npz'
    np.savez(real_out_path, **real_augmented)
    print(f"Saved: {real_out_path}")

    # Augment synthetic data
    syn_data = np.load(synthetic_path, allow_pickle=True)
    syn_augmented = dict(syn_data.items())
    syn_augmented['trust_scores'] = syn_trust
    syn_out_path = 'data/synthetic_zv_rollouts_trust.npz'
    np.savez(syn_out_path, **syn_augmented)
    print(f"Saved: {syn_out_path}")

    # Step 4: Analysis
    print("\n" + "=" * 60)
    print("TRUST ANALYSIS")
    print("=" * 60)

    # Trust distribution analysis
    print("\nTrust score distribution:")
    for threshold in [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
        n_real_above = (real_trust >= threshold).sum()
        n_syn_above = (syn_trust >= threshold).sum()
        print(f"  Trust >= {threshold:.1f}: "
              f"Real={n_real_above}/{len(real_trust)} ({100*n_real_above/len(real_trust):.0f}%), "
              f"Syn={n_syn_above}/{len(syn_trust)} ({100*n_syn_above/len(syn_trust):.0f}%)")

    # High-trust synthetic episodes (good augmentation candidates)
    high_trust_threshold = 0.5
    high_trust_syn = (syn_trust >= high_trust_threshold).sum()
    print(f"\nHigh-trust synthetic episodes (>={high_trust_threshold}): {high_trust_syn}/{len(syn_trust)}")
    print(f"  These are candidates for policy augmentation")

    # Low-trust episodes to avoid
    low_trust_threshold = 0.3
    low_trust_syn = (syn_trust < low_trust_threshold).sum()
    print(f"\nLow-trust synthetic episodes (<{low_trust_threshold}): {low_trust_syn}/{len(syn_trust)}")
    print(f"  These should be heavily downweighted or dropped")

    # Save analysis report
    report = {
        'real_trust_mean': float(real_trust.mean()),
        'real_trust_std': float(real_trust.std()),
        'syn_trust_mean': float(syn_trust.mean()),
        'syn_trust_std': float(syn_trust.std()),
        'trust_gap': float(real_trust.mean() - syn_trust.mean()),
        'roc_auc': metrics['roc_auc'],
        'val_accuracy': metrics['best_val_acc'],
        'high_trust_syn_count': int(high_trust_syn),
        'low_trust_syn_count': int(low_trust_syn),
        'n_real': int(len(real_trust)),
        'n_synthetic': int(len(syn_trust)),
    }

    import json
    os.makedirs('results', exist_ok=True)
    with open('results/trust_net_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved analysis to results/trust_net_analysis.json")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Use trust scores as importance weights in offline RL:")
    print("   - High trust → count more in loss")
    print("   - Low trust → downweight or ignore")
    print("2. Retrain offline policy with trust-weighted synthetic data")
    print("3. Compare to baseline to see if trust gating helps")
    print("=" * 60)


if __name__ == '__main__':
    main()
