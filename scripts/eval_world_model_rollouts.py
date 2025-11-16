#!/usr/bin/env python3
"""
Standalone World Model Rollout Evaluator

Single source of truth for "does this world model behave over horizon H?"

For each episode:
  - Pick a start index t
  - For horizons H in {5, 10, 20, 40, 60}:
    - Start from real z_t
    - Roll model H steps (fully autoregressive)
    - Measure:
      - std_synth / std_real
      - trust_net score
      - MSE to real endpoint
      - variance growth (z_H std / z_0 std)

Usage:
    python scripts/eval_world_model_rollouts.py --world-model checkpoints/latent_dynamics_horizon_agnostic.pt
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from scripts.train_latent_diffusion import LatentDynamicsModel
from src.valuation.trust_net import TrustNet


def extract_features_torch(z_sequence):
    """Extract episode features for trust_net."""
    global_mean = z_sequence.mean()
    global_std = z_sequence.std()
    global_min = z_sequence.min()
    global_max = z_sequence.max()
    dim_var = z_sequence.mean(dim=1).std()
    diffs = torch.abs(z_sequence[1:] - z_sequence[:-1])
    smoothness = diffs.mean()

    features = torch.stack([
        global_mean, global_std, global_min, global_max, dim_var, smoothness
    ])
    return features


def evaluate_single_rollout(world_model, z_init, actions, horizon, device):
    """
    Roll out world model for exactly `horizon` steps from z_init.

    Returns:
        z_trajectory: (H+1, latent_dim) including initial state
        final_z: (latent_dim,) final predicted z
    """
    z_current = z_init.unsqueeze(0).to(device)  # (1, latent_dim)
    z_trajectory = [z_init.to(device)]

    with torch.no_grad():
        for t in range(min(horizon, len(actions))):
            a_t = actions[t].unsqueeze(0).to(device)
            z_next, _ = world_model(z_current, a_t)
            z_trajectory.append(z_next.squeeze(0))
            z_current = z_next

    z_trajectory = torch.stack(z_trajectory, dim=0)
    return z_trajectory


def main():
    parser = argparse.ArgumentParser(description='Evaluate world model rollouts')
    parser.add_argument('--world-model', type=str, default='checkpoints/latent_dynamics_horizon_agnostic.pt')
    parser.add_argument('--dataset', type=str, default='data/physics_zv_rollouts.npz')
    parser.add_argument('--trust-net', type=str, default='checkpoints/trust_net.pt')
    parser.add_argument('--horizons', type=str, default='5,10,20,40,60')
    parser.add_argument('--num-starts-per-episode', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    horizons = [int(h) for h in args.horizons.split(',')]

    print("="*70)
    print("WORLD MODEL ROLLOUT EVALUATOR")
    print("="*70)
    print(f"Device: {device}")
    print(f"Horizons to test: {horizons}")
    print()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = np.load(args.dataset, allow_pickle=True)
    n_episodes = int(data['n_episodes'])
    latent_dim = int(data['latent_dim'])

    episodes = []
    all_z = []
    for ep in range(n_episodes):
        z_seq = data[f'ep_{ep}_z_sequence']
        actions = data[f'ep_{ep}_actions']
        episodes.append({
            'z_sequence': torch.FloatTensor(z_seq),
            'actions': torch.FloatTensor(actions),
            'length': len(actions),
        })
        all_z.extend(z_seq)
    all_z = np.array(all_z)
    action_dim = episodes[0]['actions'].shape[1]

    real_z_mean = all_z.mean()
    real_z_std = all_z.std()
    print(f"Real z_V: mean={real_z_mean:.6f}, std={real_z_std:.6f}")
    print(f"Loaded {n_episodes} episodes, avg length {np.mean([e['length'] for e in episodes]):.1f}")

    # Load world model
    print(f"\nLoading world model from {args.world_model}...")
    wm_ckpt = torch.load(args.world_model, map_location=device, weights_only=False)

    # Detect model type
    model_type = wm_ckpt.get('model_type', 'mlp')
    print(f"Model type: {model_type}")

    if model_type == 'contractive':
        # New contractive/stable model
        from src.world_model.contractive_dynamics import StableWorldModel
        world_model = StableWorldModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=wm_ckpt.get('hidden_dim', 256),
            n_layers=wm_ckpt.get('n_layers', 3),
            alpha_init=wm_ckpt.get('alpha_init', 0.3),
            max_delta=wm_ckpt.get('max_delta', 0.15),
        )
        world_model.load_state_dict(wm_ckpt['model_state_dict'])
    else:
        # Old MLP model
        world_model = LatentDynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=wm_ckpt.get('hidden_dim', 256),
        )
        world_model.load_state_dict(wm_ckpt['model_state_dict'])

    world_model = world_model.to(device)
    world_model.eval()

    # Load trust_net
    print(f"Loading trust_net from {args.trust_net}...")
    trust_ckpt = torch.load(args.trust_net, map_location=device, weights_only=False)
    trust_net = TrustNet(input_dim=6, hidden_dim=64)
    trust_net.load_state_dict(trust_ckpt['model_state_dict'])
    trust_net = trust_net.to(device)
    trust_net.eval()
    trust_mean = torch.FloatTensor(trust_ckpt['X_mean']).to(device)
    trust_std = torch.FloatTensor(trust_ckpt['X_std']).to(device)

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)

    results_by_horizon = {h: {
        'trust_scores': [],
        'std_ratios': [],
        'mse_to_endpoint': [],
        'variance_growth': [],
        'max_abs_z': [],
    } for h in horizons}

    for ep_idx in range(n_episodes):
        ep = episodes[ep_idx]
        z_seq_real = ep['z_sequence']
        actions = ep['actions']
        T = ep['length']

        # Sample start points - always include start=0 for all horizons
        start_positions = [0]  # Always start from beginning
        # Add more start positions if episode is long enough
        if T > max(horizons):
            step = max(1, (T - max(horizons)) // args.num_starts_per_episode)
            for s in range(step, T - max(horizons) + 1, step):
                start_positions.append(s)

        for start_idx in start_positions:
            z_init = z_seq_real[start_idx]

            for H in horizons:
                if start_idx + H > T:
                    continue

                # Roll out
                z_traj = evaluate_single_rollout(
                    world_model, z_init, actions[start_idx:start_idx+H], H, device
                )

                actual_H = len(z_traj) - 1  # exclude initial state
                if actual_H < H:
                    continue

                # Compute metrics
                # 1. Std ratio
                synth_std = z_traj.std().item()
                std_ratio = synth_std / real_z_std

                # 2. Trust score
                features = extract_features_torch(z_traj)
                feat_norm = (features - trust_mean) / trust_std
                trust_score = trust_net(feat_norm.unsqueeze(0)).item()

                # 3. MSE to real endpoint
                z_real_endpoint = z_seq_real[start_idx + H]
                mse = ((z_traj[-1].cpu() - z_real_endpoint) ** 2).mean().item()

                # 4. Variance growth (final vs initial)
                # Look at variance over last few steps vs first few steps
                if len(z_traj) >= 4:
                    early_var = z_traj[:3].var().item()
                    late_var = z_traj[-3:].var().item()
                    var_growth = late_var / (early_var + 1e-8)
                else:
                    var_growth = 1.0

                # 5. Max absolute z value
                max_abs = z_traj.abs().max().item()

                # Store
                results_by_horizon[H]['trust_scores'].append(trust_score)
                results_by_horizon[H]['std_ratios'].append(std_ratio)
                results_by_horizon[H]['mse_to_endpoint'].append(mse)
                results_by_horizon[H]['variance_growth'].append(var_growth)
                results_by_horizon[H]['max_abs_z'].append(max_abs)

    # Summarize
    print("\n" + "="*70)
    print("SUMMARY BY HORIZON")
    print("="*70)

    summary = {}
    for H in horizons:
        results = results_by_horizon[H]
        n_samples = len(results['trust_scores'])
        if n_samples == 0:
            print(f"\nHorizon {H}: No samples")
            continue

        trust = np.array(results['trust_scores'])
        std_ratios = np.array(results['std_ratios'])
        mse = np.array(results['mse_to_endpoint'])
        var_growth = np.array(results['variance_growth'])
        max_abs = np.array(results['max_abs_z'])

        summary[H] = {
            'n_samples': n_samples,
            'trust_mean': float(trust.mean()),
            'trust_std': float(trust.std()),
            'trust_above_0.5': int((trust >= 0.5).sum()),
            'trust_above_0.9': int((trust >= 0.9).sum()),
            'std_ratio_mean': float(std_ratios.mean()),
            'std_ratio_std': float(std_ratios.std()),
            'std_ratio_max': float(std_ratios.max()),
            'mse_mean': float(mse.mean()),
            'mse_std': float(mse.std()),
            'variance_growth_mean': float(var_growth.mean()),
            'variance_growth_max': float(var_growth.max()),
            'max_abs_z_mean': float(max_abs.mean()),
            'max_abs_z_max': float(max_abs.max()),
        }

        print(f"\nHorizon H={H} ({n_samples} samples):")
        print(f"  Trust score:  {trust.mean():.6f} +/- {trust.std():.6f}")
        print(f"    Above 0.5:  {(trust >= 0.5).sum()}/{n_samples} ({100*(trust >= 0.5).mean():.1f}%)")
        print(f"    Above 0.9:  {(trust >= 0.9).sum()}/{n_samples} ({100*(trust >= 0.9).mean():.1f}%)")
        print(f"  Std ratio:    {std_ratios.mean():.3f}x (max {std_ratios.max():.3f}x)")
        print(f"  MSE endpoint: {mse.mean():.6f} +/- {mse.std():.6f}")
        print(f"  Var growth:   {var_growth.mean():.3f}x (max {var_growth.max():.3f}x)")
        print(f"  Max |z|:      {max_abs.mean():.4f} (max {max_abs.max():.4f})")

    # Detect pathology
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    pathologies = []

    # Check for variance explosion
    if 60 in summary and summary[60]['std_ratio_mean'] > 1.5:
        ratio = summary[60]['std_ratio_mean']
        pathologies.append(f"VARIANCE EXPLOSION: 60-step rollouts have {ratio:.1f}x real std")

    # Check for trust collapse
    if 60 in summary and summary[60]['trust_mean'] < 0.01:
        pathologies.append(f"TRUST COLLAPSE: 60-step rollouts have trust {summary[60]['trust_mean']:.6f}")

    # Check for MSE explosion
    if 60 in summary and summary[60]['mse_mean'] > 0.1:
        pathologies.append(f"PREDICTION ERROR: 60-step MSE = {summary[60]['mse_mean']:.6f}")

    # Check for horizon degradation
    if 10 in summary and 60 in summary:
        trust_10 = summary[10]['trust_mean']
        trust_60 = summary[60]['trust_mean']
        if trust_10 > 0.5 and trust_60 < 0.1:
            pathologies.append(f"HORIZON MISMATCH: Trust {trust_10:.3f} @ 10 steps, but {trust_60:.6f} @ 60 steps")

    # Check for eigenvalue > 1
    if 60 in summary and summary[60]['variance_growth_max'] > 10:
        pathologies.append(f"UNSTABLE DYNAMICS: max variance growth = {summary[60]['variance_growth_max']:.1f}x")

    if pathologies:
        print("DETECTED ISSUES:")
        for p in pathologies:
            print(f"  - {p}")
        print("\nROOT CAUSE: Transition function f(z_t, a_t) is not a stable operator.")
        print("The model learned to fool trust_net on short segments but has eigenvalues > 1.")
    else:
        print("Model appears stable across all tested horizons.")

    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = 'results/world_model_rollout_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump({
            'world_model': args.world_model,
            'dataset': args.dataset,
            'horizons_tested': horizons,
            'real_z_mean': float(real_z_mean),
            'real_z_std': float(real_z_std),
            'summary': summary,
            'pathologies': pathologies,
        }, f, indent=2)
    print(f"\nSaved detailed results to {output_path}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
