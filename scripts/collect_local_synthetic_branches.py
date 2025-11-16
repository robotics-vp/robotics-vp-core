#!/usr/bin/env python3
"""
Collect Local Synthetic Branches from Stable World Model

Conservative augmentation strategy:
- Start from REAL z_t (known-good states)
- Roll forward only H_short steps (5-10)
- Gate hard: trust > 0.9, std ratio in [0.8, 1.2]
- Tag with brick_id and economic metrics

This is the safest way to use the world model:
"Fill in local bubbles around real states" instead of hallucinating entire episodes.

Usage:
    python scripts/collect_local_synthetic_branches.py
    python scripts/collect_local_synthetic_branches.py --horizon 10 --min-trust 0.95
"""
# NOTE: Experimental configuration;
# actual synthetic weighting is DL-driven (trust × w_econ).
# TODO: migrate to full PolicyProfile after demo.

import os
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from src.world_model.contractive_dynamics import StableWorldModel
from src.valuation.trust_net import TrustNet
from src.config.internal_profile import get_internal_experiment_profile


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


def load_brick_manifest(manifest_path):
    """Load brick manifest if available."""
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return None


def get_episode_brick_id(ep_idx, brick_manifest):
    """Get brick ID for an episode."""
    if brick_manifest is None:
        return -1  # Unknown brick

    # brick_manifest is a list of brick objects
    for brick in brick_manifest:
        if ep_idx in brick.get('episode_ids', []):
            # Extract numeric brick ID from string like "brick_0"
            brick_id_str = brick.get('brick_id', 'brick_-1')
            try:
                return int(brick_id_str.split('_')[1])
            except:
                return -1
    return -1


def main():
    # Load experiment profile for defaults
    profile = get_internal_experiment_profile("default")

    parser = argparse.ArgumentParser(description='Collect local synthetic branches')
    parser.add_argument('--world-model', type=str, default=profile['world_model_path'])
    parser.add_argument('--dataset', type=str, default=profile['real_data_path'])
    parser.add_argument('--trust-net', type=str, default=profile['trust_net_path'])
    parser.add_argument('--brick-manifest', type=str, default=profile['brick_manifest_path'])
    parser.add_argument('--output', type=str, default=profile['synthetic_branches_path'])

    # Branch parameters
    parser.add_argument('--horizon', type=int, default=profile['max_branch_horizon'],
                        help='Branch length (steps)')
    parser.add_argument('--branches-per-episode', type=int, default=profile['branches_per_episode'],
                        help='Branches to sample per episode')

    # Gating thresholds
    parser.add_argument('--min-trust', type=float, default=profile['min_trust_threshold'],
                        help='Minimum trust score')
    parser.add_argument('--min-std-ratio', type=float, default=profile['min_std_ratio'],
                        help='Minimum std ratio')
    parser.add_argument('--max-std-ratio', type=float, default=profile['max_std_ratio'],
                        help='Maximum std ratio')

    # Objective conditioning (for future use)
    parser.add_argument('--objective-dim', type=int, default=profile['objective_dim'],
                        help='Dimension of objective vector')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print("COLLECTING LOCAL SYNTHETIC BRANCHES")
    print("="*70)
    print(f"Branch horizon: {args.horizon} steps")
    print(f"Branches per episode: {args.branches_per_episode}")
    print(f"Trust threshold: >= {args.min_trust}")
    print(f"Std ratio range: [{args.min_std_ratio}, {args.max_std_ratio}]")
    print()

    # Load real data
    print(f"Loading real data from {args.dataset}...")
    data = np.load(args.dataset, allow_pickle=True)
    n_episodes = int(data['n_episodes'])
    latent_dim = int(data['latent_dim'])

    episodes = []
    all_z = []
    for ep in range(n_episodes):
        z_seq = data[f'ep_{ep}_z_sequence']
        actions = data[f'ep_{ep}_actions']
        episodes.append({
            'z_sequence': torch.FloatTensor(z_seq).to(device),
            'actions': torch.FloatTensor(actions).to(device),
            'length': len(actions),
        })
        all_z.extend(z_seq)
    all_z = np.array(all_z)
    action_dim = episodes[0]['actions'].shape[1]

    real_z_mean = all_z.mean()
    real_z_std = all_z.std()
    print(f"Real z_V: mean={real_z_mean:.6f}, std={real_z_std:.6f}")
    print(f"Loaded {n_episodes} episodes")

    # Load brick manifest
    print(f"\nLoading brick manifest from {args.brick_manifest}...")
    brick_manifest = load_brick_manifest(args.brick_manifest)
    if brick_manifest:
        print(f"Loaded {len(brick_manifest)} bricks")
    else:
        print("No brick manifest found, brick_id will be -1")

    # Load world model
    print(f"\nLoading stable world model from {args.world_model}...")
    wm_ckpt = torch.load(args.world_model, map_location=device, weights_only=False)
    world_model = StableWorldModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=wm_ckpt.get('hidden_dim', 256),
        n_layers=wm_ckpt.get('n_layers', 3),
        alpha_init=wm_ckpt.get('alpha_init', 0.3),
        max_delta=wm_ckpt.get('max_delta', 0.15),
    )
    world_model.load_state_dict(wm_ckpt['model_state_dict'])
    world_model = world_model.to(device)
    world_model.eval()
    print(f"Model alpha: {world_model.dynamics.alpha.item():.4f}")

    # Load trust_net
    print(f"Loading trust_net from {args.trust_net}...")
    trust_ckpt = torch.load(args.trust_net, map_location=device, weights_only=False)
    trust_net = TrustNet(input_dim=6, hidden_dim=64)
    trust_net.load_state_dict(trust_ckpt['model_state_dict'])
    trust_net = trust_net.to(device)
    trust_net.eval()
    trust_mean = torch.FloatTensor(trust_ckpt['X_mean']).to(device)
    trust_std_norm = torch.FloatTensor(trust_ckpt['X_std']).to(device)

    # Collect branches
    print("\n" + "="*70)
    print("GENERATING BRANCHES")
    print("="*70)

    branches = []
    stats = {
        'total_attempted': 0,
        'passed_trust': 0,
        'passed_std': 0,
        'passed_all': 0,
        'by_brick': {},
    }

    for ep_idx in range(n_episodes):
        ep = episodes[ep_idx]
        z_real = ep['z_sequence']
        actions = ep['actions']
        T = ep['length']
        brick_id = get_episode_brick_id(ep_idx, brick_manifest)

        if brick_id not in stats['by_brick']:
            stats['by_brick'][brick_id] = {'attempted': 0, 'passed': 0}

        # Sample start positions uniformly
        if T <= args.horizon:
            continue

        max_start = T - args.horizon
        start_positions = np.random.choice(max_start, size=min(args.branches_per_episode, max_start), replace=False)

        for start_t in start_positions:
            stats['total_attempted'] += 1
            stats['by_brick'][brick_id]['attempted'] += 1

            # Roll out from real z_start
            z_init = z_real[start_t]
            actions_segment = actions[start_t:start_t + args.horizon]

            with torch.no_grad():
                z_traj = world_model.rollout(z_init, actions_segment)

            # Compute trust score
            features = extract_features_torch(z_traj)
            feat_norm = (features - trust_mean) / trust_std_norm
            trust_score = trust_net(feat_norm.unsqueeze(0)).item()

            # Compute std ratio
            synth_std = z_traj.std().item()
            std_ratio = synth_std / real_z_std

            # Gate: trust
            if trust_score < args.min_trust:
                continue
            stats['passed_trust'] += 1

            # Gate: std ratio
            if std_ratio < args.min_std_ratio or std_ratio > args.max_std_ratio:
                continue
            stats['passed_std'] += 1
            stats['passed_all'] += 1
            stats['by_brick'][brick_id]['passed'] += 1

            # Save branch
            # Default objective vector from profile (for future conditioning)
            # This can be set based on episode-level goals or task specifications
            objective_vector = np.array(profile['default_objective_vector'], dtype=np.float32)

            branch = {
                'z_sequence': z_traj.cpu().numpy(),
                'actions': actions_segment.cpu().numpy(),
                'source_episode': ep_idx,
                'source_timestep': start_t,
                'horizon': args.horizon,
                'trust_score': trust_score,
                'std_ratio': std_ratio,
                'brick_id': brick_id,
                'objective_vector': objective_vector,
            }
            branches.append(branch)

        if (ep_idx + 1) % 10 == 0:
            print(f"  Processed {ep_idx + 1}/{n_episodes} episodes, "
                  f"collected {len(branches)} branches")

    # Summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    print(f"Total branches attempted: {stats['total_attempted']}")
    print(f"Passed trust gate (>= {args.min_trust}): {stats['passed_trust']} "
          f"({100*stats['passed_trust']/max(1,stats['total_attempted']):.1f}%)")
    print(f"Passed std ratio gate [{args.min_std_ratio}, {args.max_std_ratio}]: {stats['passed_std']} "
          f"({100*stats['passed_std']/max(1,stats['total_attempted']):.1f}%)")
    print(f"Final branches collected: {stats['passed_all']}")

    if branches:
        trust_scores = np.array([b['trust_score'] for b in branches])
        std_ratios = np.array([b['std_ratio'] for b in branches])
        print(f"\nCollected branch statistics:")
        print(f"  Trust: {trust_scores.mean():.6f} +/- {trust_scores.std():.6f}")
        print(f"  Std ratio: {std_ratios.mean():.4f} +/- {std_ratios.std():.4f}")

        print(f"\nBy brick:")
        for brick_id in sorted(stats['by_brick'].keys()):
            brick_stats = stats['by_brick'][brick_id]
            pct = 100 * brick_stats['passed'] / max(1, brick_stats['attempted'])
            print(f"  Brick {brick_id}: {brick_stats['passed']}/{brick_stats['attempted']} ({pct:.1f}%)")

    # Save
    if len(branches) == 0:
        print("\nWARNING: No branches passed gating! Check thresholds.")
        return

    print(f"\nSaving {len(branches)} branches to {args.output}...")

    # Prepare data for npz
    save_data = {
        'n_branches': len(branches),
        'horizon': args.horizon,
        'latent_dim': latent_dim,
        'action_dim': action_dim,
        'objective_dim': args.objective_dim,
        'real_z_mean': real_z_mean,
        'real_z_std': real_z_std,
        'min_trust_threshold': args.min_trust,
        'min_std_ratio': args.min_std_ratio,
        'max_std_ratio': args.max_std_ratio,
    }

    # Save each branch
    for i, branch in enumerate(branches):
        save_data[f'branch_{i}_z_sequence'] = branch['z_sequence']
        save_data[f'branch_{i}_actions'] = branch['actions']
        save_data[f'branch_{i}_source_episode'] = branch['source_episode']
        save_data[f'branch_{i}_source_timestep'] = branch['source_timestep']
        save_data[f'branch_{i}_trust_score'] = branch['trust_score']
        save_data[f'branch_{i}_std_ratio'] = branch['std_ratio']
        save_data[f'branch_{i}_brick_id'] = branch['brick_id']
        save_data[f'branch_{i}_objective_vector'] = branch['objective_vector']

    np.savez(args.output, **save_data)
    print(f"Saved to {args.output}")

    # Save metadata
    metadata = {
        'world_model': args.world_model,
        'dataset': args.dataset,
        'horizon': args.horizon,
        'branches_per_episode': args.branches_per_episode,
        'objective_dim': args.objective_dim,
        'min_trust': args.min_trust,
        'min_std_ratio': args.min_std_ratio,
        'max_std_ratio': args.max_std_ratio,
        'total_attempted': stats['total_attempted'],
        'passed_trust': stats['passed_trust'],
        'passed_std': stats['passed_std'],
        'final_branches': len(branches),
        'pass_rate': 100 * len(branches) / max(1, stats['total_attempted']),
        'avg_trust': float(trust_scores.mean()) if branches else 0,
        'avg_std_ratio': float(std_ratios.mean()) if branches else 0,
        'by_brick': stats['by_brick'],
    }

    metadata_path = args.output.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"Collected {len(branches)} trusted local synthetic branches")
    print(f"Ready for trust + econ-weighted offline RL A/B test")


if __name__ == '__main__':
    main()
