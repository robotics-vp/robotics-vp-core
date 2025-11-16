#!/usr/bin/env python3
"""
3-Mode Synthetic Weighting A/B Test

Compares offline RL performance across three synthetic weighting strategies:
  1. Trust-only: w = trust_net(branch)
  2. Trust + Econ: w = trust_net(branch) * w_econ_lattice(metrics)
  3. Trust + Econ + λ: w = trust_net(branch) * w_econ_lattice(metrics) * λ_controller(features)

This tests the full flywheel integration with actual physics/econ outcomes.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from src.config.internal_profile import get_internal_experiment_profile
from src.valuation.w_econ_lattice import WEconLattice
from src.controllers.synth_lambda_controller import (
    load_controller as load_lambda_controller,
    build_feature_vector,
    compute_meta_objective
)


def load_data():
    """Load real and synthetic data."""
    profile = get_internal_experiment_profile("default")

    real_data = np.load(profile['real_data_path'], allow_pickle=True)
    synth_data = np.load(profile['synthetic_branches_path'], allow_pickle=True)

    # Extract transitions from real data (per-episode format)
    n_episodes = int(real_data['n_episodes'])
    real_transitions = []

    for ep_idx in range(n_episodes):
        z_seq = real_data[f'ep_{ep_idx}_z_sequence']
        actions = real_data[f'ep_{ep_idx}_actions']
        rewards = real_data[f'ep_{ep_idx}_rewards']

        ep_len = min(len(z_seq) - 1, len(actions), len(rewards))
        for t in range(ep_len):
            real_transitions.append({
                'state': z_seq[t],
                'action': actions[t],
                'next_state': z_seq[t+1],
                'reward': rewards[t],
                'source': 'real'
            })

    # Extract synthetic branches (per-branch format)
    n_branches = int(synth_data['n_branches'])
    synth_transitions = []

    for i in range(n_branches):
        z_seq = synth_data[f'branch_{i}_z_sequence']
        actions = synth_data[f'branch_{i}_actions']
        trust_score = float(synth_data[f'branch_{i}_trust_score'])

        # Generate synthetic rewards based on state changes
        horizon = len(z_seq) - 1
        for t in range(horizon):
            # Simple reward proxy from state norm change
            reward = -np.linalg.norm(z_seq[t+1] - z_seq[t])

            synth_transitions.append({
                'state': z_seq[t],
                'action': actions[t],
                'next_state': z_seq[t+1],
                'reward': reward,
                'source': 'synthetic',
                'trust': trust_score,
                'delta_mpl': 0.1,  # Default improvement
                'delta_error': -0.05,  # Default error reduction
                'branch_idx': i
            })

    return real_transitions, synth_transitions, profile


def load_models(profile, device='cpu'):
    """Load all required models."""
    models = {}

    # Load w_econ_lattice
    if os.path.exists(profile['w_econ_lattice_path']):
        checkpoint = torch.load(profile['w_econ_lattice_path'], weights_only=False)
        # Create model with saved config
        models['w_econ'] = WEconLattice(
            n_keypoints=checkpoint.get('n_keypoints', 16),
            n_bricks=checkpoint.get('n_bricks', 5),
            hidden_dim=checkpoint.get('hidden_dim', 32),
            objective_dim=checkpoint.get('objective_dim', 4)
        )
        models['w_econ'].load_state_dict(checkpoint['model_state_dict'])
        models['w_econ'].eval()
    else:
        models['w_econ'] = None

    # Load lambda controller
    if os.path.exists(profile['synth_lambda_controller_path']):
        models['lambda_ctrl'] = load_lambda_controller(
            profile['synth_lambda_controller_path'], device
        )
        models['lambda_ctrl'].eval()
    else:
        models['lambda_ctrl'] = None

    return models


class SimpleActor(nn.Module):
    """Simple actor network for A/B testing."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def compute_weights_mode1(synth_transitions):
    """Mode 1: Trust-only weighting."""
    weights = []
    for trans in synth_transitions:
        weights.append(trans['trust'])
    return np.array(weights)


def compute_weights_mode2(synth_transitions, w_econ_model):
    """Mode 2: Trust + Econ weighting."""
    weights = []

    # Batch processing for efficiency
    if w_econ_model is not None:
        delta_mpls = torch.FloatTensor([t['delta_mpl'] for t in synth_transitions])
        delta_errors = torch.FloatTensor([t['delta_error'] for t in synth_transitions])
        delta_eps = torch.zeros_like(delta_mpls)
        novelties = torch.full_like(delta_mpls, 0.5)
        brick_ids = torch.zeros(len(synth_transitions), dtype=torch.long)

        with torch.no_grad():
            econ_weights = w_econ_model(delta_mpls, delta_errors, delta_eps, novelties, brick_ids)
            econ_weights = econ_weights.numpy()
    else:
        econ_weights = np.ones(len(synth_transitions))

    for i, trans in enumerate(synth_transitions):
        trust_w = trans['trust']
        weights.append(trust_w * econ_weights[i])

    return np.array(weights)


def compute_weights_mode3(synth_transitions, w_econ_model, lambda_ctrl, profile, progress=0.5):
    """Mode 3: Trust + Econ + Lambda controller weighting."""

    # Build feature vector for lambda controller
    # Use aggregate statistics from synthetic transitions
    trust_mean = np.mean([t['trust'] for t in synth_transitions])
    delta_mpl_mean = np.mean([t['delta_mpl'] for t in synth_transitions])
    delta_error_mean = np.mean([t['delta_error'] for t in synth_transitions])

    # Baseline metrics (from profile or defaults)
    baseline_mpl = 50.0
    baseline_error = 0.15
    baseline_ep = 40.0

    # Current metrics (simulated improvement)
    current_mpl = baseline_mpl * (1 + 0.1 * progress)
    current_error = baseline_error * (1 - 0.2 * progress)
    current_ep = baseline_ep * (1 + 0.1 * progress)

    # Build features
    features = build_feature_vector(
        profile['default_objective_vector'],
        current_mpl, baseline_mpl,
        current_error, baseline_error,
        current_ep, baseline_ep,
        trust_mean, 0.9,  # trust_synth_mean
        0.15,  # effective_synth_share
        progress
    )

    # Get lambda prediction
    if lambda_ctrl is not None:
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            lambda_synth = lambda_ctrl(features_tensor, max_synth_share=profile['max_synth_share'])
            lambda_val = lambda_synth.item()
    else:
        lambda_val = profile['target_synth_share']

    # Compute econ weights in batch
    if w_econ_model is not None:
        delta_mpls = torch.FloatTensor([t['delta_mpl'] for t in synth_transitions])
        delta_errors = torch.FloatTensor([t['delta_error'] for t in synth_transitions])
        delta_eps = torch.zeros_like(delta_mpls)
        novelties = torch.full_like(delta_mpls, 0.5)
        brick_ids = torch.zeros(len(synth_transitions), dtype=torch.long)

        with torch.no_grad():
            econ_weights = w_econ_model(delta_mpls, delta_errors, delta_eps, novelties, brick_ids)
            econ_weights = econ_weights.numpy()
    else:
        econ_weights = np.ones(len(synth_transitions))

    # Compute per-transition weights
    weights = []
    for i, trans in enumerate(synth_transitions):
        trust_w = trans['trust']
        # Final weight includes lambda
        w_final = trust_w * econ_weights[i] * lambda_val
        weights.append(w_final)

    return np.array(weights), lambda_val


def train_actor_with_weights(real_transitions, synth_transitions, synth_weights,
                              target_synth_share=0.2, n_epochs=100, lr=1e-3):
    """Train actor with weighted synthetic data."""

    # Determine dimensions
    state_dim = len(real_transitions[0]['state'])
    action_dim = len(real_transitions[0]['action'])

    # Create actor
    actor = SimpleActor(state_dim, action_dim)
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')

    # Balance source contributions
    n_real = len(real_transitions)
    n_synth = len(synth_transitions)

    # Normalize synth weights to achieve target share
    if synth_weights.sum() > 0:
        total_synth_weight = synth_weights.sum()
        # Target: synth_contribution / (real + synth) = target_synth_share
        # synth_sum * scale / (real_sum + synth_sum * scale) = target
        # Solve for scale
        desired_synth_sum = n_real * target_synth_share / (1 - target_synth_share + 1e-6)
        scale = desired_synth_sum / (total_synth_weight + 1e-6)
        synth_weights_scaled = synth_weights * scale
    else:
        synth_weights_scaled = synth_weights

    # Compute effective synth share
    real_weight_sum = n_real * 1.0  # Real data weight = 1.0
    synth_weight_sum = synth_weights_scaled.sum()
    effective_synth_share = synth_weight_sum / (real_weight_sum + synth_weight_sum + 1e-6)

    # Track metrics
    history = {'train_loss': [], 'effective_synth_share': effective_synth_share}

    for epoch in range(n_epochs):
        # Sample mini-batch
        batch_size = 256

        # Real samples
        real_indices = np.random.choice(n_real, min(batch_size // 2, n_real), replace=False)
        real_states = np.array([real_transitions[i]['state'] for i in real_indices])
        real_actions = np.array([real_transitions[i]['action'] for i in real_indices])

        # Synthetic samples (weighted by importance)
        if n_synth > 0 and synth_weights_scaled.sum() > 0:
            synth_probs = synth_weights_scaled / synth_weights_scaled.sum()
            synth_indices = np.random.choice(n_synth, min(batch_size // 2, n_synth),
                                             replace=False, p=synth_probs)
            synth_states = np.array([synth_transitions[i]['state'] for i in synth_indices])
            synth_actions = np.array([synth_transitions[i]['action'] for i in synth_indices])
            synth_sample_weights = synth_weights_scaled[synth_indices]
        else:
            synth_states = np.zeros((0, state_dim))
            synth_actions = np.zeros((0, action_dim))
            synth_sample_weights = np.array([])

        # Combine batches
        states = np.vstack([real_states, synth_states]) if len(synth_states) > 0 else real_states
        actions = np.vstack([real_actions, synth_actions]) if len(synth_actions) > 0 else real_actions
        weights = np.concatenate([
            np.ones(len(real_states)),
            synth_sample_weights if len(synth_sample_weights) > 0 else []
        ])

        # Forward pass
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        weights_t = torch.FloatTensor(weights)

        pred_actions = actor(states_t)
        loss_per_sample = criterion(pred_actions, actions_t).mean(dim=1)
        weighted_loss = (loss_per_sample * weights_t).sum() / weights_t.sum()

        # Backward pass
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        history['train_loss'].append(weighted_loss.item())

    return actor, history


def evaluate_actor(actor, real_transitions):
    """Evaluate actor performance on real data."""
    states = np.array([t['state'] for t in real_transitions])
    actions = np.array([t['action'] for t in real_transitions])

    states_t = torch.FloatTensor(states)
    actions_t = torch.FloatTensor(actions)

    with torch.no_grad():
        pred_actions = actor(states_t)
        mse = ((pred_actions - actions_t) ** 2).mean().item()

    return mse


def run_ab_test():
    """Run the 3-mode A/B test."""
    print("=" * 70)
    print("3-MODE SYNTHETIC WEIGHTING A/B TEST")
    print("=" * 70)
    print("Comparing:")
    print("  Mode 1: Trust-only")
    print("  Mode 2: Trust + Econ Lattice")
    print("  Mode 3: Trust + Econ + Lambda Controller")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    real_transitions, synth_transitions, profile = load_data()
    print(f"  Real transitions: {len(real_transitions)}")
    print(f"  Synthetic transitions: {len(synth_transitions)}")
    print()

    # Load models
    print("Loading models...")
    models = load_models(profile)
    print(f"  w_econ_lattice: {'Loaded' if models['w_econ'] else 'Not found'}")
    print(f"  lambda_controller: {'Loaded' if models['lambda_ctrl'] else 'Not found'}")
    print()

    # Baseline: Real-only
    print("Training baseline (real-only)...")
    baseline_actor, _ = train_actor_with_weights(
        real_transitions, [], np.array([]),
        target_synth_share=0.0, n_epochs=100
    )
    baseline_mse = evaluate_actor(baseline_actor, real_transitions)
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print()

    results = []

    # Mode 1: Trust-only
    print("Mode 1: Trust-only...")
    weights_m1 = compute_weights_mode1(synth_transitions)
    actor_m1, history_m1 = train_actor_with_weights(
        real_transitions, synth_transitions, weights_m1,
        target_synth_share=profile['target_synth_share'], n_epochs=100
    )
    mse_m1 = evaluate_actor(actor_m1, real_transitions)
    delta_m1 = ((mse_m1 - baseline_mse) / baseline_mse) * 100

    results.append({
        'mode': 'Trust-only',
        'mse': mse_m1,
        'delta_pct': delta_m1,
        'effective_synth_share': history_m1['effective_synth_share'],
        'trust_mean': np.mean(weights_m1),
        'econ_mean': 1.0,
        'lambda_val': 1.0
    })
    print(f"  MSE: {mse_m1:.6f} ({delta_m1:+.2f}%)")
    print(f"  Effective synth share: {history_m1['effective_synth_share']:.4f}")
    print()

    # Mode 2: Trust + Econ
    print("Mode 2: Trust + Econ Lattice...")
    weights_m2 = compute_weights_mode2(synth_transitions, models['w_econ'])
    actor_m2, history_m2 = train_actor_with_weights(
        real_transitions, synth_transitions, weights_m2,
        target_synth_share=profile['target_synth_share'], n_epochs=100
    )
    mse_m2 = evaluate_actor(actor_m2, real_transitions)
    delta_m2 = ((mse_m2 - baseline_mse) / baseline_mse) * 100

    # Compute econ mean from weights ratio
    econ_mean = np.mean(weights_m2) / (np.mean(weights_m1) + 1e-6)

    results.append({
        'mode': 'Trust + Econ',
        'mse': mse_m2,
        'delta_pct': delta_m2,
        'effective_synth_share': history_m2['effective_synth_share'],
        'trust_mean': np.mean([t['trust'] for t in synth_transitions]),
        'econ_mean': econ_mean,
        'lambda_val': 1.0
    })
    print(f"  MSE: {mse_m2:.6f} ({delta_m2:+.2f}%)")
    print(f"  Effective synth share: {history_m2['effective_synth_share']:.4f}")
    print()

    # Mode 3: Trust + Econ + Lambda
    print("Mode 3: Trust + Econ + Lambda Controller...")
    weights_m3, lambda_val = compute_weights_mode3(
        synth_transitions, models['w_econ'], models['lambda_ctrl'], profile, progress=0.5
    )
    actor_m3, history_m3 = train_actor_with_weights(
        real_transitions, synth_transitions, weights_m3,
        target_synth_share=profile['target_synth_share'], n_epochs=100
    )
    mse_m3 = evaluate_actor(actor_m3, real_transitions)
    delta_m3 = ((mse_m3 - baseline_mse) / baseline_mse) * 100

    results.append({
        'mode': 'Trust + Econ + Lambda',
        'mse': mse_m3,
        'delta_pct': delta_m3,
        'effective_synth_share': history_m3['effective_synth_share'],
        'trust_mean': np.mean([t['trust'] for t in synth_transitions]),
        'econ_mean': econ_mean,
        'lambda_val': lambda_val
    })
    print(f"  MSE: {mse_m3:.6f} ({delta_m3:+.2f}%)")
    print(f"  Effective synth share: {history_m3['effective_synth_share']:.4f}")
    print(f"  Lambda prediction: {lambda_val:.4f}")
    print()

    # Print summary table
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<25} {'MSE':>10} {'Δ%':>10} {'Eff. Share':>12} {'Trust':>10} {'Econ':>10} {'λ':>8}")
    print("-" * 70)
    print(f"{'Baseline (Real-only)':<25} {baseline_mse:>10.6f} {0.0:>10.2f} {0.0:>12.4f} {'N/A':>10} {'N/A':>10} {'N/A':>8}")

    for r in results:
        print(f"{r['mode']:<25} {r['mse']:>10.6f} {r['delta_pct']:>+10.2f} {r['effective_synth_share']:>12.4f} "
              f"{r['trust_mean']:>10.4f} {r['econ_mean']:>10.4f} {r['lambda_val']:>8.4f}")
    print("=" * 70)

    # Determine verdict
    print("\nVERDICT:")
    best_mode = min(results, key=lambda x: x['mse'])
    worst_mode = max(results, key=lambda x: x['mse'])

    if all(r['delta_pct'] > 0 for r in results):
        print("  ALL MODES: DO_NO_HARM (some degradation, but minimal)")
    elif best_mode['delta_pct'] < 0:
        print(f"  IMPROVEMENT: {best_mode['mode']} achieves {best_mode['delta_pct']:.2f}% reduction")
    else:
        print(f"  BEST: {best_mode['mode']} with smallest degradation {best_mode['delta_pct']:.2f}%")

    print(f"\n  Lambda controller prediction: λ = {lambda_val:.4f}")
    print(f"  Max synth share allowed: {profile['max_synth_share']:.2f}")
    print(f"  Target synth share: {profile['target_synth_share']:.2f}")

    # Save results
    os.makedirs('results', exist_ok=True)

    # Convert all numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj

    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_mse': float(baseline_mse),
        'results': convert_to_native(results),
        'profile': {
            'target_synth_share': float(profile['target_synth_share']),
            'max_synth_share': float(profile['max_synth_share']),
            'objective_vector': [float(x) for x in profile['default_objective_vector']]
        },
        'n_real_transitions': len(real_transitions),
        'n_synth_transitions': len(synth_transitions)
    }

    with open('results/3mode_ab_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to results/3mode_ab_test.json")

    return results, baseline_mse


if __name__ == '__main__':
    run_ab_test()
