#!/usr/bin/env python3
"""
4-Mode Synthetic Weighting A/B Test (With λ as Budget Controller)

Compares offline RL performance across four synthetic weighting strategies:
  1. Baseline: Real-only (no synthetic)
  2. Trust-only: w = trust_net(branch)
  3. Trust + Econ (J-trained): w = trust × w_econ_lattice (trained on J outcomes)
  4. Trust + Econ + λ-Budget: w = trust × w_econ, with λ controlling overall synth budget

Key change: λ is NOT an additional per-sample gate, but a global budget controller.
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
)
from src.controllers.synthetic_weight_controller import SyntheticWeightController


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
            reward = -np.linalg.norm(z_seq[t+1] - z_seq[t])

            synth_transitions.append({
                'state': z_seq[t],
                'action': actions[t],
                'next_state': z_seq[t+1],
                'reward': reward,
                'source': 'synthetic',
                'trust': trust_score,
                'delta_mpl': 0.1,  # Simulated improvement
                'delta_error': -0.05,  # Simulated error reduction
                'branch_idx': i
            })

    return real_transitions, synth_transitions, profile


def load_models(profile, device='cpu'):
    """Load all required models."""
    models = {}

    # Load w_econ_lattice (now J-trained)
    if os.path.exists(profile['w_econ_lattice_path']):
        checkpoint = torch.load(profile['w_econ_lattice_path'], weights_only=False)
        models['w_econ'] = WEconLattice(
            n_keypoints=checkpoint.get('n_keypoints', 16),
            n_bricks=checkpoint.get('n_bricks', 5),
            hidden_dim=checkpoint.get('hidden_dim', 32),
            objective_dim=checkpoint.get('objective_dim', 4)
        )
        models['w_econ'].load_state_dict(checkpoint['model_state_dict'])
        models['w_econ'].eval()
        models['w_econ_method'] = checkpoint.get('training_method', 'unknown')
    else:
        models['w_econ'] = None
        models['w_econ_method'] = 'none'

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


def compute_quality_weights(synth_transitions, w_econ_model):
    """
    Compute per-sample quality weights: w_quality = trust × w_econ.

    This is the "quality gate" - not the final budget.
    """
    # Batch processing
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

    trust_arr = np.array([t['trust'] for t in synth_transitions], dtype=np.float32)
    quality_weights = trust_arr * econ_weights

    return np.array(quality_weights), np.array(econ_weights), trust_arr, np.mean(econ_weights)


def get_lambda_budget(synth_transitions, lambda_ctrl, profile, progress=0.5):
    """
    Get global λ budget from controller.

    λ controls how much synthetic data to use overall, NOT per-sample weighting.
    """
    trust_mean = np.mean([t['trust'] for t in synth_transitions])

    # Baseline metrics
    baseline_mpl = 50.0
    baseline_error = 0.15
    baseline_ep = 40.0

    # Current metrics (simulated)
    current_mpl = baseline_mpl * (1 + 0.1 * progress)
    current_error = baseline_error * (1 - 0.2 * progress)
    current_ep = baseline_ep * (1 + 0.1 * progress)

    # Build features
    features = build_feature_vector(
        profile['default_objective_vector'],
        current_mpl, baseline_mpl,
        current_error, baseline_error,
        current_ep, baseline_ep,
        trust_mean, 0.9,
        0.15,
        progress
    )

    # Get λ prediction
    if lambda_ctrl is not None:
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            lambda_synth = lambda_ctrl(features_tensor, max_synth_share=profile['max_synth_share'])
            lambda_val = lambda_synth.item()
    else:
        lambda_val = profile['target_synth_share']

    return lambda_val


def train_actor_with_weighted_budget(
    real_transitions,
    synth_transitions,
    quality_weights,
    target_synth_share,
    n_epochs=100,
    lr=1e-3,
    scaled_weights=None,
    weight_debug=None,
):
    """
    Train actor with quality-gated synthetic data and controlled budget.

    Key change: target_synth_share controls the BUDGET, not an additional gate.
    Quality weights determine which samples are preferred, budget controls how many.
    """
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

    # Scale quality weights to achieve target synth share
    # target_synth_share = synth_contribution / (real + synth)
    # If real has total weight = n_real, and we want synth/(real+synth) = target,
    # then synth_total_weight = n_real * target / (1 - target)

    scale = 1.0
    if scaled_weights is None:
        if quality_weights.sum() > 0 and target_synth_share > 0:
            desired_synth_total = n_real * target_synth_share / (1 - target_synth_share + 1e-6)
            scale = desired_synth_total / (quality_weights.sum() + 1e-6)
            scaled_weights = quality_weights * scale
        else:
            scaled_weights = quality_weights
        effective_synth_share = scaled_weights.sum() / (n_real + scaled_weights.sum() + 1e-6)
    else:
        effective_synth_share = scaled_weights.sum() / (n_real + scaled_weights.sum() + 1e-6)
        scale = weight_debug.get("scale_factor", 1.0) if weight_debug else 1.0

    history = {
        'train_loss': [],
        'effective_synth_share': effective_synth_share,
        'target_synth_share': target_synth_share,
        'quality_weight_mean': np.mean(quality_weights),
        'scale_factor': scale if quality_weights.sum() > 0 and target_synth_share > 0 else 1.0,
    }

    for epoch in range(n_epochs):
        # Sample mini-batch
        batch_size = 256

        # Real samples (weight = 1.0)
        real_indices = np.random.choice(n_real, min(batch_size // 2, n_real), replace=False)
        real_states = np.array([real_transitions[i]['state'] for i in real_indices])
        real_actions = np.array([real_transitions[i]['action'] for i in real_indices])

        # Synthetic samples (weighted by quality)
        if n_synth > 0 and scaled_weights.sum() > 0:
            synth_probs = scaled_weights / scaled_weights.sum()
            synth_indices = np.random.choice(n_synth, min(batch_size // 2, n_synth),
                                             replace=False, p=synth_probs)
            synth_states = np.array([synth_transitions[i]['state'] for i in synth_indices])
            synth_actions = np.array([synth_transitions[i]['action'] for i in synth_indices])
            synth_sample_weights = scaled_weights[synth_indices]
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


def run_4mode_ab_test():
    """Run the 4-mode A/B test with λ as budget controller."""
    print("=" * 70)
    print("4-MODE SYNTHETIC WEIGHTING A/B TEST")
    print("λ AS BUDGET CONTROLLER (NOT EXTRA GATE)")
    print("=" * 70)
    print("Modes:")
    print("  1. Baseline (Real-only)")
    print("  2. Trust-only")
    print("  3. Trust + Econ (J-trained lattice)")
    print("  4. Trust + Econ + λ-Budget (λ controls synth share, not per-sample)")
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
    print(f"  w_econ_lattice: {'Loaded' if models['w_econ'] else 'Not found'} "
          f"(method: {models['w_econ_method']})")
    print(f"  lambda_controller: {'Loaded' if models['lambda_ctrl'] else 'Not found'}")
    print()

    results = []

    # Mode 1: Baseline (Real-only)
    print("Mode 1: Baseline (Real-only)...")
    baseline_actor, baseline_history = train_actor_with_weighted_budget(
        real_transitions, [], np.array([]),
        target_synth_share=0.0, n_epochs=100
    )
    baseline_mse = evaluate_actor(baseline_actor, real_transitions)
    results.append({
        'mode': 'Baseline (Real-only)',
        'mse': float(baseline_mse),
        'delta_pct': 0.0,
        'effective_synth_share': 0.0,
        'quality_weight_mean': 0.0,
        'lambda_budget': 0.0,
        'trust_mean': 0.0,
        'econ_mean': 0.0,
    })
    print(f"  MSE: {baseline_mse:.6f} (baseline)")
    print()

    controller = SyntheticWeightController(
        max_synth_share=profile.get("max_synth_share", 0.4),
        econ_weight_cap=profile.get("econ_weight_cap", 1.0),
        trust_floor=profile.get("min_trust_threshold", 0.0),
        default_lambda=profile.get("target_synth_share", 0.2),
    )

    # Mode 2: Trust-only (λ = default target share)
    print("Mode 2: Trust-only...")
    trust_only_weights = controller.compute_weights(
        trust=np.array([t['trust'] for t in synth_transitions], dtype=np.float32),
        econ=np.ones(len(synth_transitions), dtype=np.float32),
        n_real=len(real_transitions),
        mode="trust_only",
        lambda_target=profile['target_synth_share'],
    )
    actor_m2, history_m2 = train_actor_with_weighted_budget(
        real_transitions, synth_transitions, trust_only_weights['quality'],
        target_synth_share=profile['target_synth_share'], n_epochs=100,
        scaled_weights=trust_only_weights['weights'], weight_debug=trust_only_weights['debug']
    )
    mse_m2 = evaluate_actor(actor_m2, real_transitions)
    delta_m2 = ((mse_m2 - baseline_mse) / baseline_mse) * 100

    results.append({
        'mode': 'Trust-only',
        'mse': float(mse_m2),
        'delta_pct': float(delta_m2),
        'effective_synth_share': float(history_m2['effective_synth_share']),
        'quality_weight_mean': float(history_m2['quality_weight_mean']),
        'lambda_budget': float(profile['target_synth_share']),
        'trust_mean': float(np.mean(trust_only_weights['trust'])),
        'econ_mean': 1.0,
    })
    print(f"  MSE: {mse_m2:.6f} ({delta_m2:+.2f}%)")
    print(f"  Effective synth share: {history_m2['effective_synth_share']:.4f}")
    print()

    # Mode 3: Trust + Econ (J-trained, default budget)
    print("Mode 3: Trust + Econ (J-trained lattice)...")
    quality_weights, econ_weights, trust_arr, econ_mean = compute_quality_weights(synth_transitions, models['w_econ'])
    trust_econ_weights = controller.compute_weights(
        trust=trust_arr,
        econ=econ_weights,
        n_real=len(real_transitions),
        mode="trust_econ",
        lambda_target=profile['target_synth_share'],
    )
    actor_m3, history_m3 = train_actor_with_weighted_budget(
        real_transitions, synth_transitions, trust_econ_weights['quality'],
        target_synth_share=profile['target_synth_share'], n_epochs=100,
        scaled_weights=trust_econ_weights['weights'], weight_debug=trust_econ_weights['debug']
    )
    mse_m3 = evaluate_actor(actor_m3, real_transitions)
    delta_m3 = ((mse_m3 - baseline_mse) / baseline_mse) * 100

    results.append({
        'mode': 'Trust + Econ (J-trained)',
        'mse': float(mse_m3),
        'delta_pct': float(delta_m3),
        'effective_synth_share': float(history_m3['effective_synth_share']),
        'quality_weight_mean': float(history_m3['quality_weight_mean']),
        'lambda_budget': float(profile['target_synth_share']),
        'trust_mean': float(np.mean(trust_arr)),
        'econ_mean': float(econ_mean),
    })
    print(f"  MSE: {mse_m3:.6f} ({delta_m3:+.2f}%)")
    print(f"  Effective synth share: {history_m3['effective_synth_share']:.4f}")
    print(f"  Econ weight mean: {econ_mean:.4f}")
    print()

    # Mode 4: Trust + Econ + λ-Budget (λ controls budget, not extra gate)
    print("Mode 4: Trust + Econ + λ-Budget...")
    lambda_budget = get_lambda_budget(synth_transitions, models['lambda_ctrl'], profile, progress=0.5)
    trust_econ_lambda = controller.compute_weights(
        trust=trust_arr,
        econ=econ_weights,
        n_real=len(real_transitions),
        mode="trust_econ_lambda",
        lambda_target=lambda_budget,
    )
    actor_m4, history_m4 = train_actor_with_weighted_budget(
        real_transitions, synth_transitions, trust_econ_lambda['quality'],
        target_synth_share=lambda_budget,  # λ as budget, not extra gate!
        n_epochs=100,
        scaled_weights=trust_econ_lambda['weights'],
        weight_debug=trust_econ_lambda['debug']
    )
    mse_m4 = evaluate_actor(actor_m4, real_transitions)
    delta_m4 = ((mse_m4 - baseline_mse) / baseline_mse) * 100

    results.append({
        'mode': 'Trust + Econ + λ-Budget',
        'mse': float(mse_m4),
        'delta_pct': float(delta_m4),
        'effective_synth_share': float(history_m4['effective_synth_share']),
        'quality_weight_mean': float(history_m4['quality_weight_mean']),
        'lambda_budget': float(lambda_budget),
        'trust_mean': float(np.mean([t['trust'] for t in synth_transitions])),
        'econ_mean': float(econ_mean),
    })
    print(f"  MSE: {mse_m4:.6f} ({delta_m4:+.2f}%)")
    print(f"  Effective synth share: {history_m4['effective_synth_share']:.4f}")
    print(f"  λ-budget prediction: {lambda_budget:.4f}")
    print()

    # Print summary table
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Mode':<30} {'MSE':>10} {'Δ%':>10} {'Eff.Share':>12} {'λ-Budget':>10} {'Econ':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['mode']:<30} {r['mse']:>10.6f} {r['delta_pct']:>+10.2f} "
              f"{r['effective_synth_share']:>12.4f} {r['lambda_budget']:>10.4f} {r['econ_mean']:>10.4f}")
    print("=" * 90)

    # Verdict
    print("\nVERDICT:")
    best_mode = min(results[1:], key=lambda x: x['mse'])  # Exclude baseline
    print(f"  BEST: {best_mode['mode']} with {best_mode['delta_pct']:+.2f}% change")

    if all(r['delta_pct'] >= 0 for r in results[1:]):
        print("  STATUS: All modes show degradation (do-no-harm violated)")
    elif all(r['delta_pct'] <= 0 for r in results[1:]):
        print("  STATUS: ALL MODES IMPROVE on baseline!")
    else:
        improving = [r for r in results[1:] if r['delta_pct'] < 0]
        print(f"  STATUS: {len(improving)}/{len(results)-1} modes show improvement")

    print(f"\n  λ-controller prediction: {lambda_budget:.4f}")
    print(f"  Max synth share allowed: {profile['max_synth_share']:.2f}")
    print(f"  Default target share: {profile['target_synth_share']:.2f}")
    print(f"  w_econ_lattice training method: {models['w_econ_method']}")

    # Save results
    os.makedirs('results', exist_ok=True)

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
        'w_econ_method': models['w_econ_method'],
        'n_real_transitions': len(real_transitions),
        'n_synth_transitions': len(synth_transitions),
        'lambda_budget_prediction': float(lambda_budget),
    }

    with open('results/4mode_ab_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to results/4mode_ab_test.json")

    return results, baseline_mse


if __name__ == '__main__':
    run_4mode_ab_test()
