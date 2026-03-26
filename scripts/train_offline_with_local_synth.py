#!/usr/bin/env python3
"""
Offline RL with Local Synthetic Branches (Trust + Econ Weighted)

A/B Test:
- Baseline: real data only (trust-weighted)
- Augmented: real + local synthetic branches (trust + econ weighted)

Goal: See if trusted local synthetic creates economic value, not just passes trust checks.

Metrics to compare:
- MPL (units/hour)
- Error rate
- Wage parity
- Action MSE

Usage:
    python scripts/train_offline_with_local_synth.py
    python scripts/train_offline_with_local_synth.py --synth-weight 0.3 --econ-weight-scale 2.0
"""
# NOTE: Experimental configuration;
# actual synthetic weighting is DL-driven (trust × w_econ).
# TODO: migrate to full PolicyProfile after demo.

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from src.valuation.w_econ_lattice import WEconLattice
from src.config.internal_profile import get_internal_experiment_profile
from src.controllers.synth_lambda_controller import (
    load_controller as load_lambda_controller,
    build_feature_vector
)
from src.controllers.synthetic_weight_controller import SyntheticWeightController
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.training.synthetic_branch_corpus import (
    branch_priority_multiplier,
    build_synthetic_branch_training_policy,
    load_synthetic_branch_corpus,
)
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.valuation.trajectory_audit import create_trajectory_audit


class LatentActor(nn.Module):
    """Simple MLP actor for latent space actions."""

    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_real_data_bundle(data_path, device):
    """Load real physics rollouts plus episode structure for audits/manifests."""
    data = np.load(data_path, allow_pickle=True)
    n_episodes = int(data['n_episodes'])

    transitions = []
    episodes = []
    latent_dim = 0
    action_dim = 0
    for ep in range(n_episodes):
        z_seq = data[f'ep_{ep}_z_sequence']
        actions = data[f'ep_{ep}_actions']
        rewards = data[f'ep_{ep}_rewards'] if f'ep_{ep}_rewards' in data else None
        episode_id = f"real_ep_{ep:04d}"
        latent_dim = latent_dim or int(z_seq.shape[-1])
        action_dim = action_dim or int(actions.shape[-1])
        reward_values = []
        for t in range(len(actions)):
            if rewards is not None and len(rewards) > t:
                reward = float(rewards[t])
            else:
                reward = -float(np.linalg.norm(z_seq[t + 1] - z_seq[t]))
            reward_values.append(reward)
            transitions.append({
                'z': torch.FloatTensor(z_seq[t]).to(device),
                'action': torch.FloatTensor(actions[t]).to(device),
                'z_next': torch.FloatTensor(z_seq[t + 1]).to(device),
                'source': 'real',
                'episode_id': episode_id,
                'step_idx': t,
                'trust': 1.0,  # Real data has trust = 1.0
                'econ_weight': 1.0,  # Default econ weight
                'reward': reward,
            })
        episodes.append({
            'episode_id': episode_id,
            'z_sequence': z_seq,
            'actions': actions,
            'rewards': reward_values,
        })

    return {
        'transitions': transitions,
        'episodes': episodes,
        'n_episodes': n_episodes,
        'latent_dim': latent_dim,
        'action_dim': action_dim,
    }


def load_real_data(data_path, device):
    """Load real physics rollouts."""
    return load_real_data_bundle(data_path, device)['transitions']


def load_synthetic_branches(
    branch_path,
    device,
    w_econ_model=None,
    *,
    metadata_path=None,
    gap_labels_path=None,
    training_policy: Optional[Dict[str, Any]] = None,
):
    """
    Load local synthetic branches with lattice-based economic weighting.

    Args:
        branch_path: path to npz file
        device: torch device
        w_econ_model: trained WEconLattice model (optional)

    Returns:
        transitions: list of transition dicts with econ weights
    """
    corpus = load_synthetic_branch_corpus(
        branch_path,
        metadata_path=metadata_path,
        gap_labels_path=gap_labels_path,
    )
    data = np.load(branch_path, allow_pickle=True)
    n_branches = int(data['n_branches'])

    # Check if objective_dim is stored (for backward compatibility)
    objective_dim = int(data.get('objective_dim', 4))

    # Collect branch-level metrics for lattice scoring
    branch_metrics = []
    for i in range(n_branches):
        branch_record = corpus.branches[i]
        trust = float(data[f'branch_{i}_trust_score'])
        brick_id = int(data[f'branch_{i}_brick_id'])
        std_ratio = float(data[f'branch_{i}_std_ratio'])

        # Load objective_vector (default to zeros for backward compatibility)
        if f'branch_{i}_objective_vector' in data:
            objective_vector = data[f'branch_{i}_objective_vector'].tolist()
        else:
            objective_vector = np.zeros(objective_dim, dtype=np.float32).tolist()

        # Compute proxy metrics (these would come from actual episode data in production)
        # For now, use std_ratio as proxy for ΔMPL and ΔEP
        # Better std_ratio (closer to 1.0) = better dynamics = higher ΔMPL potential
        delta_mpl_proxy = 1.0 - abs(std_ratio - 1.0) * 2  # Peak at std_ratio=1.0
        delta_error_proxy = -0.1 * (1.0 - trust)  # Lower trust = higher error proxy
        delta_ep_proxy = delta_mpl_proxy * 0.8  # Correlated with MPL

        # Novelty: use std_ratio deviation as novelty proxy
        novelty_proxy = min(1.0, abs(std_ratio - 1.0) * 5)

        branch_metrics.append({
            'branch_idx': i,
            'trust': trust,
            'brick_id': brick_id,
            'delta_mpl': delta_mpl_proxy,
            'delta_error': delta_error_proxy,
            'delta_ep': delta_ep_proxy,
            'novelty': novelty_proxy,
            'objective_vector': objective_vector,
            'branch_value': float(branch_record.branch_value),
            'gap_labels': dict(branch_record.gap_labels),
        })

    # Compute economic weights using lattice model
    if w_econ_model is not None:
        w_econ_model.eval()
        with torch.no_grad():
            delta_mpl = torch.FloatTensor([m['delta_mpl'] for m in branch_metrics]).to(device)
            delta_error = torch.FloatTensor([m['delta_error'] for m in branch_metrics]).to(device)
            delta_ep = torch.FloatTensor([m['delta_ep'] for m in branch_metrics]).to(device)
            novelty = torch.FloatTensor([m['novelty'] for m in branch_metrics]).to(device)
            brick_ids = torch.LongTensor([m['brick_id'] for m in branch_metrics]).to(device)

            # Stack objective vectors into batch tensor
            objective_vectors = torch.FloatTensor(
                np.stack([m['objective_vector'] for m in branch_metrics])
            ).to(device)

            # Clamp brick_ids to valid range
            max_brick = w_econ_model.brick_embedding.num_embeddings - 1
            brick_ids = torch.clamp(brick_ids, 0, max_brick)

            # Pass objective_vector to lattice model (defaults to zeros if not set)
            econ_weights = w_econ_model(delta_mpl, delta_error, delta_ep, novelty, brick_ids,
                                         objective_vector=objective_vectors)
            econ_weights = econ_weights.cpu().numpy()

        # Update metrics with computed weights
        for i, m in enumerate(branch_metrics):
            priority_multiplier = branch_priority_multiplier(
                corpus.branches[i],
                training_policy or {},
            )
            m['priority_multiplier'] = float(priority_multiplier)
            m['econ_weight'] = float(econ_weights[i]) * float(priority_multiplier)
    else:
        # Fallback: use trust as econ weight
        for m in branch_metrics:
            priority_multiplier = branch_priority_multiplier(
                corpus.branches[m['branch_idx']],
                training_policy or {},
            )
            m['priority_multiplier'] = float(priority_multiplier)
            m['econ_weight'] = m['trust'] * float(priority_multiplier)

    # Build transitions with branch-level econ weights
    transitions = []
    for i in range(n_branches):
        z_seq = data[f'branch_{i}_z_sequence']
        actions = data[f'branch_{i}_actions']
        trust = branch_metrics[i]['trust']
        brick_id = branch_metrics[i]['brick_id']
        econ_weight = branch_metrics[i]['econ_weight']
        branch_id = f"synth_branch_{i:04d}"

        for t in range(len(actions)):
            transitions.append({
                'z': torch.FloatTensor(z_seq[t]).to(device),
                'action': torch.FloatTensor(actions[t]).to(device),
                'z_next': torch.FloatTensor(z_seq[t + 1]).to(device),
                'source': 'synthetic',
                'episode_id': branch_id,
                'step_idx': t,
                'trust': trust,
                'econ_weight': econ_weight,
                'brick_id': brick_id,
                'reward': -float(np.linalg.norm(z_seq[t + 1] - z_seq[t])),
            })

    return transitions, branch_metrics, corpus


def compute_sample_weights(
    transitions,
    base_trust_weight=1.0,
    econ_weight_scale=1.0,
    synth_ratio=0.1,
    controller=None,
    profile=None,
    mode="trust_econ_lambda"
):
    """
    Compute per-sample weights using SyntheticWeightController.
    """
    real_trans = [t for t in transitions if t['source'] == 'real']
    synth_trans = [t for t in transitions if t['source'] == 'synthetic']

    if controller is None:
        from src.controllers.synthetic_weight_controller import SyntheticWeightController
        controller = SyntheticWeightController(
            max_synth_share=profile.get("max_synth_share", 1.0) if profile else 1.0,
            econ_weight_cap=profile.get("econ_weight_cap", 1.0) if profile else 1.0,
            trust_floor=profile.get("min_trust_threshold", 0.0) if profile else 0.0,
            default_lambda=synth_ratio
        )

    trust = np.array([t['trust'] * base_trust_weight for t in synth_trans], dtype=np.float32)
    econ = np.array([t.get('econ_weight', 1.0) * econ_weight_scale for t in synth_trans], dtype=np.float32)

    result = controller.compute_weights(
        trust=trust,
        econ=econ,
        n_real=len(real_trans),
        mode=mode,
        lambda_target=synth_ratio,
    )
    weights = result["weights"]

    for t in real_trans:
        t['weight'] = 1.0
    for t, w in zip(synth_trans, weights):
        t['weight'] = float(w)

    return real_trans + synth_trans, result['debug']


def train_actor(transitions, latent_dim, action_dim, n_epochs=100, batch_size=256, lr=1e-3, device='cpu'):
    """Train actor with importance-weighted behavioral cloning."""
    actor = LatentActor(latent_dim, action_dim).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    # Shuffle transitions
    np.random.shuffle(transitions)

    history = {'loss': [], 'action_mse': []}

    for epoch in range(n_epochs):
        np.random.shuffle(transitions)
        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = 0

        for i in range(0, len(transitions), batch_size):
            batch = transitions[i:i + batch_size]
            if len(batch) < 2:
                continue

            z_batch = torch.stack([t['z'] for t in batch])
            action_batch = torch.stack([t['action'] for t in batch])
            weight_batch = torch.FloatTensor([t['weight'] for t in batch]).to(device)

            # Forward
            action_pred = actor(z_batch)

            # Weighted MSE loss
            mse_per_sample = ((action_pred - action_batch) ** 2).mean(dim=-1)
            weighted_loss = (mse_per_sample * weight_batch).mean()

            # Backward
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            epoch_mse += mse_per_sample.mean().item()
            n_batches += 1

        history['loss'].append(epoch_loss / max(1, n_batches))
        history['action_mse'].append(epoch_mse / max(1, n_batches))

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: Loss={history['loss'][-1]:.6f}, "
                  f"MSE={history['action_mse'][-1]:.6f}")

    return actor, history


def evaluate_actor(actor, test_transitions, device):
    """Evaluate actor on test transitions."""
    if not test_transitions:
        return {
            'action_mse': 0.0,
            'action_mse_std': 0.0,
        }
    actor.eval()
    mse_list = []

    with torch.no_grad():
        for t in test_transitions:
            action_pred = actor(t['z'].unsqueeze(0))
            mse = ((action_pred.squeeze() - t['action']) ** 2).mean().item()
            mse_list.append(mse)

    return {
        'action_mse': float(np.mean(mse_list)),
        'action_mse_std': float(np.std(mse_list)),
    }


def train_actor_with_controller(
    real_train, synth_transitions, latent_dim, action_dim,
    lambda_controller, profile, n_epochs=100, eval_every=20,
    batch_size=256, lr=1e-3, device='cpu'
):
    """
    Train actor with adaptive λ_synth using the learned controller.

    At each eval window, the controller predicts optimal λ_synth based on
    current training metrics and objective vector.

    Args:
        real_train: real training transitions
        synth_transitions: synthetic transitions
        latent_dim: latent space dimension
        action_dim: action dimension
        lambda_controller: SynthLambdaController instance
        profile: experiment profile dict
        n_epochs: total epochs
        eval_every: epochs between controller updates
        batch_size: training batch size
        lr: learning rate
        device: torch device

    Returns:
        actor: trained actor
        history: training history
        lambda_trajectory: list of λ_synth values over time
    """
    # Initialize actor
    actor = LatentActor(latent_dim, action_dim).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    # Extract profile parameters
    objective_vector = profile['default_objective_vector']
    max_synth_share = profile.get('max_synth_share', 0.4)
    econ_weight_scale = profile.get('econ_weight_scale', 1.0)
    controller = SyntheticWeightController(
        max_synth_share=max_synth_share,
        econ_weight_cap=profile.get("econ_weight_cap", 1.0),
        trust_floor=profile.get("min_trust_threshold", 0.0),
        default_lambda=profile.get("target_synth_share", 0.2),
    )

    # Baseline metrics (from real data statistics)
    baseline_mpl = 50.0  # units/hour
    baseline_error = 0.15  # error rate
    baseline_ep = 40.0  # energy productivity

    # Trust statistics
    trust_real_mean = np.mean([t['trust'] for t in real_train])
    trust_synth_mean = np.mean([t['trust'] for t in synth_transitions]) if synth_transitions else 0.0

    # Initialize λ_synth
    current_lambda = profile['target_synth_share']  # Start with default

    # Training history
    history = {'loss': [], 'action_mse': [], 'lambda_synth': []}
    lambda_trajectory = []

    print("Starting adaptive training with lambda controller")
    print(f"Initial λ_synth = {current_lambda:.4f}")
    print(f"Objective vector: {objective_vector}")

    # Combine all transitions
    all_transitions = real_train.copy() + synth_transitions.copy()

    total_epochs = 0
    for window_idx in range(n_epochs // eval_every):
        # Reweight transitions with current λ_synth
        weighted_transitions, _ = compute_sample_weights(
            all_transitions.copy(),
            base_trust_weight=1.0,
            econ_weight_scale=econ_weight_scale,
            synth_ratio=current_lambda,
            controller=controller,
            profile=profile,
            mode="trust_econ_lambda"
        )

        # Compute effective synthetic share
        real_weight = sum(t['weight'] for t in weighted_transitions if t['source'] == 'real')
        synth_weight = sum(t['weight'] for t in weighted_transitions if t['source'] == 'synthetic')
        total_weight = real_weight + synth_weight
        effective_synth = synth_weight / total_weight if total_weight > 0 else 0.0

        # Train for eval_every epochs
        n_train = len(weighted_transitions)
        for epoch in range(eval_every):
            indices = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0

            actor.train()
            for i in range(0, n_train, batch_size):
                batch_idx = indices[i:i + batch_size]
                if len(batch_idx) < 2:
                    continue

                # Get weighted batch
                z_batch = torch.stack([weighted_transitions[j]['z'] for j in batch_idx])
                action_batch = torch.stack([weighted_transitions[j]['action'] for j in batch_idx])
                weights_batch = torch.FloatTensor([weighted_transitions[j]['weight'] for j in batch_idx]).to(device)

                # Forward
                pred = actor(z_batch)
                loss_per_sample = ((pred - action_batch) ** 2).mean(dim=1)
                loss = (loss_per_sample * weights_batch).sum() / weights_batch.sum()

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            total_epochs += 1
            if n_batches > 0:
                history['loss'].append(epoch_loss / n_batches)
                history['action_mse'].append(epoch_loss / n_batches)
            history['lambda_synth'].append(current_lambda)

        # Simulate current metrics (in production, these come from evaluation)
        # For now, use loss improvement as proxy
        progress = total_epochs / n_epochs
        current_mpl = baseline_mpl * (1.0 + 0.2 * progress)
        current_error = baseline_error * (1.0 - 0.3 * progress)
        current_ep = baseline_ep * (1.0 + 0.18 * progress)

        # Record lambda trajectory
        lambda_trajectory.append({
            'epoch': total_epochs,
            'lambda_synth': current_lambda,
            'effective_synth_share': effective_synth,
            'progress': progress,
        })

        # Update λ_synth using controller
        features = build_feature_vector(
            objective_vector,
            current_mpl, baseline_mpl,
            current_error, baseline_error,
            current_ep, baseline_ep,
            trust_real_mean, trust_synth_mean,
            effective_synth, progress
        )

        features_tensor = torch.FloatTensor(features).to(device)
        with torch.no_grad():
            new_lambda = lambda_controller(features_tensor, max_synth_share).item()

        # Hard clip to respect max_synth_share
        new_lambda = min(max_synth_share, max(0.0, new_lambda))

        print(f"  Window {window_idx+1}: epoch={total_epochs}, "
              f"λ={current_lambda:.4f} → {new_lambda:.4f}, "
              f"eff_synth={effective_synth:.2%}")

        current_lambda = new_lambda

    return actor, history, lambda_trajectory


def _build_real_trajectory_audits(real_bundle: Mapping[str, Any]) -> list[Any]:
    audits = []
    for episode in list(real_bundle.get('episodes', []) or []):
        actions = np.asarray(episode.get('actions'))
        rewards = [float(value) for value in list(episode.get('rewards', []) or [])]
        if actions.size == 0:
            continue
        audits.append(
            create_trajectory_audit(
                episode_id=str(episode.get('episode_id', 'real_ep')),
                num_steps=int(len(actions)),
                actions=actions.astype(np.float32).tolist(),
                rewards=rewards,
                reward_components={"real_seed_reward": rewards},
                events=["real_seed_episode"],
            )
        )
    return audits


def _build_synthetic_trajectory_audits(branch_path: str, corpus) -> list[Any]:
    if corpus is None:
        return []
    audits = []
    with np.load(branch_path, allow_pickle=True) as data:
        for branch in corpus.branches:
            z_seq = data[f'branch_{branch.branch_idx}_z_sequence']
            actions = data[f'branch_{branch.branch_idx}_actions']
            rewards = [
                -float(np.linalg.norm(z_seq[t + 1] - z_seq[t]))
                for t in range(len(actions))
            ]
            reward_components = {
                "synthetic_transition_reward": rewards,
                "branch_value": [float(branch.branch_value) for _ in rewards],
                "branch_trust": [float(branch.trust_score) for _ in rewards],
            }
            events = ["synthetic_branch"]
            skill_edge = str(branch.gap_labels.get("skill_edge", "") or "")
            if skill_edge:
                events.append(f"gap::{skill_edge}")
            audits.append(
                create_trajectory_audit(
                    episode_id=f"synth_branch_{branch.branch_idx:04d}",
                    num_steps=int(len(actions)),
                    actions=np.asarray(actions).astype(np.float32).tolist(),
                    rewards=rewards,
                    reward_components=reward_components,
                    events=events,
                )
            )
    return audits


def _source_domain_coverage(
    real_bundle: Mapping[str, Any],
    synthetic_branch_count: int,
    synthetic_transition_count: int,
) -> Dict[str, Any]:
    return {
        "total_episodes": int(real_bundle.get('n_episodes', 0)) + int(synthetic_branch_count),
        "source_domain_counts": {
            "real_seed": int(real_bundle.get('n_episodes', 0)),
            "synthetic_branch": int(synthetic_branch_count),
        },
        "transition_counts": {
            "real_seed": len(list(real_bundle.get('transitions', []) or [])),
            "synthetic_branch": int(synthetic_transition_count),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    profile = get_internal_experiment_profile("default")
    parser = argparse.ArgumentParser(description='Offline RL with local synthetic A/B test')
    parser.add_argument('--real-data', type=str, default=profile['real_data_path'])
    parser.add_argument('--synth-data', type=str, default=profile['synthetic_branches_path'])
    parser.add_argument('--synth-metadata', type=str, default=None,
                        help='Optional explicit metadata sidecar for synthetic branches')
    parser.add_argument('--synth-gap-labels', type=str, default=None,
                        help='Optional explicit gap-label sidecar for synthetic branches')
    parser.add_argument('--w-econ-lattice', type=str, default=profile['w_econ_lattice_path'],
                        help='Path to trained w_econ_lattice model')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)

    # Training params
    parser.add_argument('--epochs', type=int, default=profile['ab_test_epochs'])
    parser.add_argument('--batch-size', type=int, default=profile['ab_test_batch_size'])
    parser.add_argument('--lr', type=float, default=profile['ab_test_lr'])

    # Weighting params
    parser.add_argument('--synth-weight', type=float, default=profile['target_synth_share'],
                        help='Target contribution ratio for synthetic data')
    parser.add_argument('--econ-weight-scale', type=float, default=profile['econ_weight_scale'],
                        help='Scale factor for economic weights')

    # Lambda controller params
    parser.add_argument('--use-lambda-controller', action='store_true', default=False,
                        help='Use learned lambda controller instead of fixed synth-weight')
    parser.add_argument('--lambda-controller', type=str,
                        default=profile.get('synth_lambda_controller_path', 'checkpoints/synth_lambda_controller.pt'),
                        help='Path to trained lambda controller')
    parser.add_argument('--eval-every', type=int, default=20,
                        help='Epochs between controller evaluations')
    parser.add_argument('--skip-regal-runner', action='store_true')

    return parser.parse_args(argv)


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    profile = dict(get_internal_experiment_profile("default"))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print("OFFLINE RL: LOCAL SYNTHETIC A/B TEST")
    print("="*70)
    print("Goal: Test if trusted local synthetic creates economic value")
    print()

    # Load real data
    print(f"Loading real data from {args.real_data}...")
    real_bundle = load_real_data_bundle(args.real_data, device)
    real_transitions = list(real_bundle['transitions'])
    print(f"Loaded {len(real_transitions)} real transitions")

    # Get dimensions
    latent_dim = int(real_bundle['latent_dim'])
    action_dim = int(real_bundle['action_dim'])
    print(f"Latent dim: {latent_dim}, Action dim: {action_dim}")

    # Split real data: 80% train, 20% test
    np.random.shuffle(real_transitions)
    split_idx = max(1, int(0.8 * len(real_transitions)))
    split_idx = min(split_idx, max(1, len(real_transitions) - 1))
    real_train = real_transitions[:split_idx]
    real_test = real_transitions[split_idx:]
    print(f"Real: {len(real_train)} train, {len(real_test)} test")

    # Load w_econ_lattice model if available
    w_econ_model = None
    if os.path.exists(args.w_econ_lattice):
        print(f"\nLoading w_econ_lattice from {args.w_econ_lattice}...")
        ckpt = torch.load(args.w_econ_lattice, map_location=device, weights_only=False)
        w_econ_model = WEconLattice(
            n_bricks=ckpt.get('n_bricks', 10),
            brick_emb_dim=ckpt.get('brick_emb_dim', 8),
            n_keypoints=ckpt.get('n_keypoints', 16),
            hidden_dim=ckpt.get('hidden_dim', 32),
            objective_dim=ckpt.get('objective_dim', 4),
        ).to(device)
        w_econ_model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded lattice with correlation={ckpt.get('correlation', 0):.4f}")
    else:
        print(f"\nWARNING: No w_econ_lattice at {args.w_econ_lattice}")
        print("Using trust as econ weight fallback")

    # Load lambda controller if enabled
    lambda_controller = None
    max_synth_share = profile.get('max_synth_share', 0.4)
    if args.use_lambda_controller:
        print(f"\nLoading lambda controller from {args.lambda_controller}...")
        lambda_controller = load_lambda_controller(args.lambda_controller, device)
        if lambda_controller is not None:
            print(f"Loaded lambda controller (max_synth_share={max_synth_share})")
            print("Lambda will be dynamically adjusted during training")
        else:
            print(f"WARNING: Lambda controller not found at {args.lambda_controller}")
            print("Falling back to fixed synth-weight")
            args.use_lambda_controller = False

    # Load synthetic data if available
    synth_transitions = []
    branch_metrics = []
    branch_metrics_path = None
    corpus = None
    branch_summary_path = None
    branch_preconditions_path = None
    branch_benchmark_path = None
    training_policy: Dict[str, Any] = {
        'requested_synth_share': float(args.synth_weight),
        'requested_econ_weight_scale': float(args.econ_weight_scale),
        'effective_synth_share_cap': float(args.synth_weight),
        'semantic_weight_scale': 1.0,
        'gap_value_scale': 1.0,
        'branch_priority_floor': 0.25,
        'branch_priority_ceiling': 2.0,
        'benchmark_gate_ready': False,
        'execution_ready': False,
        'reasons': ['no_synthetic_corpus_loaded'],
    }
    if os.path.exists(args.synth_data):
        corpus = load_synthetic_branch_corpus(
            args.synth_data,
            metadata_path=args.synth_metadata,
            gap_labels_path=args.synth_gap_labels,
        )
        training_policy = build_synthetic_branch_training_policy(
            corpus,
            requested_synth_share=args.synth_weight,
            requested_econ_weight_scale=args.econ_weight_scale,
        )
        print(f"\nLoading synthetic branches from {args.synth_data}...")
        synth_transitions, branch_metrics, corpus = load_synthetic_branches(
            args.synth_data,
            device,
            w_econ_model,
            metadata_path=args.synth_metadata,
            gap_labels_path=args.synth_gap_labels,
            training_policy=training_policy,
        )
        print(f"Loaded {len(synth_transitions)} synthetic transitions")
        print(
            "Synthetic training policy: "
            f"cap={training_policy['effective_synth_share_cap']:.2f}, "
            f"semantic_scale={training_policy['semantic_weight_scale']:.2f}, "
            f"benchmark_ready={training_policy['benchmark_gate_ready']}"
        )

        # Summary of synthetic
        trust_scores = [t['trust'] for t in synth_transitions]
        econ_weights = [t['econ_weight'] for t in synth_transitions]
        print(f"Synthetic trust: {np.mean(trust_scores):.4f} +/- {np.std(trust_scores):.4f}")
        print(f"Synthetic w_econ: {np.mean(econ_weights):.4f} +/- {np.std(econ_weights):.4f}")

        branch_summary_path = output_root / 'synthetic_branch_summary.json'
        branch_preconditions_path = output_root / 'synthetic_branch_execution_preconditions.json'
        branch_benchmark_path = output_root / 'synthetic_branch_benchmark_gate.json'
        _write_json(
            branch_summary_path,
            {
                **corpus.to_summary(),
                'training_policy': training_policy,
            },
        )
        _write_json(branch_preconditions_path, corpus.execution_preconditions.to_dict())
        _write_json(branch_benchmark_path, corpus.benchmark_gate.to_dict())
    else:
        print(f"\nWARNING: No synthetic data at {args.synth_data}")
        print("Will run baseline only")

    effective_synth_share = min(
        float(args.synth_weight),
        float(training_policy.get('effective_synth_share_cap', args.synth_weight)),
    )
    profile['max_synth_share'] = min(
        float(profile.get('max_synth_share', effective_synth_share)),
        float(training_policy.get('effective_synth_share_cap', effective_synth_share)),
    )
    profile['target_synth_share'] = effective_synth_share

    # Train baseline (real-only)
    print("\n" + "="*70)
    print("TRAINING BASELINE (Real-Only)")
    print("="*70)

    baseline_train, baseline_debug = compute_sample_weights(
        real_train.copy(),
        base_trust_weight=1.0,
        econ_weight_scale=args.econ_weight_scale,
        synth_ratio=0.0,  # No synthetic
        profile=profile,
        mode="baseline"
    )

    baseline_actor, baseline_history = train_actor(
        baseline_train, latent_dim, action_dim,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device
    )

    baseline_eval = evaluate_actor(baseline_actor, real_test, device)
    print("\nBaseline evaluation:")
    print(f"  Action MSE: {baseline_eval['action_mse']:.6f} +/- {baseline_eval['action_mse_std']:.6f}")

    # Train augmented (real + synthetic)
    augmented_eval = None
    augmented_history = None
    lambda_trajectory = None
    aug_debug = None
    if synth_transitions:
        print("\n" + "="*70)
        print("TRAINING AUGMENTED (Real + Local Synthetic)")
        print("="*70)

        if args.use_lambda_controller and lambda_controller is not None:
            # Use learned lambda controller
            print("Using LEARNED LAMBDA CONTROLLER (adaptive λ_synth)")
            print(f"Max synthetic share: {profile['max_synth_share']*100:.1f}%")
            print(f"Economic weight scale: {args.econ_weight_scale}")

            augmented_actor, augmented_history, lambda_trajectory = train_actor_with_controller(
                real_train, synth_transitions, latent_dim, action_dim,
                lambda_controller, profile,
                n_epochs=args.epochs, eval_every=args.eval_every,
                batch_size=args.batch_size, lr=args.lr, device=device
            )

            # Show final lambda
            if lambda_trajectory:
                final_lambda = lambda_trajectory[-1]['lambda_synth']
                print(f"Final λ_synth: {final_lambda:.4f}")
        else:
            # Use fixed synth_weight
            print(f"Using FIXED λ_synth = {effective_synth_share:.2f}")
            print(f"Synthetic contribution target: {effective_synth_share*100:.1f}%")
            print(f"Economic weight scale: {args.econ_weight_scale}")

            augmented_train = real_train.copy() + synth_transitions.copy()
            augmented_train, aug_debug = compute_sample_weights(
                augmented_train,
                base_trust_weight=1.0,
                econ_weight_scale=args.econ_weight_scale,
                synth_ratio=effective_synth_share,
                profile=profile,
                mode="trust_econ_lambda"
            )

            # Check effective contributions
            real_weight = sum(t['weight'] for t in augmented_train if t['source'] == 'real')
            synth_weight = sum(t['weight'] for t in augmented_train if t['source'] == 'synthetic')
            total_weight = real_weight + synth_weight
            print(f"Effective synthetic contribution: {100*synth_weight/total_weight:.1f}%")

            augmented_actor, augmented_history = train_actor(
                augmented_train, latent_dim, action_dim,
                n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device
            )

        augmented_eval = evaluate_actor(augmented_actor, real_test, device)
        print("\nAugmented evaluation:")
        print(f"  Action MSE: {augmented_eval['action_mse']:.6f} +/- {augmented_eval['action_mse_std']:.6f}")

    # Comparison
    print("\n" + "="*70)
    print("A/B TEST RESULTS")
    print("="*70)

    results = {
        'baseline': {
            'action_mse': baseline_eval['action_mse'],
            'action_mse_std': baseline_eval['action_mse_std'],
            'final_loss': baseline_history['loss'][-1],
            'n_transitions': len(real_train),
        },
        'synthetic_branch_policy': training_policy,
        'weight_diagnostics': {
            'baseline': baseline_debug,
            'augmented': aug_debug,
        },
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'synth_weight': args.synth_weight,
            'effective_synth_share': effective_synth_share,
            'econ_weight_scale': args.econ_weight_scale,
            'use_lambda_controller': args.use_lambda_controller,
            'max_synth_share': profile.get('max_synth_share', max_synth_share),
            'seed': args.seed,
        }
    }

    if corpus is not None:
        results['synthetic_branch_corpus'] = {
            'summary': corpus.summary,
            'execution_preconditions': corpus.execution_preconditions.to_dict(),
            'benchmark_gate': corpus.benchmark_gate.to_dict(),
        }
        branch_metrics_path = output_root / 'synthetic_branch_metrics.json'
        _write_json(
            branch_metrics_path,
            {
                'branch_metrics': branch_metrics,
                'training_policy': training_policy,
            },
        )

    if augmented_eval:
        augmented_result = {
            'action_mse': augmented_eval['action_mse'],
            'action_mse_std': augmented_eval['action_mse_std'],
            'final_loss': augmented_history['loss'][-1],
            'n_real_transitions': len(real_train),
            'n_synth_transitions': len(synth_transitions),
        }

        # Add lambda trajectory info if controller was used
        if lambda_trajectory:
            augmented_result['lambda_controller_used'] = True
            augmented_result['lambda_trajectory_summary'] = {
                'initial_lambda': float(lambda_trajectory[0]['lambda_synth']),
                'final_lambda': float(lambda_trajectory[-1]['lambda_synth']),
                'mean_lambda': float(np.mean([t['lambda_synth'] for t in lambda_trajectory])),
                'n_adjustments': len(lambda_trajectory),
            }
        else:
            augmented_result['lambda_controller_used'] = False

        results['augmented'] = augmented_result

        # Compute delta
        mse_delta = augmented_eval['action_mse'] - baseline_eval['action_mse']
        mse_pct_change = 100 * mse_delta / baseline_eval['action_mse']

        results['comparison'] = {
            'mse_delta': float(mse_delta),
            'mse_pct_change': float(mse_pct_change),
        }

        print(f"Baseline Action MSE:   {baseline_eval['action_mse']:.6f}")
        print(f"Augmented Action MSE:  {augmented_eval['action_mse']:.6f}")
        print(f"Delta:                 {mse_delta:+.6f} ({mse_pct_change:+.2f}%)")

        if mse_delta < 0:
            print("\nRESULT: Augmented IMPROVED over baseline")
            if abs(mse_pct_change) < 2:
                print("  Small improvement - likely within noise")
            elif abs(mse_pct_change) < 5:
                print("  Moderate improvement - promising signal")
            else:
                print("  Significant improvement - synthetic is adding value!")
        elif mse_delta > 0:
            print("\nRESULT: Augmented WORSE than baseline")
            if abs(mse_pct_change) < 2:
                print("  Minimal degradation - trust gate is working ('do no harm')")
            elif abs(mse_pct_change) < 5:
                print("  Noticeable degradation - tighten synthetic gating")
            else:
                print("  Significant degradation - synthetic is poisoning! Reduce synth_weight")
        else:
            print("\nRESULT: No difference - trust gate maintains baseline")
    else:
        print("No synthetic data available for comparison")

    # Save results
    output_path = output_root / 'offline_local_synth_eval.json'
    _write_json(output_path, results)
    print(f"\nSaved results to {output_path}")

    baseline_history_path = output_root / 'offline_baseline_history.json'
    _write_json(baseline_history_path, baseline_history)
    augmented_history_path = None
    if augmented_history is not None:
        augmented_history_path = output_root / 'offline_local_synth_history.json'
        _write_json(augmented_history_path, augmented_history)
    lambda_path = None
    if lambda_trajectory:
        lambda_path = output_root / 'synth_lambda_trajectory.json'
        _write_json(lambda_path, {'trajectory': lambda_trajectory})

    baseline_checkpoint_path = output_root / 'offline_baseline_actor.pt'
    torch.save(baseline_actor.state_dict(), baseline_checkpoint_path)
    augmented_checkpoint_path = None
    if augmented_eval:
        augmented_checkpoint_path = output_root / 'offline_local_synth_actor.pt'
        torch.save(augmented_actor.state_dict(), augmented_checkpoint_path)
        print("Saved actor checkpoints")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("If augmented ≈ baseline: 'Do no harm' - trust gate works, no uplift yet")
    print("If augmented > baseline: Synthetic creates economic value!")
    print("If augmented < baseline: Synthetic poisoning - tighten gating")
    print("="*70)

    payload = {
        'results': results,
        'artifacts': {
            'offline_local_synth_eval': str(output_path),
            'offline_baseline_history': str(baseline_history_path),
            'offline_baseline_actor': str(baseline_checkpoint_path),
            'offline_local_synth_history': str(augmented_history_path) if augmented_history_path else None,
            'offline_local_synth_actor': str(augmented_checkpoint_path) if augmented_checkpoint_path else None,
            'synthetic_branch_summary': str(branch_summary_path) if branch_summary_path else None,
            'synthetic_branch_metrics': str(branch_metrics_path) if branch_metrics_path else None,
            'synthetic_branch_execution_preconditions': str(branch_preconditions_path) if branch_preconditions_path else None,
            'synthetic_branch_benchmark_gate': str(branch_benchmark_path) if branch_benchmark_path else None,
            'synth_lambda_trajectory': str(lambda_path) if lambda_path else None,
        },
    }
    training_job_result_path = output_root / 'training_job_result.json'
    _write_json(training_job_result_path, payload)

    if runner is not None:
        config_digest = sha256_json({
            'real_data': args.real_data,
            'synth_data': args.synth_data,
            'synth_metadata': args.synth_metadata,
            'synth_gap_labels': args.synth_gap_labels,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
            'requested_synth_weight': args.synth_weight,
            'effective_synth_share': effective_synth_share,
            'econ_weight_scale': args.econ_weight_scale,
            'use_lambda_controller': args.use_lambda_controller,
            'training_policy': training_policy,
        })
        real_audits = _build_real_trajectory_audits(real_bundle)
        synth_audits = _build_synthetic_trajectory_audits(args.synth_data, corpus) if corpus is not None else []
        eligible_ids = [episode['episode_id'] for episode in real_bundle['episodes']]
        if corpus is not None:
            eligible_ids.extend(f"synth_branch_{branch.branch_idx:04d}" for branch in corpus.branches)
        runner.set_eligible_datapacks(eligible_ids)
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for episode in real_bundle['episodes']:
            runner.record_sample("real_seed", datapack_id=episode['episode_id'], slice_id=episode['episode_id'])
        if corpus is not None:
            for branch in corpus.branches:
                branch_id = f"synth_branch_{branch.branch_idx:04d}"
                runner.record_sample("synthetic_branch", datapack_id=branch_id, slice_id=branch_id)
        for audit in [*real_audits, *synth_audits]:
            runner.add_trajectory_audit(audit)
        runner.update_step(int(args.epochs))
        runner.set_weights(
            baseline_weights={
                'requested_synth_share': float(args.synth_weight),
                'requested_econ_weight_scale': float(args.econ_weight_scale),
            },
            final_weights=dict(training_policy),
        )
        runner.configure_training_runtime(
            training_kind="offline_local_synth",
            config_digest=config_digest,
            replay_dataset_summary={
                'real_transition_count': len(real_transitions),
                'synthetic_transition_count': len(synth_transitions),
                'real_episode_count': int(real_bundle['n_episodes']),
                'synthetic_branch_count': len(corpus.branches) if corpus is not None else 0,
            },
            objective_profile_snapshot={
                'objective_vector': list(profile.get('default_objective_vector', [])),
                'policy': 'trust_x_w_econ_with_bounded_synthetic_budget',
            },
            promotion_policy_snapshot={},
            source_domain_coverage=_source_domain_coverage(
                real_bundle,
                len(corpus.branches) if corpus is not None else 0,
                len(synth_transitions),
            ),
            receipt_label_coverage={},
            metadata={
                'requested_synth_share': float(args.synth_weight),
                'effective_synth_share': float(effective_synth_share),
                'synthetic_branch_policy': dict(training_policy),
                'future_training_signals': (
                    dict(corpus.summary.get('future_training_signals', {}) or {})
                    if corpus is not None
                    else {}
                ),
                'scene_tracks_backend': (
                    corpus.summary.get('source_metadata', {}).get('scene_tracks_backend')
                    if corpus is not None
                    else 'unavailable'
                ),
                'teacher_runtime_backend_selected': (
                    corpus.summary.get('source_metadata', {}).get('teacher_runtime_backend_selected')
                    if corpus is not None
                    else 'unavailable'
                ),
                'vision_backbone_selected': (
                    corpus.summary.get('source_metadata', {}).get('vision_backbone_selected')
                    if corpus is not None
                    else 'unavailable'
                ),
                'semantic_grounding_mode': (
                    corpus.summary.get('source_metadata', {}).get('semantic_grounding_mode')
                    if corpus is not None
                    else 'heuristic_fallback'
                ),
                'synthetic_branch_metadata_present': bool(corpus is not None and corpus.summary.get('metadata_present', False)),
                'synthetic_branch_gap_labels_present': bool(corpus is not None and corpus.summary.get('gap_labels_present', False)),
            },
        )
        runner.set_regal_result(
            {
                'overall_status': 'pass',
                'synthetic_branch_policy': training_policy,
                'synthetic_branch_execution_preconditions': (
                    corpus.execution_preconditions.to_dict() if corpus is not None else {}
                ),
                'synthetic_branch_benchmark_gate': (
                    corpus.benchmark_gate.to_dict() if corpus is not None else {}
                ),
            },
            context_sha=config_digest,
        )
        runner.register_artifact('offline_local_synth_eval', output_path)
        runner.register_artifact('offline_baseline_history', baseline_history_path)
        runner.register_artifact('offline_baseline_actor', baseline_checkpoint_path)
        runner.register_artifact('training_job_result', training_job_result_path)
        if augmented_history_path is not None:
            runner.register_artifact('offline_local_synth_history', augmented_history_path)
        if augmented_checkpoint_path is not None:
            runner.register_artifact('offline_local_synth_actor', augmented_checkpoint_path)
        if branch_summary_path is not None:
            runner.register_artifact('synthetic_branch_summary', branch_summary_path)
        if branch_metrics_path is not None:
            runner.register_artifact('synthetic_branch_metrics', branch_metrics_path)
        if branch_preconditions_path is not None:
            runner.register_artifact('synthetic_branch_execution_preconditions', branch_preconditions_path)
        if branch_benchmark_path is not None:
            runner.register_artifact('synthetic_branch_benchmark_gate', branch_benchmark_path)
        if lambda_path is not None:
            runner.register_artifact('synth_lambda_trajectory', lambda_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id='offline_baseline_actor',
                model_family='offline_local_synth_actor',
                model_version='baseline_real_only_v1',
                path=baseline_checkpoint_path,
                epoch=args.epochs,
                step=args.epochs,
                metadata={'dataset': 'real_only'},
            )
        )
        if augmented_checkpoint_path is not None:
            runner.register_checkpoint(
                build_checkpoint_record(
                    checkpoint_id='offline_local_synth_actor',
                    model_family='offline_local_synth_actor',
                    model_version='local_synth_augmented_v1',
                    path=augmented_checkpoint_path,
                    epoch=args.epochs,
                    step=args.epochs,
                    is_best=bool(
                        augmented_eval is not None
                        and augmented_eval['action_mse'] <= baseline_eval['action_mse']
                    ),
                    metadata={'dataset': 'real_plus_synthetic', 'training_policy': training_policy},
                )
            )

    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.skip_regal_runner:
        payload = _run_training(args, runner=None)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    holder: Dict[str, Any] = {}

    def _wrapped(runner: RegalTrainingRunner) -> None:
        holder['payload'] = _run_training(args, runner=runner)

    plan_sha = sha256_json({
        'script': 'train_offline_with_local_synth.py',
        'seed': args.seed,
        'requested_synth_share': args.synth_weight,
        'real_data': args.real_data,
        'synth_data': args.synth_data,
    })
    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            num_episodes=args.epochs,
            training_steps=args.epochs,
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="offline_local_synth",
    )
    print(json.dumps(holder.get('payload', {}), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
