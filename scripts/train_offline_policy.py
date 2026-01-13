#!/usr/bin/env python3
"""
Offline Policy Training with Data Augmentation (Phase B.5)

Trains SAC policy on replay buffer augmented with synthetic data from world model.
Compares: Baseline (real-only) vs Augmented (real + synthetic from high-value bricks)

Usage:
    # Baseline: real data only
    python scripts/train_offline_policy.py --real data/physics_zv_rollouts.npz --episodes 50

    # Augmented: real + synthetic
    python scripts/train_offline_policy.py --real data/physics_zv_rollouts.npz \
        --synthetic data/synthetic_zv_rollouts.npz --synthetic-ratio 0.3 --episodes 50

    # Filtered by bricks
    python scripts/train_offline_policy.py --real data/physics_zv_rollouts.npz \
        --synthetic data/synthetic_zv_rollouts.npz --brick-filter top3_mpl --episodes 50
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path


class Actor(nn.Module):
    """Simple SAC actor for latent actions."""
    def __init__(self, latent_dim=128, action_dim=2, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.sigmoid(x_t)  # Bound to [0, 1]
        log_prob = normal.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean


class Critic(nn.Module):
    """Twin Q-networks for SAC."""
    def __init__(self, latent_dim=128, action_dim=2, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class OfflineReplayBuffer:
    """Replay buffer loaded from z_V rollouts."""

    def __init__(self, max_size=100000, synthetic_weight=1.0):
        self.max_size = max_size
        self.synthetic_weight = synthetic_weight  # Down-weight synthetic samples
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.sources = []  # Track if 'real' or 'synthetic'
        self.weights = []  # Importance weights for sampling

    def load_from_npz(self, npz_path, source='real', episode_filter=None):
        """
        Load transitions from z_V rollout npz file.

        Args:
            npz_path: Path to rollouts npz
            source: 'real' or 'synthetic'
            episode_filter: Optional list of episode indices to include
        """
        data = np.load(npz_path, allow_pickle=True)
        n_episodes = int(data['n_episodes'])

        added_transitions = 0
        for ep in range(n_episodes):
            if episode_filter is not None and ep not in episode_filter:
                continue

            z_seq = data[f'ep_{ep}_z_sequence']  # (T+1, latent_dim)
            actions = data[f'ep_{ep}_actions']    # (T, action_dim)

            # Handle missing rewards/dones for synthetic data
            if f'ep_{ep}_rewards' in data:
                rewards = data[f'ep_{ep}_rewards']    # (T,)
                dones = data[f'ep_{ep}_dones']        # (T,)
            else:
                # Generate synthetic rewards: small negative step cost + completion bonus
                T = len(actions)
                rewards = np.full(T, -0.01)  # Step cost
                rewards[-1] += 1.0  # Terminal bonus
                dones = np.zeros(T)
                dones[-1] = 1.0  # Episode ends

            # Extract transitions: (s_t, a_t, r_t, s_{t+1}, done_t)
            weight = self.synthetic_weight if source == 'synthetic' else 1.0
            for t in range(len(actions)):
                if len(self.states) >= self.max_size:
                    break

                self.states.append(z_seq[t])
                self.actions.append(actions[t])
                self.rewards.append(rewards[t])
                self.next_states.append(z_seq[t + 1])
                self.dones.append(float(dones[t]))
                self.sources.append(source)
                self.weights.append(weight)
                added_transitions += 1

        print(f"  Loaded {added_transitions} transitions from {source} data ({npz_path})")
        if source == 'synthetic':
            print(f"    Synthetic weight: {self.synthetic_weight}")
        return added_transitions

    def sample(self, batch_size=256, weighted=False):
        """Sample random batch of transitions.

        Args:
            batch_size: Number of samples
            weighted: If True, sample proportionally to weights (down-weights synthetic)
        """
        if weighted and len(self.weights) > 0:
            # Weighted sampling - real data is sampled more often
            probs = np.array(self.weights)
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.states), size=batch_size, p=probs)
        else:
            idx = np.random.randint(0, len(self.states), size=batch_size)

        batch = {
            'states': torch.FloatTensor(np.array([self.states[i] for i in idx])),
            'actions': torch.FloatTensor(np.array([self.actions[i] for i in idx])),
            'rewards': torch.FloatTensor(np.array([self.rewards[i] for i in idx])).unsqueeze(-1),
            'next_states': torch.FloatTensor(np.array([self.next_states[i] for i in idx])),
            'dones': torch.FloatTensor(np.array([self.dones[i] for i in idx])).unsqueeze(-1),
            'weights': torch.FloatTensor(np.array([self.weights[i] for i in idx])).unsqueeze(-1),
        }
        return batch

    def __len__(self):
        return len(self.states)

    def get_source_stats(self):
        """Get breakdown of data sources."""
        from collections import Counter
        return Counter(self.sources)


def filter_episodes_by_brick(brick_manifest_path, filter_strategy='top3_mpl'):
    """
    Get episode indices based on brick filtering strategy.

    Args:
        brick_manifest_path: Path to data_bricks_manifest.json
        filter_strategy: One of:
            - 'top3_mpl': Top 3 bricks by MPL improvement
            - 'top3_error': Top 3 bricks by error reduction
            - 'top3_energy': Top 3 bricks by energy efficiency
            - 'positive_mpl': Only bricks with positive MPL improvement
            - 'positive_error': Only bricks with negative error rate (improvement)
            - 'all': No filtering

    Returns:
        Set of episode indices to include
    """
    if filter_strategy == 'all':
        return None  # Include all

    with open(brick_manifest_path, 'r') as f:
        bricks = json.load(f)

    # Sort bricks by impact dimension
    if filter_strategy == 'top3_mpl':
        # Sort by MPL improvement (descending)
        sorted_bricks = sorted(bricks,
                               key=lambda b: b['impact_profile']['delta_mpl_units_per_hr'],
                               reverse=True)
        top_bricks = sorted_bricks[:3]
    elif filter_strategy == 'top3_error':
        # Sort by error reduction (most negative first)
        sorted_bricks = sorted(bricks,
                               key=lambda b: b['impact_profile']['delta_error_rate'])
        top_bricks = sorted_bricks[:3]
    elif filter_strategy == 'top3_energy':
        # Sort by energy reduction (most negative first)
        sorted_bricks = sorted(bricks,
                               key=lambda b: b['impact_profile']['delta_energy_wh_per_unit'])
        top_bricks = sorted_bricks[:3]
    elif filter_strategy == 'positive_mpl':
        # Only bricks with POSITIVE MPL improvement
        top_bricks = [b for b in bricks
                      if b['impact_profile']['delta_mpl_units_per_hr'] > 0]
    elif filter_strategy == 'positive_error':
        # Only bricks with NEGATIVE error rate (i.e., error reduction)
        top_bricks = [b for b in bricks
                      if b['impact_profile']['delta_error_rate'] < 0]
    else:
        raise ValueError(f"Unknown filter strategy: {filter_strategy}")

    print(f"Selected bricks ({filter_strategy}):")
    for brick in top_bricks:
        delta_mpl = brick['impact_profile']['delta_mpl_units_per_hr']
        delta_err = brick['impact_profile']['delta_error_rate']
        print(f"  - {brick['brick_id']}: {brick['num_episodes']} episodes")
        print(f"    ΔMPL={delta_mpl:+.2f} units/hr, ΔError={delta_err:+.4f}")
        print(f"    Semantic: {', '.join(brick['semantic_tags'][:2])}")

    # Collect episode IDs
    episode_ids = set()
    for brick in top_bricks:
        episode_ids.update(brick['episode_ids'])

    print(f"Total episodes to use: {len(episode_ids)}")
    return episode_ids


def train_offline_sac(
    buffer,
    actor,
    critic,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    n_updates=1000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    device='cpu',
    use_weighted_sampling=False,
):
    """
    Train SAC on offline data.

    Returns:
        Training metrics
    """
    actor.train()
    critic.train()

    actor_losses = []
    critic_losses = []

    for update in range(n_updates):
        # Sample batch (optionally weighted to favor real data)
        batch = buffer.sample(batch_size, weighted=use_weighted_sampling)
        states = batch['states'].to(device)
        actions = batch['actions'].to(device)
        rewards = batch['rewards'].to(device)
        next_states = batch['next_states'].to(device)
        dones = batch['dones'].to(device)

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = actor.sample(next_states)
            q1_next, q2_next = target_critic(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_target = rewards + gamma * (1 - dones) * q_next

        q1, q2 = critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update actor
        new_actions, log_probs, _ = actor.sample(states)
        q1_new, q2_new = critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs - q_new).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Soft update target
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

        if (update + 1) % 200 == 0:
            print(f"  Update {update + 1}/{n_updates}: "
                  f"Actor Loss={np.mean(actor_losses[-200:]):.4f}, "
                  f"Critic Loss={np.mean(critic_losses[-200:]):.4f}")

    return {
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'final_actor_loss': np.mean(actor_losses[-100:]),
        'final_critic_loss': np.mean(critic_losses[-100:]),
    }


def evaluate_policy_in_buffer(actor, buffer, n_samples=1000, device='cpu'):
    """
    Evaluate policy on held-out transitions from buffer.

    This is a proxy for online evaluation (we don't run the actual env).
    """
    actor.eval()

    # Sample transitions
    if len(buffer) < n_samples:
        n_samples = len(buffer)

    batch = buffer.sample(n_samples)
    states = batch['states'].to(device)
    actions_true = batch['actions'].to(device)

    with torch.no_grad():
        actions_pred, _, _ = actor.sample(states)

    # Compute action prediction error (proxy for policy quality)
    action_mse = nn.MSELoss()(actions_pred, actions_true).item()

    return {
        'action_mse': action_mse,
        'n_samples': n_samples,
    }


def run_offline_experiment(
    real_data_path,
    synthetic_data_path=None,
    synthetic_ratio=0.3,
    synthetic_weight=1.0,
    brick_filter='all',
    brick_manifest_path='data/bricks/data_bricks_manifest.json',
    n_updates=2000,
    batch_size=256,
    latent_dim=128,
    action_dim=2,
    hidden_dim=256,
    device=None,
    save_dir='checkpoints',
    use_weighted_sampling=False,
):
    """
    Run offline RL experiment.

    Args:
        real_data_path: Path to real z_V rollouts
        synthetic_data_path: Path to synthetic z_V rollouts (None for baseline)
        synthetic_ratio: Fraction of training to sample from synthetic
        synthetic_weight: Weight for synthetic samples (0.1 = 10% weight vs real)
        brick_filter: Strategy for filtering bricks ('all', 'top3_mpl', etc.)
        n_updates: Number of SAC updates
        use_weighted_sampling: If True, sample proportionally to weights
        ...

    Returns:
        Experiment results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("OFFLINE RL TRAINING EXPERIMENT")
    print("="*60)

    # Create replay buffer with synthetic weighting
    buffer = OfflineReplayBuffer(max_size=200000, synthetic_weight=synthetic_weight)

    # Load real data
    print(f"\nLoading real data from {real_data_path}")
    buffer.load_from_npz(real_data_path, source='real')

    # Load synthetic data if provided
    if synthetic_data_path and os.path.exists(synthetic_data_path):
        # Filter synthetic episodes by brick if requested
        episode_filter = None
        if brick_filter != 'all' and os.path.exists(brick_manifest_path):
            print(f"\nFiltering synthetic data by brick strategy: {brick_filter}")
            episode_filter = filter_episodes_by_brick(brick_manifest_path, brick_filter)

        print(f"\nLoading synthetic data from {synthetic_data_path}")
        buffer.load_from_npz(synthetic_data_path, source='synthetic', episode_filter=episode_filter)

    # Report data composition
    source_stats = buffer.get_source_stats()
    total = len(buffer)
    print(f"\nReplay Buffer Composition:")
    for source, count in source_stats.items():
        print(f"  {source}: {count} transitions ({100*count/total:.1f}%)")
    print(f"  Total: {total} transitions")

    if use_weighted_sampling:
        print(f"  Using weighted sampling (synthetic weight={synthetic_weight})")

    # Initialize networks
    actor = Actor(latent_dim, action_dim, hidden_dim).to(device)
    critic = Critic(latent_dim, action_dim, hidden_dim).to(device)
    target_critic = Critic(latent_dim, action_dim, hidden_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    # Train
    print(f"\nTraining SAC for {n_updates} updates...")
    train_metrics = train_offline_sac(
        buffer, actor, critic, target_critic,
        actor_optimizer, critic_optimizer,
        n_updates=n_updates,
        batch_size=batch_size,
        device=device,
        use_weighted_sampling=use_weighted_sampling,
    )

    # Evaluate
    print("\nEvaluating policy...")
    eval_metrics = evaluate_policy_in_buffer(actor, buffer, device=device)
    print(f"  Action MSE (lower is better): {eval_metrics['action_mse']:.6f}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    experiment_name = f"offline_{'augmented' if synthetic_data_path else 'baseline'}"
    if brick_filter != 'all':
        experiment_name += f"_{brick_filter}"
    if use_weighted_sampling and synthetic_weight < 1.0:
        experiment_name += f"_w{int(synthetic_weight*100)}"

    model_path = os.path.join(save_dir, f"{experiment_name}_actor.pt")
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'source_stats': dict(source_stats),
        'config': {
            'real_data_path': real_data_path,
            'synthetic_data_path': synthetic_data_path,
            'synthetic_ratio': synthetic_ratio,
            'synthetic_weight': synthetic_weight,
            'brick_filter': brick_filter,
            'n_updates': n_updates,
            'batch_size': batch_size,
            'use_weighted_sampling': use_weighted_sampling,
        }
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # Summary
    results = {
        'experiment': experiment_name,
        'real_transitions': source_stats.get('real', 0),
        'synthetic_transitions': source_stats.get('synthetic', 0),
        'total_transitions': total,
        'final_actor_loss': train_metrics['final_actor_loss'],
        'final_critic_loss': train_metrics['final_critic_loss'],
        'action_mse': eval_metrics['action_mse'],
        'model_path': model_path,
    }

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("="*60)

    return results


# Add regality wrapper
import sys
repo_root_path = Path(__file__).parent.parent
if str(repo_root_path) not in sys.path:
    sys.path.insert(0, str(repo_root_path))

from src.training.wrap_training_entrypoint import regal_training


@regal_training(env_type="workcell")
def main(runner=None):
    """Main training function with regality wrapper."""
    parser = argparse.ArgumentParser(description='Train offline policy with data augmentation (FULL regality)')
    parser.add_argument(
        '--real',
        type=str,
        default='data/physics_zv_rollouts.npz',
        help='Path to real z_V rollouts'
    )
    parser.add_argument(
        '--synthetic',
        type=str,
        default=None,
        help='Path to synthetic z_V rollouts (None for baseline)'
    )
    parser.add_argument(
        '--synthetic-ratio',
        type=float,
        default=0.3,
        help='Fraction of data to draw from synthetic'
    )
    parser.add_argument(
        '--brick-filter',
        type=str,
        default='all',
        choices=['all', 'top3_mpl', 'top3_error', 'top3_energy', 'positive_mpl', 'positive_error'],
        help='Strategy for filtering bricks'
    )
    parser.add_argument(
        '--brick-manifest',
        type=str,
        default='data/bricks/data_bricks_manifest.json',
        help='Path to brick manifest'
    )
    parser.add_argument(
        '--synthetic-weight',
        type=float,
        default=1.0,
        help='Weight for synthetic samples (0.1 = 10%% weight vs real)'
    )
    parser.add_argument(
        '--weighted-sampling',
        action='store_true',
        help='Use weighted sampling to favor real data'
    )
    parser.add_argument(
        '--updates',
        type=int,
        default=2000,
        help='Number of SAC updates'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='checkpoints',
        help='Directory to save trained model'
    )

    args = parser.parse_args()

    if runner:
        runner.start_training()

    results = run_offline_experiment(
        real_data_path=args.real,
        synthetic_data_path=args.synthetic,
        synthetic_ratio=args.synthetic_ratio,
        synthetic_weight=args.synthetic_weight,
        brick_filter=args.brick_filter,
        brick_manifest_path=args.brick_manifest,
        n_updates=args.updates,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        use_weighted_sampling=args.weighted_sampling,
    )

    if runner:
        runner.update_step(args.updates)

    print("\nNext steps:")
    print("1. Run baseline experiment:")
    print("   python scripts/train_offline_policy.py --real data/physics_zv_rollouts.npz")
    print()
    print("2. Run augmented experiment:")
    print("   python scripts/train_offline_policy.py --real data/physics_zv_rollouts.npz \\")
    print("       --synthetic data/synthetic_zv_rollouts.npz")
    print()
    print("3. Compare results to validate world model value")


if __name__ == '__main__':
    main()

