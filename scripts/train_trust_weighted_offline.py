#!/usr/bin/env python3
"""
Trust-Weighted Offline RL Training (Phase B Priority 1)

Trains offline SAC using trust scores as importance weights.
Since synthetic trust ≈ 0.0006, synthetic data is effectively ignored.

Expected result: Performance ≈ baseline (proves "do no harm" gate works)

Usage:
    python scripts/train_trust_weighted_offline.py
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from scripts.train_offline_policy import Actor, Critic, OfflineReplayBuffer


class TrustWeightedReplayBuffer(OfflineReplayBuffer):
    """Replay buffer that uses trust scores as per-transition weights."""

    def load_from_trust_augmented_npz(self, npz_path, source='real'):
        """
        Load transitions from trust-augmented z_V rollout npz file.
        Uses trust_scores field for per-episode weighting.

        Args:
            npz_path: Path to trust-augmented rollouts (contains trust_scores)
            source: 'real' or 'synthetic'
        """
        data = np.load(npz_path, allow_pickle=True)
        n_episodes = int(data['n_episodes'])

        # Get trust scores (per-episode)
        if 'trust_scores' in data:
            trust_scores = data['trust_scores']
            print(f"  Found trust_scores for {source}: mean={trust_scores.mean():.6f}, "
                  f"std={trust_scores.std():.6f}")
        else:
            # No trust scores - use 1.0 for real, 0.0 for synthetic
            print(f"  WARNING: No trust_scores in {npz_path}, using defaults")
            trust_scores = np.ones(n_episodes) if source == 'real' else np.zeros(n_episodes)

        added_transitions = 0
        for ep in range(n_episodes):
            z_seq = data[f'ep_{ep}_z_sequence']  # (T+1, latent_dim)
            actions = data[f'ep_{ep}_actions']    # (T, action_dim)

            # Trust score for this episode
            ep_trust = float(trust_scores[ep])

            # Handle missing rewards/dones for synthetic data
            if f'ep_{ep}_rewards' in data:
                rewards = data[f'ep_{ep}_rewards']
                dones = data[f'ep_{ep}_dones']
            else:
                # Generate synthetic rewards
                T = len(actions)
                rewards = np.full(T, -0.01)  # Step cost
                rewards[-1] += 1.0  # Terminal bonus
                dones = np.zeros(T)
                dones[-1] = 1.0

            # Add transitions with trust-based weights
            for t in range(len(actions)):
                if len(self.states) >= self.max_size:
                    break

                self.states.append(z_seq[t])
                self.actions.append(actions[t])
                self.rewards.append(rewards[t])
                self.next_states.append(z_seq[t + 1])
                self.dones.append(float(dones[t]))
                self.sources.append(source)
                self.weights.append(ep_trust)  # Use actual trust score!
                added_transitions += 1

        print(f"  Loaded {added_transitions} transitions from {source} data")
        print(f"    Weight range: [{min(self.weights[-added_transitions:]):.6f}, "
              f"{max(self.weights[-added_transitions:]):.6f}]")
        return added_transitions


def train_trust_weighted_sac(
    buffer,
    actor,
    critic,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    n_updates=2000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    device='cpu',
    use_weighted_loss=True,
):
    """
    Train SAC with trust-weighted loss.

    Key difference from baseline:
    - Loss is weighted by trust scores
    - Synthetic with trust ≈ 0 contributes ~0 to gradient
    """
    actor.train()
    critic.train()

    actor_losses = []
    critic_losses = []
    effective_batch_sizes = []

    for update in range(n_updates):
        # Sample batch (weighted sampling to favor high-trust data)
        batch = buffer.sample(batch_size, weighted=True)
        states = batch['states'].to(device)
        actions = batch['actions'].to(device)
        rewards = batch['rewards'].to(device)
        next_states = batch['next_states'].to(device)
        dones = batch['dones'].to(device)
        weights = batch['weights'].to(device)

        # Track effective batch size (sum of weights)
        effective_batch_size = weights.sum().item()
        effective_batch_sizes.append(effective_batch_size)

        # Update critic with weighted loss
        with torch.no_grad():
            next_actions, next_log_probs, _ = actor.sample(next_states)
            q1_next, q2_next = target_critic(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_target = rewards + gamma * (1 - dones) * q_next

        q1, q2 = critic(states, actions)

        if use_weighted_loss:
            # Weighted MSE loss: weight each sample by trust score
            critic_loss = (weights * (q1 - q_target)**2).mean() + \
                          (weights * (q2 - q_target)**2).mean()
        else:
            critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update actor with weighted loss
        new_actions, log_probs, _ = actor.sample(states)
        q1_new, q2_new = critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        if use_weighted_loss:
            # Weighted actor loss
            actor_loss = (weights * (alpha * log_probs - q_new)).mean()
        else:
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
            avg_eff_batch = np.mean(effective_batch_sizes[-200:])
            print(f"  Update {update + 1}/{n_updates}: "
                  f"Actor={np.mean(actor_losses[-200:]):.4f}, "
                  f"Critic={np.mean(critic_losses[-200:]):.4f}, "
                  f"EffBatch={avg_eff_batch:.1f}/{batch_size}")

    return {
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'final_actor_loss': np.mean(actor_losses[-100:]),
        'final_critic_loss': np.mean(critic_losses[-100:]),
        'avg_effective_batch_size': np.mean(effective_batch_sizes),
    }


def main():
    parser = argparse.ArgumentParser(description='Trust-weighted offline RL')
    parser.add_argument('--real', type=str, default='data/physics_zv_rollouts_trust.npz',
                        help='Path to trust-augmented real rollouts')
    parser.add_argument('--synthetic', type=str, default='data/synthetic_zv_rollouts_trust.npz',
                        help='Path to trust-augmented synthetic rollouts')
    parser.add_argument('--updates', type=int, default=2000,
                        help='Number of SAC updates')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--no-weighted-loss', action='store_true',
                        help='Disable weighted loss (use weighted sampling only)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("TRUST-WEIGHTED OFFLINE RL TRAINING")
    print("="*60)
    print("Goal: Confirm trust-weighted policy ≈ baseline")
    print("Expected: Synthetic data (trust≈0) effectively ignored")
    print("="*60)

    # Load trust-augmented data
    buffer = TrustWeightedReplayBuffer(max_size=200000)

    print(f"\nLoading trust-augmented real data...")
    buffer.load_from_trust_augmented_npz(args.real, source='real')

    if args.synthetic and os.path.exists(args.synthetic):
        print(f"\nLoading trust-augmented synthetic data...")
        buffer.load_from_trust_augmented_npz(args.synthetic, source='synthetic')

    # Report composition
    source_stats = buffer.get_source_stats()
    total = len(buffer)
    print(f"\nReplay Buffer Composition:")
    for source, count in source_stats.items():
        print(f"  {source}: {count} transitions ({100*count/total:.1f}%)")
    print(f"  Total: {total} transitions")

    # Analyze weight distribution
    weights = np.array(buffer.weights)
    print(f"\nWeight Distribution:")
    print(f"  Mean: {weights.mean():.6f}")
    print(f"  Std:  {weights.std():.6f}")
    print(f"  Min:  {weights.min():.6f}")
    print(f"  Max:  {weights.max():.6f}")

    # Effective data contribution
    real_weights = [w for w, s in zip(buffer.weights, buffer.sources) if s == 'real']
    syn_weights = [w for w, s in zip(buffer.weights, buffer.sources) if s == 'synthetic']

    total_weight = sum(buffer.weights)
    real_contribution = sum(real_weights) / total_weight * 100
    syn_contribution = sum(syn_weights) / total_weight * 100

    print(f"\nEffective Contribution:")
    print(f"  Real: {real_contribution:.2f}%")
    print(f"  Synthetic: {syn_contribution:.2f}%")
    print(f"  (Synthetic should be ~0% if trust≈0)")

    # Initialize networks
    latent_dim = 128
    action_dim = 2
    hidden_dim = 256

    actor = Actor(latent_dim, action_dim, hidden_dim).to(device)
    critic = Critic(latent_dim, action_dim, hidden_dim).to(device)
    target_critic = Critic(latent_dim, action_dim, hidden_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    # Train
    print(f"\nTraining SAC with trust-weighted loss for {args.updates} updates...")
    train_metrics = train_trust_weighted_sac(
        buffer, actor, critic, target_critic,
        actor_optimizer, critic_optimizer,
        n_updates=args.updates,
        batch_size=args.batch_size,
        device=device,
        use_weighted_loss=not args.no_weighted_loss,
    )

    # Evaluate on held-out data (proxy evaluation)
    print("\nEvaluating policy on buffer data...")
    actor.eval()
    n_eval = min(1000, len(buffer))
    batch = buffer.sample(n_eval)
    states = batch['states'].to(device)
    actions_true = batch['actions'].to(device)

    with torch.no_grad():
        actions_pred, _, _ = actor.sample(states)

    action_mse = nn.MSELoss()(actions_pred, actions_true).item()
    print(f"  Action MSE: {action_mse:.6f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    model_path = 'checkpoints/offline_trust_weighted_actor.pt'
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'train_metrics': train_metrics,
        'eval_metrics': {'action_mse': action_mse},
        'source_stats': dict(source_stats),
        'weight_stats': {
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'real_contribution_pct': real_contribution,
            'syn_contribution_pct': syn_contribution,
        },
        'config': {
            'real_data': args.real,
            'synthetic_data': args.synthetic,
            'n_updates': args.updates,
            'use_weighted_loss': not args.no_weighted_loss,
        }
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # Summary
    print("\n" + "="*60)
    print("TRUST-WEIGHTED TRAINING SUMMARY")
    print("="*60)
    print(f"Real transitions: {source_stats.get('real', 0)}")
    print(f"Synthetic transitions: {source_stats.get('synthetic', 0)}")
    print(f"Real contribution: {real_contribution:.2f}%")
    print(f"Synthetic contribution: {syn_contribution:.2f}%")
    print(f"Final actor loss: {train_metrics['final_actor_loss']:.6f}")
    print(f"Final critic loss: {train_metrics['final_critic_loss']:.6f}")
    print(f"Avg effective batch: {train_metrics['avg_effective_batch_size']:.1f}/{args.batch_size}")
    print(f"Action MSE: {action_mse:.6f}")
    print("="*60)

    # Save results for comparison
    results = {
        'experiment': 'trust_weighted',
        'real_transitions': source_stats.get('real', 0),
        'synthetic_transitions': source_stats.get('synthetic', 0),
        'real_contribution_pct': real_contribution,
        'syn_contribution_pct': syn_contribution,
        'final_actor_loss': train_metrics['final_actor_loss'],
        'final_critic_loss': train_metrics['final_critic_loss'],
        'avg_effective_batch_size': train_metrics['avg_effective_batch_size'],
        'action_mse': action_mse,
        'model_path': model_path,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/trust_weighted_training.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to results/trust_weighted_training.json")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Compare trust-weighted vs baseline performance:")
    print(f"   python scripts/eval_offline_policy.py \\")
    print(f"       --model checkpoints/offline_baseline_actor.pt \\")
    print(f"       --compare {model_path}")
    print()
    print("2. If performance ≈ baseline:")
    print("   - Trust gate works correctly")
    print("   - Synthetic data was rightfully ignored")
    print("   - Safe to fix world model and re-test")
    print("="*60)


if __name__ == '__main__':
    main()
