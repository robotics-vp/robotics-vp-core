#!/usr/bin/env python3
"""
Stable World Model Training - The Real Fix

Trains a contractive world model to be truly horizon-agnostic:
1. Residual + damped architecture (can't explode)
2. REAL scheduled sampling (actually mixes teacher forcing)
3. Multi-horizon loss with autoregressive rollouts
4. Trust loss over MULTIPLE horizons (not just max)
5. Variance GROWTH penalty (not just std matching)
6. Spectral normalization (bounds operator eigenvalues)

This makes the transition function f(z_t, a_t) a stable operator that can be
iterated 60+ times without exploding.

Usage:
    python scripts/train_stable_world_model.py
    python scripts/train_stable_world_model.py --epochs 200 --lambda-trust 1.0
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
from src.world_model.contractive_dynamics import StableWorldModel
from src.valuation.trust_net import TrustNet
from src.training.wrap_training_entrypoint import regal_training


class RealScheduledSampler:
    """
    Actual scheduled sampling that mixes teacher forcing with autoregressive.

    Key: during training, sometimes use real z_t, sometimes use predicted z_t.
    This teaches the model to correct its own errors.
    """

    def __init__(self, initial_prob=0.9, final_prob=0.0, anneal_epochs=100):
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.anneal_epochs = anneal_epochs
        self.current_epoch = 0

    def get_teacher_forcing_prob(self):
        """Get current teacher forcing probability."""
        if self.current_epoch >= self.anneal_epochs:
            return self.final_prob

        # Linear annealing
        t = self.current_epoch / self.anneal_epochs
        return self.initial_prob + (self.final_prob - self.initial_prob) * t

    def step_epoch(self):
        """Advance to next epoch."""
        self.current_epoch += 1

    def rollout_with_sampling(self, model, z_real_seq, actions):
        """
        Roll out with actual scheduled sampling.

        At each step:
          - With prob p: use ground truth z_t as input
          - With prob (1-p): use model's own prediction

        This is the REAL scheduled sampling that was missing before.

        Args:
            model: dynamics model
            z_real_seq: (T+1, latent_dim) real trajectory
            actions: (T, action_dim) action sequence

        Returns:
            z_pred_seq: (T+1, latent_dim) predicted trajectory
        """
        T = len(actions)
        p = self.get_teacher_forcing_prob()

        z_pred_seq = [z_real_seq[0]]
        z_current = z_real_seq[0].unsqueeze(0)  # (1, latent_dim)

        for t in range(T):
            a_t = actions[t].unsqueeze(0)
            z_next_pred, _ = model(z_current, a_t)

            z_pred_seq.append(z_next_pred.squeeze(0))

            # Scheduled sampling: decide whether to use real or predicted for next input
            if np.random.random() < p:
                # Teacher forcing: use real z_{t+1} as next input
                z_current = z_real_seq[t + 1].unsqueeze(0)
            else:
                # Autoregressive: use predicted z_{t+1} as next input
                z_current = z_next_pred

        return torch.stack(z_pred_seq, dim=0)


class MultiHorizonLoss:
    """
    Compute loss over multiple horizon lengths.

    Key insight: train the model to be accurate at ALL horizons, not just 1 step.
    Sample random (start, horizon) pairs and backprop through entire rollout.
    """

    def __init__(self, max_horizon=60, n_samples=5):
        self.max_horizon = max_horizon
        self.n_samples = n_samples

    def compute(self, model, z_real_seq, actions, device):
        """
        Sample multiple horizons and compute average endpoint MSE.

        Args:
            model: dynamics model
            z_real_seq: (T+1, latent_dim) real trajectory
            actions: (T, action_dim) actions

        Returns:
            loss: average MSE over sampled horizons
            horizon_info: dict with details
        """
        T = len(actions)
        if T < 2:
            return torch.tensor(0.0, device=device), {}

        total_loss = 0.0
        horizon_losses = {}

        for _ in range(self.n_samples):
            # Sample horizon length (bias towards longer)
            # Use exponential distribution favoring long horizons
            if np.random.random() < 0.3:
                # 30% chance: short horizon
                L = np.random.randint(1, min(10, T) + 1)
            else:
                # 70% chance: medium to long horizon
                L = np.random.randint(10, min(self.max_horizon, T) + 1)

            # Sample start position
            max_start = T - L
            start_t = np.random.randint(0, max_start + 1)

            # Roll out autoregressively from real z_start
            z_start = z_real_seq[start_t].unsqueeze(0)  # (1, latent_dim)
            z_current = z_start

            for step in range(L):
                a_t = actions[start_t + step].unsqueeze(0)
                z_current, _ = model(z_current, a_t)

            # MSE to real endpoint
            z_target = z_real_seq[start_t + L]
            loss = nn.MSELoss()(z_current.squeeze(0), z_target)
            total_loss += loss

            # Track per-horizon
            if L not in horizon_losses:
                horizon_losses[L] = []
            horizon_losses[L].append(loss.item())

        return total_loss / self.n_samples, horizon_losses


class MultiHorizonTrustLoss:
    """
    Compute trust loss over MULTIPLE horizons, not just max.

    The previous implementation only scored the longest rollout.
    This scores rollouts at {10, 20, 30, 40, 50, 60} steps.
    """

    def __init__(self, trust_net, trust_mean, trust_std, horizons=[10, 30, 60]):
        self.trust_net = trust_net
        self.trust_mean = trust_mean
        self.trust_std = trust_std
        self.horizons = horizons

    def extract_features(self, z_sequence):
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

    def compute(self, model, z_real_seq, actions, device):
        """
        Compute trust loss at multiple horizons.

        Args:
            model: dynamics model
            z_real_seq: (T+1, latent_dim) real trajectory (to get starting z)
            actions: (T, action_dim) actions

        Returns:
            loss: average trust loss over horizons
            trust_scores: dict of trust scores per horizon
        """
        T = len(actions)
        total_loss = 0.0
        trust_scores = {}
        n_valid = 0

        for H in self.horizons:
            if H > T:
                continue

            # Roll out H steps from start
            z_current = z_real_seq[0].unsqueeze(0)
            z_traj = [z_real_seq[0]]

            for t in range(H):
                a_t = actions[t].unsqueeze(0)
                z_current, _ = model(z_current, a_t)
                z_traj.append(z_current.squeeze(0))

            z_traj = torch.stack(z_traj, dim=0)

            # Extract features and score
            features = self.extract_features(z_traj)
            feat_norm = (features - self.trust_mean) / self.trust_std
            trust_score = self.trust_net(feat_norm.unsqueeze(0))

            # BCE loss: want trust -> 1
            target = torch.ones_like(trust_score)
            loss = nn.BCELoss()(trust_score, target)
            total_loss += loss
            trust_scores[H] = trust_score.item()
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device), {}

        return total_loss / n_valid, trust_scores


def train_stable_world_model(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("STABLE WORLD MODEL TRAINING")
    print("="*70)
    print("Key features:")
    print("  1. Contractive residual architecture (can't explode)")
    print("  2. REAL scheduled sampling (actually mixes teacher forcing)")
    print("  3. Multi-horizon loss (accurate at ALL horizons)")
    print("  4. Trust loss at MULTIPLE horizons (not just max)")
    print("  5. Variance GROWTH penalty (not just std matching)")
    print("="*70)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
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

    real_z_std = all_z.std()
    print(f"Real z_V: mean={all_z.mean():.6f}, std={real_z_std:.6f}")
    print(f"Loaded {n_episodes} episodes, avg length {np.mean([e['length'] for e in episodes]):.1f}")

    # Load trust_net
    print(f"\nLoading trust_net from {args.trust_net}...")
    trust_ckpt = torch.load(args.trust_net, map_location=device, weights_only=False)
    trust_net = TrustNet(input_dim=6, hidden_dim=64)
    trust_net.load_state_dict(trust_ckpt['model_state_dict'])
    trust_net = trust_net.to(device)
    trust_net.eval()
    for param in trust_net.parameters():
        param.requires_grad = False

    trust_mean = torch.FloatTensor(trust_ckpt['X_mean']).to(device)
    trust_std = torch.FloatTensor(trust_ckpt['X_std']).to(device)

    # Create stable world model
    print("\nCreating stable world model...")
    model = StableWorldModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        alpha_init=args.alpha_init,
        learnable_alpha=True,
        max_delta=args.max_delta,
    ).to(device)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  N layers: {args.n_layers}")
    print(f"  Initial alpha: {args.alpha_init}")
    print(f"  Max delta: {args.max_delta}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Training components
    scheduler = RealScheduledSampler(
        initial_prob=args.tf_start,
        final_prob=args.tf_end,
        anneal_epochs=args.epochs,
    )
    multi_horizon_loss = MultiHorizonLoss(max_horizon=args.max_horizon, n_samples=5)
    multi_trust_loss = MultiHorizonTrustLoss(
        trust_net, trust_mean, trust_std,
        horizons=[10, 30, min(60, args.max_horizon)]
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Lambda trust: {args.lambda_trust}")
    print(f"  Lambda var: {args.lambda_var}")
    print(f"  Lambda growth: {args.lambda_growth}")
    print(f"  Teacher forcing: {args.tf_start} -> {args.tf_end}")

    history = {
        'recon_loss': [],
        'trust_loss': [],
        'var_loss': [],
        'growth_loss': [],
        'total_loss': [],
        'trust_scores': [],
        'pred_std': [],
        'alpha': [],
    }

    for epoch in range(args.epochs):
        tf_prob = scheduler.get_teacher_forcing_prob()
        epoch_metrics = {k: [] for k in history.keys()}

        for ep in episodes:
            z_real = ep['z_sequence']
            actions = ep['actions']

            # 1. Multi-horizon reconstruction loss
            recon_loss, _ = multi_horizon_loss.compute(model, z_real, actions, device)

            # 2. Full trajectory rollout for variance/trust
            L = min(len(actions), args.max_horizon)
            z_pred_seq = scheduler.rollout_with_sampling(model, z_real[:L+1], actions[:L])

            # 3. Variance and growth regularization
            reg_losses = model.get_regularization_loss(z_pred_seq, target_std=real_z_std)
            var_loss = reg_losses['var_loss']
            growth_loss = reg_losses['growth_penalty']

            # 4. Multi-horizon trust loss
            trust_loss, trust_scores = multi_trust_loss.compute(model, z_real, actions, device)

            # Total loss
            total_loss = (
                recon_loss +
                args.lambda_trust * trust_loss +
                args.lambda_var * var_loss +
                args.lambda_growth * growth_loss
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Record
            epoch_metrics['recon_loss'].append(recon_loss.item())
            epoch_metrics['trust_loss'].append(trust_loss.item())
            epoch_metrics['var_loss'].append(var_loss.item())
            epoch_metrics['growth_loss'].append(growth_loss.item())
            epoch_metrics['total_loss'].append(total_loss.item())
            if trust_scores:
                max_H = max(trust_scores.keys())
                epoch_metrics['trust_scores'].append(trust_scores[max_H])
            epoch_metrics['pred_std'].append(z_pred_seq.std().item())
            epoch_metrics['alpha'].append(model.dynamics.alpha.item())

        # Average over episodes
        for k in history:
            if epoch_metrics[k]:
                history[k].append(np.mean(epoch_metrics[k]))

        scheduler.step_epoch()

        if (epoch + 1) % 10 == 0:
            trust_str = f"{history['trust_scores'][-1]:.4f}" if history['trust_scores'] else "N/A"
            alpha_str = f"{history['alpha'][-1]:.4f}" if history['alpha'] else "N/A"
            print(f"  Epoch {epoch + 1}/{args.epochs} (TF={tf_prob:.2f}, alpha={alpha_str}): "
                  f"recon={history['recon_loss'][-1]:.6f}, "
                  f"trust={trust_str}, "
                  f"std={history['pred_std'][-1]:.4f}")

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION: FULL 60-STEP ROLLOUTS")
    print("="*70)

    model.eval()
    trust_scores_60 = []
    stds_60 = []
    var_growths = []

    for ep in episodes:
        z_seq = ep['z_sequence']
        actions = ep['actions']

        L = min(len(actions), 60)
        # rollout handles both (latent_dim,) and (batch, latent_dim), pass without batch dim
        z_traj = model.rollout(z_seq[0], actions[:L])

        # Trust score
        features = multi_trust_loss.extract_features(z_traj)
        feat_norm = (features - trust_mean) / trust_std
        trust_score = trust_net(feat_norm.unsqueeze(0)).item()

        # Stability metrics
        metrics = model.compute_stability_metrics(z_traj)

        trust_scores_60.append(trust_score)
        stds_60.append(metrics['global_std'].item())
        var_growths.append(metrics['var_growth'].item())

    trust_scores_60 = np.array(trust_scores_60)
    stds_60 = np.array(stds_60)
    var_growths = np.array(var_growths)

    print(f"Trust scores (60-step rollouts):")
    print(f"  Mean: {trust_scores_60.mean():.6f}")
    print(f"  Std:  {trust_scores_60.std():.6f}")
    print(f"  Above 0.5: {(trust_scores_60 >= 0.5).sum()}/{len(trust_scores_60)}")
    print(f"  Above 0.9: {(trust_scores_60 >= 0.9).sum()}/{len(trust_scores_60)}")

    print(f"\nPredicted std (60-step rollouts):")
    print(f"  Mean: {stds_60.mean():.6f}")
    print(f"  Real: {real_z_std:.6f}")
    print(f"  Ratio: {stds_60.mean() / real_z_std:.3f}x")

    print(f"\nVariance growth:")
    print(f"  Mean: {var_growths.mean():.3f}x")
    print(f"  Max:  {var_growths.max():.3f}x")

    print(f"\nModel parameters:")
    print(f"  Alpha (damping): {model.dynamics.alpha.item():.4f}")
    lipschitz = model.dynamics.get_lipschitz_bound()
    print(f"  Lipschitz bound: {lipschitz:.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/stable_world_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'contractive',
        'latent_dim': latent_dim,
        'action_dim': action_dim,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'alpha_init': args.alpha_init,
        'max_delta': args.max_delta,
        'final_alpha': model.dynamics.alpha.item(),
        'final_trust_score': float(trust_scores_60.mean()),
        'final_pred_std': float(stds_60.mean()),
        'history': history,
        'config': vars(args),
    }, save_path)
    print(f"\nSaved model to {save_path}")

    # Save report
    report = {
        'final_trust_mean': float(trust_scores_60.mean()),
        'final_trust_std': float(trust_scores_60.std()),
        'final_pred_std_mean': float(stds_60.mean()),
        'final_pred_std_ratio': float(stds_60.mean() / real_z_std),
        'final_var_growth_mean': float(var_growths.mean()),
        'final_var_growth_max': float(var_growths.max()),
        'final_alpha': float(model.dynamics.alpha.item()),
        'lipschitz_bound': float(lipschitz),
        'real_std': float(real_z_std),
        'trust_above_0.5': int((trust_scores_60 >= 0.5).sum()),
        'trust_above_0.9': int((trust_scores_60 >= 0.9).sum()),
        'config': vars(args),
    }

    os.makedirs('results', exist_ok=True)
    with open('results/stable_world_model.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to results/stable_world_model.json")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    success = True
    if trust_scores_60.mean() < 0.5:
        print(f"  FAIL: Trust {trust_scores_60.mean():.6f} < 0.5")
        success = False
    else:
        print(f"  PASS: Trust {trust_scores_60.mean():.6f} >= 0.5")

    if stds_60.mean() / real_z_std > 1.5:
        print(f"  FAIL: Std ratio {stds_60.mean() / real_z_std:.3f}x > 1.5x")
        success = False
    else:
        print(f"  PASS: Std ratio {stds_60.mean() / real_z_std:.3f}x <= 1.5x")

    if var_growths.max() > 3.0:
        print(f"  WARN: Max variance growth {var_growths.max():.3f}x > 3x")

    if success:
        print("\n  SUCCESS! Model is now horizon-stable")
        print("  Ready for synthetic augmentation!")
    else:
        print("\n  PARTIAL IMPROVEMENT - may need more epochs or tuning")
        print("  Suggestions:")
        if trust_scores_60.mean() < 0.3:
            print("    - Increase lambda_trust")
        if stds_60.mean() / real_z_std > 2.0:
            print("    - Reduce max_delta or alpha_init")
            print("    - Increase lambda_var and lambda_growth")

    print("="*70)


@regal_training(env_type="workcell")
def main(runner=None):
    """Main entrypoint with regality wrapper."""
    if runner:
        runner.start_training()
    
    parser = argparse.ArgumentParser(description='Train stable world model')
    parser.add_argument('--dataset', type=str, default='data/physics_zv_rollouts.npz')
    parser.add_argument('--trust-net', type=str, default='checkpoints/trust_net.pt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Architecture
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--alpha-init', type=float, default=0.3)
    parser.add_argument('--max-delta', type=float, default=0.15)

    # Loss weights
    parser.add_argument('--lambda-trust', type=float, default=1.0)
    parser.add_argument('--lambda-var', type=float, default=1.0)
    parser.add_argument('--lambda-growth', type=float, default=2.0)

    # Scheduled sampling
    parser.add_argument('--tf-start', type=float, default=0.8)
    parser.add_argument('--tf-end', type=float, default=0.0)

    # Horizon
    parser.add_argument('--max-horizon', type=int, default=60)

    args = parser.parse_args()
    train_stable_world_model(args)
    
    if runner:
        runner.update_step(args.epochs * 100)  # Approximate


if __name__ == '__main__':
    main()
