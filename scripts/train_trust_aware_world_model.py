#!/usr/bin/env python3
"""
Trust-Aware World Model Training (Integrative Fix)

Trains latent dynamics model with trust_net as a frozen critic.
The world model is incentivized to generate synthetic latents that:
1. Reconstruct well (MSE loss)
2. Look real to trust_net (trust loss)
3. Match real z_V distribution in feature space (feature matching loss)

This integrates the world model into the trust/valuation flywheel instead of
treating it as an isolated subsystem.

Usage:
    python scripts/train_trust_aware_world_model.py
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from scripts.train_latent_diffusion import ZVTransitionDataset, LatentDynamicsModel
from src.valuation.trust_net import TrustNet, extract_episode_features


class TrustAwareWorldModelTrainer:
    """
    Trains world model with trust-aware regularization.

    Loss = L_recon + lambda_trust * L_trust + lambda_feat * L_feat

    Where:
    - L_recon: MSE reconstruction loss on (z_t, a_t) -> z_{t+1}
    - L_trust: BCE(trust_net(z_syn), 1) - make synthetic look real
    - L_feat: ||mean(phi(z_real)) - mean(phi(z_syn))||^2 - feature matching
    """

    def __init__(
        self,
        world_model,
        trust_net,
        trust_net_stats,
        real_data_path,
        device='cpu',
        lambda_trust=0.1,
        lambda_feat=0.01,
        rollout_steps=10,
    ):
        self.world_model = world_model.to(device)
        self.trust_net = trust_net.to(device)
        self.trust_net.eval()  # Frozen!
        for param in self.trust_net.parameters():
            param.requires_grad = False

        self.trust_mean = torch.FloatTensor(trust_net_stats['X_mean']).to(device)
        self.trust_std = torch.FloatTensor(trust_net_stats['X_std']).to(device)

        self.real_data_path = real_data_path
        self.device = device
        self.lambda_trust = lambda_trust
        self.lambda_feat = lambda_feat
        self.rollout_steps = rollout_steps

        # Load real z_V statistics for feature matching
        self._load_real_stats()

    def _load_real_stats(self):
        """Load real z_V episode features for matching."""
        data = np.load(self.real_data_path, allow_pickle=True)
        n_episodes = int(data['n_episodes'])

        real_features = []
        for ep in range(n_episodes):
            z_seq = data[f'ep_{ep}_z_sequence']
            feat = extract_episode_features(z_seq)
            real_features.append(feat)

        self.real_features = np.array(real_features)
        self.real_features_mean = torch.FloatTensor(
            self.real_features.mean(axis=0)
        ).to(self.device)
        self.real_features_std = torch.FloatTensor(
            self.real_features.std(axis=0) + 1e-6
        ).to(self.device)

        # Also compute z_V distribution stats
        all_z = []
        for ep in range(n_episodes):
            z_seq = data[f'ep_{ep}_z_sequence']
            all_z.extend(z_seq)
        all_z = np.array(all_z)

        self.real_z_mean = torch.FloatTensor(all_z.mean(axis=0)).to(self.device)
        self.real_z_std = torch.FloatTensor(all_z.std(axis=0) + 1e-6).to(self.device)

        print(f"Loaded {n_episodes} real episodes for feature matching")
        print(f"  Real z_V mean: {all_z.mean():.6f}, std: {all_z.std():.6f}")

    def extract_episode_features_torch(self, z_sequence):
        """
        Extract episode features from z_V sequence (PyTorch version).

        Args:
            z_sequence: (batch, T+1, latent_dim) tensor

        Returns:
            features: (batch, 6) tensor
        """
        # Global mean across time and dims
        global_mean = z_sequence.mean(dim=(1, 2))  # (batch,)

        # Global std
        global_std = z_sequence.std(dim=(1, 2))  # (batch,)

        # Min and max
        global_min = z_sequence.min(dim=2)[0].min(dim=1)[0]  # (batch,)
        global_max = z_sequence.max(dim=2)[0].max(dim=1)[0]  # (batch,)

        # Variance across dimensions (per-timestep mean, then std across dims)
        per_time_mean = z_sequence.mean(dim=2)  # (batch, T+1)
        dim_var = per_time_mean.std(dim=1)  # (batch,)

        # Temporal smoothness (mean absolute difference)
        diffs = torch.abs(z_sequence[:, 1:, :] - z_sequence[:, :-1, :])
        smoothness = diffs.mean(dim=(1, 2))  # (batch,)

        features = torch.stack([
            global_mean, global_std, global_min, global_max, dim_var, smoothness
        ], dim=1)  # (batch, 6)

        return features

    def rollout_synthetic(self, z_init, actions):
        """
        Roll out world model to generate synthetic z_V sequence.

        Args:
            z_init: (batch, latent_dim) initial states
            actions: (batch, T, action_dim) action sequence

        Returns:
            z_sequence: (batch, T+1, latent_dim) synthetic trajectory
        """
        batch_size, T, action_dim = actions.shape

        z_sequence = [z_init]
        z_current = z_init

        for t in range(T):
            a_t = actions[:, t, :]
            z_next, _ = self.world_model(z_current, a_t)
            z_sequence.append(z_next)
            z_current = z_next

        z_sequence = torch.stack(z_sequence, dim=1)  # (batch, T+1, latent_dim)
        return z_sequence

    def compute_trust_loss(self, synthetic_features):
        """
        Compute trust loss: BCE(trust_net(z_syn), 1).

        The world model is rewarded for generating synthetic that trust_net
        classifies as real.

        Args:
            synthetic_features: (batch, 6) episode features

        Returns:
            trust_loss: scalar
        """
        # Normalize features
        feat_norm = (synthetic_features - self.trust_mean) / self.trust_std

        # Get trust scores
        trust_scores = self.trust_net(feat_norm)  # (batch, 1)

        # BCE loss: want trust_scores -> 1 (look real)
        target = torch.ones_like(trust_scores)
        loss = nn.BCELoss()(trust_scores, target)

        return loss, trust_scores.mean().item()

    def compute_feature_matching_loss(self, synthetic_features):
        """
        Feature matching loss: ||mean(feat_real) - mean(feat_syn)||^2.

        Ensures synthetic episodes have similar feature distribution to real.

        Args:
            synthetic_features: (batch, 6) episode features

        Returns:
            feat_loss: scalar
        """
        syn_feat_mean = synthetic_features.mean(dim=0)  # (6,)

        # MSE between mean features
        loss = ((syn_feat_mean - self.real_features_mean) ** 2).mean()

        return loss

    def train_step(self, batch, optimizer):
        """
        Single training step with trust-aware loss.

        Args:
            batch: (z_current, actions, z_next) from dataloader
            optimizer: Optimizer for world model

        Returns:
            losses: Dict of loss components
        """
        z_current, actions, z_next = batch
        z_current = z_current.to(self.device)
        actions = actions.to(self.device)
        z_next = z_next.to(self.device)

        # ===== 1. Reconstruction loss (standard MSE) =====
        z_next_pred, log_var = self.world_model(z_current, actions)
        recon_loss = nn.MSELoss()(z_next_pred, z_next)

        # ===== 2. Trust loss (needs full episode rollout) =====
        # Sample initial states from batch
        batch_size = min(16, len(z_current))  # Smaller batch for rollouts
        init_idx = torch.randperm(len(z_current))[:batch_size]
        z_init = z_current[init_idx]

        # Generate synthetic episode by rolling out
        # Use random actions from the batch distribution
        syn_actions = []
        for _ in range(self.rollout_steps):
            # Sample random action from batch
            a_idx = torch.randint(0, len(actions), (batch_size,))
            syn_actions.append(actions[a_idx])
        syn_actions = torch.stack(syn_actions, dim=1)  # (batch_size, T, action_dim)

        z_syn_sequence = self.rollout_synthetic(z_init, syn_actions)

        # Extract features from synthetic episodes
        syn_features = self.extract_episode_features_torch(z_syn_sequence)

        # Trust loss
        trust_loss, mean_trust = self.compute_trust_loss(syn_features)

        # ===== 3. Feature matching loss =====
        feat_loss = self.compute_feature_matching_loss(syn_features)

        # ===== 4. Total loss =====
        total_loss = (
            recon_loss +
            self.lambda_trust * trust_loss +
            self.lambda_feat * feat_loss
        )

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track synthetic z_V statistics
        syn_z_std = z_syn_sequence.std().item()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'trust_loss': trust_loss.item(),
            'feat_loss': feat_loss.item(),
            'mean_trust': mean_trust,
            'syn_z_std': syn_z_std,
        }

    def train(self, dataset, n_epochs=50, batch_size=64, lr=1e-4):
        """
        Train world model with trust-aware loss.

        Args:
            dataset: ZVTransitionDataset
            n_epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            Training history
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.world_model.parameters(), lr=lr)

        history = {
            'recon_loss': [],
            'trust_loss': [],
            'feat_loss': [],
            'mean_trust': [],
            'syn_z_std': [],
        }

        print(f"\nTraining with trust-aware loss:")
        print(f"  lambda_trust: {self.lambda_trust}")
        print(f"  lambda_feat: {self.lambda_feat}")
        print(f"  rollout_steps: {self.rollout_steps}")
        print(f"  Real z_V std: {self.real_z_std.mean().item():.6f}")

        for epoch in range(n_epochs):
            epoch_losses = {k: [] for k in history.keys()}

            for batch in dataloader:
                losses = self.train_step(batch, optimizer)
                for k, v in losses.items():
                    if k != 'total_loss':
                        epoch_losses[k].append(v)

            # Average over epoch
            for k in history.keys():
                history[k].append(np.mean(epoch_losses[k]))

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}: "
                      f"recon={history['recon_loss'][-1]:.6f}, "
                      f"trust={history['trust_loss'][-1]:.4f}, "
                      f"feat={history['feat_loss'][-1]:.6f}, "
                      f"syn_trust={history['mean_trust'][-1]:.4f}, "
                      f"syn_std={history['syn_z_std'][-1]:.4f}")

        return history


def main():
    parser = argparse.ArgumentParser(description='Train trust-aware world model')
    parser.add_argument('--dataset', type=str, default='data/physics_zv_rollouts.npz',
                        help='Path to real z_V rollouts')
    parser.add_argument('--trust-net', type=str, default='checkpoints/trust_net.pt',
                        help='Path to trained trust_net checkpoint')
    parser.add_argument('--world-model', type=str, default='checkpoints/latent_diffusion_zv.pt',
                        help='Path to pretrained world model (optional)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lambda-trust', type=float, default=0.1,
                        help='Weight for trust loss')
    parser.add_argument('--lambda-feat', type=float, default=0.01,
                        help='Weight for feature matching loss')
    parser.add_argument('--rollout-steps', type=int, default=10,
                        help='Steps to roll out for trust evaluation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("TRUST-AWARE WORLD MODEL TRAINING")
    print("="*60)
    print("Integrating trust_net as differentiable critic in world model loss")
    print("Goal: Synthetic z_V should look real to trust_net (trust → higher)")
    print("="*60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = ZVTransitionDataset(args.dataset)

    # Load trust_net (frozen critic)
    print(f"\nLoading frozen trust_net from {args.trust_net}...")
    trust_ckpt = torch.load(args.trust_net, map_location=device, weights_only=False)

    trust_net = TrustNet(input_dim=6, hidden_dim=64)
    trust_net.load_state_dict(trust_ckpt['model_state_dict'])

    trust_stats = {
        'X_mean': trust_ckpt['X_mean'],
        'X_std': trust_ckpt['X_std'],
    }
    print(f"  Trust_net validation acc: {trust_ckpt['metrics']['best_val_acc']:.3f}")
    print(f"  Current trust gap: {trust_ckpt['metrics']['trust_gap']:.4f}")

    # Initialize or load world model
    latent_dim = dataset.latent_dim
    action_dim = dataset.actions.shape[1]

    if os.path.exists(args.world_model):
        print(f"\nLoading pretrained world model from {args.world_model}...")
        wm_ckpt = torch.load(args.world_model, map_location=device, weights_only=False)
        world_model = LatentDynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=wm_ckpt['hidden_dim'],
        )
        world_model.load_state_dict(wm_ckpt['model_state_dict'])
        print(f"  Original MSE: {wm_ckpt['final_mse']:.6f}")
    else:
        print(f"\nInitializing new world model...")
        world_model = LatentDynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=256,
        )

    # Create trainer
    trainer = TrustAwareWorldModelTrainer(
        world_model=world_model,
        trust_net=trust_net,
        trust_net_stats=trust_stats,
        real_data_path=args.dataset,
        device=device,
        lambda_trust=args.lambda_trust,
        lambda_feat=args.lambda_feat,
        rollout_steps=args.rollout_steps,
    )

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(
        dataset=dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Evaluate improvement
    print("\n" + "="*60)
    print("TRAINING COMPLETE - ANALYZING RESULTS")
    print("="*60)

    initial_trust = history['mean_trust'][0]
    final_trust = history['mean_trust'][-1]
    initial_std = history['syn_z_std'][0]
    final_std = history['syn_z_std'][-1]

    print(f"Synthetic trust score:")
    print(f"  Initial: {initial_trust:.6f}")
    print(f"  Final:   {final_trust:.6f}")
    print(f"  Change:  {final_trust - initial_trust:+.6f} ({100*(final_trust/initial_trust - 1):+.1f}%)")

    print(f"\nSynthetic z_V std:")
    print(f"  Initial: {initial_std:.6f}")
    print(f"  Final:   {final_std:.6f}")
    print(f"  Real:    {trainer.real_z_std.mean().item():.6f}")
    print(f"  Ratio (final/real): {final_std / trainer.real_z_std.mean().item():.3f}x")

    # Save improved model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/latent_dynamics_trust_aware.pt'
    torch.save({
        'model_state_dict': world_model.state_dict(),
        'model_type': 'mlp',
        'latent_dim': latent_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'final_mse': history['recon_loss'][-1],
        'final_trust': final_trust,
        'final_syn_std': final_std,
        'history': history,
        'config': {
            'lambda_trust': args.lambda_trust,
            'lambda_feat': args.lambda_feat,
            'rollout_steps': args.rollout_steps,
            'epochs': args.epochs,
        }
    }, save_path)
    print(f"\nSaved trust-aware world model to {save_path}")

    # Save training report
    report = {
        'initial_trust': initial_trust,
        'final_trust': final_trust,
        'trust_improvement': final_trust - initial_trust,
        'initial_syn_std': initial_std,
        'final_syn_std': final_std,
        'real_z_std': trainer.real_z_std.mean().item(),
        'final_std_ratio': final_std / trainer.real_z_std.mean().item(),
        'final_recon_loss': history['recon_loss'][-1],
        'config': {
            'lambda_trust': args.lambda_trust,
            'lambda_feat': args.lambda_feat,
            'rollout_steps': args.rollout_steps,
            'epochs': args.epochs,
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/trust_aware_world_model.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to results/trust_aware_world_model.json")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if final_trust > 0.1:
        print("  SUCCESS: Synthetic trust improved significantly")
        print("  World model now generates more realistic latents")
        print("  Ready for next step: regenerate synthetic data and re-score")
    elif final_trust > initial_trust * 10:
        print("  PARTIAL SUCCESS: Trust improved but still low")
        print("  May need to tune lambda_trust or train longer")
    else:
        print("  LIMITED IMPROVEMENT: Trust barely moved")
        print("  Consider: higher lambda_trust, more epochs, or architectural changes")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Sample new synthetic episodes with improved world model:")
    print(f"   python scripts/sample_zv_rollouts.py --model {save_path}")
    print()
    print("2. Re-score with trust_net:")
    print("   python scripts/train_trust_net.py  # Uses new synthetic")
    print()
    print("3. If trust improves: test trust-weighted offline RL")
    print("4. If trust still low: increase lambda_trust or epochs")
    print("="*60)


if __name__ == '__main__':
    main()
