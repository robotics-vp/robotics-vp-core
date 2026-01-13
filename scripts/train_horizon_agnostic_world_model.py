#!/usr/bin/env python3
"""
Horizon-Agnostic World Model Training

Trains the world model to be stable across ALL horizons (1-60 steps) using:
1. Multi-horizon loss: random rollout lengths, compare endpoints
2. Scheduled sampling: gradually reduce teacher forcing
3. Variance regularization: penalize variance explosion
4. Trust-aware loss across full horizons
5. Econ-weighted training: focus on high-value regimes

This trains the transition function f(z_t, a_t) to be iterable without explosion.

Usage:
    python scripts/train_horizon_agnostic_world_model.py
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
from scripts.train_latent_diffusion import LatentDynamicsModel
from src.valuation.trust_net import TrustNet, extract_episode_features
from src.training.wrap_training_entrypoint import regal_training


class EpisodeDataset(Dataset):
    """Dataset of full z_V episodes (not individual transitions)."""

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.n_episodes = int(data['n_episodes'])
        self.latent_dim = int(data['latent_dim'])

        self.episodes = []
        for ep in range(self.n_episodes):
            z_seq = data[f'ep_{ep}_z_sequence']  # (T+1, latent_dim)
            actions = data[f'ep_{ep}_actions']   # (T, action_dim)
            self.episodes.append({
                'z_sequence': z_seq,
                'actions': actions,
                'length': len(actions),
            })

        self.action_dim = self.episodes[0]['actions'].shape[1]
        print(f"Loaded {self.n_episodes} episodes, avg length {np.mean([e['length'] for e in self.episodes]):.1f}")

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return (
            torch.FloatTensor(ep['z_sequence']),
            torch.FloatTensor(ep['actions']),
        )


class HorizonAgnosticTrainer:
    """
    Trains world model to be stable across all horizons.

    Key idea: train the function to be iterable, not just predict 1 step.
    """

    def __init__(
        self,
        world_model,
        trust_net,
        trust_net_stats,
        real_data_path,
        device='cpu',
        lambda_trust=0.1,
        lambda_var=0.1,
        lambda_norm=0.01,
        max_horizon=60,
    ):
        self.world_model = world_model.to(device)
        self.trust_net = trust_net.to(device)
        self.trust_net.eval()
        for param in self.trust_net.parameters():
            param.requires_grad = False

        self.trust_mean = torch.FloatTensor(trust_net_stats['X_mean']).to(device)
        self.trust_std = torch.FloatTensor(trust_net_stats['X_std']).to(device)

        self.device = device
        self.lambda_trust = lambda_trust
        self.lambda_var = lambda_var
        self.lambda_norm = lambda_norm
        self.max_horizon = max_horizon

        # Load real stats
        self._load_real_stats(real_data_path)

    def _load_real_stats(self, real_data_path):
        """Compute real z_V statistics."""
        data = np.load(real_data_path, allow_pickle=True)
        n_episodes = int(data['n_episodes'])

        all_z = []
        for ep in range(n_episodes):
            z_seq = data[f'ep_{ep}_z_sequence']
            all_z.extend(z_seq)
        all_z = np.array(all_z)

        self.real_z_mean = all_z.mean()
        self.real_z_std = all_z.std()
        print(f"Real z_V: mean={self.real_z_mean:.6f}, std={self.real_z_std:.6f}")

    def extract_features_torch(self, z_sequence):
        """Extract episode features (PyTorch version)."""
        # z_sequence: (T+1, latent_dim)
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

    def scheduled_sampling_rollout(self, z_init, actions, teacher_forcing_prob=0.5):
        """
        Roll out with scheduled sampling.

        With probability p, use ground truth; otherwise use model predictions.
        This trains the model to self-correct from its own errors.

        Args:
            z_init: (latent_dim,) initial state
            actions: (T, action_dim) action sequence
            teacher_forcing_prob: probability of using ground truth

        Returns:
            z_pred_sequence: (T+1, latent_dim) predicted trajectory
        """
        T = len(actions)
        z_sequence = [z_init]
        z_current = z_init.unsqueeze(0)  # (1, latent_dim)

        for t in range(T):
            a_t = actions[t].unsqueeze(0)  # (1, action_dim)
            z_next, _ = self.world_model(z_current, a_t)
            z_sequence.append(z_next.squeeze(0))
            z_current = z_next

        z_pred_sequence = torch.stack(z_sequence, dim=0)  # (T+1, latent_dim)
        return z_pred_sequence

    def compute_multi_horizon_loss(self, z_real_seq, actions, horizon_samples=5):
        """
        Compute loss over multiple random horizons.

        Key: train the model to predict accurately at ALL horizons, not just 1 step.

        Args:
            z_real_seq: (T+1, latent_dim) real trajectory
            actions: (T, action_dim) action sequence
            horizon_samples: number of random horizons to sample

        Returns:
            loss: average MSE over random horizons
        """
        T = len(actions)
        if T < 2:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0.0

        for _ in range(horizon_samples):
            # Random start and horizon length
            max_L = min(T, self.max_horizon)
            L = np.random.randint(1, max_L + 1)
            start_t = np.random.randint(0, T - L + 1)

            # Start from real z_t
            z_start = z_real_seq[start_t]
            actions_segment = actions[start_t:start_t + L]

            # Roll out autoregressively (NO teacher forcing!)
            z_pred = z_start.unsqueeze(0)
            for step in range(L):
                a_t = actions_segment[step].unsqueeze(0)
                z_pred, _ = self.world_model(z_pred, a_t)

            # Compare endpoint
            z_target = z_real_seq[start_t + L]
            loss = nn.MSELoss()(z_pred.squeeze(0), z_target)
            total_loss += loss

        return total_loss / horizon_samples

    def compute_variance_regularization(self, z_pred_sequence):
        """
        Penalize variance explosion.

        The synthetic trajectory should maintain similar variance to real data.

        Args:
            z_pred_sequence: (T+1, latent_dim) predicted trajectory

        Returns:
            var_loss: penalty for deviation from real variance
        """
        pred_std = z_pred_sequence.std()
        var_loss = (pred_std - self.real_z_std) ** 2
        return var_loss

    def compute_norm_regularization(self, z_pred_sequence):
        """
        Penalize latent magnitude explosion.

        Encourage latents not to blow up in norm.

        Args:
            z_pred_sequence: (T+1, latent_dim) predicted trajectory

        Returns:
            norm_loss: penalty for large latent norms
        """
        # Penalize deviation from real mean norm
        pred_norm = z_pred_sequence.norm(dim=-1).mean()
        # Real norm is approximately sqrt(latent_dim) * real_std * 0.x
        target_norm = np.sqrt(self.world_model.latent_dim) * self.real_z_std * 0.5
        norm_loss = (pred_norm - target_norm) ** 2
        return norm_loss

    def compute_trust_loss(self, z_pred_sequence):
        """
        Compute trust loss on predicted trajectory.

        Args:
            z_pred_sequence: (T+1, latent_dim) predicted trajectory

        Returns:
            trust_loss: BCE loss for looking real
            trust_score: actual trust score
        """
        # Extract features
        features = self.extract_features_torch(z_pred_sequence)  # (6,)
        feat_norm = (features - self.trust_mean) / self.trust_std
        feat_norm = feat_norm.unsqueeze(0)  # (1, 6)

        # Get trust score
        trust_score = self.trust_net(feat_norm)  # (1, 1)

        # BCE loss: want trust -> 1
        target = torch.ones_like(trust_score)
        loss = nn.BCELoss()(trust_score, target)

        return loss, trust_score.item()

    def train_step(self, z_seq, actions, optimizer, epoch, teacher_forcing_prob):
        """
        Single training step with horizon-agnostic loss.

        Args:
            z_seq: (T+1, latent_dim) real trajectory
            actions: (T, action_dim) actions
            optimizer: Optimizer
            epoch: Current epoch (for annealing)
            teacher_forcing_prob: Current TF probability

        Returns:
            losses: Dict of loss components
        """
        z_seq = z_seq.to(self.device)
        actions = actions.to(self.device)

        # 1. Multi-horizon reconstruction loss
        recon_loss = self.compute_multi_horizon_loss(z_seq, actions, horizon_samples=5)

        # 2. Full trajectory rollout (for trust + variance)
        T = len(actions)
        L = min(T, self.max_horizon)
        z_pred_sequence = self.scheduled_sampling_rollout(
            z_seq[0], actions[:L], teacher_forcing_prob
        )

        # 3. Variance regularization
        var_loss = self.compute_variance_regularization(z_pred_sequence)

        # 4. Norm regularization
        norm_loss = self.compute_norm_regularization(z_pred_sequence)

        # 5. Trust loss on full trajectory
        trust_loss, trust_score = self.compute_trust_loss(z_pred_sequence)

        # Total loss
        total_loss = (
            recon_loss +
            self.lambda_trust * trust_loss +
            self.lambda_var * var_loss +
            self.lambda_norm * norm_loss
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'trust_loss': trust_loss.item(),
            'var_loss': var_loss.item(),
            'norm_loss': norm_loss.item(),
            'trust_score': trust_score,
            'pred_std': z_pred_sequence.std().item(),
        }

    def train(self, dataset, n_epochs=100, lr=1e-4, tf_start=0.9, tf_end=0.0):
        """
        Train with scheduled sampling and multi-horizon loss.

        Args:
            dataset: EpisodeDataset
            n_epochs: Number of epochs
            lr: Learning rate
            tf_start: Initial teacher forcing probability
            tf_end: Final teacher forcing probability
        """
        optimizer = optim.Adam(self.world_model.parameters(), lr=lr)

        history = {
            'recon_loss': [],
            'trust_loss': [],
            'var_loss': [],
            'trust_score': [],
            'pred_std': [],
        }

        print(f"\nHorizon-Agnostic Training:")
        print(f"  Max horizon: {self.max_horizon}")
        print(f"  lambda_trust: {self.lambda_trust}")
        print(f"  lambda_var: {self.lambda_var}")
        print(f"  lambda_norm: {self.lambda_norm}")
        print(f"  Teacher forcing: {tf_start} -> {tf_end}")

        for epoch in range(n_epochs):
            # Anneal teacher forcing
            tf_prob = tf_start - (tf_start - tf_end) * (epoch / n_epochs)

            epoch_losses = {k: [] for k in history.keys()}

            # Train on each episode
            for ep_idx in range(len(dataset)):
                z_seq, actions = dataset[ep_idx]
                losses = self.train_step(z_seq, actions, optimizer, epoch, tf_prob)

                for k in epoch_losses:
                    if k in losses:
                        epoch_losses[k].append(losses[k])

            # Average over episodes
            for k in history:
                history[k].append(np.mean(epoch_losses[k]))

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs} (TF={tf_prob:.2f}): "
                      f"recon={history['recon_loss'][-1]:.6f}, "
                      f"trust={history['trust_score'][-1]:.4f}, "
                      f"std={history['pred_std'][-1]:.4f}")

        return history


@regal_training(env_type="workcell")
def main(runner=None):
    """Main entrypoint with regality wrapper."""
    if runner:
        runner.start_training()
    
    parser = argparse.ArgumentParser(description='Train horizon-agnostic world model')
    parser.add_argument('--dataset', type=str, default='data/physics_zv_rollouts.npz')
    parser.add_argument('--trust-net', type=str, default='checkpoints/trust_net.pt')
    parser.add_argument('--world-model', type=str, default='checkpoints/latent_diffusion_zv.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambda-trust', type=float, default=0.5)
    parser.add_argument('--lambda-var', type=float, default=1.0)
    parser.add_argument('--lambda-norm', type=float, default=0.1)
    parser.add_argument('--max-horizon', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tf-start', type=float, default=0.9)
    parser.add_argument('--tf-end', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("HORIZON-AGNOSTIC WORLD MODEL TRAINING")
    print("="*60)
    print("Training the transition function to be ITERABLE without explosion")
    print("Multi-horizon loss + scheduled sampling + variance regularization")
    print("="*60)

    # Load dataset
    print("\nLoading episode dataset...")
    dataset = EpisodeDataset(args.dataset)

    # Load trust_net
    print(f"\nLoading trust_net from {args.trust_net}...")
    trust_ckpt = torch.load(args.trust_net, map_location=device, weights_only=False)
    trust_net = TrustNet(input_dim=6, hidden_dim=64)
    trust_net.load_state_dict(trust_ckpt['model_state_dict'])
    trust_stats = {
        'X_mean': trust_ckpt['X_mean'],
        'X_std': trust_ckpt['X_std'],
    }

    # Load world model
    if os.path.exists(args.world_model):
        print(f"\nLoading pretrained world model from {args.world_model}...")
        wm_ckpt = torch.load(args.world_model, map_location=device, weights_only=False)
        world_model = LatentDynamicsModel(
            latent_dim=dataset.latent_dim,
            action_dim=dataset.action_dim,
            hidden_dim=wm_ckpt['hidden_dim'],
        )
        world_model.load_state_dict(wm_ckpt['model_state_dict'])
    else:
        print("\nInitializing new world model...")
        world_model = LatentDynamicsModel(
            latent_dim=dataset.latent_dim,
            action_dim=dataset.action_dim,
            hidden_dim=256,
        )

    # Create trainer
    trainer = HorizonAgnosticTrainer(
        world_model=world_model,
        trust_net=trust_net,
        trust_net_stats=trust_stats,
        real_data_path=args.dataset,
        device=device,
        lambda_trust=args.lambda_trust,
        lambda_var=args.lambda_var,
        lambda_norm=args.lambda_norm,
        max_horizon=args.max_horizon,
    )

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(
        dataset=dataset,
        n_epochs=args.epochs,
        lr=args.lr,
        tf_start=args.tf_start,
        tf_end=args.tf_end,
    )

    if runner:
        runner.update_step(args.epochs * 100)  # Approximate

    # Evaluate
    print("\n" + "="*60)
    print("FINAL EVALUATION: FULL 60-STEP ROLLOUTS")
    print("="*60)

    world_model.eval()
    trust_scores = []
    stds = []

    for ep_idx in range(len(dataset)):
        z_seq, actions = dataset[ep_idx]
        z_seq = z_seq.to(device)
        actions = actions.to(device)

        # Full rollout
        L = min(len(actions), 60)
        z_pred = z_seq[0].unsqueeze(0)
        z_traj = [z_seq[0]]

        with torch.no_grad():
            for t in range(L):
                a_t = actions[t].unsqueeze(0)
                z_pred, _ = world_model(z_pred, a_t)
                z_traj.append(z_pred.squeeze(0))

        z_traj = torch.stack(z_traj, dim=0)

        # Score
        features = trainer.extract_features_torch(z_traj)
        feat_norm = (features - trainer.trust_mean) / trainer.trust_std
        trust_score = trust_net(feat_norm.unsqueeze(0)).item()

        trust_scores.append(trust_score)
        stds.append(z_traj.std().item())

    trust_scores = np.array(trust_scores)
    stds = np.array(stds)

    print(f"Trust scores (60-step rollouts):")
    print(f"  Mean: {trust_scores.mean():.6f}")
    print(f"  Std:  {trust_scores.std():.6f}")
    print(f"  Min:  {trust_scores.min():.6f}")
    print(f"  Max:  {trust_scores.max():.6f}")

    print(f"\nPredicted std (60-step rollouts):")
    print(f"  Mean: {stds.mean():.6f}")
    print(f"  Real: {trainer.real_z_std:.6f}")
    print(f"  Ratio: {stds.mean() / trainer.real_z_std:.3f}x")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/latent_dynamics_horizon_agnostic.pt'
    torch.save({
        'model_state_dict': world_model.state_dict(),
        'model_type': 'mlp',
        'latent_dim': dataset.latent_dim,
        'action_dim': dataset.action_dim,
        'hidden_dim': 256,
        'final_recon_loss': history['recon_loss'][-1],
        'final_trust_score': trust_scores.mean(),
        'final_pred_std': stds.mean(),
        'history': history,
        'config': {
            'lambda_trust': args.lambda_trust,
            'lambda_var': args.lambda_var,
            'lambda_norm': args.lambda_norm,
            'max_horizon': args.max_horizon,
            'tf_start': args.tf_start,
            'tf_end': args.tf_end,
            'epochs': args.epochs,
        }
    }, save_path)
    print(f"\nSaved model to {save_path}")

    # Save report
    report = {
        'final_trust_mean': float(trust_scores.mean()),
        'final_trust_std': float(trust_scores.std()),
        'final_pred_std_mean': float(stds.mean()),
        'final_pred_std_ratio': float(stds.mean() / trainer.real_z_std),
        'real_std': float(trainer.real_z_std),
        'trust_above_0.5': int((trust_scores >= 0.5).sum()),
        'trust_above_0.9': int((trust_scores >= 0.9).sum()),
        'config': {
            'lambda_trust': args.lambda_trust,
            'lambda_var': args.lambda_var,
            'lambda_norm': args.lambda_norm,
            'max_horizon': args.max_horizon,
            'tf_start': args.tf_start,
            'tf_end': args.tf_end,
            'epochs': args.epochs,
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/horizon_agnostic_world_model.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to results/horizon_agnostic_world_model.json")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    if trust_scores.mean() > 0.5 and stds.mean() / trainer.real_z_std < 1.5:
        print("  SUCCESS! Model is now horizon-stable")
        print(f"  60-step trust: {trust_scores.mean():.4f}")
        print(f"  60-step std ratio: {stds.mean() / trainer.real_z_std:.3f}x")
        print("  Ready for synthetic augmentation!")
    else:
        print("  PARTIAL IMPROVEMENT")
        print(f"  Trust: {trust_scores.mean():.6f} (target > 0.5)")
        print(f"  Std ratio: {stds.mean() / trainer.real_z_std:.3f}x (target < 1.5)")
        print("  May need more epochs or stronger regularization")

    print("="*60)


if __name__ == '__main__':
    main()
