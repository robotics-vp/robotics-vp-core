#!/usr/bin/env python3
"""
Train Latent Dynamics Model on z_V Sequences (Phase B.2)

Trains a temporal model to predict z_V dynamics:
- Next-step prediction: z_{t+1} = f(z_t, a_t)
- Small-horizon rollouts in latent space
- Can use simple MLP, 1D U-Net, or Transformer

Usage:
    python scripts/train_latent_diffusion.py --dataset data/physics_zv_rollouts.npz --epochs 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ZVTransitionDataset(Dataset):
    """
    Dataset of (z_t, a_t, z_{t+1}) transitions from z_V rollouts.
    """
    def __init__(self, npz_path):
        """Load transitions from npz file."""
        data = np.load(npz_path, allow_pickle=True)

        self.n_episodes = int(data['n_episodes'])
        self.latent_dim = int(data['latent_dim'])

        # Extract transitions
        self.z_current = []
        self.actions = []
        self.z_next = []
        self.episode_ids = []

        for ep in range(self.n_episodes):
            z_seq = data[f'ep_{ep}_z_sequence']  # (T+1, latent_dim)
            actions = data[f'ep_{ep}_actions']   # (T, action_dim)

            # Create transitions
            for t in range(len(actions)):
                self.z_current.append(z_seq[t])
                self.actions.append(actions[t])
                self.z_next.append(z_seq[t + 1])
                self.episode_ids.append(ep)

        self.z_current = np.array(self.z_current)
        self.actions = np.array(self.actions)
        self.z_next = np.array(self.z_next)
        self.episode_ids = np.array(self.episode_ids)

        print(f"Loaded {len(self)} transitions from {self.n_episodes} episodes")

    def __len__(self):
        return len(self.z_current)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.z_current[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.z_next[idx]),
        )


class LatentDynamicsModel(nn.Module):
    """
    Simple MLP model for z_V dynamics prediction.

    Predicts: z_{t+1} = f(z_t, a_t)
    """
    def __init__(self, latent_dim=128, action_dim=2, hidden_dim=256):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Encoder for (z_t, a_t)
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Predict residual (delta)
        self.residual_head = nn.Linear(hidden_dim, latent_dim)

        # Also predict uncertainty (optional)
        self.uncertainty_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, a_t):
        """
        Predict z_{t+1} given z_t and a_t.

        Args:
            z_t: (B, latent_dim) current state
            a_t: (B, action_dim) action

        Returns:
            z_next_pred: (B, latent_dim) predicted next state
            uncertainty: (B, latent_dim) prediction uncertainty (log variance)
        """
        # Concatenate state and action
        x = torch.cat([z_t, a_t], dim=-1)  # (B, latent_dim + action_dim)

        # Encode
        h = self.encoder(x)  # (B, hidden_dim)

        # Predict residual
        delta = self.residual_head(h)  # (B, latent_dim)

        # Predict uncertainty
        log_var = self.uncertainty_head(h)  # (B, latent_dim)

        # Next state = current + residual
        z_next_pred = z_t + delta

        return z_next_pred, log_var

    def sample_next(self, z_t, a_t):
        """Sample next state with uncertainty."""
        z_next_mean, log_var = self.forward(z_t, a_t)
        std = torch.exp(0.5 * log_var)
        z_next = z_next_mean + std * torch.randn_like(std)
        return z_next


class TemporalTransformer(nn.Module):
    """
    Transformer-based model for z_V sequence prediction.

    Predicts multi-step rollouts in latent space.
    """
    def __init__(self, latent_dim=128, action_dim=2, hidden_dim=256, num_heads=4, num_layers=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Embeddings
        self.state_embed = nn.Linear(latent_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_sequence, action_sequence):
        """
        Predict next z_V given sequence of (z, a) pairs.

        Args:
            z_sequence: (B, T, latent_dim)
            action_sequence: (B, T, action_dim)

        Returns:
            z_next: (B, latent_dim) predicted next state
        """
        B, T, _ = z_sequence.shape

        # Embed states and actions
        z_emb = self.state_embed(z_sequence)      # (B, T, hidden_dim)
        a_emb = self.action_embed(action_sequence)  # (B, T, hidden_dim)

        # Combine (interleave or add)
        x = z_emb + a_emb  # (B, T, hidden_dim)

        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]

        # Transformer
        h = self.transformer(x)  # (B, T, hidden_dim)

        # Take last position
        h_last = h[:, -1, :]  # (B, hidden_dim)

        # Project to latent space
        z_next = self.output_proj(h_last)  # (B, latent_dim)

        return z_next


def gaussian_nll_loss(pred_mean, pred_log_var, target):
    """
    Gaussian negative log-likelihood loss with uncertainty.

    Args:
        pred_mean: (B, D) predicted mean
        pred_log_var: (B, D) predicted log variance
        target: (B, D) ground truth

    Returns:
        loss: scalar
    """
    # NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
    inv_var = torch.exp(-pred_log_var)
    mse = (target - pred_mean) ** 2
    nll = 0.5 * (pred_log_var + mse * inv_var)
    return nll.mean()


def train_latent_dynamics(
    dataset_path,
    n_epochs=100,
    batch_size=64,
    lr=1e-4,
    model_type='mlp',
    hidden_dim=256,
    save_dir='checkpoints',
    device=None,
):
    """
    Train latent dynamics model on z_V transitions.

    Args:
        dataset_path: Path to npz dataset
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        model_type: 'mlp' or 'transformer'
        hidden_dim: Hidden dimension
        save_dir: Directory to save model
        device: Torch device

    Returns:
        model, save_path
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = ZVTransitionDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    latent_dim = dataset.latent_dim
    action_dim = dataset.actions.shape[1]

    if model_type == 'mlp':
        model = LatentDynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        print(f"Model: LatentDynamicsModel (MLP)")
    elif model_type == 'transformer':
        model = TemporalTransformer(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        print(f"Model: TemporalTransformer")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print()

    train_log = []

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        epoch_mse = []

        for batch_idx, (z_t, a_t, z_next) in enumerate(dataloader):
            z_t = z_t.to(device)
            a_t = a_t.to(device)
            z_next = z_next.to(device)

            # Forward pass
            if model_type == 'mlp':
                z_next_pred, log_var = model(z_t, a_t)
                loss = gaussian_nll_loss(z_next_pred, log_var, z_next)
                mse = ((z_next_pred - z_next) ** 2).mean()
            else:
                # For transformer, use single step for now
                z_next_pred = model(z_t.unsqueeze(1), a_t.unsqueeze(1))
                loss = nn.MSELoss()(z_next_pred, z_next)
                mse = loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_mse.append(mse.item())

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_mse = np.mean(epoch_mse)

        train_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'mse': avg_mse,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"MSE: {avg_mse:.6f}")

    # Save model
    save_path = os.path.join(save_dir, 'latent_diffusion_zv.pt')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': model_type,
        'latent_dim': latent_dim,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim,
        'n_epochs': n_epochs,
        'train_log': train_log,
        'final_mse': avg_mse,
    }
    torch.save(checkpoint, save_path)

    print(f"\n✅ Saved model to {save_path}")
    print(f"   Final MSE: {avg_mse:.6f}")

    # Save training log
    log_path = os.path.join(save_dir, 'latent_dynamics_train.csv')
    import csv
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'mse'])
        writer.writeheader()
        writer.writerows(train_log)
    print(f"   Training log saved to {log_path}")

    return model, save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train latent dynamics model on z_V')
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/physics_zv_rollouts.npz',
        help='Path to z_V rollouts dataset'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='mlp',
        choices=['mlp', 'transformer'],
        help='Model architecture'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='checkpoints',
        help='Directory to save model'
    )

    args = parser.parse_args()

    print("="*60)
    print("Phase B.2: Training Latent Dynamics Model on z_V")
    print("="*60)

    model, save_path = train_latent_dynamics(
        dataset_path=args.dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        save_dir=args.save_dir,
    )

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print(f"1. Sample synthetic z_V trajectories:")
    print(f"   python scripts/sample_zv_rollouts.py \\")
    print(f"     --model {save_path} \\")
    print(f"     --samples 50")
    print()
    print("2. Run ΔMPL/novelty valuation on synthetic data:")
    print(f"   python scripts/validate_dmpl_novelty.py \\")
    print(f"     --synthetic data/synthetic_zv_rollouts.npz")
    print("="*60)
