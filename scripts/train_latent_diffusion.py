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

# Regality wrapper (Phase 10: Unitree G1 primary; workcell is curriculum)
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from src.training.wrap_training_entrypoint import regal_training


class ZVTransitionDataset(Dataset):
    """
    Dataset of (z_t, a_t, z_{t+1}) transitions from z_V rollouts.
    """
    def __init__(self, npz_path, use_scene_tracks=False, use_semantic_conditioning=False, semantic_sidecar_path=None):
        """Load transitions from npz file.

        Args:
            npz_path: Path to transitions .npz
            use_scene_tracks: Include scene tracks if available
            use_semantic_conditioning: Load semantic conditioning from sidecar
            semantic_sidecar_path: Path to JSON file with semantic conditioning vectors.
                Defaults to npz_path with .json extension.
        """
        data = np.load(npz_path, allow_pickle=True)

        self.n_episodes = int(data['n_episodes'])
        self.latent_dim = int(data['latent_dim'])
        self.use_scene_tracks = use_scene_tracks
        self.use_semantic_conditioning = use_semantic_conditioning

        # Extract transitions
        self.z_current = []
        self.actions = []
        self.z_next = []
        self.scene_tracks = []
        self.episode_ids = []

        has_tracks = 'ep_0_scene_tracks' in data

        if self.use_scene_tracks and not has_tracks:
            print(f"WARNING: use_scene_tracks=True but 'scene_tracks' not found in {npz_path}")

        for ep in range(self.n_episodes):
            z_seq = data[f'ep_{ep}_z_sequence']  # (T+1, latent_dim)
            actions = data[f'ep_{ep}_actions']   # (T, action_dim)
            
            tracks = None
            if self.use_scene_tracks and has_tracks:
                 tracks = data[f'ep_{ep}_scene_tracks'] # (T+1, K, D)

            # Create transitions
            for t in range(len(actions)):
                self.z_current.append(z_seq[t])
                self.actions.append(actions[t])
                self.z_next.append(z_seq[t + 1])
                if self.use_scene_tracks and has_tracks:
                    self.scene_tracks.append(tracks[t])
                self.episode_ids.append(ep)

        self.z_current = np.array(self.z_current)
        self.actions = np.array(self.actions)
        self.z_next = np.array(self.z_next)
        self.scene_tracks = np.array(self.scene_tracks) if (self.use_scene_tracks and has_tracks) else None
        self.episode_ids = np.array(self.episode_ids)

        # Load semantic conditioning (Section H)
        self.semantic_cond = None
        self.semantic_cond_dim = 0
        if self.use_semantic_conditioning:
            self._load_semantic_conditioning(
                semantic_sidecar_path or str(npz_path).replace('.npz', '_semantic_cond.json')
            )

        print(f"Loaded {len(self)} transitions from {self.n_episodes} episodes")
        if self.scene_tracks is not None:
            print(f"  Includes scene tracks: {self.scene_tracks.shape}")
        if self.semantic_cond is not None:
            print(f"  Includes semantic conditioning: dim={self.semantic_cond_dim}")

    def _load_semantic_conditioning(self, path):
        """Load semantic conditioning vectors from JSON sidecar.

        Expected format: list of dicts with 'episode_id' and 'cond_vector'
        (float list), or a dict mapping episode_id -> cond_vector.
        """
        import json
        try:
            with open(path) as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"WARNING: Could not load semantic conditioning from {path}: {e}")
            self.use_semantic_conditioning = False
            return

        # Build per-episode conditioning vectors
        ep_cond = {}
        if isinstance(raw, list):
            for item in raw:
                ep_id = int(item.get('episode_id', -1))
                vec = item.get('cond_vector', [])
                if ep_id >= 0 and vec:
                    ep_cond[ep_id] = np.array(vec, dtype=np.float32)
        elif isinstance(raw, dict):
            for k, v in raw.items():
                ep_cond[int(k)] = np.array(v, dtype=np.float32)

        if not ep_cond:
            print(f"WARNING: No semantic conditioning found in {path}")
            self.use_semantic_conditioning = False
            return

        # Determine dim from first vector
        sample_vec = next(iter(ep_cond.values()))
        self.semantic_cond_dim = len(sample_vec)

        # Expand episode-level conditioning to per-transition
        per_transition = []
        default_cond = np.zeros(self.semantic_cond_dim, dtype=np.float32)
        for ep_id in self.episode_ids:
            per_transition.append(ep_cond.get(int(ep_id), default_cond))
        self.semantic_cond = np.array(per_transition)

    def __len__(self):
        return len(self.z_current)

    def __getitem__(self, idx):
        item = [
            torch.FloatTensor(self.z_current[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.z_next[idx]),
        ]
        
        if self.scene_tracks is not None:
             item.append(torch.FloatTensor(self.scene_tracks[idx]))

        if self.semantic_cond is not None:
            item.append(torch.FloatTensor(self.semantic_cond[idx]))

        return tuple(item)


class LatentDynamicsModel(nn.Module):
    """
    Simple MLP model for z_V dynamics prediction.

    Predicts: z_{t+1} = f(z_t, a_t)
    """
    def __init__(self, latent_dim=128, action_dim=2, hidden_dim=256, semantic_cond_dim=0):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.semantic_cond_dim = semantic_cond_dim

        # Input dim: z_t + a_t + (optional) semantic conditioning
        input_dim = latent_dim + action_dim + semantic_cond_dim

        # Encoder for (z_t, a_t, [semantic_cond])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

    def forward(self, z_t, a_t, semantic_cond=None):
        """
        Predict z_{t+1} given z_t and a_t, optionally conditioned on
        semantic WM topology.

        Args:
            z_t: (B, latent_dim) current state
            a_t: (B, action_dim) action
            semantic_cond: (B, semantic_cond_dim) optional semantic conditioning
                           from WM topology (skill-edge targets, env-primitive
                           targets, risk/affordance families, etc.)

        Returns:
            z_next_pred: (B, latent_dim) predicted next state
            uncertainty: (B, latent_dim) prediction uncertainty (log variance)
        """
        # Concatenate state, action, and optional semantic conditioning
        if self.semantic_cond_dim > 0:
            if semantic_cond is None:
                semantic_cond = torch.zeros(
                    (z_t.shape[0], self.semantic_cond_dim),
                    dtype=z_t.dtype,
                    device=z_t.device,
                )
            x = torch.cat([z_t, a_t, semantic_cond], dim=-1)
        else:
            x = torch.cat([z_t, a_t], dim=-1)

        # Encode
        h = self.encoder(x)

        # Predict residual
        delta = self.residual_head(h)

        # Predict uncertainty
        log_var = self.uncertainty_head(h)

        # Next state = current + residual
        z_next_pred = z_t + delta

        return z_next_pred, log_var

    def sample_next(self, z_t, a_t, semantic_cond=None):
        """Sample next state with uncertainty."""
        z_next_mean, log_var = self.forward(z_t, a_t, semantic_cond=semantic_cond)
        std = torch.exp(0.5 * log_var)
        z_next = z_next_mean + std * torch.randn_like(std)
        return z_next


class TemporalTransformer(nn.Module):
    """
    Transformer-based model for z_V sequence prediction.

    Predicts multi-step rollouts in latent space.
    """
    def __init__(
        self,
        latent_dim=128,
        action_dim=2,
        hidden_dim=256,
        num_heads=4,
        num_layers=3,
        semantic_cond_dim=0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.semantic_cond_dim = semantic_cond_dim

        # Embeddings
        self.state_embed = nn.Linear(latent_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_cond_dim, hidden_dim) if semantic_cond_dim > 0 else None

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

    def forward(self, z_sequence, action_sequence, semantic_cond=None):
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
        if self.semantic_proj is not None:
            if semantic_cond is None:
                semantic_cond = torch.zeros(
                    (z_sequence.shape[0], self.semantic_cond_dim),
                    dtype=z_sequence.dtype,
                    device=z_sequence.device,
                )
            x = x + self.semantic_proj(semantic_cond).unsqueeze(1)

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
    use_scene_tracks=False,
    use_semantic_conditioning=False,
    semantic_sidecar_path=None,
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
        use_scene_tracks: Whether to include scene tracks in dataset
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    # use_scene_tracks defaults to False unless specified
    dataset = ZVTransitionDataset(
        dataset_path,
        use_scene_tracks=use_scene_tracks,
        use_semantic_conditioning=use_semantic_conditioning,
        semantic_sidecar_path=semantic_sidecar_path,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    latent_dim = dataset.latent_dim
    action_dim = dataset.actions.shape[1]

    if model_type == 'mlp':
        model = LatentDynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            semantic_cond_dim=dataset.semantic_cond_dim,
        ).to(device)
        print("Model: LatentDynamicsModel (MLP)")
    elif model_type == 'transformer':
        model = TemporalTransformer(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            semantic_cond_dim=dataset.semantic_cond_dim,
        ).to(device)
        print("Model: TemporalTransformer")
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

        for batch_idx, batch in enumerate(dataloader):
            z_t = batch[0]
            a_t = batch[1]
            z_next = batch[2]
            semantic_cond = batch[-1] if dataset.semantic_cond is not None else None
            z_t = z_t.to(device)
            a_t = a_t.to(device)
            z_next = z_next.to(device)
            if semantic_cond is not None:
                semantic_cond = semantic_cond.to(device)

            # Forward pass
            if model_type == 'mlp':
                z_next_pred, log_var = model(z_t, a_t, semantic_cond=semantic_cond)
                loss = gaussian_nll_loss(z_next_pred, log_var, z_next)
                mse = ((z_next_pred - z_next) ** 2).mean()
            else:
                # For transformer, use single step for now
                z_next_pred = model(z_t.unsqueeze(1), a_t.unsqueeze(1), semantic_cond=semantic_cond)
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
        'semantic_cond_dim': dataset.semantic_cond_dim,
        'use_semantic_conditioning': bool(dataset.semantic_cond is not None),
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


@regal_training(env_type="unitree_g1")
def main(runner=None):
    """Main entrypoint with regality wrapper."""
    if runner:
        runner.start_training()
    
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
    parser.add_argument(
        '--use-scene-tracks',
        action='store_true',
        help='Use scene_tracks_v1 if available in dataset'
    )
    parser.add_argument(
        '--use-semantic-conditioning',
        action='store_true',
        help='Condition on semantic WM topology (requires sidecar JSON)'
    )
    parser.add_argument(
        '--semantic-sidecar',
        type=str,
        default=None,
        help='Path to semantic conditioning sidecar JSON'
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
        use_scene_tracks=args.use_scene_tracks,
        use_semantic_conditioning=args.use_semantic_conditioning,
        semantic_sidecar_path=args.semantic_sidecar,
    )

    if runner:
        runner.update_step(args.epochs * 100)  # Approximate

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Sample synthetic z_V trajectories:")
    print("   python scripts/sample_zv_rollouts.py \\")
    print(f"     --model {save_path} \\")
    print("     --samples 50")
    print()
    print("2. Run ΔMPL/novelty valuation on synthetic data:")
    print("   python scripts/validate_dmpl_novelty.py \\")
    print("     --synthetic data/synthetic_zv_rollouts.npz")
    print("="*60)


if __name__ == '__main__':
    main()
