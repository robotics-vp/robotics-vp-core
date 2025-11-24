"""
Train Spatial RNN for temporal coherence in visual features.

Losses:
- Next-step prediction MSE
- Temporal smoothness (L2 on deltas)
- Optional InfoNCE across time

Outputs:
- Deterministic checkpoints (JSON-safe)
- Loss components logged per step (JSON lines)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Cannot train Spatial RNN.")
    sys.exit(1)

from src.vision.spatial_rnn import SpatialRNN
from src.utils.json_safe import to_json_safe


class SpatialRNNDataset:
    """
    Dataset for Spatial RNN training.

    Yields sequences of feature pyramids with task/episode/timestep metadata.
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 16,
        levels: List[str] = None,
        feature_dim: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.levels = levels or ["P3", "P4", "P5"]
        self.feature_dim = feature_dim

        # For now, generate synthetic data (in real usage, load from disk)
        self.num_sequences = 100
        self.sequences = self._generate_synthetic_sequences()

    def _generate_synthetic_sequences(self) -> List[Dict[str, Any]]:
        """Generate synthetic sequences for testing."""
        sequences = []
        np.random.seed(42)

        for i in range(self.num_sequences):
            seq = {
                "task_id": f"task_{i % 5}",
                "episode_id": f"episode_{i}",
                "features": [],
                "metadata": {
                    "sequence_index": i,
                    "sequence_length": self.sequence_length,
                }
            }

            # Generate smooth trajectory with noise
            for t in range(self.sequence_length):
                pyramid = {}
                for level in self.levels:
                    # Smooth trajectory + noise
                    base = np.sin(t / 5.0) + np.random.randn() * 0.1
                    feat = np.random.randn(self.feature_dim) * 0.5 + base
                    pyramid[level] = feat.astype(np.float32)

                seq["features"].append(pyramid)

            sequences.append(seq)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.sequences[idx]


def compute_temporal_smoothness_loss(hidden_sequence: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute temporal smoothness loss: L2 on deltas.

    Args:
        hidden_sequence: List of hidden states [B, C, H, W]

    Returns:
        Smoothness loss (scalar)
    """
    if len(hidden_sequence) < 2:
        return torch.tensor(0.0)

    deltas = []
    for t in range(len(hidden_sequence) - 1):
        delta = hidden_sequence[t + 1] - hidden_sequence[t]
        deltas.append(delta)

    if not deltas:
        return torch.tensor(0.0)

    # Mean L2 norm of deltas
    loss = torch.mean(torch.stack([torch.norm(d) for d in deltas]))
    return loss


def train_epoch(
    model: SpatialRNN,
    dataset: SpatialRNNDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_prediction: float = 1.0,
    alpha_smoothness: float = 0.1,
    log_file: Path = None,
) -> Dict[str, float]:
    """
    Train one epoch.

    Returns:
        Dict with average losses
    """
    model.train()

    total_prediction_loss = 0.0
    total_smoothness_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    for idx in range(len(dataset)):
        seq_data = dataset[idx]
        features = seq_data["features"]

        if len(features) < 2:
            continue

        # Split into input (t=0..T-1) and target (t=1..T)
        inputs = features[:-1]
        targets = features[1:]

        # Forward pass with hidden state tracking
        hidden_states = {level: None for level in dataset.levels}
        hidden_sequence = []

        prediction_loss = 0.0
        for t in range(len(inputs)):
            current = inputs[t]
            target_next = targets[t]

            # Update hidden states
            for level in dataset.levels:
                if level not in current:
                    continue
                feat = current[level]
                feat_tensor = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0).to(device)
                hidden_states[level] = model.cells[level](feat_tensor, hidden_states.get(level))

            # Track for smoothness
            if hidden_states.get(dataset.levels[0]) is not None:
                hidden_sequence.append(hidden_states[dataset.levels[0]].clone())

            # Compute prediction loss
            for level in dataset.levels:
                if level not in target_next or hidden_states.get(level) is None:
                    continue
                pred = hidden_states[level].mean(dim=[2, 3])  # [B, hidden_dim]
                target_feat = torch.from_numpy(np.asarray(target_next[level], dtype=np.float32)).to(device)
                if target_feat.dim() == 1:
                    target_feat = target_feat.unsqueeze(0)
                # Match dimensions
                if target_feat.shape[1] < model.hidden_dim:
                    target_feat = F.pad(target_feat, (0, model.hidden_dim - target_feat.shape[1]))
                target_feat = target_feat[:, :model.hidden_dim]

                loss = F.mse_loss(pred, target_feat)
                prediction_loss += loss

        # Temporal smoothness
        smoothness_loss = compute_temporal_smoothness_loss(hidden_sequence)

        # Total loss
        loss = alpha_prediction * prediction_loss + alpha_smoothness * smoothness_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        total_prediction_loss += prediction_loss.item()
        total_smoothness_loss += smoothness_loss.item()
        total_loss += loss.item()
        num_batches += 1

        # Log to JSON lines
        if log_file is not None:
            log_entry = {
                "event": "train_step",
                "step": idx,
                "task_id": seq_data["task_id"],
                "episode_id": seq_data["episode_id"],
                "loss": float(loss.item()),
                "prediction_loss": float(prediction_loss.item()),
                "smoothness_loss": float(smoothness_loss.item()),
                "metadata": seq_data.get("metadata", {}),
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(to_json_safe(log_entry)) + "\n")

    # Average losses
    avg_losses = {
        "prediction_loss": total_prediction_loss / max(1, num_batches),
        "smoothness_loss": total_smoothness_loss / max(1, num_batches),
        "total_loss": total_loss / max(1, num_batches),
    }

    return avg_losses


def save_checkpoint(
    model: SpatialRNN,
    epoch: int,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    checkpoint_path: Path,
    seed: int,
):
    """Save deterministic checkpoint (JSON-safe)."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "trained_steps": epoch,
        "seed": seed,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Spatial RNN")
    parser.add_argument("--data_dir", type=str, default="data/spatial_rnn", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/spatial_rnn", help="Output directory")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--feature_dim", type=int, default=8, help="Feature dimension")
    parser.add_argument("--levels", type=str, nargs="+", default=["P3", "P4", "P5"], help="Pyramid levels")
    parser.add_argument("--mode", type=str, default="convgru", choices=["convgru", "s4d"], help="RNN mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--alpha_prediction", type=float, default=1.0, help="Prediction loss weight")
    parser.add_argument("--alpha_smoothness", type=float, default=0.1, help="Smoothness loss weight")
    parser.add_argument("--sequence_length", type=int, default=16, help="Sequence length")
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training_log.jsonl"

    # Dataset
    dataset = SpatialRNNDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        levels=args.levels,
        feature_dim=args.feature_dim,
    )
    print(f"Dataset: {len(dataset)} sequences")

    # Model
    model = SpatialRNN(
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        levels=args.levels,
        mode=args.mode,
        seed=args.seed,
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training config
    config = {
        "hidden_dim": args.hidden_dim,
        "feature_dim": args.feature_dim,
        "levels": args.levels,
        "mode": args.mode,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "alpha_prediction": args.alpha_prediction,
        "alpha_smoothness": args.alpha_smoothness,
        "sequence_length": args.sequence_length,
    }

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        losses = train_epoch(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            device=device,
            alpha_prediction=args.alpha_prediction,
            alpha_smoothness=args.alpha_smoothness,
            log_file=log_file,
        )

        print(f"  Prediction Loss: {losses['prediction_loss']:.6f}")
        print(f"  Smoothness Loss: {losses['smoothness_loss']:.6f}")
        print(f"  Total Loss: {losses['total_loss']:.6f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        save_checkpoint(
            model=model,
            epoch=epoch + 1,
            config=config,
            metrics=losses,
            checkpoint_path=checkpoint_path,
            seed=args.seed,
        )

    print(f"\nTraining complete! Logs saved to {log_file}")


if __name__ == "__main__":
    main()
