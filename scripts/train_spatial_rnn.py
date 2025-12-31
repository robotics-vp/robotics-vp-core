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
from typing import Any, Dict, List, Optional

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
from src.utils.training_env import should_use_amp, device_info, run_with_oom_recovery
from src.utils.logging_schema import make_training_log_entry, write_training_log_entry
from src.utils.failure_sentinel import FailureSentinel
import hashlib


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


def compute_prediction_loss(
    model: SpatialRNN,
    input_seq: List[Any],
    target_seq: List[Any],
    device: torch.device,
) -> torch.Tensor:
    if len(input_seq) != len(target_seq):
        raise ValueError(f"Sequence and targets must have same length, got {len(input_seq)} vs {len(target_seq)}")

    hidden_states = {level: None for level in model.levels}
    total_loss = torch.tensor(0.0, device=device)
    num_steps = len(input_seq)

    for t in range(num_steps - 1):
        current = input_seq[t]
        target_next = target_seq[t + 1]

        if isinstance(current, dict):
            for level in model.levels:
                if level not in current:
                    continue
                feat = current[level]
                feat_tensor = torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device).unsqueeze(0)
                if model.use_checkpointing:
                    from src.utils.training_env import checkpoint_if_enabled
                    feat_tensor.requires_grad_(True)
                    hidden_states[level] = checkpoint_if_enabled(model.cells[level], feat_tensor, hidden_states.get(level), enabled=True)
                else:
                    hidden_states[level] = model.cells[level](feat_tensor, hidden_states.get(level))
        else:
            feat = np.asarray(current, dtype=np.float32).flatten()
            feat_per_level = len(feat) // len(model.levels)
            for i, level in enumerate(model.levels):
                start = i * feat_per_level
                end = start + feat_per_level if i < len(model.levels) - 1 else len(feat)
                level_feat = feat[start:end]
                if len(level_feat) < model.feature_dim:
                    level_feat = np.pad(level_feat, (0, model.feature_dim - len(level_feat)))
                level_feat = level_feat[:model.feature_dim]
                feat_tensor = torch.as_tensor(level_feat, device=device).unsqueeze(0)
                hidden_states[level] = model.cells[level](feat_tensor, hidden_states.get(level))

        if isinstance(target_next, dict):
            for level in model.levels:
                if level not in target_next or hidden_states.get(level) is None:
                    continue
                pred = hidden_states[level].mean(dim=[2, 3])
                target_feat = torch.as_tensor(np.asarray(target_next[level], dtype=np.float32), device=device)
                if target_feat.dim() == 1:
                    target_feat = target_feat.unsqueeze(0)
                if target_feat.shape[1] < model.hidden_dim:
                    target_feat = F.pad(target_feat, (0, model.hidden_dim - target_feat.shape[1]))
                target_feat = target_feat[:, :model.hidden_dim]
                total_loss = total_loss + F.mse_loss(pred, target_feat)
        else:
            target_feat = torch.as_tensor(np.asarray(target_next, dtype=np.float32).flatten(), device=device)
            pred_parts = []
            for level in model.levels:
                if hidden_states.get(level) is not None:
                    pred_parts.append(hidden_states[level].mean(dim=[2, 3]))
            if pred_parts:
                pred = torch.cat(pred_parts, dim=1).flatten()
                min_len = min(pred.numel(), target_feat.numel())
                total_loss = total_loss + F.mse_loss(pred[:min_len], target_feat[:min_len])

    return total_loss / max(1, num_steps - 1)


def train_epoch(
    model: SpatialRNN,
    dataset: SpatialRNNDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_prediction: float,
    alpha_smoothness: float,
    log_file: str,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    epoch_idx: int,
    run_name: str,
    config_digest: str,
    sentinel: FailureSentinel,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    total_prediction_loss = 0.0
    total_smoothness_loss = 0.0
    total_loss_val = 0.0
    num_sequences = 0

    for idx, sample in enumerate(dataset):
        features = sample.get("features", [])
        if len(features) < 2:
            continue

        optimizer.zero_grad(set_to_none=True)

        with sentinel.monitor(epoch_idx * len(dataset) + idx, model):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                # Split into input sequence (0 to T-1) and targets (1 to T)
                input_seq = features[:-1]
                target_seq = features[1:]

                # Compute prediction loss
                prediction_loss = compute_prediction_loss(model, input_seq, target_seq, device)
                smoothness_loss = torch.tensor(0.0, device=device)
                loss = alpha_prediction * prediction_loss + alpha_smoothness * smoothness_loss

            # Backward pass
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_prediction_loss += prediction_loss.item()
        total_smoothness_loss += smoothness_loss.item()
        total_loss_val += loss.item()
        num_sequences += 1

        # Log step (every 10 steps)
        if idx % 10 == 0:
            log_entry = make_training_log_entry(
                run_name=run_name,
                step=epoch_idx * len(dataset) + idx,
                epoch=epoch_idx,
                phase="train",
                loss=loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                task_id="spatial_rnn",
                seed=0,
                config_digest=config_digest,
                amp_enabled=use_amp,
                gpu_mem_mb=None,
                gpu_util_pct=None,
                extra={
                    "prediction_loss": prediction_loss.item(),
                    "smoothness_loss": smoothness_loss.item()
                }
            )
            write_training_log_entry(log_file, log_entry)

    return {
        "prediction_loss": total_prediction_loss / max(1, num_sequences),
        "smoothness_loss": total_smoothness_loss / max(1, num_sequences),
        "total_loss": total_loss_val / max(1, num_sequences),
    }




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
    parser.add_argument("--use-mixed-precision", action="store_true", help="Enable mixed precision training (AMP)")
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

    # AMP Setup
    use_amp = args.use_mixed_precision or should_use_amp({"training": {"amp": args.use_mixed_precision}})
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    print(f"AMP Enabled: {use_amp}")

    config_digest = hashlib.md5(json.dumps(vars(args), sort_keys=True).encode()).hexdigest()[:8]

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        losses = run_with_oom_recovery(
            train_epoch,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            device=device,
            alpha_prediction=args.alpha_prediction,
            alpha_smoothness=args.alpha_smoothness,
            log_file=log_file,
            scaler=scaler,
            use_amp=use_amp,
            epoch_idx=epoch,
            run_name=f"spatial_rnn_{args.seed}",
            config_digest=config_digest,
            sentinel=FailureSentinel(),
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
