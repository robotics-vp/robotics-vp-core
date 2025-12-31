#!/usr/bin/env python3
"""
Training harness for CtRL-Sim-style BehaviourModel.

This script:
1. Loads trajectory dataset from export_lsd_vector_scene_dataset.py
2. Trains autoregressive BehaviourModel to predict next actions
3. Supports optional tilt regularization for return-conditional sampling
4. Saves checkpoints and logs training metrics

Usage:
    # Smoke test (minimal training):
    python scripts/train_behaviour_model.py \
        --dataset-path data/lsd_vector_scene_dataset \
        --num-epochs 1 \
        --batch-size 4 \
        --output-dir checkpoints/behaviour_model_smoke

    # Full training:
    python scripts/train_behaviour_model.py \
        --dataset-path data/lsd_vector_scene_dataset \
        --num-epochs 50 \
        --batch-size 32 \
        --lr 1e-4 \
        --tilt-regularization 0.1 \
        --output-dir checkpoints/behaviour_model
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# PyTorch imports (optional for scaffolding)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Training will use dummy implementation.")


@dataclass
class TrainingConfig:
    """Configuration for BehaviourModel training."""
    dataset_path: str
    output_dir: str = "checkpoints/behaviour_model"
    num_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    tilt_regularization: float = 0.0
    seed: int = 42
    val_split: float = 0.1
    log_interval: int = 10
    save_interval: int = 5
    max_seq_len: int = 100
    num_action_bins: int = 64


class TrajectoryDataset:
    """Dataset of trajectories for behaviour model training."""

    def __init__(
        self,
        dataset_path: str,
        max_seq_len: int = 100,
        num_action_bins: int = 64,
    ):
        self.dataset_path = Path(dataset_path)
        self.max_seq_len = max_seq_len
        self.num_action_bins = num_action_bins
        self.episodes: List[Dict[str, Any]] = []
        self.action_coder = None

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from shards."""
        index_path = self.dataset_path / "index.json"
        if not index_path.exists():
            raise ValueError(f"Dataset index not found: {index_path}")

        with open(index_path, "r") as f:
            index = json.load(f)

        # Load all shards
        for shard_info in index.get("shards", []):
            shard_path = self.dataset_path / shard_info["file_path"]
            if shard_path.exists():
                with open(shard_path, "r") as f:
                    shard_episodes = json.load(f)
                    self.episodes.extend(shard_episodes)

        print(f"Loaded {len(self.episodes)} episodes from {len(index.get('shards', []))} shards")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single episode as training example."""
        episode = self.episodes[idx]
        trajectory = episode.get("trajectory", [])

        # Extract actions and rewards
        actions = []
        rewards = []
        for step in trajectory[:self.max_seq_len]:
            action = step.get("action", [0.5, 0.5])
            if isinstance(action, (list, tuple)) and len(action) >= 2:
                actions.append(action[:2])
            else:
                actions.append([0.5, 0.5])
            rewards.append(step.get("reward", 0.0))

        # Pad if needed
        while len(actions) < self.max_seq_len:
            actions.append([0.5, 0.5])
            rewards.append(0.0)

        # Discretize actions for classification
        action_tokens = self._discretize_actions(actions)

        # Get difficulty features as conditioning
        difficulty = episode.get("difficulty_features", {})
        condition_vector = [
            difficulty.get("graph_density", 0.5),
            difficulty.get("route_length", 30.0) / 100.0,  # Normalize
            difficulty.get("tilt", 0.0),
            difficulty.get("num_dynamic_agents", 2.0) / 10.0,  # Normalize
        ]

        # Compute return for tilt conditioning
        total_return = sum(rewards)

        return {
            "action_tokens": np.array(action_tokens, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "condition": np.array(condition_vector, dtype=np.float32),
            "total_return": total_return,
            "seq_len": min(len(trajectory), self.max_seq_len),
        }

    def _discretize_actions(self, actions: List[List[float]]) -> List[int]:
        """Discretize continuous actions to tokens."""
        tokens = []
        for action in actions:
            # Map [0, 1] to bins
            speed_bin = int(np.clip(action[0], 0, 1) * (self.num_action_bins - 1))
            care_bin = int(np.clip(action[1], 0, 1) * (self.num_action_bins - 1))
            # Combined token: speed * num_bins + care
            token = speed_bin * self.num_action_bins + care_bin
            tokens.append(token)
        return tokens


if TORCH_AVAILABLE:
    class BehaviourModelPyTorch(nn.Module):
        """PyTorch implementation of CtRL-Sim-style BehaviourModel."""

        def __init__(
            self,
            num_action_tokens: int = 4096,
            d_model: int = 256,
            num_heads: int = 4,
            num_layers: int = 4,
            condition_dim: int = 4,
            max_seq_len: int = 100,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_model = d_model
            self.num_action_tokens = num_action_tokens

            # Token embedding
            self.token_embedding = nn.Embedding(num_action_tokens, d_model)

            # Positional encoding
            self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

            # Condition projection
            self.condition_proj = nn.Linear(condition_dim, d_model)

            # Transformer decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

            # Output head
            self.output_head = nn.Linear(d_model, num_action_tokens)

            # Causal mask
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            )

        def forward(
            self,
            action_tokens: torch.Tensor,
            condition: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                action_tokens: (batch, seq_len) int64 tensor of action tokens
                condition: (batch, condition_dim) float tensor

            Returns:
                logits: (batch, seq_len, num_action_tokens)
            """
            batch_size, seq_len = action_tokens.shape

            # Embed tokens
            x = self.token_embedding(action_tokens)  # (batch, seq, d_model)

            # Add positional encoding
            x = x + self.pos_encoding[:, :seq_len, :]

            # Project condition as memory for cross-attention
            cond_embed = self.condition_proj(condition).unsqueeze(1)  # (batch, 1, d_model)

            # Transformer forward with causal mask
            mask = self.causal_mask[:seq_len, :seq_len]
            x = self.transformer(x, cond_embed, tgt_mask=mask)

            # Output logits
            logits = self.output_head(x)

            return logits


def create_dataloaders(
    config: TrainingConfig,
) -> Tuple[Any, Any]:
    """Create train and validation dataloaders."""
    dataset = TrajectoryDataset(
        dataset_path=config.dataset_path,
        max_seq_len=config.max_seq_len,
        num_action_bins=config.num_action_bins,
    )

    # Split into train/val
    num_val = int(len(dataset) * config.val_split)
    num_train = len(dataset) - num_val

    indices = list(range(len(dataset)))
    random.Random(config.seed).shuffle(indices)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    if TORCH_AVAILABLE:
        from torch.utils.data import Subset

        class TorchDataset(Dataset):
            def __init__(self, base_dataset, indices):
                self.base = base_dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                item = self.base[self.indices[idx]]
                return {
                    "action_tokens": torch.from_numpy(item["action_tokens"]),
                    "rewards": torch.from_numpy(item["rewards"]),
                    "condition": torch.from_numpy(item["condition"]),
                    "total_return": torch.tensor(item["total_return"]),
                    "seq_len": item["seq_len"],
                }

        train_dataset = TorchDataset(dataset, train_indices)
        val_dataset = TorchDataset(dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    else:
        # Dummy loaders for scaffolding
        return [], []


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    config: TrainingConfig,
    epoch: int,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        action_tokens = batch["action_tokens"]
        condition = batch["condition"]
        seq_lens = batch["seq_len"]

        # Forward pass
        logits = model(action_tokens[:, :-1], condition)  # Predict next token

        # Compute loss
        targets = action_tokens[:, 1:]  # Shift targets
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Tilt regularization (optional)
        if config.tilt_regularization > 0:
            # Encourage higher log-probs for high-return sequences
            returns = batch["total_return"]
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            mean_log_prob = selected_log_probs.mean(dim=1)
            tilt_loss = -config.tilt_regularization * (returns * mean_log_prob).mean()
            loss = loss + tilt_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == targets).sum().item()
        total_tokens += targets.numel()

        if verbose and (batch_idx + 1) % config.log_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}


def validate(
    model,
    val_loader,
    criterion,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            action_tokens = batch["action_tokens"]
            condition = batch["condition"]

            logits = model(action_tokens[:, :-1], condition)
            targets = action_tokens[:, 1:]

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()

    avg_loss = total_loss / len(val_loader) if val_loader else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}


def train_behaviour_model(
    config: TrainingConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train BehaviourModel on trajectory dataset.

    Returns:
        Training summary
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)

    if verbose:
        print("=" * 60)
        print("BehaviourModel Training")
        print("=" * 60)
        print(f"Dataset: {config.dataset_path}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.lr}")
        print(f"Tilt regularization: {config.tilt_regularization}")
        print(f"Output: {output_dir}")
        print()

    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch not available. Running dummy training.")
        # Save dummy checkpoint
        dummy_results = {
            "status": "dummy_training",
            "reason": "pytorch_not_available",
            "config": {
                "dataset_path": config.dataset_path,
                "num_epochs": config.num_epochs,
            },
        }
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(dummy_results, f, indent=2)
        return dummy_results

    # Set torch seed
    torch.manual_seed(config.seed)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    if len(train_loader) == 0:
        print("ERROR: No training data loaded")
        return {"status": "error", "reason": "no_data"}

    # Create model
    num_action_tokens = config.num_action_bins * config.num_action_bins
    model = BehaviourModelPyTorch(
        num_action_tokens=num_action_tokens,
        d_model=256,
        num_heads=4,
        num_layers=4,
        condition_dim=4,
        max_seq_len=config.max_seq_len,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if verbose:
        print(f"Device: {device}")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        print()

    # Optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        if verbose:
            print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")

        # Move data to device
        def move_batch(batch):
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

        # Wrap dataloaders to move to device
        train_iter = (move_batch(b) for b in train_loader)

        # Train
        train_metrics = train_epoch(
            model, list(train_iter), optimizer, criterion, config, epoch, verbose
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])

        # Validate
        val_iter = (move_batch(b) for b in val_loader)
        val_metrics = validate(model, list(val_iter), criterion)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if verbose:
            print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0 or val_metrics["loss"] < best_val_loss:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "config": {
                    "num_action_tokens": num_action_tokens,
                    "d_model": 256,
                    "num_heads": 4,
                    "num_layers": 4,
                    "max_seq_len": config.max_seq_len,
                },
            }

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(checkpoint, output_dir / "best_model.pt")
                if verbose:
                    print(f"  Saved best model (val_loss={val_metrics['loss']:.4f})")

            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")

    elapsed = time.time() - start_time

    # Save training summary
    summary = {
        "status": "completed",
        "total_epochs": config.num_epochs,
        "total_time_s": elapsed,
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0,
        "history": history,
    }

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print()
        print("=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Total time: {elapsed:.1f}s")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Final train acc: {summary['final_train_acc']:.4f}")
        print(f"Final val acc: {summary['final_val_acc']:.4f}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CtRL-Sim-style BehaviourModel"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to exported dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/behaviour_model",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--tilt-regularization",
        type=float,
        default=0.0,
        help="Tilt regularization weight (0 = disabled)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tilt_regularization=args.tilt_regularization,
        seed=args.seed,
    )

    try:
        train_behaviour_model(config, verbose=not args.quiet)
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
