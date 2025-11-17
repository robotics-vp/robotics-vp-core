#!/usr/bin/env python3
"""
Train Orchestration Transformer on heuristic teacher sequences.

Supervised learning: imitate the heuristic policy that maps
(context_features) -> tool_sequence.

No Phase B/RL integration - purely advisory orchestration.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.orchestrator.orchestration_transformer import (
    OrchestrationTransformer,
    TOOL_NAMES,
    _encode_ctx,
)
from src.orchestrator.training_dataset import (
    build_training_dataset,
    dataset_to_tensors,
    save_dataset,
)


def create_dummy_instruction_tokens(batch_size: int, vocab_size: int = 128, seq_len: int = 8):
    """
    Create dummy instruction tokens for training.

    In full implementation, these would be tokenized natural language instructions.
    For now, we use random tokens as placeholders.
    """
    return torch.randint(1, vocab_size, (batch_size, seq_len))


def train_epoch(
    model: OrchestrationTransformer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    vocab_size: int = 128,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_ctx, batch_tools in dataloader:
        batch_size = batch_ctx.shape[0]

        # Create dummy instruction tokens (would be real in full implementation)
        instr_tokens = create_dummy_instruction_tokens(batch_size, vocab_size)

        optimizer.zero_grad()

        # Forward pass
        tool_logits, arg_vec = model(instr_tokens, batch_ctx)

        # Compute loss for first tool prediction only (simplification)
        # In full implementation, we'd predict entire sequence
        target_first_tool = batch_tools[:, 0]
        loss = criterion(tool_logits, target_first_tool)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        pred = torch.argmax(tool_logits, dim=-1)
        correct += (pred == target_first_tool).sum().item()
        total += batch_size

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader)

    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate(
    model: OrchestrationTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    vocab_size: int = 128,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_ctx, batch_tools in dataloader:
            batch_size = batch_ctx.shape[0]
            instr_tokens = create_dummy_instruction_tokens(batch_size, vocab_size)

            tool_logits, arg_vec = model(instr_tokens, batch_ctx)
            target_first_tool = batch_tools[:, 0]
            loss = criterion(tool_logits, target_first_tool)

            total_loss += loss.item()
            pred = torch.argmax(tool_logits, dim=-1)
            correct += (pred == target_first_tool).sum().item()
            total += batch_size

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader)

    return {"loss": avg_loss, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Train Orchestration Transformer")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=96, help="Hidden dimension")
    parser.add_argument("--ctx-dim", type=int, default=36, help="Context feature dimension")
    parser.add_argument("--vocab-size", type=int, default=128, help="Instruction vocabulary size")
    parser.add_argument("--save-dir", type=str, default="checkpoints/orchestrator", help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Orchestration Transformer Training")
    print("=" * 60)
    print(f"Samples: {args.num_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden dim: {args.hidden}")
    print(f"Context dim: {args.ctx_dim}")
    print("=" * 60)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate training dataset
    print("\nGenerating training dataset with heuristic teacher...")
    samples = build_training_dataset(num_samples=args.num_samples)
    print(f"Generated {len(samples)} samples")

    # Convert to tensors
    X, Y, tool_names = dataset_to_tensors(samples)
    print(f"Context features shape: {X.shape}")
    print(f"Target tools shape: {Y.shape}")

    # Adjust context dimension if needed
    actual_ctx_dim = X.shape[1]
    if actual_ctx_dim != args.ctx_dim:
        print(f"Adjusting ctx_dim from {args.ctx_dim} to {actual_ctx_dim}")
        args.ctx_dim = actual_ctx_dim

    # Train/val split
    num_val = int(len(samples) * args.val_split)
    num_train = len(samples) - num_val

    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).long()

    train_dataset = TensorDataset(X_tensor[:num_train], Y_tensor[:num_train])
    val_dataset = TensorDataset(X_tensor[num_train:], Y_tensor[num_train:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {num_train}, Val samples: {num_val}")

    # Create model
    model = OrchestrationTransformer(
        vocab_size=args.vocab_size,
        hidden=args.hidden,
        ctx_dim=args.ctx_dim,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\nTraining...")
    best_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, args.vocab_size)
        val_metrics = evaluate(model, val_loader, criterion, args.vocab_size)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            }
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            # Save best model
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f}"
            )

    # Save final model
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    print(f"\nSaved final model to {save_dir / 'final_model.pt'}")

    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {save_dir / 'training_history.json'}")

    # Save dataset for reference
    dataset_path = save_dir / "training_dataset.json"
    save_dataset(samples, str(dataset_path))

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Final train accuracy: {history[-1]['train_acc']:.3f}")
    print(f"Final val accuracy: {history[-1]['val_acc']:.3f}")
    print(f"\nCheckpoint saved to: {save_dir / 'best_model.pt'}")

    # Show tool distribution
    print("\nTool distribution in training data:")
    tool_counts = {name: 0 for name in TOOL_NAMES}
    for seq in tool_names:
        for tool in seq:
            tool_counts[tool] += 1
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()
