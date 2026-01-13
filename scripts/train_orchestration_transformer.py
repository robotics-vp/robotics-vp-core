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
import sys
from pathlib import Path

# Regality wrapper
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from src.training.wrap_training_entrypoint import regal_training

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
    build_mixed_training_dataset,
    split_dataset_by_source,
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


def evaluate_subset(
    model: OrchestrationTransformer,
    samples,
    criterion: nn.Module,
    vocab_size: int = 128,
    batch_size: int = 32,
) -> dict:
    """Evaluate on a specific subset of samples."""
    if len(samples) == 0:
        return {"loss": 0.0, "accuracy": 0.0, "count": 0}

    X, Y, _ = dataset_to_tensors(samples)
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).long()
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return evaluate(model, loader, criterion, vocab_size)


@regal_training(env_type="workcell")
def main(runner=None):
    """Main entrypoint with regality wrapper."""
    if runner:
        runner.start_training()
    
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
    parser.add_argument("--use-mixed-dataset", action="store_true", help="Use mixed heuristic + econ/semantic dataset")
    parser.add_argument("--econ-semantic-ratio", type=float, default=0.5, help="Fraction of samples from econ/semantic (when using mixed)")
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
    if args.use_mixed_dataset:
        print("\nGenerating MIXED dataset (heuristic + econ/semantic)...")
        num_econ_semantic = int(args.num_samples * args.econ_semantic_ratio)
        num_heuristic = args.num_samples - num_econ_semantic
        samples, dataset_stats = build_mixed_training_dataset(
            num_heuristic=num_heuristic,
            num_econ_semantic=num_econ_semantic,
        )
        print(f"Generated {len(samples)} samples")
        print(f"  - Heuristic: {dataset_stats['heuristic_count']}")
        print(f"  - Econ/Semantic: {dataset_stats['econ_semantic_count']}")
        if "profile_distribution" in dataset_stats:
            print(f"  - Profile dist: {dataset_stats['profile_distribution']}")
            print(f"  - Preset dist: {dataset_stats['preset_distribution']}")

        # Save dataset stats
        with open(save_dir / "dataset_stats.json", "w") as f:
            json.dump(dataset_stats, f, indent=2)
    else:
        print("\nGenerating training dataset with heuristic teacher...")
        samples = build_training_dataset(num_samples=args.num_samples)
        print(f"Generated {len(samples)} samples")
        dataset_stats = None

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

    # Evaluate separately on heuristic vs econ/semantic subsets (if using mixed dataset)
    if args.use_mixed_dataset:
        print("\n" + "=" * 60)
        print("Subset Performance Analysis")
        print("=" * 60)

        # Load best model for evaluation
        model.load_state_dict(torch.load(save_dir / "best_model.pt"))

        # Split validation samples by source type
        heuristic_samples, econ_semantic_samples = split_dataset_by_source(samples)

        # Evaluate on heuristic samples
        if heuristic_samples:
            heur_metrics = evaluate_subset(model, heuristic_samples, criterion, args.vocab_size, args.batch_size)
            print(f"Heuristic samples ({len(heuristic_samples)}):")
            print(f"  Accuracy: {heur_metrics['accuracy']:.3f}")
            print(f"  Loss: {heur_metrics['loss']:.4f}")

        # Evaluate on econ/semantic samples
        if econ_semantic_samples:
            econ_metrics = evaluate_subset(model, econ_semantic_samples, criterion, args.vocab_size, args.batch_size)
            print(f"Econ/Semantic samples ({len(econ_semantic_samples)}):")
            print(f"  Accuracy: {econ_metrics['accuracy']:.3f}")
            print(f"  Loss: {econ_metrics['loss']:.4f}")

            # Check if model tends to pick right profiles in specific regimes
            print("\nProfile selection analysis (econ/semantic samples):")
            correct_by_urgency = {}
            for sample in econ_semantic_samples:
                if sample.econ_semantic_summary:
                    urg = sample.econ_semantic_summary.urgency_level
                    if urg not in correct_by_urgency:
                        correct_by_urgency[urg] = {"total": 0, "safe_profile": 0}
                    correct_by_urgency[urg]["total"] += 1
                    if sample.econ_semantic_summary.chosen_profile == "SAFE":
                        correct_by_urgency[urg]["safe_profile"] += 1

            for urg_level in ["critical", "high", "moderate", "none"]:
                if urg_level in correct_by_urgency:
                    data = correct_by_urgency[urg_level]
                    safe_pct = 100 * data["safe_profile"] / data["total"] if data["total"] > 0 else 0
                    print(f"  {urg_level}: {data['total']} samples, {safe_pct:.1f}% use SAFE profile")

        # Save subset metrics
        subset_metrics = {
            "heuristic": heur_metrics if heuristic_samples else {},
            "econ_semantic": econ_metrics if econ_semantic_samples else {},
        }
        with open(save_dir / "subset_metrics.json", "w") as f:
            json.dump(subset_metrics, f, indent=2)
        print(f"\nSaved subset metrics to {save_dir / 'subset_metrics.json'}")

    if runner:
        runner.update_step(args.epochs * 100)  # Approximate


if __name__ == "__main__":
    main()
