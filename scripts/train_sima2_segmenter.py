"""
Train Neural SIMA-2 Segmenter.

Bootstraps from heuristic segmenter outputs:
- Loads pseudo-labels from heuristic segmentation
- Trains boundary detector + primitive classifier
- Logs precision/recall/F1 per epoch
- Activates neural segmenter when F1 ≥ 85%
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Cannot train neural segmenter.")
    sys.exit(1)

from src.sima2.segmenter_nn import (
    NeuralSegmenter,
    compute_segmentation_loss,
    compute_f1_score,
    save_checkpoint,
)
from src.utils.json_safe import to_json_safe


class SIMA2SegmentationDataset(Dataset):
    """
    Dataset for SIMA-2 segmentation training.

    Fields (from heuristic segmenter):
    - image: [C, H, W]
    - boundary_mask: [1, H, W]
    - primitive_id: int
    - tags: List[str]
    - task_id, episode_id, timestep: metadata
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        # For now, generate synthetic data (in real usage, load from heuristic segmenter outputs)
        np.random.seed(42 if split == "train" else 43)
        self.num_samples = 1000 if split == "train" else 200
        self.samples = self._generate_synthetic_samples()

    def _generate_synthetic_samples(self) -> List[Dict[str, Any]]:
        """Generate synthetic training data."""
        samples = []

        for i in range(self.num_samples):
            # Synthetic image
            image = np.random.randn(3, self.image_size, self.image_size).astype(np.float32)

            # Synthetic boundary mask (random blobs)
            mask = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
            num_blobs = np.random.randint(1, 5)
            for _ in range(num_blobs):
                cx = np.random.randint(0, self.image_size)
                cy = np.random.randint(0, self.image_size)
                radius = np.random.randint(10, 30)
                y, x = np.ogrid[:self.image_size, :self.image_size]
                blob_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
                mask[0][blob_mask] = 1.0

            # Primitive class
            primitive_id = np.random.randint(0, 10)

            samples.append({
                "image": image,
                "boundary_mask": mask,
                "primitive_id": primitive_id,
                "task_id": f"task_{i % 5}",
                "episode_id": f"episode_{i}",
                "timestep": i,
                "tags": ["synthetic"],
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        return {
            "image": torch.from_numpy(sample["image"]),
            "boundary_mask": torch.from_numpy(sample["boundary_mask"]),
            "primitive_id": torch.tensor(sample["primitive_id"], dtype=torch.long),
            "task_id": sample["task_id"],
            "episode_id": sample["episode_id"],
            "timestep": sample["timestep"],
        }


def train_epoch(
    model: NeuralSegmenter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 0.25,
    gamma: float = 2.0,
    lambda_boundary: float = 1.0,
    lambda_primitive: float = 0.5,
    use_dice: bool = True,
    log_file: Path = None,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    total_boundary_loss = 0.0
    total_primitive_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        boundary_masks = batch["boundary_mask"].to(device)
        primitive_ids = batch["primitive_id"].to(device)

        # Forward pass
        outputs = model(images)

        # Compute loss
        losses = compute_segmentation_loss(
            boundary_logits=outputs["boundary_logits"],
            boundary_targets=boundary_masks,
            primitive_logits=outputs["primitive_logits"],
            primitive_targets=primitive_ids,
            alpha=alpha,
            gamma=gamma,
            lambda_boundary=lambda_boundary,
            lambda_primitive=lambda_primitive,
            use_dice=use_dice,
        )

        loss = losses["total_loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        total_loss += loss.item()
        total_boundary_loss += losses["boundary_loss"].item()
        total_primitive_loss += losses["primitive_loss"].item()
        num_batches += 1

        # Log to JSON lines
        if log_file is not None and batch_idx % 10 == 0:
            log_entry = {
                "event": "train_step",
                "batch": batch_idx,
                "loss": float(loss.item()),
                "boundary_loss": float(losses["boundary_loss"].item()),
                "focal_loss": float(losses["focal_loss"].item()),
                "dice_loss": float(losses["dice_loss"].item()),
                "primitive_loss": float(losses["primitive_loss"].item()),
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(to_json_safe(log_entry)) + "\n")

    # Average losses
    avg_losses = {
        "total_loss": total_loss / max(1, num_batches),
        "boundary_loss": total_boundary_loss / max(1, num_batches),
        "primitive_loss": total_primitive_loss / max(1, num_batches),
    }

    return avg_losses


def evaluate(
    model: NeuralSegmenter,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    all_precisions = []
    all_recalls = []
    all_f1s = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            boundary_masks = batch["boundary_mask"].to(device)

            outputs = model(images)

            precision, recall, f1 = compute_f1_score(
                outputs["boundary_logits"],
                boundary_masks,
                threshold=threshold,
            )

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

    avg_metrics = {
        "precision": float(np.mean(all_precisions)),
        "recall": float(np.mean(all_recalls)),
        "f1": float(np.mean(all_f1s)),
    }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Neural SIMA-2 Segmenter")
    parser.add_argument("--data_dir", type=str, default="data/sima2_segmentation", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sima2_segmenter", help="Output directory")
    parser.add_argument("--num_primitives", type=int, default=10, help="Number of primitive classes")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--lambda_boundary", type=float, default=1.0, help="Boundary loss weight")
    parser.add_argument("--lambda_primitive", type=float, default=0.5, help="Primitive loss weight")
    parser.add_argument("--use_dice", action="store_true", help="Use Dice loss")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder")
    parser.add_argument("--f1_threshold", type=float, default=0.85, help="F1 threshold for activation")
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

    # Datasets
    train_dataset = SIMA2SegmentationDataset(
        data_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
    )
    val_dataset = SIMA2SegmentationDataset(
        data_dir=args.data_dir,
        split="val",
        image_size=args.image_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Model
    model = NeuralSegmenter(
        in_channels=3,
        num_primitives=args.num_primitives,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training config
    config = {
        "num_primitives": args.num_primitives,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "lambda_boundary": args.lambda_boundary,
        "lambda_primitive": args.lambda_primitive,
        "use_dice": args.use_dice,
        "freeze_encoder": args.freeze_encoder,
        "f1_threshold": args.f1_threshold,
    }

    # Training loop
    best_f1 = 0.0
    activation_epoch = None

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_losses = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=args.alpha,
            gamma=args.gamma,
            lambda_boundary=args.lambda_boundary,
            lambda_primitive=args.lambda_primitive,
            use_dice=args.use_dice,
            log_file=log_file,
        )

        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print(f"  Train Loss: {train_losses['total_loss']:.6f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")

        # Check if F1 threshold met
        if val_metrics["f1"] >= args.f1_threshold and activation_epoch is None:
            activation_epoch = epoch + 1
            print(f"  *** F1 threshold ({args.f1_threshold}) met! Neural segmenter ready for activation. ***")

        # Track best F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        metrics = {**train_losses, **val_metrics}
        save_checkpoint(
            model=model,
            epoch=epoch + 1,
            config=config,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            seed=args.seed,
        )

        # Log epoch summary
        epoch_log = {
            "event": "epoch_complete",
            "epoch": epoch + 1,
            "train_loss": float(train_losses["total_loss"]),
            "val_f1": float(val_metrics["f1"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "best_f1": float(best_f1),
            "activation_ready": val_metrics["f1"] >= args.f1_threshold,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(to_json_safe(epoch_log)) + "\n")

    print(f"\nTraining complete!")
    print(f"Best F1: {best_f1:.4f}")
    if activation_epoch is not None:
        print(f"Neural segmenter activated at epoch {activation_epoch}")
    else:
        print(f"F1 threshold ({args.f1_threshold}) not met. Continue training or lower threshold.")
    print(f"Logs saved to {log_file}")


if __name__ == "__main__":
    main()
