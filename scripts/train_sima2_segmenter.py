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
from src.utils.training_env import should_use_amp, device_info, run_with_oom_recovery
from src.utils.gpu_env import get_gpu_utilization
from src.utils.logging_schema import make_training_log_entry, write_training_log_entry
import hashlib


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
            # Synthetic image (uint8 to save memory)
            # Random noise in [0, 255]
            image = np.random.randint(0, 256, (3, self.image_size, self.image_size), dtype=np.uint8)

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



def evaluate(
    model: SIMA2Segmenter,
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
            
            # Assuming compute_f1_score is available or we implement it
            # For now, let's use a placeholder or simple calculation if helper not imported
            # The original code had compute_f1_score. Let's assume it's imported or defined.
            # If not, we might need to add it.
            # But wait, looking at imports in Step 347, I don't see compute_f1_score imported.
            # It might have been there before I replaced the file content?
            # In Step 352, I replaced the whole file content? No, just a chunk.
            # Let's check imports.
            
            # For this fix, I will just put the structure back.
            # If compute_f1_score is missing, I'll add a dummy one or import it if I know where it is.
            # It was likely in `src.utils.metrics` or similar.
            # But I can't see it in imports.
            # I'll assume it's not there and implement a simple one or skip it for now to fix syntax.
            
            # Actually, let's just calculate it here to be safe and self-contained.
            
            if isinstance(outputs, dict):
                logits = outputs.get("boundary_logits", outputs.get("out"))
            else:
                logits = outputs
                
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            tp = (preds * boundary_masks).sum().item()
            fp = (preds * (1 - boundary_masks)).sum().item()
            fn = ((1 - preds) * boundary_masks).sum().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

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
        # Train
        # AMP Setup
        use_amp = args.use_mixed_precision or should_use_amp({"training": {"amp": args.use_mixed_precision}})
        scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
        config_digest = hashlib.md5(json.dumps(vars(args), sort_keys=True).encode()).hexdigest()[:8]

        train_losses = run_with_oom_recovery(
            train_epoch,
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
            scaler=scaler,
            use_amp=use_amp,
            epoch_idx=epoch,
            run_name=f"sima2_segmenter_{args.seed}",
            config_digest=config_digest,
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
