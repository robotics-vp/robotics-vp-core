#!/usr/bin/env python3
"""Train MotionHierarchyNode on a synthetic chain dataset."""
from __future__ import annotations

import argparse

from src.vision.motion_hierarchy.trainer import run_synthetic_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MotionHierarchyNode on synthetic data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device string")

    args = parser.parse_args()
    run_synthetic_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
