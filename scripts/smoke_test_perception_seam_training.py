#!/usr/bin/env python3
"""Proof-of-life smoke test for the perception seam training pipeline.

Exercises the full path: synthetic data → EvidenceFusionSeam → trainer →
validation → benchmark gate → metrics artifact (JSON receipts).

This script does NOT require GPU, real providers, or external datasets.
It uses the synthetic data generators to verify that every component in the
training pipeline compiles, runs forward/backward, emits receipts, and
produces a structured artifact.

Usage::

    python3 scripts/smoke_test_perception_seam_training.py [--steps N] [--artifact-dir DIR]

Produces::

    artifacts/perception_seam_proof_of_life.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.perception_seam_data import (
    generate_synthetic_evidence_fusion_samples,
    create_evidence_fusion_loader,
)
from src.training.perception_seam_trainer import (
    PerceptionSeamTrainer,
    SeamTrainingConfig,
)
from src.training.perception_seam_benchmarks import (
    EvidenceFusionBenchmark,
)
from src.world_model.perception_grounding.seam_registry import (
    PerceptionSeamRegistry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger("perception_seam_proof_of_life")


def run_proof_of_life(
    max_steps: int = 30,
    artifact_dir: str | Path | None = None,
) -> dict:
    """Run end-to-end proof-of-life for EvidenceFusionSeam training.

    Returns:
        Summary dict written to the artifact JSON.
    """
    artifact_dir = Path(artifact_dir or REPO_ROOT / "artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "perception_seam_proof_of_life.json"

    t0 = time.time()

    # ── 1. Generate synthetic data ──────────────────────────────────────
    logger.info("Generating synthetic multi-provider samples...")
    all_samples = generate_synthetic_evidence_fusion_samples(
        n_samples=200, n_providers=4, d_feature=128, seed=42,
    )
    # 80/20 train/val split
    split = int(0.8 * len(all_samples))
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]

    train_loader = create_evidence_fusion_loader(
        train_samples, batch_size=16, shuffle=True, d_feature=128,
    )
    val_loader = create_evidence_fusion_loader(
        val_samples, batch_size=16, shuffle=False, d_feature=128,
    )
    logger.info(
        f"  train={len(train_samples)} samples / {len(train_loader)} batches, "
        f"val={len(val_samples)} samples / {len(val_loader)} batches"
    )

    # ── 2. Set up registry + seam ───────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="seam_ckpt_") as ckpt_dir:
        registry = PerceptionSeamRegistry(checkpoint_dir=ckpt_dir)
        registry.register_seam(
            seam_type="evidence_fusion",
            seam_id="fusion_proof_of_life",
            posture="auto",
        )
        seam = registry.load_seam("fusion_proof_of_life")
        param_count = sum(p.numel() for p in seam.parameters() if p.requires_grad)
        logger.info(f"  EvidenceFusionSeam loaded: {param_count:,} trainable params")

        # ── 3. Configure and run trainer ────────────────────────────────
        config = SeamTrainingConfig(
            learning_rate=3e-4,
            weight_decay=0.01,
            batch_size=16,
            max_steps=max_steps,
            max_epochs=100,  # won't hit this — max_steps stops us first
            gradient_accumulation_steps=1,
            val_check_interval=10,
            checkpoint_interval=max_steps,  # checkpoint at end
            early_stopping_patience=999,  # don't early-stop in smoke
            benchmark_gate_interval=max_steps,  # gate at end
            promotion_threshold=0.6,
            log_interval=5,
            device="cpu",
        )

        trainer = PerceptionSeamTrainer(
            registry=registry,
            seam_id="fusion_proof_of_life",
            config=config,
        )

        logger.info(f"Starting training for {max_steps} steps...")
        summary = trainer.fit(train_loader, val_loader)
        logger.info("Training complete.")

        # ── 4. Run standalone benchmark evaluation ──────────────────────
        benchmark = EvidenceFusionBenchmark()
        bench_result = benchmark.evaluate(seam, val_loader)
        logger.info(
            f"Benchmark gate: score={bench_result.overall_score:.4f}, "
            f"passed={bench_result.overall_passed}"
        )

        # ── 5. Assemble and write artifact ──────────────────────────────
        elapsed = time.time() - t0

        artifact = {
            "proof_of_life": True,
            "seam_type": "evidence_fusion",
            "param_count": param_count,
            "max_steps": max_steps,
            "training_summary": summary,
            "benchmark": {
                "overall_score": bench_result.overall_score,
                "overall_passed": bench_result.overall_passed,
                "promotion_decision": bench_result.promotion_decision,
                "metrics": [
                    {"name": m.name, "value": m.value, "threshold": m.threshold, "passed": m.passed}
                    for m in bench_result.metrics
                ],
            },
            "receipts": {
                "training_count": len(trainer.training_receipts),
                "validation_count": len(trainer.validation_receipts),
                "benchmark_count": len(trainer.benchmark_receipts),
                "sample_training_receipt": (
                    trainer.training_receipts[-1].to_dict()
                    if trainer.training_receipts else None
                ),
                "sample_validation_receipt": (
                    trainer.validation_receipts[-1].to_dict()
                    if trainer.validation_receipts else None
                ),
                "sample_benchmark_receipt": (
                    trainer.benchmark_receipts[-1].to_dict()
                    if trainer.benchmark_receipts else None
                ),
            },
            "data": {
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "data_source": "synthetic (generate_synthetic_evidence_fusion_samples)",
            },
            "elapsed_seconds": round(elapsed, 2),
            "device": config.device,
            "torch_version": torch.__version__,
        }

        # Also save full receipts alongside
        receipts_path = artifact_dir / "perception_seam_proof_of_life_receipts.json"
        trainer.save_receipts(receipts_path)

        artifact_path.write_text(json.dumps(artifact, indent=2, default=str))
        logger.info(f"Artifact written to {artifact_path}")
        logger.info(f"Full receipts written to {receipts_path}")
        logger.info(f"Elapsed: {elapsed:.1f}s")

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perception seam training proof-of-life smoke test",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Number of training steps (default: 30)",
    )
    parser.add_argument(
        "--artifact-dir", type=str, default=None,
        help="Output directory for artifacts (default: artifacts/)",
    )
    args = parser.parse_args()

    artifact = run_proof_of_life(
        max_steps=args.steps,
        artifact_dir=args.artifact_dir,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PROOF-OF-LIFE SUMMARY")
    print("=" * 60)
    print(f"  Seam type:      {artifact['seam_type']}")
    print(f"  Param count:    {artifact['param_count']:,}")
    print(f"  Steps:          {artifact['training_summary']['total_steps']}")
    print(f"  Final val loss: {artifact['training_summary']['best_val_loss']:.4f}")
    print(f"  Benchmark:      score={artifact['benchmark']['overall_score']:.4f} "
          f"passed={artifact['benchmark']['overall_passed']} "
          f"decision={artifact['benchmark']['promotion_decision']}")
    print(f"  Receipts:       {artifact['receipts']['training_count']} train, "
          f"{artifact['receipts']['validation_count']} val, "
          f"{artifact['receipts']['benchmark_count']} benchmark")
    print(f"  Elapsed:        {artifact['elapsed_seconds']}s")
    print(f"  Artifact:       artifacts/perception_seam_proof_of_life.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
