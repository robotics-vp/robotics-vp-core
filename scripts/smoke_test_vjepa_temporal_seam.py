#!/usr/bin/env python3
"""Proof-of-life smoke test for the V-JEPA temporal seam training pipeline.

Exercises the full path: temporal samples -> VJEPATemporalAlignmentSeam ->
trainer -> benchmark gate -> typed proof artifacts.

This script does NOT require GPU, real providers, or external datasets.
It uses synthetic temporal samples or DROID-shaped mock LeRobot episodes to
verify that the local training pipeline compiles, runs forward/backward, emits
receipts, and produces structured artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.perception_proof_of_life_utils import (
    load_lerobot_episodes_from_path,
    make_mock_lerobot_episode,
)
from src.dataset_bridges.lerobot_perception_adapter import (
    LeRobotPerceptionAdapterConfig,
    adapt_lerobot_episodes_for_vjepa_temporal,
)
from src.training.perception_seam_benchmarks import VJEPATemporalBenchmark
from src.training.perception_seam_data import (
    VJEPATemporalSample,
    create_vjepa_temporal_loader,
    generate_synthetic_vjepa_temporal_samples,
)
from src.training.perception_seam_trainer import (
    PerceptionSeamTrainer,
    SeamTrainingConfig,
)
from src.training.training_manifest import (
    TRAINING_RUNTIME_MANIFEST_SCHEMA_VERSION,
    TrainingRuntimeManifest,
    write_training_runtime_manifest,
)
from src.utils.config_digest import sha256_file, sha256_json
from src.world_model.perception_grounding.benchmark_evidence import (
    build_perception_benchmark_evidence,
    write_perception_benchmark_evidence,
)
from src.world_model.perception_grounding.seam_registry import PerceptionSeamRegistry


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger("vjepa_temporal_seam_proof_of_life")


def _load_temporal_samples(
    *,
    data_source: str,
    seed: int,
    lerobot_rows_path: str | Path | None = None,
    max_episodes: int = 4,
    max_steps_per_episode: int | None = None,
) -> tuple[list[VJEPATemporalSample], dict[str, object]]:
    if data_source == "synthetic":
        samples = generate_synthetic_vjepa_temporal_samples(
            n_samples=120,
            n_temporal_steps=4,
            n_objects=10,
            d_vjepa=1024,
            d_wm=128,
            d_out=128,
            seed=seed,
        )
        return samples, {
            "dataset_kind": "synthetic_vjepa_temporal",
            "data_source": "synthetic",
            "source_domain_counts": {"synthetic_vjepa_temporal": len(samples)},
            "temporal_window_size": 4,
            "provider_count": 1,
            "mock_data": False,
        }
    if data_source == "mock_lerobot_droid":
        episodes = [
            make_mock_lerobot_episode(
                episode_idx=episode_idx,
                num_steps=max_steps_per_episode or 24,
                seed=seed,
                camera_format="droid",
            )
            for episode_idx in range(max_episodes)
        ]
        adapter_config = LeRobotPerceptionAdapterConfig(
            temporal_window_size=4,
            temporal_stride=4,
            n_objects=10,
            d_vjepa=1024,
            d_wm=128,
            d_out=128,
        )
        samples = adapt_lerobot_episodes_for_vjepa_temporal(episodes, adapter_config)
        return samples, {
            "dataset_kind": "mock_lerobot_droid_vjepa_temporal",
            "data_source": "mock_lerobot_droid",
            "source_domain_counts": {"mock_lerobot_droid": len(samples)},
            "episode_count": len(episodes),
            "steps_per_episode": 24,
            "temporal_window_size": 4,
            "temporal_stride": 4,
            "provider_count": 3,
            "mock_data": True,
            "not_external_dataset": True,
        }
    if data_source == "local_lerobot_rows":
        if lerobot_rows_path is None:
            raise ValueError(
                "lerobot_rows_path is required when data_source=local_lerobot_rows"
            )
        episodes = load_lerobot_episodes_from_path(
            lerobot_rows_path,
            max_episodes=max_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )
        adapter_config = LeRobotPerceptionAdapterConfig(
            temporal_window_size=4,
            temporal_stride=4,
            n_objects=10,
            d_vjepa=1024,
            d_wm=128,
            d_out=128,
        )
        samples = adapt_lerobot_episodes_for_vjepa_temporal(episodes, adapter_config)
        source_domain_counts: dict[str, int] = {}
        for episode, _steps in episodes:
            source_domain_counts[episode.source_domain] = (
                source_domain_counts.get(episode.source_domain, 0) + 1
            )
        return samples, {
            "dataset_kind": "local_lerobot_rows_vjepa_temporal",
            "data_source": "local_lerobot_rows",
            "source_domain_counts": source_domain_counts,
            "episode_count": len(episodes),
            "max_steps_per_episode": max_steps_per_episode,
            "temporal_window_size": 4,
            "temporal_stride": 4,
            "provider_count": 0,
            "mock_data": False,
            "not_external_dataset": False,
            "lerobot_rows_path": str(Path(lerobot_rows_path).resolve()),
        }
    raise ValueError(f"unsupported data_source={data_source!r}")


def run_proof_of_life(
    max_steps: int = 40,
    artifact_dir: str | Path | None = None,
    *,
    seed: int = 42,
    data_source: str = "synthetic",
    lerobot_rows_path: str | Path | None = None,
    max_episodes: int = 4,
    max_steps_per_episode: int | None = None,
) -> dict:
    artifact_dir = Path(artifact_dir or REPO_ROOT / "artifacts").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "vjepa_temporal_seam_proof_of_life.json"
    metric_report_path = artifact_dir / "vjepa_temporal_metric_report.json"
    benchmark_evidence_path = artifact_dir / "vjepa_temporal_benchmark_evidence.json"
    receipts_path = artifact_dir / "vjepa_temporal_proof_of_life_receipts.json"
    registry_summary_path = artifact_dir / "vjepa_temporal_registry_summary.json"
    manifest_path = artifact_dir / "training_runtime_manifest.json"
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(seed)

    logger.info("Preparing %s temporal samples...", data_source)
    all_samples, data_summary = _load_temporal_samples(
        data_source=data_source,
        seed=seed,
        lerobot_rows_path=lerobot_rows_path,
        max_episodes=max_episodes,
        max_steps_per_episode=max_steps_per_episode,
    )
    split = int(0.8 * len(all_samples))
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]

    train_loader = create_vjepa_temporal_loader(
        train_samples,
        batch_size=8,
        shuffle=True,
        n_temporal_steps=4,
        max_objects=32,
    )
    val_loader = create_vjepa_temporal_loader(
        val_samples,
        batch_size=8,
        shuffle=False,
        n_temporal_steps=4,
        max_objects=32,
    )
    logger.info(
        "  train=%s samples / %s batches, val=%s samples / %s batches",
        len(train_samples),
        len(train_loader),
        len(val_samples),
        len(val_loader),
    )

    registry = PerceptionSeamRegistry(checkpoint_dir=checkpoint_dir)
    registry.register_seam(
        seam_type="vjepa_temporal_alignment",
        seam_id="vjepa_temporal_proof_of_life",
        posture="auto",
    )
    seam = registry.load_seam("vjepa_temporal_proof_of_life")
    param_count = sum(p.numel() for p in seam.parameters() if p.requires_grad)
    logger.info(
        "  VJEPATemporalAlignmentSeam loaded: %s trainable params", f"{param_count:,}"
    )

    config = SeamTrainingConfig(
        learning_rate=3e-4,
        weight_decay=0.01,
        batch_size=8,
        max_steps=max_steps,
        max_epochs=100,
        gradient_accumulation_steps=1,
        val_check_interval=10,
        checkpoint_interval=max_steps,
        early_stopping_patience=999,
        benchmark_gate_interval=max_steps,
        promotion_threshold=0.6,
        log_interval=10,
        lr_scheduler="none",
        warmup_steps=0,
        device="cpu",
    )
    trainer = PerceptionSeamTrainer(
        registry=registry,
        seam_id="vjepa_temporal_proof_of_life",
        config=config,
    )
    initial_val_loss, initial_val_components, initial_val_metrics = trainer._validate(
        val_loader
    )
    logger.info(
        "Starting training for %s steps (initial_val_loss=%.4f)...",
        max_steps,
        initial_val_loss,
    )
    summary = trainer.fit(train_loader, val_loader)
    final_val_loss, final_val_components, final_val_metrics = trainer._validate(
        val_loader
    )
    best_val_loss = min(float(summary["best_val_loss"]), final_val_loss)
    loss_improvement = initial_val_loss - best_val_loss
    loss_improvement_fraction = loss_improvement / max(abs(initial_val_loss), 1e-9)
    loss_decreased = loss_improvement > 0.0

    benchmark = VJEPATemporalBenchmark()
    bench_result = benchmark.evaluate(seam, val_loader)
    logger.info(
        "Benchmark gate: score=%.4f, passed=%s",
        bench_result.overall_score,
        bench_result.overall_passed,
    )

    elapsed = time.time() - t0
    ended_at = datetime.now(timezone.utc).isoformat()

    trainer.save_receipts(receipts_path)
    registry_summary_path.write_text(
        json.dumps(registry.summary(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    descriptor = registry.get_descriptor("vjepa_temporal_proof_of_life")
    checkpoint_path = Path(descriptor.checkpoint_path).resolve() if descriptor else None
    checkpoint_digest = (
        sha256_file(checkpoint_path)
        if checkpoint_path is not None and checkpoint_path.exists()
        else None
    )

    metric_report = {
        "schema_version": "perception_seam_metric_report_v1",
        "seam_type": "vjepa_temporal_alignment",
        "seam_id": "vjepa_temporal_proof_of_life",
        "run_kind": f"local_cpu_{data_source}_proof_of_life",
        "evidence_source_provisional": True,
        "promotion_eligible": False,
        "promotion_claim": "explicitly_held",
        "metrics": {
            "initial_val_loss": float(initial_val_loss),
            "final_val_loss": float(final_val_loss),
            "best_val_loss": float(best_val_loss),
            "loss_improvement": float(loss_improvement),
            "loss_improvement_fraction": float(loss_improvement_fraction),
            "loss_decreased": bool(loss_decreased),
            "benchmark_score": float(bench_result.overall_score),
            "benchmark_passed": bool(bench_result.overall_passed),
            "benchmark_promotion_decision": bench_result.promotion_decision,
            "training_steps": int(summary["total_steps"]),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "data_source": data_source,
        },
        "initial_validation": {
            "component_losses": initial_val_components,
            "metrics": initial_val_metrics,
        },
        "final_validation": {
            "component_losses": final_val_components,
            "metrics": final_val_metrics,
        },
    }
    metric_report_path.write_text(
        json.dumps(metric_report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    evidence_metrics = {
        "benchmark_evidence_present": True,
        "evidence_source_provisional": True,
        "evidence_truth_class": f"{data_source}_proof_of_life",
        "token_source_kind": f"{data_source}_temporal_windows",
        "source_record_count": len(val_samples),
        "annotation_supervision_score": max(0.0, min(1.0, loss_improvement_fraction)),
        "held_out_label_agreement": float(bench_result.overall_score),
        "downstream_usefulness_score": float(bench_result.overall_score),
        "receipt_consistency": 1.0 if trainer.training_receipts else 0.0,
        "gate_score": float(bench_result.overall_score),
        "promotion_eligible": False,
    }
    benchmark_evidence = build_perception_benchmark_evidence(
        subsystem_key="vjepa_temporal_alignment",
        metrics=evidence_metrics,
        source_record_count=len(val_samples),
        source_artifact_path=metric_report_path,
        metadata={
            "emitter": "vjepa_temporal_seam_proof_of_life_smoke",
            "metric_report_path": str(metric_report_path),
            "metric_report_digest": sha256_file(metric_report_path),
            "training_manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_digest": checkpoint_digest,
            "loss_decreased": bool(loss_decreased),
            "data_source": data_source,
            "promotion_claim": "not_implied_by_local_proof_of_life",
        },
    )
    benchmark_evidence_digest = write_perception_benchmark_evidence(
        benchmark_evidence_path,
        benchmark_evidence,
    )

    artifact_paths = {
        "proof_of_life_summary": str(artifact_path),
        "metric_report": str(metric_report_path),
        "benchmark_evidence": str(benchmark_evidence_path),
        "receipts": str(receipts_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "registry_summary": str(registry_summary_path),
    }
    config_snapshot = asdict(config)
    plan_snapshot = {
        "script": "scripts/smoke_test_vjepa_temporal_seam.py",
        "seam_type": "vjepa_temporal_alignment",
        "run_kind": f"local_cpu_{data_source}_proof_of_life",
        "max_steps": max_steps,
        "seed": seed,
        "data_source": data_source,
        "max_episodes": max_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "lerobot_rows_path": (
            str(Path(lerobot_rows_path).resolve()) if lerobot_rows_path else None
        ),
        "promotion_claim": "explicitly_held",
    }

    manifest = TrainingRuntimeManifest(
        schema_version=TRAINING_RUNTIME_MANIFEST_SCHEMA_VERSION,
        run_id=(
            f"vjepa_temporal_proof_of_life_{data_source}_seed_{seed}_steps_{max_steps}"
        ),
        training_kind="perception_vjepa_temporal_proof_of_life",
        status="completed",
        seed=seed,
        plan_id="phase2_vjepa_temporal_local_proof_of_life",
        plan_sha=sha256_json(plan_snapshot),
        started_at=started_at,
        ended_at=ended_at,
        config_path=None,
        config_digest=sha256_json(config_snapshot),
        replay_dataset_dir=None,
        replay_manifest_digest=None,
        replay_dataset_summary={
            **data_summary,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        },
        objective_profile_snapshot={
            "objective": "vjepa_temporal_alignment_loss",
            "device": config.device,
            "max_steps": max_steps,
        },
        promotion_policy_snapshot={
            "promotion_eligible": False,
            "promotion_claim": "explicitly_held",
            "reason": "local proof-of-life is provisional plumbing evidence",
        },
        source_domain_coverage={
            "total_records": len(all_samples),
            "source_domain_counts": dict(data_summary.get("source_domain_counts", {})),
        },
        receipt_label_coverage={
            "training_receipts": len(trainer.training_receipts),
            "validation_receipts": len(trainer.validation_receipts),
            "benchmark_receipts": len(trainer.benchmark_receipts),
        },
        inferential_learnability_summary={
            "initial_val_loss": float(initial_val_loss),
            "final_val_loss": float(final_val_loss),
            "best_val_loss": float(best_val_loss),
            "loss_improvement": float(loss_improvement),
            "loss_improvement_fraction": float(loss_improvement_fraction),
            "loss_decreased": bool(loss_decreased),
            "benchmark_score": float(bench_result.overall_score),
            "promotion_eligible": False,
        },
        artifact_paths=artifact_paths,
        checkpoint_registry_path=str(registry_summary_path),
        checkpoint_registry_digest=sha256_file(registry_summary_path),
        promotion_evidence_path=str(benchmark_evidence_path),
        promotion_evidence_digest=benchmark_evidence_digest,
        metadata={
            "epistemic_status": "proof_of_life",
            "execution_plane": "local",
            "gpu_required": False,
            "external_dataset_required": data_source == "local_lerobot_rows",
            "torch_version": torch.__version__,
            "data_source": data_source,
            "mock_data": bool(data_summary.get("mock_data", False)),
            "lerobot_rows_path": data_summary.get("lerobot_rows_path"),
        },
    )
    manifest_digest = write_training_runtime_manifest(manifest_path, manifest)

    artifact = {
        "schema_version": "perception_seam_proof_of_life_v2",
        "proof_of_life": True,
        "promotion_eligible": False,
        "promotion_claim": "explicitly_held",
        "seam_type": "vjepa_temporal_alignment",
        "param_count": param_count,
        "max_steps": max_steps,
        "seed": seed,
        "data_source": data_source,
        "training_summary": summary,
        "loss_proof": {
            "initial_val_loss": float(initial_val_loss),
            "final_val_loss": float(final_val_loss),
            "best_val_loss": float(best_val_loss),
            "loss_improvement": float(loss_improvement),
            "loss_improvement_fraction": float(loss_improvement_fraction),
            "loss_decreased": bool(loss_decreased),
        },
        "benchmark": {
            "overall_score": bench_result.overall_score,
            "overall_passed": bench_result.overall_passed,
            "promotion_decision": bench_result.promotion_decision,
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "threshold": metric.threshold,
                    "passed": metric.passed,
                }
                for metric in bench_result.metrics
            ],
        },
        "receipts": {
            "training_count": len(trainer.training_receipts),
            "validation_count": len(trainer.validation_receipts),
            "benchmark_count": len(trainer.benchmark_receipts),
            "sample_training_receipt": (
                trainer.training_receipts[-1].to_dict()
                if trainer.training_receipts
                else None
            ),
            "sample_validation_receipt": (
                trainer.validation_receipts[-1].to_dict()
                if trainer.validation_receipts
                else None
            ),
            "sample_benchmark_receipt": (
                trainer.benchmark_receipts[-1].to_dict()
                if trainer.benchmark_receipts
                else None
            ),
        },
        "data": {
            **data_summary,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        },
        "artifact_paths": {
            **artifact_paths,
            "training_manifest": str(manifest_path),
        },
        "artifact_digests": {
            "metric_report": sha256_file(metric_report_path),
            "benchmark_evidence": benchmark_evidence_digest,
            "receipts": sha256_file(receipts_path),
            "checkpoint": checkpoint_digest,
            "registry_summary": sha256_file(registry_summary_path),
            "training_manifest": manifest_digest,
        },
        "elapsed_seconds": round(elapsed, 2),
        "device": config.device,
        "torch_version": torch.__version__,
    }
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    logger.info("Artifact written to %s", artifact_path)
    logger.info("Metric report written to %s", metric_report_path)
    logger.info("Benchmark evidence written to %s", benchmark_evidence_path)
    logger.info("Training manifest written to %s", manifest_path)
    logger.info("Full receipts written to %s", receipts_path)
    logger.info("Elapsed: %.1fs", elapsed)

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V-JEPA temporal seam training proof-of-life smoke test",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of training steps (default: 40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data and seam initialization (default: 42)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Output directory for artifacts (default: artifacts/)",
    )
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "mock_lerobot_droid", "local_lerobot_rows"],
        default="synthetic",
        help=(
            "Local proof data source. mock_lerobot_droid exercises the "
            "LeRobot temporal adapter path without requiring external data; "
            "local_lerobot_rows consumes a local LeRobot-like JSON/JSONL row "
            "bundle."
        ),
    )
    parser.add_argument(
        "--lerobot-rows-path",
        type=str,
        default=None,
        help="Path to a local LeRobot-like JSON or JSONL row bundle.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=4,
        help="Maximum episodes to load for mock or local LeRobot row sources.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Optional cap on steps per episode for mock or local LeRobot row sources.",
    )
    parser.add_argument(
        "--require-loss-decrease",
        action="store_true",
        help="Exit nonzero if the best validation loss does not improve.",
    )
    args = parser.parse_args()

    artifact = run_proof_of_life(
        max_steps=args.steps,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        data_source=args.data_source,
        lerobot_rows_path=args.lerobot_rows_path,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
    )

    print("\n" + "=" * 60)
    print("V-JEPA TEMPORAL PROOF-OF-LIFE SUMMARY")
    print("=" * 60)
    print(f"  Seam type:      {artifact['seam_type']}")
    print(f"  Param count:    {artifact['param_count']:,}")
    print(f"  Steps:          {artifact['training_summary']['total_steps']}")
    print(f"  Data source:    {artifact['data_source']}")
    print(f"  Initial loss:   {artifact['loss_proof']['initial_val_loss']:.4f}")
    print(f"  Best val loss:  {artifact['loss_proof']['best_val_loss']:.4f}")
    print(
        f"  Loss decreased: {artifact['loss_proof']['loss_decreased']} "
        f"({artifact['loss_proof']['loss_improvement']:.4f})"
    )
    print(
        f"  Benchmark:      score={artifact['benchmark']['overall_score']:.4f} "
        f"passed={artifact['benchmark']['overall_passed']} "
        f"decision={artifact['benchmark']['promotion_decision']}"
    )
    print(
        f"  Receipts:       {artifact['receipts']['training_count']} train, "
        f"{artifact['receipts']['validation_count']} val, "
        f"{artifact['receipts']['benchmark_count']} benchmark"
    )
    print(f"  Elapsed:        {artifact['elapsed_seconds']}s")
    print(f"  Artifact:       {artifact['artifact_paths']['proof_of_life_summary']}")
    print(f"  Manifest:       {artifact['artifact_paths']['training_manifest']}")
    print(f"  Evidence:       {artifact['artifact_paths']['benchmark_evidence']}")
    print("=" * 60)
    if args.require_loss_decrease and not artifact["loss_proof"]["loss_decreased"]:
        raise SystemExit("best validation loss did not decrease")


if __name__ == "__main__":
    main()
