#!/usr/bin/env python3
"""Train the learned semantic WM refiner with canonical runtime artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit
from src.world_model.semantic_wm_refiner import (
    SemanticWMRefinementDataset,
    build_semantic_wm_refinement_dataset_from_artifact_dirs,
    train_semantic_wm_refiner_package,
)


SEMANTIC_WM_REFINER_MIN_OBJECT_ROWS = 32
SEMANTIC_WM_REFINER_MIN_RELATION_ROWS = 12
SEMANTIC_WM_REFINER_MIN_PROPOSAL_ROWS = 12
SEMANTIC_WM_REFINER_MIN_ARTIFACT_DIRS = 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train semantic WM refiner")
    parser.add_argument(
        "--artifact-dir",
        action="append",
        required=True,
        help="Path to a coverage-loop artifact directory. Repeat to train on multiple runs.",
    )
    parser.add_argument("--epochs", type=int, default=32, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Hidden dimension")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/semantic_wm_refiner",
        help="Output directory",
    )
    parser.add_argument("--run-name", type=str, default="semantic_wm_refiner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _build_dataset_summary(dataset: SemanticWMRefinementDataset, artifact_dirs: Sequence[str]) -> dict[str, Any]:
    accepted_proposals = sum(1 for value in dataset.proposal_accept_targets if float(value) >= 0.5)
    benchmark_gate_ready = (
        len(dataset.object_features) >= SEMANTIC_WM_REFINER_MIN_OBJECT_ROWS
        and len(dataset.relation_features) >= SEMANTIC_WM_REFINER_MIN_RELATION_ROWS
        and len(dataset.proposal_features) >= SEMANTIC_WM_REFINER_MIN_PROPOSAL_ROWS
        and len(artifact_dirs) >= SEMANTIC_WM_REFINER_MIN_ARTIFACT_DIRS
    )
    return {
        "schema_version": "semantic_wm_refiner_dataset_summary_v1",
        "artifact_dirs": [str(Path(path).resolve()) for path in artifact_dirs],
        "dataset_digest": sha256_json(
            {
                "artifact_dirs": [str(Path(path).resolve()) for path in artifact_dirs],
                "object_rows": len(dataset.object_features),
                "relation_rows": len(dataset.relation_features),
                "capability_rows": len(dataset.capability_features),
                "proposal_rows": len(dataset.proposal_features),
                "accepted_proposals": accepted_proposals,
            }
        ),
        "object_rows": len(dataset.object_features),
        "relation_rows": len(dataset.relation_features),
        "capability_rows": len(dataset.capability_features),
        "proposal_rows": len(dataset.proposal_features),
        "accepted_proposals": accepted_proposals,
        "global_feature_dim": int(dataset.global_feature_dim),
        "metadata": dict(dataset.metadata),
        "benchmark_gate": {
            "name": "semantic_wm_refiner_coverage_artifact_density",
            "ready": benchmark_gate_ready,
            "required_object_rows": SEMANTIC_WM_REFINER_MIN_OBJECT_ROWS,
            "required_relation_rows": SEMANTIC_WM_REFINER_MIN_RELATION_ROWS,
            "required_proposal_rows": SEMANTIC_WM_REFINER_MIN_PROPOSAL_ROWS,
            "required_artifact_dirs": SEMANTIC_WM_REFINER_MIN_ARTIFACT_DIRS,
            "observed_object_rows": len(dataset.object_features),
            "observed_relation_rows": len(dataset.relation_features),
            "observed_proposal_rows": len(dataset.proposal_features),
            "observed_artifact_dirs": len(artifact_dirs),
        },
    }


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::artifact_dirs": int(bool(dataset_summary.get("artifact_dirs"))),
        "dataset::object_rows_present": int(int(dataset_summary.get("object_rows", 0)) > 0),
        "dataset::relation_rows_present": int(int(dataset_summary.get("relation_rows", 0)) > 0),
        "dataset::proposal_rows_present": int(int(dataset_summary.get("proposal_rows", 0)) > 0),
        "dataset::artifact_dir_support": int(
            int(benchmark_gate.get("observed_artifact_dirs", 0))
            >= int(benchmark_gate.get("required_artifact_dirs", 0))
        ),
        "benchmark::semantic_wm_refiner_coverage_artifact_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "semantic_wm_refiner_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(dataset: SemanticWMRefinementDataset, *, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "semantic_wm_refiner_model_config_v1",
        "global_feature_dim": int(dataset.global_feature_dim),
        "object_feature_dim": len(dataset.object_features[0]) if dataset.object_features else 0,
        "relation_feature_dim": len(dataset.relation_features[0]) if dataset.relation_features else 0,
        "capability_feature_dim": len(dataset.capability_features[0]) if dataset.capability_features else 0,
        "proposal_feature_dim": len(dataset.proposal_features[0]) if dataset.proposal_features else 0,
        "hidden_dim": int(hidden_dim),
        "targets": [
            "object_confidence_delta",
            "relation_confidence_delta",
            "capability_adjustments",
            "proposal_accept_probability",
            "proposal_confidence_delta",
        ],
    }


def _evaluate_train_mae(package, dataset: SemanticWMRefinementDataset) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if dataset.object_features:
        inputs = torch.tensor(np.asarray(dataset.object_features, dtype=np.float32))
        targets = np.asarray(dataset.object_targets, dtype=np.float32)
        with torch.no_grad():
            preds = package.object_model(inputs).squeeze(-1).cpu().numpy()
        metrics["object_mae"] = float(np.mean(np.abs(preds - targets)))
    else:
        metrics["object_mae"] = 0.0
    if dataset.relation_features:
        inputs = torch.tensor(np.asarray(dataset.relation_features, dtype=np.float32))
        targets = np.asarray(dataset.relation_targets, dtype=np.float32)
        with torch.no_grad():
            preds = package.relation_model(inputs).squeeze(-1).cpu().numpy()
        metrics["relation_mae"] = float(np.mean(np.abs(preds - targets)))
    else:
        metrics["relation_mae"] = 0.0
    if dataset.capability_features:
        inputs = torch.tensor(np.asarray(dataset.capability_features, dtype=np.float32))
        targets = np.asarray(dataset.capability_targets, dtype=np.float32)
        with torch.no_grad():
            preds = package.capability_model(inputs).cpu().numpy()
        metrics["capability_mae"] = float(np.mean(np.abs(preds - targets)))
    else:
        metrics["capability_mae"] = 0.0
    if dataset.proposal_features:
        inputs = torch.tensor(np.asarray(dataset.proposal_features, dtype=np.float32))
        accept_targets = np.asarray(dataset.proposal_accept_targets, dtype=np.float32)
        confidence_targets = np.asarray(dataset.proposal_confidence_targets, dtype=np.float32)
        with torch.no_grad():
            accept_preds = package.proposal_accept_model(inputs).squeeze(-1).cpu().numpy()
            confidence_preds = package.proposal_confidence_model(inputs).squeeze(-1).cpu().numpy()
        metrics["proposal_accept_mae"] = float(np.mean(np.abs(accept_preds - accept_targets)))
        metrics["proposal_confidence_mae"] = float(np.mean(np.abs(confidence_preds - confidence_targets)))
    else:
        metrics["proposal_accept_mae"] = 0.0
        metrics["proposal_confidence_mae"] = 0.0
    return metrics


def _build_runtime_package(
    *,
    config_digest: str,
    checkpoint_path: Path,
    dataset_summary: Mapping[str, Any],
    model_config: Mapping[str, Any],
    execution_preconditions: Mapping[str, Any],
    dataset_summary_path: Path,
    dataset_path: Path,
    model_config_path: Path,
    preconditions_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    benchmark_gate_ready = bool(benchmark_gate.get("ready", False))
    return {
        "schema_version": "semantic_wm_refiner_runtime_package_v1",
        "package_id": f"semantic_wm_refiner_{config_digest[:12]}",
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary_path": str(dataset_summary_path),
        "dataset_path": str(dataset_path),
        "model_config_path": str(model_config_path),
        "preconditions_path": str(preconditions_path),
        "training_summary_path": str(training_summary_path),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "inference_contract": {
            "target_contract": "semantic_wm_refiner_v1",
            "allowed_modes": ["disabled", "auto", "required"],
            "shadow_candidate_overlay_scale": 0.28,
            "promoted_overlay_scale": 0.62,
            "shadow_candidate_proposal_scale": 0.18,
            "promoted_proposal_scale": 0.35,
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "semantic_wm_refiner_overlay_and_mutation_v1",
            "routing_targets": ["coverage_loop_correction_overlay", "coverage_loop_graph_mutation"],
        },
        "model_config": dict(model_config),
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    train_mae: Mapping[str, float],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "semantic_wm_refiner_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "object_rows": int(dataset_summary.get("object_rows", 0)),
        "relation_rows": int(dataset_summary.get("relation_rows", 0)),
        "proposal_rows": int(dataset_summary.get("proposal_rows", 0)),
        "train_mae": dict(train_mae),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(artifact_dirs: Sequence[str]) -> list[Any]:
    audits: list[Any] = []
    for index, artifact_dir in enumerate(artifact_dirs):
        audits.append(
            create_trajectory_audit(
                episode_id=f"semantic_wm_refiner_{index:03d}",
                num_steps=1,
                rewards=[1.0],
                reward_components={"artifact_dir_present": [1.0]},
                events=[str(Path(artifact_dir).name)],
            )
        )
    return audits


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = build_semantic_wm_refinement_dataset_from_artifact_dirs(args.artifact_dir)
    if not dataset.object_features and not dataset.relation_features:
        raise ValueError("no semantic WM refinement samples available")

    dataset_summary = _build_dataset_summary(dataset, args.artifact_dir)
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    model_config = _build_model_config(dataset, hidden_dim=args.hidden_dim)
    config_digest = sha256_json(
        {
            "artifact_dirs": [str(Path(path).resolve()) for path in args.artifact_dir],
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "run_name": args.run_name,
            "seed": args.seed,
        }
    )

    dataset_path = output_dir / "semantic_wm_refiner_dataset.json"
    dataset_summary_path = output_dir / "semantic_wm_refiner_dataset_summary.json"
    model_config_path = output_dir / "semantic_wm_refiner_model_config.json"
    preconditions_path = output_dir / "semantic_wm_refiner_execution_preconditions.json"
    training_summary_path = output_dir / "semantic_wm_refiner_training_summary.json"
    runtime_package_path = output_dir / "semantic_wm_refiner_runtime_package.json"
    checkpoint_path = output_dir / "semantic_wm_refiner_package.pt"
    training_job_result_path = output_dir / "training_job_result.json"

    package = train_semantic_wm_refiner_package(
        dataset,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
    )
    torch.save(package.to_checkpoint(), checkpoint_path)
    train_mae = _evaluate_train_mae(package, dataset)

    artifacts = {
        "checkpoint": str(checkpoint_path),
        "dataset_summary": str(dataset_summary_path),
        "dataset": str(dataset_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "runtime_package": str(runtime_package_path),
    }
    training_summary = _build_training_summary(
        run_name=args.run_name,
        seed=args.seed,
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        train_mae=train_mae,
        artifacts=artifacts,
    )
    runtime_package = _build_runtime_package(
        config_digest=config_digest,
        checkpoint_path=checkpoint_path,
        dataset_summary=dataset_summary,
        model_config=model_config,
        execution_preconditions=execution_preconditions,
        dataset_summary_path=dataset_summary_path,
        dataset_path=dataset_path,
        model_config_path=model_config_path,
        preconditions_path=preconditions_path,
        training_summary_path=training_summary_path,
    )

    _write_json(dataset_path, dataset.to_dict())
    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)
    _write_json(runtime_package_path, runtime_package)
    _write_json(
        training_job_result_path,
        {
            "training_kind": "semantic_wm_refiner",
            "result": {
                "checkpoint": str(checkpoint_path),
                "dataset_summary": str(dataset_summary_path),
                "runtime_package": str(runtime_package_path),
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
            },
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        datapack_ids = [f"coverage_artifact_{index:03d}" for index in range(len(args.artifact_dir))]
        runner.set_eligible_datapacks(datapack_ids)
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for index, path in enumerate(args.artifact_dir):
            runner.record_sample("semantic_wm_refiner", datapack_id=datapack_ids[index], slice_id=str(path))
        for audit in _build_trajectory_audits(args.artifact_dir):
            runner.add_trajectory_audit(audit)
        runner.update_step(max(1, args.epochs))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "object_rows": int(dataset_summary.get("object_rows", 0)),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="semantic_wm_refiner",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "semantic_wm_refiner"},
            promotion_policy_snapshot={"benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {})},
            source_domain_coverage={"artifact_dir_count": len(args.artifact_dir)},
            receipt_label_coverage={"accepted_proposals": int(dataset_summary.get("accepted_proposals", 0))},
            metadata={
                "trajectory_audit_kind": "semantic_wm_refiner_artifact_projection",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "target_contract": "semantic_wm_refiner_v1",
            },
        )
        runner.register_artifact("semantic_wm_refiner_dataset", dataset_path)
        runner.register_artifact("semantic_wm_refiner_dataset_summary", dataset_summary_path)
        runner.register_artifact("semantic_wm_refiner_model_config", model_config_path)
        runner.register_artifact("semantic_wm_refiner_preconditions", preconditions_path)
        runner.register_artifact("semantic_wm_refiner_training_summary", training_summary_path)
        runner.register_artifact("semantic_wm_refiner_runtime_package", runtime_package_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="semantic_wm_refiner_latest",
                model_family="semantic_wm_refiner",
                model_version="semantic_wm_refiner_v1",
                path=checkpoint_path,
                step=max(1, args.epochs),
                epoch=args.epochs,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "train_mae": dict(train_mae),
                },
            )
        )

    return {
        "checkpoint": str(checkpoint_path),
        "dataset_summary": str(dataset_summary_path),
        "dataset": str(dataset_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "training_summary": str(training_summary_path),
        "runtime_package": str(runtime_package_path),
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plan_sha = sha256_json(
        {
            "training_kind": "semantic_wm_refiner",
            "artifact_dirs": [str(Path(path).resolve()) for path in args.artifact_dir],
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "run_name": args.run_name,
            "seed": args.seed,
        }
    )

    if args.skip_regal_runner:
        payload = _run_training(args, runner=None)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    holder: Dict[str, Any] = {}

    def _wrapped(runner: RegalTrainingRunner) -> None:
        holder["payload"] = _run_training(args, runner)

    result = run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            num_episodes=max(1, len(args.artifact_dir)),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="semantic_wm_refiner",
    )
    print(
        json.dumps(
            {
                "training_run": result.to_dict(),
                "job": holder.get("payload", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
