#!/usr/bin/env python3
"""Train a learned gen2sim validity helper from synthetic-branch assessment traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.config.internal_profile import get_internal_experiment_profile
from src.evidence.gen2sim_validity import (
    GEN2SIM_FEATURE_NAMES,
    build_gen2sim_feature_vector,
    coerce_gen2sim_validity_assessment,
)
from src.evidence.gen2sim_validity_training import (
    TORCH_AVAILABLE,
    Gen2SimValidityTrainingRow,
    LearnedGen2SimValidityModel,
    train_gen2sim_validity_model,
)
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.training.synthetic_branch_corpus import load_synthetic_branch_corpus
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit

if TORCH_AVAILABLE:  # pragma: no branch
    import torch
else:  # pragma: no cover - explicit errors below
    torch = None

GEN2SIM_MIN_RECORDS = 64
GEN2SIM_MIN_GROUNDED_RECORDS = 16
GEN2SIM_MIN_EMPIRICAL_RECEIPTS = 24


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    profile = get_internal_experiment_profile("default")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-corpus", type=str, default=profile["synthetic_branches_path"])
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--gap-labels", type=str, default=None)
    parser.add_argument("--gen2sim-validity", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="checkpoints/gen2sim_validity")
    parser.add_argument("--run-name", type=str, default="gen2sim_validity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _build_training_rows(corpus) -> list[Gen2SimValidityTrainingRow]:
    rows: list[Gen2SimValidityTrainingRow] = []
    for branch in corpus.branches:
        assessment = coerce_gen2sim_validity_assessment(branch.gen2sim_validity)
        if assessment is None:
            continue
        row_context = {
            "trust_score": branch.trust_score,
            "std_ratio": branch.std_ratio,
            "branch_value": branch.branch_value,
            "gap_labels": branch.gap_labels,
            "metadata": {
                **dict(corpus.metadata),
                "branch_value": branch.branch_value,
                "trust_score": branch.trust_score,
                "std_ratio": branch.std_ratio,
                "coverage_gap_contribution": branch.gap_labels.get(
                    "coverage_gap_contribution",
                    0.0,
                ),
                "economic_priority": branch.gap_labels.get("economic_priority", 0.0),
            },
            "objective_vector": branch.objective_vector,
        }
        rows.append(
            Gen2SimValidityTrainingRow(
                subject_id=f"synth_branch_{branch.branch_idx:04d}",
                feature_vector=build_gen2sim_feature_vector(row_context),
                target_validity_score=float(assessment.validity_score),
                target_value_support_score=float(assessment.value_support_score),
                promotion_stage=str(assessment.promotion_stage),
                metadata={
                    "branch_idx": int(branch.branch_idx),
                    "assessment_id": assessment.assessment_id,
                },
            )
        )
    return rows


def _build_dataset_summary(
    corpus,
    rows: Sequence[Gen2SimValidityTrainingRow],
) -> dict[str, Any]:
    assessments = [
        coerce_gen2sim_validity_assessment(branch.gen2sim_validity)
        for branch in corpus.branches
    ]
    assessments = [assessment for assessment in assessments if assessment is not None]
    stage_counts = Counter(assessment.promotion_stage for assessment in assessments)
    grounded_count = sum(
        1
        for assessment in assessments
        if bool(assessment.metadata.get("benchmark_signals", {}).get("benchmark_eligible", False))
    )
    empirical_receipt_count = sum(
        1
        for assessment in assessments
        if bool(assessment.metadata.get("empirical_outcome_observed", False))
    )
    benchmark_gate_ready = (
        len(rows) >= GEN2SIM_MIN_RECORDS
        and grounded_count >= GEN2SIM_MIN_GROUNDED_RECORDS
        and empirical_receipt_count >= GEN2SIM_MIN_EMPIRICAL_RECEIPTS
    )
    return {
        "schema_version": "gen2sim_validity_dataset_summary_v1",
        "branch_corpus_path": str(Path(corpus.npz_path).resolve()),
        "dataset_digest": sha256_json(
            {
                "branch_corpus": str(Path(corpus.npz_path).resolve()),
                "row_count": len(rows),
                "grounded_count": grounded_count,
                "empirical_receipt_count": empirical_receipt_count,
                "stage_counts": dict(sorted(stage_counts.items())),
            }
        ),
        "row_count": len(rows),
        "feature_dim": len(GEN2SIM_FEATURE_NAMES),
        "avg_target_validity_score": float(
            np.mean([row.target_validity_score for row in rows]) if rows else 0.0
        ),
        "avg_target_value_support_score": float(
            np.mean([row.target_value_support_score for row in rows]) if rows else 0.0
        ),
        "promotion_stage_counts": dict(sorted(stage_counts.items())),
        "grounded_count": grounded_count,
        "empirical_receipt_count": empirical_receipt_count,
        "corpus_summary": dict(corpus.summary),
        "benchmark_gate": {
            "name": "gen2sim_validity_empirical_density",
            "ready": benchmark_gate_ready,
            "required_records": GEN2SIM_MIN_RECORDS,
            "required_grounded_records": GEN2SIM_MIN_GROUNDED_RECORDS,
            "required_empirical_receipts": GEN2SIM_MIN_EMPIRICAL_RECEIPTS,
            "observed_records": len(rows),
            "observed_grounded_records": grounded_count,
            "observed_empirical_receipts": empirical_receipt_count,
        },
    }


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::branch_corpus_present": int(bool(dataset_summary.get("branch_corpus_path"))),
        "dataset::non_empty": int(int(dataset_summary.get("row_count", 0)) > 0),
        "dataset::grounded_support": int(
            int(dataset_summary.get("grounded_count", 0))
            >= int(benchmark_gate.get("required_grounded_records", 0))
        ),
        "dataset::empirical_receipt_support": int(
            int(dataset_summary.get("empirical_receipt_count", 0))
            >= int(benchmark_gate.get("required_empirical_receipts", 0))
        ),
        "benchmark::gen2sim_validity_empirical_density": int(
            bool(benchmark_gate.get("ready", False))
        ),
    }
    return {
        "schema_version": "gen2sim_validity_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "gen2sim_validity_model_config_v1",
        "input_dim": len(GEN2SIM_FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "targets": ["validity_score", "value_support_score"],
    }


def _evaluate_train_mse(
    model: LearnedGen2SimValidityModel,
    rows: Sequence[Gen2SimValidityTrainingRow],
) -> Dict[str, float]:
    if torch is None:
        return {"validity_mse": 0.0, "value_support_mse": 0.0}
    X = torch.from_numpy(
        np.asarray([row.feature_vector for row in rows], dtype=np.float32)
    )
    y_validity = np.asarray([row.target_validity_score for row in rows], dtype=np.float32)
    y_value_support = np.asarray(
        [row.target_value_support_score for row in rows],
        dtype=np.float32,
    )
    with torch.no_grad():
        pred_validity, pred_value_support = model(X)
    return {
        "validity_mse": float(
            np.mean((pred_validity.squeeze(-1).cpu().numpy() - y_validity) ** 2)
        ),
        "value_support_mse": float(
            np.mean((pred_value_support.squeeze(-1).cpu().numpy() - y_value_support) ** 2)
        ),
    }


def _build_runtime_package(
    *,
    config_digest: str,
    checkpoint_path: Path,
    dataset_summary: Mapping[str, Any],
    model_config: Mapping[str, Any],
    execution_preconditions: Mapping[str, Any],
    dataset_summary_path: Path,
    model_config_path: Path,
    preconditions_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    benchmark_gate_ready = bool(benchmark_gate.get("ready", False))
    return {
        "schema_version": "gen2sim_validity_runtime_package_v1",
        "package_id": f"gen2sim_validity_{config_digest[:12]}",
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary_path": str(dataset_summary_path),
        "model_config_path": str(model_config_path),
        "preconditions_path": str(preconditions_path),
        "training_summary_path": str(training_summary_path),
        "model_config": dict(model_config),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "inference_contract": {
            "blend_policy": {
                "shadow_candidate_max_delta": 0.12,
                "promoted_max_delta": 0.25,
            },
            "targets": ["validity_score", "value_support_score"],
            "feature_names": list(GEN2SIM_FEATURE_NAMES),
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "bounded_gen2sim_validity_helper_v1",
            "routing_targets": ["regal_data_value", "synthetic_branch_admission"],
        },
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    history: Mapping[str, Any],
    train_mse: Mapping[str, float],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "gen2sim_validity_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "row_count": int(dataset_summary.get("row_count", 0)),
        "train_loss": float(list(history.get("loss", [0.0]))[-1] if history.get("loss") else 0.0),
        "train_validity_mse": float(train_mse.get("validity_mse", 0.0)),
        "train_value_support_mse": float(train_mse.get("value_support_mse", 0.0)),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(rows: Sequence[Gen2SimValidityTrainingRow]) -> list[Any]:
    audits: list[Any] = []
    for row in rows:
        admission = float(
            row.target_validity_score * (0.75 + (0.25 * row.target_value_support_score))
        )
        audits.append(
            create_trajectory_audit(
                episode_id=row.subject_id,
                num_steps=1,
                rewards=[admission],
                reward_components={
                    "target_validity_score": [float(row.target_validity_score)],
                    "target_value_support_score": [float(row.target_value_support_score)],
                },
                events=[f"promotion_stage:{row.promotion_stage}", "gen2sim_validity_training_row"],
            )
        )
    return audits


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for gen2sim validity training")

    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    corpus = load_synthetic_branch_corpus(
        args.branch_corpus,
        metadata_path=args.metadata,
        gap_labels_path=args.gap_labels,
        gen2sim_validity_path=args.gen2sim_validity,
    )
    rows = _build_training_rows(corpus)
    if not rows:
        raise ValueError("No gen2sim validity rows found in the corpus")

    config_digest = sha256_json(
        {
            "branch_corpus": args.branch_corpus,
            "metadata": args.metadata,
            "gap_labels": args.gap_labels,
            "gen2sim_validity": args.gen2sim_validity,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
        }
    )

    dataset_summary = _build_dataset_summary(corpus, rows)
    dataset_summary_path = output_root / "gen2sim_validity_dataset_summary.json"
    _write_json(dataset_summary_path, dataset_summary)

    model_config = _build_model_config(hidden_dim=args.hidden_dim)
    model_config_path = output_root / "gen2sim_validity_model_config.json"
    _write_json(model_config_path, model_config)

    execution_preconditions = _build_execution_preconditions(dataset_summary)
    preconditions_path = output_root / "gen2sim_validity_execution_preconditions.json"
    _write_json(preconditions_path, execution_preconditions)

    rows_path = output_root / "gen2sim_validity_rows.json"
    _write_json(
        rows_path,
        {
            "rows": [
                {
                    "subject_id": row.subject_id,
                    "feature_vector": row.feature_vector,
                    "target_validity_score": row.target_validity_score,
                    "target_value_support_score": row.target_value_support_score,
                    "promotion_stage": row.promotion_stage,
                    "metadata": dict(row.metadata),
                }
                for row in rows
            ]
        },
    )

    checkpoint_path = output_root / "gen2sim_validity_model.pt"
    model, history = train_gen2sim_validity_model(
        rows,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_path=str(checkpoint_path),
    )
    train_mse = _evaluate_train_mse(model, rows)

    training_summary_path = output_root / "gen2sim_validity_training_summary.json"
    runtime_package_path = output_root / "gen2sim_validity_package.json"
    training_summary = _build_training_summary(
        run_name=args.run_name,
        seed=args.seed,
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        history=history,
        train_mse=train_mse,
        artifacts={
            "dataset_summary": str(dataset_summary_path),
            "model_config": str(model_config_path),
            "execution_preconditions": str(preconditions_path),
            "rows": str(rows_path),
            "checkpoint": str(checkpoint_path),
        },
    )
    _write_json(training_summary_path, training_summary)

    runtime_package = _build_runtime_package(
        config_digest=config_digest,
        checkpoint_path=checkpoint_path,
        dataset_summary=dataset_summary,
        model_config=model_config,
        execution_preconditions=execution_preconditions,
        dataset_summary_path=dataset_summary_path,
        model_config_path=model_config_path,
        preconditions_path=preconditions_path,
        training_summary_path=training_summary_path,
    )
    _write_json(runtime_package_path, runtime_package)

    payload = {
        "dataset_summary": dataset_summary,
        "execution_preconditions": execution_preconditions,
        "training_summary": training_summary,
        "runtime_package": runtime_package,
        "artifacts": {
            "dataset_summary": str(dataset_summary_path),
            "model_config": str(model_config_path),
            "execution_preconditions": str(preconditions_path),
            "training_summary": str(training_summary_path),
            "runtime_package": str(runtime_package_path),
            "rows": str(rows_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    training_job_result_path = output_root / "training_job_result.json"
    _write_json(training_job_result_path, payload)

    if runner is not None:
        runner.set_eligible_datapacks([row.subject_id for row in rows])
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for row in rows:
            runner.record_sample("gen2sim_validity_row", datapack_id=row.subject_id, slice_id=row.subject_id)
        for audit in _build_trajectory_audits(rows):
            runner.add_trajectory_audit(audit)
        runner.update_step(int(args.epochs))
        runner.set_weights(
            baseline_weights={"epochs": float(args.epochs)},
            final_weights={"lr": float(args.lr), "hidden_dim": float(args.hidden_dim)},
        )
        runner.configure_training_runtime(
            training_kind="gen2sim_validity_helper",
            config_digest=config_digest,
            replay_dataset_summary={
                "row_count": len(rows),
                "grounded_count": int(dataset_summary.get("grounded_count", 0)),
                "empirical_receipt_count": int(dataset_summary.get("empirical_receipt_count", 0)),
            },
            objective_profile_snapshot={
                "feature_names": list(GEN2SIM_FEATURE_NAMES),
                "targets": ["validity_score", "value_support_score"],
            },
            promotion_policy_snapshot={"runtime_package": runtime_package},
            source_domain_coverage={
                "source_domain_counts": {"synthetic_branch": len(rows)},
                "transition_counts": {"synthetic_branch": len(rows)},
                "total_episodes": len(rows),
            },
            receipt_label_coverage={
                "empirical_receipt_count": int(dataset_summary.get("empirical_receipt_count", 0))
            },
            metadata={
                "branch_corpus_path": str(Path(args.branch_corpus).resolve()),
                "benchmark_gate_ready": bool(
                    dataset_summary.get("benchmark_gate", {}).get("ready", False)
                ),
                "promotion_stage": runtime_package.get("promotion_stage", "shadow_candidate"),
            },
        )
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "dataset_summary": dataset_summary,
                "execution_preconditions": execution_preconditions,
                "runtime_package": runtime_package,
            },
            context_sha=config_digest,
        )
        runner.register_artifact("gen2sim_validity_dataset_summary", dataset_summary_path)
        runner.register_artifact("gen2sim_validity_model_config", model_config_path)
        runner.register_artifact("gen2sim_validity_execution_preconditions", preconditions_path)
        runner.register_artifact("gen2sim_validity_rows", rows_path)
        runner.register_artifact("gen2sim_validity_training_summary", training_summary_path)
        runner.register_artifact("gen2sim_validity_runtime_package", runtime_package_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="gen2sim_validity_model",
                model_family="gen2sim_validity_helper",
                model_version="bounded_gen2sim_validity_helper_v1",
                path=checkpoint_path,
                epoch=args.epochs,
                step=args.epochs,
                metadata={"dataset_digest": dataset_summary.get("dataset_digest")},
            )
        )

    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.skip_regal_runner:
        payload = _run_training(args, runner=None)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    holder: Dict[str, Any] = {}

    def _wrapped(runner: RegalTrainingRunner) -> None:
        holder["payload"] = _run_training(args, runner=runner)

    plan_sha = sha256_json(
        {
            "script": "train_gen2sim_validity.py",
            "branch_corpus": args.branch_corpus,
            "seed": args.seed,
        }
    )
    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=args.save_dir,
            seed=args.seed,
            num_episodes=args.epochs,
            training_steps=args.epochs,
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="gen2sim_validity_helper",
    )
    print(json.dumps(holder.get("payload", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
