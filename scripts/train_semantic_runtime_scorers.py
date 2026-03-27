#!/usr/bin/env python3
"""Train semantic runtime scorers from replay-backed runtime rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.orchestrator.semantic_runtime_learning import (
    SemanticRuntimeLearningCorpus,
    build_semantic_runtime_learning_corpus,
)
from src.orchestrator.semantic_runtime_scorer_training import (
    TORCH_AVAILABLE,
    SemanticRuntimeScorerTrainingDataset,
    build_semantic_runtime_scorer_training_dataset,
    save_semantic_runtime_scorer_checkpoint,
    train_semantic_runtime_scorer_net,
    write_semantic_runtime_scorer_training_dataset,
)
from src.orchestrator.semantic_runtime_scorers import (
    score_semantic_runtime_learning_row,
    train_semantic_runtime_scorer_package,
    write_semantic_runtime_scorer_package,
)
from src.replay.dataset import load_replay_dataset
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit


SEMANTIC_RUNTIME_SCORER_MIN_ROWS = 64
SEMANTIC_RUNTIME_SCORER_MIN_EXECUTION_READY = 16
SEMANTIC_RUNTIME_SCORER_MIN_ROUTE_SUCCESS = 16
SEMANTIC_RUNTIME_SCORER_MIN_SEMANTIC_GROUNDED = 16
SEMANTIC_RUNTIME_SCORER_MIN_COUNTERFACTUALS = 32


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dataset", required=True, help="Path to canonical replay dataset directory")
    parser.add_argument("--output-dir", required=True, help="Directory to write scorer artifacts")
    parser.add_argument("--max-counterfactuals", type=int, default=3, help="Maximum shadow counterfactuals per row")
    parser.add_argument(
        "--trainer",
        choices=["linear", "torch", "both"],
        default="both",
        help="Which training artifacts to emit",
    )
    parser.add_argument("--epochs", type=int, default=24, help="Epochs for the torch scorer trainer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the torch scorer trainer")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width for the torch scorer trainer")
    parser.add_argument("--run-name", type=str, default="semantic_runtime_scorers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _build_dataset_summary(
    *,
    replay_dataset_path: str,
    corpus: SemanticRuntimeLearningCorpus,
    training_dataset: SemanticRuntimeScorerTrainingDataset,
) -> dict[str, Any]:
    counterfactual_count = int(training_dataset.metadata.get("counterfactual_count", 0))
    benchmark_gate_ready = (
        int(corpus.summary.get("row_count", 0)) >= SEMANTIC_RUNTIME_SCORER_MIN_ROWS
        and int(corpus.summary.get("bounded_ready_count", 0)) >= SEMANTIC_RUNTIME_SCORER_MIN_EXECUTION_READY
        and int(corpus.summary.get("route_success_count", 0)) >= SEMANTIC_RUNTIME_SCORER_MIN_ROUTE_SUCCESS
        and int(corpus.summary.get("semantic_grounded_count", 0)) >= SEMANTIC_RUNTIME_SCORER_MIN_SEMANTIC_GROUNDED
        and counterfactual_count >= SEMANTIC_RUNTIME_SCORER_MIN_COUNTERFACTUALS
    )
    return {
        "schema_version": "semantic_runtime_scorer_dataset_summary_v1",
        "replay_dataset_path": str(Path(replay_dataset_path).resolve()),
        "dataset_digest": sha256_json(
            {
                "replay_dataset_path": str(Path(replay_dataset_path).resolve()),
                "row_count": int(corpus.summary.get("row_count", 0)),
                "bounded_ready_count": int(corpus.summary.get("bounded_ready_count", 0)),
                "route_success_count": int(corpus.summary.get("route_success_count", 0)),
                "semantic_grounded_count": int(corpus.summary.get("semantic_grounded_count", 0)),
                "counterfactual_count": counterfactual_count,
                "authority_distribution": dict(corpus.summary.get("authority_distribution", {}) or {}),
            }
        ),
        "row_count": int(corpus.summary.get("row_count", 0)),
        "counterfactual_count": counterfactual_count,
        "bounded_ready_count": int(corpus.summary.get("bounded_ready_count", 0)),
        "route_success_count": int(corpus.summary.get("route_success_count", 0)),
        "authority_success_count": int(corpus.summary.get("authority_success_count", 0)),
        "semantic_grounded_count": int(corpus.summary.get("semantic_grounded_count", 0)),
        "authority_distribution": dict(corpus.summary.get("authority_distribution", {}) or {}),
        "feature_dims": {
            "meta_route": len(training_dataset.meta_route_feature_names),
            "orchestration_route": len(training_dataset.orchestration_route_feature_names),
            "authority": len(training_dataset.authority_feature_names),
            "counterfactual": len(training_dataset.counterfactual_feature_names),
        },
        "corpus_summary": dict(corpus.summary),
        "training_dataset_metadata": dict(training_dataset.metadata),
        "benchmark_gate": {
            "name": "semantic_runtime_runtime_row_density",
            "ready": benchmark_gate_ready,
            "required_rows": SEMANTIC_RUNTIME_SCORER_MIN_ROWS,
            "required_execution_ready_rows": SEMANTIC_RUNTIME_SCORER_MIN_EXECUTION_READY,
            "required_route_success_rows": SEMANTIC_RUNTIME_SCORER_MIN_ROUTE_SUCCESS,
            "required_semantic_grounded_rows": SEMANTIC_RUNTIME_SCORER_MIN_SEMANTIC_GROUNDED,
            "required_counterfactuals": SEMANTIC_RUNTIME_SCORER_MIN_COUNTERFACTUALS,
            "observed_rows": int(corpus.summary.get("row_count", 0)),
            "observed_execution_ready_rows": int(corpus.summary.get("bounded_ready_count", 0)),
            "observed_route_success_rows": int(corpus.summary.get("route_success_count", 0)),
            "observed_semantic_grounded_rows": int(corpus.summary.get("semantic_grounded_count", 0)),
            "observed_counterfactuals": counterfactual_count,
        },
    }


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::replay_dataset_present": int(bool(dataset_summary.get("replay_dataset_path"))),
        "dataset::non_empty": int(int(dataset_summary.get("row_count", 0)) > 0),
        "dataset::execution_ready_support": int(
            int(dataset_summary.get("bounded_ready_count", 0))
            >= int(benchmark_gate.get("required_execution_ready_rows", 0))
        ),
        "dataset::route_success_support": int(
            int(dataset_summary.get("route_success_count", 0))
            >= int(benchmark_gate.get("required_route_success_rows", 0))
        ),
        "dataset::semantic_grounded_support": int(
            int(dataset_summary.get("semantic_grounded_count", 0))
            >= int(benchmark_gate.get("required_semantic_grounded_rows", 0))
        ),
        "dataset::counterfactual_support": int(
            int(dataset_summary.get("counterfactual_count", 0))
            >= int(benchmark_gate.get("required_counterfactuals", 0))
        ),
        "benchmark::semantic_runtime_runtime_row_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "semantic_runtime_scorer_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, training_dataset: SemanticRuntimeScorerTrainingDataset, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "semantic_runtime_scorer_model_config_v1",
        "feature_dims": {
            "meta_route": len(training_dataset.meta_route_feature_names),
            "orchestration_route": len(training_dataset.orchestration_route_feature_names),
            "authority": len(training_dataset.authority_feature_names),
            "counterfactual": len(training_dataset.counterfactual_feature_names),
        },
        "feature_names": {
            "meta_route": list(training_dataset.meta_route_feature_names),
            "orchestration_route": list(training_dataset.orchestration_route_feature_names),
            "authority": list(training_dataset.authority_feature_names),
            "counterfactual": list(training_dataset.counterfactual_feature_names),
        },
        "hidden_dim": int(hidden_dim),
        "targets": [
            "meta_route_success_probability",
            "orchestration_route_success_probability",
            "authority_success_probability",
            "predicted_regret",
            "counterfactual_value_score",
        ],
    }


def _build_runtime_package(
    *,
    config_digest: str,
    scorer_package_path: Path,
    torch_checkpoint_path: Optional[Path],
    dataset_summary: Mapping[str, Any],
    model_config: Mapping[str, Any],
    execution_preconditions: Mapping[str, Any],
    dataset_summary_path: Path,
    training_dataset_path: Path,
    model_config_path: Path,
    preconditions_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    benchmark_gate_ready = bool(benchmark_gate.get("ready", False))
    return {
        "schema_version": "semantic_runtime_scorer_runtime_package_v1",
        "package_id": f"semantic_runtime_scorer_{config_digest[:12]}",
        "scorer_package_path": str(scorer_package_path),
        "torch_checkpoint_path": str(torch_checkpoint_path) if torch_checkpoint_path is not None else "",
        "dataset_summary_path": str(dataset_summary_path),
        "training_dataset_path": str(training_dataset_path),
        "model_config_path": str(model_config_path),
        "preconditions_path": str(preconditions_path),
        "training_summary_path": str(training_summary_path),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "inference_contract": {
            "target_contract": "semantic_runtime_scorer_v1",
            "helper_blend_policy": "bounded_semantic_runtime_shadow_advisory_v1",
            "allowed_modes": ["disabled", "auto", "required"],
            "legacy_package_path": str(scorer_package_path),
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "semantic_runtime_scorer_runtime_row_v1",
            "routing_targets": ["shadow_advisory", "queue_selection", "replay_sampling"],
        },
        "model_config": dict(model_config),
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    trainer_mode: str,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    linear_summary: Mapping[str, Any],
    torch_summary: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "semantic_runtime_scorer_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "trainer_mode": trainer_mode,
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "row_count": int(dataset_summary.get("row_count", 0)),
        "counterfactual_count": int(dataset_summary.get("counterfactual_count", 0)),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "linear_summary": dict(linear_summary),
        "torch_summary": dict(torch_summary),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(corpus: SemanticRuntimeLearningCorpus) -> list[Any]:
    audits: list[Any] = []
    for row in corpus.rows:
        route_success = float(row.inferential_summary.get("route_success_label", 0.0) or 0.0)
        authority_success = float(row.inferential_summary.get("authority_success_label", 0.0) or 0.0)
        quality = float(row.outcome_summary.get("quality_score", 0.0) or 0.0)
        reward_signal = float(row.outcome_summary.get("reward_signal", 0.0) or 0.0)
        audits.append(
            create_trajectory_audit(
                episode_id=row.episode_id,
                num_steps=max(1, len(row.counterfactuals) + 1),
                rewards=[quality + route_success - float(row.inferential_summary.get("estimated_regret", 0.0) or 0.0)],
                reward_components={
                    "route_success": [route_success],
                    "authority_success": [authority_success],
                    "quality_score": [quality],
                    "reward_signal": [reward_signal],
                },
                events=[
                    f"authority_gt:{row.meta_transformer_target.get('authority_gt', 'unknown')}",
                    f"objective:{row.meta_transformer_target.get('objective_preset', 'balanced')}",
                ],
            )
        )
    return audits


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    replay_bundle = load_replay_dataset(args.replay_dataset)
    corpus = build_semantic_runtime_learning_corpus(
        replay_bundle,
        max_counterfactuals=max(args.max_counterfactuals, 1),
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    training_dataset = build_semantic_runtime_scorer_training_dataset(corpus)
    dataset_summary = _build_dataset_summary(
        replay_dataset_path=args.replay_dataset,
        corpus=corpus,
        training_dataset=training_dataset,
    )
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    model_config = _build_model_config(training_dataset=training_dataset, hidden_dim=args.hidden_dim)
    config_digest = sha256_json(
        {
            "replay_dataset": str(Path(args.replay_dataset).resolve()),
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "max_counterfactuals": args.max_counterfactuals,
            "trainer": args.trainer,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "run_name": args.run_name,
            "seed": args.seed,
        }
    )

    training_dataset_path = Path(
        write_semantic_runtime_scorer_training_dataset(
            output_root / "semantic_runtime_scorer_training_dataset.json",
            training_dataset,
        )
    )
    dataset_summary_path = output_root / "semantic_runtime_scorer_dataset_summary.json"
    model_config_path = output_root / "semantic_runtime_scorer_model_config.json"
    preconditions_path = output_root / "semantic_runtime_scorer_execution_preconditions.json"
    training_summary_path = output_root / "semantic_runtime_scorer_training_summary.json"
    runtime_package_path = output_root / "semantic_runtime_scorer_runtime_package.json"
    training_job_result_path = output_root / "training_job_result.json"
    legacy_package_path = output_root / "semantic_runtime_scorer_package.json"
    scores_path = output_root / "semantic_runtime_shadow_scores.jsonl"

    scorer_package = train_semantic_runtime_scorer_package(corpus)
    write_semantic_runtime_scorer_package(legacy_package_path, scorer_package)
    linear_summary: Dict[str, Any] = {
        "trained": True,
        **dict(scorer_package.summary or {}),
    }
    with scores_path.open("w", encoding="utf-8") as handle:
        for row in corpus.rows:
            handle.write(json.dumps(score_semantic_runtime_learning_row(scorer_package, row).to_dict(), sort_keys=True) + "\n")

    torch_summary: Dict[str, Any] = {"torch_available": TORCH_AVAILABLE, "trained": False}
    torch_checkpoint_path: Optional[Path] = None
    if args.trainer in {"torch", "both"}:
        torch_result = train_semantic_runtime_scorer_net(
            training_dataset,
            epochs=max(args.epochs, 1),
            learning_rate=float(args.lr),
            hidden_dim=max(args.hidden_dim, 8),
        )
        torch_summary = {
            key: value
            for key, value in dict(torch_result.get("summary", torch_result)).items()
            if key != "model"
        }
        checkpoint_ref = save_semantic_runtime_scorer_checkpoint(
            output_root / "semantic_runtime_scorer_model.pt",
            torch_result,
        )
        torch_checkpoint_path = Path(checkpoint_ref) if checkpoint_ref else None

    artifacts = {
        "legacy_scorer_package": str(legacy_package_path) if legacy_package_path.exists() else "",
        "shadow_scores": str(scores_path) if scores_path.exists() else "",
        "training_dataset": str(training_dataset_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "runtime_package": str(runtime_package_path),
        "torch_checkpoint": str(torch_checkpoint_path) if torch_checkpoint_path is not None else "",
    }
    training_summary = _build_training_summary(
        run_name=args.run_name,
        seed=args.seed,
        trainer_mode=args.trainer,
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        linear_summary=linear_summary,
        torch_summary=torch_summary,
        artifacts=artifacts,
    )
    runtime_package = _build_runtime_package(
        config_digest=config_digest,
        scorer_package_path=legacy_package_path,
        torch_checkpoint_path=torch_checkpoint_path,
        dataset_summary=dataset_summary,
        model_config=model_config,
        execution_preconditions=execution_preconditions,
        dataset_summary_path=dataset_summary_path,
        training_dataset_path=training_dataset_path,
        model_config_path=model_config_path,
        preconditions_path=preconditions_path,
        training_summary_path=training_summary_path,
    )

    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)
    _write_json(runtime_package_path, runtime_package)

    result = {
        "legacy_scorer_package": str(legacy_package_path) if legacy_package_path.exists() else "",
        "training_dataset": str(training_dataset_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "training_summary": str(training_summary_path),
        "runtime_package": str(runtime_package_path),
        "shadow_scores": str(scores_path) if scores_path.exists() else "",
        "torch_checkpoint": str(torch_checkpoint_path) if torch_checkpoint_path is not None else "",
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }
    _write_json(
        training_job_result_path,
        {
            "training_kind": "semantic_runtime_scorers",
            "result": result,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        datapack_ids = [row.episode_id for row in corpus.rows]
        runner.set_eligible_datapacks(datapack_ids)
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for row in corpus.rows:
            runner.record_sample(
                row.task_id,
                datapack_id=row.episode_id,
                slice_id=row.sample_id,
            )
        for audit in _build_trajectory_audits(corpus):
            runner.add_trajectory_audit(audit)
        runner.update_step(max(1, args.epochs))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "row_count": len(corpus.rows),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="semantic_runtime_scorers",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "semantic_runtime_scorer"},
            promotion_policy_snapshot={"benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {})},
            source_domain_coverage={"source_domains": list(corpus.summary.get("source_domains", []))},
            receipt_label_coverage={
                "execution_ready_rows": int(corpus.summary.get("bounded_ready_count", 0)),
                "semantic_grounded_rows": int(corpus.summary.get("semantic_grounded_count", 0)),
            },
            metadata={
                "trajectory_audit_kind": "semantic_runtime_scorer_runtime_projection",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "target_contract": "semantic_runtime_scorer_v1",
            },
        )
        runner.register_artifact("semantic_runtime_scorer_training_dataset", training_dataset_path)
        runner.register_artifact("semantic_runtime_scorer_dataset_summary", dataset_summary_path)
        runner.register_artifact("semantic_runtime_scorer_model_config", model_config_path)
        runner.register_artifact("semantic_runtime_scorer_preconditions", preconditions_path)
        runner.register_artifact("semantic_runtime_scorer_training_summary", training_summary_path)
        runner.register_artifact("semantic_runtime_scorer_runtime_package", runtime_package_path)
        if legacy_package_path.exists():
            runner.register_artifact("semantic_runtime_scorer_legacy_package", legacy_package_path)
        if scores_path.exists():
            runner.register_artifact("semantic_runtime_scorer_shadow_scores", scores_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        if torch_checkpoint_path is not None and torch_checkpoint_path.exists():
            runner.register_checkpoint(
                build_checkpoint_record(
                    checkpoint_id="semantic_runtime_scorer_latest",
                    model_family="semantic_runtime_scorer",
                    model_version="semantic_runtime_scorer_v1",
                    path=torch_checkpoint_path,
                    step=max(1, args.epochs),
                    epoch=args.epochs,
                    metadata={
                        "config_digest": config_digest,
                        "dataset_digest": dataset_summary.get("dataset_digest"),
                        "trainer_mode": args.trainer,
                    },
                )
            )
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plan_sha = sha256_json(
        {
            "training_kind": "semantic_runtime_scorers",
            "replay_dataset": str(Path(args.replay_dataset).resolve()),
            "max_counterfactuals": args.max_counterfactuals,
            "trainer": args.trainer,
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

    replay_bundle = load_replay_dataset(args.replay_dataset)
    result = run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            num_episodes=max(1, len(replay_bundle.episodes)),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="semantic_runtime_scorers",
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
