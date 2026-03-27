#!/usr/bin/env python3
"""Train the shadow replay BC policy under the canonical regal-aware runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.replay_policy_trainer import load_training_config, train_replay_policy
from src.objectives.profile_loader import load_contract_profile
from src.orchestrator.shadow_advisory import build_shadow_advisory_output
from src.regality.promotion_policy import load_regal_promotion_policy
from src.regality.promotion_reporting import (
    build_promotion_evidence_report,
    write_promotion_evidence_report,
)
from src.replay.dataset import load_replay_dataset
from src.replay.receipt_ingest import resolve_receipt_label_bundle, write_receipt_label_bundle
from src.rl.episode_sampling import DataPackRLSampler, replay_episode_to_rl_episode_descriptor
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.training.training_manifest import (
    build_replay_dataset_summary,
    build_replay_trajectory_audits,
    build_source_domain_coverage,
)
from src.utils.config_digest import sha256_json


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train replay BC policy with regal-aware runtime")
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--objective-profile", type=str, default="balanced_contract")
    parser.add_argument("--promotion-policy", type=str, default="configs/regality/promotion_default.yaml")
    parser.add_argument("--receipt-label-dir", type=str, default=None)
    parser.add_argument("--receipt-label-mode", type=str, default="synthetic_shadow")
    parser.add_argument("--queue-selection-mode", type=str, default="compare_only")
    parser.add_argument("--queue-strategy", type=str, default="balanced")
    parser.add_argument("--max-queue-episodes", type=int, default=None)
    parser.add_argument("--queue-max-upweight", type=float, default=2.0)
    parser.add_argument("--queue-max-downweight", type=float, default=0.5)
    parser.add_argument("--queue-allow-slice-removal-on-integrity-failure", action="store_true")
    parser.add_argument("--queue-policy-helper-mode", type=str, default="disabled")
    parser.add_argument("--queue-policy-package-path", type=str, default=None)
    parser.add_argument("--sampler-policy-helper-mode", type=str, default="disabled")
    parser.add_argument("--sampler-policy-package-path", type=str, default=None)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _select_episode_ids(
    *,
    dataset_dir: str,
    dataset,
    output_root: Path,
    args: argparse.Namespace,
    receipt_label_dir: str,
) -> tuple[list[str], Dict[str, str], Dict[str, Any], Dict[str, Any]]:
    advisory = build_shadow_advisory_output(
        replay_dataset_dir=dataset_dir,
        promotion_policy_path=args.promotion_policy,
        receipt_label_dir=receipt_label_dir,
        receipt_label_mode=args.receipt_label_mode,
    )
    advisory_path = output_root / "shadow_advisory.json"
    queue_path = output_root / "live_queue_selection.json"
    scorer_preconditions_path = output_root / "semantic_runtime_scorer_preconditions.json"
    scorer_work_orders_path = output_root / "semantic_runtime_scorer_work_orders.json"
    inferential_summary_path = output_root / "inferential_learnability_summary.json"
    inferential_work_orders_path = output_root / "inferential_work_orders.json"
    _write_json(advisory_path, advisory)
    _write_json(queue_path, advisory["live_queue_selection"])
    _write_json(scorer_preconditions_path, advisory["semantic_runtime_scorer_preconditions"])
    _write_json(scorer_work_orders_path, {"work_orders": advisory["semantic_runtime_scorer_work_orders"]})
    _write_json(inferential_summary_path, advisory["inferential_learnability_summary"])
    _write_json(inferential_work_orders_path, {"work_orders": advisory["inferential_work_orders"]})

    descriptors = [replay_episode_to_rl_episode_descriptor(episode) for episode in dataset.episodes]
    sampler = DataPackRLSampler(
        existing_descriptors=descriptors,
        live_queue_selection=advisory["live_queue_selection"],
        queue_dispatch_mode=args.queue_selection_mode,
        queue_max_upweight=args.queue_max_upweight,
        queue_max_downweight=args.queue_max_downweight,
        queue_allow_slice_removal_on_integrity_failure=args.queue_allow_slice_removal_on_integrity_failure,
        queue_policy_helper_mode=args.queue_policy_helper_mode,
        queue_policy_package_path=args.queue_policy_package_path,
        sampler_policy_helper_mode=args.sampler_policy_helper_mode,
        sampler_policy_package_path=args.sampler_policy_package_path,
    )
    dispatch = sampler.dispatch_queue(
        batch_size=args.max_queue_episodes or len(descriptors),
        seed=int(load_training_config(args.config).get("training", {}).get("seed", 42)),
        strategy=args.queue_strategy,
    )
    dispatch_path = output_root / "queue_dispatch_comparison.json"
    sampler_policy_receipt_path = output_root / "sampler_policy_receipt.json"
    _write_json(dispatch_path, dispatch)
    _write_json(
        sampler_policy_receipt_path,
        dict(dispatch.get("sampler_policy_receipt", sampler.last_sampler_policy_artifact or {})),
    )
    selected_episode_ids = [
        str(row.get("pack_id") or row.get("episode_id"))
        for row in dispatch.get("ordered_descriptors", [])
        if str(row.get("pack_id") or row.get("episode_id"))
    ]
    return selected_episode_ids, {
        "shadow_advisory": str(advisory_path),
        "live_queue_selection": str(queue_path),
        "semantic_runtime_scorer_preconditions": str(scorer_preconditions_path),
        "semantic_runtime_scorer_work_orders": str(scorer_work_orders_path),
        "inferential_learnability_summary": str(inferential_summary_path),
        "inferential_work_orders": str(inferential_work_orders_path),
        "queue_dispatch_comparison": str(dispatch_path),
        "sampler_policy_receipt": str(sampler_policy_receipt_path),
    }, advisory, dispatch


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_replay_dataset(args.dataset_dir)
    config = load_training_config(args.config)
    seed = int(config.get("training", {}).get("seed", 42))
    contract = load_contract_profile(args.objective_profile)
    promotion_policy = load_regal_promotion_policy(args.promotion_policy)
    receipt_bundle = resolve_receipt_label_bundle(
        dataset=dataset,
        receipt_label_dir=args.receipt_label_dir,
        allow_synthetic=True,
        label_mode=args.receipt_label_mode,
    )
    receipt_paths = write_receipt_label_bundle(receipt_bundle, output_root / "receipt_labels")
    selected_episode_ids, queue_paths, advisory, dispatch = _select_episode_ids(
        dataset_dir=args.dataset_dir,
        dataset=dataset,
        output_root=output_root,
        args=args,
        receipt_label_dir=str(Path(receipt_paths["bundle"]).parent),
    )
    report = build_promotion_evidence_report(
        dataset=dataset,
        promotion_policy=promotion_policy,
        receipt_bundle=receipt_bundle,
        evidence_pointers={
            "dataset_dir": args.dataset_dir,
            "queue_dispatch_comparison": queue_paths["queue_dispatch_comparison"],
        },
    )
    promotion_paths = write_promotion_evidence_report(output_root, report)
    result = train_replay_policy(
        dataset_dir=args.dataset_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume,
        episode_ids=selected_episode_ids,
    )

    if runner is not None:
        runner.set_eligible_datapacks([episode.episode_id for episode in dataset.episodes])
        runner.set_sampler_config(seed=seed, config_sha=result.config_digest)
        selected_set = set(selected_episode_ids) or {episode.episode_id for episode in dataset.episodes}
        for episode in dataset.episodes:
            if episode.episode_id in selected_set:
                runner.record_sample(episode.task_id, datapack_id=episode.episode_id, slice_id=episode.episode_id)
            else:
                runner.record_rejection(episode.episode_id, "queue_dispatch_not_selected")
        for audit in build_replay_trajectory_audits(dataset):
            if audit.episode_id in selected_set:
                runner.add_trajectory_audit(audit)
        runner.update_step(result.train_steps)
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "promotion_summary": report.summary,
                "receipt_label_coverage": receipt_bundle.coverage_summary(),
            },
            context_sha=promotion_policy.config_digest,
        )
        runner.configure_training_runtime(
            training_kind="shadow_replay_policy",
            config_path=args.config,
            config_digest=result.config_digest,
            replay_dataset_dir=args.dataset_dir,
            replay_manifest_digest=dataset.manifest.manifest_hash,
            replay_dataset_summary=build_replay_dataset_summary(dataset),
            objective_profile_snapshot=contract.to_dict(),
            promotion_policy_snapshot=promotion_policy.to_dict(),
            source_domain_coverage=build_source_domain_coverage(dataset),
            receipt_label_coverage=receipt_bundle.coverage_summary(),
            inferential_learnability_summary=dict(
                advisory.get("inferential_learnability_summary", {}) or {}
            ),
            inferential_work_order_summary=dict(
                advisory.get("adaptation_budget", {}).get("summary", {}) or {}
            ),
            artifact_schema_compatibility=list(dataset.manifest.metadata.get("schema_compatibility", []) or []),
            metadata={
                "selected_episode_count": len(selected_episode_ids),
                "queue_selection_mode": args.queue_selection_mode,
                "queue_strategy": args.queue_strategy,
                "advisory_dataset_digest": advisory.get("dataset_digest"),
            },
        )
        runner.register_artifact("replay_dataset_manifest", Path(args.dataset_dir) / "manifest.json")
        runner.register_artifact("receipt_label_bundle", receipt_paths["bundle"])
        runner.register_artifact("receipt_label_summary", receipt_paths["summary"])
        runner.register_artifact("shadow_advisory", queue_paths["shadow_advisory"])
        runner.register_artifact("live_queue_selection", queue_paths["live_queue_selection"])
        runner.register_artifact("semantic_runtime_scorer_preconditions", queue_paths["semantic_runtime_scorer_preconditions"])
        runner.register_artifact("semantic_runtime_scorer_work_orders", queue_paths["semantic_runtime_scorer_work_orders"])
        runner.register_artifact("inferential_learnability_summary", queue_paths["inferential_learnability_summary"])
        runner.register_artifact("inferential_work_orders", queue_paths["inferential_work_orders"])
        runner.register_artifact("queue_dispatch_comparison", queue_paths["queue_dispatch_comparison"])
        runner.register_artifact("sampler_policy_receipt", queue_paths["sampler_policy_receipt"])
        runner.register_artifact("regal_promotion_eval", promotion_paths["json"])
        runner.register_artifact("regal_promotion_eval_markdown", promotion_paths["markdown"])
        runner.register_artifact("replay_policy_metrics", result.metrics_path)
        runner.register_artifact("replay_policy_summary", result.summary_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="replay_policy_latest",
                model_family="shadow_replay_policy",
                model_version="replay_policy_bc_v1",
                path=result.checkpoint_path,
                step=result.train_steps,
                epoch=result.epochs,
                metadata={"config_digest": result.config_digest, "dataset_digest": result.dataset_digest},
            )
        )
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="replay_policy_best",
                model_family="shadow_replay_policy",
                model_version="replay_policy_bc_v1",
                path=result.best_checkpoint_path,
                step=result.train_steps,
                epoch=result.epochs,
                is_best=True,
                metadata={"best_val_mse": result.best_val_mse, "dataset_digest": result.dataset_digest},
            )
        )

    payload = {
        "training_result": result.to_dict(),
        "selected_episode_ids": selected_episode_ids,
        "receipt_label_coverage": receipt_bundle.coverage_summary(),
        "queue_dispatch_summary": dict(dispatch.get("summary", {})),
        "promotion_summary": report.summary,
        "artifacts": {
            **receipt_paths,
            **queue_paths,
            **promotion_paths,
            "replay_policy_summary": result.summary_path,
            "replay_policy_metrics": result.metrics_path,
        },
    }
    _write_json(output_root / "training_job_result.json", payload)
    if runner is not None:
        runner.register_artifact("training_job_result", output_root / "training_job_result.json")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_training_config(args.config)
    seed = int(config.get("training", {}).get("seed", 42))
    dataset = load_replay_dataset(args.dataset_dir)
    if args.skip_regal_runner:
        payload = _run_training(args, runner=None)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    runner_result_holder: Dict[str, Any] = {}

    def _wrapped_training(runner: RegalTrainingRunner) -> None:
        runner_result_holder["payload"] = _run_training(args, runner=runner)

    plan_sha = sha256_json(
        {
            "training_kind": "shadow_replay_policy",
            "dataset_digest": dataset.manifest.dataset_digest,
            "config_digest": sha256_json(config),
            "objective_profile": args.objective_profile,
            "promotion_policy": args.promotion_policy,
            "queue_selection_mode": args.queue_selection_mode,
            "queue_strategy": args.queue_strategy,
        }
    )
    result = run_training_with_regality(
        training_fn=_wrapped_training,
        config=TrainingRunConfig(
            output_dir=args.output_dir,
            seed=seed,
            num_episodes=dataset.manifest.num_episodes,
            training_steps=int(config.get("training", {}).get("epochs", 0)),
        ),
        plan_sha=plan_sha,
        plan_id="shadow_replay_policy",
    )
    payload = {
        "training_run": result.to_dict(),
        "job": runner_result_holder.get("payload", {}),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
