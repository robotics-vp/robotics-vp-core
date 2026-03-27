#!/usr/bin/env python3
"""Train the bounded D4 knob calibration helper with canonical runtime artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.regal.knob_model_training import (
    KNOB_FEATURE_NAMES,
    TORCH_AVAILABLE,
    KnobTrainingRow,
    build_knob_feature_vector,
    build_knob_training_dataset,
    generate_synthetic_knob_training_rows,
    load_knob_training_dataset,
    save_knob_training_dataset,
    target_triplet_from_policy,
    train_knob_calibration_model,
)
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit

if TORCH_AVAILABLE:  # pragma: no branch
    import torch
else:  # pragma: no cover - explicit failures in caller
    torch = None


KNOB_MODEL_MIN_ROWS = 96
KNOB_MODEL_MIN_RUNTIME_RECEIPTS = 24
KNOB_MODEL_MIN_PROMOTED_ROWS = 8


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Path to a knob training dataset JSON payload",
    )
    parser.add_argument(
        "--receipt-path",
        type=str,
        default=None,
        help="Path to knob-policy receipt JSON or JSONL exported from runtime/smoke runs",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=0,
        help="Append heuristic-bootstrap synthetic rows when positive",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--save-dir", type=str, default="checkpoints/knob_model")
    parser.add_argument("--run-name", type=str, default="knob_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _load_receipt_payloads(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Knob receipt path not found: {path}")
    if path.suffix == ".jsonl":
        payloads: list[Mapping[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                payloads.append(payload)
        return payloads

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        receipts = payload.get("receipts")
        if isinstance(receipts, list):
            return [item for item in receipts if isinstance(item, Mapping)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    raise ValueError(f"Unsupported knob receipt payload in {path}")


def _coerce_row_from_receipt(payload: Mapping[str, Any], idx: int) -> Optional[KnobTrainingRow]:
    regime_features = payload.get("regime_features")
    base_config = payload.get("base_config")
    target_policy = (
        payload.get("target_policy")
        or payload.get("applied_policy")
        or payload.get("knob_policy")
        or payload.get("policy")
    )
    if not isinstance(regime_features, Mapping) or not isinstance(base_config, Mapping):
        return None
    if not isinstance(target_policy, Mapping):
        return None

    row_id = str(
        payload.get("receipt_id")
        or payload.get("row_id")
        or payload.get("run_id")
        or f"receipt_row_{idx}"
    )
    promotion_stage = str(
        payload.get("promotion_stage")
        or target_policy.get("promotion_stage")
        or "shadow_candidate"
    )
    target_source = str(payload.get("target_source") or "runtime_receipt")
    metadata = dict(payload.get("metadata", {}) or {})
    if "knob_policy_sha" in payload:
        metadata.setdefault("knob_policy_sha", payload.get("knob_policy_sha"))
    if "run_manifest_path" in payload:
        metadata.setdefault("run_manifest_path", payload.get("run_manifest_path"))
    return KnobTrainingRow(
        row_id=row_id,
        regime_features=dict(regime_features),
        base_config=dict(base_config),
        target_policy=dict(target_policy),
        target_source=target_source,
        promotion_stage=promotion_stage,
        metadata=metadata,
    )


def _load_rows_from_receipts(path: str | Path) -> list[KnobTrainingRow]:
    payloads = _load_receipt_payloads(Path(path))
    rows: list[KnobTrainingRow] = []
    for idx, payload in enumerate(payloads):
        row = _coerce_row_from_receipt(payload, idx)
        if row is not None:
            rows.append(row)
    return rows


def _dedupe_rows(rows: Iterable[KnobTrainingRow]) -> list[KnobTrainingRow]:
    deduped: Dict[str, KnobTrainingRow] = {}
    for row in rows:
        deduped[row.row_id] = row
    return list(deduped.values())


def _build_dataset_rows(args: argparse.Namespace) -> list[KnobTrainingRow]:
    rows: list[KnobTrainingRow] = []
    if args.dataset_json:
        rows.extend(load_knob_training_dataset(args.dataset_json).rows)
    if args.receipt_path:
        rows.extend(_load_rows_from_receipts(args.receipt_path))
    if int(args.synthetic_samples) > 0:
        rows.extend(
            generate_synthetic_knob_training_rows(
                int(args.synthetic_samples),
                seed=int(args.seed),
            )
        )
    rows = _dedupe_rows(rows)
    if not rows:
        raise ValueError(
            "No knob training rows found; provide --dataset-json, --receipt-path, or --synthetic-samples"
        )
    return rows


def _build_dataset_summary(
    rows: Sequence[KnobTrainingRow],
    dataset_path: Path,
    *,
    dataset_json: Optional[str],
    receipt_path: Optional[str],
    synthetic_samples: int,
) -> dict[str, Any]:
    dataset = build_knob_training_dataset(rows)
    target_source_counts = dict(dataset.summary.get("target_source_counts", {}) or {})
    promotion_stage_counts = dict(dataset.summary.get("promotion_stage_counts", {}) or {})
    runtime_receipt_rows = int(target_source_counts.get("runtime_receipt", 0))
    synthetic_bootstrap_rows = int(target_source_counts.get("heuristic_bootstrap", 0))
    promoted_rows = int(promotion_stage_counts.get("promoted", 0))
    benchmark_gate_ready = (
        len(rows) >= KNOB_MODEL_MIN_ROWS
        and runtime_receipt_rows >= KNOB_MODEL_MIN_RUNTIME_RECEIPTS
        and promoted_rows >= KNOB_MODEL_MIN_PROMOTED_ROWS
    )
    return {
        "schema_version": "knob_model_dataset_summary_v1",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_digest": dataset.summary.get("dataset_digest"),
        "num_rows": len(rows),
        "feature_dim": len(KNOB_FEATURE_NAMES),
        "target_source_counts": target_source_counts,
        "promotion_stage_counts": promotion_stage_counts,
        "runtime_receipt_rows": runtime_receipt_rows,
        "synthetic_bootstrap_rows": synthetic_bootstrap_rows,
        "input_sources": {
            "dataset_json": dataset_json,
            "receipt_path": receipt_path,
            "synthetic_samples": int(synthetic_samples),
        },
        "benchmark_gate": {
            "name": "knob_model_runtime_receipt_density",
            "ready": benchmark_gate_ready,
            "required_rows": KNOB_MODEL_MIN_ROWS,
            "required_runtime_receipt_rows": KNOB_MODEL_MIN_RUNTIME_RECEIPTS,
            "required_promoted_rows": KNOB_MODEL_MIN_PROMOTED_ROWS,
            "observed_rows": len(rows),
            "observed_runtime_receipt_rows": runtime_receipt_rows,
            "observed_promoted_rows": promoted_rows,
        },
    }


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::training_dataset_present": int(bool(dataset_summary.get("dataset_path"))),
        "dataset::non_empty": int(int(dataset_summary.get("num_rows", 0)) > 0),
        "dataset::runtime_receipts_present": int(
            int(dataset_summary.get("runtime_receipt_rows", 0))
            >= int(benchmark_gate.get("required_runtime_receipt_rows", 0))
        ),
        "dataset::promoted_support_present": int(
            int(benchmark_gate.get("observed_promoted_rows", 0))
            >= int(benchmark_gate.get("required_promoted_rows", 0))
        ),
        "benchmark::knob_model_runtime_receipt_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "knob_model_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "knob_model_config_v1",
        "input_dim": len(KNOB_FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "targets": [
            "gain_multiplier_override",
            "conservative_multiplier_override",
            "patience_override",
        ],
        "target_contract": "bounded_knob_calibration_triplet_v1",
    }


def _evaluate_train_metrics(model: Any, rows: Sequence[KnobTrainingRow]) -> dict[str, float]:
    if torch is None:
        return {
            "normalized_gain_mse": 0.0,
            "normalized_conservative_mse": 0.0,
            "normalized_patience_mse": 0.0,
        }
    X = np.asarray(
        [build_knob_feature_vector(row.regime_features, row.base_config) for row in rows],
        dtype=np.float32,
    )
    y = np.asarray(
        [target_triplet_from_policy(row.target_policy, row.base_config) for row in rows],
        dtype=np.float32,
    )
    with torch.no_grad():
        predictions = torch.sigmoid(model(torch.from_numpy(X))).cpu().numpy()
    mse = np.mean((predictions - y) ** 2, axis=0)
    return {
        "normalized_gain_mse": float(mse[0]),
        "normalized_conservative_mse": float(mse[1]),
        "normalized_patience_mse": float(mse[2]),
    }


def _build_runtime_package(
    *,
    config_digest: str,
    checkpoint_path: Path,
    dataset_summary: Mapping[str, Any],
    model_config: Mapping[str, Any],
    execution_preconditions: Mapping[str, Any],
    dataset_path: Path,
    dataset_summary_path: Path,
    model_config_path: Path,
    preconditions_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    benchmark_gate_ready = bool(benchmark_gate.get("ready", False))
    return {
        "schema_version": "knob_model_runtime_package_v1",
        "package_id": f"knob_model_{config_digest[:12]}",
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "dataset_summary_path": str(dataset_summary_path),
        "model_config_path": str(model_config_path),
        "preconditions_path": str(preconditions_path),
        "training_summary_path": str(training_summary_path),
        "model_config": dict(model_config),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "inference_contract": {
            "helper_blend_policy": {
                "shadow_candidate_helper_weight": 0.2,
                "promoted_helper_weight": 0.55,
            },
            "feature_names": list(KNOB_FEATURE_NAMES),
            "targets": list(model_config.get("targets", [])),
            "runtime_receipt_contract": "knob_policy_receipt_v1",
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "bounded_knob_calibration_triplet_v1",
            "runtime_targets": [
                "homeostatic_plan_writer",
                "run_closed_loop_smoke",
            ],
        },
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    history: Mapping[str, Any],
    train_metrics: Mapping[str, float],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "knob_model_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "num_rows": int(dataset_summary.get("num_rows", 0)),
        "train_loss": float(list(history.get("loss", [0.0]))[-1] if history.get("loss") else 0.0),
        "train_metrics": dict(train_metrics),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(rows: Sequence[KnobTrainingRow]) -> list[Any]:
    audits: list[Any] = []
    for row in rows:
        target_policy = dict(row.target_policy)
        rewards = [
            float(target_policy.get("gain_multiplier_override") or 1.0),
            float(target_policy.get("conservative_multiplier_override") or 1.0),
            float(target_policy.get("patience_override") or 1.0),
        ]
        reward_components = {
            "gain_multiplier_override": [rewards[0]],
            "conservative_multiplier_override": [rewards[1]],
            "patience_override": [rewards[2]],
        }
        audits.append(
            create_trajectory_audit(
                episode_id=row.row_id,
                num_steps=1,
                rewards=[sum(rewards) / float(len(rewards))],
                reward_components=reward_components,
                events=[
                    f"target_source:{row.target_source}",
                    f"promotion_stage:{row.promotion_stage}",
                    "knob_model_training_row",
                ],
            )
        )
    return audits


def _train(
    *,
    args: argparse.Namespace,
    runner: Optional[RegalTrainingRunner],
) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required to train the knob model")

    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _build_dataset_rows(args)
    dataset = build_knob_training_dataset(rows)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = output_root / "knob_model_dataset.json"
    dataset_summary_path = output_root / "knob_model_dataset_summary.json"
    model_config_path = output_root / "knob_model_config.json"
    preconditions_path = output_root / "knob_model_execution_preconditions.json"
    training_summary_path = output_root / "knob_model_training_summary.json"
    runtime_package_path = output_root / "knob_model_package.json"
    checkpoint_path = output_root / "knob_model.pt"
    training_job_result_path = output_root / "training_job_result.json"

    save_knob_training_dataset(dataset, dataset_path)
    dataset_summary = _build_dataset_summary(
        rows,
        dataset_path,
        dataset_json=args.dataset_json,
        receipt_path=args.receipt_path,
        synthetic_samples=args.synthetic_samples,
    )
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    model_config = _build_model_config(hidden_dim=args.hidden_dim)
    config_digest = sha256_json(
        {
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "run_name": args.run_name,
        }
    )

    model, history = train_knob_calibration_model(
        rows,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        save_path=str(checkpoint_path),
    )
    train_metrics = _evaluate_train_metrics(model, rows)

    artifacts = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "runtime_package": str(runtime_package_path),
    }
    training_summary = _build_training_summary(
        run_name=args.run_name,
        seed=args.seed,
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        history=history,
        train_metrics=train_metrics,
        artifacts=artifacts,
    )
    runtime_package = _build_runtime_package(
        config_digest=config_digest,
        checkpoint_path=checkpoint_path,
        dataset_summary=dataset_summary,
        model_config=model_config,
        execution_preconditions=execution_preconditions,
        dataset_path=dataset_path,
        dataset_summary_path=dataset_summary_path,
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
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "training_summary": str(training_summary_path),
        "runtime_package": str(runtime_package_path),
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }
    _write_json(
        training_job_result_path,
        {
            "training_kind": "knob_model",
            "result": result,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        runner.set_eligible_datapacks([row.row_id for row in rows])
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for row in rows:
            runner.record_sample(
                row.target_source,
                datapack_id=row.row_id,
                slice_id=f"{row.target_source}:{row.row_id}",
            )
        for audit in _build_trajectory_audits(rows):
            runner.add_trajectory_audit(audit)
        runner.update_step(max(1, args.epochs))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "num_rows": len(rows),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="knob_model",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "knob_calibration"},
            promotion_policy_snapshot={
                "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {})
            },
            source_domain_coverage={
                "target_source_counts": dict(dataset_summary.get("target_source_counts", {}) or {})
            },
            receipt_label_coverage={
                "promotion_stage_counts": dict(dataset_summary.get("promotion_stage_counts", {}) or {})
            },
            metadata={
                "trajectory_audit_kind": "knob_policy_projection",
                "receipt_path": args.receipt_path,
                "synthetic_samples": int(args.synthetic_samples),
                "target_contract": "bounded_knob_calibration_triplet_v1",
            },
        )
        runner.register_artifact("knob_model_dataset", dataset_path)
        runner.register_artifact("knob_model_dataset_summary", dataset_summary_path)
        runner.register_artifact("knob_model_config", model_config_path)
        runner.register_artifact("knob_model_preconditions", preconditions_path)
        runner.register_artifact("knob_model_training_summary", training_summary_path)
        runner.register_artifact("knob_model_runtime_package", runtime_package_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="knob_model_latest",
                model_family="knob_model",
                model_version="knob_model_v1",
                path=checkpoint_path,
                step=max(1, args.epochs),
                epoch=args.epochs,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "train_metrics": dict(train_metrics),
                },
            )
        )
    return result


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    return _train(args=args, runner=runner)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    estimated_rows = len(_build_dataset_rows(args))
    plan_sha = sha256_json(
        {
            "training_kind": "knob_model",
            "dataset_json": args.dataset_json,
            "receipt_path": args.receipt_path,
            "synthetic_samples": args.synthetic_samples,
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
            output_dir=args.save_dir,
            seed=args.seed,
            num_episodes=max(1, estimated_rows),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="knob_model",
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
