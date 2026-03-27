#!/usr/bin/env python3
"""Train the sim/synth/physics backend selector with canonical runtime artifacts."""

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

from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.world_model.sim_synth_physics.backend_selector import (
    BACKEND_LABELS,
    FIDELITY_LABELS,
    RANDOMIZATION_LABELS,
    TORCH_AVAILABLE,
    train_backend_selector,
)
from src.world_model.sim_synth_physics.training_corpus import (
    build_backend_selector_rows_from_receipts,
    harvest_sim_synth_receipt_bundles,
    load_sim_synth_receipt_bundles,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


BACKEND_SELECTOR_MIN_ROWS = 100
BACKEND_SELECTOR_MIN_BACKENDS = 2
BACKEND_SELECTOR_MIN_FIDELITIES = 2
BACKEND_SELECTOR_MIN_RUNTIME_RECEIPTS = 24


def _artifact_ref(path: Path, *, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None, help="JSONL or JSON dataset path")
    parser.add_argument(
        "--receipt-path",
        type=str,
        default=None,
        help="JSON or JSONL runtime receipt bundles exported from sim/synth/physics WM runs",
    )
    parser.add_argument(
        "--receipt-dir",
        action="append",
        default=None,
        help="Directory to auto-harvest live sim/synth runtime receipts from (may be repeated)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="checkpoints/sim_synth_backend_selector")
    parser.add_argument("--run-name", type=str, default="sim_synth_backend_selector")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if dataset_path.suffix == ".json":
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        raise ValueError("Expected dataset JSON to contain a list of rows")
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
    return rows


def _resolve_receipt_bundles(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if getattr(args, "receipt_path", None):
        bundles = load_sim_synth_receipt_bundles(args.receipt_path)
        return bundles, {
            "receipt_path": getattr(args, "receipt_path", None),
            "receipt_dirs": [],
            "receipt_source_kind": "explicit_bundle",
        }
    receipt_dirs = [str(item) for item in list(getattr(args, "receipt_dir", []) or []) if str(item).strip()]
    if not receipt_dirs and getattr(args, "dataset", None) is None:
        receipt_dirs = [str(Path.cwd()), str(Path(getattr(args, "save_dir")).resolve())]
    if not receipt_dirs:
        return [], {
            "receipt_path": None,
            "receipt_dirs": [],
            "receipt_source_kind": "none",
        }
    bundles = harvest_sim_synth_receipt_bundles(receipt_dirs)
    return bundles, {
        "receipt_path": None,
        "receipt_dirs": receipt_dirs,
        "receipt_source_kind": "harvested_runtime_receipts",
    }


def _build_dataset_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    receipt_source = {
        "receipt_path": None,
        "receipt_dirs": [],
        "receipt_source_kind": "none",
        "receipt_bundle_count": 0,
    }
    if getattr(args, "dataset", None):
        rows.extend(_load_rows(args.dataset))
    bundles, receipt_source = _resolve_receipt_bundles(args)
    if bundles:
        rows.extend(build_backend_selector_rows_from_receipts(bundles))
        receipt_source["receipt_bundle_count"] = len(bundles)
    if not rows:
        raise ValueError(
            "No backend-selector training rows found; provide --dataset, --receipt-path, or --receipt-dir"
        )
    return rows, receipt_source


def _build_dataset_summary(
    rows: Sequence[Mapping[str, Any]],
    dataset_path: Path,
    *,
    dataset_arg: Optional[str],
    receipt_source: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    backend_counts = Counter(
        str(row.get("target_backend", row.get("heuristic_backend", "other")) or "other")
        for row in rows
    )
    fidelity_counts = Counter(
        str(
            row.get("target_fidelity_tier", row.get("heuristic_fidelity_tier", "branch_balanced"))
            or "branch_balanced"
        )
        for row in rows
    )
    randomization_counts = Counter(
        str(
            row.get(
                "target_domain_randomization_regime",
                row.get("heuristic_domain_randomization_regime", "steady_state"),
            )
            or "steady_state"
        )
        for row in rows
    )
    target_source_counts = Counter(str(row.get("target_source", "dataset_rows") or "dataset_rows") for row in rows)
    runtime_receipt_rows = int(target_source_counts.get("runtime_receipt", 0))
    benchmark_gate_ready = (
        len(rows) >= BACKEND_SELECTOR_MIN_ROWS
        and len([label for label in backend_counts if label in BACKEND_LABELS]) >= BACKEND_SELECTOR_MIN_BACKENDS
        and len([label for label in fidelity_counts if label in FIDELITY_LABELS]) >= BACKEND_SELECTOR_MIN_FIDELITIES
        and runtime_receipt_rows >= BACKEND_SELECTOR_MIN_RUNTIME_RECEIPTS
    )
    return {
        "schema_version": "sim_synth_backend_selector_dataset_summary_v1",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_digest": sha256_json(
            {
                "dataset_path": str(dataset_path.resolve()),
                "row_count": len(rows),
                "backend_counts": dict(sorted(backend_counts.items())),
                "fidelity_counts": dict(sorted(fidelity_counts.items())),
                "randomization_counts": dict(sorted(randomization_counts.items())),
            }
        ),
        "row_count": len(rows),
        "backend_counts": dict(sorted(backend_counts.items())),
        "fidelity_counts": dict(sorted(fidelity_counts.items())),
        "randomization_counts": dict(sorted(randomization_counts.items())),
        "target_source_counts": dict(sorted(target_source_counts.items())),
        "runtime_receipt_rows": runtime_receipt_rows,
        "input_sources": {
            "dataset": dataset_arg,
            "receipt_path": None if receipt_source is None else receipt_source.get("receipt_path"),
            "receipt_dirs": [] if receipt_source is None else list(receipt_source.get("receipt_dirs", []) or []),
            "receipt_source_kind": "none"
            if receipt_source is None
            else str(receipt_source.get("receipt_source_kind", "none")),
            "receipt_bundle_count": 0
            if receipt_source is None
            else int(receipt_source.get("receipt_bundle_count", 0) or 0),
        },
        "benchmark_gate": {
            "name": "sim_synth_backend_selector_dataset_density",
            "ready": benchmark_gate_ready,
            "required_rows": BACKEND_SELECTOR_MIN_ROWS,
            "required_distinct_backends": BACKEND_SELECTOR_MIN_BACKENDS,
            "required_distinct_fidelities": BACKEND_SELECTOR_MIN_FIDELITIES,
            "required_runtime_receipt_rows": BACKEND_SELECTOR_MIN_RUNTIME_RECEIPTS,
            "observed_rows": len(rows),
            "observed_distinct_backends": len([label for label in backend_counts if label in BACKEND_LABELS]),
            "observed_distinct_fidelities": len([label for label in fidelity_counts if label in FIDELITY_LABELS]),
            "observed_runtime_receipt_rows": runtime_receipt_rows,
        },
    }


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::dataset_present": int(bool(dataset_summary.get("dataset_path"))),
        "dataset::non_empty": int(int(dataset_summary.get("row_count", 0)) > 0),
        "dataset::backend_diversity": int(
            int(benchmark_gate.get("observed_distinct_backends", 0))
            >= int(benchmark_gate.get("required_distinct_backends", 0))
        ),
        "dataset::fidelity_diversity": int(
            int(benchmark_gate.get("observed_distinct_fidelities", 0))
            >= int(benchmark_gate.get("required_distinct_fidelities", 0))
        ),
        "dataset::runtime_receipts_present": int(
            int(benchmark_gate.get("observed_runtime_receipt_rows", 0))
            >= int(benchmark_gate.get("required_runtime_receipt_rows", 0))
        ),
        "benchmark::backend_selector_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "sim_synth_backend_selector_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "sim_synth_backend_selector_model_config_v1",
        "hidden_dim": int(hidden_dim),
        "backend_labels": list(BACKEND_LABELS),
        "fidelity_labels": list(FIDELITY_LABELS),
        "randomization_labels": list(RANDOMIZATION_LABELS),
        "target_contract": "backend_fidelity_randomization_selection_v1",
    }


def _evaluate_train_accuracy(model: Any, rows: Sequence[Mapping[str, Any]]) -> float:
    if torch is None:
        return 0.0
    correct = 0
    total = 0
    for row in rows:
        prediction = model.predict_context(context=row)
        target_backend = str(row.get("target_backend", row.get("heuristic_backend", "other")) or "other")
        target_fidelity = str(
            row.get("target_fidelity_tier", row.get("heuristic_fidelity_tier", "branch_balanced"))
            or "branch_balanced"
        )
        target_randomization = str(
            row.get(
                "target_domain_randomization_regime",
                row.get("heuristic_domain_randomization_regime", "steady_state"),
            )
            or "steady_state"
        )
        correct += int(
            prediction["preferred_backend"] == target_backend
            and prediction["fidelity_tier"] == target_fidelity
            and prediction["domain_randomization_regime"] == target_randomization
        )
        total += 1
    return float(correct / max(total, 1))


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
        "schema_version": "sim_synth_backend_selector_runtime_package_v1",
        "package_id": f"sim_synth_backend_selector_{config_digest[:12]}",
        "checkpoint_path": _artifact_ref(checkpoint_path, base_dir=checkpoint_path.parent),
        "dataset_summary_path": _artifact_ref(dataset_summary_path, base_dir=checkpoint_path.parent),
        "model_config_path": _artifact_ref(model_config_path, base_dir=checkpoint_path.parent),
        "preconditions_path": _artifact_ref(preconditions_path, base_dir=checkpoint_path.parent),
        "training_summary_path": _artifact_ref(training_summary_path, base_dir=checkpoint_path.parent),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "inference_contract": {
            "helper_blend_policy": "bounded_backend_selector_helper_v1",
            "allowed_modes": ["disabled", "auto", "required"],
            "shadow_candidate_helper_weight": 0.12,
            "promoted_helper_weight": 0.35,
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "backend_fidelity_randomization_selection_v1",
            "routing_targets": ["sim_synth_physics.physics_context", "coverage_loop"],
            "target_hardware_class": "unitree_g1_r1_class",
            "subsystem_posture": "complete_subsystem_until_data_gpu_assets_are_bottleneck",
        },
        "model_config": dict(model_config),
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    train_accuracy: float,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "sim_synth_backend_selector_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "row_count": int(dataset_summary.get("row_count", 0)),
        "train_accuracy": float(train_accuracy),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _train(*, args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the backend selector")
    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows, receipt_source = _build_dataset_rows(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    compiled_dataset_path = output_root / "sim_synth_backend_selector_rows.jsonl"
    _write_jsonl(compiled_dataset_path, rows)
    dataset_summary = _build_dataset_summary(
        rows,
        compiled_dataset_path,
        dataset_arg=getattr(args, "dataset", None),
        receipt_source=receipt_source,
    )
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    model_config = _build_model_config(args.hidden_dim)
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

    checkpoint_path = output_root / "sim_synth_backend_selector.pt"
    dataset_summary_path = output_root / "sim_synth_backend_selector_dataset_summary.json"
    model_config_path = output_root / "sim_synth_backend_selector_model_config.json"
    preconditions_path = output_root / "sim_synth_backend_selector_execution_preconditions.json"
    training_summary_path = output_root / "sim_synth_backend_selector_training_summary.json"
    runtime_package_path = output_root / "sim_synth_backend_selector_package.json"
    training_job_result_path = output_root / "training_job_result.json"

    model = train_backend_selector(
        rows,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_path=str(checkpoint_path),
    )
    train_accuracy = _evaluate_train_accuracy(model, rows)
    artifacts = {
        "checkpoint": str(checkpoint_path),
        "compiled_dataset": str(compiled_dataset_path),
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
        train_accuracy=train_accuracy,
        artifacts=artifacts,
    )
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

    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)
    _write_json(runtime_package_path, runtime_package)

    result = {
        "checkpoint": str(checkpoint_path),
        "compiled_dataset": str(compiled_dataset_path),
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
            "training_kind": "sim_synth_backend_selector",
            "result": result,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        runner.set_eligible_datapacks([f"backend_row:{idx}" for idx, _ in enumerate(rows)])
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        runner.update_step(max(1, args.epochs))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "row_count": len(rows),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="sim_synth_backend_selector",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "sim_synth_backend_selector"},
            promotion_policy_snapshot={},
            source_domain_coverage={"backend_counts": dict(dataset_summary.get("backend_counts", {}) or {})},
            receipt_label_coverage={"rows": int(dataset_summary.get("row_count", 0))},
            metadata={
                "target_contract": "backend_fidelity_randomization_selection_v1",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
            },
        )
        runner.register_artifact("sim_synth_backend_selector_dataset_summary", dataset_summary_path)
        runner.register_artifact("sim_synth_backend_selector_compiled_dataset", compiled_dataset_path)
        runner.register_artifact("sim_synth_backend_selector_model_config", model_config_path)
        runner.register_artifact("sim_synth_backend_selector_preconditions", preconditions_path)
        runner.register_artifact("sim_synth_backend_selector_training_summary", training_summary_path)
        runner.register_artifact("sim_synth_backend_selector_runtime_package", runtime_package_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="sim_synth_backend_selector_latest",
                model_family="sim_synth_backend_selector",
                model_version="sim_synth_backend_selector_v1",
                path=checkpoint_path,
                step=max(1, args.epochs),
                epoch=args.epochs,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "train_accuracy": train_accuracy,
                },
            )
        )
    return result


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    return _train(args=args, runner=runner)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plan_sha = sha256_json(
        {
            "training_kind": "sim_synth_backend_selector",
            "dataset": args.dataset,
            "receipt_path": getattr(args, "receipt_path", None),
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "run_name": args.run_name,
            "seed": args.seed,
        }
    )
    if args.skip_regal_runner:
        print(json.dumps(_run_training(args, runner=None), indent=2, sort_keys=True))
        return

    holder: Dict[str, Any] = {}

    def _wrapped(runner: RegalTrainingRunner) -> None:
        holder["payload"] = _run_training(args, runner)

    result = run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=args.save_dir,
            seed=args.seed,
            num_episodes=max(1, len(_build_dataset_rows(args))),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="sim_synth_backend_selector",
    )
    print(json.dumps({"training_run": result.to_dict(), "job": holder.get("payload", {})}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
