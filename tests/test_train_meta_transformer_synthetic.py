import json
from argparse import Namespace
from pathlib import Path

from scripts.train_meta_transformer_synthetic import _run_training
from src.orchestrator.meta_transformer_training import (
    generate_meta_transformer_dataset,
    save_meta_transformer_dataset,
)
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


def _write_runtime_export(root: Path, sample_count: int = 8) -> Path:
    export_dir = root / "runtime_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = export_dir / "meta_transformer_runtime_dataset.json"
    summary_path = export_dir / "semantic_runtime_learning_summary.json"
    save_meta_transformer_dataset(generate_meta_transformer_dataset(sample_count), str(dataset_path))
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "semantic_runtime_learning_summary_v1",
                "total_rows": sample_count,
                "route_success_count": sample_count // 2,
                "authority_success_count": sample_count // 2,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return export_dir


def test_meta_transformer_training_emits_artifacts(tmp_path: Path) -> None:
    export_dir = _write_runtime_export(tmp_path, sample_count=6)
    output_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    args = Namespace(
        runtime_export_dir=str(export_dir),
        dataset_json=None,
        runtime_summary_json=None,
        synthetic_samples=0,
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        run_name="meta_unit",
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        max_semantic_tokens=8,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        val_fraction=0.25,
        seed=5,
        skip_regal_runner=True,
    )

    result = _run_training(args, runner=None)

    assert Path(result["checkpoint"]).exists()
    assert Path(result["best_checkpoint"]).exists()
    assert Path(result["dataset_summary"]).exists()
    assert Path(result["training_summary"]).exists()
    assert (output_dir / "training_job_result.json").exists()
    assert result["benchmark_gate_ready"] is False

    summary = json.loads(Path(result["dataset_summary"]).read_text(encoding="utf-8"))
    assert summary["dataset_source"] == "semantic_runtime_export"
    assert summary["runtime_summary"]["total_rows"] == 6


def test_meta_transformer_training_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    export_dir = _write_runtime_export(tmp_path, sample_count=6)
    output_dir = tmp_path / "runner_results"
    checkpoint_dir = tmp_path / "runner_checkpoints"
    args = Namespace(
        runtime_export_dir=str(export_dir),
        dataset_json=None,
        runtime_summary_json=None,
        synthetic_samples=0,
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        run_name="meta_runner",
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        max_semantic_tokens=8,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        val_fraction=0.25,
        seed=9,
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=9,
            num_episodes=1,
            training_steps=1,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="meta_transformer_runtime_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((output_dir / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert manifest["training_kind"] == "meta_transformer_runtime"
    assert manifest["artifact_paths"]["meta_transformer_training_summary"].endswith("meta_transformer_training_summary.json")
    assert manifest["metadata"]["trajectory_audit_kind"] == "meta_transformer_sample_projection"
    assert checkpoint_registry["checkpoints"][0]["model_family"] == "meta_transformer"
    assert holder["payload"]["benchmark_gate_ready"] is False
