import json
from argparse import Namespace
from pathlib import Path

from scripts.eval_orchestration_transformer import predict_tool_sequence_for_sample
from scripts.train_orchestration_transformer import _run_training
from src.orchestrator.orchestration_transformer import OrchestrationTransformer
from src.orchestrator.training_dataset import (
    context_to_sample,
    generate_synthetic_context,
    instruction_tokens_from_sample,
    load_dataset_samples,
    save_dataset,
)
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


def _write_runtime_export(root: Path, sample_count: int = 8) -> Path:
    export_dir = root / "runtime_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = export_dir / "orchestration_runtime_dataset.json"
    summary_path = export_dir / "semantic_runtime_learning_summary.json"
    samples = []
    for index in range(sample_count):
        sample = context_to_sample(generate_synthetic_context(seed=100 + index))
        sample.source_type = "semantic_runtime_corpus"
        sample.metadata.update(
            {
                "sample_id": f"runtime_sample_{index}",
                "source_domain": "workcell",
                "objective_preset": "balanced",
                "instruction_text": f"task close drawer iteration {index}",
                "runtime_instruction": f"task close drawer iteration {index}",
                "execution_mode": "bounded_execution",
            }
        )
        samples.append(sample)
    save_dataset(samples, str(dataset_path))
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


def test_orchestration_instruction_tokens_are_deterministic_and_preserved(tmp_path: Path) -> None:
    sample = context_to_sample(generate_synthetic_context(seed=7))
    sample.metadata["instruction_text"] = "task open drawer carefully"
    expected_tokens = instruction_tokens_from_sample(sample, vocab_size=64, seq_len=12).tolist()

    dataset_path = tmp_path / "dataset.json"
    save_dataset([sample], str(dataset_path))
    loaded = load_dataset_samples(str(dataset_path))[0]
    loaded_tokens = instruction_tokens_from_sample(loaded, vocab_size=64, seq_len=12).tolist()

    model = OrchestrationTransformer(
        vocab_size=64,
        hidden=32,
        ctx_dim=int(sample.context_features.shape[0]),
    )
    prediction = predict_tool_sequence_for_sample(
        model,
        loaded,
        vocab_size=64,
        instruction_seq_len=12,
    )

    assert loaded_tokens == expected_tokens
    assert prediction["instruction_tokens"] == expected_tokens
    assert isinstance(prediction["predicted_tool_sequence"], list)


def test_orchestration_training_emits_artifacts(tmp_path: Path) -> None:
    export_dir = _write_runtime_export(tmp_path, sample_count=6)
    output_dir = tmp_path / "results"
    args = Namespace(
        runtime_export_dir=str(export_dir),
        dataset_json=None,
        runtime_summary_json=None,
        num_samples=6,
        use_mixed_dataset=False,
        econ_semantic_ratio=0.5,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        hidden=32,
        ctx_dim=0,
        vocab_size=64,
        instruction_seq_len=12,
        seed=13,
        val_split=0.25,
        save_dir=str(output_dir),
        run_name="orchestration_unit",
        skip_regal_runner=True,
    )

    result = _run_training(args, runner=None)

    assert Path(result["checkpoint"]).exists()
    assert Path(result["best_checkpoint"]).exists()
    assert Path(result["dataset"]).exists()
    assert Path(result["dataset_summary"]).exists()
    assert Path(result["model_config"]).exists()
    assert Path(result["training_summary"]).exists()
    assert (output_dir / "training_job_result.json").exists()
    assert result["benchmark_gate_ready"] is False

    summary = json.loads(Path(result["dataset_summary"]).read_text(encoding="utf-8"))
    model_config = json.loads(Path(result["model_config"]).read_text(encoding="utf-8"))
    assert summary["dataset_source"] == "semantic_runtime_export"
    assert summary["runtime_summary"]["total_rows"] == 6
    assert summary["source_type_counts"]["semantic_runtime_corpus"] == 6
    assert model_config["tool_prediction_contract"] == "bounded_tool_sequence_v2"
    assert model_config["max_tool_steps"] >= 1


def test_orchestration_training_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    export_dir = _write_runtime_export(tmp_path, sample_count=6)
    output_dir = tmp_path / "runner_results"
    args = Namespace(
        runtime_export_dir=str(export_dir),
        dataset_json=None,
        runtime_summary_json=None,
        num_samples=6,
        use_mixed_dataset=False,
        econ_semantic_ratio=0.5,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        hidden=32,
        ctx_dim=0,
        vocab_size=64,
        instruction_seq_len=12,
        seed=19,
        val_split=0.25,
        save_dir=str(output_dir),
        run_name="orchestration_runner",
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=19,
            num_episodes=1,
            training_steps=1,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="orchestration_transformer_runtime_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((output_dir / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert manifest["training_kind"] == "orchestration_transformer"
    assert manifest["artifact_paths"]["orchestration_training_summary"].endswith(
        "orchestration_training_summary.json"
    )
    assert manifest["metadata"]["trajectory_audit_kind"] == "orchestration_sample_projection"
    assert checkpoint_registry["checkpoints"][0]["model_family"] == "orchestration_transformer"
    assert holder["payload"]["benchmark_gate_ready"] is False
