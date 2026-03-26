import json
from argparse import Namespace
from pathlib import Path

import torch

from scripts.train_vla_recap_offline import _run_training, train_offline
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


def _write_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(8):
        rows.append(
            {
                "task_id": "recap_task",
                "episode_id": f"ep_{idx // 4}",
                "timestep": idx,
                "advantage": float((idx % 3) - 1),
                "metrics": {
                    "mpl": 40.0 + idx,
                    "energy_cost": 0.3 + 0.02 * idx,
                    "error_rate": 0.01 * (idx % 4),
                },
                "sampler_strategy": "balanced",
                "curriculum_phase": "early",
                "objective_preset": "baseline",
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def test_train_vla_recap_offline_direct_api_writes_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "recap.jsonl"
    output_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    _write_dataset(dataset_path)

    result = train_offline(
        dataset_paths=[str(dataset_path)],
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        advantage_bins=[-1.0, 0.0, 1.0],
        metrics=["mpl", "energy_cost", "error_rate"],
        num_atoms=5,
        hidden_dim=8,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        seed=11,
        log_csv=True,
        run_name="recap_unit",
    )

    assert Path(result["checkpoint"]).exists()
    assert Path(result["best_checkpoint"]).exists()
    assert Path(result["csv"]).exists()
    assert Path(result["dataset_summary"]).exists()
    assert Path(result["training_summary"]).exists()
    assert (output_dir / "training_job_result.json").exists()
    assert result["benchmark_gate_ready"] is False

    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    assert checkpoint["metrics"] == ["mpl", "energy_cost", "error_rate"]
    assert "model_state_dict" in checkpoint
    assert checkpoint["history"]

    summary = json.loads(Path(result["dataset_summary"]).read_text(encoding="utf-8"))
    assert summary["num_rows"] == 8
    assert summary["benchmark_gate"]["ready"] is False


def test_train_vla_recap_offline_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    dataset_path = tmp_path / "recap.jsonl"
    output_dir = tmp_path / "runner_results"
    checkpoint_dir = tmp_path / "runner_checkpoints"
    _write_dataset(dataset_path)

    args = Namespace(
        datasets=[str(dataset_path)],
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        advantage_bins=[-1.0, 0.0, 1.0],
        metrics=["mpl", "energy_cost", "error_rate"],
        num_atoms=5,
        hidden_dim=8,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        seed=13,
        run_name="recap_runner",
        no_csv=False,
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=13,
            num_episodes=1,
            training_steps=1,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="vla_recap_offline_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((output_dir / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert manifest["training_kind"] == "vla_recap_offline"
    assert manifest["artifact_paths"]["recap_training_summary"].endswith("recap_training_summary.json")
    assert manifest["artifact_paths"]["training_job_result"].endswith("training_job_result.json")
    assert manifest["metadata"]["trajectory_audit_kind"] == "recap_row_projection"
    assert checkpoint_registry["checkpoints"][0]["model_family"] == "vla_recap"
    assert holder["payload"]["benchmark_gate_ready"] is False
