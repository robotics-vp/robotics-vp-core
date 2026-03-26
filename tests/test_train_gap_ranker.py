import json
from argparse import Namespace
from pathlib import Path

from scripts.train_gap_ranker import _run_training
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


def _write_outcome_store(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "edge_key": "task:drawer -> skill:grasp",
            "fill_method": "diffusion",
            "gap_features": {"economic_priority": 0.8, "trust_priority": 0.4, "readiness": 0.3},
            "pre_evidence_count": 0,
            "post_evidence_count": 1,
            "coverage_delta": 0.2,
            "wall_time_s": 10.0,
            "quality_score": 0.7,
        },
        {
            "edge_key": "task:drawer -> risk:collision",
            "fill_method": "real_sim",
            "gap_features": {"economic_priority": 0.3, "trust_priority": 0.8, "readiness": 0.6},
            "pre_evidence_count": 0,
            "post_evidence_count": 1,
            "coverage_delta": 0.1,
            "wall_time_s": 8.0,
            "quality_score": 0.6,
        },
        {
            "edge_key": "skill:grasp -> prim:force_control",
            "fill_method": "diffusion",
            "gap_features": {"economic_priority": 0.7, "trust_priority": 0.5, "readiness": 0.4},
            "pre_evidence_count": 1,
            "post_evidence_count": 2,
            "coverage_delta": 0.05,
            "wall_time_s": 12.0,
            "quality_score": 0.5,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return path


def test_train_gap_ranker_emits_artifacts(tmp_path: Path) -> None:
    outcome_store = _write_outcome_store(tmp_path / "fill_outcomes.jsonl")
    output_dir = tmp_path / "results"
    args = Namespace(
        outcome_store=str(outcome_store),
        epochs=1,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="gap_ranker_unit",
        seed=7,
        skip_regal_runner=True,
    )

    result = _run_training(args, runner=None)

    assert Path(result["checkpoint"]).exists()
    assert Path(result["dataset_summary"]).exists()
    assert Path(result["model_config"]).exists()
    assert Path(result["preconditions"]).exists()
    assert Path(result["training_summary"]).exists()
    assert Path(result["runtime_package"]).exists()
    assert (output_dir / "training_job_result.json").exists()
    assert result["benchmark_gate_ready"] is False

    runtime_package = json.loads(Path(result["runtime_package"]).read_text(encoding="utf-8"))
    assert runtime_package["promotion_stage"] == "shadow_candidate"
    assert runtime_package["metadata"]["training_contract"] == "bounded_gap_agenda_helper_v1"


def test_train_gap_ranker_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    outcome_store = _write_outcome_store(tmp_path / "fill_outcomes.jsonl")
    output_dir = tmp_path / "runner_results"
    args = Namespace(
        outcome_store=str(outcome_store),
        epochs=1,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="gap_ranker_runner",
        seed=11,
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=11,
            num_episodes=1,
            training_steps=1,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="gap_ranker_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((output_dir / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert manifest["training_kind"] == "gap_ranker"
    assert manifest["artifact_paths"]["gap_ranker_runtime_package"].endswith("gap_ranker_package.json")
    assert manifest["metadata"]["trajectory_audit_kind"] == "gap_ranker_fill_outcome_projection"
    assert checkpoint_registry["checkpoints"][0]["model_family"] == "gap_ranker"
    assert holder["payload"]["benchmark_gate_ready"] is False
