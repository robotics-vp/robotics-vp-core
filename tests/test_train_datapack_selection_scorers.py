import json
from argparse import Namespace
from pathlib import Path

from scripts.train_datapack_selection_scorers import _run_training
from src.orchestrator.datapack_selection_training import TORCH_AVAILABLE
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


def _write_run_log(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "scenario_id": "run_a",
            "eval_metrics": {
                "mpl_units_per_hour": 80.0,
                "wage_parity": 0.9,
                "reward_scalar_sum": 8.0,
                "error_rate": 0.01,
                "anti_reward_hacking_suspicious": 0.0,
            },
            "selection_summary": {
                "required_tags": ["warehouse", "grasp"],
                "robot_family": "G1",
                "objective_hint": "baseline",
                "candidate_count": 2,
                "selection_policy": "heuristic_plus_learned_helper",
                "selected_ids": ["dp_good"],
                "selection_helper_status": {
                    "status": "available",
                    "promotion_stage": "shadow_candidate",
                },
                "selection_context": {
                    "required_tag_count_norm": 0.25,
                    "gap_pressure": 0.5,
                    "candidate_pool_size_norm": 0.2,
                    "benchmark_ready_ratio": 0.5,
                    "execution_ready_ratio": 0.5,
                    "history_density": 0.2,
                    "cold_start_pressure": 0.8,
                    "objective_present": 1.0,
                    "robot_specificity": 1.0,
                },
                "top_candidates": [
                    {
                        "datapack_id": "dp_good",
                        "selection_features": {
                            "tag_coverage": 1.0,
                            "exact_tag_match": 1.0,
                            "gap_fill_score": 0.5,
                            "objective_match": 1.0,
                            "history_support_score": 0.5,
                            "quality_score": 0.9,
                            "novelty_score": 0.3,
                            "semantic_grounding_non_heuristic": 1.0,
                            "benchmark_eligible": 1.0,
                            "execution_ready": 1.0,
                            "cold_start_bonus": 0.0,
                            "max_arh_penalty": 0.0,
                            "mean_adjusted_mpl_norm": 0.8,
                            "mean_reward_norm": 0.7,
                            "scenario_count_norm": 0.2,
                            "eval_count_norm": 0.2,
                        },
                    },
                    {
                        "datapack_id": "dp_alt",
                        "selection_features": {
                            "tag_coverage": 1.0,
                            "exact_tag_match": 1.0,
                            "gap_fill_score": 0.0,
                            "objective_match": 1.0,
                            "history_support_score": 0.1,
                            "quality_score": 0.2,
                            "novelty_score": 0.9,
                            "semantic_grounding_non_heuristic": 0.0,
                            "benchmark_eligible": 0.0,
                            "execution_ready": 0.0,
                            "cold_start_bonus": 1.0,
                            "max_arh_penalty": 0.0,
                            "mean_adjusted_mpl_norm": 0.3,
                            "mean_reward_norm": 0.5,
                            "scenario_count_norm": 0.0,
                            "eval_count_norm": 0.0,
                        },
                    },
                ],
            },
        }
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return path


def test_train_datapack_selection_scorers_emits_artifacts(tmp_path: Path) -> None:
    run_log = _write_run_log(tmp_path / "logs" / "semantic_runs.jsonl")
    output_dir = tmp_path / "results"
    args = Namespace(
        run_logs=[str(run_log)],
        run_log_dirs=[],
        output_dir=str(output_dir),
        run_name="selection_unit",
        seed=7,
        skip_regal_runner=True,
    )

    result = _run_training(args, runner=None)

    assert Path(result["dataset"]).exists()
    assert Path(result["dataset_summary"]).exists()
    assert Path(result["model_config"]).exists()
    assert Path(result["preconditions"]).exists()
    assert Path(result["scorer_package"]).exists()
    assert Path(result["training_summary"]).exists()
    assert (output_dir / "training_job_result.json").exists()
    assert result["benchmark_gate_ready"] is False

    summary = json.loads(Path(result["dataset_summary"]).read_text(encoding="utf-8"))
    model_config = json.loads(Path(result["model_config"]).read_text(encoding="utf-8"))
    training_summary = json.loads(Path(result["training_summary"]).read_text(encoding="utf-8"))
    assert summary["num_logs"] == 1
    assert summary["num_examples"] >= 2
    assert summary["feature_contract"]["scoring_contract"] == "neural_feature_mlp_with_context_conditioned_adjustment_v2"
    assert model_config["scoring_contract"] == "neural_feature_mlp_with_context_conditioned_adjustment_v2"
    expected_model_kind = (
        "neural_feature_mlp_with_context_conditioned_adjustment_v2"
        if TORCH_AVAILABLE
        else "linear_feature_weights_plus_context_conditioned_adjustment_v1"
    )
    assert training_summary["package_summary"]["model_kind"] == expected_model_kind


def test_train_datapack_selection_scorers_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    run_log = _write_run_log(tmp_path / "logs" / "semantic_runs.jsonl")
    output_dir = tmp_path / "runner_results"
    args = Namespace(
        run_logs=[str(run_log)],
        run_log_dirs=[],
        output_dir=str(output_dir),
        run_name="selection_runner",
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
        plan_id="semantic_datapack_selection_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((output_dir / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert manifest["training_kind"] == "semantic_datapack_selection"
    assert manifest["artifact_paths"]["datapack_selection_training_summary"].endswith(
        "datapack_selection_training_summary.json"
    )
    assert manifest["metadata"]["trajectory_audit_kind"] == "selection_receipt_projection"
    assert checkpoint_registry["checkpoints"][0]["model_family"] == "semantic_datapack_selection"
    assert holder["payload"]["benchmark_gate_ready"] is False
