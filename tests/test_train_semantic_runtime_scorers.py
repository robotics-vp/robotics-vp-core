from __future__ import annotations

import json
from pathlib import Path

from scripts.train_semantic_runtime_scorers import _run_training, parse_args
from src.replay.dataset import ReplayDatasetBuilder
from src.shadow_runtime.control_plane import run_shadow_control_plane
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.utils.config_digest import sha256_json


def _replay_dataset_path(tmp_path: Path) -> Path:
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=13,
        episodes=3,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)
    return dataset_dir


def test_train_semantic_runtime_scorers_emits_runtime_package(tmp_path: Path) -> None:
    dataset_dir = _replay_dataset_path(tmp_path)
    args = parse_args(
        [
            "--replay-dataset",
            str(dataset_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--trainer",
            "linear",
            "--skip-regal-runner",
        ]
    )

    result = _run_training(args, runner=None)
    runtime_package = json.loads(Path(result["runtime_package"]).read_text(encoding="utf-8"))

    assert runtime_package["promotion_stage"] == "shadow_candidate"
    assert runtime_package["inference_contract"]["target_contract"] == "semantic_runtime_scorer_v1"
    assert Path(result["legacy_scorer_package"]).exists()


def test_regality_wrapper_registers_semantic_runtime_scorer_artifacts(tmp_path: Path) -> None:
    dataset_dir = _replay_dataset_path(tmp_path)
    output_dir = tmp_path / "runner"

    def _wrapped(runner) -> None:
        args = parse_args(
            [
                "--replay-dataset",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--trainer",
                "linear",
            ]
        )
        _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=7,
            num_episodes=3,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha=sha256_json({"plan": "semantic_runtime_scorer_test"}),
        plan_id="semantic_runtime_scorer_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_kind"] == "semantic_runtime_scorers"
    assert manifest["artifact_paths"]["semantic_runtime_scorer_runtime_package"].endswith(
        "semantic_runtime_scorer_runtime_package.json"
    )
