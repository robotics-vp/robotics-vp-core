from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.train_pipeline_stage_policy import _run_training, parse_args
from src.orchestrator.pipeline_manager import (
    PipelineIteration,
    PipelineStage,
    StageResult,
    StageStatus,
    create_default_pipeline_manager,
)
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.utils.config_digest import sha256_json


def _state_path(tmp_path: Path) -> Path:
    manager = create_default_pipeline_manager()
    manager.metadata["execution_precondition_summary"] = {
        "report_count": 1,
        "ready_count": 1,
        "blocked_count": 0,
        "mean_readiness_score": 1.0,
        "blocking_preconditions": {},
        "satisfied_preconditions": {"artifact::runtime_packet_ref": 1},
    }
    iteration = PipelineIteration(
        iteration_id="iter_a",
        iteration_number=1,
        started_at=datetime.utcnow().isoformat(),
        completed_at=datetime.utcnow().isoformat(),
        is_complete=True,
        summary_metrics={
            "mpl_delta": 0.4,
            "error_rate": 0.19,
            "energy_efficiency": 0.44,
        },
    )
    for stage in PipelineStage:
        iteration.stage_results[stage.value] = StageResult(
            stage=stage,
            status=StageStatus.COMPLETED,
            started_at=iteration.started_at,
            completed_at=iteration.completed_at,
            duration_seconds=2.0,
        )
    manager.iterations.append(iteration)
    path = tmp_path / "pipeline_state.json"
    path.write_text(json.dumps(manager.to_dict(), indent=2), encoding="utf-8")
    return path


def test_run_training_emits_runtime_package(tmp_path: Path) -> None:
    state_path = _state_path(tmp_path)
    args = parse_args(
        [
            "--state-json",
            str(state_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-regal-runner",
            "--epochs",
            "2",
        ]
    )

    result = _run_training(args, runner=None)

    assert Path(result["training_summary"]).exists()
    package = json.loads(Path(result["runtime_package"]).read_text(encoding="utf-8"))
    assert package["promotion_stage"] == "shadow_candidate"
    assert package["inference_contract"]["target_contract"] == "pipeline_manager_stage_policy_v1"


def test_regality_wrapper_registers_pipeline_stage_artifacts(tmp_path: Path) -> None:
    state_path = _state_path(tmp_path)
    output_dir = tmp_path / "runner"

    def _wrapped(runner):
        args = parse_args(
            [
                "--state-json",
                str(state_path),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "2",
            ]
        )
        _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=7,
            num_episodes=1,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha=sha256_json({"plan": "pipeline_stage_policy_test"}),
        plan_id="pipeline_stage_policy_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_kind"] == "pipeline_stage_policy"
    assert manifest["artifact_paths"]["pipeline_stage_policy_runtime_package"].endswith(
        "pipeline_stage_policy_package.json"
    )
