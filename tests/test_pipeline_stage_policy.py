from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.orchestrator.pipeline_manager import (
    PipelineIteration,
    PipelineManager,
    PipelineStage,
    StageResult,
    StageStatus,
    create_default_pipeline_manager,
)
from src.orchestrator.pipeline_stage_policy import (
    PIPELINE_STAGE_LABELS,
    build_pipeline_stage_feature_map,
    heuristic_stage_priority_distribution,
)
from src.orchestrator.pipeline_stage_policy_runtime import (
    resolve_pipeline_stage_policy_helper,
)
from src.orchestrator.pipeline_stage_policy_training import (
    build_pipeline_stage_training_dataset,
    train_pipeline_stage_policy_model,
)


def _ready_execution_summary() -> dict[str, object]:
    return {
        "report_count": 1,
        "ready_count": 1,
        "blocked_count": 0,
        "mean_readiness_score": 1.0,
        "blocking_preconditions": {},
        "satisfied_preconditions": {
            "artifact::runtime_packet_ref": 1,
        },
    }


def _manager_state() -> PipelineManager:
    manager = create_default_pipeline_manager()
    manager.metadata["execution_precondition_summary"] = _ready_execution_summary()
    manager.config["objective_preset"] = "safety"
    for idx, (mpl_delta, error_rate, energy_efficiency) in enumerate(
        [(0.5, 0.18, 0.42), (1.4, 0.08, 0.55)],
        start=1,
    ):
        iteration = PipelineIteration(
            iteration_id=f"iter_{idx}",
            iteration_number=idx,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            is_complete=True,
            summary_metrics={
                "mpl_delta": mpl_delta,
                "error_rate": error_rate,
                "energy_efficiency": energy_efficiency,
            },
        )
        for stage in PipelineStage:
            iteration.stage_results[stage.value] = StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                started_at=iteration.started_at,
                completed_at=iteration.completed_at,
                duration_seconds=1.0 + idx,
                recommendations=["increase data collection"] if stage == PipelineStage.DATA_COLLECTION else [],
            )
        manager.iterations.append(iteration)
    return manager


def _write_package(tmp_path: Path) -> Path:
    manager = _manager_state()
    dataset = build_pipeline_stage_training_dataset([manager.to_dict(), manager.to_dict()])
    checkpoint_path = tmp_path / "pipeline_stage_policy.pt"
    _, training_result = train_pipeline_stage_policy_model(
        dataset,
        epochs=2,
        hidden_dim=16,
        save_path=str(checkpoint_path),
    )
    assert training_result["checkpoint_path"] is not None
    package_path = tmp_path / "pipeline_stage_policy_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "pipeline_stage_policy_test",
                "checkpoint_path": checkpoint_path.name,
                "model_config": {
                    "input_dim": len(dataset.summary["feature_names"]),
                    "hidden_dim": 16,
                },
                "benchmark_gate": {"ready": False},
                "execution_preconditions": {"benchmark_gate_ready": False},
                "promotion_stage": "shadow_candidate",
                "inference_contract": {
                    "helper_blend_policy": {
                        "shadow_candidate_helper_weight": 0.12,
                        "promoted_helper_weight": 0.35,
                        "shadow_candidate_max_stage_delta": 0.18,
                        "promoted_max_stage_delta": 0.4,
                        "shadow_candidate_max_config_delta": 0.18,
                        "promoted_max_config_delta": 0.35,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return package_path


def test_pipeline_stage_feature_map_extracts_progress_and_stage_history() -> None:
    manager = _manager_state()
    activation_plan = manager.build_iteration_activation_plan()
    feature_map = build_pipeline_stage_feature_map(
        config=manager.config,
        iterations=manager.iterations,
        progress=manager.get_progress_metrics(),
        execution_summary=manager.metadata["execution_precondition_summary"],
        shell_activation=activation_plan["shell_activation"],
        last_results=manager._last_iteration_summary(),
        suggested_config={"increase_safety_weight": 1.0},
    )
    distribution = heuristic_stage_priority_distribution(feature_map)

    assert feature_map["execution_mean_readiness"] == 1.0
    assert feature_map["objective_is_safety"] == 1.0
    assert set(distribution) == set(PIPELINE_STAGE_LABELS)
    assert abs(sum(distribution.values()) - 1.0) < 1e-6


def test_pipeline_manager_applies_stage_policy_helper(tmp_path: Path) -> None:
    manager = _manager_state()
    package_path = _write_package(tmp_path)
    manager.config["pipeline_stage_policy_helper_mode"] = "auto"
    manager.config["pipeline_stage_policy_package_path"] = str(package_path)

    preview = manager.preview_next_iteration()

    assert preview["policy_source"] == "heuristic_plus_learned_helper"
    assert preview["promotion_stage"] == "shadow_candidate"
    assert preview["stage_policy_trace"]["helper_trace"]["helper_weight"] == pytest.approx(0.12)
    assert preview["stage_activation_plan"]["policy_source"] == "heuristic_plus_learned_helper"
    assert preview["stage_activation_plan"]["stages"][0]["priority_score"] >= preview["stage_activation_plan"]["stages"][-1]["priority_score"]


def test_required_pipeline_stage_helper_enforces_benchmark_gate(tmp_path: Path) -> None:
    package_path = _write_package(tmp_path)
    with pytest.raises(ValueError):
        resolve_pipeline_stage_policy_helper(
            helper_mode="required",
            package_path=package_path,
        )
