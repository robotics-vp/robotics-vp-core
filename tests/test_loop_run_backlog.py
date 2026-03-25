from __future__ import annotations

from pathlib import Path

from src.orchestrator.loop_run_backlog import LoopRunBacklogItem, evaluate_loop_run_item


def test_loop_run_item_ready_when_preconditions_are_satisfied(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    monkeypatch.setenv("DROID_DATASET_ROOT", str(data_root))
    item = LoopRunBacklogItem(
        loop_run_id="bootstrap",
        title="bootstrap",
        command="python3 scripts/bootstrap_semantic_workcell_loop.py",
        auto_trigger=True,
        required_capabilities={"mujoco_available": True},
        required_python_modules=[],
        required_commands=["python3"],
        required_paths=[],
        required_env_path_vars=["DROID_DATASET_ROOT"],
    )

    assessment = evaluate_loop_run_item(
        item,
        host_capabilities={"mujoco_available": True},
    )

    assert assessment.ready is True


def test_loop_run_item_blocks_when_benchmark_gate_is_not_ready() -> None:
    item = LoopRunBacklogItem(
        loop_run_id="benchmark",
        title="benchmark",
        command="python3 scripts/bootstrap_semantic_workcell_loop.py --backend-policy real",
        auto_trigger=False,
        required_capabilities={"mujoco_available": True},
        benchmark_gate={
            "require_real_scene_tracks": True,
            "require_teacher_runtime": True,
        },
    )

    assessment = evaluate_loop_run_item(
        item,
        host_capabilities={
            "mujoco_available": True,
            "gpu_available": False,
            "transformers_available": False,
            "openvla_model_ref_present": False,
            "openvla_model_path_ready": False,
            "opencv_available": False,
            "sam3d_objects_repo_available": False,
            "sam3d_body_repo_available": False,
            "sam3d_objects_checkpoint_available": False,
            "sam3d_body_checkpoint_available": False,
        },
    )

    assert assessment.ready is False
    assert "signal_bool::scene_tracks_backend_real" in assessment.pending_requirements
