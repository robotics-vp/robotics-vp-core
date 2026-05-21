from __future__ import annotations

from src.evidence.benchmark_gating import (
    build_benchmark_gate_report,
    collect_benchmark_gating_signals,
)


def test_collect_benchmark_gating_signals_marks_real_grounding() -> None:
    signals = collect_benchmark_gating_signals(
        {
            "scene_tracks_backend": "real",
            "semantic_memory_grounded": True,
            "openvla_backend_selected": "real",
            "vision_backbone_selected": "real",
        }
    )

    assert signals["scene_tracks_backend_real"] is True
    assert signals["teacher_runtime_real"] is True
    assert signals["vision_backbone_real"] is True
    assert signals["semantic_grounding_non_heuristic"] is True
    assert signals["benchmark_eligible"] is True


def test_collect_benchmark_gating_signals_respects_calibration_requirement() -> None:
    signals = collect_benchmark_gating_signals(
        {
            "scene_tracks_backend": "real",
            "semantic_memory_grounded": True,
            "openvla_backend_selected": "real",
            "vision_backbone_selected": "real",
            "require_camera_calibration": True,
        }
    )

    assert signals["reconstruction_calibrated"] is False
    assert signals["camera_calibration_required"] is True
    assert signals["benchmark_eligible"] is False


def test_benchmark_gate_blocks_missing_camera_calibration_when_required() -> None:
    report = build_benchmark_gate_report(
        subject_id="loop",
        subject_kind="loop_run",
        metadata={
            "scene_tracks_backend": "real",
            "semantic_memory_grounded": True,
            "openvla_backend_selected": "real",
            "vision_backbone_selected": "real",
            "reconstruction_calibration_class": "camera_missing",
        },
        require_real_scene_tracks=True,
        require_teacher_runtime=True,
        require_vision_backbone=True,
        require_camera_calibration=True,
    )

    assert report.ready is False
    assert "signal_bool::reconstruction_calibrated" in report.blocking_preconditions
    assert "blocked::camera_calibration_missing" in report.blocking_preconditions


def test_benchmark_gate_passes_when_camera_calibrated_required() -> None:
    report = build_benchmark_gate_report(
        subject_id="loop",
        subject_kind="loop_run",
        metadata={
            "scene_tracks_backend": "real",
            "semantic_memory_grounded": True,
            "openvla_backend_selected": "real",
            "vision_backbone_selected": "real",
            "reconstruction_calibration_class": "camera_calibrated",
        },
        require_real_scene_tracks=True,
        require_teacher_runtime=True,
        require_vision_backbone=True,
        require_camera_calibration=True,
    )

    assert report.ready is True


def test_benchmark_gate_blocks_passthrough_and_stub_paths() -> None:
    report = build_benchmark_gate_report(
        subject_id="loop",
        subject_kind="loop_run",
        metadata={
            "scene_tracks_backend": "passthrough",
            "semantic_memory_grounded": True,
            "openvla_backend_selected": "stub",
            "vision_backbone_selected": "stub",
        },
        require_real_scene_tracks=True,
        require_teacher_runtime=True,
        require_vision_backbone=True,
    )

    assert report.ready is False
    assert "blocked::scene_tracks_passthrough_selected" in report.blocking_preconditions
    assert "blocked::teacher_runtime_stub_selected" in report.blocking_preconditions
    assert "blocked::vision_backbone_stub_selected" in report.blocking_preconditions


def test_benchmark_gate_passes_real_paths() -> None:
    report = build_benchmark_gate_report(
        subject_id="loop",
        subject_kind="loop_run",
        metadata={
            "scene_tracks_backend": "real",
            "semantic_memory_grounded": True,
            "openvla_backend_selected": "real",
            "vision_backbone_selected": "real",
        },
        require_real_scene_tracks=True,
        require_teacher_runtime=True,
        require_vision_backbone=True,
    )

    assert report.ready is True
