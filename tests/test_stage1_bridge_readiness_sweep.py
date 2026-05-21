from __future__ import annotations

from scripts.economic_world_model.sweep_stage1_bridge_readiness import (
    run_stage1_bridge_readiness_sweep,
)


def test_stage1_bridge_readiness_sweep_covers_manifest_variants(tmp_path) -> None:
    summary = run_stage1_bridge_readiness_sweep(output_dir=tmp_path, quiet=True)

    assert summary["version"] == "stage1_bridge_readiness_sweep_v1"
    assert summary["status"] == "ok"
    assert summary["scenario_count"] == 5
    assert summary["admission_count"] == 5
    assert summary["rlds_episode_count"] == 5
    assert summary["lerobot_row_count"] == 5
    assert summary["benchmark_ready_count"] == 2
    assert summary["shadow_only_count"] == 3
    assert summary["promotion_eligible"] is False

    reports = {row["scenario_id"]: row for row in summary["scenario_reports"]}
    assert reports["calibrated_inline_real"]["observed"]["benchmark_ready"] is True
    assert reports["top_level_calibration_real"]["observed"]["benchmark_ready"] is True
    assert (
        reports["missing_calibration_real"]["observed"]["calibration_class"]
        == "camera_missing"
    )
    assert (
        "blocked::camera_calibration_missing"
        in reports["missing_calibration_real"]["observed"][
            "benchmark_blocking_preconditions"
        ]
    )
    assert (
        reports["unknown_artifact_calibrated"]["observed"]["grounding_class"]
        == "scene_tracks_ref_unqualified"
    )
    assert (
        "blocked::scene_tracks_passthrough_selected"
        in reports["passthrough_calibrated"]["observed"][
            "benchmark_blocking_preconditions"
        ]
    )
