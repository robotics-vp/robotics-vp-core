from __future__ import annotations

from scripts.economic_world_model.economic_wm_entry_preflight import (
    run_economic_wm_entry_preflight,
)
from src.economics.economic_wm_entry import (
    evaluate_economic_wm_entry_preflight,
    load_economic_wm_entry_preflight_report,
)


def _ok_sweep_report() -> dict:
    scenarios = []
    for idx, ready in enumerate([True, False, False, True, False]):
        scenarios.append(
            {
                "scenario_id": f"scenario_{idx}",
                "passed": True,
                "observed": {
                    "benchmark_ready": ready,
                    "rlds_benchmark_ready": ready,
                    "lerobot_benchmark_ready": ready,
                },
            }
        )
    return {
        "version": "stage1_bridge_readiness_sweep_v1",
        "status": "ok",
        "scenario_count": 5,
        "admission_count": 5,
        "rlds_episode_count": 5,
        "lerobot_row_count": 5,
        "benchmark_ready_count": 2,
        "shadow_only_count": 3,
        "promotion_eligible": False,
        "failures": [],
        "scenario_reports": scenarios,
    }


def test_economic_wm_entry_preflight_separates_scaffold_from_training() -> None:
    report = evaluate_economic_wm_entry_preflight(
        stage1_sweep_report=_ok_sweep_report()
    )

    assert report.ready_for_scaffold is True
    assert report.ready_for_training is False
    assert report.readiness_class == "scaffold_ready_training_blocked"
    assert "gpu_training_not_run" in report.training_blockers
    assert report.scaffold_blockers == []


def test_economic_wm_entry_preflight_blocks_failed_sweep() -> None:
    payload = _ok_sweep_report()
    payload["status"] = "failed"
    payload["failures"] = ["boom"]
    report = evaluate_economic_wm_entry_preflight(stage1_sweep_report=payload)

    assert report.ready_for_scaffold is False
    assert report.readiness_class == "blocked"
    assert "stage1_bridge_sweep_failed" in report.scaffold_blockers
    assert "stage1_bridge_sweep_has_failures" in report.scaffold_blockers


def test_economic_wm_entry_preflight_script_round_trip(tmp_path) -> None:
    payload = run_economic_wm_entry_preflight(output_dir=tmp_path)

    assert payload["ready_for_scaffold"] is True
    assert payload["ready_for_training"] is False
    assert payload["readiness_class"] == "scaffold_ready_training_blocked"
    assert payload["counts"]["scenario_count"] == 5
    loaded = load_economic_wm_entry_preflight_report(payload["report_path"])
    assert loaded.report_id == payload["report_id"]
