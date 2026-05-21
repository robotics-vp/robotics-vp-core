from __future__ import annotations

import json

from scripts.economic_world_model.build_economic_wm_scaffold import (
    run_build_economic_wm_scaffold,
)
from src.economics.economic_wm_entry import evaluate_economic_wm_entry_preflight
from src.world_model.economic_world_model import (
    build_economic_wm_scaffold_report,
    load_economic_wm_scaffold_report,
)


def _preflight_report():
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
    return evaluate_economic_wm_entry_preflight(
        stage1_sweep_report={
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
    )


def test_economic_wm_scaffold_builds_state_and_allocation_envelope() -> None:
    report = build_economic_wm_scaffold_report(_preflight_report())

    assert report.version == "economic_wm_scaffold_report_v1"
    assert report.ready_for_scaffold is True
    assert report.ready_for_training is False
    assert report.promotion_eligible is False
    assert "gpu_training_not_run" in report.training_blockers
    assert report.economic_state.version == "economic_state_v1"
    assert report.economic_state.regime == "scaffold_ready_training_blocked"
    assert report.economic_state.resource_reservoirs["replay_datapack_inventory"] == 5.0
    assert report.economic_state.flow_fields["benchmark_ready_flow"] == 0.4
    assert report.economic_state.flow_fields["replay_export_flow"] == 1.0
    assert report.economic_state.dissipation_fields["gpu_training_friction"] == 1.0
    assert report.allocation_envelope.authority_class == "scaffold_only"
    assert report.allocation_envelope.reward_math_mutation is False
    assert report.allocation_envelope.promotion_eligible is False
    assert "gpu_training" in report.allocation_envelope.denied_actions
    assert "build_economic_wm_scaffold" in report.allocation_envelope.allowed_actions


def test_build_economic_wm_scaffold_script_from_preflight_report(tmp_path) -> None:
    preflight = _preflight_report()
    preflight_path = tmp_path / "entry_preflight.json"
    preflight_path.write_text(json.dumps(preflight.to_dict()), encoding="utf-8")

    payload = run_build_economic_wm_scaffold(
        output_dir=tmp_path / "scaffold",
        entry_preflight_report_path=preflight_path,
    )

    assert payload["version"] == "economic_wm_scaffold_report_v1"
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_scaffold"] is True
    assert payload["ready_for_training"] is False
    assert payload["allocation_envelope"]["reward_math_mutation"] is False
    assert payload["allocation_envelope"]["promotion_eligible"] is False
    loaded = load_economic_wm_scaffold_report(
        payload["artifact_refs"]["scaffold_report_path"]
    )
    assert loaded.scaffold_id == payload["scaffold_id"]
    assert loaded.ready_for_scaffold is True
    assert (tmp_path / "scaffold" / "economic_wm_scaffold_report_v1.md").exists()
