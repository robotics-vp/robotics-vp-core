import json
from pathlib import Path

from src.world_model.sim_synth_physics.training_corpus import harvest_sim_synth_receipt_bundles


def test_harvest_sim_synth_receipt_bundles_builds_bundle_from_live_dir(tmp_path: Path) -> None:
    receipt_dir = tmp_path / "live_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "episode_sim_synth_world_state_v1.json").write_text(
        json.dumps(
            {
                "state_id": "sim_state_1",
                "simulation_agenda": {"jobs": [{"job_id": "job_1"}]},
                "physics_context": {"backend": "pybullet", "metadata": {}},
                "synthetic_branch_plans": [{"plan_id": "plan_1", "source_job_id": "job_1"}],
                "version": "sim_synth_physics_world_state_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_physics_calibration_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "cal_1",
                "backend": "isaac",
                "fidelity_tier": "high_fidelity",
                "calibration_profile": "default",
                "quality_score": 0.8,
                "version": "physics_calibration_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_simulation_outcome_receipt_v1.jsonl").write_text(
        json.dumps(
            {
                "receipt_id": "outcome_1",
                "job_id": "job_1",
                "branch_plan_id": "plan_1",
                "status": "completed",
                "version": "simulation_outcome_receipt_v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bundles = harvest_sim_synth_receipt_bundles([receipt_dir])

    assert len(bundles) == 1
    assert bundles[0]["world_state"]["state_id"] == "sim_state_1"
    assert bundles[0]["physics_calibration_receipt"]["receipt_id"] == "cal_1"
    assert bundles[0]["simulation_outcome_receipts"][0]["receipt_id"] == "outcome_1"


def test_harvest_sim_synth_receipt_bundles_ignores_incomplete_dirs(tmp_path: Path) -> None:
    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    (incomplete_dir / "episode_simulation_outcome_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "outcome_only",
                "job_id": "job_only",
                "branch_plan_id": "plan_only",
                "status": "completed",
                "version": "simulation_outcome_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    bundles = harvest_sim_synth_receipt_bundles([incomplete_dir])

    assert bundles == []
