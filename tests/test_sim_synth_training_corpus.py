import json
from pathlib import Path

from src.world_model.sim_synth_physics.training_corpus import (
    build_backend_selector_rows_from_receipts,
    build_branch_planner_rows_from_receipts,
    harvest_sim_synth_receipt_bundles,
)


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
    (receipt_dir / "episode_physics_adaptation_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "adapt_1",
                "policy_id": "policy_1",
                "backend": "pybullet",
                "target_hardware_class": "unitree_g1_r1_class",
                "domain_randomization_profile": "humanoid_shadow_randomization",
                "system_identification_profile": "humanoid_shadow_system_id",
                "readiness_score": 0.4,
                "version": "physics_adaptation_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_backend_execution_binding_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "binding_1",
                "binding_id": "binding_state_1",
                "backend": "isaac",
                "binding_status": "assets_missing",
                "executor_entrypoint": "src.envs.physics.backend_factory:make_backend",
                "asset_profile": "unitree_humanoid_shadow_assets",
                "metadata": {
                    "required_assets": ["unitree_robot_description"],
                    "missing_assets": ["unitree_robot_description"],
                },
                "version": "backend_execution_binding_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_robot_asset_contract_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "asset_1",
                "contract_id": "asset_contract_1",
                "asset_profile": "unitree_humanoid_shadow_assets",
                "target_hardware_class": "unitree_g1_r1_class",
                "readiness_score": 0.25,
                "required_assets": ["unitree_robot_description", "sensor_extrinsics"],
                "available_assets": [],
                "missing_assets": ["unitree_robot_description", "sensor_extrinsics"],
                "calibration_contracts": ["whole_body_joint_map"],
                "observation_contracts": ["imu_state_v1"],
                "action_contracts": ["whole_body_joint_command_v1"],
                "version": "robot_asset_contract_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_backend_shadow_execution_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "shadow_1",
                "backend": "isaac",
                "execution_mode": "shadow_contract",
                "execution_status": "shadow_executed_with_asset_gaps",
                "episode_ids": ["shadow_episode_1"],
                "artifact_refs": ["/tmp/shadow_episode_1/rgb.json"],
                "version": "backend_shadow_execution_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_render_provider_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "provider_1",
                "branch_plan_id": "plan_1",
                "provider_id": "render_provider_1",
                "provider_kind": "lsd_scene_graph",
                "provider_status": "ready",
                "render_mode": "lsd_vector_scene",
                "counterfactual_mode": "none",
                "materialization_status": "scene_materialized",
                "materialization_mode": "scene_config",
                "materialization_entrypoint": "src.motor_backend.factory:make_motor_backend",
                "artifact_refs": ["/tmp/render_provider_1/lsd_vector_scene_config.json"],
                "version": "render_provider_receipt_v1",
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
    assert bundles[0]["physics_adaptation_receipt"]["receipt_id"] == "adapt_1"
    assert bundles[0]["backend_execution_binding_receipt"]["receipt_id"] == "binding_1"
    assert bundles[0]["robot_asset_contract_receipt"]["receipt_id"] == "asset_1"
    assert bundles[0]["backend_shadow_execution_receipt"]["receipt_id"] == "shadow_1"
    assert bundles[0]["physics_calibration_receipt"]["receipt_id"] == "cal_1"
    assert bundles[0]["render_provider_receipts"][0]["receipt_id"] == "provider_1"
    assert bundles[0]["simulation_outcome_receipts"][0]["receipt_id"] == "outcome_1"

    backend_rows = build_backend_selector_rows_from_receipts(bundles)
    assert backend_rows[0]["target_hardware_class"] == "unitree_g1_r1_class"
    assert backend_rows[0]["target_system_identification_profile"] == "humanoid_shadow_system_id"
    assert backend_rows[0]["target_source"] == "runtime_receipt"
    assert backend_rows[0]["metadata"]["robot_asset_contract_receipt_id"] == "asset_1"
    assert backend_rows[0]["metadata"]["robot_asset_readiness_score"] == 0.25
    assert backend_rows[0]["metadata"]["robot_asset_missing_assets"] == [
        "unitree_robot_description",
        "sensor_extrinsics",
    ]
    assert backend_rows[0]["metadata"]["backend_shadow_execution_receipt_id"] == "shadow_1"
    assert (
        backend_rows[0]["metadata"]["backend_shadow_execution_status"]
        == "shadow_executed_with_asset_gaps"
    )

    branch_rows = build_branch_planner_rows_from_receipts(bundles)
    assert branch_rows[0]["target_render_materialization_status"] == "scene_materialized"
    assert branch_rows[0]["target_render_materialization_mode"] == "scene_config"
    assert branch_rows[0]["metadata"]["robot_asset_contract_receipt_id"] == "asset_1"
    assert branch_rows[0]["metadata"]["robot_asset_readiness_score"] == 0.25
    assert (
        branch_rows[0]["metadata"]["render_artifact_refs"]
        == ["/tmp/render_provider_1/lsd_vector_scene_config.json"]
    )


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
