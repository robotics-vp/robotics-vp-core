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
    (receipt_dir / "episode_backend_runtime_bridge_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "bridge_1",
                "bridge_id": "bridge_state_1",
                "backend": "isaac",
                "bridge_status": "runtime_targets_missing",
                "execution_authority": "shadow_runtime",
                "transport_profile": "isaac_shadow_bridge",
                "planner_rate_hz": 10.0,
                "control_rate_hz": 250.0,
                "observation_rate_hz": 60.0,
                "action_decimation": 4,
                "latency_budget_ms": 8.0,
                "bridge_readiness_score": 0.55,
                "action_contracts": ["whole_body_joint_command_v1"],
                "observation_contracts": ["imu_state_v1"],
                "telemetry_contracts": ["watchdog_state_v1"],
                "safety_channels": ["joint_limit_guard_v1", "watchdog_v1"],
                "metadata": {
                    "runtime_target_contract": {
                        "missing_required_target_ids": ["unitree_sdk2_root"]
                    }
                },
                "version": "backend_runtime_bridge_receipt_v1",
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
    (receipt_dir / "episode_backend_runtime_execution_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "runtime_1",
                "backend": "isaac",
                "execution_mode": "workcell_isaaclab_evaluate_policy",
                "execution_status": "runtime_execution_completed",
                "policy_id": "policy_isaac_1",
                "artifact_refs": ["/tmp/runtime_episode_1/trajectory.npz"],
                "version": "backend_runtime_execution_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_backend_runtime_adapter_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "adapter_1",
                "backend": "isaac",
                "adapter_family": "isaac_unitree",
                "adapter_entrypoint": "isaaclab_unitree_sim",
                "consumer_mode": "external_sim_launch",
                "adapter_status": "external_launch_completed",
                "execution_path": "external_launch",
                "metadata": {
                    "realization": {
                        "realization_path": "external_launch_delegate",
                        "realization_status": "external_launch_delegate_ready",
                    }
                },
                "executed": True,
                "version": "backend_runtime_adapter_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_backend_runtime_launch_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "launch_1",
                "backend": "isaac",
                "launch_profile": "unitree_sim_isaaclab",
                "launch_status": "launch_completed",
                "executed": True,
                "command": "python sim_main.py --task peg_in_hole",
                "cwd": "/tmp/unitree_sim_isaaclab",
                "version": "backend_runtime_launch_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_backend_runtime_outcome_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "runtime_outcome_1",
                "backend": "isaac",
                "outcome_profile": "unitree_sim_isaaclab",
                "outcome_status": "runtime_outputs_harvested",
                "executed": True,
                "harvested_output_count": 2,
                "artifact_refs": [
                    "/tmp/unitree_sim_isaaclab/logs/run_1/metrics.json",
                    "/tmp/unitree_sim_isaaclab/logs/run_1/policy.onnx",
                ],
                "metadata": {
                    "structured_outputs": {
                        "ready_surfaces": [
                            "metrics_surface_ready",
                            "policy_surface_ready",
                        ],
                        "metric_keys": ["metrics.score"],
                        "primary_policy_ref": "/tmp/unitree_sim_isaaclab/logs/run_1/policy.onnx",
                    }
                },
                "version": "backend_runtime_outcome_receipt_v1",
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
    assert bundles[0]["backend_runtime_bridge_receipt"]["receipt_id"] == "bridge_1"
    assert bundles[0]["backend_runtime_execution_receipt"]["receipt_id"] == "runtime_1"
    assert bundles[0]["backend_runtime_adapter_receipt"]["receipt_id"] == "adapter_1"
    assert bundles[0]["backend_runtime_launch_receipt"]["receipt_id"] == "launch_1"
    assert bundles[0]["backend_runtime_outcome_receipt"]["receipt_id"] == "runtime_outcome_1"
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
    assert backend_rows[0]["metadata"]["backend_runtime_bridge_receipt_id"] == "bridge_1"
    assert backend_rows[0]["metadata"]["backend_runtime_bridge_status"] == "runtime_targets_missing"
    assert backend_rows[0]["metadata"]["bridge_execution_authority"] == "shadow_runtime"
    assert backend_rows[0]["metadata"]["bridge_transport_profile"] == "isaac_shadow_bridge"
    assert backend_rows[0]["metadata"]["bridge_missing_runtime_targets"] == [
        "unitree_sdk2_root"
    ]
    assert backend_rows[0]["metadata"]["backend_shadow_execution_receipt_id"] == "shadow_1"
    assert (
        backend_rows[0]["metadata"]["backend_shadow_execution_status"]
        == "shadow_executed_with_asset_gaps"
    )
    assert backend_rows[0]["metadata"]["backend_runtime_execution_receipt_id"] == "runtime_1"
    assert (
        backend_rows[0]["metadata"]["backend_runtime_execution_status"]
        == "runtime_execution_completed"
    )
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_receipt_id"] == "adapter_1"
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_status"] == "external_launch_completed"
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_execution_path"] == "external_launch"
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_realization_path"] == "external_launch_delegate"
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_realization_status"] == "external_launch_delegate_ready"
    assert backend_rows[0]["metadata"]["backend_runtime_launch_receipt_id"] == "launch_1"
    assert backend_rows[0]["metadata"]["backend_runtime_launch_status"] == "launch_completed"
    assert backend_rows[0]["metadata"]["backend_runtime_launch_executed"] is True
    assert (
        backend_rows[0]["metadata"]["backend_runtime_outcome_receipt_id"]
        == "runtime_outcome_1"
    )
    assert backend_rows[0]["metadata"]["backend_runtime_outcome_status"] == "runtime_outputs_harvested"
    assert backend_rows[0]["metadata"]["backend_runtime_output_count"] == 2
    assert backend_rows[0]["metadata"]["backend_runtime_ready_surfaces"] == [
        "metrics_surface_ready",
        "policy_surface_ready",
    ]
    assert backend_rows[0]["metadata"]["backend_runtime_primary_policy_ref"].endswith(
        "policy.onnx"
    )
    assert backend_rows[0]["metadata"]["backend_runtime_metric_keys"] == ["metrics.score"]

    branch_rows = build_branch_planner_rows_from_receipts(bundles)
    assert branch_rows[0]["target_render_materialization_status"] == "scene_materialized"
    assert branch_rows[0]["target_render_materialization_mode"] == "scene_config"
    assert branch_rows[0]["metadata"]["robot_asset_contract_receipt_id"] == "asset_1"
    assert branch_rows[0]["metadata"]["robot_asset_readiness_score"] == 0.25
    assert branch_rows[0]["metadata"]["backend_runtime_bridge_receipt_id"] == "bridge_1"
    assert branch_rows[0]["metadata"]["backend_runtime_bridge_status"] == "runtime_targets_missing"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_receipt_id"] == "adapter_1"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_status"] == "external_launch_completed"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_execution_path"] == "external_launch"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_realization_path"] == "external_launch_delegate"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_realization_status"] == "external_launch_delegate_ready"
    assert branch_rows[0]["metadata"]["backend_runtime_launch_receipt_id"] == "launch_1"
    assert branch_rows[0]["metadata"]["backend_runtime_launch_status"] == "launch_completed"
    assert (
        branch_rows[0]["metadata"]["backend_runtime_outcome_receipt_id"]
        == "runtime_outcome_1"
    )
    assert branch_rows[0]["metadata"]["backend_runtime_outcome_status"] == "runtime_outputs_harvested"
    assert branch_rows[0]["metadata"]["backend_runtime_output_count"] == 2
    assert branch_rows[0]["metadata"]["backend_runtime_ready_surfaces"] == [
        "metrics_surface_ready",
        "policy_surface_ready",
    ]
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


def test_backend_selector_rows_prefer_external_runtime_outcomes_when_no_concrete_runtime() -> None:
    bundles = [
        {
            "bundle_id": "bundle_1",
            "world_state": {
                "state_id": "sim_state_2",
                "simulation_agenda": {"jobs": [{"job_id": "job_1"}]},
                "physics_context": {
                    "backend": "isaac",
                    "fidelity_tier": "high_fidelity",
                    "domain_randomization_regime": "benchmark_focus",
                    "metadata": {},
                },
            },
            "backend_runtime_bridge_receipt": {
                "receipt_id": "bridge_2",
                "bridge_status": "runtime_bridge_ready",
                "execution_authority": "shadow_runtime",
                "transport_profile": "isaaclab_unitree_dds_bridge",
                "bridge_readiness_score": 0.8,
                "metadata": {
                    "runtime_target_contract": {"missing_required_target_ids": []}
                },
            },
            "backend_runtime_launch_receipt": {
                "receipt_id": "launch_2",
                "launch_status": "launch_completed",
                "executed": True,
            },
            "backend_runtime_outcome_receipt": {
                "receipt_id": "outcome_runtime_2",
                "outcome_status": "runtime_outputs_harvested",
                "harvested_output_count": 4,
                "metadata": {
                    "structured_outputs": {
                        "ready_surfaces": ["dataset_surface_ready"],
                        "metric_keys": [],
                        "primary_policy_ref": "",
                    }
                },
            },
            "backend_shadow_execution_receipt": {
                "receipt_id": "shadow_2",
                "execution_status": "shadow_work_order_materialized",
            },
        }
    ]

    rows = build_backend_selector_rows_from_receipts(bundles)

    assert rows[0]["target_source"] == "external_runtime_outcome_receipt"
    assert rows[0]["metadata"]["backend_runtime_outcome_receipt_id"] == "outcome_runtime_2"
    assert rows[0]["metadata"]["backend_runtime_output_count"] == 4
    assert rows[0]["metadata"]["backend_runtime_ready_surfaces"] == [
        "dataset_surface_ready"
    ]
