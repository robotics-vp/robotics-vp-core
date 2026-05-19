import json
from pathlib import Path

from src.world_model.sim_synth_physics.training_corpus import (
    build_backend_selector_rows_from_receipts,
    build_branch_planner_rows_from_receipts,
    harvest_sim_synth_receipt_bundles,
    select_phase1x_positive_training_rows,
    validate_runtime_receipt_manifest,
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
                "physics_execution_contract": {
                    "contract_id": "contract_1",
                    "requested_backend": "pybullet",
                    "resolved_backend": "isaac",
                    "fidelity_tier": "high_fidelity",
                    "domain_randomization_regime": "benchmark_focus",
                    "calibration_profile": "shadow_replay",
                    "backend_selection_policy": "heuristic_plus_learned_backend_selector",
                    "adapter_name": "workcell_isaaclab",
                    "route_status": "fallback",
                    "version": "physics_execution_contract_v1",
                },
                "synthetic_branch_plans": [
                    {
                        "plan_id": "plan_1",
                        "source_job_id": "job_1",
                        "metadata": {
                            "branch_helper_status": {
                                "status": "loaded",
                                "promotion_stage": "shadow_candidate",
                                "benchmark_gate_ready": False,
                            },
                            "branch_helper_trace": {
                                "generation_mode": "neural_branch_candidate",
                                "expected_yield_score": 0.91,
                            },
                            "branch_helper_resolution": "heuristic_due_to_shadow_candidate",
                            "branch_helper_resolution_reason": "benchmark_gate_not_ready",
                            "branch_helper_payload_applied": False,
                            "scene_hierarchy_ref": {
                                "hierarchy_id": "scene_h_1",
                                "scene_id": "workcell_scene_1",
                                "scene_kind": "workcell",
                                "materialization_status": "asset_contract_ready",
                            },
                            "scene_materialization_status": "asset_contract_ready",
                        },
                    }
                ],
                "metadata": {
                    "compiled_receipt_inventory": {
                        "inventory_id": "inventory_1",
                        "runtime_depth_projection": {
                            "binding_status": "assets_missing",
                            "bridge_status": "runtime_targets_missing",
                            "upstream_runtime_pack_status": "pack_partial",
                        },
                    }
                },
                "version": "sim_synth_physics_world_state_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_runtime_receipt_manifest_v1.json").write_text(
        json.dumps(
            {
                "manifest_id": "runtime_manifest_1",
                "world_state_id": "sim_state_1",
                "physics_execution_contract_id": "contract_1",
                "compiled_receipt_inventory_id": "inventory_1",
                "manifest_status": "complete",
                "missing_required_families": [],
                "optional_not_emitted_families": [
                    "backend_runtime_launch_receipt_v1"
                ],
                "receipt_family_counts": {
                    "branch_validity_receipt_v1": 1,
                    "replay_validity_receipt_v1": 1,
                    "sensor_alignment_receipt_v1": 1,
                },
                "emitted_receipt_count": 3,
                "artifact_entries": [],
                "version": "sim_synth_runtime_receipt_manifest_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_physics_execution_contract_v1.json").write_text(
        json.dumps(
            {
                "contract_id": "contract_1",
                "requested_backend": "pybullet",
                "resolved_backend": "isaac",
                "fidelity_tier": "high_fidelity",
                "domain_randomization_regime": "benchmark_focus",
                "calibration_profile": "shadow_replay",
                "backend_selection_policy": "heuristic_plus_learned_backend_selector",
                "adapter_name": "workcell_isaaclab",
                "route_status": "fallback",
                "version": "physics_execution_contract_v1",
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
    (receipt_dir / "episode_task_measurement_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "task_measure_1",
                "surface_id": "task_surface_1",
                "task_definition_contract_id": "task_contract_1",
                "task_family": "drawer_vase",
                "benchmark_gate_ready": False,
                "measurement_values": {
                    "coverage_gap_score": 0.7,
                    "promotion_readiness": 0.2,
                },
                "measurement_status": {
                    "coverage_gap_score": "available",
                    "promotion_readiness": "shadow_only",
                },
                "version": "task_measurement_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_sim_real_gap_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "sim_real_gap_1",
                "source_backend": "isaac",
                "target_hardware_class": "unitree_g1_r1_class",
                "comparison_scope": "planning_window",
                "gap_score": 0.42,
                "realism_confidence": 0.58,
                "status": "estimated",
                "branch_plan_ids": ["plan_1"],
                "version": "sim_real_gap_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_backend_mismatch_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "backend_mismatch_1",
                "reference_backend": "pybullet",
                "candidate_backend": "isaac",
                "mismatch_score": 0.25,
                "calibration_staleness_score": 0.2,
                "status": "mismatch_estimated",
                "version": "backend_mismatch_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_surrogate_physics_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "surrogate_physics_1",
                "provider_id": "surrogate_provider_1",
                "forecast_scope": "branch_preview",
                "forecast_status": "contract_reserved",
                "surrogate_confidence": 0.0,
                "branch_plan_ids": ["plan_1"],
                "version": "surrogate_physics_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_surrogate_calibration_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "surrogate_calibration_1",
                "provider_id": "surrogate_provider_1",
                "reference_backend": "isaac",
                "calibration_status": "not_calibrated",
                "calibration_score": 0.0,
                "staleness_score": 1.0,
                "version": "surrogate_calibration_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_branch_validity_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "branch_validity_1",
                "branch_plan_id": "plan_1",
                "job_id": "job_1",
                "validity_score": 0.66,
                "admission_score": 0.61,
                "admissible": True,
                "evidence_status": "local_estimate",
                "reject_reasons": [],
                "metadata": {
                    "scene_materialization_status": "asset_contract_ready",
                },
                "version": "branch_validity_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_sensor_alignment_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "sensor_alignment_1",
                "scene_hierarchy_id": "scene_h_1",
                "sensor_profile": "rgbd_front",
                "alignment_score": 0.95,
                "status": "geometry_contract_validated",
                "checks": {
                    "intrinsics": "valid",
                    "extrinsics": "valid",
                    "round_trip": "passed",
                },
                "metrics": {"round_trip_max_pixel_error": 0.0},
                "metadata": {
                    "scene_materialization_status": "asset_contract_ready",
                },
                "version": "sensor_alignment_receipt_v1",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (receipt_dir / "episode_replay_validity_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "replay_validity_1",
                "branch_plan_id": "plan_1",
                "outcome_receipt_id": "outcome_1",
                "validity_score": 0.72,
                "task_consistency_score": 0.68,
                "transfer_consistency_score": 0.58,
                "status": "training_validity_estimated",
                "reject_reasons": [],
                "metadata": {
                    "evidence_status": "local_estimate",
                },
                "version": "replay_validity_receipt_v1",
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
    (receipt_dir / "episode_gen2sim_admission_receipt_v1.json").write_text(
        json.dumps(
            {
                "receipt_id": "gen2sim_1",
                "admission_id": "gen2sim_state_1",
                "benchmark_gate_ready": False,
                "admissible_branch_ids": ["plan_1"],
                "blocked_branch_ids": [],
                "selection_policy": "receipt_gated_with_inferential_contracts",
                "rationale": "1 branch admissible",
                "metadata": {
                    "admissible_branch_count": 1,
                    "blocked_branch_count": 0,
                },
                "version": "gen2sim_admission_receipt_v1",
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
                    },
                    "runtime_layout_contract": {
                        "install_ready_profiles": ["unitree_sim_isaaclab"],
                        "install_partial_profiles": [],
                        "install_blocked_profiles": ["isaaclab_core"],
                    },
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
                "metadata": {
                    "runtime_binding": {
                        "host_preflight_status": "preflight_blocked",
                        "host_preflight_missing_components": [
                            "asset::unitree_robot_description"
                        ],
                        "host_preflight_ready_components": [
                            "target::unitree_sim_isaaclab_root"
                        ],
                        "host_preflight_verified_components": [
                            "target::unitree_sim_isaaclab_root"
                        ],
                        "host_preflight_symbolic_components": ["policy_ref"],
                        "selected_ref_evidence": {
                            "policy_ref": {
                                "verification_status": "symbolic_ref",
                            }
                        },
                    }
                },
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
                "metadata": {
                    "missing_preconditions": ["asset::unitree_robot_description"],
                    "notes": ["Launch blocked until verified robot assets are present."],
                },
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
                    },
                    "selected_ref_validation": {
                        "status": "selected_refs_matched",
                        "mismatched_components": [],
                        "missing_components": [],
                    },
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
    assert bundles[0]["physics_execution_contract"]["contract_id"] == "contract_1"
    assert bundles[0]["physics_adaptation_receipt"]["receipt_id"] == "adapt_1"
    assert bundles[0]["backend_execution_binding_receipt"]["receipt_id"] == "binding_1"
    assert bundles[0]["robot_asset_contract_receipt"]["receipt_id"] == "asset_1"
    assert bundles[0]["gen2sim_admission_receipt"]["receipt_id"] == "gen2sim_1"
    assert bundles[0]["backend_runtime_bridge_receipt"]["receipt_id"] == "bridge_1"
    assert bundles[0]["backend_runtime_execution_receipt"]["receipt_id"] == "runtime_1"
    assert bundles[0]["backend_runtime_adapter_receipt"]["receipt_id"] == "adapter_1"
    assert bundles[0]["backend_runtime_launch_receipt"]["receipt_id"] == "launch_1"
    assert bundles[0]["backend_runtime_outcome_receipt"]["receipt_id"] == "runtime_outcome_1"
    assert bundles[0]["backend_shadow_execution_receipt"]["receipt_id"] == "shadow_1"
    assert bundles[0]["physics_calibration_receipt"]["receipt_id"] == "cal_1"
    assert bundles[0]["task_measurement_receipt"]["receipt_id"] == "task_measure_1"
    assert bundles[0]["sim_real_gap_receipt"]["receipt_id"] == "sim_real_gap_1"
    assert bundles[0]["backend_mismatch_receipt"]["receipt_id"] == "backend_mismatch_1"
    assert bundles[0]["surrogate_physics_receipt"]["receipt_id"] == "surrogate_physics_1"
    assert (
        bundles[0]["surrogate_calibration_receipt"]["receipt_id"]
        == "surrogate_calibration_1"
    )
    assert bundles[0]["branch_validity_receipts"][0]["receipt_id"] == "branch_validity_1"
    assert bundles[0]["sensor_alignment_receipt"]["receipt_id"] == "sensor_alignment_1"
    assert bundles[0]["replay_validity_receipts"][0]["receipt_id"] == "replay_validity_1"
    assert bundles[0]["runtime_receipt_manifest"]["manifest_id"] == "runtime_manifest_1"
    manifest_validation = validate_runtime_receipt_manifest(bundles[0])
    assert manifest_validation["validation_status"] == "validated"
    assert manifest_validation["mismatched_families"] == []
    assert bundles[0]["render_provider_receipts"][0]["receipt_id"] == "provider_1"
    assert bundles[0]["simulation_outcome_receipts"][0]["receipt_id"] == "outcome_1"

    backend_rows = build_backend_selector_rows_from_receipts(bundles)
    assert backend_rows[0]["target_hardware_class"] == "unitree_g1_r1_class"
    assert backend_rows[0]["target_system_identification_profile"] == "humanoid_shadow_system_id"
    assert backend_rows[0]["target_source"] == "runtime_receipt"
    assert backend_rows[0]["training_admissibility"]["status"] == "positive_training"
    assert backend_rows[0]["training_admissibility"]["positive_training_admissible"] is True
    assert backend_rows[0]["training_admissibility"]["negative_supervision_eligible"] is False
    assert backend_rows[0]["metadata"]["training_admissibility_status"] == "positive_training"
    assert backend_rows[0]["metadata"]["positive_training_admissible"] is True
    assert backend_rows[0]["metadata"]["robot_asset_contract_receipt_id"] == "asset_1"
    assert backend_rows[0]["metadata"]["task_measurement_receipt_id"] == "task_measure_1"
    assert backend_rows[0]["metadata"]["task_measurement_values"]["coverage_gap_score"] == 0.7
    assert backend_rows[0]["metadata"]["sim_real_gap_receipt_id"] == "sim_real_gap_1"
    assert backend_rows[0]["metadata"]["sim_real_gap_score"] == 0.42
    assert backend_rows[0]["metadata"]["backend_mismatch_receipt_id"] == "backend_mismatch_1"
    assert backend_rows[0]["metadata"]["backend_mismatch_score"] == 0.25
    assert backend_rows[0]["metadata"]["surrogate_physics_receipt_id"] == "surrogate_physics_1"
    assert (
        backend_rows[0]["metadata"]["surrogate_calibration_receipt_id"]
        == "surrogate_calibration_1"
    )
    assert backend_rows[0]["metadata"]["physics_execution_contract_id"] == "contract_1"
    assert backend_rows[0]["metadata"]["physics_route_status"] == "fallback"
    assert backend_rows[0]["metadata"]["compiled_receipt_inventory_id"] == "inventory_1"
    assert backend_rows[0]["metadata"]["runtime_receipt_manifest_id"] == "runtime_manifest_1"
    assert backend_rows[0]["metadata"]["runtime_receipt_manifest_status"] == "complete"
    assert backend_rows[0]["metadata"]["runtime_receipt_missing_required_families"] == []
    assert backend_rows[0]["metadata"]["runtime_receipt_emitted_count"] == 3
    assert (
        backend_rows[0]["metadata"]["runtime_receipt_family_counts"][
            "replay_validity_receipt_v1"
        ]
        == 1
    )
    assert (
        backend_rows[0]["metadata"]["runtime_receipt_manifest_validation_status"]
        == "validated"
    )
    assert backend_rows[0]["metadata"]["runtime_receipt_manifest_mismatched_families"] == []
    assert backend_rows[0]["metadata"]["compiled_runtime_binding_status"] == "assets_missing"
    assert (
        backend_rows[0]["metadata"]["compiled_runtime_bridge_status"]
        == "runtime_targets_missing"
    )
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
    assert backend_rows[0]["metadata"]["branch_validity_receipt_ids"] == [
        "branch_validity_1"
    ]
    assert backend_rows[0]["metadata"]["branch_validity_admissible_count"] == 1
    assert backend_rows[0]["metadata"]["branch_validity_reject_count"] == 0
    assert backend_rows[0]["metadata"]["branch_validity_reject_reasons"] == []
    assert backend_rows[0]["metadata"]["sensor_alignment_receipt_id"] == "sensor_alignment_1"
    assert backend_rows[0]["metadata"]["sensor_alignment_status"] == "geometry_contract_validated"
    assert backend_rows[0]["metadata"]["sensor_alignment_score"] == 0.95
    assert backend_rows[0]["metadata"]["sensor_alignment_checks"]["round_trip"] == "passed"
    assert backend_rows[0]["metadata"]["replay_validity_receipt_ids"] == [
        "replay_validity_1"
    ]
    assert backend_rows[0]["metadata"]["replay_validity_reject_count"] == 0
    assert backend_rows[0]["metadata"]["replay_validity_reject_reasons"] == []
    assert backend_rows[0]["metadata"]["gen2sim_admission_receipt_id"] == "gen2sim_1"
    assert backend_rows[0]["metadata"]["gen2sim_admissible_branch_count"] == 1
    assert backend_rows[0]["metadata"]["backend_runtime_execution_receipt_id"] == "runtime_1"
    assert (
        backend_rows[0]["metadata"]["backend_runtime_execution_status"]
        == "runtime_execution_completed"
    )
    assert (
        backend_rows[0]["metadata"]["backend_runtime_binding_host_preflight_status"]
        == "preflight_blocked"
    )
    assert (
        backend_rows[0]["metadata"]["backend_runtime_binding_host_preflight_missing_components"]
        == ["asset::unitree_robot_description"]
    )
    assert (
        backend_rows[0]["metadata"]["backend_runtime_binding_host_preflight_ready_components"]
        == ["target::unitree_sim_isaaclab_root"]
    )
    assert (
        backend_rows[0]["metadata"]["backend_runtime_binding_host_preflight_verified_components"]
        == ["target::unitree_sim_isaaclab_root"]
    )
    assert (
        backend_rows[0]["metadata"]["backend_runtime_binding_host_preflight_symbolic_components"]
        == ["policy_ref"]
    )
    assert backend_rows[0]["metadata"]["backend_runtime_layout_install_ready_profiles"] == [
        "unitree_sim_isaaclab"
    ]
    assert backend_rows[0]["metadata"]["backend_runtime_layout_install_blocked_profiles"] == [
        "isaaclab_core"
    ]
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_receipt_id"] == "adapter_1"
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_status"] == "external_launch_completed"
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_execution_path"] == "external_launch"
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_realization_path"] == "external_launch_delegate"
    assert backend_rows[0]["metadata"]["backend_runtime_adapter_realization_status"] == "external_launch_delegate_ready"
    assert backend_rows[0]["metadata"]["backend_runtime_launch_receipt_id"] == "launch_1"
    assert backend_rows[0]["metadata"]["backend_runtime_launch_status"] == "launch_completed"
    assert backend_rows[0]["metadata"]["backend_runtime_launch_executed"] is True
    assert backend_rows[0]["metadata"]["backend_runtime_launch_missing_preconditions"] == [
        "asset::unitree_robot_description"
    ]
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
    assert (
        backend_rows[0]["metadata"]["backend_runtime_selected_ref_validation_status"]
        == "selected_refs_matched"
    )
    assert backend_rows[0]["metadata"]["backend_runtime_metric_keys"] == ["metrics.score"]

    branch_rows = build_branch_planner_rows_from_receipts(bundles)
    assert branch_rows[0]["target_render_materialization_status"] == "scene_materialized"
    assert branch_rows[0]["target_render_materialization_mode"] == "scene_config"
    assert branch_rows[0]["training_admissibility"]["status"] == "positive_training"
    assert branch_rows[0]["training_admissibility"]["positive_training_admissible"] is True
    assert branch_rows[0]["training_admissibility"]["negative_supervision_eligible"] is False
    assert branch_rows[0]["metadata"]["training_admissibility_status"] == "positive_training"
    assert branch_rows[0]["metadata"]["positive_training_admissible"] is True
    assert branch_rows[0]["metadata"]["robot_asset_contract_receipt_id"] == "asset_1"
    assert branch_rows[0]["metadata"]["scene_hierarchy_ref"]["hierarchy_id"] == "scene_h_1"
    assert branch_rows[0]["metadata"]["scene_materialization_status"] == "asset_contract_ready"
    assert branch_rows[0]["metadata"]["branch_validity_receipt_id"] == "branch_validity_1"
    assert branch_rows[0]["metadata"]["branch_validity_score"] == 0.66
    assert branch_rows[0]["metadata"]["branch_admission_score"] == 0.61
    assert branch_rows[0]["metadata"]["branch_validity_admissible"] is True
    assert branch_rows[0]["metadata"]["branch_validity_evidence_status"] == "local_estimate"
    assert branch_rows[0]["metadata"]["branch_reject_reasons"] == []
    assert branch_rows[0]["metadata"]["sensor_alignment_receipt_id"] == "sensor_alignment_1"
    assert branch_rows[0]["metadata"]["sensor_alignment_status"] == "geometry_contract_validated"
    assert branch_rows[0]["metadata"]["sensor_alignment_score"] == 0.95
    assert branch_rows[0]["metadata"]["sensor_alignment_checks"]["intrinsics"] == "valid"
    assert branch_rows[0]["metadata"]["replay_validity_receipt_id"] == "replay_validity_1"
    assert branch_rows[0]["metadata"]["replay_validity_score"] == 0.72
    assert branch_rows[0]["metadata"]["replay_validity_status"] == "training_validity_estimated"
    assert branch_rows[0]["metadata"]["replay_task_consistency_score"] == 0.68
    assert branch_rows[0]["metadata"]["replay_transfer_consistency_score"] == 0.58
    assert branch_rows[0]["metadata"]["replay_reject_reasons"] == []
    assert branch_rows[0]["metadata"]["task_measurement_receipt_id"] == "task_measure_1"
    assert branch_rows[0]["metadata"]["sim_real_gap_receipt_id"] == "sim_real_gap_1"
    assert branch_rows[0]["metadata"]["sim_real_gap_score"] == 0.42
    assert branch_rows[0]["metadata"]["backend_mismatch_receipt_id"] == "backend_mismatch_1"
    assert branch_rows[0]["metadata"]["backend_mismatch_score"] == 0.25
    assert branch_rows[0]["metadata"]["physics_execution_contract_id"] == "contract_1"
    assert branch_rows[0]["metadata"]["physics_route_status"] == "fallback"
    assert branch_rows[0]["metadata"]["compiled_receipt_inventory_id"] == "inventory_1"
    assert branch_rows[0]["metadata"]["runtime_receipt_manifest_id"] == "runtime_manifest_1"
    assert branch_rows[0]["metadata"]["runtime_receipt_manifest_status"] == "complete"
    assert branch_rows[0]["metadata"]["runtime_receipt_missing_required_families"] == []
    assert branch_rows[0]["metadata"]["runtime_receipt_emitted_count"] == 3
    assert (
        branch_rows[0]["metadata"]["runtime_receipt_family_counts"][
            "branch_validity_receipt_v1"
        ]
        == 1
    )
    assert (
        branch_rows[0]["metadata"]["runtime_receipt_manifest_validation_status"]
        == "validated"
    )
    assert branch_rows[0]["metadata"]["runtime_receipt_manifest_mismatched_families"] == []
    assert branch_rows[0]["metadata"]["compiled_runtime_binding_status"] == "assets_missing"
    assert (
        branch_rows[0]["metadata"]["compiled_runtime_bridge_status"]
        == "runtime_targets_missing"
    )
    assert branch_rows[0]["metadata"]["robot_asset_readiness_score"] == 0.25
    assert branch_rows[0]["metadata"]["adaptation_receipt_id"] == "adapt_1"
    assert branch_rows[0]["metadata"]["gen2sim_admission_receipt_id"] == "gen2sim_1"
    assert branch_rows[0]["metadata"]["gen2sim_admissible_branch_count"] == 1
    assert branch_rows[0]["metadata"]["backend_runtime_bridge_receipt_id"] == "bridge_1"
    assert branch_rows[0]["metadata"]["backend_runtime_bridge_status"] == "runtime_targets_missing"
    assert (
        branch_rows[0]["metadata"]["backend_runtime_binding_host_preflight_status"]
        == "preflight_blocked"
    )
    assert (
        branch_rows[0]["metadata"]["backend_runtime_binding_host_preflight_missing_components"]
        == ["asset::unitree_robot_description"]
    )
    assert (
        branch_rows[0]["metadata"]["backend_runtime_binding_host_preflight_ready_components"]
        == ["target::unitree_sim_isaaclab_root"]
    )
    assert (
        branch_rows[0]["metadata"]["backend_runtime_binding_host_preflight_verified_components"]
        == ["target::unitree_sim_isaaclab_root"]
    )
    assert (
        branch_rows[0]["metadata"]["backend_runtime_binding_host_preflight_symbolic_components"]
        == ["policy_ref"]
    )
    assert branch_rows[0]["metadata"]["backend_runtime_layout_install_ready_profiles"] == [
        "unitree_sim_isaaclab"
    ]
    assert branch_rows[0]["metadata"]["backend_runtime_layout_install_blocked_profiles"] == [
        "isaaclab_core"
    ]
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_receipt_id"] == "adapter_1"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_status"] == "external_launch_completed"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_execution_path"] == "external_launch"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_realization_path"] == "external_launch_delegate"
    assert branch_rows[0]["metadata"]["backend_runtime_adapter_realization_status"] == "external_launch_delegate_ready"
    assert branch_rows[0]["metadata"]["backend_runtime_launch_receipt_id"] == "launch_1"
    assert branch_rows[0]["metadata"]["backend_runtime_launch_status"] == "launch_completed"
    assert branch_rows[0]["metadata"]["backend_runtime_launch_missing_preconditions"] == [
        "asset::unitree_robot_description"
    ]
    assert (
        branch_rows[0]["metadata"]["backend_runtime_outcome_receipt_id"]
        == "runtime_outcome_1"
    )
    assert branch_rows[0]["metadata"]["backend_runtime_outcome_status"] == "runtime_outputs_harvested"
    assert branch_rows[0]["metadata"]["backend_runtime_output_count"] == 2
    assert (
        branch_rows[0]["metadata"]["backend_runtime_selected_ref_validation_status"]
        == "selected_refs_matched"
    )
    assert branch_rows[0]["metadata"]["backend_shadow_execution_receipt_id"] == "shadow_1"
    assert branch_rows[0]["metadata"]["backend_shadow_execution_status"] == "shadow_executed_with_asset_gaps"
    assert branch_rows[0]["metadata"]["calibration_receipt_id"] == "cal_1"
    assert branch_rows[0]["metadata"]["calibration_quality_score"] == 0.8
    assert (
        branch_rows[0]["metadata"]["branch_helper_resolution"]
        == "heuristic_due_to_shadow_candidate"
    )
    assert (
        branch_rows[0]["metadata"]["branch_helper_resolution_reason"]
        == "benchmark_gate_not_ready"
    )
    assert branch_rows[0]["metadata"]["branch_helper_payload_applied"] is False
    assert (
        branch_rows[0]["metadata"]["branch_helper_trace_generation_mode"]
        == "neural_branch_candidate"
    )
    assert branch_rows[0]["metadata"]["branch_helper_trace_expected_yield_score"] == 0.91
    assert branch_rows[0]["metadata"]["backend_runtime_ready_surfaces"] == [
        "metrics_surface_ready",
        "policy_surface_ready",
    ]
    assert (
        branch_rows[0]["metadata"]["render_artifact_refs"]
        == ["/tmp/render_provider_1/lsd_vector_scene_config.json"]
    )

    filtered_bundles = json.loads(json.dumps(bundles))
    filtered_bundles[0]["replay_validity_receipts"][0]["status"] = "training_filtered_estimate"
    filtered_bundles[0]["replay_validity_receipts"][0]["reject_reasons"] = [
        "sensor_alignment_unready"
    ]
    filtered_backend_rows = build_backend_selector_rows_from_receipts(filtered_bundles)
    assert (
        filtered_backend_rows[0]["training_admissibility"]["status"]
        == "negative_supervision"
    )
    assert (
        filtered_backend_rows[0]["training_admissibility"]["positive_training_admissible"]
        is False
    )
    assert (
        filtered_backend_rows[0]["training_admissibility"]["negative_supervision_eligible"]
        is True
    )
    assert filtered_backend_rows[0]["metadata"]["training_admissibility_reasons"] == [
        "replay_reject_reasons_present"
    ]
    filtered_branch_rows = build_branch_planner_rows_from_receipts(filtered_bundles)
    assert (
        filtered_branch_rows[0]["training_admissibility"]["status"]
        == "negative_supervision"
    )
    assert (
        filtered_branch_rows[0]["training_admissibility"]["positive_training_admissible"]
        is False
    )
    assert (
        filtered_branch_rows[0]["training_admissibility"]["negative_supervision_eligible"]
        is True
    )
    assert filtered_branch_rows[0]["metadata"]["training_admissibility_reasons"] == [
        "replay_reject_reasons_present",
        "replay_validity_filtered",
    ]
    selected_backend_rows, backend_selection_summary = select_phase1x_positive_training_rows(
        [backend_rows[0], filtered_backend_rows[0]]
    )
    assert selected_backend_rows == [backend_rows[0]]
    assert backend_selection_summary["source_row_count"] == 2
    assert backend_selection_summary["selected_row_count"] == 1
    assert backend_selection_summary["excluded_row_count"] == 1
    assert backend_selection_summary["positive_training_row_count"] == 1
    assert backend_selection_summary["negative_supervision_row_count"] == 1
    assert backend_selection_summary["status_counts"] == {
        "negative_supervision": 1,
        "positive_training": 1,
    }
    selected_branch_rows, branch_selection_summary = select_phase1x_positive_training_rows(
        [branch_rows[0], filtered_branch_rows[0]]
    )
    assert selected_branch_rows == [branch_rows[0]]
    assert branch_selection_summary["source_row_count"] == 2
    assert branch_selection_summary["selected_row_count"] == 1
    assert branch_selection_summary["excluded_row_count"] == 1
    assert branch_selection_summary["positive_training_row_count"] == 1
    assert branch_selection_summary["negative_supervision_row_count"] == 1


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
                    },
                    "selected_ref_validation": {
                        "status": "no_expected_selected_refs",
                        "mismatched_components": [],
                        "missing_components": [],
                    },
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


def test_backend_selector_rows_do_not_prefer_mismatched_runtime_outcomes() -> None:
    bundles = [
        {
            "bundle_id": "bundle_mismatch",
            "world_state": {
                "state_id": "sim_state_mismatch",
                "simulation_agenda": {"jobs": [{"job_id": "job_1"}]},
                "physics_context": {
                    "backend": "isaac",
                    "fidelity_tier": "high_fidelity",
                    "domain_randomization_regime": "benchmark_focus",
                    "metadata": {},
                },
            },
            "backend_runtime_bridge_receipt": {
                "receipt_id": "bridge_mismatch",
                "bridge_status": "runtime_bridge_ready",
                "execution_authority": "shadow_runtime",
                "transport_profile": "isaaclab_unitree_dds_bridge",
                "bridge_readiness_score": 0.8,
                "metadata": {
                    "runtime_target_contract": {"missing_required_target_ids": []}
                },
            },
            "backend_runtime_launch_receipt": {
                "receipt_id": "launch_mismatch",
                "launch_status": "launch_completed",
                "executed": True,
            },
            "backend_runtime_outcome_receipt": {
                "receipt_id": "outcome_runtime_mismatch",
                "outcome_status": "runtime_outputs_harvested",
                "harvested_output_count": 2,
                "metadata": {
                    "structured_outputs": {
                        "ready_surfaces": ["policy_surface_ready"],
                        "metric_keys": [],
                        "primary_policy_ref": "/tmp/run/policy.onnx",
                    },
                    "selected_ref_validation": {
                        "status": "selected_refs_mismatched",
                        "mismatched_components": ["policy_ref"],
                        "missing_components": [],
                    },
                },
            },
        }
    ]

    rows = build_backend_selector_rows_from_receipts(bundles)

    assert rows[0]["target_source"] == "external_launch_receipt"
    assert (
        rows[0]["metadata"]["backend_runtime_selected_ref_validation_status"]
        == "selected_refs_mismatched"
    )
