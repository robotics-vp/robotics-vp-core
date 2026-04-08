from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.holosoma_executable_adapter import (
    build_holosoma_executable_adapter_request,
)


def test_holosoma_executable_adapter_request_marks_motion_train() -> None:
    request = build_holosoma_executable_adapter_request(
        task_id="humanoid_wbt_g1",
        policy_ref="",
        preferred_profile="holosoma_motion_bank",
        launch_spec={
            "command": "python scripts/local_holosoma_smoke.py --task-id humanoid_wbt_g1",
            "root": "/tmp/motions",
        },
        runtime_target_contract={
            "ready_target_ids": ["holosoma_motion_root", "holosoma_root"],
            "python_bridge_available": True,
            "required_target_ids": ["holosoma_motion_root"],
            "targets": [
                {"target_id": "holosoma_motion_root", "ref": "/tmp/motions"},
                {"target_id": "holosoma_root", "ref": "/tmp/holosoma"},
            ],
        },
        policy_contract={"policy_ready": False},
        normalized_robot_asset_manifest={
            "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"}
        },
        robot_contract_context={
            "robot_asset_contract_id": "robot_contract_g1",
            "calibration_contracts": ["imu"],
            "observation_contracts": ["joint_state"],
            "action_contracts": ["joint_targets"],
        },
        output_contract={
            "profile_id": "holosoma_motion_bank",
            "source_specs": [{"source_id": "runtime_root", "artifact_kind": "runtime_outputs"}],
        },
    )

    assert request["deployment_mode"] == "motion_train"
    assert request["adapter_entrypoint"] == "holosoma_motion_train"
    assert request["supports_local_runtime_binding"] is True
    assert request["missing_preconditions"] == []
    assert request["env_overrides"]["HOLOSOMA_MOTION_TRAIN_ENABLED"] == "1"
