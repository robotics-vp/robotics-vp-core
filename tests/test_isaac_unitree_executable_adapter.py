from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.isaac_unitree_executable_adapter import (
    build_isaac_unitree_executable_adapter_request,
)


def test_executable_adapter_request_marks_teleop_mode_and_env_overrides() -> None:
    request = build_isaac_unitree_executable_adapter_request(
        task_id="teleop_eval",
        policy_ref="/tmp/g1.onnx",
        preferred_profile="xr_teleoperate",
        launch_spec={
            "command": "python ${XR_TELEOPERATE_ROOT}/teleop/run_teleop.py --task teleop_eval --policy /tmp/g1.onnx",
            "root": "/tmp/xr_teleoperate",
        },
        runtime_target_contract={
            "ready_target_ids": [
                "xr_teleoperate_root",
                "unitree_sdk2_root",
                "unitree_sdk2_python_root",
                "teleimager_root",
                "unitree_asset_root",
            ],
            "python_bridge_available": False,
            "targets": [
                {"target_id": "xr_teleoperate_root", "ref": "/tmp/xr_teleoperate"},
                {"target_id": "unitree_sdk2_root", "ref": "/tmp/sdk2"},
                {"target_id": "unitree_sdk2_python_root", "ref": "/tmp/sdk2_python"},
                {"target_id": "teleimager_root", "ref": "/tmp/teleimager"},
                {"target_id": "unitree_asset_root", "ref": "/tmp/assets"},
            ],
        },
        deployment_contract={
            "robot_variant": "unitree_g1",
            "placement_class": "unitree_onboard_plus_companion",
            "physical_deploy_ready": False,
            "deployment_modes": [
                {
                    "mode_id": "teleop_bridge",
                    "required_target_ids": [
                        "unitree_sdk2_root",
                        "unitree_sdk2_python_root",
                        "teleimager_root",
                        "xr_teleoperate_root",
                    ],
                    "required_asset_ids": [
                        "unitree_robot_description",
                        "camera_extrinsics",
                        "imu_extrinsics",
                        "safety_watchdog_profile",
                    ],
                    "missing_preconditions": [],
                }
            ],
        },
        normalized_robot_asset_manifest={
            "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"},
            "camera_extrinsics": {"present": True, "value": "/assets/camera.json"},
            "imu_extrinsics": {"present": True, "value": "/assets/imu.json"},
            "safety_watchdog_profile": {"present": True, "value": "/assets/watchdog.yaml"},
        },
        robot_contract_context={
            "robot_asset_contract_id": "robot_contract_g1",
            "calibration_contracts": ["camera", "imu"],
            "observation_contracts": ["rgb", "joint_state"],
            "action_contracts": ["joint_targets"],
        },
        output_contract={
            "profile_id": "xr_teleoperate",
            "source_specs": [
                {"source_id": "runtime_root", "artifact_kind": "runtime_outputs"},
                {"source_id": "teleop_certs", "artifact_kind": "deploy_contracts"},
            ],
        },
    )

    assert request["deployment_mode"] == "teleop_bridge"
    assert request["adapter_entrypoint"] == "unitree_xr_teleop_bridge"
    assert request["robot_asset_contract_id"] == "robot_contract_g1"
    assert request["env_overrides"]["UNITREE_TELEOP_ENABLED"] == "1"
    assert request["env_overrides"]["XR_TELEOPERATE_ROOT"] == "/tmp/xr_teleoperate"
    assert request["asset_refs"]["camera_extrinsics"] == "/assets/camera.json"
    assert request["output_expectations"]["artifact_kinds"] == [
        "deploy_contracts",
        "runtime_outputs",
    ]


def test_executable_adapter_request_preserves_missing_physical_preconditions() -> None:
    request = build_isaac_unitree_executable_adapter_request(
        task_id="walk_forward",
        policy_ref="/tmp/r1.onnx",
        preferred_profile="unitree_lerobot",
        launch_spec={"command": "python eval_policy.py", "root": "/tmp/lerobot"},
        runtime_target_contract={
            "ready_target_ids": [
                "unitree_sdk2_root",
                "unitree_il_lerobot_root",
                "unitree_asset_root",
            ],
            "python_bridge_available": True,
            "targets": [],
        },
        deployment_contract={
            "robot_variant": "unitree_r1",
            "placement_class": "unitree_onboard_plus_companion",
            "physical_deploy_ready": False,
            "deployment_modes": [
                {
                    "mode_id": "lerobot_eval",
                    "required_target_ids": [
                        "unitree_sdk2_root",
                        "unitree_il_lerobot_root",
                        "unitree_asset_root",
                    ],
                    "required_asset_ids": [
                        "unitree_robot_description",
                        "whole_body_joint_map",
                    ],
                    "missing_preconditions": [],
                }
            ],
        },
        normalized_robot_asset_manifest={
            "unitree_robot_description": {"present": True},
            "whole_body_joint_map": {"present": True},
        },
        output_contract={"profile_id": "unitree_lerobot", "source_specs": []},
    )

    assert request["deployment_mode"] == "lerobot_eval"
    assert request["adapter_entrypoint"] == "unitree_lerobot_eval"
    assert request["supports_local_python_bridge"] is True
    assert request["missing_preconditions"] == []
