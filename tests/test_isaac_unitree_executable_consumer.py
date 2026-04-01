from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.isaac_unitree_executable_consumer import (
    build_isaac_unitree_executable_adapter_consumer,
)


def test_executable_consumer_prefers_local_python_bridge_for_sim_eval() -> None:
    consumer = build_isaac_unitree_executable_adapter_consumer(
        {
            "request_id": "req_1",
            "preferred_profile": "unitree_sim_isaaclab",
            "adapter_entrypoint": "isaaclab_unitree_sim",
            "deployment_mode": "sim_eval",
            "supports_local_python_bridge": True,
            "task_id": "peg_in_hole",
            "policy_ref": "/tmp/g1.onnx",
            "command": "python sim_main.py",
            "cwd": "/tmp/unitree_sim_isaaclab",
            "missing_preconditions": [],
            "env_overrides": {"UNITREE_DEPLOYMENT_MODE": "sim_eval"},
            "notes": ["request present"],
        }
    )

    assert consumer["consumer_mode"] == "local_python_bridge"
    assert consumer["consumer_status"] == "local_python_bridge_ready"
    assert consumer["external_runtime_required"] is False


def test_executable_consumer_blocks_external_launch_without_command() -> None:
    consumer = build_isaac_unitree_executable_adapter_consumer(
        {
            "request_id": "req_2",
            "preferred_profile": "unitree_lerobot",
            "adapter_entrypoint": "unitree_lerobot_eval",
            "deployment_mode": "lerobot_eval",
            "supports_local_python_bridge": False,
            "task_id": "walk_forward",
            "policy_ref": "/tmp/g1.onnx",
            "command": "",
            "cwd": "/tmp/unitree_lerobot",
            "missing_preconditions": [],
        }
    )

    assert consumer["consumer_mode"] == "external_lerobot_eval"
    assert consumer["consumer_status"] == "consumer_blocked"
    assert "launch_command" in consumer["missing_preconditions"]
