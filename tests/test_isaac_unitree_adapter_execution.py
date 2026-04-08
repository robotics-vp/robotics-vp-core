from __future__ import annotations

from src.world_model.sim_synth_physics.adapters import isaac_unitree_adapter_execution as module


def _request(**overrides):
    payload = {
        "request_id": "req_1",
        "adapter_family": "isaac_unitree",
        "adapter_entrypoint": "isaaclab_unitree_sim",
        "deployment_mode": "sim_eval",
        "task_id": "peg_in_hole",
        "policy_ref": "/tmp/g1.onnx",
        "robot_variant": "unitree_g1",
        "placement_class": "companion_gpu_required",
        "command": "python sim_main.py --task peg_in_hole --policy /tmp/g1.onnx",
        "cwd": "/tmp/unitree_sim_isaaclab",
        "env_overrides": {"UNITREE_DEPLOYMENT_MODE": "sim_eval"},
        "notes": ["request present"],
    }
    payload.update(overrides)
    return payload


def _consumer(**overrides):
    payload = {
        "consumer_id": "consumer_1",
        "consumer_mode": "external_sim_launch",
        "consumer_status": "external_launch_consumer_ready",
        "command": "python sim_main.py --task peg_in_hole --policy /tmp/g1.onnx",
        "cwd": "/tmp/unitree_sim_isaaclab",
        "env_overrides": {"UNITREE_CONSUMER_MODE": "external_sim_launch"},
        "missing_preconditions": [],
        "notes": ["consumer present"],
    }
    payload.update(overrides)
    return payload


def test_prepare_adapter_execution_marks_external_launch_ready() -> None:
    execution = module.prepare_isaac_unitree_adapter_execution(_request(), _consumer())

    assert execution["execution_path"] == "external_launch"
    assert execution["adapter_status"] == "external_launch_ready"
    assert execution["local_bridge_available"] is False


def test_prepare_adapter_execution_marks_missing_local_bridge(monkeypatch) -> None:
    monkeypatch.setattr(module, "_has_local_bridge_module", lambda: False)

    execution = module.prepare_isaac_unitree_adapter_execution(
        _request(),
        _consumer(
            consumer_mode="local_python_bridge",
            consumer_status="local_python_bridge_ready",
        ),
    )

    assert execution["execution_path"] == "local_python_bridge"
    assert execution["adapter_status"] == "local_bridge_missing"
    assert "local_python_bridge_module" in execution["missing_preconditions"]


def test_finalize_adapter_execution_builds_receipt() -> None:
    execution = module.prepare_isaac_unitree_adapter_execution(_request(), _consumer())
    execution = module.finalize_isaac_unitree_adapter_execution(
        execution,
        launch_result={
            "status": "launch_completed",
            "executed": True,
            "returncode": 0,
        },
    )
    receipt = module.build_isaac_unitree_adapter_receipt(
        execution,
        artifact_refs=["/tmp/backend_runtime_adapter_execution.json"],
    )

    assert execution["adapter_status"] == "external_launch_completed"
    assert receipt.adapter_status == "external_launch_completed"
    assert receipt.execution_path == "external_launch"
    assert receipt.metadata["execution_id"] == execution["execution_id"]
    assert receipt.artifact_refs == ["/tmp/backend_runtime_adapter_execution.json"]
