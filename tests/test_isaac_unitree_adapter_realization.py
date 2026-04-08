from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.isaac_unitree_adapter_realization import (
    EXTERNAL_DELEGATE_RUNNER,
    LOCAL_BACKEND_FACTORY,
    LOCAL_BACKEND_NAME,
    build_isaac_unitree_adapter_realization,
)


def _request(**overrides):
    payload = {
        "request_id": "req_1",
        "adapter_family": "isaac_unitree",
        "adapter_entrypoint": "isaaclab_unitree_sim",
        "preferred_profile": "unitree_sim_isaaclab",
        "deployment_mode": "sim_eval",
        "task_id": "peg_in_hole",
        "policy_ref": "/tmp/g1.onnx",
        "command": "python sim_main.py",
        "cwd": "/tmp/unitree_sim_isaaclab",
        "output_expectations": {"artifact_kinds": ["runtime_outputs"]},
    }
    payload.update(overrides)
    return payload


def _consumer(**overrides):
    payload = {
        "consumer_id": "consumer_1",
        "consumer_mode": "external_sim_launch",
        "consumer_status": "external_launch_consumer_ready",
        "command": "python sim_main.py",
        "cwd": "/tmp/unitree_sim_isaaclab",
        "env_overrides": {"UNITREE_CONSUMER_MODE": "external_sim_launch"},
    }
    payload.update(overrides)
    return payload


def _execution(**overrides):
    payload = {
        "execution_id": "exec_1",
        "execution_path": "external_launch",
        "adapter_status": "external_launch_ready",
        "deployment_mode": "sim_eval",
    }
    payload.update(overrides)
    return payload


def test_build_adapter_realization_for_external_delegate() -> None:
    realization = build_isaac_unitree_adapter_realization(
        executable_adapter_request=_request(),
        executable_adapter_consumer=_consumer(),
        adapter_execution=_execution(),
        runtime_bundle={"preferred_profile": "unitree_sim_isaaclab"},
        launch_spec={"preferred_profile": "unitree_sim_isaaclab"},
    )

    assert realization["realization_path"] == "external_launch_delegate"
    assert realization["realization_status"] == "external_launch_delegate_ready"
    assert realization["delegate_runner"] == EXTERNAL_DELEGATE_RUNNER


def test_build_adapter_realization_for_local_backend_factory() -> None:
    realization = build_isaac_unitree_adapter_realization(
        executable_adapter_request=_request(),
        executable_adapter_consumer=_consumer(
            consumer_mode="local_python_bridge",
            consumer_status="local_python_bridge_ready",
        ),
        adapter_execution=_execution(
            execution_path="local_python_bridge",
            adapter_status="local_bridge_handed_off",
        ),
        runtime_bundle={"preferred_profile": "unitree_sim_isaaclab"},
        launch_spec={"preferred_profile": "unitree_sim_isaaclab"},
        binding_payload={
            "executor_entrypoint": "src.motor_backend.factory:make_motor_backend",
            "binding_status": "runtime_ready",
        },
    )

    assert realization["realization_path"] == "local_backend_factory"
    assert realization["realization_status"] == "local_backend_factory_ready"
    assert realization["backend_name"] == LOCAL_BACKEND_NAME
    assert realization["factory_entrypoint"] == LOCAL_BACKEND_FACTORY
