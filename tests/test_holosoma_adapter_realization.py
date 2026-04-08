from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.holosoma_adapter_realization import (
    EXTERNAL_DELEGATE_RUNNER,
    LOCAL_BACKEND_FACTORY,
    LOCAL_BACKEND_NAME,
    build_holosoma_adapter_realization,
)


def _request(**overrides):
    payload = {
        "request_id": "req_1",
        "adapter_family": "holosoma",
        "adapter_entrypoint": "holosoma_repo_eval",
        "preferred_profile": "holosoma_repo",
        "deployment_mode": "sim_eval",
        "task_id": "humanoid_wbt_g1",
        "policy_ref": "/tmp/policy.ckpt",
        "command": "python -m holosoma.eval --task-id humanoid_wbt_g1 --policy /tmp/policy.ckpt",
        "cwd": "/tmp/holosoma",
        "output_expectations": {"artifact_kinds": ["runtime_outputs"]},
    }
    payload.update(overrides)
    return payload


def _consumer(**overrides):
    payload = {
        "consumer_id": "consumer_1",
        "consumer_mode": "external_runtime_launch",
        "consumer_status": "external_launch_consumer_ready",
        "command": "python -m holosoma.eval --task-id humanoid_wbt_g1 --policy /tmp/policy.ckpt",
        "cwd": "/tmp/holosoma",
        "env_overrides": {"HOLOSOMA_CONSUMER_MODE": "external_runtime_launch"},
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


def test_build_holosoma_adapter_realization_for_external_delegate() -> None:
    realization = build_holosoma_adapter_realization(
        executable_adapter_request=_request(),
        executable_adapter_consumer=_consumer(),
        adapter_execution=_execution(),
        runtime_bundle={"preferred_profile": "holosoma_repo"},
        launch_spec={"preferred_profile": "holosoma_repo"},
    )

    assert realization["realization_path"] == "external_launch_delegate"
    assert realization["realization_status"] == "external_launch_delegate_ready"
    assert realization["delegate_runner"] == EXTERNAL_DELEGATE_RUNNER


def test_build_holosoma_adapter_realization_for_local_backend_factory() -> None:
    realization = build_holosoma_adapter_realization(
        executable_adapter_request=_request(),
        executable_adapter_consumer=_consumer(
            consumer_mode="local_runtime_binding",
            consumer_status="local_runtime_binding_ready",
        ),
        adapter_execution=_execution(
            execution_path="local_runtime_binding",
            adapter_status="local_runtime_binding_handed_off",
        ),
        runtime_bundle={"preferred_profile": "holosoma_repo"},
        launch_spec={"preferred_profile": "holosoma_repo"},
        binding_payload={
            "executor_entrypoint": "src.motor_backend.factory:make_motor_backend",
            "binding_status": "runtime_ready",
        },
    )

    assert realization["realization_path"] == "local_backend_factory"
    assert realization["realization_status"] == "local_backend_factory_ready"
    assert realization["backend_name"] == LOCAL_BACKEND_NAME
    assert realization["factory_entrypoint"] == LOCAL_BACKEND_FACTORY

