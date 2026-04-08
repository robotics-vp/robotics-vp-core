from __future__ import annotations

from src.world_model.sim_synth_physics.adapters import holosoma_adapter_execution as module


def _request(**overrides):
    payload = {
        "request_id": "req_1",
        "adapter_family": "holosoma",
        "adapter_entrypoint": "holosoma_repo_eval",
        "deployment_mode": "sim_eval",
        "task_id": "humanoid_wbt_g1",
        "policy_ref": "/tmp/policy.ckpt",
        "command": "python -m holosoma.eval --task-id humanoid_wbt_g1 --policy /tmp/policy.ckpt",
        "cwd": "/tmp/holosoma",
        "env_overrides": {"HOLOSOMA_DEPLOYMENT_MODE": "sim_eval"},
        "notes": ["request present"],
    }
    payload.update(overrides)
    return payload


def _consumer(**overrides):
    payload = {
        "consumer_id": "consumer_1",
        "consumer_mode": "local_runtime_binding",
        "consumer_status": "local_runtime_binding_ready",
        "command": "python -m holosoma.eval --task-id humanoid_wbt_g1 --policy /tmp/policy.ckpt",
        "cwd": "/tmp/holosoma",
        "env_overrides": {"HOLOSOMA_CONSUMER_MODE": "local_runtime_binding"},
        "missing_preconditions": [],
        "notes": ["consumer present"],
    }
    payload.update(overrides)
    return payload


def test_prepare_holosoma_adapter_execution_marks_local_binding_ready(monkeypatch) -> None:
    monkeypatch.setattr(module, "_has_local_runtime_module", lambda: True)

    execution = module.prepare_holosoma_adapter_execution(_request(), _consumer())

    assert execution["execution_path"] == "local_runtime_binding"
    assert execution["adapter_status"] == "local_runtime_binding_ready"


def test_finalize_holosoma_adapter_execution_builds_receipt() -> None:
    execution = module.prepare_holosoma_adapter_execution(
        _request(),
        _consumer(
            consumer_mode="external_runtime_launch",
            consumer_status="external_launch_consumer_ready",
        ),
    )
    execution = module.finalize_holosoma_adapter_execution(
        execution,
        launch_result={"status": "launch_completed", "executed": True, "returncode": 0},
    )
    receipt = module.build_holosoma_adapter_receipt(
        execution,
        artifact_refs=["/tmp/backend_runtime_adapter_execution.json"],
    )

    assert execution["adapter_status"] == "external_launch_completed"
    assert receipt.adapter_status == "external_launch_completed"
    assert receipt.execution_path == "external_launch"

