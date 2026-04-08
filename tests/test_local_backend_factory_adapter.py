from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.local_backend_factory_adapter import (
    SUPPORTED_FACTORY_ENTRYPOINT,
    build_local_backend_factory_invocation,
    materialize_local_backend_factory_invocation,
)


def test_build_local_backend_factory_invocation_ready() -> None:
    invocation = build_local_backend_factory_invocation(
        backend="isaac",
        executable_adapter_request={
            "request_id": "req_1",
            "adapter_family": "isaac_unitree",
            "adapter_entrypoint": "isaaclab_unitree_sim",
            "task_id": "peg_in_hole",
            "policy_ref": "/tmp/policy.onnx",
        },
        executable_adapter_consumer={
            "consumer_id": "consumer_1",
            "consumer_mode": "local_python_bridge",
            "command": "python sim_main.py",
            "cwd": "/tmp/unitree_sim_isaaclab",
        },
        adapter_execution={
            "execution_id": "exec_1",
            "execution_path": "local_python_bridge",
            "missing_preconditions": [],
        },
        adapter_realization={
            "realization_id": "real_1",
            "realization_path": "local_backend_factory",
            "realization_status": "local_backend_factory_ready",
            "backend_name": "workcell_isaaclab",
            "factory_entrypoint": SUPPORTED_FACTORY_ENTRYPOINT,
        },
        binding_payload={"binding_status": "ready"},
    )

    assert invocation["invocation_status"] == "local_backend_invocation_ready"
    assert invocation["backend_name"] == "workcell_isaaclab"


def test_materialize_local_backend_factory_invocation_returns_result(monkeypatch) -> None:
    invocation = {
        "invocation_id": "inv_1",
        "backend": "holosoma",
        "backend_name": "holosoma",
        "factory_entrypoint": SUPPORTED_FACTORY_ENTRYPOINT,
        "invocation_status": "local_backend_invocation_ready",
        "backend_config": {},
        "task_id": "humanoid_wbt_g1",
        "policy_ref": "/tmp/policy.onnx",
        "notes": ["ready"],
    }
    sentinel = object()
    monkeypatch.setattr(
        "src.world_model.sim_synth_physics.adapters.local_backend_factory_adapter.make_motor_backend",
        lambda backend_name, econ_meter, store, backend_config=None: sentinel,
    )

    backend, result = materialize_local_backend_factory_invocation(
        invocation,
        econ_meter=object(),
        store=object(),
    )

    assert backend is sentinel
    assert result["result_status"] == "local_backend_materialized"

