from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase1_runtime_launch import main as runtime_launch_main
from src.world_model.sim_synth_physics import runtime_launch as runtime_launch_module
from src.world_model.sim_synth_physics.adapters.isaac_unitree_adapter_execution import (
    LOCAL_BRIDGE_MODULE,
)
from src.world_model.sim_synth_physics.runtime_launch import (
    build_backend_runtime_launch_receipt,
    execute_backend_runtime_launch,
    prepare_backend_runtime_launch,
)
from scripts.run_isaac_unitree_executable_adapter import (
    main as run_isaac_unitree_executable_adapter_main,
)


def _isaac_bundle() -> dict[str, object]:
    return {
        "backend": "isaac",
        "runtime_target_contract": {
            "runtime_targets_ready": True,
            "missing_required_target_ids": [],
            "unresolved_one_of_groups": [],
            "targets": [
                {"target_id": "unitree_sim_isaaclab_root", "ref": "/tmp/unitree_sim_isaaclab"},
                {"target_id": "unitree_sdk2_root", "ref": "/tmp/unitree_sdk2"},
            ],
        },
        "policy_contract": {"policy_ready": True},
        "executable_adapter_request": {
            "deployment_mode": "sim_eval",
            "adapter_entrypoint": "isaaclab_unitree_sim",
            "command": "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py --task peg_in_hole --policy /tmp/g1.onnx --headless",
            "cwd": "/tmp/unitree_sim_isaaclab",
            "env_overrides": {
                "UNITREE_DEPLOYMENT_MODE": "sim_eval",
                "UNITREE_ROBOT_VARIANT": "unitree_g1",
            },
            "missing_preconditions": [],
            "notes": ["Executable adapter request present."],
        },
        "executable_adapter_consumer": {
            "consumer_mode": "external_sim_launch",
            "consumer_status": "external_launch_consumer_ready",
            "command": "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py --task peg_in_hole --policy /tmp/g1.onnx --headless",
            "cwd": "/tmp/unitree_sim_isaaclab",
            "env_overrides": {"UNITREE_CONSUMER_MODE": "external_sim_launch"},
            "missing_preconditions": [],
            "notes": ["Executable adapter consumer present."],
        },
    }


def test_prepare_backend_runtime_launch_ready(monkeypatch) -> None:
    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)

    plan = prepare_backend_runtime_launch(
        _isaac_bundle(),
        {
            "backend": "isaac",
            "preferred_profile": "unitree_sim_isaaclab",
            "policy_ready": True,
            "command": "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py --task peg_in_hole --policy /tmp/g1.onnx --headless",
            "root": "/tmp/unitree_sim_isaaclab",
            "policy_ref": "/tmp/g1.onnx",
        },
    )

    assert plan["status"] == "ready_for_launch"
    assert plan["env_overrides"]["UNITREE_SIM_ISAACLAB_ROOT"] == "/tmp/unitree_sim_isaaclab"
    assert plan["env_overrides"]["UNITREE_SDK2_ROOT"] == "/tmp/unitree_sdk2"
    assert plan["env_overrides"]["UNITREE_DEPLOYMENT_MODE"] == "sim_eval"
    assert plan["env_overrides"]["UNITREE_CONSUMER_MODE"] == "external_sim_launch"
    assert plan["executable_adapter_request"]["adapter_entrypoint"] == "isaaclab_unitree_sim"
    assert plan["executable_adapter_consumer"]["consumer_mode"] == "external_sim_launch"


def test_prepare_backend_runtime_launch_blocks_when_policy_and_gpu_missing(monkeypatch) -> None:
    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: False)

    plan = prepare_backend_runtime_launch(
        {
            "backend": "holosoma",
            "runtime_target_contract": {
                "runtime_targets_ready": False,
                "missing_required_target_ids": ["holosoma_motion_root"],
                "unresolved_one_of_groups": [["holosoma_root", "holosoma_policy_root"]],
                "targets": [],
            },
            "policy_contract": {"policy_ready": False},
        },
        {"backend": "holosoma", "command": "", "policy_ready": False},
    )

    assert plan["status"] == "blocked"
    assert "linux_host" in plan["missing_preconditions"]
    assert "cuda_gpu" in plan["missing_preconditions"]
    assert "policy_checkpoint" in plan["missing_preconditions"]


def test_prepare_backend_runtime_launch_consumes_non_asset_host_preflight_truth(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)

    bundle = _isaac_bundle()
    bundle["runtime_binding"] = {
        "host_preflight_status": "preflight_blocked",
        "host_preflight_missing_components": [
            "target::unitree_sdk2_root",
            "asset::unitree_robot_description",
        ],
        "missing_components": [],
    }

    plan = prepare_backend_runtime_launch(
        bundle,
        {
            "backend": "isaac",
            "preferred_profile": "unitree_sim_isaaclab",
            "policy_ready": True,
            "command": "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py --task peg_in_hole --policy /tmp/g1.onnx --headless",
            "root": "/tmp/unitree_sim_isaaclab",
            "policy_ref": "/tmp/g1.onnx",
            "runtime_binding": bundle["runtime_binding"],
        },
    )

    assert plan["status"] == "blocked"
    assert "target::unitree_sdk2_root" in plan["missing_preconditions"]
    assert "asset::unitree_robot_description" in plan["missing_preconditions"]
    assert plan["host_preflight_status"] == "preflight_blocked"


def test_run_phase1_runtime_launch_script_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "backend_runtime_bundle.json").write_text(
        json.dumps(_isaac_bundle(), indent=2),
        encoding="utf-8",
    )
    (runtime_root / "backend_launch_spec.json").write_text(
        json.dumps(
            {
                "backend": "isaac",
                "preferred_profile": "unitree_sim_isaaclab",
                "policy_ready": True,
                "command": "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py --task peg_in_hole --policy /tmp/g1.onnx --headless",
                "root": "/tmp/unitree_sim_isaaclab",
                "policy_ref": "/tmp/g1.onnx",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "launch_report.json"

    payload = runtime_launch_main(
        [
            "--runtime-root",
            str(runtime_root),
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    assert payload["result"]["status"] == "ready_for_launch"
    assert payload["result"]["executed"] is False
    assert payload["receipt"]["launch_status"] == "launch_prepared"
    assert payload["receipt"]["backend"] == "isaac"


def test_execute_backend_runtime_launch_stays_dry_without_execute(monkeypatch) -> None:
    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)

    result = execute_backend_runtime_launch(
        _isaac_bundle(),
        {
            "backend": "isaac",
            "preferred_profile": "unitree_sim_isaaclab",
            "policy_ready": True,
            "command": "echo runtime_launch",
            "root": ".",
            "policy_ref": "/tmp/g1.onnx",
        },
        execute=False,
    )

    assert result["status"] == "ready_for_launch"
    assert result["executed"] is False


def test_build_backend_runtime_launch_receipt_maps_ready_status() -> None:
    receipt = build_backend_runtime_launch_receipt(
        _isaac_bundle(),
        {
            "backend": "isaac",
            "preferred_profile": "unitree_sim_isaaclab",
            "command": "echo hello",
            "root": "/tmp/unitree_sim_isaaclab",
        },
        {
            "backend": "isaac",
            "preferred_profile": "unitree_sim_isaaclab",
            "status": "ready_for_launch",
            "command": "echo hello",
            "cwd": "/tmp/unitree_sim_isaaclab",
        "executed": False,
        "executable_adapter_consumer": {"consumer_mode": "external_sim_launch"},
    },
    )

    assert receipt.launch_status == "launch_prepared"
    assert receipt.executed is False
    assert receipt.launch_profile == "unitree_sim_isaaclab"


def test_run_phase1_runtime_launch_script_writes_pure_receipt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "backend_runtime_bundle.json").write_text(
        json.dumps(_isaac_bundle(), indent=2),
        encoding="utf-8",
    )
    (runtime_root / "backend_launch_spec.json").write_text(
        json.dumps(
            {
                "backend": "isaac",
                "preferred_profile": "unitree_sim_isaaclab",
                "policy_ready": True,
                "command": "echo runtime_launch",
                "root": ".",
                "policy_ref": "/tmp/g1.onnx",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "launch_receipt.json"

    runtime_launch_main(
        [
            "--runtime-root",
            str(runtime_root),
            "--receipt-output",
            str(receipt_path),
        ]
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["version"] == "backend_runtime_launch_receipt_v1"
    assert receipt["launch_status"] == "launch_prepared"


def test_run_isaac_unitree_executable_adapter_script_writes_adapter_request(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    bundle = _isaac_bundle()
    launch_spec = {
        "backend": "isaac",
        "preferred_profile": "unitree_sim_isaaclab",
        "policy_ready": True,
        "command": "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py --task peg_in_hole --policy /tmp/g1.onnx --headless",
        "root": "/tmp/unitree_sim_isaaclab",
        "policy_ref": "/tmp/g1.onnx",
        "executable_adapter_request": bundle["executable_adapter_request"],
    }
    (runtime_root / "backend_runtime_bundle.json").write_text(
        json.dumps(bundle, indent=2),
        encoding="utf-8",
    )
    (runtime_root / "backend_launch_spec.json").write_text(
        json.dumps(launch_spec, indent=2),
        encoding="utf-8",
    )
    output_path = tmp_path / "adapter_report.json"

    payload = run_isaac_unitree_executable_adapter_main(
        [
            "--runtime-root",
            str(runtime_root),
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    assert payload["executable_adapter_request"]["deployment_mode"] == "sim_eval"
    assert payload["executable_adapter_consumer"]["consumer_mode"] == "external_sim_launch"
    assert payload["adapter_execution"]["execution_path"] == "external_launch"
    assert payload["adapter_execution"]["adapter_status"] == "external_launch_ready"
    assert payload["adapter_realization"]["realization_path"] == "external_launch_delegate"
    assert payload["adapter_realization"]["realization_status"] == "external_launch_delegate_ready"
    assert payload["adapter_execution"]["local_bridge_module"] == LOCAL_BRIDGE_MODULE
    assert payload["adapter_receipt"]["version"] == "backend_runtime_adapter_receipt_v1"
    assert payload["adapter_receipt"]["adapter_status"] == "external_launch_ready"
    assert (
        payload["adapter_receipt"]["metadata"]["realization"]["realization_path"]
        == "external_launch_delegate"
    )
    assert payload["receipt"]["launch_status"] == "launch_prepared"


def test_run_phase1_runtime_launch_harvests_outcomes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)

    runtime_root = tmp_path / "unitree_sim_isaaclab"
    runtime_root.mkdir()
    logs_dir = runtime_root / "logs" / "run_1"
    logs_dir.mkdir(parents=True)
    (logs_dir / "policy.onnx").write_text("x", encoding="utf-8")
    (logs_dir / "metrics.json").write_text("{}", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_path = policy_root / "g1_policy.onnx"
    policy_path.write_text("x", encoding="utf-8")
    (runtime_root / "backend_runtime_bundle.json").write_text(
        json.dumps(
            {
                "backend": "isaac",
                "preferred_profile": "unitree_sim_isaaclab",
                "runtime_target_contract": {
                    "runtime_targets_ready": True,
                    "missing_required_target_ids": [],
                    "unresolved_one_of_groups": [],
                    "targets": [
                        {"target_id": "unitree_sim_isaaclab_root", "ref": str(runtime_root)},
                        {"target_id": "unitree_sdk2_root", "ref": str(tmp_path / "sdk2")},
                    ],
                },
                "policy_contract": {
                    "policy_ready": True,
                    "policy_root": str(policy_root),
                    "policy_ref": str(policy_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (runtime_root / "backend_launch_spec.json").write_text(
        json.dumps(
            {
                "backend": "isaac",
                "preferred_profile": "unitree_sim_isaaclab",
                "policy_ready": True,
                "command": "echo runtime_launch",
                "root": str(runtime_root),
                "policy_ref": str(policy_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outcome_path = tmp_path / "runtime_outcome_receipt.json"

    payload = runtime_launch_main(
        [
            "--runtime-root",
            str(runtime_root),
            "--harvest-outcomes",
            "--outcome-output",
            str(outcome_path),
        ]
    )

    assert payload["outcome_receipt"]["outcome_status"] == "launch_not_executed"
    assert payload["output_summary"]["harvested_output_count"] == 0
    outcome_receipt = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome_receipt["version"] == "backend_runtime_outcome_receipt_v1"
