from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase1_runtime_launch import main as runtime_launch_main
from src.world_model.sim_synth_physics import runtime_launch as runtime_launch_module
from src.world_model.sim_synth_physics.runtime_launch import (
    execute_backend_runtime_launch,
    prepare_backend_runtime_launch,
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
