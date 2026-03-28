from __future__ import annotations

from pathlib import Path

from src.world_model.sim_synth_physics.runtime_outcomes import (
    build_backend_runtime_outcome_receipt,
    build_backend_runtime_output_contract,
    harvest_backend_runtime_outcomes,
)
from src.world_model.sim_synth_physics.runtime_launch import (
    build_backend_runtime_launch_receipt,
)


def test_runtime_outcomes_harvest_unitree_sim_outputs(tmp_path: Path) -> None:
    runtime_root = tmp_path / "unitree_sim_isaaclab"
    runtime_root.mkdir()
    (runtime_root / "sim_main.py").write_text("", encoding="utf-8")
    logs_dir = runtime_root / "logs" / "run_1"
    logs_dir.mkdir(parents=True)
    (logs_dir / "policy.onnx").write_text("x", encoding="utf-8")
    (logs_dir / "metrics.json").write_text("{}", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_ref = policy_root / "g1_policy.onnx"
    policy_ref.write_text("x", encoding="utf-8")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "g1.usd").write_text("usd", encoding="utf-8")

    runtime_bundle = {
        "backend": "isaac",
        "preferred_profile": "unitree_sim_isaaclab",
        "runtime_target_contract": {
            "targets": [
                {"target_id": "unitree_sim_isaaclab_root", "ref": str(runtime_root)},
                {"target_id": "unitree_asset_root", "ref": str(asset_root)},
            ]
        },
        "policy_contract": {
            "policy_root": str(policy_root),
            "policy_ref": str(policy_ref),
        },
    }
    launch_spec = {
        "backend": "isaac",
        "preferred_profile": "unitree_sim_isaaclab",
        "root": str(runtime_root),
        "policy_ref": str(policy_ref),
        "command": "python sim_main.py --task g1",
    }
    output_contract = build_backend_runtime_output_contract(runtime_bundle, launch_spec)
    output_summary = harvest_backend_runtime_outcomes(output_contract, executed=True)
    launch_receipt = build_backend_runtime_launch_receipt(
        runtime_bundle,
        launch_spec,
        {
            "backend": "isaac",
            "preferred_profile": "unitree_sim_isaaclab",
            "status": "launch_completed",
            "executed": True,
            "command": launch_spec["command"],
            "cwd": str(runtime_root),
        },
    )
    outcome_receipt = build_backend_runtime_outcome_receipt(
        runtime_bundle=runtime_bundle,
        launch_receipt=launch_receipt,
        output_summary=output_summary,
    )

    assert output_contract["profile_id"] == "unitree_sim_isaaclab"
    assert output_summary["outcome_status"] == "runtime_outputs_harvested"
    assert output_summary["harvested_output_count"] >= 3
    assert outcome_receipt.outcome_status == "runtime_outputs_harvested"
    assert outcome_receipt.harvested_output_count >= 3
    assert any(ref.endswith("metrics.json") for ref in outcome_receipt.artifact_refs)
