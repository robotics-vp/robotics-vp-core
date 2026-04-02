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
    assert (
        output_summary["structured_outputs"]["surface_ready"]["policy_surface_ready"] is True
    )
    assert (
        output_summary["structured_outputs"]["surface_ready"]["metrics_surface_ready"] is True
    )
    assert output_summary["structured_outputs"]["primary_policy_ref"].endswith("policy.onnx")
    assert output_summary["structured_outputs"]["runtime_metrics_refs"]
    assert outcome_receipt.outcome_status == "runtime_outputs_harvested"
    assert outcome_receipt.harvested_output_count >= 3
    assert any(ref.endswith("metrics.json") for ref in outcome_receipt.artifact_refs)
    assert (
        outcome_receipt.metadata["structured_outputs"]["surface_ready"]["policy_surface_ready"]
        is True
    )


def test_runtime_outcomes_harvest_local_runtime_artifacts_without_launch(tmp_path: Path) -> None:
    episode_dir = tmp_path / "rollouts" / "scenario_1" / "episode_000"
    episode_dir.mkdir(parents=True)
    trajectory_path = episode_dir / "trajectory.npz"
    trajectory_path.write_bytes(b"fake")
    metrics_path = tmp_path / "backend_runtime_metrics.json"
    metrics_path.write_text('{"success_rate": 1.0}', encoding="utf-8")
    policy_path = tmp_path / "trained_policy.onnx"
    policy_path.write_text("x", encoding="utf-8")

    runtime_bundle = {
        "backend": "holosoma",
        "preferred_profile": "holosoma_repo",
        "runtime_target_contract": {"targets": []},
        "policy_contract": {"policy_root": str(tmp_path), "policy_ref": ""},
    }
    launch_spec = {
        "backend": "holosoma",
        "preferred_profile": "holosoma_repo",
        "root": str(tmp_path),
        "policy_ref": "",
        "command": "",
    }
    output_contract = build_backend_runtime_output_contract(runtime_bundle, launch_spec)
    output_summary = harvest_backend_runtime_outcomes(
        output_contract,
        executed=True,
        explicit_artifact_refs=[str(trajectory_path), str(metrics_path)],
        explicit_policy_ref=str(policy_path),
    )
    outcome_receipt = build_backend_runtime_outcome_receipt(
        runtime_bundle=runtime_bundle,
        launch_receipt=None,
        output_summary=output_summary,
    )

    assert output_summary["outcome_status"] == "runtime_outputs_harvested"
    assert output_summary["structured_outputs"]["surface_ready"]["dataset_surface_ready"] is True
    assert output_summary["structured_outputs"]["surface_ready"]["metrics_surface_ready"] is True
    assert output_summary["structured_outputs"]["surface_ready"]["policy_surface_ready"] is True
    assert output_summary["structured_outputs"]["counts"]["dataset_episode_count"] == 1
    assert outcome_receipt.outcome_status == "runtime_outputs_harvested"
    assert outcome_receipt.executed is True
    assert outcome_receipt.metadata["harvest_mode"] == "local_runtime_execution"
    assert outcome_receipt.metadata["launch_receipt_id"] == ""
