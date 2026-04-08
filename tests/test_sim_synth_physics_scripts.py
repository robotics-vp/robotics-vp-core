from __future__ import annotations

import json
from pathlib import Path

from scripts.compile_sim_synth_physics_plan import main as compile_plan_main
from scripts.run_sim_synth_physics_loop import main as run_loop_main
from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph


def _make_test_graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("hrl:grasp_handle", "skill", "Grasp Handle"),
            CoverageNode("prim:locate_handle", "env_primitive", "Locate Handle"),
        ],
        edges=[
            CoverageEdge(
                "hrl:grasp_handle",
                "prim:locate_handle",
                "requires",
                evidence_count=0,
                economic_priority=0.8,
                trust_priority=0.4,
                promotion_readiness=0.3,
            )
        ],
    )


def _write_graph(path: Path) -> Path:
    path.write_text(json.dumps(_make_test_graph().to_dict(), indent=2), encoding="utf-8")
    return path


def test_compile_sim_synth_physics_plan_writes_world_state_and_diffusion_bundle(tmp_path: Path) -> None:
    graph_path = _write_graph(tmp_path / "coverage_graph.json")

    result = compile_plan_main(
        [
            "--coverage-graph",
            str(graph_path),
            "--output-dir",
            str(tmp_path / "plan"),
            "--limit",
            "2",
        ]
    )

    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert Path(result["world_state_path"]).exists()
    assert Path(result["diffusion_plans_path"]).exists()
    assert summary["world_state_id"] == result["world_state_id"]
    assert summary["diffusion_plan_count"] >= 1


def test_run_sim_synth_physics_loop_writes_canonical_receipts(tmp_path: Path) -> None:
    graph_path = _write_graph(tmp_path / "coverage_graph.json")

    result = run_loop_main(
        [
            "--coverage-graph",
            str(graph_path),
            "--output-dir",
            str(tmp_path / "loop"),
            "--limit",
            "2",
        ]
    )

    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["world_state_id"] == result["world_state_id"]
    assert Path(result["artifact_paths"]["physics_execution_contract"]).exists()
    assert Path(result["artifact_paths"]["physics_adaptation_receipt"]).exists()
    assert Path(result["artifact_paths"]["backend_execution_binding_receipt"]).exists()
    assert Path(result["artifact_paths"]["robot_asset_contract_receipt"]).exists()
    assert Path(result["artifact_paths"]["backend_runtime_bridge_receipt"]).exists()
    assert Path(result["artifact_paths"]["backend_runtime_work_orders"]).exists()
    assert Path(result["artifact_paths"]["physics_calibration_receipt"]).exists()
    assert Path(result["artifact_paths"]["render_provider_receipts"]).exists()
    assert result["backend_execution_binding_receipt_id"]
    assert result["robot_asset_contract_receipt_id"]
    assert result["backend_runtime_bridge_receipt_id"]
    assert result["backend_runtime_bridge_status"] in {
        "runtime_bridge_ready",
        "runtime_targets_missing",
        "runtime_assets_missing",
        "shadow_bridge_only",
        "planning_only",
    }
    assert result["bridge_execution_authority"] in {
        "planning_only",
        "runtime_request_only",
        "shadow_runtime",
        "concrete_runtime",
    }
    assert result["backend_runtime_work_order_count"] >= 0
    if result["backend_runtime_work_order_count"]:
        assert result["backend_runtime_work_order_statuses"]
    assert result["robot_asset_readiness_score"] >= 0.0
    assert result["backend_runtime_execution_status"] in {
        "",
        "runtime_request_materialized_with_preconditions",
        "runtime_launch_prepared",
        "runtime_external_launch_completed",
        "runtime_external_launch_failed",
        "runtime_execution_completed",
        "runtime_training_completed",
        "runtime_execution_failed",
    }
    assert result["backend_runtime_launch_status"] in {
        "",
        "launch_blocked",
        "launch_prepared",
        "launch_completed",
        "launch_failed",
    }
    assert result["backend_runtime_outcome_status"] in {
        "",
        "launch_not_executed",
        "runtime_outputs_harvested",
        "runtime_outputs_missing",
        "outcome_sources_missing",
    }
    assert result["backend_runtime_output_count"] >= 0
    assert result["backend_shadow_execution_status"] in {
        "",
        "shadow_executed",
        "shadow_executed_with_asset_gaps",
        "shadow_work_order_materialized",
        "shadow_work_order_materialized_with_preconditions",
    }
    assert Path(result["artifact_paths"]["training_feedback_manifest"]).exists()
    assert result["render_provider_receipt_count"] >= 1
    assert result["materialized_render_provider_count"] >= 1
    assert result["outcome_receipt_count"] >= 1
