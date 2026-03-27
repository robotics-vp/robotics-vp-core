#!/usr/bin/env python3
"""Run the sim/synth/physics WM planning loop and emit canonical receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.world_model.semantic_coverage_graph import SemanticCoverageGraph
from src.world_model.sim_synth_physics import (
    SimSynthPhysicsRuntime,
    SimSynthPhysicsRuntimeConfig,
)


def _load_graph(path: str | Path) -> SemanticCoverageGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SemanticCoverageGraph.from_dict(payload)


def _load_mapping(path: str | None) -> Optional[Mapping[str, Any]]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected mapping payload in {path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sim/synth/physics WM planning loop")
    parser.add_argument("--coverage-graph", required=True, help="Path to coverage_graph.json")
    parser.add_argument(
        "--output-dir",
        default="artifacts/sim_synth_physics_loop",
        help="Directory to write WM loop artifacts",
    )
    parser.add_argument("--semantic-context", help="Optional JSON mapping with semantic context")
    parser.add_argument("--economic-context", help="Optional JSON mapping with economic context")
    parser.add_argument("--embodiment-context", help="Optional JSON mapping with embodiment context")
    parser.add_argument("--benchmark-signals", help="Optional JSON mapping with benchmark signals")
    parser.add_argument("--backend-selector-package", help="Optional backend selector package path")
    parser.add_argument("--branch-planner-package", help="Optional branch planner package path")
    parser.add_argument("--default-backend", default="pybullet")
    parser.add_argument("--default-objective", default="balanced")
    parser.add_argument("--fallback-backend", default="pybullet")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--economic-weight", type=float, default=1.0)
    parser.add_argument("--trust-weight", type=float, default=1.0)
    parser.add_argument("--readiness-weight", type=float, default=1.0)
    parser.add_argument("--gap-ranker-mode", choices=("disabled", "auto", "required"), default="auto")
    parser.add_argument("--backend-selector-mode", choices=("disabled", "auto", "required"), default="auto")
    parser.add_argument("--branch-planner-mode", choices=("disabled", "auto", "required"), default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    graph = _load_graph(args.coverage_graph)
    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(
            economic_weight=float(args.economic_weight),
            trust_weight=float(args.trust_weight),
            readiness_weight=float(args.readiness_weight),
            agenda_limit=int(args.limit),
            default_backend=str(args.default_backend),
            default_objective=str(args.default_objective),
            gap_ranker_mode=str(args.gap_ranker_mode),
            backend_selector_mode=str(args.backend_selector_mode),
            branch_planner_mode=str(args.branch_planner_mode),
            fallback_backend=str(args.fallback_backend),
        )
    )
    result = runtime.run_planning_window(
        graph,
        semantic_context=_load_mapping(args.semantic_context),
        economic_context=_load_mapping(args.economic_context),
        embodiment_context=_load_mapping(args.embodiment_context),
        benchmark_signals=_load_mapping(args.benchmark_signals),
        backend_selector=args.backend_selector_package,
        branch_planner=args.branch_planner_package,
        output_dir=args.output_dir,
    )
    summary_path = Path(args.output_dir) / "sim_synth_physics_loop_summary.json"
    return {
        "world_state_id": result.world_state.state_id,
        "physics_execution_contract_id": result.physics_execution_contract.contract_id,
        "physics_adaptation_receipt_id": result.physics_adaptation_receipt.receipt_id,
        "backend_execution_binding_receipt_id": result.backend_execution_binding_receipt.receipt_id,
        "robot_asset_contract_receipt_id": result.robot_asset_contract_receipt.receipt_id,
        "robot_asset_readiness_score": float(result.robot_asset_contract_receipt.readiness_score),
        "backend_shadow_execution_receipt_id": (
            None
            if result.backend_shadow_execution_receipt is None
            else result.backend_shadow_execution_receipt.receipt_id
        ),
        "backend_shadow_execution_status": (
            ""
            if result.backend_shadow_execution_receipt is None
            else result.backend_shadow_execution_receipt.execution_status
        ),
        "physics_calibration_receipt_id": result.physics_calibration_receipt.receipt_id,
        "render_provider_receipt_count": len(result.render_provider_receipts),
        "materialized_render_provider_count": sum(
            1
            for receipt in result.render_provider_receipts
            if str(receipt.materialization_status)
            not in {"", "planned_only", "materialization_blocked"}
        ),
        "outcome_receipt_count": len(result.outcome_receipts),
        "summary_path": str(summary_path.resolve()),
        "artifact_paths": dict(result.artifact_paths),
    }


if __name__ == "__main__":
    main()
