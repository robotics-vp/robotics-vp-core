#!/usr/bin/env python3
"""Compile canonical sim/synth/physics WM planning artifacts."""

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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile sim/synth/physics WM planning artifacts")
    parser.add_argument("--coverage-graph", required=True, help="Path to coverage_graph.json")
    parser.add_argument(
        "--output-dir",
        default="artifacts/sim_synth_physics_plan",
        help="Directory to write compiled planning artifacts",
    )
    parser.add_argument("--semantic-context", help="Optional JSON mapping with semantic context")
    parser.add_argument("--economic-context", help="Optional JSON mapping with economic context")
    parser.add_argument("--embodiment-context", help="Optional JSON mapping with embodiment context")
    parser.add_argument("--benchmark-signals", help="Optional JSON mapping with benchmark signals")
    parser.add_argument("--backend-selector-package", help="Optional backend selector package path")
    parser.add_argument("--branch-planner-package", help="Optional branch planner package path")
    parser.add_argument("--default-backend", default="pybullet")
    parser.add_argument("--default-objective", default="balanced")
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
        )
    )
    world_state, diffusion_plans = runtime.compile_world_state_and_diffusion_plans(
        graph,
        semantic_context=_load_mapping(args.semantic_context),
        economic_context=_load_mapping(args.economic_context),
        embodiment_context=_load_mapping(args.embodiment_context),
        benchmark_signals=_load_mapping(args.benchmark_signals),
        backend_selector=args.backend_selector_package,
        branch_planner=args.branch_planner_package,
        limit=int(args.limit),
    )
    output_dir = Path(args.output_dir)
    world_state_path = output_dir / "sim_synth_physics_world_state.json"
    diffusion_plans_path = output_dir / "gap_driven_diffusion_plans.json"
    summary_path = output_dir / "sim_synth_physics_plan_summary.json"
    _write_json(world_state_path, world_state.to_dict())
    _write_json(
        diffusion_plans_path,
        {
            "version": "gap_driven_diffusion_plan_bundle_v1",
            "plans": [plan.to_dict() for plan in diffusion_plans],
        },
    )
    summary = {
        "version": "sim_synth_physics_plan_summary_v1",
        "world_state_id": world_state.state_id,
        "agenda_id": world_state.simulation_agenda.agenda_id,
        "job_count": len(world_state.simulation_agenda.jobs),
        "branch_plan_count": len(world_state.synthetic_branch_plans),
        "admissible_branch_count": len(
            getattr(world_state.gen2sim_admission, "admissible_branch_ids", []) or []
        ),
        "blocked_branch_count": len(
            getattr(world_state.gen2sim_admission, "blocked_branch_ids", []) or []
        ),
        "diffusion_plan_count": len(diffusion_plans),
        "world_state_path": str(world_state_path.resolve()),
        "diffusion_plans_path": str(diffusion_plans_path.resolve()),
    }
    _write_json(summary_path, summary)
    return {
        "world_state_path": str(world_state_path.resolve()),
        "diffusion_plans_path": str(diffusion_plans_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "world_state_id": world_state.state_id,
    }


if __name__ == "__main__":
    main()
