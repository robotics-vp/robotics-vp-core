#!/usr/bin/env python3
"""
Preview orchestrated run specs (advisory only).
"""
import argparse
import json
import os

from src.orchestrator.context import build_orchestrator_context_from_datapacks
from src.orchestrator.orchestration_transformer import OrchestrationTransformer, propose_orchestrated_plan
from src.orchestrator.experiment_config import orchestration_plan_to_run_specs
from src.robot.backend import RobotRunSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", type=str, default="results/orchestrated_econ_plan_example.json")
    parser.add_argument("--datapacks-dir", type=str, default="data/datapacks")
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    parser.add_argument("--env", type=str, default="drawer_vase_arm")
    parser.add_argument("--engine", type=str, default="pybullet")
    parser.add_argument("--task", type=str, default="fragility")
    parser.add_argument("--customer-segment", type=str, default="industrial_high_wage")
    parser.add_argument("--market-region", type=str, default="US_NE")
    parser.add_argument("--instruction", type=str, default="maximize throughput subject to moderate energy cost and low breakage")
    parser.add_argument("--out-json", type=str, default="results/orchestrated_run_specs.json")
    args = parser.parse_args()

    if os.path.exists(args.plan_json):
        with open(args.plan_json, "r") as f:
            saved = json.load(f)
        # Build minimal ctx from saved
        ctx_data = saved.get("context", {})
        # fallback: reconstruct context
        ctx = build_orchestrator_context_from_datapacks(
            base_dir=args.datapacks_dir,
            env_name=ctx_data.get("env_name", args.env),
            engine_type=ctx_data.get("engine_type", args.engine),
            task_type=ctx_data.get("task_type", args.task),
            customer_segment=ctx_data.get("customer_segment", args.customer_segment),
            market_region=ctx_data.get("market_region", args.market_region),
            interventions_path=args.interventions if os.path.exists(args.interventions) else None,
        )
        # Use saved plan if present
        plan_steps = saved.get("plan_steps", [])
        # Build a dummy OrchestratorResult-like object
        from types import SimpleNamespace
        result = SimpleNamespace(
            steps=plan_steps,
            chosen_backend=saved.get("chosen_backend", args.engine),
            energy_profile_weights=saved.get("energy_profile_weights", saved.get("plan", {}).get("energy_profile_weights", {})),
            objective_preset=saved.get("objective_preset", saved.get("plan", {}).get("objective_preset", "balanced")),
            data_mix_weights=saved.get("data_mix_weights", saved.get("plan", {}).get("data_mix_weights", {})),
        )
    else:
        ctx = build_orchestrator_context_from_datapacks(
            base_dir=args.datapacks_dir,
            env_name=args.env,
            engine_type=args.engine,
            task_type=args.task,
            customer_segment=args.customer_segment,
            market_region=args.market_region,
            interventions_path=args.interventions if os.path.exists(args.interventions) else None,
        )
        model = OrchestrationTransformer()
        result = propose_orchestrated_plan(model, ctx, args.instruction, steps=4)

    run_specs = orchestration_plan_to_run_specs(result, ctx)

    robot_specs = [
        RobotRunSpec(
            run_id=rs.run_id + "-robot",
            env_name=rs.env_name,
            engine_type=rs.engine_type,
            skill_sequence=[],
            objective_profile={"preset": rs.objective_preset},
            energy_profile_mix=rs.energy_profile_mix,
            notes="Robot run stub derived from orchestrated plan",
        )
        for rs in run_specs
    ]

    print("=== Orchestrated Run Specs ===")
    for rs in run_specs:
        print(json.dumps(rs.to_dict(), indent=2))
    print("=== Robot Run Specs (stub) ===")
    for rrs in robot_specs:
        print(json.dumps(rrs.to_dict(), indent=2))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "policy_runs": [rs.to_dict() for rs in run_specs],
                "robot_runs": [rrs.to_dict() for rrs in robot_specs],
            },
            f,
            indent=2,
        )
    print(f"Wrote run specs to {args.out_json}")


if __name__ == "__main__":
    main()
