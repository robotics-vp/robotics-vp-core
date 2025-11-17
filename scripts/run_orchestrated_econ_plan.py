#!/usr/bin/env python3
"""
Run the orchestration transformer in advisory mode to propose an econ plan.

No RL/reward changes; this is planning/logging only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch

from src.orchestrator.context import build_orchestrator_context_from_datapacks
from src.orchestrator.orchestration_transformer import OrchestrationTransformer, propose_orchestrated_plan
from src.orchestrator.economic_controller import EconomicController
from scripts.run_energy_interventions import PROFILES, run_scripted_drawer_open
from src.envs.drawer_vase_arm_env import DrawerVaseArmEnv


def maybe_run_single_rollout(env_name: str, dominant_profile: str):
    if env_name != "drawer_vase_arm":
        return None
    env = DrawerVaseArmEnv()
    profile = PROFILES.get(dominant_profile, PROFILES["BASE"])
    info_hist = run_scripted_drawer_open(env, profile)
    last = info_hist[-1]
    return {
        "mpl": last.get("mpl_t", 0.0) if "mpl_t" in last else last.get("mpl_episode", 0.0),
        "error": float(last.get("errors", 0.0)),
        "energy_Wh": float(last.get("energy_Wh", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="drawer_vase_arm")
    parser.add_argument("--engine", type=str, default="pybullet")
    parser.add_argument("--task", type=str, default="fragility")
    parser.add_argument("--customer-segment", type=str, default="industrial_high_wage")
    parser.add_argument("--market-region", type=str, default="US_NE")
    parser.add_argument("--instruction", type=str, default="maximize throughput subject to moderate energy cost and low breakage")
    parser.add_argument("--datapacks-dir", type=str, default="data/datapacks")
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    parser.add_argument("--out-json", type=str, default="results/orchestrated_econ_plan_example.json")
    args = parser.parse_args()

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
    plan = propose_orchestrated_plan(model, ctx, args.instruction, steps=4)

    econ = EconomicController.from_econ_params(econ_params=None, objective_profile=None)
    constraint_bundle = {"mpl_min_human": None, "energy_budget_Wh": None, "error_max": None}
    frontiers = econ.compute_pareto_frontiers([])
    filtered = {name: econ.filter_frontier_by_constraints(f, constraint_bundle) for name, f in frontiers.items()}
    evaluated = {name: econ.evaluate_frontier_for_spread(f) for name, f in filtered.items()}
    best = econ.select_frontier_optimum(evaluated.get("objective", []), ctx.objective_vector)

    # Pick dominant energy profile to optionally test
    dominant_profile = max(plan.energy_profile_weights.items(), key=lambda kv: kv[1])[0]
    realized = maybe_run_single_rollout(args.env, dominant_profile)

    print("=== Orchestrated Econ Plan ===")
    print(f"Context: env={ctx.env_name}, engine={ctx.engine_type}, task={ctx.task_type}")
    print(f"Customer: segment={ctx.customer_segment}, region={ctx.market_region}")
    print(f"Objective vector: {ctx.objective_vector}")
    print("Plan steps:")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}) {step.tool_call.name} args={step.tool_call.args}")
    print("Expected deltas (from energy profile mix):")
    print(f"  ΔMPL ~ {plan.expected_delta_mpl:.2f}, Δerror ~ {plan.expected_delta_error:.3f}, Δenergy_Wh ~ {plan.expected_delta_energy_Wh:.4f}")
    if realized:
        print("Realized single rollout (dominant profile):", realized)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        "context": ctx.__dict__,
        "plan_steps": [ {"name": s.tool_call.name, "args": s.tool_call.args} for s in plan.steps ],
        "expected": {
            "delta_mpl": plan.expected_delta_mpl,
            "delta_error": plan.expected_delta_error,
            "delta_energy_Wh": plan.expected_delta_energy_Wh,
        },
        "energy_profile_weights": plan.energy_profile_weights,
        "data_mix_weights": plan.data_mix_weights,
        "objective_preset": plan.objective_preset,
        "chosen_backend": plan.chosen_backend,
        "pareto": {
            "best": best,
            "frontiers": evaluated,
        },
        "realized": realized,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote plan summary to {args.out_json}")


if __name__ == "__main__":
    main()
