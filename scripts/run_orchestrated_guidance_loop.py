#!/usr/bin/env python3
"""
Orchestrated guidance loop (single iteration).

Advisory-only: annotates datapacks with guidance profiles and emits diffusion requests.
"""
import argparse
import json
import os
import uuid

from src.orchestrator.context import build_orchestrator_context_from_datapacks
from src.orchestrator.orchestration_transformer import OrchestrationTransformer, propose_orchestrated_plan
from src.orchestrator.guidance import annotate_datapacks_with_guidance
from src.orchestrator.diffusion_requests import build_diffusion_prompt_from_guidance
from src.valuation.datapack_repo import DataPackRepo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapacks-dir", type=str, default="data/datapacks")
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    parser.add_argument("--env", type=str, default="drawer_vase_arm")
    parser.add_argument("--engine", type=str, default="pybullet")
    parser.add_argument("--task", type=str, default="fragility")
    parser.add_argument("--customer-segment", type=str, default="industrial_high_wage")
    parser.add_argument("--market-region", type=str, default="US_NE")
    parser.add_argument("--instruction", type=str, default="minimize cost subject to MPL >= human and low breakage")
    parser.add_argument("--out-guidance", type=str, default="data/datapacks/guidance_overlays.jsonl")
    parser.add_argument("--out-diffusion", type=str, default="data/diffusion_requests.jsonl")
    args = parser.parse_args()

    repo = DataPackRepo(base_dir=args.datapacks_dir)
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
    plan_id = str(uuid.uuid4())

    annotated = annotate_datapacks_with_guidance(repo, ctx, plan_id, plan.steps, max_packs=80)

    # Build diffusion prompts from guidance
    prompts = []
    for dp in annotated:
        if not dp.guidance_profile:
            continue
        prompts.append(build_diffusion_prompt_from_guidance(dp, dp.guidance_profile))

    repo.write_guidance_overlays(annotated, args.out_guidance)

    if prompts:
        os.makedirs(os.path.dirname(args.out_diffusion), exist_ok=True)
        with open(args.out_diffusion, "w") as f:
            for p in prompts:
                f.write(json.dumps(p.to_dict()) + "\n")

    # Console summary
    print("Orchestrated Guidance Loop:")
    print(f"  Context: {ctx.env_name} / {ctx.engine_type} / {ctx.task_type}")
    print(f"  Customer: {ctx.customer_segment} / {ctx.market_region}")
    print(f"  Objective: {ctx.objective_vector}")
    print("  Plan steps:")
    for i, s in enumerate(plan.steps, 1):
        print(f"    {i}) {s.tool_call.name} args={s.tool_call.args}")
    print(f"  Annotated datapacks: {len(annotated)}")
    print(f"  Diffusion requests: {len(prompts)}")
    if prompts:
        pos = len([p for p in prompts if p.difficulty_hint != 'hard_neg'])
        neg = len(prompts) - pos
        print(f"    Breakdown: positive-like {pos}, hard_neg {neg}")

    summary = {
        "plan_id": plan_id,
        "context": ctx.__dict__,
        "plan": {
            "steps": [{"name": s.tool_call.name, "args": s.tool_call.args} for s in plan.steps],
            "energy_profile_weights": plan.energy_profile_weights,
            "data_mix_weights": plan.data_mix_weights,
            "objective_preset": plan.objective_preset,
            "chosen_backend": plan.chosen_backend,
            "expected": {
                "delta_mpl": plan.expected_delta_mpl,
                "delta_error": plan.expected_delta_error,
                "delta_energy_Wh": plan.expected_delta_energy_Wh,
            },
        },
        "annotated_count": len(annotated),
        "diffusion_requests": len(prompts),
    }
    os.makedirs(os.path.dirname(args.out_guidance), exist_ok=True)
    with open("results/orchestrated_guidance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
