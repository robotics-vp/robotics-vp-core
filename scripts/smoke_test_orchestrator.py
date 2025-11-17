#!/usr/bin/env python3
"""
Smoke test for the orchestration transformer scaffold.
"""
import torch

from src.orchestrator.orchestration_transformer import OrchestrationTransformer, decode_tool, _encode_ctx, _hash_tokens
from src.orchestrator.context import OrchestratorContext


def run_case(instr: str, objective_vec):
    model = OrchestrationTransformer()
    tokens = _hash_tokens(instr, model.instr_embed.num_embeddings)
    ctx = OrchestratorContext(
        env_name="drawer_vase_arm",
        engine_type="pybullet",
        task_type="fragility",
        customer_segment="industrial_high_wage",
        market_region="US_NE",
        objective_vector=objective_vec,
        wage_human=25.0,
        energy_price_kWh=0.12,
        mean_delta_mpl=1.0,
        mean_delta_error=0.1,
        mean_delta_j=0.5,
        mean_trust=0.8,
        mean_w_econ=0.9,
        profile_summaries={
            "BASE": {"mpl": 5.0, "error": 0.2, "energy_Wh": 0.1},
            "BOOST": {"mpl": 6.0, "error": 0.25, "energy_Wh": 0.15},
            "SAVER": {"mpl": 4.0, "error": 0.22, "energy_Wh": 0.08},
            "SAFE": {"mpl": 3.5, "error": 0.15, "energy_Wh": 0.09},
        },
    )
    ctx_vec = torch.from_numpy(_encode_ctx(ctx)).unsqueeze(0)
    logits, arg_vec = model(tokens, ctx_vec)
    tool = decode_tool(logits)
    print(f"Instruction: {instr}")
    print(f"Objective vec: {objective_vec}")
    print(f"Chosen tool: {tool}")
    print(f"Logits: {logits.detach().numpy()}")
    print(f"Arg vec: {arg_vec.detach().numpy()}")
    print("---")


def main():
    run_case("open drawer without hitting vase", [1.0, 0.5, -0.2, 0.2, 0.0])  # throughput-ish
    run_case("minimize energy usage", [0.2, 0.0, -1.0, 0.1, 0.0])            # energy saver
    run_case("generate novel trajectories", [0.3, 0.0, -0.1, 0.2, 1.0])      # novelty


if __name__ == "__main__":
    main()
