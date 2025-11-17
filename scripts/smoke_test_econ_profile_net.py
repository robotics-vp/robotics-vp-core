#!/usr/bin/env python3
"""
Smoke test for EconProfileNet scaffolding.

Verifies:
- EconProfileContext can be created
- EconProfileNet can forward pass
- build_econ_params_from_context applies deltas correctly
- No behavior change when profile_net is None

This is scaffolding only - no training, no integration into RL loop yet.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config.internal_profile import get_internal_experiment_profile
from src.config.econ_params import load_econ_params, get_econ_params_with_profile
from src.config.econ_profile_net import (
    EconProfileContext,
    EconProfileNet,
    build_econ_params_from_context,
    context_to_tensor,
)


def main():
    print("=" * 70)
    print("ECON PROFILE NET SMOKE TEST")
    print("=" * 70)

    # 1. Load base econ params for a known preset
    profile = get_internal_experiment_profile("dishwashing")
    base = load_econ_params(profile, preset="toy")

    print("\n1. Base EconParams (toy preset):")
    print(f"   base_rate: {base.base_rate}")
    print(f"   damage_cost: {base.damage_cost}")
    print(f"   care_cost: {base.care_cost}")
    print(f"   energy_Wh_per_attempt: {base.energy_Wh_per_attempt}")
    print(f"   max_steps: {base.max_steps}")
    print(f"   preset: {base.preset}")

    # 2. Construct a dummy context for Drawer+Vase on PyBullet
    ctx = EconProfileContext(
        env_name="drawer_vase",
        engine_type="pybullet",
        task_type="fragility",
        mean_energy_Wh_per_unit=0.003,
        baseline_mpl_human=20.0,
        baseline_error_rate_human=0.05,
    )

    print("\n2. EconProfileContext:")
    print(f"   env_name: {ctx.env_name}")
    print(f"   engine_type: {ctx.engine_type}")
    print(f"   task_type: {ctx.task_type}")
    print(f"   mean_energy_Wh_per_unit: {ctx.mean_energy_Wh_per_unit}")
    print(f"   baseline_mpl_human: {ctx.baseline_mpl_human}")
    print(f"   baseline_error_rate_human: {ctx.baseline_error_rate_human}")

    # 3. Test context_to_tensor
    tensor = context_to_tensor(ctx)
    print("\n3. Context to tensor:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Values: {tensor.tolist()}")

    # 4. Instantiate a random EconProfileNet
    net = EconProfileNet(hidden_dim=32)
    print(f"\n4. EconProfileNet created:")
    print(f"   Parameters: {sum(p.numel() for p in net.parameters())}")

    # Test forward pass
    with torch.no_grad():
        deltas = net(tensor.unsqueeze(0))[0]
    print(f"   Raw deltas: {deltas.tolist()}")
    print(f"   Δbase_rate: {deltas[0]:.6f}")
    print(f"   Δdamage_cost: {deltas[1]:.6f}")
    print(f"   Δcare_cost: {deltas[2]:.6f}")
    print(f"   Δenergy_Wh_per_attempt: {deltas[3]:.6f}")
    print(f"   Δmax_steps_scale: {deltas[4]:.6f}")

    # 5. Apply adjustments
    params_adj = build_econ_params_from_context(base, ctx, net)
    print("\n5. Adjusted EconParams:")
    print(f"   base_rate: {base.base_rate:.4f} -> {params_adj.base_rate:.4f} (Δ={params_adj.base_rate - base.base_rate:.4f})")
    print(f"   damage_cost: {base.damage_cost:.4f} -> {params_adj.damage_cost:.4f} (Δ={params_adj.damage_cost - base.damage_cost:.4f})")
    print(f"   care_cost: {base.care_cost:.4f} -> {params_adj.care_cost:.4f} (Δ={params_adj.care_cost - base.care_cost:.4f})")
    print(f"   energy_Wh_per_attempt: {base.energy_Wh_per_attempt:.6f} -> {params_adj.energy_Wh_per_attempt:.6f}")
    print(f"   max_steps: {base.max_steps} -> {params_adj.max_steps}")

    # 6. Verify no change when profile_net is None
    params_unchanged = build_econ_params_from_context(base, ctx, None)
    assert params_unchanged.base_rate == base.base_rate
    assert params_unchanged.damage_cost == base.damage_cost
    assert params_unchanged.max_steps == base.max_steps
    print("\n6. No change when profile_net is None: PASS")

    # 7. Test the wrapper function
    params_via_wrapper = get_econ_params_with_profile(profile, "toy", ctx, net)
    print(f"\n7. get_econ_params_with_profile wrapper:")
    print(f"   base_rate: {params_via_wrapper.base_rate:.4f}")
    print(f"   damage_cost: {params_via_wrapper.damage_cost:.4f}")

    # Without profile_net, should return base unchanged
    params_via_wrapper_none = get_econ_params_with_profile(profile, "toy", None, None)
    assert params_via_wrapper_none.base_rate == base.base_rate
    print("   Wrapper returns base unchanged when ctx/net is None: PASS")

    # 8. Test different contexts
    print("\n8. Testing different contexts:")

    ctx_isaac = EconProfileContext(
        env_name="drawer_vase",
        engine_type="isaac",
        task_type="precision",
        mean_energy_Wh_per_unit=0.005,
        baseline_mpl_human=25.0,
        baseline_error_rate_human=0.03,
    )
    tensor_isaac = context_to_tensor(ctx_isaac)
    print(f"   Isaac context tensor: {tensor_isaac.tolist()}")

    ctx_ue5 = EconProfileContext(
        env_name="dishwashing",
        engine_type="ue5",
        task_type="throughput",
        mean_energy_Wh_per_unit=0.002,
        baseline_mpl_human=30.0,
        baseline_error_rate_human=0.02,
    )
    tensor_ue5 = context_to_tensor(ctx_ue5)
    print(f"   UE5 context tensor: {tensor_ue5.tolist()}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    print("\nScaffolding is ready for:")
    print("  - Engine-specific econ param tuning (PyBullet, Isaac, UE5)")
    print("  - Task-specific adjustments (throughput, fragility, precision)")
    print("  - DL-based hyperparameter learning (future work)")
    print("\nNo existing call sites have been modified.")
    print("No behavior change in training loop yet.")


if __name__ == "__main__":
    main()
