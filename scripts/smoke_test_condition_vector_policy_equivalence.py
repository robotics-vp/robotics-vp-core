#!/usr/bin/env python3
"""
Smoke test for policy conditioning equivalence.

Validates that enabling the policy conditioning flag with zero-initialized
weights produces outputs identical to the condition-disabled path.
"""
import torch
from types import SimpleNamespace

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.rl.trunk_net import TrunkNet


def _build_stub_obs():
    return SimpleNamespace(
        latent=[0.1, 0.2, 0.3, 0.4],
        state_summary={"speed": 0.5, "errors": 1},
    )


def _build_condition():
    builder = ConditionVectorBuilder()
    return builder.build(
        episode_config={"task_id": "cv_policy_smoke", "env_id": "stub_env", "backend": "stub_backend", "objective_preset": "balanced"},
        econ_state={"target_mpl": 1.0, "current_wage_parity": 1.0, "energy_budget_wh": 0.0},
        curriculum_phase="warmup",
        sima2_trust=None,
        datapack_metadata={"tags": ["smoke_test"]},
        episode_step=0,
        episode_metadata={"episode_id": "ep_smoke"},
    )


def main():
    torch.manual_seed(0)
    obs = _build_stub_obs()
    condition = _build_condition()

    condition_dim = max(len(condition.to_vector()), 16)
    hidden_dim = 16

    # Baseline trunk: conditioning disabled
    torch.manual_seed(0)
    trunk_disabled = TrunkNet(
        vision_dim=4,
        state_dim=2,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        use_condition_film=False,
        use_condition_vector=False,
        use_condition_vector_for_policy=False,
    )

    # Policy-conditioning trunk: same weights + zero-initialized FiLM block
    torch.manual_seed(0)
    trunk_enabled = TrunkNet(
        vision_dim=4,
        state_dim=2,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        use_condition_film=False,
        use_condition_vector=False,
        use_condition_vector_for_policy=True,
        condition_fusion_mode="film",
        condition_film_hidden_dim=hidden_dim,
        condition_context_dim=hidden_dim,
    )
    base_state = trunk_disabled.state_dict()
    enabled_state = trunk_enabled.state_dict()
    for key, value in base_state.items():
        if key in enabled_state and enabled_state[key].shape == value.shape:
            enabled_state[key] = value.clone()
    trunk_enabled.load_state_dict(enabled_state, strict=False)

    with torch.no_grad():
        baseline = trunk_disabled(obs, use_condition=False)
        conditioned = trunk_enabled.condition_policy_features(baseline, condition)
        enabled_out = trunk_enabled(obs, use_condition=False)

    assert torch.allclose(baseline, enabled_out, atol=1e-7), "Base trunk output changed when enabling policy conditioning."
    assert conditioned is not None, "Conditioned features should be returned when policy flag is on."
    assert torch.allclose(baseline, conditioned, atol=1e-7), "Zero-initialized conditioning should not change outputs."

    print("[PASS] Policy conditioning equivalence holds for disabled vs zero-initialized weights.")
    print(f"[SUMMARY] baseline_norm={baseline.abs().sum().item():.6f} conditioned_norm={conditioned.abs().sum().item():.6f}")


if __name__ == "__main__":
    main()
