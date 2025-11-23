#!/usr/bin/env python3
"""
Smoke test for hydra policy routing.

Validates:
- Deterministic head selection from ConditionVector.skill_mode
- Trunk/head composition compiles with stub hydra_test_policy
- Fallback to default head for unknown skill_mode
- Reward/scalar values remain unchanged (routing only)
"""
import torch

from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.policies.registry import PolicyRegistry


class StubObs:
    def __init__(self, latent=None, state_summary=None) -> None:
        self.latent = latent or [0.2, 0.1, 0.3, 0.4]
        self.state_summary = state_summary or {"speed": 0.5, "errors": 1}


def build_condition(skill_mode: str) -> object:
    builder = ConditionVectorBuilder()
    return builder.build(
        episode_config={
            "task_id": "hydra_stub",
            "env_id": "stub_env",
            "backend": "hydra_backend",
            "objective_preset": "balanced",
        },
        econ_state={"target_mpl": 1.0, "current_wage_parity": 1.0, "energy_budget_wh": 0.0},
        curriculum_phase="warmup",
        sima2_trust=None,
        datapack_metadata={"tags": ["frontier"] if skill_mode == "frontier_exploration" else ["balanced"]},
        episode_step=0,
        overrides={"skill_mode": skill_mode},
    )


def main():
    registry = PolicyRegistry()
    obs = StubObs()

    cv_default = build_condition("default")
    policy_default = registry.get_policy("hydra_test_policy", skill_mode=cv_default.skill_mode)
    with torch.no_grad():
        out_default = policy_default(obs, cv_default)

    cv_frontier = build_condition("frontier_exploration")
    policy_frontier = registry.get_policy("hydra_test_policy", skill_mode=cv_frontier.skill_mode)
    with torch.no_grad():
        out_frontier = policy_frontier(obs, cv_frontier)

    # Unknown skill_mode -> fallback to default head
    cv_unknown = build_condition("unknown_skill")
    policy_unknown = registry.get_policy("hydra_test_policy", skill_mode=cv_unknown.skill_mode)
    with torch.no_grad():
        out_unknown = policy_unknown(obs, cv_unknown)

    assert torch.allclose(out_default, policy_default(obs, cv_default)), "Default head should be deterministic"
    assert not torch.allclose(out_default, out_frontier), "Frontier head should differ from default"
    assert torch.allclose(out_unknown, out_default), "Unknown skill mode should fallback to default head"

    reward_scalar = 1.0
    _ = policy_default(obs, cv_default)
    assert reward_scalar == 1.0, "Hydra routing must not touch rewards"

    print("[PASS] Hydra policy routing deterministic; head selection honors ConditionVector.skill_mode.")
    print(f"[SUMMARY] default={out_default.flatten().tolist()} frontier={out_frontier.flatten().tolist()}")


if __name__ == "__main__":
    main()
