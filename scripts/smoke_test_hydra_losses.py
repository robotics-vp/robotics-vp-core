#!/usr/bin/env python3
"""
Smoke test to ensure Hydra loss scaffolding compiles and is compatible with
HydraActor/HydraCritic routing primitives.
"""
from pathlib import Path
import sys
from typing import Dict

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observation.condition_vector import ConditionVector
from src.rl.hydra_heads import HydraActor
from src.rl.hydra_losses import (
    PerHeadActorLoss,
    PerMetricCriticLoss,
    compute_hydra_losses,
    compute_gradient_stats,
)


class _DummyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, obs, condition=None):
        # Return base and conditioned features to exercise _unpack_trunk_output
        base = self.proj(torch.as_tensor(obs, dtype=torch.float32))
        return base, base * 0.0


class _DummyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(4, 2)

    def forward(self, trunk_features, condition=None, conditioned_features=None):
        return self.out(trunk_features)


def _make_condition(skill_mode: str) -> ConditionVector:
    return ConditionVector(
        task_id="smoke",
        env_id="sim",
        backend_id="stub",
        target_mpl=1.0,
        current_wage_parity=1.0,
        energy_budget_wh=1.0,
        skill_mode=skill_mode,
        ood_risk_level=0.1,
        recovery_priority=0.0,
        novelty_tier=0,
        sima2_trust_score=0.5,
        recap_goodness_bucket="bronze",
        objective_preset="balanced",
    )


def main():
    trunk = _DummyTrunk()
    heads: Dict[str, nn.Module] = {
        "manipulation": _DummyHead(),
        "navigation": _DummyHead(),
    }
    actor = HydraActor(trunk=trunk, heads=heads, default_skill_mode="manipulation")
    condition = _make_condition("manipulation_grasp")
    actor_out = actor(torch.zeros(1, 4), condition)
    assert actor_out.shape[-1] == 2, "HydraActor should emit dummy logits"

    actor_losses = {
        "manipulation": PerHeadActorLoss("manipulation", ["manipulation"]),
        "navigation": PerHeadActorLoss("navigation", ["navigation"]),
    }
    critic_losses = {
        "mpl": PerMetricCriticLoss("mpl", "mpl"),
        "damage": PerMetricCriticLoss("damage", "damage"),
    }
    loss_result = compute_hydra_losses(
        actor_losses,
        critic_losses,
        head_predictions={"manipulation": {"logits": actor_out}},
        head_targets={"manipulation": {"actions": torch.ones(1, 2)}},
        value_predictions={"mpl": {"value": 0.0}, "damage": {"value": 0.0}},
        value_targets={"mpl": {"target": 0.0}, "damage": {"target": 0.0}},
        skill_context=condition.skill_mode,
    ).to_dict()

    # Ensure deterministic keys and JSON-safe values exist
    assert set(loss_result["actor"].keys()) == {"manipulation", "navigation"}
    assert "isolation" in loss_result
    assert loss_result["metadata"]["skill_context"] == condition.skill_mode
    assert compute_gradient_stats({})["grad_norm"] == 0.0

    print("[smoke_test_hydra_losses] PASS")


if __name__ == "__main__":
    main()
