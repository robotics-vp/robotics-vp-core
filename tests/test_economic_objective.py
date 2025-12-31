"""Unit tests for economic objective compiler."""
from src.objectives.economic_objective import EconomicObjectiveSpec, compile_economic_overlay


def test_compile_overlay_signs():
    obj = EconomicObjectiveSpec(
        mpl_weight=1.5,
        energy_weight=0.2,
        error_weight=0.3,
        novelty_weight=0.4,
        risk_weight=0.7,
    )
    overlay = compile_economic_overlay(obj)
    assert overlay.reward_scales["mpl_per_hour"] == 1.5
    assert overlay.reward_scales["energy_kwh"] == -0.2
    assert overlay.reward_scales["error_rate"] == -0.3
    assert overlay.reward_scales["novelty_score"] == 0.4
    assert overlay.reward_scales["risk_score"] == -0.7


def test_compile_overlay_extra_weights():
    obj = EconomicObjectiveSpec(extra_weights={"custom_signal": 2.5})
    overlay = compile_economic_overlay(obj)
    assert overlay.reward_scales["custom_signal"] == 2.5
