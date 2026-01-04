"""Unit tests for economic objective compiler."""
from src.objectives.economic_objective import EconomicObjectiveSpec, compile_economic_overlay
from src.objectives.loader import load_objective_spec


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


def test_load_objective_spec(tmp_path):
    config_path = tmp_path / "objective.yaml"
    config_path.write_text(
        "mpl_weight: 2.0\n"
        "energy_weight: 0.5\n"
        "error_weight: 1.5\n"
        "novelty_weight: 0.3\n"
        "risk_weight: 4.0\n"
        "extra_weights:\n"
        "  time_to_completion: -0.25\n"
    )
    obj = load_objective_spec(config_path)
    assert obj.mpl_weight == 2.0
    assert obj.energy_weight == 0.5
    assert obj.error_weight == 1.5
    assert obj.novelty_weight == 0.3
    assert obj.risk_weight == 4.0
    assert obj.extra_weights == {"time_to_completion": -0.25}
