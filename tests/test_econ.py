"""
Unit tests for economic calculations.
"""
import pytest
from src.economics.mpl import mpl
from src.economics.wage import implied_robot_wage
from src.economics.reward import econ_lagrangian_reward


def test_mpl_basic():
    """Test MPL calculation."""
    # 60 units in 1 hour = 60/hr
    assert mpl(60, 1.0) == 60.0

    # 30 units in 0.5 hours = 60/hr
    assert mpl(30, 0.5) == 60.0

    # Zero time should return 0
    assert mpl(60, 0.0) == 0.0


def test_implied_wage():
    """Test robot wage calculation."""
    # Perfect execution: no errors
    wage = implied_robot_wage(
        price_per_unit=0.30,
        mp_r=60.0,
        error_rate=0.0,
        damage_cost=1.0
    )
    assert wage == 0.30 * 60.0  # $18/hr

    # With 5% errors
    wage = implied_robot_wage(
        price_per_unit=0.30,
        mp_r=60.0,
        error_rate=0.05,
        damage_cost=1.0
    )
    # Revenue: 0.30 * 60 = $18
    # Damage: 1.0 * (0.05 * 60) = $3
    # Net: $15
    assert abs(wage - 15.0) < 0.01


def test_lagrangian_reward():
    """Test Lagrangian reward function."""
    # No constraint violation (err < e*)
    reward = econ_lagrangian_reward(
        mp_r=60.0,
        err_rate=0.05,
        price_per_unit=0.30,
        damage_cost=1.0,
        lam=0.1,
        err_target=0.06,
        energy_cost_per_hour=0.0
    )

    # Profit: 0.30*60 - 1.0*(0.05*60) = 18 - 3 = 15
    # Penalty: 0 (err < target)
    # Reward = 15
    assert abs(reward - 15.0) < 0.01

    # With constraint violation (err > e*)
    reward = econ_lagrangian_reward(
        mp_r=60.0,
        err_rate=0.10,  # 10% > 6% target
        price_per_unit=0.30,
        damage_cost=1.0,
        lam=1.0,
        err_target=0.06,
        energy_cost_per_hour=0.0
    )

    # Profit: 0.30*60 - 1.0*(0.10*60) = 18 - 6 = 12
    # Penalty: 1.0 * (0.10 - 0.06) = 0.04
    # Reward = 12 - 0.04 = 11.96
    assert abs(reward - 11.96) < 0.01


def test_dual_ascent():
    """Test λ dual ascent update."""
    lam = 0.0
    eta = 0.1
    e_star = 0.06

    # Error below target: λ should decrease (but floor at 0)
    e = 0.04
    lam_new = max(0.0, lam + eta * (e - e_star))
    assert lam_new == 0.0  # Floored at 0

    # Error above target: λ should increase
    e = 0.08
    lam_new = max(0.0, lam + eta * (e - e_star))
    assert abs(lam_new - 0.002) < 1e-6  # 0.1 * 0.02, with tolerance

    # Multiple updates
    lam = 0.0
    for _ in range(10):
        e = 0.08  # Consistently violating
        lam = max(0.0, lam + eta * (e - e_star))

    assert lam > 0.01  # Should have grown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
