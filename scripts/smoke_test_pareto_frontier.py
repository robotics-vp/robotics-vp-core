#!/usr/bin/env python3
"""
Deterministic Pareto frontier smoke test.
"""
from src.orchestrator.economic_controller import EconomicController
from types import SimpleNamespace


def make_dp(mpl, energy, error, name):
    dp = SimpleNamespace()
    dp.attribution = SimpleNamespace(delta_mpl=mpl, delta_error=error, delta_ep=0.0, rebate_pct=0.0, attributable_spread_capture=0.0, data_premium=0.0)
    dp.energy = SimpleNamespace(total_Wh=energy, Wh_per_unit=energy)
    dp.condition = SimpleNamespace(econ_preset=name)
    return dp


def main():
    dps = [
        make_dp(10, 5, 0.1, "A"),
        make_dp(9, 3, 0.2, "B"),
        make_dp(8, 2, 0.05, "C"),
        make_dp(11, 6, 0.3, "D"),
    ]
    econ = EconomicController.from_econ_params(econ_params=None, objective_profile=None)
    frontiers = econ.compute_pareto_frontiers(dps)
    constraints = {"mpl_min_human": 8.5, "energy_budget_Wh": 5.5, "error_max": 0.25}
    filtered = {name: econ.filter_frontier_by_constraints(f, constraints) for name, f in frontiers.items()}
    evaluated = {name: econ.evaluate_frontier_for_spread(f) for name, f in filtered.items()}
    best = econ.select_frontier_optimum(evaluated.get("objective", []), [1.0, 0.2, 0.1])

    print("Frontiers:", frontiers)
    print("Filtered:", filtered)
    print("Best point:", best)


if __name__ == "__main__":
    main()
