#!/usr/bin/env python3
"""
Offline econ surface solver over datapacks (no training).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from src.valuation.datapack_repo import DataPackRepo
from src.config.objective_profile import get_objective_presets
from src.orchestrator.economic_controller import EconomicController
from src.valuation.datapack_validators import validate_datapack_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapack-dir", type=str, required=True)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--engine-type", type=str, default=None)
    parser.add_argument("--task-type", type=str, default=None)
    parser.add_argument("--customer-segment", type=str, default=None)
    parser.add_argument("--mpl-human", type=float, default=None)
    parser.add_argument("--energy-budget", type=float, default=None)
    parser.add_argument("--error-max", type=float, default=None)
    parser.add_argument("--out-json", type=str, default="results/objective_surface_summary.json")
    args = parser.parse_args()

    repo = DataPackRepo(base_dir=args.datapack_dir)
    dps = [dp for dp in repo.iter_all(args.env_name) or []]
    if args.engine_type:
        dps = [dp for dp in dps if dp.condition.engine_type == args.engine_type]
    if args.customer_segment:
        dps = [dp for dp in dps if getattr(dp.condition, "customer_segment", None) == args.customer_segment]

    total_warnings = 0
    for dp in dps:
        total_warnings += len(validate_datapack_meta(dp))
    if total_warnings:
        print(f"[validation] Found {total_warnings} schema warnings in datapacks; continuing with analysis.")

    presets = get_objective_presets()
    constraint_bundle = {
        "mpl_min_human": args.mpl_human,
        "energy_budget_Wh": args.energy_budget,
        "error_max": args.error_max,
    }

    results = []
    econ = EconomicController.from_econ_params(econ_params=None, objective_profile=None)
    for preset_name, obj_vec in presets.items():
        frontiers = econ.compute_pareto_frontiers(dps)
        filtered = {}
        for name, f in frontiers.items():
            filtered[name] = econ.filter_frontier_by_constraints(f, constraint_bundle)
        evaluated = {name: econ.evaluate_frontier_for_spread(f) for name, f in filtered.items()}
        best = econ.select_frontier_optimum(evaluated.get("objective", []), obj_vec)
        results.append(
            {
                "preset": preset_name,
                "objective_vector": obj_vec,
                "constraints": constraint_bundle,
                "frontiers": evaluated,
                "best": best,
            }
        )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    for r in results:
        best = r["best"]
        if best:
            print(f"{r['preset']} -> best utility point: mpl={best.get('mpl')} err={best.get('error')} wh={best.get('energy_Wh')}")


if __name__ == "__main__":
    main()
