#!/usr/bin/env python3
"""
Offline econ surface solver over datapacks (no training).
"""
import argparse
import json
import os
import numpy as np
from itertools import product

from src.valuation.datapack_repo import DataPackRepo
from src.config.objective_profile import get_objective_presets


def estimate_metrics(datapacks):
    if not datapacks:
        return None
    mpl = np.mean([dp.attribution.delta_mpl for dp in datapacks])
    error = np.mean([dp.attribution.delta_error for dp in datapacks])
    ep = np.mean([dp.energy.total_Wh for dp in datapacks])
    wh_per_unit = np.mean([dp.energy.Wh_per_unit for dp in datapacks])
    return {"mpl": float(mpl), "error": float(error), "energy_Wh": float(ep), "wh_per_unit": float(wh_per_unit)}


def utility(obj_vec, metrics):
    return obj_vec[0] * metrics["mpl"] - obj_vec[1] * metrics["error"] - obj_vec[2] * metrics["energy_Wh"]


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
    if args.customer_segement:
        dps = [dp for dp in dps if getattr(dp.condition, "customer_segment", None) == args.customer_segement]

    presets = get_objective_presets()
    constraint_sets = [
        {"mpl_min": args.mpl_human, "energy_max": args.energy_budget, "error_max": args.error_max}
    ]
    results = []
    for preset_name, obj_vec in presets.items():
        for constraint in constraint_sets:
            metrics = estimate_metrics(dps)
            if not metrics:
                continue
            valid = True
            if constraint["mpl_min"] is not None and metrics["mpl"] < constraint["mpl_min"]:
                valid = False
            if constraint["energy_max"] is not None and metrics["energy_Wh"] > constraint["energy_max"]:
                valid = False
            if constraint["error_max"] is not None and metrics["error"] > constraint["error_max"]:
                valid = False
            u = utility(obj_vec, metrics) if valid else None
            results.append(
                {
                    "preset": preset_name,
                    "objective_vector": obj_vec,
                    "constraints": constraint,
                    "metrics": metrics,
                    "valid": valid,
                    "utility": u,
                }
            )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    for r in results:
        print(
            f"{r['preset']} valid={r['valid']} mpl={r['metrics']['mpl']:.3f} "
            f"err={r['metrics']['error']:.3f} wh={r['metrics']['energy_Wh']:.3f} util={r['utility']}"
        )


if __name__ == "__main__":
    main()
