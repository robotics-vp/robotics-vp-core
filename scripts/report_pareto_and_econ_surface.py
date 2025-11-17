#!/usr/bin/env python3
"""
Generate a small advisory report of econ/Pareto surfaces from datapacks.

Read-only: does not change rewards or training behavior.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from src.orchestrator.economic_controller import EconomicController
from src.config.objective_profile import get_objective_presets
from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_validators import validate_datapack_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapack-dir", type=str, default="data/datapacks/phase_c")
    parser.add_argument("--env-name", type=str, default="drawer_vase")
    parser.add_argument("--out-json", type=str, default="results/pareto_econ_surface_report.json")
    args = parser.parse_args()

    repo = DataPackRepo(base_dir=args.datapack_dir)
    datapacks = [dp for dp in repo.iter_all(args.env_name) or []]
    warn_count = 0
    for dp in datapacks:
        warn_count += len(validate_datapack_meta(dp))
    if warn_count:
        print(f"[report] {warn_count} schema warnings detected; continuing with analysis.")

    econ = EconomicController.from_econ_params(econ_params=None, objective_profile=None)
    presets = get_objective_presets()

    report = []
    for preset_name, obj_vec in presets.items():
        frontiers = econ.compute_pareto_frontiers(datapacks)
        filtered = {k: econ.filter_frontier_by_constraints(v, {}) for k, v in frontiers.items()}
        scored = {k: econ.evaluate_frontier_for_spread(v) for k, v in filtered.items()}
        best = econ.select_frontier_optimum(scored.get("objective", []), obj_vec)
        report.append(
            {
                "preset": preset_name,
                "objective_vector": obj_vec.to_list() if hasattr(obj_vec, "to_list") else obj_vec,
                "frontier_points": {k: v for k, v in scored.items()},
                "best": best,
            }
        )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)

    for entry in report:
        best = entry.get("best")
        print(f"{entry['preset']}: best={best}")


if __name__ == "__main__":
    main()
