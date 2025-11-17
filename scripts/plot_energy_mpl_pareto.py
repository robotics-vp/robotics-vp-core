#!/usr/bin/env python3
"""
Compute and print the Pareto frontier for MPL vs energy using energy interventions.
"""
import argparse
import json
import os
import numpy as np


def dominates(a, b):
    # a dominates b if energy <= and mpl >= with at least one strict
    return (a["energy"] <= b["energy"] and a["mpl"] >= b["mpl"]) and (
        a["energy"] < b["energy"] or a["mpl"] > b["mpl"]
    )


def load_interventions(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            summ = rec.get("summary", {})
            rows.append(
                {
                    "env": rec.get("env", "unknown"),
                    "profile": rec.get("profile", "BASE"),
                    "mpl": summ.get("mpl_episode", summ.get("mpl_t", 0.0)),
                    "error": summ.get("error_rate_episode", 0.0),
                    "energy": summ.get("energy_Wh", 0.0),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    parser.add_argument("--env", type=str, default="drawer_vase_arm")
    parser.add_argument("--out-json", type=str, default="results/energy_mpl_pareto_summary.json")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows = [r for r in load_interventions(args.interventions) if r["env"] == args.env]
    if not rows:
        print("No rows for env", args.env)
        return

    profiles = {}
    for r in rows:
        profiles.setdefault(r["profile"], {"mpl": [], "error": [], "energy": []})
        profiles[r["profile"]]["mpl"].append(r["mpl"])
        profiles[r["profile"]]["error"].append(r["error"])
        profiles[r["profile"]]["energy"].append(r["energy"])

    summary = {}
    for prof, vals in profiles.items():
        summary[prof] = {
            "mpl_mean": float(np.mean(vals["mpl"])),
            "mpl_std": float(np.std(vals["mpl"])),
            "error_mean": float(np.mean(vals["error"])),
            "energy_mean": float(np.mean(vals["energy"])),
            "energy_std": float(np.std(vals["energy"])),
        }

    # Determine Pareto frontier
    items = [
        {"profile": prof, "mpl": v["mpl_mean"], "energy": v["energy_mean"]}
        for prof, v in summary.items()
    ]
    pareto = []
    for a in items:
        if not any(dominates(b, a) for b in items if b is not a):
            pareto.append(a["profile"])

    for prof in summary:
        summary[prof]["pareto_frontier"] = prof in pareto

    print("Profile\tMPL_mean\tErr_mean\tEnergy_Wh_mean\tPareto")
    for prof, vals in summary.items():
        print(
            f"{prof}\t{vals['mpl_mean']:.2f}\t{vals['error_mean']:.3f}\t{vals['energy_mean']:.4f}\t{vals['pareto_frontier']}"
        )

    out = {"env": args.env, "summary": summary, "pareto_profiles": pareto}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    if args.plot:
        import matplotlib.pyplot as plt

        colors = {"BASE": "blue", "BOOST": "red", "SAVER": "green", "SAFE": "orange"}
        for prof, vals in summary.items():
            plt.scatter(vals["energy_mean"], vals["mpl_mean"], c=colors.get(prof, "gray"), label=prof)
        plt.xlabel("Energy (Wh)")
        plt.ylabel("MPL (units/hr)")
        plt.title(f"Energy vs MPL Pareto ({args.env})")
        plt.legend()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/energy_mpl_pareto.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
