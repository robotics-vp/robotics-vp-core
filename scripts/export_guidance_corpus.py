#!/usr/bin/env python3
"""
Export guidance corpus (structured “why good/bad”) for SIMA/VLA/meta-judges.
"""
import argparse
import json
import os

from src.valuation.datapack_repo import DataPackRepo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapacks-dir", type=str, default="data/datapacks")
    parser.add_argument("--guidance-overlays", type=str, default="data/datapacks/guidance_overlays.jsonl")
    parser.add_argument("--env", type=str, default="drawer_vase_arm")
    parser.add_argument("--out", type=str, default="results/guidance_corpus.jsonl")
    args = parser.parse_args()

    repo = DataPackRepo(base_dir=args.datapacks_dir)
    dps = list(repo.iter_all(args.env) or [])

    # Overlay lookup
    overlay = {}
    if os.path.exists(args.guidance_overlays):
        with open(args.guidance_overlays, "r") as f:
            for line in f:
                rec = json.loads(line)
                overlay[rec.get("pack_id")] = rec.get("guidance_profile")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    count = 0
    with open(args.out, "w") as f:
        for dp in dps:
            gp = dp.guidance_profile.to_dict() if dp.guidance_profile else overlay.get(dp.pack_id)
            if not gp:
                continue
            template_summary = (
                f"In env {gp['env_name']} with objective {gp['objective_vector']}, "
                f"this episode was {gp['quality_label']} because {gp['main_driver']} "
                f"with tags {gp.get('semantic_tags', [])}."
            )
            record = {
                "env_name": gp["env_name"],
                "task_type": gp["task_type"],
                "customer_segment": gp["customer_segment"],
                "objective_vector": gp["objective_vector"],
                "is_good": gp["is_good"],
                "quality_label": gp["quality_label"],
                "semantic_tags": gp.get("semantic_tags", []),
                "numeric": {
                    "delta_mpl": gp["delta_mpl"],
                    "delta_error": gp["delta_error"],
                    "delta_energy_Wh": gp["delta_energy_Wh"],
                    "delta_J": gp["delta_J"],
                },
                "template_summary": template_summary,
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    print(f"Wrote {count} records to {args.out}")


if __name__ == "__main__":
    main()
