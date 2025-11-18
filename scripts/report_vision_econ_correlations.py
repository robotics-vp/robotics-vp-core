#!/usr/bin/env python3
"""
Report econ metrics bucketed by vision metadata.
"""
import argparse
import json
import csv
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore  # noqa: E402
from src.analytics.vision_econ import summarize_vision_econ_correlations, correlation_rows_for_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Vision/Econ correlation report")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, default="")
    parser.add_argument("--output-json", type=str, default="results/reports/vision_econ_correlations.json")
    parser.add_argument("--output-csv", type=str, default="results/reports/vision_econ_correlations.csv")
    args = parser.parse_args()

    store = OntologyStore(root_dir=args.ontology_root)
    episodes = store.list_episodes(task_id=args.task_id or None)
    econ_vectors = store.list_econ_vectors()
    summary = summarize_vision_econ_correlations(episodes, econ_vectors)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, sort_keys=True, indent=2))

    if args.output_csv:
        rows = correlation_rows_for_csv(summary)
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group", "bucket", "metric", "mean"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"[report_vision_econ_correlations] wrote {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
