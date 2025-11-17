#!/usr/bin/env python3
"""
Preview sampling overrides based on latest semantic metrics.
"""
import glob
import json
import os

from src.orchestrator.semantic_metrics import load_semantic_metrics
from src.orchestrator.datapack_engine import DatapackEngine
from src.valuation.datapack_repo import DataPackRepo


def main():
    metrics_files = sorted(glob.glob("data/semantic_metrics/*.jsonl"))
    if not metrics_files:
        print("No semantic metrics files found.")
        return
    latest = metrics_files[-1]
    metrics_list = load_semantic_metrics(latest)
    if not metrics_list:
        print("No metrics in latest file.")
        return
    metrics = metrics_list[-1]
    engine = DatapackEngine(DataPackRepo(base_dir="data/datapacks"))
    overrides = engine.update_sampling_from_semantics(metrics)
    print("Latest metrics file:", latest)
    print("Overrides:", overrides)
    os.makedirs("results", exist_ok=True)
    with open("results/semantic_sampling_overrides.json", "w") as f:
        json.dump(overrides, f, indent=2)


if __name__ == "__main__":
    main()
