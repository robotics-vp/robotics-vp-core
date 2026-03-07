#!/usr/bin/env python3
"""Evaluate learned shadow pricing/data-value/regal-support models against heuristics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay.dataset import load_replay_dataset
from src.shadow_runtime.advisors import AdvisorMode, DataValueAdvisor, PricingAdvisor, RegalSupportAdvisor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned shadow pricing/value models")
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--pricing-checkpoint", required=True, type=str)
    parser.add_argument("--data-value-checkpoint", required=True, type=str)
    parser.add_argument("--regal-support-checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    dataset = load_replay_dataset(args.dataset_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pricing = PricingAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=args.pricing_checkpoint)
    data_value = DataValueAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=args.data_value_checkpoint)
    regal = RegalSupportAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=args.regal_support_checkpoint)

    rows = []
    for episode in dataset.episodes:
        rows.append(
            {
                "episode_id": episode.episode_id,
                "pricing": pricing.assess_episode(episode).to_dict(),
                "data_value": data_value.assess_episode(episode).to_dict(),
                "regal_support": regal.assess_episode(episode).to_dict(),
            }
        )
    summary = {
        "episodes": len(rows),
        "mean_pricing_delta": (
            sum(float(row["pricing"]["learned_output"].get("predicted_residual", 0.0)) for row in rows) / max(len(rows), 1)
        ),
        "mean_data_value_prediction": (
            sum(float(row["data_value"]["learned_output"].get("predicted_data_value", 0.0)) for row in rows) / max(len(rows), 1)
        ),
        "mean_regal_support": (
            sum(float(row["regal_support"]["learned_output"].get("anomaly_support_score", 0.0)) for row in rows) / max(len(rows), 1)
        ),
    }
    (output_root / "shadow_model_eval.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_root / "shadow_model_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
