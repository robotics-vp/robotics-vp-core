#!/usr/bin/env python3
"""Evaluate whether regal nodes have earned additional authority."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.calibration import summarize_calibration
from src.replay.dataset import load_replay_dataset
from src.regality.promotion_policy import PromotionMetrics, load_regal_promotion_policy


NODE_IDS = (
    "objective_integrity_regal",
    "plausibility_regal",
    "reward_safety_regal",
    "pricing_truth_regal",
    "data_value_regal",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate regal promotion readiness")
    parser.add_argument("--replay-dataset-dir", required=True, type=str)
    parser.add_argument("--promotion-policy", default="configs/regality/promotion_default.yaml", type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    dataset = load_replay_dataset(args.replay_dataset_dir)
    policy = load_regal_promotion_policy(args.promotion_policy)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    decisions = []
    for node_id in NODE_IDS:
        metrics = _metrics_for_node(node_id, dataset.episodes)
        decisions.append(policy.evaluate_node(node_id, metrics, evidence_pointers={"dataset_dir": str(args.replay_dataset_dir)}))

    payload = {
        "policy_name": policy.policy_name,
        "config_digest": policy.config_digest,
        "dataset_digest": dataset.manifest.dataset_digest,
        "node_decisions": [decision.to_dict() for decision in decisions],
        "summary": {
            "recommend_promote": sum(1 for decision in decisions if decision.outcome == "recommend_promote"),
            "recommend_hold": sum(1 for decision in decisions if decision.outcome == "recommend_hold"),
            "recommend_demote": sum(1 for decision in decisions if decision.outcome == "recommend_demote"),
        },
    }
    json_path = output_root / "regal_promotion_eval.json"
    md_path = output_root / "regal_promotion_eval.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def _metrics_for_node(node_id: str, episodes) -> PromotionMetrics:
    count = max(1, len(episodes))
    confidences = []
    outcomes = []
    predictions = []
    targets = []
    monotonic_inputs = []
    monotonic_outputs = []
    reference_vectors = []
    current_vectors = []
    for episode in episodes:
        pricing_conf = float(episode.pricing_summary.get("confidence", 0.0))
        quality = float(episode.datapack_summary.get("quality_score", 0.0))
        frontier = float(episode.datapack_summary.get("marginal_frontier_gain", 0.0))
        hard_flags = sum(1 for flag in episode.constraint_flags if str(flag.get("severity", "")) == "hard")
        if node_id == "objective_integrity_regal":
            outcome = 1.0 if episode.objective_tensor_summary.get("axes") else 0.0
            prediction = 1.0 if episode.metadata.get("objective_profile_id") else 0.5
            confidence = 0.9 if outcome else 0.2
            baseline = outcome
        elif node_id == "pricing_truth_regal":
            outcome = 1.0 if pricing_conf >= 0.5 and float(episode.pricing_summary.get("net_customer_rate", 0.0)) >= 0.0 else 0.0
            prediction = min(1.0, pricing_conf + 0.1)
            confidence = pricing_conf
            baseline = 1.0 if episode.regal_summary.get("pricing_recommendation", "publish") != "suppress" else 0.0
        elif node_id == "data_value_regal":
            outcome = 1.0 if frontier > 0.0 or quality > 0.5 else 0.0
            prediction = min(1.0, max(0.0, 0.5 + frontier))
            confidence = min(1.0, max(0.0, quality))
            baseline = 1.0 if episode.regal_summary.get("datapack_recommendation", "keep") != "downweight" else 0.0
        elif node_id == "reward_safety_regal":
            outcome = 1.0 if hard_flags == 0 else 0.0
            prediction = 1.0 if float(episode.total_reward) >= 0.0 else 0.0
            confidence = 0.85 if hard_flags == 0 else 0.35
            baseline = 1.0 if episode.regal_summary.get("deploy_recommendation", "allow_shadow") == "allow_shadow" else 0.0
        else:
            outcome = 1.0 if hard_flags == 0 and quality > 0.4 else 0.0
            prediction = 1.0 if pricing_conf > 0.4 else 0.0
            confidence = 0.8 if outcome else 0.3
            baseline = 1.0 if episode.regal_summary.get("overall_status", "pass") == "pass" else 0.0
        confidences.append(confidence)
        outcomes.append(outcome)
        predictions.append(prediction)
        targets.append(baseline)
        monotonic_inputs.append(float(episode.total_steps))
        monotonic_outputs.append(confidence)
        reference_vectors.append(list(episode.condition_vector_values))
        current_vectors.append(list(episode.condition_vector_values))
    calibration = summarize_calibration(
        confidences=confidences,
        outcomes=outcomes,
        predictions=predictions,
        targets=targets,
        monotonic_inputs=monotonic_inputs,
        monotonic_outputs=monotonic_outputs,
        reference_vectors=reference_vectors,
        current_vectors=current_vectors,
        metadata={"node_id": node_id},
    )
    agreement = sum(1 for prediction, target in zip(predictions, targets) if abs(prediction - target) <= 0.25) / float(count)
    false_positive = sum(1 for prediction, outcome in zip(predictions, outcomes) if prediction >= 0.5 and outcome < 0.5) / float(count)
    false_negative = sum(1 for prediction, outcome in zip(predictions, outcomes) if prediction < 0.5 and outcome >= 0.5) / float(count)
    return PromotionMetrics(
        replay_coverage=min(1.0, float(count) / 10.0 + 0.4),
        downstream_label_count=count,
        deployment_receipt_count=sum(1 for episode in episodes if episode.ledger_event_ids),
        calibration_error=calibration.expected_calibration_error,
        baseline_agreement=agreement,
        monotonicity=calibration.monotonicity_score,
        sign_consistency=calibration.sign_consistency,
        false_positive_rate=false_positive,
        false_negative_rate=false_negative,
        drift_score=calibration.drift_score,
        residual_gain=0.0,
        calibration_summary=calibration,
        metadata={"node_id": node_id},
    )


def _markdown(payload: dict) -> str:
    lines = [
        "# Regal Promotion Evaluation",
        "",
        f"- Policy: {payload['policy_name']}",
        f"- Promote: {payload['summary']['recommend_promote']}",
        f"- Hold: {payload['summary']['recommend_hold']}",
        f"- Demote: {payload['summary']['recommend_demote']}",
        "",
        "## Node Decisions",
    ]
    for decision in payload["node_decisions"]:
        lines.extend(
            [
                f"### {decision['node_id']}",
                f"- Current stage: {decision['current_stage']}",
                f"- Recommendation: {decision['outcome']} -> {decision['recommended_stage']}",
                f"- Reasons: {', '.join(decision['reasons'])}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
