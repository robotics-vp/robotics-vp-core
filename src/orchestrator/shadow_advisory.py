"""Advisory-only trainer/orchestrator outputs from the shadow learning stack."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Optional

from src.economics.inferential_training_gate import InferentialTrainingCandidate, InferentialTrainingGate
from src.phase_h.advisory_integration import MAX_MULTIPLIER, MIN_MULTIPLIER
from src.orchestrator.adaptation_budgeting import evaluate_adaptation_budget
from src.orchestrator.queue_selection import build_live_queue_selection
from src.replay.receipt_ingest import resolve_receipt_label_bundle
from src.replay.dataset import load_replay_dataset
from src.replay.compatibility import check_replay_manifest_compatibility
from src.regality.promotion_policy import load_regal_promotion_policy
from src.rl.econ_regal_sampling import recommend_sampling
from src.shadow_runtime.advisors import (
    AdvisorMode,
    DataValueAdvisor,
    PolicyAdvisor,
    PricingAdvisor,
    RegalSupportAdvisor,
)


def build_shadow_advisory_output(
    *,
    replay_dataset_dir: str,
    policy_advisor: Optional[PolicyAdvisor] = None,
    pricing_advisor: Optional[PricingAdvisor] = None,
    data_value_advisor: Optional[DataValueAdvisor] = None,
    regal_support_advisor: Optional[RegalSupportAdvisor] = None,
    promotion_policy_path: str = "configs/regality/promotion_default.yaml",
    receipt_label_dir: Optional[str] = None,
    receipt_label_mode: str = "synthetic_shadow",
) -> Dict[str, Any]:
    dataset = load_replay_dataset(replay_dataset_dir)
    promotion_policy = load_regal_promotion_policy(promotion_policy_path)
    manifest_compatibility = check_replay_manifest_compatibility(dataset.manifest, expected_schema_version=dataset.manifest.schema_version)
    receipt_bundle = resolve_receipt_label_bundle(
        dataset=dataset,
        receipt_label_dir=receipt_label_dir,
        allow_synthetic=True,
        label_mode=receipt_label_mode,
    )
    deployment_by_episode = {
        row.episode_id: row for row in receipt_bundle.deployment_outcomes
    }
    receipt_by_episode = {
        row.episode_id: row for row in receipt_bundle.deployment_receipts
    }
    adaptation_by_episode = {
        str(row.metadata.get("episode_id", "")): row
        for row in receipt_bundle.adaptation_outcomes
    }
    steps_by_episode = defaultdict(list)
    for step in dataset.steps:
        steps_by_episode[step.episode_id].append(step)

    episode_outputs = []
    budget_candidates = []
    for episode in dataset.episodes:
        policy_result = (policy_advisor or PolicyAdvisor()).summarize_episode(steps_by_episode.get(episode.episode_id, []))
        pricing_result = (pricing_advisor or PricingAdvisor()).assess_episode(episode)
        data_value_result = (data_value_advisor or DataValueAdvisor()).assess_episode(episode)
        regal_support_result = (regal_support_advisor or RegalSupportAdvisor()).assess_episode(episode)

        policy_mae = policy_result.learned_output.get("mean_action_mae", policy_result.applied_output.get("mean_action_mae"))
        policy_uncertainty = policy_result.learned_output.get("mean_uncertainty", policy_result.applied_output.get("mean_uncertainty", 1.0))
        learned_data_value = float(data_value_result.learned_output.get("predicted_data_value", data_value_result.applied_output.get("data_share_credit", 0.0) or 0.0))
        learned_pricing_delta = float(pricing_result.learned_output.get("predicted_residual", 0.0))
        anomaly_support = float(regal_support_result.learned_output.get("anomaly_support_score", 0.0))

        deploy_recommendation = str(episode.regal_summary.get("deploy_recommendation", "allow_shadow"))
        datapack_recommendation = str(episode.regal_summary.get("datapack_recommendation", "keep"))
        pricing_recommendation = str(episode.regal_summary.get("pricing_recommendation", "publish"))
        coverage_gap = max(0.0, 1.0 - float(episode.condition_vector.get("safety_margin", 0.0) or 0.0))
        provenance_quality = float(episode.datapack_summary.get("quality_score", 0.0) or 0.0)
        data_quality = float(episode.datapack_summary.get("quality_score", 0.0) or 0.0)
        hard_flags = sum(1 for flag in episode.constraint_flags if str(flag.get("severity", "")) == "hard")
        deployment_label = deployment_by_episode.get(episode.episode_id)
        deployment_receipt = receipt_by_episode.get(episode.episode_id)
        adaptation_label = adaptation_by_episode.get(episode.episode_id)

        sampling = recommend_sampling(
            objective_profile_coverage_gap=coverage_gap,
            constraint_violation_count=hard_flags,
            uncertainty=float(policy_uncertainty or 0.0),
            datapack_value=learned_data_value,
            regal_support_score=anomaly_support,
            deploy_recommendation=deploy_recommendation,
            pricing_recommendation=pricing_recommendation,
            datapack_recommendation=datapack_recommendation,
            promotion_policy=promotion_policy,
            replay_policy_error=float(policy_mae or 0.0),
            provenance_quality=provenance_quality,
        )
        slice_weight_multiplier = max(
            MIN_MULTIPLIER,
            min(MAX_MULTIPLIER, float(sampling.weight_multiplier)),
        )

        candidate = InferentialTrainingCandidate(
            run_id=episode.run_id,
            episode_id=episode.episode_id,
            objective_profile_id=str(episode.metadata.get("objective_profile_id", "balanced_contract")),
            source_domain=episode.source_domain,
            expected_value_gain=float(episode.econ_tensor_summary.get("axes", {}).get("value_earned", 0.0)),
            compute_cost=max(0.05, 0.08 * max(1, episode.total_steps) / 10.0),
            risk_cost=float(episode.econ_tensor_summary.get("axes", {}).get("constraint_penalty", 0.0)),
            uncertainty=float(policy_uncertainty or 0.0),
            ood_score=float(episode.condition_vector.get("ood_risk_level", 0.0) or 0.0),
            data_quality=data_quality,
            provenance_quality=provenance_quality,
            pricing_summary=dict(pricing_result.applied_output),
            regal_statuses={
                "objective_integrity_regal": str(episode.regal_summary.get("overall_status", "pass")),
                "reward_safety_regal": "warn" if anomaly_support > 0.65 else "pass",
                "pricing_truth_regal": "fail" if pricing_recommendation == "suppress" else "pass",
            },
            regal_scores={
                "overall": float(episode.regal_summary.get("score", 0.75) or 0.75),
                "regal_support": anomaly_support,
            },
            replay_policy_uncertainty=float(policy_uncertainty or 0.0),
            learned_data_value=learned_data_value,
            expected_adaptation_benefit=max(0.0, learned_data_value - float(policy_mae or 0.0)),
            metadata={
                "pricing_delta": learned_pricing_delta,
                "realized_gain": (
                    float(adaptation_label.realized_gain)
                    if adaptation_label is not None
                    else None
                ),
                "realized_value": (
                    float(deployment_label.realized_value)
                    if deployment_label is not None
                    else None
                ),
            },
        )
        budget_candidates.append(candidate)

        episode_outputs.append(
            {
                "episode_id": episode.episode_id,
                "sampling_priority": sampling.priority_label,
                "sampling_priority_score": sampling.priority_score,
                "slice_weight_multiplier": slice_weight_multiplier,
                "replay_queue_tags": sampling.queue_tags,
                "replay_action": sampling.replay_action,
                "deploy_recommendation": deploy_recommendation,
                "pricing_recommendation": pricing_recommendation,
                "datapack_recommendation": datapack_recommendation,
                "sampling_recommendation": sampling.to_dict(),
                "policy_advisor": policy_result.to_dict(),
                "pricing_advisor": pricing_result.to_dict(),
                "data_value_advisor": data_value_result.to_dict(),
                "regal_support_advisor": regal_support_result.to_dict(),
                "receipt_feedback": {
                    "deployment_outcome": (
                        deployment_label.to_dict() if deployment_label is not None else None
                    ),
                    "deployment_receipt": (
                        deployment_receipt.to_dict() if deployment_receipt is not None else None
                    ),
                    "adaptation_outcome": (
                        adaptation_label.to_dict() if adaptation_label is not None else None
                    ),
                },
            }
        )

    gate = InferentialTrainingGate(promotion_policy=promotion_policy)
    budget_artifact = evaluate_adaptation_budget(gate=gate, candidates=budget_candidates)
    decisions_by_episode = {
        str(row.get("artifact_summary", {}).get("episode_id", candidate.episode_id)): row
        for row, candidate in zip(budget_artifact.decisions, budget_candidates)
    }
    for episode_output, candidate in zip(episode_outputs, budget_candidates):
        budget_decision = decisions_by_episode.get(candidate.episode_id) or gate.evaluate(candidate).to_dict()
        episode_output["inferential_budget_decision"] = budget_decision
        episode_output["collect_more_data"] = budget_decision["decision"] == "collect_more_data"
        episode_output["retrain"] = budget_decision["decision"] == "adapt_now"
        if episode_output["receipt_feedback"]["deployment_outcome"] is not None:
            episode_output["inferential_budget_decision"]["artifact_summary"]["receipt_feedback"] = {
                "realized_value": episode_output["receipt_feedback"]["deployment_outcome"]["realized_value"],
                "pricing_accepted": episode_output["receipt_feedback"]["deployment_outcome"]["pricing_accepted"],
            }

    summary = {
        "episodes": len(episode_outputs),
        "sampling_priorities": dict(Counter(output["sampling_priority"] for output in episode_outputs)),
        "collect_more_data_count": sum(1 for output in episode_outputs if output["collect_more_data"]),
        "retrain_count": sum(1 for output in episode_outputs if output["retrain"]),
        "mean_slice_weight_multiplier": (
            sum(float(output["slice_weight_multiplier"]) for output in episode_outputs) / max(len(episode_outputs), 1)
        ),
        "manifest_compatibility": manifest_compatibility.to_dict(),
        "receipt_label_coverage": receipt_bundle.coverage_summary(),
    }
    payload = {
        "summary": summary,
        "episodes": episode_outputs,
        "dataset_digest": dataset.manifest.dataset_digest,
        "promotion_policy": {
            "policy_name": promotion_policy.policy_name,
            "config_digest": promotion_policy.config_digest,
        },
        "adaptation_budget": budget_artifact.to_dict(),
        "receipt_label_coverage": receipt_bundle.coverage_summary(),
    }
    payload["live_queue_selection"] = build_live_queue_selection(payload)
    return payload
