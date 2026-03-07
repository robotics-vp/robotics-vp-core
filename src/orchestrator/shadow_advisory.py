"""Advisory-only trainer/orchestrator outputs from the shadow learning stack."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

from src.phase_h.advisory_integration import MAX_MULTIPLIER, MIN_MULTIPLIER
from src.replay.dataset import load_replay_dataset
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
) -> Dict[str, Any]:
    dataset = load_replay_dataset(replay_dataset_dir)
    steps_by_episode = defaultdict(list)
    for step in dataset.steps:
        steps_by_episode[step.episode_id].append(step)

    episode_outputs = []
    for episode in dataset.episodes:
        policy_result = (policy_advisor or PolicyAdvisor()).summarize_episode(steps_by_episode.get(episode.episode_id, []))
        pricing_result = (pricing_advisor or PricingAdvisor()).assess_episode(episode)
        data_value_result = (data_value_advisor or DataValueAdvisor()).assess_episode(episode)
        regal_support_result = (regal_support_advisor or RegalSupportAdvisor()).assess_episode(episode)

        policy_mae = policy_result.applied_output.get("mean_action_mae")
        policy_uncertainty = policy_result.applied_output.get("mean_uncertainty", 1.0)
        learned_data_value = float(data_value_result.learned_output.get("predicted_data_value", data_value_result.applied_output.get("data_share_credit", 0.0) or 0.0))
        learned_pricing_delta = float(pricing_result.learned_output.get("predicted_residual", 0.0))
        anomaly_support = float(regal_support_result.learned_output.get("anomaly_support_score", 0.0))

        deploy_recommendation = str(episode.regal_summary.get("deploy_recommendation", "allow_shadow"))
        datapack_recommendation = str(episode.regal_summary.get("datapack_recommendation", "keep"))
        pricing_recommendation = str(episode.regal_summary.get("pricing_recommendation", "publish"))

        priority_score = 0.5
        if deploy_recommendation == "require_review":
            priority_score += 0.10
        if datapack_recommendation in {"review", "reward_credit"}:
            priority_score += 0.10
        if learned_data_value > 1.0:
            priority_score += 0.15
        if learned_pricing_delta < -2.0:
            priority_score += 0.10
        if policy_mae is not None and float(policy_mae) > 0.25:
            priority_score += 0.15
        if anomaly_support > 0.65:
            priority_score += 0.10
        priority_score = max(0.0, min(1.0, priority_score))

        sampling_priority = "low"
        if priority_score >= 0.70:
            sampling_priority = "high"
        elif priority_score >= 0.45:
            sampling_priority = "medium"

        weight_delta = 0.0
        weight_delta += min(0.15, learned_data_value / 20.0)
        weight_delta -= min(0.20, max(0.0, anomaly_support - 0.4))
        if policy_mae is not None:
            weight_delta -= min(0.10, max(0.0, float(policy_mae) - 0.20))
        slice_weight_multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, 1.0 + weight_delta))

        tags = [f"skill:{episode.skill_mode}", f"pricing:{pricing_recommendation}", f"datapack:{datapack_recommendation}"]
        if deploy_recommendation != "allow_shadow":
            tags.append(f"deploy:{deploy_recommendation}")
        if learned_pricing_delta < -2.0:
            tags.append("pricing_discount_candidate")
        if anomaly_support > 0.65:
            tags.append("regal_anomaly_support_high")
        if policy_mae is not None and float(policy_mae) > 0.25:
            tags.append("policy_error_high")
        if policy_uncertainty and float(policy_uncertainty) > 0.45:
            tags.append("policy_uncertainty_high")

        collect_more_data = any(
            [
                deploy_recommendation != "allow_shadow",
                datapack_recommendation in {"review", "downweight"},
                anomaly_support > 0.75,
            ]
        )
        retrain = any(
            [
                policy_mae is not None and float(policy_mae) > 0.20,
                learned_pricing_delta < -1.0,
                pricing_recommendation == "publish_discounted",
            ]
        )

        episode_outputs.append(
            {
                "episode_id": episode.episode_id,
                "sampling_priority": sampling_priority,
                "sampling_priority_score": priority_score,
                "slice_weight_multiplier": slice_weight_multiplier,
                "replay_queue_tags": sorted(tags),
                "collect_more_data": bool(collect_more_data),
                "retrain": bool(retrain),
                "deploy_recommendation": deploy_recommendation,
                "pricing_recommendation": pricing_recommendation,
                "datapack_recommendation": datapack_recommendation,
                "policy_advisor": policy_result.to_dict(),
                "pricing_advisor": pricing_result.to_dict(),
                "data_value_advisor": data_value_result.to_dict(),
                "regal_support_advisor": regal_support_result.to_dict(),
            }
        )

    summary = {
        "episodes": len(episode_outputs),
        "sampling_priorities": dict(Counter(output["sampling_priority"] for output in episode_outputs)),
        "collect_more_data_count": sum(1 for output in episode_outputs if output["collect_more_data"]),
        "retrain_count": sum(1 for output in episode_outputs if output["retrain"]),
        "mean_slice_weight_multiplier": (
            sum(float(output["slice_weight_multiplier"]) for output in episode_outputs) / max(len(episode_outputs), 1)
        ),
    }
    return {
        "summary": summary,
        "episodes": episode_outputs,
        "dataset_digest": dataset.manifest.dataset_digest,
    }
