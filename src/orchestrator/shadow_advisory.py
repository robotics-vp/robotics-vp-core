"""Advisory-only trainer/orchestrator outputs from the shadow learning stack."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.economics.inferential_reward import compile_signal_yield
from src.economics.inferential_training_gate import InferentialTrainingCandidate, InferentialTrainingGate
from src.epiplexity.metadata import (
    extract_epiplexity_summary_confidence,
    extract_epiplexity_summary_metric,
    load_epiplexity_overlay_map,
)
from src.phase_h.advisory_integration import MAX_MULTIPLIER, MIN_MULTIPLIER
from src.orchestrator.adaptation_budgeting import evaluate_adaptation_budget
from src.orchestrator.queue_selection import build_live_queue_selection
from src.orchestrator.semantic_runtime_learning import build_semantic_runtime_learning_row
from src.orchestrator.semantic_runtime_scorer_runtime import (
    load_semantic_runtime_scorer_from_runtime_package,
)
from src.orchestrator.semantic_runtime_scorers import (
    coerce_semantic_runtime_scorer_package,
    load_semantic_runtime_scorer_package,
    score_semantic_runtime_learning_row,
)
from src.replay.receipt_ingest import resolve_receipt_label_bundle
from src.replay.dataset import load_replay_dataset
from src.replay.compatibility import check_replay_manifest_compatibility
from src.regality.promotion_policy import load_regal_promotion_policy
from src.rl.econ_regal_sampling import recommend_sampling
from src.shadow_runtime.advisors import (
    DataValueAdvisor,
    PolicyAdvisor,
    PricingAdvisor,
    RegalSupportAdvisor,
)


def _semantic_runtime_scorer_candidate_paths(
    *,
    replay_dataset_dir: str,
    semantic_runtime_scorer_package_path: Optional[str] = None,
) -> list[str]:
    candidate_paths: list[Path] = []
    if semantic_runtime_scorer_package_path:
        candidate_paths.append(Path(semantic_runtime_scorer_package_path))
    replay_root = Path(replay_dataset_dir)
    candidate_paths.extend(
        [
            replay_root / "semantic_runtime_scorer_runtime_package.json",
            replay_root / "semantic_runtime_scorer_package.json",
            replay_root.parent / "semantic_runtime_scorers" / "semantic_runtime_scorer_runtime_package.json",
            replay_root.parent / "semantic_runtime_scorers" / "semantic_runtime_scorer_package.json",
        ]
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidate_paths:
        resolved = str(candidate.resolve()) if candidate.is_absolute() else str(candidate)
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def _resolve_semantic_runtime_scorer_package(
    *,
    replay_dataset_dir: str,
    semantic_runtime_scorer_package: Optional[Any] = None,
    semantic_runtime_scorer_package_path: Optional[str] = None,
) -> tuple[Optional[Any], Optional[str], Dict[str, Any]]:
    if semantic_runtime_scorer_package is not None:
        return (
            coerce_semantic_runtime_scorer_package(semantic_runtime_scorer_package),
            None,
            {
                "contract_type": "direct_object",
                "benchmark_gate": {},
                "execution_preconditions": {},
                "promotion_stage": "shadow_candidate",
            },
        )

    candidate_paths = _semantic_runtime_scorer_candidate_paths(
        replay_dataset_dir=replay_dataset_dir,
        semantic_runtime_scorer_package_path=semantic_runtime_scorer_package_path,
    )
    if semantic_runtime_scorer_package_path and not Path(semantic_runtime_scorer_package_path).exists():
        raise FileNotFoundError(
            f"semantic runtime scorer package not found: {semantic_runtime_scorer_package_path}"
        )
    for candidate_ref in candidate_paths:
        candidate = Path(candidate_ref)
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and "scorer_package_path" in payload:
                scorer_package, runtime_package = load_semantic_runtime_scorer_from_runtime_package(candidate)
                return (
                    scorer_package,
                    str(candidate),
                    {
                        "contract_type": "runtime_package",
                        "benchmark_gate": dict(runtime_package.benchmark_gate or {}),
                        "execution_preconditions": dict(runtime_package.execution_preconditions or {}),
                        "promotion_stage": str(runtime_package.promotion_stage or "shadow_candidate"),
                        "package_id": runtime_package.package_id,
                    },
                )
            return (
                load_semantic_runtime_scorer_package(candidate),
                str(candidate),
                {
                    "contract_type": "legacy_package",
                    "benchmark_gate": {},
                    "execution_preconditions": {},
                    "promotion_stage": "legacy_fallback",
                },
            )
    return None, None, {
        "contract_type": "missing",
        "benchmark_gate": {},
        "execution_preconditions": {},
        "promotion_stage": "heuristic_fallback",
    }


def _build_semantic_runtime_scorer_preconditions(
    *,
    replay_dataset_dir: str,
    manifest_compatibility,
    semantic_runtime_package: Optional[Any],
    semantic_runtime_package_ref: Optional[str],
    semantic_runtime_contract: Mapping[str, Any],
    candidate_paths: Sequence[str],
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    package_ready = semantic_runtime_package is not None
    benchmark_gate = dict(semantic_runtime_contract.get("benchmark_gate", {}) or {})
    execution_preconditions = dict(semantic_runtime_contract.get("execution_preconditions", {}) or {})
    contract_type = str(semantic_runtime_contract.get("contract_type", "missing") or "missing")
    satisfied_preconditions = {
        "artifact::semantic_runtime_scorer_package": int(package_ready),
        "artifact::semantic_runtime_scorer_runtime_package": int(contract_type == "runtime_package"),
        "replay_manifest::compatible": int(bool(getattr(manifest_compatibility, "compatible", False))),
        "benchmark::semantic_runtime_runtime_row_density": int(bool(benchmark_gate.get("ready", False))),
    }
    unsatisfied = [
        key for key, value in sorted(satisfied_preconditions.items()) if not value
    ]
    preconditions = {
        "schema_version": "semantic_runtime_scorer_preconditions_v2",
        "replay_dataset_dir": replay_dataset_dir,
        "semantic_runtime_scorer_package_ref": semantic_runtime_package_ref,
        "candidate_paths": list(candidate_paths),
        "ready": package_ready,
        "fallback_active": (not package_ready) or contract_type == "legacy_package",
        "contract_type": contract_type,
        "promotion_stage": str(semantic_runtime_contract.get("promotion_stage", "heuristic_fallback")),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": execution_preconditions,
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
        "satisfied_preconditions": satisfied_preconditions,
        "unsatisfied_preconditions": unsatisfied,
        "manifest_compatibility": manifest_compatibility.to_dict(),
    }
    work_orders: list[Dict[str, Any]] = []
    if not package_ready:
        work_orders.append(
            {
                "order_type": "runtime_precondition",
                "subject_kind": "semantic_runtime_scorer_package",
                "subject_id": replay_dataset_dir,
                "ready": False,
                "blocking": True,
                "reason": "semantic_runtime_scorer_package_missing",
                "recommended_entrypoint": "scripts/train_semantic_runtime_scorers.py",
                "candidate_paths": list(candidate_paths),
                "required_preconditions": [
                    "artifact::semantic_runtime_scorer_package",
                    "artifact::semantic_runtime_scorer_runtime_package",
                    "replay_manifest::compatible",
                ],
            }
        )
    elif contract_type == "legacy_package":
        work_orders.append(
            {
                "order_type": "runtime_contract_upgrade",
                "subject_kind": "semantic_runtime_scorer_package",
                "subject_id": replay_dataset_dir,
                "ready": True,
                "blocking": False,
                "reason": "semantic_runtime_scorer_runtime_package_missing",
                "recommended_entrypoint": "scripts/train_semantic_runtime_scorers.py",
                "candidate_paths": list(candidate_paths),
                "required_preconditions": [
                    "artifact::semantic_runtime_scorer_runtime_package",
                    "replay_manifest::compatible",
                ],
            }
        )
    return preconditions, work_orders


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
    epiplexity_overlay_path: Optional[str] = None,
    semantic_runtime_scorer_package: Optional[Any] = None,
    semantic_runtime_scorer_package_path: Optional[str] = None,
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
    datapack_by_episode = {
        str(row.metadata.get("episode_id", "")): row
        for row in receipt_bundle.datapack_contributions
    }
    epiplexity_by_pack_id = load_epiplexity_overlay_map(epiplexity_overlay_path) if epiplexity_overlay_path else {}
    steps_by_episode = defaultdict(list)
    for step in dataset.steps:
        steps_by_episode[step.episode_id].append(step)
    windows_by_episode = defaultdict(list)
    for window in dataset.windows:
        windows_by_episode[window.episode_id].append(window)
    scorer_candidate_paths = _semantic_runtime_scorer_candidate_paths(
        replay_dataset_dir=replay_dataset_dir,
        semantic_runtime_scorer_package_path=semantic_runtime_scorer_package_path,
    )
    semantic_runtime_package, semantic_runtime_package_ref, semantic_runtime_contract = _resolve_semantic_runtime_scorer_package(
        replay_dataset_dir=replay_dataset_dir,
        semantic_runtime_scorer_package=semantic_runtime_scorer_package,
        semantic_runtime_scorer_package_path=semantic_runtime_scorer_package_path,
    )
    scorer_preconditions, scorer_work_orders = _build_semantic_runtime_scorer_preconditions(
        replay_dataset_dir=replay_dataset_dir,
        manifest_compatibility=manifest_compatibility,
        semantic_runtime_package=semantic_runtime_package,
        semantic_runtime_package_ref=semantic_runtime_package_ref,
        semantic_runtime_contract=semantic_runtime_contract,
        candidate_paths=scorer_candidate_paths,
    )

    episode_outputs: list[Dict[str, Any]] = []
    budget_candidates: list[InferentialTrainingCandidate] = []
    execution_preconditions_by_episode: Dict[str, Dict[str, Any]] = {
        episode.episode_id: dict(episode.metadata.get("execution_preconditions", {}) or {})
        for episode in dataset.episodes
    }
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
        datapack_id = str(
            episode.metadata.get("datapack_id")
            or episode.datapack_summary.get("datapack_id")
            or episode.episode_id
        )
        epiplexity_overlay = epiplexity_by_pack_id.get(datapack_id, {})
        provenance_quality = float(episode.datapack_summary.get("quality_score", 0.0) or 0.0)
        data_quality = float(episode.datapack_summary.get("quality_score", 0.0) or 0.0)
        frontier_gain = float(episode.datapack_summary.get("marginal_frontier_gain", 0.0) or 0.0)
        epi_delta = float(
            episode.datapack_summary.get(
                "delta_epi_per_flop",
                episode.datapack_summary.get("delta_epi_vs_baseline", 0.0),
            )
            or extract_epiplexity_summary_metric(epiplexity_overlay, metric="delta_epi_vs_baseline")
            or 0.0
        )
        epi_conf = float(
            episode.datapack_summary.get("epi_confidence", 0.0)
            or extract_epiplexity_summary_confidence(epiplexity_overlay)
            or 0.0
        )
        epi_per_flop = float(
            episode.datapack_summary.get("epi_per_flop", 0.0)
            or extract_epiplexity_summary_metric(epiplexity_overlay, metric="epi_per_flop")
            or 0.0
        )
        hard_flags = sum(1 for flag in episode.constraint_flags if str(flag.get("severity", "")) == "hard")
        deployment_label = deployment_by_episode.get(episode.episode_id)
        deployment_receipt = receipt_by_episode.get(episode.episode_id)
        adaptation_label = adaptation_by_episode.get(episode.episode_id)
        datapack_label = datapack_by_episode.get(episode.episode_id)
        use_realized_receipts = receipt_bundle.label_mode in {
            "sim_rollout",
            "training_run",
            "future_real_deployment",
        }
        transfer_score = max(
            0.0,
            1.0 - max(float(policy_uncertainty or 0.0), float(episode.condition_vector.get("ood_risk_level", 0.0) or 0.0)),
        )
        governance_penalty = 0.0
        if deploy_recommendation != "allow_shadow":
            governance_penalty += 0.15
        if pricing_recommendation == "suppress":
            governance_penalty += 0.25
        signal_yield = compile_signal_yield(
            frontier_gain=frontier_gain,
            epiplexity_delta=epi_delta,
            epiplexity_confidence=epi_conf,
            transfer_score=transfer_score,
            data_quality=data_quality,
            provenance_quality=provenance_quality,
        )
        semantic_runtime_score = None
        if semantic_runtime_package is not None:
            runtime_row = build_semantic_runtime_learning_row(
                episode,
                steps=steps_by_episode.get(episode.episode_id, []),
                windows=windows_by_episode.get(episode.episode_id, []),
                root_dir=dataset.root_dir,
                max_counterfactuals=2,
            )
            semantic_runtime_score = score_semantic_runtime_learning_row(
                semantic_runtime_package,
                runtime_row,
            )
        top_counterfactual_value = (
            float(semantic_runtime_score.counterfactual_scores[0].rescored_value)
            if semantic_runtime_score is not None and semantic_runtime_score.counterfactual_scores
            else 0.0
        )

        sampling = recommend_sampling(
            objective_profile_coverage_gap=coverage_gap,
            constraint_violation_count=hard_flags,
            uncertainty=float(policy_uncertainty or 0.0),
            datapack_value=learned_data_value,
            signal_yield_score=signal_yield.score,
            regal_support_score=anomaly_support,
            deploy_recommendation=deploy_recommendation,
            pricing_recommendation=pricing_recommendation,
            datapack_recommendation=datapack_recommendation,
            promotion_policy=promotion_policy,
            replay_policy_error=float(policy_mae or 0.0),
            provenance_quality=provenance_quality,
            semantic_runtime_route_score=(
                float(semantic_runtime_score.meta_route_success_probability)
                if semantic_runtime_score is not None
                else 0.0
            ),
            semantic_runtime_authority_confidence=(
                float(semantic_runtime_score.chosen_authority_confidence)
                if semantic_runtime_score is not None
                else 0.0
            ),
            semantic_runtime_counterfactual_value=top_counterfactual_value,
            semantic_runtime_predicted_regret=(
                float(semantic_runtime_score.predicted_regret)
                if semantic_runtime_score is not None
                else 0.0
            ),
            semantic_runtime_authority_switch_recommended=bool(
                semantic_runtime_score.authority_switch_recommended
            )
            if semantic_runtime_score is not None
            else False,
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
            expected_value_gain=(
                float(deployment_label.realized_value)
                if deployment_label is not None and use_realized_receipts
                else float(episode.econ_tensor_summary.get("axes", {}).get("value_earned", 0.0))
            ),
            compute_cost=max(0.05, 0.08 * max(1, episode.total_steps) / 10.0),
            risk_cost=float(episode.econ_tensor_summary.get("axes", {}).get("constraint_penalty", 0.0)),
            uncertainty=float(policy_uncertainty or 0.0),
            ood_score=float(episode.condition_vector.get("ood_risk_level", 0.0) or 0.0),
            data_quality=(
                max(0.0, data_quality - 0.15)
                if datapack_label is not None and datapack_label.downweight_recommended
                else data_quality
            ),
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
            expected_adaptation_benefit=(
                float(adaptation_label.realized_gain)
                if adaptation_label is not None and use_realized_receipts
                else max(0.0, learned_data_value - float(policy_mae or 0.0))
            ),
            frontier_gain=frontier_gain,
            epiplexity_delta=epi_delta,
            epiplexity_confidence=epi_conf,
            transfer_score=transfer_score,
            governance_penalty=governance_penalty,
            signal_yield_score=signal_yield.score,
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
                "realized_reward": (
                    float(deployment_label.realized_reward)
                    if deployment_label is not None and deployment_label.realized_reward is not None
                    else None
                ),
                "signal_yield": signal_yield.to_dict(),
                "epiplexity_overlay": epiplexity_overlay or None,
            },
        )
        budget_candidates.append(candidate)

        episode_outputs.append(
            {
                "episode_id": episode.episode_id,
                "sampling_priority": sampling.priority_label,
                "sampling_priority_score": sampling.priority_score,
                "signal_yield_score": signal_yield.score,
                "inferential_signal_yield": signal_yield.to_dict(),
                "epiplexity_evidence": {
                    "datapack_id": datapack_id,
                    "delta_epi_vs_baseline": epi_delta,
                    "epi_per_flop": epi_per_flop,
                    "confidence": epi_conf,
                    "overlay_joined": bool(epiplexity_overlay),
                },
                "slice_weight_multiplier": slice_weight_multiplier,
                "replay_queue_tags": sampling.queue_tags,
                "replay_action": sampling.replay_action,
                "deploy_recommendation": deploy_recommendation,
                "pricing_recommendation": pricing_recommendation,
                "datapack_recommendation": datapack_recommendation,
                "sampling_recommendation": sampling.to_dict(),
                "semantic_runtime_score": (
                    semantic_runtime_score.to_dict() if semantic_runtime_score is not None else None
                ),
                "semantic_runtime_scorer_contract": dict(semantic_runtime_contract),
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
                    "datapack_contribution": (
                        datapack_label.to_dict() if datapack_label is not None else None
                    ),
                },
                "advisor_evaluation": {
                    "pricing_alignment": (
                        {
                            "predicted_rate": float(pricing_result.applied_output.get("net_customer_rate", 0.0) or 0.0),
                            "pricing_accepted": bool(deployment_label.pricing_accepted),
                        }
                        if deployment_label is not None
                        else None
                    ),
                    "data_value_alignment": (
                        {
                            "predicted_data_value": float(learned_data_value),
                            "realized_frontier_gain": float(datapack_label.marginal_frontier_gain_realized),
                            "downweight_recommended": bool(datapack_label.downweight_recommended),
                        }
                        if datapack_label is not None
                        else None
                    ),
                    "policy_alignment": (
                        {
                            "policy_error": float(policy_mae or 0.0),
                            "task_success": deployment_label.task_success,
                            "objective_satisfied": deployment_label.objective_satisfied,
                        }
                        if deployment_label is not None
                        else None
                    ),
                },
                "execution_preconditions": execution_preconditions_by_episode.get(episode.episode_id, {}),
            }
        )

    gate = InferentialTrainingGate(promotion_policy=promotion_policy)
    budget_artifact = evaluate_adaptation_budget(
        gate=gate,
        candidates=budget_candidates,
        execution_preconditions=execution_preconditions_by_episode,
    )
    decisions_by_episode: Dict[str, Dict[str, Any]] = {
        str(row.get("artifact_summary", {}).get("episode_id", candidate.episode_id)): row
        for row, candidate in zip(budget_artifact.decisions, budget_candidates)
    }
    work_orders_by_episode: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for work_order in budget_artifact.work_orders:
        work_orders_by_episode[str(work_order.get("subject_id", ""))].append(dict(work_order))
    for episode_output, candidate in zip(episode_outputs, budget_candidates):
        budget_decision = decisions_by_episode.get(candidate.episode_id) or gate.evaluate(candidate).to_dict()
        episode_output["inferential_budget_decision"] = budget_decision
        episode_output["collect_more_data"] = budget_decision["decision"] == "collect_more_data"
        episode_output["retrain"] = budget_decision["decision"] == "adapt_now"
        episode_output["execution_work_orders"] = work_orders_by_episode.get(candidate.episode_id, [])
        if episode_output["receipt_feedback"]["deployment_outcome"] is not None:
            episode_output["inferential_budget_decision"]["artifact_summary"]["receipt_feedback"] = {
                "realized_value": episode_output["receipt_feedback"]["deployment_outcome"]["realized_value"],
                "pricing_accepted": episode_output["receipt_feedback"]["deployment_outcome"]["pricing_accepted"],
            }

    summary: Dict[str, Any] = {
        "episodes": len(episode_outputs),
        "sampling_priorities": dict(Counter(output["sampling_priority"] for output in episode_outputs)),
        "collect_more_data_count": sum(1 for output in episode_outputs if output["collect_more_data"]),
        "retrain_count": sum(1 for output in episode_outputs if output["retrain"]),
        "epiplexity_overlay_joins": sum(
            1
            for output in episode_outputs
            if bool(output.get("epiplexity_evidence", {}).get("overlay_joined"))
        ),
        "semantic_runtime_scorer_episodes": sum(
            1 for output in episode_outputs if output.get("semantic_runtime_score") is not None
        ),
        "semantic_runtime_scorer_ready": bool(scorer_preconditions.get("ready", False)),
        "semantic_runtime_scorer_fallback_active": bool(scorer_preconditions.get("fallback_active", False)),
        "semantic_runtime_scorer_contract_type": str(scorer_preconditions.get("contract_type", "missing")),
        "semantic_runtime_scorer_promotion_stage": str(
            scorer_preconditions.get("promotion_stage", "heuristic_fallback")
        ),
        "semantic_runtime_scorer_benchmark_gate_ready": bool(
            scorer_preconditions.get("benchmark_gate_ready", False)
        ),
        "semantic_runtime_scorer_package_ref": semantic_runtime_package_ref,
        "mean_semantic_runtime_route_score": (
            sum(
                float(output.get("semantic_runtime_score", {}).get("meta_route_success_probability", 0.0) or 0.0)
                for output in episode_outputs
                if output.get("semantic_runtime_score") is not None
            )
            / max(sum(1 for output in episode_outputs if output.get("semantic_runtime_score") is not None), 1)
        ),
        "mean_slice_weight_multiplier": (
            sum(float(output["slice_weight_multiplier"]) for output in episode_outputs) / max(len(episode_outputs), 1)
        ),
        "manifest_compatibility": manifest_compatibility.to_dict(),
        "receipt_label_coverage": receipt_bundle.coverage_summary(),
        "advisor_realized_feedback_count": sum(
            1
            for output in episode_outputs
            if output["advisor_evaluation"]["pricing_alignment"] is not None
            or output["advisor_evaluation"]["data_value_alignment"] is not None
            or output["advisor_evaluation"]["policy_alignment"] is not None
        ),
    }
    payload: Dict[str, Any] = {
        "summary": summary,
        "episodes": episode_outputs,
        "dataset_digest": dataset.manifest.dataset_digest,
        "promotion_policy": {
            "policy_name": promotion_policy.policy_name,
            "config_digest": promotion_policy.config_digest,
        },
        "semantic_runtime_scorer_preconditions": scorer_preconditions,
        "semantic_runtime_scorer_work_orders": scorer_work_orders,
        "adaptation_budget": budget_artifact.to_dict(),
        "adaptation_work_orders": [
            row for row in budget_artifact.work_orders
            if row.get("order_type") == "adaptation_training"
        ],
        "collection_work_orders": [
            row for row in budget_artifact.work_orders
            if row.get("order_type") == "data_collection"
        ],
        "review_work_orders": [
            row for row in budget_artifact.work_orders
            if row.get("order_type") == "human_review"
        ],
        "receipt_label_coverage": receipt_bundle.coverage_summary(),
    }
    payload["live_queue_selection"] = build_live_queue_selection(payload)
    return payload
