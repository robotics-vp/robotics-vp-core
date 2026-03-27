#!/usr/bin/env python3
"""Online SAC training with ontology logging, bounded queue influence, and live evidence artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.config.econ_params import EconParams
from src.economics.functor import ObjectiveEconFunctor
from src.economics.reward_engine import RewardEngine
from src.encoders.mlp_encoder import EncoderWithAuxiliaries
from src.envs.dishwashing_env import DishwashingEnv, summarize_episode_info
from src.logging.episode_logger import EpisodeLogger
from src.motor_backend.rollout_capture import EpisodeMetadata
from src.objectives.profile import ObjectiveProfile
from src.ontology.store import OntologyStore
from src.ontology.models import Task, Robot
from src.orchestrator.queue_selection import build_live_queue_selection
from src.orchestrator.shadow_advisory import build_shadow_advisory_output
from src.regality.promotion_policy import load_regal_promotion_policy
from src.regality.promotion_reporting import (
    build_promotion_evidence_report,
    write_promotion_evidence_report,
)
from src.replay.dataset import ReplayDatasetBuilder
from src.replay.receipt_ingest import (
    build_training_run_receipt_label_bundle,
    write_receipt_label_bundle,
)
from src.rl.episode_sampling import DataPackRLSampler
from src.rl.econ_regal_sampling import recommend_sampling
from src.rl.sac import SACAgent
from src.rl.sac_contract_aware_adapter import (
    SACContractAwareAdapter,
    SACContractAwareAdapterConfig,
)
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.training_manifest import (
    build_replay_dataset_summary,
    build_replay_trajectory_audits,
    build_source_domain_coverage,
)
from src.training.wrap_training_entrypoint import regal_training
from src.envs.workcell_env.base import EpisodeLog
from src.utils.config_digest import sha256_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online SAC training with ontology logging")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--output-dir", type=str, default="artifacts/train_sac_with_ontology_logging")
    parser.add_argument("--task-id", type=str, default="task_dishwashing")
    parser.add_argument("--robot-id", type=str, default="robot_sac")
    parser.add_argument("--econ-domain", type=str, default="default")
    parser.add_argument("--promotion-policy", type=str, default="configs/regality/promotion_default.yaml")
    parser.add_argument("--receipt-label-mode", type=str, default="training_run")
    parser.add_argument("--queue-selection-mode", type=str, default="log_only")
    parser.add_argument("--queue-max-upweight", type=float, default=2.0)
    parser.add_argument("--queue-max-downweight", type=float, default=0.5)
    parser.add_argument("--queue-allow-slice-removal-on-integrity-failure", action="store_true")
    parser.add_argument("--queue-policy-helper-mode", type=str, default="disabled")
    parser.add_argument("--queue-policy-package-path", type=str, default=None)
    parser.add_argument("--sampler-policy-helper-mode", type=str, default="disabled")
    parser.add_argument("--sampler-policy-package-path", type=str, default=None)
    parser.add_argument("--learning-starts", type=int, default=64)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=100000)
    parser.add_argument("--contract-aware-config", type=str, default=None)
    parser.add_argument(
        "--contract-aware-mode",
        type=str,
        default="disabled",
        choices=["disabled", "sidecar", "live_loss"],
    )
    parser.add_argument(
        "--objective-scalarizer",
        type=str,
        default="legacy",
        choices=["legacy", "weighted_sum", "constrained", "lexicographic", "chebyshev", "epsilon", "product"],
    )
    args, _ = parser.parse_known_args()
    return args


def _objective_profile(objective_scalarizer: str) -> Optional[ObjectiveProfile]:
    if objective_scalarizer == "legacy":
        return None
    return ObjectiveProfile(
        profile_id=f"train_sac_with_ontology_logging:{objective_scalarizer}",
        scalarizer=objective_scalarizer,
        weights={
            "throughput": 1.0,
            "error": 1.0,
            "safety": 1.0,
            "energy": 1.0,
        },
        maximize={
            "throughput": True,
            "error": False,
            "safety": True,
            "energy": False,
        },
    )


def _contract_aware_adapter(
    *,
    args: argparse.Namespace,
    output_root: Path,
    latent_dim: int,
    action_dim: int,
) -> Optional[SACContractAwareAdapter]:
    if args.contract_aware_mode == "disabled" and not args.contract_aware_config:
        return None
    payload: Dict[str, Any] = {}
    if args.contract_aware_config:
        raw = yaml.safe_load(Path(args.contract_aware_config).read_text(encoding="utf-8")) or {}
        payload = dict(raw.get("adapter", raw) or {})
    payload.setdefault("enabled", args.contract_aware_mode != "disabled")
    payload.setdefault("latent_dim", latent_dim)
    payload.setdefault("action_dim", action_dim)
    payload.setdefault("condition_dim", 1)
    payload.setdefault("artifact_dir", str(output_root / "contract_aware"))
    if args.contract_aware_mode != "disabled":
        payload["enabled"] = True
        payload["mode"] = args.contract_aware_mode
    config = SACContractAwareAdapterConfig.from_mapping(payload)
    return SACContractAwareAdapter(config)


def _status_from_summary(summary) -> str:
    if summary.termination_reason in {"catastrophic_error", "sla_violation", "zero_throughput"}:
        return "review"
    return "success"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _episode_log_payload(
    *,
    episode_id: str,
    task_id: str,
    seed: int,
    trajectory: List[Dict[str, Any]],
    info_history: List[Dict[str, Any]],
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    episode_log = EpisodeLog(
        metadata=EpisodeMetadata(
            episode_id=episode_id,
            task_id=task_id,
            robot_family="dishwasher_sac",
            seed=seed,
            env_params={"config": {"topology_type": "dishwashing_online_sac"}},
        ),
        trajectory=trajectory,
        info_history=info_history,
        metrics=metrics,
    )
    return episode_log.to_dict()


def _online_receipt_row(
    *,
    episode_id: str,
    run_id: str,
    summary,
    total_scalar_reward: float,
    sampling_recommendation: Dict[str, Any],
) -> Dict[str, Any]:
    predicted_value = float(summary.profit + 0.05 * summary.throughput_units_per_hour)
    realized_value = float(summary.profit)
    pricing_accepted = realized_value >= -0.1
    objective_satisfied = summary.error_rate_episode <= 0.12 and summary.termination_reason != "catastrophic_error"
    failure_events: List[str] = []
    if summary.termination_reason in {"catastrophic_error", "sla_violation", "zero_throughput"}:
        failure_events.append(summary.termination_reason)
    risk_events = ["high_error_rate"] if summary.error_rate_episode > 0.12 else []
    incident_events = list(failure_events) + list(risk_events)
    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "source_domain": "training_run",
        "predicted_value": predicted_value,
        "realized_value": realized_value,
        "quoted_rate": float(max(0.0, predicted_value)),
        "accepted_rate": float(max(0.0, realized_value)),
        "pricing_accepted": pricing_accepted,
        "pricing_reasons": [
            "training_run_accept" if pricing_accepted else "training_run_reject",
        ],
        "task_success": summary.termination_reason not in {"catastrophic_error", "zero_throughput"},
        "objective_satisfied": objective_satisfied,
        "realized_reward": float(total_scalar_reward),
        "failure_events": failure_events,
        "risk_events": risk_events,
        "incident_events": incident_events,
        "expected_adaptation_benefit": float(max(0.0, predicted_value * 0.1)),
        "realized_adaptation_benefit": float(max(0.0, realized_value * 0.1)),
        "adaptation_compute_cost": float(max(0.01, summary.energy_Wh * 0.01)),
        "adaptation_risk_cost": float(len(incident_events) * 0.1),
        "adaptation_review_required": bool(incident_events),
        "marginal_frontier_gain_predicted": float(max(0.0, sampling_recommendation.get("priority_score", 0.0) - 0.35)),
        "marginal_frontier_gain_realized": float(max(0.0, realized_value) * 0.01),
        "data_share_credit_predicted": float(max(0.0, predicted_value) * 0.02),
        "data_share_credit_realized": float(max(0.0, realized_value) * 0.02),
        "downweight_recommended": bool(summary.error_rate_episode > 0.12),
        "human_review_label": "needs_review" if incident_events else "pass",
        "override_label": None,
    }


def _descriptor_from_receipt_row(row: Dict[str, Any]) -> Dict[str, Any]:
    priority_score = float(row["sampling_recommendation"]["priority_score"])
    return {
        "pack_id": row["episode_id"],
        "episode_id": row["episode_id"],
        "env_name": "dishwashing_online_sac",
        "task_type": "dishwashing_online_sac",
        "backend": "training_run",
        "engine_type": "dishwashing_online_sac",
        "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
        "tier": 1 if row["task_success"] else 0,
        "trust_score": max(0.1, min(1.0, 1.0 - row["summary"]["error_rate_episode"])),
        "sampling_weight": max(0.1, 1.0 + priority_score),
        "episode_length": int(row["summary"]["steps"]),
        "semantic_tags": list(row["sampling_recommendation"]["queue_tags"]),
        "focus_areas": ["online_sac"],
        "priority": row["sampling_recommendation"]["priority_label"],
        "delta_J": float(row["receipt"]["realized_value"]),
        "w_embodiment": 1.0,
        "w_epi": float(max(0.0, row["receipt"]["realized_value"])),
        "metadata": {
            "summary": dict(row["summary"]),
            "receipt": dict(row["receipt"]),
        },
    }


def _advisory_episode_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    recommendation = dict(row["sampling_recommendation"])
    receipt = dict(row["receipt"])
    deploy_recommendation = "allow_shadow" if receipt["task_success"] else "require_review"
    datapack_recommendation = "downweight" if receipt["downweight_recommended"] else "keep"
    pricing_recommendation = "publish" if receipt["pricing_accepted"] else "publish_discounted"
    return {
        "episode_id": row["episode_id"],
        "sampling_priority_score": recommendation["priority_score"],
        "replay_queue_tags": recommendation["queue_tags"],
        "replay_action": recommendation["replay_action"],
        "deploy_recommendation": deploy_recommendation,
        "pricing_recommendation": pricing_recommendation,
        "datapack_recommendation": datapack_recommendation,
        "sampling_recommendation": recommendation,
        "receipt_feedback": {
            "deployment_outcome": {
                "task_success": receipt["task_success"],
                "objective_satisfied": receipt["objective_satisfied"],
                "realized_value": receipt["realized_value"],
                "pricing_accepted": receipt["pricing_accepted"],
            }
        },
    }


@regal_training(env_type="workcell")
def main(runner=None, _wrapped_args=None):
    """Main training function with canonical online SAC artifacts."""
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(runner.output_dir if runner is not None else args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    episode_logs_dir = output_root / "online_episode_logs"
    dataset_dir = output_root / "online_replay_dataset"
    queue_history_path = output_root / "queue_dispatch_history.jsonl"
    queue_latest_path = output_root / "queue_dispatch_comparison.json"
    sampler_policy_receipt_path = output_root / "sampler_policy_receipt.json"
    live_queue_history_path = output_root / "live_queue_selection_history.jsonl"
    live_queue_latest_path = output_root / "live_queue_selection.json"
    episode_receipts_path = output_root / "online_episode_receipts.jsonl"
    metrics_path = output_root / "online_sac_metrics.jsonl"

    econ_params = EconParams(
        price_per_unit=0.3,
        damage_cost=1.0,
        energy_Wh_per_attempt=0.05,
        time_step_s=60.0,
        base_rate=2.0,
        p_min=0.02,
        k_err=0.12,
        q_speed=1.2,
        q_care=1.5,
        care_cost=0.25,
        max_steps=240,
        max_catastrophic_errors=3,
        max_error_rate_sla=0.12,
        min_steps_for_sla=5,
        zero_throughput_patience=10,
        preset="toy",
    )
    env = DishwashingEnv(econ_params)
    obs_dim = 4
    latent_dim = 128
    encoder = EncoderWithAuxiliaries(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=256,
        use_consistency=True,
        use_contrastive=True,
    )
    contract_aware_adapter = _contract_aware_adapter(
        args=args,
        output_root=output_root,
        latent_dim=latent_dim,
        action_dim=2,
    )
    agent = SACAgent(
        encoder=encoder,
        latent_dim=latent_dim,
        action_dim=2,
        lr=3e-4,
        gamma=0.995,
        tau=5e-3,
        buffer_capacity=int(args.buffer_capacity),
        batch_size=int(args.batch_size),
        target_entropy=-2.0,
        device="cpu",
        contract_aware_adapter=contract_aware_adapter,
        sampling_artifact_dir=str(output_root / "online_sampling"),
        sampling_log_interval=1,
    )

    store = OntologyStore(root_dir=args.ontology_root)
    task = Task(
        task_id=args.task_id,
        name="Dishwashing",
        description="Online SAC dishwashing task",
        environment_id="dishwashing_env",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=18.0,
        default_energy_cost_per_wh=0.12,
    )
    robot = Robot(robot_id=args.robot_id, name="DishwasherBot")
    store.upsert_task(task)
    store.upsert_robot(robot)

    objective_profile = _objective_profile(args.objective_scalarizer)
    reward_engine = RewardEngine(
        task,
        robot,
        config={},
        econ_domain_name=args.econ_domain,
        objective_profile=objective_profile,
    )
    objective_econ_functor = ObjectiveEconFunctor()
    logger = EpisodeLogger(store=store, task=task, robot=robot)
    promotion_policy = load_regal_promotion_policy(args.promotion_policy)

    training_run_id = runner.run_id if runner is not None else f"online_sac_{args.seed}"
    episode_rows: List[Dict[str, Any]] = []
    descriptors: List[Dict[str, Any]] = []
    advisory_episodes: List[Dict[str, Any]] = []
    update_count = 0
    total_steps = 0

    for ep_idx in range(args.episodes):
        episode = logger.start_episode(
            metadata={
                "episode_index": ep_idx,
                "episode_id": f"{training_run_id}_ep_{ep_idx:04d}",
            }
        )
        obs = env.reset()
        done = False
        timestep = 0
        trajectory: List[Dict[str, Any]] = []
        info_history: List[Dict[str, Any]] = []
        total_scalar_reward = 0.0
        last_update_metrics: Dict[str, Any] = {}

        while not done:
            action, _ = agent.select_action(obs, novelty=0.5)
            next_obs, info, done = env.step(action)
            info = dict(info or {})
            raw_reward = float(info.get("profit_step", 0.0))
            scalar_reward, components = reward_engine.step_reward(raw_reward, info)
            total_scalar_reward += float(scalar_reward)
            logger.log_step(
                timestep=timestep,
                reward_scalar=scalar_reward,
                reward_components=components,
                state_summary={"obs": obs, "next_obs": next_obs},
                metadata={"env_info": dict(info)},
            )
            trajectory.append(
                {
                    "step": timestep,
                    "obs": dict(obs),
                    "action": {
                        "speed": float(action[0]),
                        "care": float(action[1]),
                    },
                    "done": bool(done),
                    "info": dict(info),
                }
            )
            info_history.append(dict(info))
            agent.store_transition(
                obs,
                action,
                scalar_reward,
                next_obs,
                done,
                novelty=0.5,
                episode_id=episode.episode_id,
                source_domain="training_run",
                metadata={"episode_index": ep_idx, "timestep": timestep},
            )
            obs = next_obs
            timestep += 1
            total_steps += 1

            if total_steps >= int(args.learning_starts):
                for _ in range(max(1, int(args.updates_per_step))):
                    last_update_metrics = agent.update()
                    if last_update_metrics:
                        update_count += 1
                        _append_jsonl(metrics_path, last_update_metrics)

        summary = summarize_episode_info(info_history)
        status = _status_from_summary(summary)
        econ = reward_engine.compute_econ_vector(episode, logger._events)
        objective_tensor = reward_engine.compute_objective_tensor_from_events(episode, logger._events)
        econ_tensor = objective_econ_functor.map(
            objective_tensor,
            constraint_flags=[],
            uncertainty=0.0,
            context={"episode_id": episode.episode_id},
        )
        logger.mark_outcome(status=status, metadata={"termination_reason": summary.termination_reason})
        logger.finalize(
            econ_vector=econ,
            econ_tensor=econ_tensor,
            objective_tensor=objective_tensor,
        )

        policy_uncertainty = float(last_update_metrics.get("alpha", 0.1)) / 2.0 if last_update_metrics else 0.1
        sampling_recommendation = recommend_sampling(
            objective_profile_coverage_gap=max(0.0, summary.error_rate_episode),
            constraint_violation_count=len([event for event in [summary.termination_reason] if event in {"catastrophic_error", "sla_violation"}]),
            uncertainty=min(1.0, policy_uncertainty),
            datapack_value=max(0.0, summary.profit),
            regal_support_score=min(1.0, summary.error_rate_episode + 0.1 * (summary.energy_Wh_per_unit > 0.2)),
            deploy_recommendation="allow_shadow" if status == "success" else "require_review",
            pricing_recommendation="publish" if summary.profit >= 0.0 else "publish_discounted",
            datapack_recommendation="downweight" if summary.error_rate_episode > 0.12 else "keep",
            promotion_policy=promotion_policy,
            replay_policy_error=float(last_update_metrics.get("critic_loss", 0.0) if last_update_metrics else 0.0),
            provenance_quality=max(0.1, 1.0 - summary.error_rate_episode),
        ).to_dict()

        receipt_row = _online_receipt_row(
            episode_id=episode.episode_id,
            run_id=training_run_id,
            summary=summary,
            total_scalar_reward=total_scalar_reward,
            sampling_recommendation=sampling_recommendation,
        )
        log_payload = _episode_log_payload(
            episode_id=episode.episode_id,
            task_id=args.task_id,
            seed=args.seed,
            trajectory=trajectory,
            info_history=info_history,
            metrics={
                "reward_total": float(total_scalar_reward),
                "steps": float(timestep),
                "time_step_s": float(econ_params.time_step_s),
                "energy_wh_per_unit": float(summary.energy_Wh_per_unit),
            },
        )
        episode_log_path = episode_logs_dir / f"{episode.episode_id}.json"
        _write_json(episode_log_path, log_payload)
        _append_jsonl(episode_receipts_path, receipt_row)

        row = {
            "episode_id": episode.episode_id,
            "summary": {
                "termination_reason": summary.termination_reason,
                "mpl_episode": float(summary.mpl_episode),
                "error_rate_episode": float(summary.error_rate_episode),
                "energy_Wh": float(summary.energy_Wh),
                "energy_Wh_per_unit": float(summary.energy_Wh_per_unit),
                "profit": float(summary.profit),
                "steps": int(timestep),
            },
            "receipt": receipt_row,
            "sampling_recommendation": sampling_recommendation,
            "task_success": bool(receipt_row["task_success"]),
        }
        episode_rows.append(row)
        descriptors.append(_descriptor_from_receipt_row(row))
        advisory_episodes.append(_advisory_episode_from_row(row))
        agent.attach_receipt_feedback(
            {
                episode.episode_id: {
                    "deployment_outcome": {
                        "task_success": receipt_row["task_success"],
                        "objective_satisfied": receipt_row["objective_satisfied"],
                        "realized_value": receipt_row["realized_value"],
                        "pricing_accepted": receipt_row["pricing_accepted"],
                    }
                }
            }
        )

        live_queue_selection = build_live_queue_selection(
            {"episodes": advisory_episodes},
            queue_name="online_sac_queue",
        )
        sampler = DataPackRLSampler(
            existing_descriptors=descriptors,
            live_queue_selection=live_queue_selection,
            queue_dispatch_mode=args.queue_selection_mode,
            queue_max_upweight=args.queue_max_upweight,
            queue_max_downweight=args.queue_max_downweight,
            queue_allow_slice_removal_on_integrity_failure=args.queue_allow_slice_removal_on_integrity_failure,
            queue_policy_helper_mode=args.queue_policy_helper_mode,
            queue_policy_package_path=args.queue_policy_package_path,
            sampler_policy_helper_mode=args.sampler_policy_helper_mode,
            sampler_policy_package_path=args.sampler_policy_package_path,
        )
        dispatch = sampler.dispatch_queue(
            batch_size=len(descriptors),
            seed=args.seed,
            strategy="balanced",
        )
        agent.apply_queue_dispatch(dispatch)
        _write_json(live_queue_latest_path, live_queue_selection)
        _append_jsonl(live_queue_history_path, live_queue_selection)
        _write_json(queue_latest_path, dispatch)
        _write_json(
            sampler_policy_receipt_path,
            dict(dispatch.get("sampler_policy_receipt", sampler.last_sampler_policy_artifact or {})),
        )
        _append_jsonl(queue_history_path, dispatch)

    dataset_bundle = ReplayDatasetBuilder()
    for path in sorted(episode_logs_dir.glob("*.json")):
        dataset_bundle.add_workcell_episode_log(
            path,
            run_id=training_run_id,
            source_domain="training_run",
            objective_profile_id="balanced_contract",
        )
    dataset = dataset_bundle.write(dataset_dir)
    receipt_bundle = build_training_run_receipt_label_bundle(
        output_root,
        replay_dataset_dir=dataset.root_dir,
        label_mode=args.receipt_label_mode,
    )
    receipt_paths = write_receipt_label_bundle(receipt_bundle, output_root / "receipt_labels")
    advisory = build_shadow_advisory_output(
        replay_dataset_dir=str(dataset.root_dir),
        promotion_policy_path=args.promotion_policy,
        receipt_label_dir=str(Path(receipt_paths["bundle"]).parent),
        receipt_label_mode=args.receipt_label_mode,
    )
    advisory_path = output_root / "online_shadow_advisory.json"
    scorer_preconditions_path = output_root / "semantic_runtime_scorer_preconditions.json"
    scorer_work_orders_path = output_root / "semantic_runtime_scorer_work_orders.json"
    inferential_summary_path = output_root / "inferential_learnability_summary.json"
    inferential_work_orders_path = output_root / "inferential_work_orders.json"
    _write_json(advisory_path, advisory)
    _write_json(scorer_preconditions_path, advisory["semantic_runtime_scorer_preconditions"])
    _write_json(scorer_work_orders_path, {"work_orders": advisory["semantic_runtime_scorer_work_orders"]})
    _write_json(inferential_summary_path, advisory["inferential_learnability_summary"])
    _write_json(inferential_work_orders_path, {"work_orders": advisory["inferential_work_orders"]})
    report = build_promotion_evidence_report(
        dataset=dataset,
        promotion_policy=promotion_policy,
        receipt_bundle=receipt_bundle,
        evidence_pointers={
            "live_queue_selection": str(live_queue_latest_path),
            "queue_dispatch_comparison": str(queue_latest_path),
            "sampler_policy_receipt": str(sampler_policy_receipt_path),
            "online_shadow_advisory": str(advisory_path),
        },
    )
    promotion_paths = write_promotion_evidence_report(output_root, report)

    checkpoint_path = output_root / "sac_online_agent.pt"
    agent.save(checkpoint_path)

    if runner is not None:
        runner.set_eligible_datapacks([episode.episode_id for episode in dataset.episodes])
        runner.set_sampler_config(
            seed=args.seed,
            config_sha=sha256_json(
                {
                    "queue_selection_mode": args.queue_selection_mode,
                    "queue_max_upweight": args.queue_max_upweight,
                    "queue_max_downweight": args.queue_max_downweight,
                    "contract_aware_mode": args.contract_aware_mode,
                }
            ),
        )
        for episode in dataset.episodes:
            runner.record_sample(episode.task_id, datapack_id=episode.episode_id, slice_id=episode.episode_id)
        for audit in build_replay_trajectory_audits(dataset):
            runner.add_trajectory_audit(audit)
        runner.update_step(total_steps)
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "promotion_summary": report.summary,
                "receipt_label_coverage": receipt_bundle.coverage_summary(),
            },
            context_sha=promotion_policy.config_digest,
        )
        runner.configure_training_runtime(
            training_kind="online_sac",
            config_path=args.contract_aware_config,
            config_digest=sha256_json(
                {
                    "objective_scalarizer": args.objective_scalarizer,
                    "queue_selection_mode": args.queue_selection_mode,
                    "contract_aware_mode": args.contract_aware_mode,
                }
            ),
            replay_dataset_dir=dataset.root_dir,
            replay_manifest_digest=dataset.manifest.manifest_hash,
            replay_dataset_summary=build_replay_dataset_summary(dataset),
            objective_profile_snapshot=(
                objective_profile.to_dict()
                if objective_profile is not None
                else {"profile_id": "legacy"}
            ),
            promotion_policy_snapshot=promotion_policy.to_dict(),
            source_domain_coverage=build_source_domain_coverage(dataset),
            receipt_label_coverage=receipt_bundle.coverage_summary(),
            inferential_learnability_summary=dict(
                advisory.get("inferential_learnability_summary", {}) or {}
            ),
            inferential_work_order_summary=dict(
                advisory.get("adaptation_budget", {}).get("summary", {}) or {}
            ),
            artifact_schema_compatibility=list(dataset.manifest.metadata.get("schema_compatibility", []) or []),
            metadata={
                "update_count": update_count,
                "queue_selection_mode": args.queue_selection_mode,
                "contract_aware_mode": args.contract_aware_mode,
            },
        )
        runner.register_artifact("online_episode_logs", episode_logs_dir)
        runner.register_artifact("online_episode_receipts", episode_receipts_path)
        runner.register_artifact("live_queue_selection", live_queue_latest_path)
        runner.register_artifact("queue_dispatch_comparison", queue_latest_path)
        runner.register_artifact("sampler_policy_receipt", sampler_policy_receipt_path)
        runner.register_artifact("online_shadow_advisory", advisory_path)
        runner.register_artifact("semantic_runtime_scorer_preconditions", scorer_preconditions_path)
        runner.register_artifact("semantic_runtime_scorer_work_orders", scorer_work_orders_path)
        runner.register_artifact("inferential_learnability_summary", inferential_summary_path)
        runner.register_artifact("inferential_work_orders", inferential_work_orders_path)
        runner.register_artifact("receipt_label_bundle", receipt_paths["bundle"])
        runner.register_artifact("receipt_label_summary", receipt_paths["summary"])
        runner.register_artifact("regal_promotion_eval", promotion_paths["json"])
        runner.register_artifact("regal_promotion_eval_markdown", promotion_paths["markdown"])
        runner.register_artifact("online_replay_dataset_manifest", Path(dataset.root_dir) / "manifest.json")
        runner.register_artifact("online_sac_metrics", metrics_path)
        sampling_artifact = output_root / "online_sampling" / "online_sac_sampling.jsonl"
        if sampling_artifact.exists():
            runner.register_artifact("online_sac_sampling", sampling_artifact)
        contract_metrics_path = output_root / "contract_aware" / "sac_contract_aware_metrics.jsonl"
        if contract_metrics_path.exists():
            runner.register_artifact("contract_aware_metrics", contract_metrics_path)
        contract_predictions_path = output_root / "contract_aware" / "sac_contract_aware_predictions.jsonl"
        if contract_predictions_path.exists():
            runner.register_artifact("contract_aware_predictions", contract_predictions_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="online_sac_latest",
                model_family="online_sac",
                model_version="sac_online_v1",
                path=checkpoint_path,
                step=total_steps,
                epoch=args.episodes,
                metadata={
                    "queue_selection_mode": args.queue_selection_mode,
                    "contract_aware_mode": args.contract_aware_mode,
                },
            )
        )

    print(
        f"[train_sac_with_ontology_logging] episodes={args.episodes} "
        f"updates={update_count} output_dir={output_root}"
    )


if __name__ == "__main__":
    main()
