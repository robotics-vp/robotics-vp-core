"""WM-owned synthetic-branch planning and local branch-collection helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from src.economics.inferential_contract import coerce_inferential_learnability_contract
from src.evidence.scene_tracks_truth import scene_tracks_truth_from_metadata

from .branch_planner_runtime import resolve_branch_planner_helper
from .common import clip01, mapping, stable_id
from .inferential import (
    adjusted_branch_yield_score,
    build_branch_plan_inferential_contract,
)
from .promotion import HelperMode, infer_branch_payload
from .render_providers import compile_branch_render_provider_state
from .state import (
    PhysicsAdaptationPolicyState,
    PhysicsContextState,
    SyntheticBranchPlan,
)


def extract_branch_features(z_sequence: torch.Tensor) -> torch.Tensor:
    """Extract trust-net features for a synthetic latent trajectory."""

    global_mean = z_sequence.mean()
    global_std = z_sequence.std()
    global_min = z_sequence.min()
    global_max = z_sequence.max()
    dim_var = z_sequence.mean(dim=1).std()
    diffs = torch.abs(z_sequence[1:] - z_sequence[:-1])
    smoothness = diffs.mean()
    return torch.stack([global_mean, global_std, global_min, global_max, dim_var, smoothness])


def compute_branch_gap_labels(
    coverage_graph: Any,
    *,
    task_id: str = "",
    env_id: str = "",
) -> dict[str, Any]:
    labels = {
        "skill_edge": "",
        "env_primitive_edge": "",
        "risk_family": "",
        "coverage_gap_contribution": 0.0,
        "economic_priority": 0.0,
    }
    if coverage_graph is None:
        return labels
    try:
        gaps = coverage_graph.rank_gaps(limit=50)
        if not gaps:
            return labels
        best = gaps[0]
        for gap in gaps:
            if task_id and task_id in str(getattr(gap, "source_id", "")):
                best = gap
                break
            if env_id and env_id in str(getattr(gap, "target_id", "")):
                best = gap
                break
        labels["skill_edge"] = f"{best.source_id} -> {best.target_id}"
        labels["coverage_gap_contribution"] = (
            best.gap_score() if callable(getattr(best, "gap_score", None)) else 0.0
        )
        labels["economic_priority"] = getattr(best, "economic_priority", 0.0)
    except Exception:
        return labels
    return labels


def collect_local_synthetic_branch_records(
    *,
    episodes: Sequence[Mapping[str, Any]],
    world_model: Any,
    trust_net: Any,
    trust_mean: torch.Tensor,
    trust_std_norm: torch.Tensor,
    real_z_std: float,
    horizon: int,
    branches_per_episode: int,
    min_trust: float,
    min_std_ratio: float,
    max_std_ratio: float,
    objective_vector: Sequence[float],
    coverage_graph: Any = None,
    brick_manifest: Any = None,
    brick_id_fn: Optional[Any] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branches: list[dict[str, Any]] = []
    stats = {
        "total_attempted": 0,
        "passed_trust": 0,
        "passed_std": 0,
        "passed_all": 0,
        "by_brick": {},
    }

    objective_array = np.asarray(objective_vector, dtype=np.float32)
    resolve_brick_id = brick_id_fn or (lambda ep_idx, manifest: -1)

    for ep_idx, episode in enumerate(episodes):
        z_real = episode["z_sequence"]
        actions = episode["actions"]
        length = int(episode["length"])
        task_id = str(episode.get("task_id", "") or "")
        env_id = str(episode.get("env_id", "") or "")
        brick_id = int(resolve_brick_id(ep_idx, brick_manifest))

        if brick_id not in stats["by_brick"]:
            stats["by_brick"][brick_id] = {"attempted": 0, "passed": 0}
        if length <= horizon:
            continue

        max_start = length - horizon
        start_positions = np.random.choice(
            max_start,
            size=min(branches_per_episode, max_start),
            replace=False,
        )
        for start_t in start_positions:
            stats["total_attempted"] += 1
            stats["by_brick"][brick_id]["attempted"] += 1
            z_init = z_real[start_t]
            actions_segment = actions[start_t : start_t + horizon]
            with torch.no_grad():
                z_traj = world_model.rollout(z_init, actions_segment)
            features = extract_branch_features(z_traj)
            feat_norm = (features - trust_mean) / trust_std_norm
            trust_score = float(trust_net(feat_norm.unsqueeze(0)).item())
            synth_std = float(z_traj.std().item())
            std_ratio = float(synth_std / max(real_z_std, 1e-8))

            if trust_score < min_trust:
                continue
            stats["passed_trust"] += 1
            if std_ratio < min_std_ratio or std_ratio > max_std_ratio:
                continue
            stats["passed_std"] += 1
            stats["passed_all"] += 1
            stats["by_brick"][brick_id]["passed"] += 1

            gap_labels = compute_branch_gap_labels(
                coverage_graph,
                task_id=task_id,
                env_id=env_id,
            )
            gap_score = float(gap_labels.get("coverage_gap_contribution", 0.0))
            branch_value = trust_score * max(0.01, gap_score) if coverage_graph else trust_score
            branches.append(
                {
                    "z_sequence": z_traj.detach().cpu().numpy(),
                    "actions": actions_segment.detach().cpu().numpy(),
                    "source_episode": ep_idx,
                    "source_timestep": int(start_t),
                    "horizon": horizon,
                    "trust_score": trust_score,
                    "std_ratio": std_ratio,
                    "brick_id": brick_id,
                    "objective_vector": objective_array,
                    "branch_value": branch_value,
                    "gap_labels": gap_labels,
                }
            )
    return branches, stats


def build_synthetic_branch_corpus_metadata(
    *,
    output_path: str,
    world_model_path: str,
    dataset_path: str,
    horizon: int,
    branches_per_episode: int,
    objective_dim: int,
    min_trust: float,
    min_std_ratio: float,
    max_std_ratio: float,
    stats: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
    coverage_graph_used: bool,
    coverage_graph_path: Optional[str],
    source_runtime_metadata: Optional[Mapping[str, Any]],
    source_runtime_metadata_artifact: Optional[str],
    scene_tracks_backend: str,
    teacher_runtime_backend_selected: str,
    vision_backbone_selected: str,
    semantic_grounding_mode: str,
    semantic_memory_grounded: bool,
    gap_labels_path: Optional[str],
    gen2sim_validity_path: Optional[str],
    gen2sim_summary: Mapping[str, Any],
) -> dict[str, Any]:
    trust_scores = np.array([float(branch.get("trust_score", 0.0)) for branch in branches], dtype=np.float32)
    std_ratios = np.array([float(branch.get("std_ratio", 0.0)) for branch in branches], dtype=np.float32)
    source_payload = dict(source_runtime_metadata or {})
    scene_truth = scene_tracks_truth_from_metadata(
        {
            **source_payload,
            "scene_tracks_backend": scene_tracks_backend,
        }
    )
    return {
        "schema_version": "synthetic_branch_corpus_metadata_v1",
        "source_type": "stable_world_model_local_branch_v1",
        "world_model": world_model_path,
        "dataset": dataset_path,
        "horizon": horizon,
        "branches_per_episode": branches_per_episode,
        "objective_dim": objective_dim,
        "min_trust": min_trust,
        "min_std_ratio": min_std_ratio,
        "max_std_ratio": max_std_ratio,
        "total_attempted": int(stats.get("total_attempted", 0)),
        "passed_trust": int(stats.get("passed_trust", 0)),
        "passed_std": int(stats.get("passed_std", 0)),
        "final_branches": len(branches),
        "pass_rate": 100 * len(branches) / max(1, int(stats.get("total_attempted", 0))),
        "avg_trust": float(trust_scores.mean()) if len(trust_scores) else 0.0,
        "avg_std_ratio": float(std_ratios.mean()) if len(std_ratios) else 0.0,
        "by_brick": dict(stats.get("by_brick", {})),
        "coverage_graph_used": bool(coverage_graph_used),
        "coverage_graph_path": coverage_graph_path if coverage_graph_used else None,
        "gap_label_sample": branches[0].get("gap_labels") if branches else None,
        "gen2sim_validity_summary": mapping(gen2sim_summary),
        "scene_tracks_backend": scene_tracks_backend,
        "teacher_runtime_backend_selected": teacher_runtime_backend_selected,
        "vision_backbone_selected": vision_backbone_selected,
        "semantic_grounding_mode": semantic_grounding_mode,
        "semantic_memory_grounded": bool(semantic_memory_grounded),
        "future_training_signals": {
            **dict(source_payload.get("future_training_signals", {}) or {}),
            **{
                key: value
                for key, value in scene_truth.items()
                if key
                in {
                    "scene_tracks_non_stub",
                    "scene_tracks_training_eligible",
                    "semantic_grounding_non_heuristic",
                    "semantic_grounding_ready",
                }
            },
            "semantic_gap_labeled": bool(gap_labels_path),
            "semantic_memory_grounded": bool(semantic_memory_grounded),
        },
        "future_training_artifacts": {
            "branch_gap_labels": gap_labels_path,
            "branch_gen2sim_validity": gen2sim_validity_path,
            "source_runtime_metadata": source_runtime_metadata_artifact,
        },
    }


def heuristic_generation_mode(job: Any, physics_context: PhysicsContextState) -> str:
    if str(getattr(job, "data_collection_intent", "")) == "validate":
        return "physics_probe"
    if str(physics_context.fidelity_tier) == "high_fidelity":
        return "geometry_guarded_rollout"
    if str(getattr(job, "data_collection_intent", "")) == "exploit":
        return "targeted_synth_rollout"
    return "coverage_branch"


def heuristic_yield_score(job: Any) -> float:
    return clip01(
        (0.4 * clip01(getattr(job, "coverage_gap_score", 0.0)))
        + (0.35 * clip01(getattr(job, "economic_priority", 0.0)))
        + (0.15 * (1.0 - clip01(getattr(job, "trust_priority", 0.0))))
        + (0.10 * clip01(getattr(job, "readiness", 0.0)))
    )


def compile_synthetic_branch_plans(
    jobs: Sequence[Any],
    *,
    physics_context: PhysicsContextState,
    physics_adaptation_policy: PhysicsAdaptationPolicyState,
    benchmark_signals: Mapping[str, Any],
    semantic_context: Optional[Mapping[str, Any]],
    economic_context: Optional[Mapping[str, Any]],
    branch_planner: Any,
    branch_planner_mode: HelperMode,
) -> list[SyntheticBranchPlan]:
    helper, helper_status = resolve_branch_planner_helper(
        branch_planner,
        mode=branch_planner_mode,
    )
    plans: list[SyntheticBranchPlan] = []
    for job in jobs:
        heuristic_mode = heuristic_generation_mode(job, physics_context)
        heuristic_score = heuristic_yield_score(job)
        helper_payload = infer_branch_payload(
            helper,
            job=job.to_dict(),
            context={
                "job": job.to_dict(),
                "physics_context": physics_context.to_dict(),
                "benchmark_signals": mapping(benchmark_signals),
                "heuristic_generation_mode": heuristic_mode,
            },
        )
        selection_policy = "heuristic_only"
        generation_mode = heuristic_mode
        expected_yield_score = heuristic_score
        if helper_payload:
            selection_policy = "heuristic_plus_learned_branch_planner"
            if str(helper_status.get("promotion_stage")) == "promoted":
                generation_mode = str(helper_payload.get("generation_mode") or heuristic_mode)
                expected_yield_score = clip01(
                    helper_payload.get("expected_yield_score", heuristic_score)
                )
        job_contract = coerce_inferential_learnability_contract(
            job.inferential_learnability_contract
        )
        plan_payload = {
            "job_id": job.job_id,
            "branch_family": f"{job.task_family}:{job.data_collection_intent}",
            "generation_mode": generation_mode,
            "render_backend": physics_context.backend,
        }
        plan_id = stable_id("branch_plan", plan_payload)
        branch_contract = build_branch_plan_inferential_contract(
            plan_id=plan_id,
            job_id=job.job_id,
            expected_yield_score=expected_yield_score,
            job_contract=job_contract,
            benchmark_signals=benchmark_signals,
            semantic_context=semantic_context,
            economic_context=economic_context,
        )
        inferential_signal_score = clip01(branch_contract.signal_yield.get("score", 0.0))
        inferential_replay_weight = clip01(branch_contract.inferential_replay_weight)
        expected_yield_score = adjusted_branch_yield_score(
            base_expected_yield_score=expected_yield_score,
            contract=branch_contract,
        )
        admission_preconditions = {
            "requires_non_heuristic_grounding": bool(
                job.data_collection_intent == "validate" and bool(job.risk_family)
            ),
            "requires_benchmark_ready": bool(job.readiness >= 0.8 and job.economic_priority >= 0.8),
            "min_readiness": 0.0,
            "min_inferential_replay_weight": (
                0.08 if job.data_collection_intent == "validate" else 0.04
            ),
        }
        plans.append(
            SyntheticBranchPlan(
                plan_id=plan_id,
                source_job_id=job.job_id,
                branch_family=f"{job.task_family}:{job.data_collection_intent}",
                generation_mode=generation_mode,
                render_backend=physics_context.backend,
                gap_target_refs=[mapping(job.coverage_targets)],
                admission_preconditions=admission_preconditions,
                expected_yield_score=expected_yield_score,
                selection_policy=selection_policy,
                render_provider=compile_branch_render_provider_state(
                    branch_plan_id=plan_id,
                    generation_mode=generation_mode,
                    branch_family=f"{job.task_family}:{job.data_collection_intent}",
                    physics_context=physics_context,
                    physics_adaptation_policy=physics_adaptation_policy,
                    benchmark_signals=benchmark_signals,
                ),
                inferential_learnability_contract=branch_contract.to_dict(),
                metadata={
                    "agenda_rank": job.rank,
                    "source_ranking_policy": job.ranking_policy,
                    "heuristic_generation_mode": heuristic_mode,
                    "heuristic_expected_yield_score": heuristic_score,
                    "inferential_signal_yield_score": inferential_signal_score,
                    "inferential_replay_weight": inferential_replay_weight,
                    "branch_helper_status": helper_status,
                    "branch_helper_trace": helper_payload,
                },
            )
        )
    return plans


__all__ = [
    "build_synthetic_branch_corpus_metadata",
    "collect_local_synthetic_branch_records",
    "compile_synthetic_branch_plans",
    "compute_branch_gap_labels",
    "extract_branch_features",
    "heuristic_generation_mode",
    "heuristic_yield_score",
]
