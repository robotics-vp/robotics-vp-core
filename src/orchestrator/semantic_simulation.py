from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Literal

from src.config.objective_profile import ObjectiveVector
from src.evidence.benchmark_gating import collect_benchmark_gating_signals
from src.evidence.preconditions import build_execution_preconditions
from src.economics.econ_meter import EconomicMeter
from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.datapacks import (
    DatapackConfig,
    datapack_configs_from_ontology,
    load_datapack_configs,
    save_datapack_config,
)
from src.motor_backend.factory import make_motor_backend
from src.motor_backend.rollout_capture import RolloutBundle, finalize_rollout_bundle
from src.objectives.economic_objective import EconomicObjectiveSpec, load_economic_objective_spec
from src.ontology.datapack_registry import register_datapack_configs
from src.ontology.query import find_datapacks, find_scenarios
from src.ontology.store import OntologyStore
from src.orchestrator.schedule import BudgetExceeded, acquire_run_budget, release_run_budget
from src.orchestrator.semantic_policy import (
    DatapackSelectionDecision,
    DatapackSelectionScorerPackage,
    MissingScenarioSpec,
    apply_arh_penalty,
    coerce_datapack_selection_scorer_package,
    detect_semantic_gaps,
    load_datapack_selection_scorer_package,
    rank_datapacks_for_intent,
    select_datapacks_for_intent,
    summarize_datapack_selection,
)
from src.orchestrator.semantic_fusion_runner import run_semantic_fusion_for_rollouts
from src.scenarios.metadata import ScenarioMetadata, build_scenario_metadata
from src.vla.rollout_labeler import label_rollouts_with_vla
from src.world_model.sim_synth_physics import (
    SimulationJobSpec,
    SimSynthPhysicsRuntime,
    SimSynthPhysicsRuntimeConfig,
)

if TYPE_CHECKING:
    from src.ontology.models import Robot


@dataclass(frozen=True)
class SemanticSimulationResult:
    scenario: ScenarioMetadata
    train_result: MotorTrainingResult
    eval_result: MotorEvalResult | None
    rollout_bundle: RolloutBundle | None
    labeled_datapacks: Sequence[DatapackConfig]
    missing_specs: Sequence[MissingScenarioSpec]
    selection_summary: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OrchestratedRunResult:
    status: Literal["completed", "deferred", "failed"]
    scenario: ScenarioMetadata | None
    simulation: SemanticSimulationResult | None = None
    reason: str | None = None


def run_semantic_simulation(
    *,
    store: OntologyStore,
    intent: str | None = None,
    tags: Sequence[str] | None,
    robot_family: str | None,
    objective_hint: str | None,
    notes: str | None = None,
    task_id: str | None = None,
    motor_backend: str = "holosoma",
    objective_config: str | None = None,
    datapack_limit: int = 1,
    num_envs: int = 1024,
    max_steps: int = 10000,
    eval_episodes: int = 0,
    seed: int | None = None,
    rollout_base_dir: str | Path | None = None,
    datapack_output_dir: str | Path = "configs/datapacks",
    robot_id: str = "robot_default",
    run_log_path: str | Path = "data/logs/semantic_runs.jsonl",
    selection_scorer_package: Mapping[str, Any] | None = None,
    selection_scorer_package_path: str | None = None,
    selection_scorer_mode: Literal["disabled", "auto", "required"] = "auto",
) -> OrchestratedRunResult:
    run_id = str(uuid.uuid4())
    estimated_steps = max(0, int(num_envs) * int(max_steps))

    try:
        acquire_run_budget(estimated_steps)
    except BudgetExceeded as exc:
        result = OrchestratedRunResult(status="deferred", scenario=None, reason=str(exc))
        _append_run_log(
            run_log_path,
            _build_run_log_payload(
                intent=intent,
                tags=tags,
                robot_family=robot_family,
                objective_hint=objective_hint,
                scenario=None,
                simulation=None,
                status=result.status,
                reason=result.reason,
                motor_backend=motor_backend,
                vla_mode="disabled",
            ),
        )
        return result

    scenario: ScenarioMetadata | None = None
    simulation: SemanticSimulationResult | None = None
    status: Literal["completed", "deferred", "failed"] = "failed"
    reason: str | None = None
    steps_used = 0
    vla_mode = "disabled"

    try:
        datapack_records = find_datapacks(
            store,
            tags=tags,
            robot_family=robot_family,
            objective_hint=objective_hint,
            task_id=task_id,
            limit=datapack_limit,
        )
        candidates = datapack_configs_from_ontology(datapack_records)
        candidate_metadata_by_id = _candidate_metadata_by_id(datapack_records)

        scenario_records = find_scenarios(
            store,
            datapack_tags=tags,
            robot_families=[robot_family] if robot_family else None,
            objective_name=objective_hint,
            motor_backend=motor_backend,
        )
        (
            resolved_selection_scorer_package,
            selection_scorer_package_ref,
            selection_helper_status,
        ) = _resolve_datapack_selection_scorer_package(
            selection_scorer_package=selection_scorer_package,
            selection_scorer_package_path=selection_scorer_package_path,
            selection_scorer_mode=selection_scorer_mode,
        )

        ranked_selected = rank_datapacks_for_intent(
            tags or [],
            robot_family,
            objective_hint,
            candidates,
            scenario_records,
            candidate_metadata_by_id=candidate_metadata_by_id,
            selection_scorer_package=resolved_selection_scorer_package,
            source="ontology",
        )
        selected = [row.datapack for row in ranked_selected]

        missing_specs = detect_semantic_gaps(tags or [], robot_family, scenario_records)

        fallback_ranked: list[DatapackSelectionDecision] = []
        if missing_specs:
            exploratory = _resolve_local_datapacks(
                tags=tags,
                robot_family=robot_family,
                objective_hint=objective_hint,
            )
            if exploratory:
                fallback_ranked = rank_datapacks_for_intent(
                    tags or [],
                    robot_family,
                    objective_hint,
                    exploratory,
                    scenario_records,
                    selection_scorer_package=resolved_selection_scorer_package,
                    source="local_yaml",
                )
        ranked_combined = _merge_ranked_datapacks(ranked_selected, fallback_ranked)
        if ranked_combined:
            selected = [row.datapack for row in ranked_combined]
        if not selected:
            fallback_only = _resolve_local_datapacks(
                tags=tags,
                robot_family=robot_family,
                objective_hint=objective_hint,
            )
            selected = select_datapacks_for_intent(
                tags or [],
                robot_family,
                objective_hint,
                fallback_only,
                scenario_records,
                selection_scorer_package=resolved_selection_scorer_package,
                source="local_yaml",
            )
            ranked_combined = rank_datapacks_for_intent(
                tags or [],
                robot_family,
                objective_hint,
                fallback_only,
                scenario_records,
                selection_scorer_package=resolved_selection_scorer_package,
                source="local_yaml",
            )

        if not selected:
            raise ValueError("No datapacks matched the requested semantic filters.")

        if datapack_limit:
            selected = list(selected)[: datapack_limit]
            ranked_combined = list(ranked_combined)[: datapack_limit]
        selection_summary = summarize_datapack_selection(
            ranked_combined,
            selected=ranked_combined[: len(selected)],
            tags=tags or [],
            robot_family=robot_family,
            objective_hint=objective_hint,
            selection_helper_status=selection_helper_status,
            selection_context=_selection_context_from_ranked(ranked_combined),
        )
        if selection_scorer_package_ref:
            selection_summary = {
                **selection_summary,
                "scorer_package_ref": selection_scorer_package_ref,
            }

        resolved_task_id = task_id or (datapack_records[0].task_id if datapack_records else None)
        if not resolved_task_id:
            raise ValueError("task_id is required when datapacks are resolved from local YAML.")

        task = store.get_task(resolved_task_id)
        if not task:
            raise ValueError(f"Task '{resolved_task_id}' not found in ontology.")

        objective_spec, objective_name = _resolve_objective_spec(objective_hint, objective_config)

        robot = _ensure_robot(store, robot_id, task.default_energy_cost_per_wh)
        econ_meter = EconomicMeter(task=task, robot=robot)
        backend = make_motor_backend(motor_backend, econ_meter, store)
        if backend is None:
            raise ValueError(f"Motor backend '{motor_backend}' is not configured.")

        register_datapack_configs(store, resolved_task_id, selected)

        scenario = build_scenario_metadata(
            run_id=run_id,
            task_id=resolved_task_id,
            motor_backend=motor_backend,
            objective_name=objective_name,
            objective=objective_spec,
            datapacks=selected,
            notes=_merge_notes(notes, missing_specs),
        )

        train_result = backend.train_policy(
            task_id=resolved_task_id,
            objective=objective_spec,
            datapack_ids=[cfg.id for cfg in selected],
            datapack_configs=selected,
            num_envs=num_envs,
            max_steps=max_steps,
            scenario_id=scenario.scenario_id,
            rollout_base_dir=rollout_base_dir,
            seed=seed,
        )
        steps_used = int(train_result.raw_metrics.get("train_steps", estimated_steps) or estimated_steps)

        eval_result: MotorEvalResult | None = None
        eval_metrics: dict[str, float] = {}
        if eval_episodes > 0:
            eval_result = backend.evaluate_policy(
                policy_id=train_result.policy_id,
                task_id=resolved_task_id,
                objective=objective_spec,
                num_episodes=eval_episodes,
                scenario_id=scenario.scenario_id,
                rollout_base_dir=rollout_base_dir,
                seed=seed,
            )
            eval_metrics = apply_arh_penalty(eval_result.econ_metrics)
            eval_result = MotorEvalResult(
                policy_id=eval_result.policy_id,
                raw_metrics=eval_result.raw_metrics,
                econ_metrics=eval_metrics,
                rollout_bundle=eval_result.rollout_bundle,
            )

        train_metrics = apply_arh_penalty(train_result.econ_metrics)
        train_result = MotorTrainingResult(
            policy_id=train_result.policy_id,
            raw_metrics=train_result.raw_metrics,
            econ_metrics=train_metrics,
            rollout_bundle=train_result.rollout_bundle,
        )
        store.record_scenario(
            scenario=scenario,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
        )

        rollout_bundle: RolloutBundle | None = None
        if eval_result and eval_result.rollout_bundle:
            rollout_bundle = eval_result.rollout_bundle
        elif rollout_base_dir:
            rollout_bundle = finalize_rollout_bundle(scenario.scenario_id, Path(rollout_base_dir))
        if rollout_bundle and selection_summary:
            _persist_selection_summary_to_rollouts(
                rollout_bundle=rollout_bundle,
                selection_summary=selection_summary,
            )

        labeled_datapacks: list[DatapackConfig] = []
        if rollout_base_dir and rollout_bundle and selected:
            labeled = label_rollouts_with_vla(rollout_bundle, base_datapack=selected[0])
            fusion_summaries = run_semantic_fusion_for_rollouts(
                rollout_bundle,
                summary_path=Path(rollout_base_dir) / rollout_bundle.scenario_id / "semantic_fusion_summary.jsonl",
            )
            labeled = _enrich_labeled_datapacks_with_fusion(labeled, fusion_summaries)
            for cfg in labeled:
                save_datapack_config(cfg, datapack_output_dir)
            register_datapack_configs(store, resolved_task_id, labeled)
            labeled_datapacks = list(labeled)
            vla_mode = _infer_vla_mode(labeled_datapacks)
            scenario = build_scenario_metadata(
                run_id=run_id,
                task_id=resolved_task_id,
                motor_backend=motor_backend,
                objective_name=objective_name,
                objective=objective_spec,
                datapacks=[*selected, *labeled_datapacks],
                notes=_merge_notes(notes, missing_specs),
            )
            store.record_scenario(
                scenario=scenario,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )

        simulation = SemanticSimulationResult(
            scenario=scenario,
            train_result=train_result,
            eval_result=eval_result,
            rollout_bundle=rollout_bundle,
            labeled_datapacks=labeled_datapacks,
            missing_specs=missing_specs,
            selection_summary=selection_summary,
        )
        status = "completed"
    except Exception as exc:
        reason = str(exc)
    finally:
        release_run_budget(steps_used or estimated_steps)
        _append_run_log(
            run_log_path,
            _build_run_log_payload(
                intent=intent,
                tags=tags,
                robot_family=robot_family,
                objective_hint=objective_hint,
                scenario=scenario,
                simulation=simulation,
                status=status,
                reason=reason,
                motor_backend=motor_backend,
                vla_mode=vla_mode,
            ),
        )

    return OrchestratedRunResult(status=status, scenario=scenario, simulation=simulation, reason=reason)


def _enrich_labeled_datapacks_with_fusion(
    datapacks: Sequence[DatapackConfig],
    fusion_summaries: Sequence[Mapping[str, Any]],
) -> list[DatapackConfig]:
    if not datapacks:
        return []
    fusion_summary = _aggregate_semantic_fusion_summary(fusion_summaries)
    future_training_artifacts = dict(fusion_summary["artifact_refs"])
    future_training_signals = {
        "semantic_fusion_ready": bool(fusion_summary["ready_count"] > 0),
        "semantic_fusion_blocked": bool(fusion_summary["blocked_count"] > 0),
    }
    enriched: list[DatapackConfig] = []
    for cfg in datapacks:
        metadata = dict(cfg.metadata or {})
        metadata_artifacts = dict(metadata.get("future_training_artifacts", {}) or {})
        metadata_artifacts.update(future_training_artifacts)
        benchmark_payload = {
            "scene_tracks_backend": metadata.get("scene_tracks_backend", ""),
            "teacher_runtime_backend_selected": metadata.get("teacher_runtime_backend_selected", ""),
            "vision_backbone_selected": metadata.get("vision_backbone_selected", ""),
            "semantic_grounding_mode": metadata.get("semantic_grounding_mode", ""),
            "semantic_memory_grounded": metadata.get("semantic_memory_grounded", False),
            "grounded_track_object_count": metadata.get("grounded_track_object_count", 0),
        }
        benchmark_signals = collect_benchmark_gating_signals(benchmark_payload)
        execution_preconditions = build_execution_preconditions(
            subject_id=cfg.id,
            subject_kind="vla_labeled_datapack",
            artifact_refs=metadata_artifacts,
            required_artifact_refs=["teacher_trace_ref", "vla_semantic_evidence_ref"],
            soft_required_artifact_refs=[
                "semantic_fusion_path",
                "semantic_world_model_path",
                "semantic_snapshot_path",
                "orchestrator_advisory_path",
            ],
            signal_values={
                **benchmark_signals,
                "semantic_fusion_ready": future_training_signals["semantic_fusion_ready"],
                "semantic_fusion_quality_mean": float(fusion_summary["quality_mean"]),
                "semantic_fusion_ready_fraction": float(fusion_summary["ready_fraction"]),
            },
            required_boolean_signals={
                "semantic_grounding_non_heuristic": True,
                "teacher_runtime_real": True,
                "vision_backbone_real": True,
                "semantic_fusion_ready": True,
            },
            metadata={
                "selection_contract": "vla_rollout_labeler_v2",
                "semantic_fusion_summary": fusion_summary["summary"],
            },
        )
        metadata["benchmark_signals"] = benchmark_signals
        metadata["execution_preconditions"] = execution_preconditions.to_dict()
        metadata["future_training_artifacts"] = metadata_artifacts
        metadata["future_training_signals"] = {
            **dict(metadata.get("future_training_signals", {}) or {}),
            **future_training_signals,
            "benchmark_eligible": bool(benchmark_signals.get("benchmark_eligible", False)),
        }
        metadata["semantic_fusion"] = fusion_summary["summary"]
        artifacts = dict(metadata.get("artifacts", {}) or {})
        artifacts.update(fusion_summary["catalog"])
        metadata["artifacts"] = artifacts
        quality_score = max(float(cfg.quality_score), float(fusion_summary["quality_mean"]))
        metadata["quality_score"] = quality_score
        enriched.append(replace(cfg, quality_score=quality_score, metadata=metadata))
    return enriched


def _aggregate_semantic_fusion_summary(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in summaries]
    ready_rows = [row for row in rows if row.get("semantic_fusion_status") == "ready"]
    blocked_rows = [row for row in rows if row.get("semantic_fusion_status") == "blocked"]
    artifact_keys = (
        "semantic_fusion_path",
        "semantic_fusion_failure_path",
        "evidence_bus_path",
        "belief_state_path",
        "semantic_world_model_path",
        "semantic_snapshot_path",
        "orchestrator_advisory_path",
    )
    artifact_refs: dict[str, list[str]] = {}
    for row in rows:
        for key in artifact_keys:
            value = row.get(key)
            if value in (None, "", [], {}):
                continue
            artifact_refs.setdefault(key, [])
            if str(value) not in artifact_refs[key]:
                artifact_refs[key].append(str(value))
    quality_mean = (
        sum(float(row.get("semantic_fusion_quality_score", 0.0) or 0.0) for row in ready_rows)
        / float(max(len(ready_rows), 1))
    )
    ready_fraction = float(len(ready_rows)) / float(max(len(rows), 1))
    if ready_rows and blocked_rows:
        status = "mixed"
    elif ready_rows:
        status = "ready"
    elif blocked_rows:
        status = "blocked"
    else:
        status = "missing"
    failure_reasons = sorted(
        {
            str(row.get("semantic_fusion_failure_reason"))
            for row in blocked_rows
            if row.get("semantic_fusion_failure_reason")
        }
    )
    return {
        "artifact_refs": artifact_refs,
        "catalog": {
            "semantic_fusion": list(artifact_refs.get("semantic_fusion_path", []) or []),
            "semantic_fusion_failures": list(
                artifact_refs.get("semantic_fusion_failure_path", []) or []
            ),
            "semantic_world_models": list(
                artifact_refs.get("semantic_world_model_path", []) or []
            ),
            "semantic_snapshots": list(
                artifact_refs.get("semantic_snapshot_path", []) or []
            ),
            "orchestrator_advisories": list(
                artifact_refs.get("orchestrator_advisory_path", []) or []
            ),
        },
        "ready_count": len(ready_rows),
        "blocked_count": len(blocked_rows),
        "quality_mean": quality_mean,
        "ready_fraction": ready_fraction,
        "summary": {
            "status": status,
            "ready_count": len(ready_rows),
            "blocked_count": len(blocked_rows),
            "quality_mean": quality_mean,
            "ready_fraction": ready_fraction,
            "failure_reasons": failure_reasons,
        },
    }


def _resolve_objective_spec(
    objective_hint: str | None,
    objective_config: str | None,
) -> tuple[EconomicObjectiveSpec, str]:
    if objective_config:
        path = Path(objective_config)
        return load_economic_objective_spec(path), path.stem

    if objective_hint:
        hint_path = Path(objective_hint)
        if hint_path.exists():
            return load_economic_objective_spec(hint_path), hint_path.stem
        config_path = Path("configs/objectives") / f"{objective_hint}.yaml"
        if config_path.exists():
            return load_economic_objective_spec(config_path), config_path.stem
        try:
            obj_vec = ObjectiveVector.from_preset(objective_hint)
            return EconomicObjectiveSpec.from_objective_vector(obj_vec), objective_hint
        except Exception:
            return EconomicObjectiveSpec(), objective_hint

    return EconomicObjectiveSpec(), "default"


def _resolve_local_datapacks(
    *,
    tags: Sequence[str] | None,
    robot_family: str | None,
    objective_hint: str | None,
) -> list[DatapackConfig]:
    datapack_dir = Path("configs/datapacks")
    if not datapack_dir.exists():
        return []
    configs: list[DatapackConfig] = []
    for path in sorted(datapack_dir.glob("*.yml")) + sorted(datapack_dir.glob("*.yaml")):
        configs.extend(load_datapack_configs([path]))

    tag_set = {t.strip().lower() for t in tags or [] if t and str(t).strip()}
    robot_norm = robot_family.strip().lower() if robot_family else None
    objective_norm = objective_hint.strip().lower() if objective_hint else None

    filtered: list[DatapackConfig] = []
    for cfg in configs:
        cfg_tags = {t.lower() for t in cfg.tags}
        cfg_robot = {t.lower() for t in cfg.robot_families}
        cfg_objective = cfg.objective_hint.lower() if cfg.objective_hint else None

        if tag_set and not tag_set.issubset(cfg_tags):
            continue
        if robot_norm and robot_norm not in cfg_robot:
            continue
        if objective_norm and (not cfg_objective or objective_norm not in cfg_objective):
            continue
        filtered.append(cfg)
    return filtered


def _resolve_datapack_selection_scorer_package(
    *,
    selection_scorer_package: Mapping[str, Any] | None,
    selection_scorer_package_path: str | None,
    selection_scorer_mode: Literal["disabled", "auto", "required"],
) -> tuple[Mapping[str, Any] | None, str | None, dict[str, Any]]:
    if selection_scorer_mode not in {"disabled", "auto", "required"}:
        raise ValueError(f"Unsupported selection_scorer_mode: {selection_scorer_mode}")
    if selection_scorer_mode == "disabled":
        return (
            None,
            None,
            {
                "mode": selection_scorer_mode,
                "status": "disabled",
                "promotion_stage": "disabled",
                "benchmark_gate_ready": False,
                "effective_max_adjustment": 0.0,
            },
        )
    if selection_scorer_package is not None:
        package = coerce_datapack_selection_scorer_package(selection_scorer_package)
        return _finalize_datapack_selection_scorer_package(
            package,
            selection_scorer_mode=selection_scorer_mode,
            package_ref=None,
        )
    if selection_scorer_package_path:
        package = load_datapack_selection_scorer_package(selection_scorer_package_path)
        return _finalize_datapack_selection_scorer_package(
            package,
            selection_scorer_mode=selection_scorer_mode,
            package_ref=str(Path(selection_scorer_package_path).resolve()),
        )
    candidate_paths = [
        Path("artifacts/semantic_selection/datapack_selection_scorer_package.json"),
        Path("artifacts/semantic_runtime/datapack_selection_scorer_package.json"),
        Path("artifacts/semantic_runtime_scorers/datapack_selection_scorer_package.json"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            package = load_datapack_selection_scorer_package(candidate)
            return _finalize_datapack_selection_scorer_package(
                package,
                selection_scorer_mode=selection_scorer_mode,
                package_ref=str(candidate.resolve()),
            )
    if selection_scorer_mode == "required":
        raise FileNotFoundError(
            "selection_scorer_mode='required' but no datapack selection scorer package was found"
        )
    return (
        None,
        None,
        {
            "mode": selection_scorer_mode,
            "status": "heuristic_fallback",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
            "effective_max_adjustment": 0.0,
        },
    )


def _finalize_datapack_selection_scorer_package(
    package: DatapackSelectionScorerPackage | None,
    *,
    selection_scorer_mode: Literal["disabled", "auto", "required"],
    package_ref: str | None,
) -> tuple[Mapping[str, Any] | None, str | None, dict[str, Any]]:
    if package is None:
        return (
            None,
            package_ref,
            {
                "mode": selection_scorer_mode,
                "status": "heuristic_fallback",
                "promotion_stage": "heuristic_fallback",
                "benchmark_gate_ready": False,
                "effective_max_adjustment": 0.0,
            },
        )
    benchmark_gate = dict(package.metadata.get("benchmark_gate", {}) or {})
    benchmark_gate_ready = bool(benchmark_gate.get("ready", False))
    raw_max_adjustment = float(package.max_adjustment)
    effective_max_adjustment = raw_max_adjustment
    promotion_stage = "promoted" if benchmark_gate_ready else "shadow_candidate"
    effective_package = package
    if not benchmark_gate_ready:
        shadow_cap = min(
            raw_max_adjustment,
            max(float(package.min_adjustment), raw_max_adjustment * 0.35 if raw_max_adjustment > 0.0 else 0.0),
        )
        effective_max_adjustment = max(float(package.min_adjustment), shadow_cap)
        effective_package = DatapackSelectionScorerPackage(
            package_id=package.package_id,
            schema_version=package.schema_version,
            feature_weights=dict(package.feature_weights),
            context_weights=dict(package.context_weights),
            bias=float(package.bias),
            context_bias=float(package.context_bias),
            min_adjustment=float(package.min_adjustment),
            max_adjustment=float(effective_max_adjustment),
            metadata={
                **dict(package.metadata),
                "selection_helper_promotion": {
                    "raw_max_adjustment": raw_max_adjustment,
                    "effective_max_adjustment": effective_max_adjustment,
                    "benchmark_gate_ready": benchmark_gate_ready,
                    "promotion_stage": promotion_stage,
                },
            },
        )
    if selection_scorer_mode == "required" and not benchmark_gate_ready:
        raise FileNotFoundError(
            "selection_scorer_mode='required' but datapack selection scorer package is not benchmark-gated ready"
        )
    return (
        effective_package.to_dict(),
        package_ref,
        {
            "mode": selection_scorer_mode,
            "status": "available",
            "promotion_stage": promotion_stage,
            "benchmark_gate_ready": benchmark_gate_ready,
            "raw_max_adjustment": raw_max_adjustment,
            "effective_max_adjustment": effective_max_adjustment,
            "conditioning_policy": (
                "context_conditioned_max_adjustment"
                if package.context_weights
                else "unconditioned_max_adjustment"
            ),
            "package_id": package.package_id,
        },
    )


def _selection_context_from_ranked(
    ranked: Sequence[DatapackSelectionDecision],
) -> dict[str, Any]:
    if not ranked:
        return {}
    scorer_trace = dict(ranked[0].scorer_trace or {})
    context_trace = dict(scorer_trace.get("context_trace", {}) or {})
    context = context_trace.get("context")
    if isinstance(context, Mapping):
        return {
            str(key): float(value)
            for key, value in dict(context).items()
        }
    return {}


def _merge_notes(notes: str | None, missing_specs: Sequence[MissingScenarioSpec]) -> str | None:
    if not missing_specs:
        return notes
    gap_notes = "; ".join([",".join(spec.tags) for spec in missing_specs])
    if notes:
        return f"{notes}; gaps={gap_notes}"
    return f"gaps={gap_notes}"


def _ensure_robot(store: OntologyStore, robot_id: str, energy_cost_per_wh: float) -> "Robot":
    from src.ontology.models import Robot

    robot = store.get_robot(robot_id)
    if robot:
        return robot
    robot = Robot(robot_id=robot_id, name=robot_id, energy_cost_per_wh=energy_cost_per_wh)
    store.upsert_robot(robot)
    return robot


def _append_run_log(path: str | Path, payload: Mapping[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _build_run_log_payload(
    *,
    intent: str | None,
    tags: Sequence[str] | None,
    robot_family: str | None,
    objective_hint: str | None,
    scenario: ScenarioMetadata | None,
    simulation: SemanticSimulationResult | None,
    status: str,
    reason: str | None,
    motor_backend: str,
    vla_mode: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "intent": intent,
        "tags": list(tags or []),
        "robot_family": robot_family,
        "objective_hint": objective_hint,
        "motor_backend": motor_backend,
        "vla_mode": vla_mode,
        "status": status,
        "reason": reason,
        "scenario_id": scenario.scenario_id if scenario else None,
        "new_datapacks": len(simulation.labeled_datapacks) if simulation else 0,
    }

    if simulation:
        payload["train_metrics"] = _select_core_metrics(simulation.train_result.econ_metrics)
        if simulation.eval_result:
            payload["eval_metrics"] = _select_core_metrics(simulation.eval_result.econ_metrics)
        payload["selection_summary"] = dict(simulation.selection_summary or {})
    return payload


def _persist_selection_summary_to_rollouts(
    *,
    rollout_bundle: RolloutBundle,
    selection_summary: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": "selection_summary_sidecar_v1",
        "scenario_id": rollout_bundle.scenario_id,
        "selection_summary": dict(selection_summary),
    }
    for episode in list(rollout_bundle.episodes or []):
        episode_dir = episode.trajectory_path.parent
        episode_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = episode_dir / f"{episode.metadata.episode_id}_selection_summary_v1.json"
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata_payload = {}
        metadata_payload["selection_summary_path"] = str(sidecar_path.resolve())
        metadata_payload["selection_summary"] = dict(selection_summary)
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _candidate_metadata_by_id(datapacks: Sequence[Any]) -> dict[str, dict[str, Any]]:
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for datapack in datapacks:
        metadata_by_id[str(datapack.datapack_id)] = {
            "source_type": getattr(datapack, "source_type", ""),
            "storage_uri": getattr(datapack, "storage_uri", ""),
            "quality_score": float(getattr(datapack, "quality_score", 0.0) or 0.0),
            "novelty_score": float(getattr(datapack, "novelty_score", 0.0) or 0.0),
            "metadata": dict(getattr(datapack, "metadata", {}) or {}),
        }
    return metadata_by_id


def _merge_ranked_datapacks(
    primary: Sequence[DatapackSelectionDecision],
    secondary: Sequence[DatapackSelectionDecision],
) -> list[DatapackSelectionDecision]:
    merged: dict[str, DatapackSelectionDecision] = {}
    for row in list(primary) + list(secondary):
        existing = merged.get(row.datapack.id)
        if existing is None or row.score > existing.score:
            merged[row.datapack.id] = row
    return sorted(
        merged.values(),
        key=lambda row: (row.score, row.source == "ontology", row.datapack.id),
        reverse=True,
    )


def _select_core_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    keys = ("mpl_units_per_hour", "wage_parity", "energy_cost", "error_rate", "reward_scalar_sum")
    out: dict[str, float] = {}
    for key in keys:
        if key in metrics:
            out[key] = float(metrics[key])
    if "anti_reward_hacking_suspicious" in metrics:
        out["anti_reward_hacking_suspicious"] = float(metrics["anti_reward_hacking_suspicious"])
    if "arh_excluded" in metrics:
        out["arh_excluded"] = float(metrics["arh_excluded"])
    return out


def _infer_vla_mode(datapacks: Sequence[DatapackConfig]) -> str:
    tags: set[str] = set()
    for cfg in datapacks:
        tags.update([str(tag) for tag in cfg.tags])
    if "vla_error" in tags:
        return "error_fallback"
    if "vla:available" in tags:
        return "openvla"
    return "stub"


def get_recent_runs(
    path: str | Path,
    *,
    limit: int = 20,
    status: str | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with log_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if status and record.get("status") != status:
                continue
            if backend and record.get("motor_backend") != backend:
                continue
            records.append(record)
    if limit <= 0:
        return records
    return records[-limit:]


# ==============================================================================
# Coverage-gap-driven simulation agenda (Phase C)
# ==============================================================================

SimulationAgendaItem = SimulationJobSpec


def compile_simulation_agenda(
    coverage_graph: Any,
    *,
    economic_weight: float = 1.0,
    trust_weight: float = 1.0,
    readiness_weight: float = 1.0,
    limit: int = 10,
    default_backend: str = "pybullet",
    default_objective: str = "balanced",
    gap_ranker: Any = None,
    gap_ranker_mode: Literal["disabled", "auto", "required"] = "auto",
    backend_selector: Any = None,
    backend_selector_mode: Literal["disabled", "auto", "required"] = "auto",
    branch_planner: Any = None,
    branch_planner_mode: Literal["disabled", "auto", "required"] = "auto",
) -> list[dict[str, Any]]:
    """Compile a ranked simulation agenda from the semantic coverage graph.

    This function now delegates agenda ownership to the canonical
    sim/synth/physics world-model compiler and returns the legacy agenda
    artifact view for downstream compatibility.

    Parameters
    ----------
    coverage_graph : SemanticCoverageGraph
    economic_weight, trust_weight, readiness_weight
        Weights for gap ranking.
    limit : int
        Maximum agenda items.
    default_backend, default_objective
        Fallbacks when the coverage edge doesn't specify a backend/objective.

    Returns
    -------
    list of dict (``simulation_agenda_v1`` items)
    """
    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(
            economic_weight=economic_weight,
            trust_weight=trust_weight,
            readiness_weight=readiness_weight,
            agenda_limit=limit,
            default_backend=default_backend,
            default_objective=default_objective,
            gap_ranker_mode=gap_ranker_mode,
            backend_selector_mode=backend_selector_mode,
            branch_planner_mode=branch_planner_mode,
        )
    )
    return runtime.compile_legacy_agenda(
        coverage_graph,
        gap_ranker=gap_ranker,
        backend_selector=backend_selector,
        branch_planner=branch_planner,
    )
