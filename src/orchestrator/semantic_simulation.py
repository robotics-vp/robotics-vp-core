from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Literal

from src.config.objective_profile import ObjectiveVector
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
    MissingScenarioSpec,
    apply_arh_penalty,
    detect_semantic_gaps,
    select_datapacks_for_intent,
)
from src.orchestrator.semantic_fusion_runner import run_semantic_fusion_for_rollouts
from src.scenarios.metadata import ScenarioMetadata, build_scenario_metadata
from src.vla.rollout_labeler import label_rollouts_with_vla


@dataclass(frozen=True)
class SemanticSimulationResult:
    scenario: ScenarioMetadata
    train_result: MotorTrainingResult
    eval_result: MotorEvalResult | None
    rollout_bundle: RolloutBundle | None
    labeled_datapacks: Sequence[DatapackConfig]
    missing_specs: Sequence[MissingScenarioSpec]


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

        scenario_records = find_scenarios(
            store,
            datapack_tags=tags,
            robot_families=[robot_family] if robot_family else None,
            objective_name=objective_hint,
            motor_backend=motor_backend,
        )

        selected = select_datapacks_for_intent(
            tags or [],
            robot_family,
            objective_hint,
            candidates,
            scenario_records,
        )

        missing_specs = detect_semantic_gaps(tags or [], robot_family, scenario_records)

        if missing_specs:
            exploratory = _resolve_local_datapacks(tags=tags, robot_family=robot_family, objective_hint=objective_hint)
            if exploratory:
                selected = exploratory
        if not selected:
            selected = _resolve_local_datapacks(tags=tags, robot_family=robot_family, objective_hint=objective_hint)

        if not selected:
            raise ValueError("No datapacks matched the requested semantic filters.")

        if datapack_limit:
            selected = list(selected)[: datapack_limit]

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

        labeled_datapacks: list[DatapackConfig] = []
        if rollout_base_dir and rollout_bundle and selected:
            labeled = label_rollouts_with_vla(rollout_bundle, base_datapack=selected[0])
            run_semantic_fusion_for_rollouts(
                rollout_bundle,
                summary_path=Path(rollout_base_dir) / rollout_bundle.scenario_id / "semantic_fusion_summary.jsonl",
            )
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
    return payload


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
