from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.datapacks import (
    DatapackConfig,
    datapack_configs_from_ontology,
    load_datapack_configs,
    save_datapack_config,
)
from src.motor_backend.factory import make_motor_backend
from src.motor_backend.rollout_capture import RolloutCaptureConfig, RolloutCaptureResult, capture_rollouts
from src.objectives.economic_objective import EconomicObjectiveSpec, load_economic_objective_spec
from src.ontology.datapack_registry import register_datapack_configs
from src.ontology.query import find_datapacks
from src.ontology.store import OntologyStore
from src.orchestrator.semantic_policy import apply_arh_penalty
from src.scenarios.metadata import ScenarioMetadata, build_scenario_metadata
from src.vla.rollout_labeler import RolloutLabeler, StubRolloutLabeler
from src.config.objective_profile import ObjectiveVector


@dataclass(frozen=True)
class SemanticSimulationResult:
    scenario: ScenarioMetadata
    train_result: MotorTrainingResult
    eval_result: MotorEvalResult | None
    rollout_capture: RolloutCaptureResult | None
    labeled_datapacks: Sequence[DatapackConfig]


def run_semantic_simulation(
    *,
    store: OntologyStore,
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
    rollout_capture_config: RolloutCaptureConfig | None = None,
    rollout_labeler: RolloutLabeler | None = None,
    datapack_output_dir: str | Path = "configs/datapacks",
    robot_id: str = "robot_default",
) -> SemanticSimulationResult:
    datapack_records = find_datapacks(
        store,
        tags=tags,
        robot_family=robot_family,
        objective_hint=objective_hint,
        task_id=task_id,
        limit=datapack_limit,
    )
    datapack_configs = datapack_configs_from_ontology(datapack_records)

    if not datapack_configs:
        datapack_configs = _resolve_local_datapacks(tags=tags, robot_family=robot_family, objective_hint=objective_hint)

    if not datapack_configs:
        raise ValueError("No datapacks matched the requested semantic filters.")

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

    register_datapack_configs(store, resolved_task_id, datapack_configs)

    run_id = str(uuid.uuid4())
    scenario = build_scenario_metadata(
        run_id=run_id,
        task_id=resolved_task_id,
        motor_backend=motor_backend,
        objective_name=objective_name,
        objective=objective_spec,
        datapacks=datapack_configs,
        notes=notes,
    )

    train_result = backend.train_policy(
        task_id=resolved_task_id,
        objective=objective_spec,
        datapack_ids=[cfg.id for cfg in datapack_configs],
        datapack_configs=datapack_configs,
        num_envs=num_envs,
        max_steps=max_steps,
        seed=seed,
    )

    eval_result: MotorEvalResult | None = None
    eval_metrics: dict[str, float] = {}
    if eval_episodes > 0:
        eval_result = backend.evaluate_policy(
            policy_id=train_result.policy_id,
            task_id=resolved_task_id,
            objective=objective_spec,
            num_episodes=eval_episodes,
            seed=seed,
        )
        eval_metrics = apply_arh_penalty(eval_result.econ_metrics)

    train_metrics = apply_arh_penalty(train_result.econ_metrics)
    store.record_scenario(
        scenario=scenario,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
    )

    rollout_capture_result = None
    labeled_datapacks: list[DatapackConfig] = []
    if rollout_capture_config:
        rollout_capture_result = capture_rollouts(
            policy_id=train_result.policy_id,
            scenario_id=scenario.scenario_id,
            task_id=resolved_task_id,
            datapack_ids=[cfg.id for cfg in datapack_configs],
            config=rollout_capture_config,
        )
        labeler = rollout_labeler or StubRolloutLabeler()
        base_datapack = datapack_configs[0] if datapack_configs else None
        labeled = list(labeler.label_rollouts(rollout_capture_result.output_dir, base_datapack=base_datapack))
        for cfg in labeled:
            save_datapack_config(cfg, datapack_output_dir)
        register_datapack_configs(store, resolved_task_id, labeled)
        labeled_datapacks = labeled

    return SemanticSimulationResult(
        scenario=scenario,
        train_result=train_result,
        eval_result=eval_result,
        rollout_capture=rollout_capture_result,
        labeled_datapacks=labeled_datapacks,
    )


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


def _ensure_robot(store: OntologyStore, robot_id: str, energy_cost_per_wh: float) -> "Robot":
    from src.ontology.models import Robot

    robot = store.get_robot(robot_id)
    if robot:
        return robot
    robot = Robot(robot_id=robot_id, name=robot_id, energy_cost_per_wh=energy_cost_per_wh)
    store.upsert_robot(robot)
    return robot
