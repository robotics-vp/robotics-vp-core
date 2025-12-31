from __future__ import annotations

from typing import Sequence

from src.motor_backend.datapacks import DatapackConfig
from src.ontology.models import Datapack
from src.ontology.store import OntologyStore


def register_datapack_configs(
    store: OntologyStore,
    task_id: str,
    datapack_configs: Sequence[DatapackConfig],
    source_type: str = "holosoma",
    modality: str = "motion",
) -> None:
    if not datapack_configs:
        return
    existing = {dp.datapack_id: dp for dp in store.list_datapacks()}
    stubs: list[Datapack] = []
    for cfg in datapack_configs:
        if cfg.id in existing:
            continue
        storage_uri = cfg.motion_clips[0].path if cfg.motion_clips else ""
        stubs.append(
            Datapack(
                datapack_id=cfg.id,
                source_type=source_type,
                task_id=task_id,
                modality=modality,
                storage_uri=storage_uri,
                metadata={
                    "description": cfg.description,
                    "randomization": dict(cfg.domain_randomization),
                    "curriculum": dict(cfg.curriculum),
                    "tags": list(cfg.tags),
                    "task_tags": list(cfg.task_tags),
                    "robot_families": list(cfg.robot_families),
                    "objective_hint": cfg.objective_hint,
                    "source_path": cfg.source_path,
                },
            )
        )
    if stubs:
        store.append_datapacks(stubs)
