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


def register_scene_tracks_artifact(
    *,
    store: OntologyStore,
    datapack_id: str,
    task_id: str,
    artifact_path: str,
    frame_metadata: dict,
    source_datapack: str,
) -> dict:
    """Register a SceneTracks artifact under a datapack record."""
    existing = {dp.datapack_id: dp for dp in store.list_datapacks()}
    datapack = existing.get(datapack_id)

    if datapack is None:
        datapack = Datapack(
            datapack_id=datapack_id,
            source_type="physics",
            task_id=task_id,
            modality="video",
            storage_uri=source_datapack,
            metadata={},
        )

    metadata = dict(datapack.metadata or {})
    artifacts = dict(metadata.get("artifacts") or {})
    scene_entries = list(artifacts.get("scene_tracks") or [])

    entry = {
        "path": artifact_path,
        "source_datapack": source_datapack,
        "frame_count": int(frame_metadata.get("frame_count", 0)),
        "frame_range": list(frame_metadata.get("frame_range", [])),
        "frame_indices": list(frame_metadata.get("frame_indices", [])),
        "camera": frame_metadata.get("camera_name"),
        "scene_tracks_quality": frame_metadata.get("scene_tracks_quality"),
        "scene_ir_quality": frame_metadata.get("scene_ir_quality"),
        "datapack_hash": frame_metadata.get("datapack_hash"),
        "runner_config_hash": frame_metadata.get("runner_config_hash"),
        "runner": frame_metadata.get("runner"),
    }

    scene_entries.append(entry)
    artifacts["scene_tracks"] = scene_entries
    metadata["artifacts"] = artifacts
    metadata["scene_tracks_latest"] = entry
    datapack.metadata = metadata

    store.append_datapacks([datapack])
    return entry


def get_latest_scene_tracks_artifact(
    store: OntologyStore,
    datapack_id: str,
) -> dict | None:
    """Fetch the latest SceneTracks artifact metadata for a datapack."""
    existing = {dp.datapack_id: dp for dp in store.list_datapacks()}
    datapack = existing.get(datapack_id)
    if not datapack or not isinstance(datapack.metadata, dict):
        return None
    return datapack.metadata.get("scene_tracks_latest")
