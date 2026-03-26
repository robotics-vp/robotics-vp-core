from __future__ import annotations

from typing import Any

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
    upserts: list[Datapack] = []
    for cfg in datapack_configs:
        current = existing.get(cfg.id)
        storage_uri = cfg.motion_clips[0].path if cfg.motion_clips else ""
        metadata = _merge_datapack_metadata(current.metadata if current is not None else {}, cfg)
        datapack_kwargs: dict[str, Any] = {
            "datapack_id": cfg.id,
            "source_type": current.source_type if current is not None else source_type,
            "task_id": current.task_id if current is not None else task_id,
            "modality": current.modality if current is not None else modality,
            "storage_uri": storage_uri or (current.storage_uri if current is not None else ""),
            "novelty_score": float(cfg.novelty_score),
            "quality_score": float(cfg.quality_score),
            "tags": {
                "semantic_tags": list(cfg.tags),
                "task_tags": list(cfg.task_tags),
                "robot_families": list(cfg.robot_families),
            },
            "metadata": metadata,
            "sima2_backend_id": current.sima2_backend_id if current is not None else None,
            "sima2_model_version": current.sima2_model_version if current is not None else None,
            "sima2_task_spec": dict(current.sima2_task_spec) if current is not None else {},
            "auditor_rating": current.auditor_rating if current is not None else None,
            "auditor_score": current.auditor_score if current is not None else None,
            "auditor_predicted_econ": (
                dict(current.auditor_predicted_econ)
                if current is not None and current.auditor_predicted_econ is not None
                else None
            ),
        }
        if current is not None:
            datapack_kwargs["created_at"] = current.created_at
        upserts.append(Datapack(**datapack_kwargs))
    if upserts:
        store.append_datapacks(upserts)


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


def _merge_datapack_metadata(
    base_metadata: dict[str, Any] | None,
    config: DatapackConfig,
) -> dict[str, Any]:
    metadata = dict(base_metadata or {})
    metadata.update(dict(config.metadata or {}))
    metadata["description"] = config.description
    metadata["randomization"] = dict(config.domain_randomization)
    metadata["curriculum"] = dict(config.curriculum)
    metadata["tags"] = list(config.tags)
    metadata["task_tags"] = list(config.task_tags)
    metadata["robot_families"] = list(config.robot_families)
    metadata["objective_hint"] = config.objective_hint
    metadata["source_path"] = config.source_path
    return metadata
