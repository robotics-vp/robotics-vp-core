from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.ontology.store import OntologyStore


@dataclass(frozen=True)
class DatapackBundle:
    datapack_ids: Sequence[str]
    motion_clip_paths: Sequence[str] = field(default_factory=list)
    randomization_overrides: Mapping[str, Any] = field(default_factory=dict)
    curriculum_overrides: Mapping[str, Any] = field(default_factory=dict)


class DatapackProvider:
    """Resolve datapack identifiers into motion datasets and overrides."""

    def __init__(self, store: OntologyStore):
        self._store = store

    def resolve(self, task_id: str, datapack_ids: Sequence[str]) -> DatapackBundle:
        if not datapack_ids:
            return DatapackBundle(datapack_ids=[], motion_clip_paths=[])

        datapacks = {dp.datapack_id: dp for dp in self._store.list_datapacks(task_id=task_id)}
        motion_paths: list[str] = []
        randomization: dict[str, Any] = {}
        curriculum: dict[str, Any] = {}

        for dp_id in datapack_ids:
            dp = datapacks.get(dp_id)
            if not dp:
                continue
            if dp.storage_uri:
                motion_paths.append(dp.storage_uri)
            if isinstance(dp.metadata, dict):
                if dp.metadata.get("randomization"):
                    randomization.update(dp.metadata.get("randomization", {}))
                if dp.metadata.get("curriculum"):
                    curriculum.update(dp.metadata.get("curriculum", {}))
        return DatapackBundle(
            datapack_ids=list(datapack_ids),
            motion_clip_paths=motion_paths,
            randomization_overrides=randomization,
            curriculum_overrides=curriculum,
        )
