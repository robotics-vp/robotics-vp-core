from __future__ import annotations

"""Datapack YAML loader and resolver for motor backends.

Schema (YAML):
  id: <string>
  description: <string>
  motion_clips:
    - path: <path>
      weight: <float>
  domain_randomization: <mapping>
  curriculum: <mapping>
  tags: [<string>]
  task_tags: [<string>]
  robot_families: [<string>]
  objective_hint: <string>
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.ontology.store import OntologyStore


@dataclass(frozen=True)
class MotionClipSpec:
    path: str
    weight: float = 1.0


@dataclass(frozen=True)
class DatapackConfig:
    id: str
    description: str = ""
    motion_clips: Sequence[MotionClipSpec] = field(default_factory=list)
    domain_randomization: Mapping[str, Any] = field(default_factory=dict)
    curriculum: Mapping[str, Any] = field(default_factory=dict)
    tags: Sequence[str] = field(default_factory=list)
    task_tags: Sequence[str] = field(default_factory=list)
    robot_families: Sequence[str] = field(default_factory=list)
    objective_hint: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class DatapackBundle:
    datapack_ids: Sequence[str]
    datapack_configs: Sequence[DatapackConfig] = field(default_factory=list)
    motion_clips: Sequence[MotionClipSpec] = field(default_factory=list)
    randomization_overrides: Mapping[str, Any] = field(default_factory=dict)
    curriculum_overrides: Mapping[str, Any] = field(default_factory=dict)


def load_datapack_configs(paths: Sequence[str | Path]) -> list[DatapackConfig]:
    configs: list[DatapackConfig] = []
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Datapack config not found: {p}")
        payload = yaml.safe_load(p.read_text())
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(f"Datapack config at {p} must be a mapping, got {type(payload).__name__}")
        dp_id = str(payload.get("id") or "").strip()
        if not dp_id:
            raise ValueError(f"Datapack config at {p} is missing required field 'id'")
        configs.append(
            DatapackConfig(
                id=dp_id,
                description=str(payload.get("description") or ""),
                motion_clips=_parse_motion_clips(payload.get("motion_clips") or []),
                domain_randomization=payload.get("domain_randomization", {}) or {},
                curriculum=payload.get("curriculum", {}) or {},
                tags=_parse_string_list(payload.get("tags") or []),
                task_tags=_parse_string_list(payload.get("task_tags") or []),
                robot_families=_parse_string_list(payload.get("robot_families") or []),
                objective_hint=_parse_optional_str(payload.get("objective_hint")),
                source_path=str(p),
            )
        )
    return configs


def _parse_motion_clips(raw: Sequence[Any]) -> list[MotionClipSpec]:
    clips: list[MotionClipSpec] = []
    for entry in raw:
        if isinstance(entry, str):
            clips.append(MotionClipSpec(path=entry, weight=1.0))
            continue
        if isinstance(entry, Mapping):
            path = entry.get("path")
            if not path:
                continue
            weight = entry.get("weight", 1.0)
            try:
                weight_val = float(weight)
            except (TypeError, ValueError):
                weight_val = 1.0
            clips.append(MotionClipSpec(path=str(path), weight=weight_val))
    return clips


def _parse_string_list(raw: Sequence[Any]) -> list[str]:
    values: list[str] = []
    for entry in raw:
        if entry is None:
            continue
        text = str(entry).strip()
        if text:
            values.append(text)
    return values


def _parse_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


class DatapackProvider:
    """Resolve datapack identifiers into motion datasets and overrides."""

    def __init__(self, store: OntologyStore):
        self._store = store

    def resolve(
        self,
        task_id: str,
        datapack_ids: Sequence[str],
        datapack_configs: Sequence[DatapackConfig] | None = None,
    ) -> DatapackBundle:
        motion_clips: list[MotionClipSpec] = []
        randomization: dict[str, Any] = {}
        curriculum: dict[str, Any] = {}
        ids: list[str] = []
        configs = list(datapack_configs or [])

        for cfg in configs:
            ids.append(cfg.id)
            motion_clips.extend(cfg.motion_clips)
            randomization.update(cfg.domain_randomization or {})
            curriculum.update(cfg.curriculum or {})

        if datapack_ids:
            datapacks = {dp.datapack_id: dp for dp in self._store.list_datapacks(task_id=task_id)}
            for dp_id in datapack_ids:
                ids.append(dp_id)
                dp = datapacks.get(dp_id)
                if not dp:
                    continue
                if dp.storage_uri:
                    motion_clips.append(MotionClipSpec(path=dp.storage_uri, weight=1.0))
                if isinstance(dp.metadata, dict):
                    if dp.metadata.get("randomization"):
                        randomization.update(dp.metadata.get("randomization", {}))
                    if dp.metadata.get("curriculum"):
                        curriculum.update(dp.metadata.get("curriculum", {}))

        deduped_ids = list(dict.fromkeys(ids))
        return DatapackBundle(
            datapack_ids=deduped_ids,
            datapack_configs=configs,
            motion_clips=motion_clips,
            randomization_overrides=randomization,
            curriculum_overrides=curriculum,
        )
