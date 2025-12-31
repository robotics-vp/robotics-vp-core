from __future__ import annotations

from typing import Sequence, Mapping, Any

from src.ontology.models import Datapack
from src.ontology.store import OntologyStore


def find_datapacks(
    store: OntologyStore,
    *,
    tags: Sequence[str] | None = None,
    task_tags: Sequence[str] | None = None,
    robot_family: str | None = None,
    objective_hint: str | None = None,
    task_id: str | None = None,
    limit: int | None = None,
    match_any: bool = False,
) -> list[Datapack]:
    candidates = store.list_datapacks(task_id=task_id)
    tag_set = _normalize_set(tags)
    task_tag_set = _normalize_set(task_tags)
    robot_family_norm = _normalize_text(robot_family)
    objective_norm = _normalize_text(objective_hint)

    results: list[Datapack] = []
    for dp in candidates:
        meta = dp.metadata or {}
        dp_tags = _normalize_set(meta.get("tags") or _tags_from_payload(dp.tags))
        dp_task_tags = _normalize_set(meta.get("task_tags") or [])
        dp_robot_families = _normalize_set(meta.get("robot_families") or [])
        dp_objective = _normalize_text(meta.get("objective_hint"))

        if tag_set and not _match_set(tag_set, dp_tags, match_any):
            continue
        if task_tag_set and not _match_set(task_tag_set, dp_task_tags, match_any):
            continue
        if robot_family_norm and robot_family_norm not in dp_robot_families:
            continue
        if objective_norm and (not dp_objective or objective_norm not in dp_objective):
            continue

        results.append(dp)
        if limit is not None and len(results) >= limit:
            break
    return results


def find_scenarios(
    store: OntologyStore,
    *,
    datapack_tags: Sequence[str] | None = None,
    task_tags: Sequence[str] | None = None,
    robot_families: Sequence[str] | None = None,
    objective_name: str | None = None,
    motor_backend: str | None = None,
    limit: int | None = None,
    match_any: bool = False,
) -> list[Mapping[str, Any]]:
    records = store.list_scenarios()
    dp_tag_set = _normalize_set(datapack_tags)
    task_tag_set = _normalize_set(task_tags)
    robot_family_set = _normalize_set(robot_families)
    objective_norm = _normalize_text(objective_name)
    backend_norm = _normalize_text(motor_backend)

    results: list[Mapping[str, Any]] = []
    for rec in records:
        rec_dp_tags = _normalize_set(rec.get("datapack_tags") or [])
        rec_task_tags = _normalize_set(rec.get("task_tags") or [])
        rec_robot_families = _normalize_set(rec.get("robot_families") or [])
        rec_objective = _normalize_text(rec.get("objective_name"))
        rec_backend = _normalize_text(rec.get("motor_backend"))

        if dp_tag_set and not _match_set(dp_tag_set, rec_dp_tags, match_any):
            continue
        if task_tag_set and not _match_set(task_tag_set, rec_task_tags, match_any):
            continue
        if robot_family_set and not _match_set(robot_family_set, rec_robot_families, match_any):
            continue
        if objective_norm and rec_objective != objective_norm:
            continue
        if backend_norm and rec_backend != backend_norm:
            continue

        results.append(rec)
        if limit is not None and len(results) >= limit:
            break
    return results


def _normalize_set(values: Sequence[str] | None) -> set[str]:
    if not values:
        return set()
    return {v.strip().lower() for v in values if v and str(v).strip()}


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _tags_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        for key in ("semantic_tags", "tags"):
            tags = payload.get(key)
            if tags:
                return list(tags)
    if isinstance(payload, (list, tuple)):
        return list(payload)
    return []


def _match_set(required: set[str], available: set[str], match_any: bool) -> bool:
    if match_any:
        return bool(required & available)
    return required.issubset(available)
