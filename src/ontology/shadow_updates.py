"""Additive ontology/datapack persistence helpers for the shadow control plane."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.ontology.models import Datapack, Episode, Robot, Task
from src.ontology.store import OntologyStore
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


@dataclass(frozen=True)
class ShadowDatapackCreditUpdate:
    """Datapack-facing shadow value update."""

    datapack_id: str
    episode_id: str
    run_id: str
    objective_profile_id: str
    marginal_frontier_gain: float
    data_share_credit: float
    quality_score: float
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datapack_id": self.datapack_id,
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "objective_profile_id": self.objective_profile_id,
            "marginal_frontier_gain": float(self.marginal_frontier_gain),
            "data_share_credit": float(self.data_share_credit),
            "quality_score": float(self.quality_score),
            "recommendation": self.recommendation,
            "metadata": dict(self.metadata),
        }


def persist_shadow_episode(
    *,
    store: OntologyStore,
    sidecar_dir: str | Path,
    task_id: str,
    task_name: str,
    env_id: str,
    robot_id: str,
    robot_name: str,
    episode_id: str,
    run_id: str,
    source_domain: str,
    started_at: str,
    ended_at: str,
    status: str,
    objective_tensor: Mapping[str, Any],
    econ_tensor: Mapping[str, Any],
    pricing_summary: Mapping[str, Any],
    regal_summary: Mapping[str, Any],
    datapack_update: ShadowDatapackCreditUpdate,
    episode_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist additive shadow summaries into ontology records and sidecars."""

    task = Task(
        task_id=task_id,
        name=task_name,
        environment_id=env_id,
        metadata={"shadow_economic_control_plane_v1": {"run_id": run_id, "source_domain": source_domain}},
    )
    robot = Robot(
        robot_id=robot_id,
        name=robot_name,
        metadata={"shadow_economic_control_plane_v1": {"run_id": run_id, "source_domain": source_domain}},
    )
    episode = Episode(
        episode_id=episode_id,
        task_id=task_id,
        robot_id=robot_id,
        datapack_id=datapack_update.datapack_id,
        started_at=_coerce_datetime(started_at),
        ended_at=_coerce_datetime(ended_at),
        status=status,
        metadata={
            "shadow_economic_control_plane_v1": {
                "run_id": run_id,
                "source_domain": source_domain,
                "objective_profile_id": datapack_update.objective_profile_id,
                "pricing_summary": to_json_safe(pricing_summary),
                "regal_summary": to_json_safe(regal_summary),
                "episode_metadata": dict(episode_metadata or {}),
            }
        },
    )
    datapack = Datapack(
        datapack_id=datapack_update.datapack_id,
        source_type=source_domain,
        task_id=task_id,
        modality="state",
        storage_uri=f"{sidecar_dir}/episodes/{episode_id}",
        novelty_score=max(0.0, datapack_update.marginal_frontier_gain),
        quality_score=float(datapack_update.quality_score),
        metadata={
            "shadow_economic_control_plane_v1": {
                "run_id": run_id,
                "objective_profile_id": datapack_update.objective_profile_id,
                "data_share_credit": float(datapack_update.data_share_credit),
                "marginal_frontier_gain": float(datapack_update.marginal_frontier_gain),
                "recommendation": datapack_update.recommendation,
            }
        },
        tags={"shadow": True, "source_domain": source_domain},
    )

    store.upsert_task(task)
    store.upsert_robot(robot)
    store.upsert_episode(episode)
    store.upsert_objective_tensor(episode_id, objective_tensor)
    store.upsert_econ_tensor(episode_id, econ_tensor)
    store.append_datapacks([datapack])

    sidecar_root = Path(sidecar_dir)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    sidecar_payload = {
        "task": to_json_safe(task),
        "robot": to_json_safe(robot),
        "episode": to_json_safe(episode),
        "objective_tensor": to_json_safe(objective_tensor),
        "econ_tensor": to_json_safe(econ_tensor),
        "pricing_summary": to_json_safe(pricing_summary),
        "regal_summary": to_json_safe(regal_summary),
        "datapack_credit_update": datapack_update.to_dict(),
    }
    sidecar_hash = sha256_json(sidecar_payload)
    sidecar_path = sidecar_root / f"{episode_id}_shadow_sidecar.json"
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar_payload, handle, indent=2, sort_keys=True)

    return {
        "episode_id": episode_id,
        "datapack_id": datapack_update.datapack_id,
        "sidecar_path": str(sidecar_path),
        "sidecar_hash": sidecar_hash,
    }


def _coerce_datetime(timestamp: str):
    from datetime import datetime

    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
