"""Deterministic replay dataset builder and loader."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.replay.ingest import (
    REPLAY_SCHEMA_VERSION,
    ingest_shadow_run,
    ingest_workcell_episode_log,
)
from src.replay.schema import (
    ReplayDatasetManifest,
    ReplayEpisodeRecord,
    ReplayStepRecord,
    ReplayWindowRecord,
)
from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class ReplayDatasetBundle:
    """In-memory canonical replay dataset."""

    manifest: ReplayDatasetManifest
    episodes: List[ReplayEpisodeRecord]
    steps: List[ReplayStepRecord]
    windows: List[ReplayWindowRecord]
    root_dir: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.manifest.schema_version,
            "num_episodes": self.manifest.num_episodes,
            "num_steps": self.manifest.num_steps,
            "num_windows": self.manifest.num_windows,
            "obs_dim": self.manifest.obs_dim,
            "action_dim": self.manifest.action_dim,
            "condition_dim": self.manifest.condition_dim,
            "run_ids": list(self.manifest.run_ids),
            "skill_modes": list(self.manifest.skill_modes),
            "dataset_digest": self.manifest.dataset_digest,
        }


class ReplayDatasetBuilder:
    """Build and persist canonical replay datasets from supported adapters."""

    def __init__(self) -> None:
        self._episodes: List[ReplayEpisodeRecord] = []
        self._steps: List[ReplayStepRecord] = []
        self._windows: List[ReplayWindowRecord] = []
        self._source_adapters: List[str] = []
        self._metadata_rows: List[Dict[str, Any]] = []

    def add_shadow_run(self, run_dir: str | Path) -> "ReplayDatasetBuilder":
        episodes, steps, windows, metadata = ingest_shadow_run(run_dir)
        self._episodes.extend(episodes)
        self._steps.extend(steps)
        self._windows.extend(windows)
        self._source_adapters.append("shadow_control_plane_artifacts_v1")
        self._metadata_rows.append(dict(metadata))
        return self

    def add_workcell_episode_log(
        self,
        episode_log_path: str | Path,
        *,
        run_id: Optional[str] = None,
        source_domain: str = "synthetic",
        objective_profile_id: str = "balanced_contract",
    ) -> "ReplayDatasetBuilder":
        episodes, steps, windows, metadata = ingest_workcell_episode_log(
            episode_log_path,
            run_id=run_id,
            source_domain=source_domain,
            objective_profile_id=objective_profile_id,
        )
        self._episodes.extend(episodes)
        self._steps.extend(steps)
        self._windows.extend(windows)
        self._source_adapters.append("workcell_episode_log_v1")
        self._metadata_rows.append(dict(metadata))
        return self

    def build(self) -> ReplayDatasetBundle:
        episodes = sorted(self._episodes, key=lambda row: (row.run_id, row.episode_id))
        steps = sorted(self._steps, key=lambda row: (row.run_id, row.episode_id, row.step_idx))
        windows = sorted(self._windows, key=lambda row: (row.run_id, row.episode_id, row.start_step, row.window_id))
        run_ids = sorted({row.run_id for row in episodes})
        skill_modes = sorted({row.skill_mode for row in episodes} | {row.skill_mode for row in steps})
        obs_dim = max((len(row.obs_vector) for row in steps), default=0)
        action_dim = max((len(row.action_vector) for row in steps), default=0)
        condition_dim = max((len(row.condition_vector_values) for row in steps), default=0)
        digest_payload = {
            "episodes": [row.to_dict() for row in episodes],
            "steps": [row.to_dict() for row in steps],
            "windows": [row.to_dict() for row in windows],
        }
        dataset_digest = sha256_json(digest_payload)
        manifest = ReplayDatasetManifest(
            schema_version=REPLAY_SCHEMA_VERSION,
            run_ids=run_ids,
            source_adapters=sorted(set(self._source_adapters)),
            files={
                "episodes": "episodes.jsonl",
                "steps": "steps.jsonl",
                "windows": "windows.jsonl",
                "manifest": "manifest.json",
            },
            num_episodes=len(episodes),
            num_steps=len(steps),
            num_windows=len(windows),
            obs_dim=obs_dim,
            action_dim=action_dim,
            condition_dim=condition_dim,
            skill_modes=skill_modes,
            config_digest=sha256_json({"sources": self._metadata_rows}),
            dataset_digest=dataset_digest,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"sources": list(self._metadata_rows)},
        )
        return ReplayDatasetBundle(
            manifest=manifest,
            episodes=episodes,
            steps=steps,
            windows=windows,
        )

    def write(self, output_dir: str | Path) -> ReplayDatasetBundle:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        bundle = self.build()
        _write_jsonl(output_root / "episodes.jsonl", [row.to_dict() for row in bundle.episodes])
        _write_jsonl(output_root / "steps.jsonl", [row.to_dict() for row in bundle.steps])
        _write_jsonl(output_root / "windows.jsonl", [row.to_dict() for row in bundle.windows])
        manifest_payload = bundle.manifest.to_dict()
        manifest_payload["manifest_hash"] = bundle.manifest.manifest_hash
        (output_root / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_root / "summary.json").write_text(
            json.dumps(bundle.to_summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return ReplayDatasetBundle(
            manifest=bundle.manifest,
            episodes=bundle.episodes,
            steps=bundle.steps,
            windows=bundle.windows,
            root_dir=str(output_root),
        )


def load_replay_dataset(dataset_dir: str | Path) -> ReplayDatasetBundle:
    root = Path(dataset_dir)
    manifest_payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest = ReplayDatasetManifest.from_dict(manifest_payload)
    episodes = [ReplayEpisodeRecord.from_dict(row) for row in _load_jsonl(root / manifest.files["episodes"])]
    steps = [ReplayStepRecord.from_dict(row) for row in _load_jsonl(root / manifest.files["steps"])]
    windows = [ReplayWindowRecord.from_dict(row) for row in _load_jsonl(root / manifest.files["windows"])]
    return ReplayDatasetBundle(
        manifest=manifest,
        episodes=episodes,
        steps=steps,
        windows=windows,
        root_dir=str(root),
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
