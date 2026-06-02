"""Deterministic replay dataset builder and loader."""
from __future__ import annotations

import json
from dataclasses import replace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.economics.inferential_contract import (
    build_inferential_learnability_contract,
    coerce_inferential_learnability_contract,
    summarize_inferential_learnability_contracts,
)
from src.replay.ingest import (
    REPLAY_SCHEMA_VERSION,
    ingest_rollout_bundle,
    ingest_shadow_run,
    ingest_workcell_episode_log,
)
from src.replay.importers import (
    ingest_governed_video_admission_log,
    ingest_semantic_degraded_artifacts,
)
from src.replay.compatibility import (
    build_artifact_schema_fingerprint,
    check_artifact_schema_versions,
    check_replay_manifest_compatibility,
)
from src.replay.schema import (
    ReplayDatasetManifest,
    ReplayEpisodeRecord,
    ReplayStepRecord,
    ReplayWindowRecord,
)
from src.replay.preconditions import (
    build_replay_execution_preconditions,
    summarize_replay_execution_preconditions,
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

    def add_rollout_bundle(
        self,
        rollout_root: str | Path,
        *,
        scenario_id: Optional[str] = None,
        run_id: Optional[str] = None,
        source_domain: str = "synthetic",
        objective_profile_id: str = "balanced_contract",
    ) -> "ReplayDatasetBuilder":
        episodes, steps, windows, metadata = ingest_rollout_bundle(
            rollout_root,
            scenario_id=scenario_id,
            run_id=run_id,
            source_domain=source_domain,
            objective_profile_id=objective_profile_id,
        )
        self._episodes.extend(episodes)
        self._steps.extend(steps)
        self._windows.extend(windows)
        self._source_adapters.append("rollout_capture_bundle_v1")
        self._metadata_rows.append(dict(metadata))
        return self

    def add_governed_video_admission_log(
        self,
        admission_log_path: str | Path,
        *,
        run_id: Optional[str] = None,
        source_domain: str = "governed_video_admission",
        objective_profile_id: str = "balanced_contract",
    ) -> "ReplayDatasetBuilder":
        episodes, steps, windows, metadata = ingest_governed_video_admission_log(
            admission_log_path,
            run_id=run_id,
            source_domain=source_domain,
            objective_profile_id=objective_profile_id,
        )
        self._episodes.extend(episodes)
        self._steps.extend(steps)
        self._windows.extend(windows)
        self._source_adapters.append("governed_video_admission_log_v1")
        self._metadata_rows.append(dict(metadata))
        return self

    def add_semantic_degraded_artifacts(
        self,
        root: str | Path,
        *,
        run_id: Optional[str] = None,
        source_domain: str = "semantic_negative_supervision",
        objective_profile_id: str = "balanced_contract",
    ) -> "ReplayDatasetBuilder":
        episodes, steps, windows, metadata = ingest_semantic_degraded_artifacts(
            root,
            run_id=run_id,
            source_domain=source_domain,
            objective_profile_id=objective_profile_id,
        )
        self._episodes.extend(episodes)
        self._steps.extend(steps)
        self._windows.extend(windows)
        self._source_adapters.append("semantic_degraded_import_v1")
        self._metadata_rows.append(dict(metadata))
        return self

    def add_rlds_episode(self, payload: Mapping[str, Any]) -> "ReplayDatasetBuilder":
        from src.dataset_bridges.rlds_bridge import replay_episode_from_rlds

        episode, steps = replay_episode_from_rlds(payload)
        windows = _rehydrated_windows_from_steps(episode, steps)
        self._episodes.append(episode)
        self._steps.extend(steps)
        self._windows.extend(windows)
        self._source_adapters.append("rlds_bridge_rehydration_v1")
        self._metadata_rows.append(
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "source_adapter": "rlds_bridge_rehydration_v1",
                "episode_id": episode.episode_id,
                "run_id": episode.run_id,
                "step_count": len(steps),
                "window_count": len(windows),
            }
        )
        return self

    def add_lerobot_rows(self, rows: Sequence[Mapping[str, Any]]) -> "ReplayDatasetBuilder":
        from src.dataset_bridges.lerobot_bridge import replay_episode_from_lerobot

        episode, steps = replay_episode_from_lerobot(rows)
        windows = _rehydrated_windows_from_steps(episode, steps)
        self._episodes.append(episode)
        self._steps.extend(steps)
        self._windows.extend(windows)
        self._source_adapters.append("lerobot_bridge_rehydration_v1")
        self._metadata_rows.append(
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "source_adapter": "lerobot_bridge_rehydration_v1",
                "episode_id": episode.episode_id,
                "run_id": episode.run_id,
                "step_count": len(steps),
                "window_count": len(windows),
            }
        )
        return self

    def build(self) -> ReplayDatasetBundle:
        episodes = sorted(self._episodes, key=lambda row: (row.run_id, row.episode_id))
        steps = sorted(self._steps, key=lambda row: (row.run_id, row.episode_id, row.step_idx))
        windows = sorted(self._windows, key=lambda row: (row.run_id, row.episode_id, row.start_step, row.window_id))
        steps_by_episode: Dict[str, List[ReplayStepRecord]] = {}
        for row in steps:
            steps_by_episode.setdefault(row.episode_id, []).append(row)
        windows_by_episode: Dict[str, List[ReplayWindowRecord]] = {}
        for window_row in windows:
            windows_by_episode.setdefault(window_row.episode_id, []).append(window_row)
        execution_precondition_reports = [
            build_replay_execution_preconditions(
                episode,
                steps=steps_by_episode.get(episode.episode_id, []),
                windows=windows_by_episode.get(episode.episode_id, []),
            )
            for episode in episodes
        ]
        reports_by_episode = {
            report.subject_id: report
            for report in execution_precondition_reports
        }
        enriched_contracts = []
        enriched_episodes = []
        for episode in episodes:
            execution_report = reports_by_episode[episode.episode_id]
            execution_preconditions = execution_report.to_dict()
            existing_contract = coerce_inferential_learnability_contract(
                episode.metadata.get("inferential_learnability_contract")
            )
            if existing_contract is None:
                future_signals = dict(execution_preconditions.get("metadata", {}).get("future_training_signals", {}) or {})
                quality_score = float(episode.datapack_summary.get("quality_score", 0.0) or 0.0)
                pricing_confidence = float(episode.pricing_summary.get("confidence", 0.0) or 0.0)
                epi_delta = float(
                    episode.datapack_summary.get(
                        "delta_epi_per_flop",
                        episode.datapack_summary.get("delta_epi_vs_baseline", 0.0),
                    )
                    or 0.0
                )
                epi_conf = float(episode.datapack_summary.get("epi_confidence", 0.0) or 0.0)
                existing_contract = build_inferential_learnability_contract(
                    subject_id=episode.episode_id,
                    subject_kind="replay_episode",
                    datapack_id=str(
                        episode.metadata.get("datapack_id")
                        or episode.datapack_summary.get("datapack_id")
                        or episode.episode_id
                    ),
                    frontier_gain=float(
                        episode.datapack_summary.get("marginal_frontier_gain", 0.0) or 0.0
                    ),
                    epiplexity_delta=epi_delta,
                    epiplexity_confidence=epi_conf,
                    transfer_score=max(
                        0.0,
                        1.0 - float(episode.condition_vector.get("ood_risk_level", 0.0) or 0.0),
                    ),
                    data_quality=quality_score,
                    provenance_quality=pricing_confidence,
                    trust_score=pricing_confidence,
                    overlay_joined=bool(episode.metadata.get("epiplexity_overlay_joined", False)),
                    benchmark_eligible=bool(future_signals.get("benchmark_eligible", False)),
                    semantic_grounding_non_heuristic=bool(
                        future_signals.get("semantic_grounding_non_heuristic", False)
                    ),
                    promotion_trace_complete=bool(
                        future_signals.get("promotion_trace_complete", False)
                    ),
                    budget_settlement_live=bool(
                        future_signals.get("budget_settlement_live", False)
                    ),
                    metadata={
                        "source": "replay_dataset_builder",
                        "source_domain": episode.source_domain,
                    },
                )
            metadata = {
                **dict(episode.metadata),
                "execution_preconditions": execution_preconditions,
                "inferential_learnability_contract": existing_contract.to_dict(),
            }
            enriched_contracts.append(existing_contract)
            enriched_episodes.append(replace(episode, metadata=metadata))
        episodes = enriched_episodes
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
        artifact_payload = {
            str(row.get("source_adapter", f"source_{index}")): {
                "schema_version": str(row.get("schema_version", "")),
                "config_digest": sha256_json(row),
                "dataset_digest": dataset_digest,
            }
            for index, row in enumerate(self._metadata_rows)
        }
        compatibility = check_artifact_schema_versions(
            artifact_payload,
            required_versions={key: REPLAY_SCHEMA_VERSION for key in artifact_payload},
        )
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
            metadata={
                "sources": list(self._metadata_rows),
                "schema_compatibility": [row.to_dict() for row in compatibility],
                "execution_precondition_summary": summarize_replay_execution_preconditions(
                    execution_precondition_reports
                ),
                "inferential_learnability_summary": summarize_inferential_learnability_contracts(
                    [row.to_dict() for row in enriched_contracts]
                ),
            },
            artifact_schema_fingerprint=build_artifact_schema_fingerprint(artifact_payload),
            provenance_summary={
                "source_roots": sorted(
                    {
                        str(row.get("source_root") or row.get("source_path") or row.get("scenario_id") or "")
                        for row in self._metadata_rows
                    }
                ),
                "source_adapter_count": len(set(self._source_adapters)),
            },
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
    compatibility = check_replay_manifest_compatibility(manifest, expected_schema_version=REPLAY_SCHEMA_VERSION)
    if not compatibility.compatible:
        raise ValueError(f"Replay dataset manifest incompatible: {compatibility.reasons}")
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


def _rehydrated_windows_from_steps(
    episode: ReplayEpisodeRecord,
    steps: Sequence[ReplayStepRecord],
) -> list[ReplayWindowRecord]:
    if not steps:
        return []
    obs_dims = max((len(step.obs_vector) for step in steps), default=0)
    action_dims = max((len(step.action_vector) for step in steps), default=0)
    return [
        ReplayWindowRecord(
            run_id=episode.run_id,
            episode_id=episode.episode_id,
            window_id=f"rehydrated_{steps[0].step_idx:04d}_{steps[-1].step_idx:04d}",
            start_step=int(steps[0].step_idx),
            end_step=int(steps[-1].step_idx),
            task_id=episode.task_id,
            env_id=episode.env_id,
            source_domain=episode.source_domain,
            seed=episode.seed,
            timestamp=str(steps[0].timestamp),
            reward_sum=sum(step.reward for step in steps),
            obs_vector_mean=_mean_vector([step.obs_vector for step in steps], obs_dims),
            action_vector_mean=_mean_vector([step.action_vector for step in steps], action_dims),
            condition_vector=dict(episode.condition_vector),
            condition_vector_values=list(episode.condition_vector_values),
            skill_mode=str(episode.skill_mode),
            objective_tensor_summary=dict(episode.objective_tensor_summary),
            econ_tensor_summary=dict(episode.econ_tensor_summary),
            pricing_summary=dict(episode.pricing_summary),
            constraint_flags=[dict(flag) for flag in episode.constraint_flags],
            metadata={
                "event_refs": list(episode.metadata.get("event_refs", []) or []),
                "decision_refs": list(episode.metadata.get("decision_refs", []) or []),
                "rehydrated": True,
            },
            provenance=dict(episode.provenance),
        )
    ]


def _mean_vector(vectors: Sequence[Sequence[float]], dim: int) -> list[float]:
    if dim <= 0:
        return []
    if not vectors:
        return [0.0] * dim
    totals = [0.0] * dim
    for vector in vectors:
        for index in range(min(len(vector), dim)):
            totals[index] += float(vector[index])
    return [value / float(len(vectors)) for value in totals]
