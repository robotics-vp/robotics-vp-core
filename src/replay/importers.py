"""Importer-side replay adapters for governed-video and degraded semantic artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.evidence.benchmark_gating import collect_benchmark_gating_signals
from src.replay.ingest import REPLAY_SCHEMA_VERSION
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord, ReplayWindowRecord
from src.utils.config_digest import sha256_json


def _json_rows(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _timestamp_from_path(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_artifact_refs(payload: Mapping[str, Any]) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        if value in (None, "", [], {}):
            continue
        normalized_key = str(key)
        refs[normalized_key] = value
        if normalized_key.endswith("_path"):
            refs[f"{normalized_key[:-5]}_ref"] = value
        elif normalized_key.endswith("_paths"):
            refs[f"{normalized_key[:-6]}_refs"] = value
    return dict(sorted(refs.items()))


def _future_signals_from_admission(
    row: Mapping[str, Any],
    *,
    artifact_refs: Mapping[str, Any],
) -> Dict[str, bool]:
    explicit = row.get("future_training_signals", {})
    if not isinstance(explicit, Mapping):
        explicit = {}
    semantic_world_model_path = artifact_refs.get(
        "semantic_world_model_ref"
    ) or artifact_refs.get("semantic_world_model_path")
    grounded_track_count = 0
    if semantic_world_model_path:
        try:
            payload = _json_object(Path(str(semantic_world_model_path)))
            topology = payload.get("topology", {})
            if isinstance(topology, Mapping):
                grounded_track_count = int(
                    topology.get("grounded_track_object_count", 0) or 0
                )
        except Exception:
            grounded_track_count = 0
    benchmark_signals = collect_benchmark_gating_signals(row)
    derived = {
        "replay_roundtrip_complete": True,
        "promotion_trace_complete": bool(
            artifact_refs.get("event_spine_ref")
            and artifact_refs.get("decision_ledger_ref")
            and artifact_refs.get("governance_trace_ref")
            and artifact_refs.get("counterfactual_eval_ref")
            and artifact_refs.get("value_target_pack_ref")
        ),
        "teacher_runtime_live": bool(
            artifact_refs.get("teacher_trace_ref")
            or artifact_refs.get("teacher_trace_path")
            or artifact_refs.get("teacher_contract_ref")
            or artifact_refs.get("teacher_contract_path")
            or artifact_refs.get("teacher_action_ref")
            or artifact_refs.get("teacher_action_path")
            or artifact_refs.get("teacher_action_envelope_ref")
            or artifact_refs.get("teacher_action_envelope_path")
        ),
        "scene_tracks_non_stub": bool(row.get("scene_tracks_non_stub", False)),
        "semantic_memory_grounded": grounded_track_count > 0,
        "budget_settlement_live": False,
        "teacher_runtime_real": bool(
            benchmark_signals.get("teacher_runtime_real", False)
        ),
        "vision_backbone_real": bool(
            benchmark_signals.get("vision_backbone_real", False)
        ),
        "semantic_grounding_non_heuristic": bool(
            benchmark_signals.get("semantic_grounding_non_heuristic", False)
        ),
        "benchmark_eligible": bool(benchmark_signals.get("benchmark_eligible", False)),
    }
    monotonic_explicit_keys = {
        "scene_tracks_non_stub",
        "semantic_memory_grounded",
        "budget_settlement_live",
    }
    for key, value in explicit.items():
        normalized_key = str(key)
        if normalized_key in monotonic_explicit_keys:
            derived[normalized_key] = bool(derived.get(normalized_key, False) or value)
        elif normalized_key not in derived:
            derived[normalized_key] = bool(value)
    return dict(sorted(derived.items()))


def _future_signals_from_degraded(
    payload: Mapping[str, Any],
    *,
    artifact_refs: Mapping[str, Any],
) -> Dict[str, bool]:
    explicit = payload.get("future_training_signals", {})
    if not isinstance(explicit, Mapping):
        explicit = {}
    benchmark_signals = collect_benchmark_gating_signals(payload)
    derived = {
        "replay_roundtrip_complete": True,
        "promotion_trace_complete": False,
        "teacher_runtime_live": bool(
            artifact_refs.get("teacher_trace_ref")
            or artifact_refs.get("teacher_trace_path")
            or artifact_refs.get("teacher_contract_ref")
            or artifact_refs.get("teacher_contract_path")
            or artifact_refs.get("teacher_action_ref")
            or artifact_refs.get("teacher_action_path")
            or artifact_refs.get("teacher_action_envelope_ref")
            or artifact_refs.get("teacher_action_envelope_path")
        ),
        "scene_tracks_non_stub": bool(payload.get("scene_tracks_non_stub", False)),
        "semantic_memory_grounded": False,
        "budget_settlement_live": False,
        "teacher_runtime_real": bool(
            benchmark_signals.get("teacher_runtime_real", False)
        ),
        "vision_backbone_real": bool(
            benchmark_signals.get("vision_backbone_real", False)
        ),
        "semantic_grounding_non_heuristic": bool(
            benchmark_signals.get("semantic_grounding_non_heuristic", False)
        ),
        "benchmark_eligible": bool(benchmark_signals.get("benchmark_eligible", False)),
    }
    monotonic_explicit_keys = {
        "scene_tracks_non_stub",
        "semantic_memory_grounded",
        "budget_settlement_live",
    }
    for key, value in explicit.items():
        normalized_key = str(key)
        if normalized_key in monotonic_explicit_keys:
            derived[normalized_key] = bool(derived.get(normalized_key, False) or value)
        elif normalized_key not in derived:
            derived[normalized_key] = bool(value)
    return dict(sorted(derived.items()))


def _future_artifacts(payload: Mapping[str, Any]) -> Dict[str, Any]:
    explicit = payload.get("future_training_artifacts", {})
    if not isinstance(explicit, Mapping):
        return {}
    return {
        str(key): value
        for key, value in dict(explicit).items()
        if value not in (None, "", [], {})
    }


def _single_window(
    episode: ReplayEpisodeRecord, step: ReplayStepRecord
) -> ReplayWindowRecord:
    return ReplayWindowRecord(
        run_id=episode.run_id,
        episode_id=episode.episode_id,
        window_id=f"{episode.episode_id}:0000_0000",
        start_step=0,
        end_step=0,
        task_id=episode.task_id,
        env_id=episode.env_id,
        source_domain=episode.source_domain,
        seed=episode.seed,
        timestamp=step.timestamp,
        reward_sum=float(step.reward),
        obs_vector_mean=list(step.obs_vector),
        action_vector_mean=list(step.action_vector),
        condition_vector=dict(step.condition_vector),
        condition_vector_values=list(step.condition_vector_values),
        skill_mode=episode.skill_mode,
        objective_tensor_summary=dict(episode.objective_tensor_summary),
        econ_tensor_summary=dict(episode.econ_tensor_summary),
        pricing_summary=dict(episode.pricing_summary),
        constraint_flags=[dict(flag) for flag in episode.constraint_flags],
        metadata={
            "event_refs": list(episode.metadata.get("event_refs", []) or []),
            "decision_refs": list(episode.metadata.get("decision_refs", []) or []),
            "source_adapter": episode.metadata.get("source_adapter", ""),
        },
        provenance=dict(episode.provenance),
    )


def ingest_governed_video_admission_log(
    log_path: str | Path,
    *,
    run_id: Optional[str] = None,
    source_domain: str = "governed_video_admission",
    objective_profile_id: str = "balanced_contract",
) -> tuple[
    list[ReplayEpisodeRecord],
    list[ReplayStepRecord],
    list[ReplayWindowRecord],
    Dict[str, Any],
]:
    """Convert governed-video proposal admission logs into canonical replay rows."""

    path = Path(log_path)
    rows = _json_rows(path)
    resolved_run_id = (
        run_id or f"governed_video_import_{sha256_json({'log_path': str(path)})[:12]}"
    )
    timestamp = _timestamp_from_path(path)
    episodes: list[ReplayEpisodeRecord] = []
    steps: list[ReplayStepRecord] = []
    windows: list[ReplayWindowRecord] = []

    for index, row in enumerate(rows):
        video_id = str(row.get("video_id", f"video_{index:04d}"))
        proposal_id = str(row.get("proposal_id", f"proposal_{index:04d}"))
        episode_id = f"{video_id}:{proposal_id}"
        artifact_refs = _normalize_artifact_refs(
            {
                key: value
                for key, value in dict(row).items()
                if str(key).endswith(
                    ("_path", "_paths", "_ref", "_refs", "_id", "_ids")
                )
            }
        )
        source_preconditions = dict(row.get("execution_preconditions", {}) or {})
        source_work_order = dict(row.get("execution_work_order", {}) or {})
        source_benchmark_gate = dict(row.get("benchmark_gate", {}) or {})
        future_training_signals = _future_signals_from_admission(
            row, artifact_refs=artifact_refs
        )
        future_training_artifacts = _future_artifacts(row)
        blocked = bool(row.get("blocked", False))
        plausibility_gate = (
            row.get("plausibility_gate", {})
            if isinstance(row.get("plausibility_gate"), Mapping)
            else {}
        )
        plausibility_details = (
            plausibility_gate.get("details", {})
            if isinstance(plausibility_gate.get("details"), Mapping)
            else {}
        )
        plausibility_score = _as_float(
            plausibility_details.get("plausibility_score"),
            _as_float(source_preconditions.get("readiness_score"), 0.0),
        )
        readiness_score = _as_float(source_preconditions.get("readiness_score"), 0.0)
        work_order_ready = bool(
            source_work_order.get("ready", source_preconditions.get("ready", False))
        )
        obs_vector = [
            plausibility_score,
            readiness_score,
            1.0 if blocked else 0.0,
            1.0 if work_order_ready else 0.0,
            1.0
            if future_training_signals.get("promotion_trace_complete", False)
            else 0.0,
            1.0
            if future_training_signals.get("semantic_memory_grounded", False)
            else 0.0,
        ]
        action_vector = [
            1.0 if source_work_order.get("decision") == "admit_datapack" else 0.0,
            1.0 if blocked else 0.0,
        ]
        condition_vector = {
            "blocked": blocked,
            "promotion_trace_complete": future_training_signals.get(
                "promotion_trace_complete", False
            ),
            "semantic_memory_grounded": future_training_signals.get(
                "semantic_memory_grounded", False
            ),
            "teacher_runtime_live": future_training_signals.get(
                "teacher_runtime_live", False
            ),
            "scene_tracks_non_stub": future_training_signals.get(
                "scene_tracks_non_stub", False
            ),
        }
        row_metadata: Dict[str, Any] = {
            "source_adapter": "governed_video_admission_log_v1",
            "video_id": video_id,
            "proposal_id": proposal_id,
            "blocked": blocked,
            "source_execution_preconditions": source_preconditions,
            "source_execution_work_order": source_work_order,
            "source_benchmark_gate": source_benchmark_gate,
            "benchmark_gate": source_benchmark_gate,
            "future_training_signals": future_training_signals,
            "future_training_artifacts": future_training_artifacts,
            "event_refs": [artifact_refs["event_spine_ref"]]
            if artifact_refs.get("event_spine_ref")
            else [],
            "decision_refs": [artifact_refs["decision_ledger_ref"]]
            if artifact_refs.get("decision_ledger_ref")
            else [],
            "semantic_world_model_summary": _json_object(
                Path(str(artifact_refs["semantic_world_model_ref"]))
            )
            if artifact_refs.get("semantic_world_model_ref")
            else {},
        }
        provenance = {
            "source_adapter": "governed_video_admission_log_v1",
            "proposal_admission_log_ref": str(path),
            **artifact_refs,
        }
        episode = ReplayEpisodeRecord(
            run_id=resolved_run_id,
            episode_id=episode_id,
            task_id=video_id,
            env_id="stage1_governed_video",
            source_domain=source_domain,
            seed=0,
            status="blocked" if blocked else "admitted",
            started_at=timestamp,
            ended_at=timestamp,
            total_steps=1,
            total_reward=0.0 if blocked else float(work_order_ready),
            skill_mode="review" if blocked else "exploration",
            condition_vector=condition_vector,
            condition_vector_values=list(obs_vector),
            objective_tensor_summary={"objective_profile_id": objective_profile_id},
            objective_tensor_ref=None,
            econ_tensor_summary={"plausibility_score": plausibility_score},
            econ_tensor_ref=None,
            pricing_summary={"plausibility_gate": plausibility_gate},
            pricing_tick_refs=[artifact_refs["pricing_tick_ref"]]
            if artifact_refs.get("pricing_tick_ref")
            else [],
            constraint_flags=[
                {"code": reason, "severity": "hard" if blocked else "info"}
                for reason in list(plausibility_gate.get("reason_codes", []) or [])
            ],
            regal_summary={"plausibility_gate": plausibility_gate},
            datapack_summary={
                "blocked": blocked,
                "work_order_id": source_work_order.get("work_order_id"),
                "counterfactual_eval_id": row.get("counterfactual_eval_id"),
                "value_target_pack_id": row.get("value_target_pack_id"),
            },
            ledger_event_ids=[],
            metadata=row_metadata,
            provenance=provenance,
        )
        step = ReplayStepRecord(
            run_id=resolved_run_id,
            episode_id=episode_id,
            step_idx=0,
            obs={"plausibility_gate": plausibility_gate, "blocked": blocked},
            obs_vector=obs_vector,
            action={
                "decision": source_work_order.get(
                    "decision",
                    "capture_negative_supervision" if blocked else "admit_datapack",
                )
            },
            action_vector=action_vector,
            reward=0.0 if blocked else float(work_order_ready),
            reward_decomposition={
                "admission_ready": float(work_order_ready),
                "blocked": 1.0 if blocked else 0.0,
            },
            done=True,
            task_id=video_id,
            env_id="stage1_governed_video",
            condition_vector=condition_vector,
            condition_vector_values=list(obs_vector),
            skill_mode=episode.skill_mode,
            objective_tensor_summary=dict(episode.objective_tensor_summary),
            objective_tensor_ref=None,
            econ_tensor_summary=dict(episode.econ_tensor_summary),
            econ_tensor_ref=None,
            constraint_flags=[dict(flag) for flag in episode.constraint_flags],
            pricing_tick_ref=artifact_refs.get("pricing_tick_ref"),
            ledger_event_ref=artifact_refs.get("value_ledger_receipt_ref"),
            source_domain=source_domain,
            seed=0,
            timestamp=timestamp,
            metadata=row_metadata,
            provenance=provenance,
        )
        episodes.append(episode)
        steps.append(step)
        windows.append(_single_window(episode, step))

    summary_metadata: Dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_adapter": "governed_video_admission_log_v1",
        "source_path": str(path),
        "run_id": resolved_run_id,
        "episode_count": len(episodes),
        "blocked_count": sum(1 for row in rows if row.get("blocked")),
        "admitted_count": sum(1 for row in rows if not row.get("blocked")),
    }
    return episodes, steps, windows, summary_metadata


def ingest_semantic_degraded_artifacts(
    root: str | Path,
    *,
    run_id: Optional[str] = None,
    source_domain: str = "semantic_negative_supervision",
    objective_profile_id: str = "balanced_contract",
) -> tuple[
    list[ReplayEpisodeRecord],
    list[ReplayStepRecord],
    list[ReplayWindowRecord],
    Dict[str, Any],
]:
    """Convert semantic degraded artifacts into canonical replay rows."""

    path = Path(root)
    if path.is_file():
        artifact_paths = [path]
    else:
        artifact_paths = sorted(path.rglob("*_semantic_degraded_v1.json"))
    resolved_run_id = (
        run_id or f"semantic_degraded_import_{sha256_json({'root': str(path)})[:12]}"
    )
    episodes: list[ReplayEpisodeRecord] = []
    steps: list[ReplayStepRecord] = []
    windows: list[ReplayWindowRecord] = []

    for artifact_path in artifact_paths:
        payload = _json_object(artifact_path)
        if not payload:
            continue
        episode_id = str(payload.get("episode_id", artifact_path.stem))
        failure_reason = str(payload.get("failure_reason", "semantic_degraded"))
        artifact_refs = _normalize_artifact_refs(
            dict(payload.get("artifact_refs", {}) or {})
        )
        future_training_signals = _future_signals_from_degraded(
            payload, artifact_refs=artifact_refs
        )
        future_training_artifacts = _future_artifacts(payload)
        source_preconditions = dict(payload.get("execution_preconditions", {}) or {})
        source_work_order = dict(payload.get("execution_work_order", {}) or {})
        source_benchmark_gate = dict(payload.get("benchmark_gate", {}) or {})
        timestamp = _timestamp_from_path(artifact_path)
        obs_vector = [
            _as_float(source_preconditions.get("readiness_score"), 0.0),
            1.0,
            1.0 if future_training_signals.get("teacher_runtime_live", False) else 0.0,
            1.0 if future_training_signals.get("scene_tracks_non_stub", False) else 0.0,
        ]
        row_metadata: Dict[str, Any] = {
            "source_adapter": "semantic_degraded_import_v1",
            "failure_reason": failure_reason,
            "source_execution_preconditions": source_preconditions,
            "source_execution_work_order": source_work_order,
            "source_benchmark_gate": source_benchmark_gate,
            "benchmark_gate": source_benchmark_gate,
            "future_training_signals": future_training_signals,
            "future_training_artifacts": future_training_artifacts,
            "event_refs": [artifact_refs["event_spine_ref"]]
            if artifact_refs.get("event_spine_ref")
            else [],
            "decision_refs": [artifact_refs["decision_ledger_ref"]]
            if artifact_refs.get("decision_ledger_ref")
            else [],
        }
        provenance = {
            "source_adapter": "semantic_degraded_import_v1",
            "semantic_degraded_artifact_ref": str(artifact_path),
            **artifact_refs,
        }
        episode = ReplayEpisodeRecord(
            run_id=resolved_run_id,
            episode_id=episode_id,
            task_id=episode_id,
            env_id="semantic_fusion_runner",
            source_domain=source_domain,
            seed=0,
            status="blocked",
            started_at=timestamp,
            ended_at=timestamp,
            total_steps=1,
            total_reward=0.0,
            skill_mode="review",
            condition_vector={
                "failure_reason": failure_reason,
                "teacher_runtime_live": future_training_signals.get(
                    "teacher_runtime_live", False
                ),
                "scene_tracks_non_stub": future_training_signals.get(
                    "scene_tracks_non_stub", False
                ),
            },
            condition_vector_values=list(obs_vector),
            objective_tensor_summary={"objective_profile_id": objective_profile_id},
            objective_tensor_ref=None,
            econ_tensor_summary={"failure_reason": failure_reason},
            econ_tensor_ref=None,
            pricing_summary={},
            pricing_tick_refs=[],
            constraint_flags=[{"code": failure_reason, "severity": "hard"}],
            regal_summary={"failure_reason": failure_reason},
            datapack_summary={
                "negative_supervision": True,
                "work_order_id": source_work_order.get("work_order_id"),
            },
            ledger_event_ids=[],
            metadata=row_metadata,
            provenance=provenance,
        )
        step = ReplayStepRecord(
            run_id=resolved_run_id,
            episode_id=episode_id,
            step_idx=0,
            obs={"failure_reason": failure_reason},
            obs_vector=obs_vector,
            action={
                "decision": source_work_order.get(
                    "decision", "capture_negative_supervision"
                )
            },
            action_vector=[0.0, 1.0],
            reward=0.0,
            reward_decomposition={"negative_supervision": 1.0},
            done=True,
            task_id=episode.task_id,
            env_id=episode.env_id,
            condition_vector=dict(episode.condition_vector),
            condition_vector_values=list(obs_vector),
            skill_mode=episode.skill_mode,
            objective_tensor_summary=dict(episode.objective_tensor_summary),
            objective_tensor_ref=None,
            econ_tensor_summary=dict(episode.econ_tensor_summary),
            econ_tensor_ref=None,
            constraint_flags=[dict(flag) for flag in episode.constraint_flags],
            pricing_tick_ref=None,
            ledger_event_ref=None,
            source_domain=source_domain,
            seed=0,
            timestamp=timestamp,
            metadata=row_metadata,
            provenance=provenance,
        )
        episodes.append(episode)
        steps.append(step)
        windows.append(_single_window(episode, step))

    summary_metadata: Dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_adapter": "semantic_degraded_import_v1",
        "source_root": str(path),
        "run_id": resolved_run_id,
        "episode_count": len(episodes),
        "failure_count": len(episodes),
    }
    return episodes, steps, windows, summary_metadata


__all__ = [
    "ingest_governed_video_admission_log",
    "ingest_semantic_degraded_artifacts",
]
