"""Replay-specific trace completeness and self-improvement readiness helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from src.evidence.benchmark_gating import collect_benchmark_gating_signals
from src.evidence.preconditions import (
    ExecutionPreconditionsReport,
    build_execution_preconditions,
    summarize_execution_preconditions,
)
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord, ReplayWindowRecord


REPLAY_REQUIRED_ARTIFACT_REFS = (
    "runtime_packet_ref",
    "event_spine_ref",
    "decision_ledger_ref",
)


def collect_replay_artifact_refs(
    episode: ReplayEpisodeRecord,
    *,
    steps: Optional[Iterable[ReplayStepRecord]] = None,
    windows: Optional[Iterable[ReplayWindowRecord]] = None,
) -> Dict[str, Any]:
    """Collect canonical replay refs from episode/step/window envelopes."""

    refs: Dict[str, Any] = {}
    step_rows = list(steps or [])
    window_rows = list(windows or [])
    envelopes = [episode.provenance, episode.metadata]
    for step in step_rows:
        envelopes.extend([step.provenance, step.metadata])
    for window in window_rows:
        envelopes.extend([window.provenance, window.metadata])
    for payload in envelopes:
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            if key.endswith(
                ("_ref", "_refs", "_id", "_ids", "_path", "_paths")
            ) and value not in (None, "", [], {}):
                refs[str(key)] = value
    return refs


def _future_training_signals(
    episode: ReplayEpisodeRecord,
    refs: Mapping[str, Any],
) -> Dict[str, bool]:
    metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
    explicit = metadata.get("future_training_signals", {})
    if not isinstance(explicit, Mapping):
        explicit = {}
    source_adapter = str(
        metadata.get("source_adapter") or episode.provenance.get("source_adapter") or ""
    )
    semantic_world_model_summary = metadata.get("semantic_world_model_summary", {})
    grounded_track_count = 0
    if isinstance(semantic_world_model_summary, Mapping):
        topology = semantic_world_model_summary.get("topology", {})
        if isinstance(topology, Mapping):
            grounded_track_count = int(
                topology.get("grounded_track_object_count", 0) or 0
            )
    promotion_trace_complete = bool(
        refs.get("event_spine_ref")
        and refs.get("decision_ledger_ref")
        and refs.get("governance_trace_ref")
        and (
            refs.get("counterfactual_eval_ref") or refs.get("counterfactual_eval_path")
        )
        and (refs.get("value_target_pack_ref") or refs.get("value_target_pack_path"))
    )
    benchmark_signals = collect_benchmark_gating_signals(metadata)
    derived = {
        "replay_roundtrip_complete": source_adapter
        in {
            "rlds_bridge_rehydration_v1",
            "lerobot_bridge_rehydration_v1",
            "governed_video_admission_log_v1",
            "semantic_degraded_import_v1",
        },
        "promotion_trace_complete": promotion_trace_complete,
        "teacher_runtime_live": bool(
            refs.get("teacher_trace_ref")
            or refs.get("teacher_trace_path")
            or refs.get("teacher_contract_ref")
            or refs.get("teacher_contract_path")
            or refs.get("teacher_action_ref")
            or refs.get("teacher_action_path")
            or refs.get("teacher_action_envelope_ref")
            or refs.get("teacher_action_envelope_path")
        ),
        "scene_tracks_non_stub": bool(metadata.get("scene_tracks_non_stub", False)),
        "semantic_memory_grounded": bool(
            grounded_track_count > 0 or metadata.get("semantic_memory_grounded", False)
        ),
        "budget_settlement_live": bool(metadata.get("budget_settlement_live", False)),
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
    for key, value in explicit.items():
        derived[str(key)] = bool(value)
    return dict(sorted(derived.items()))


def _future_training_artifact_refs(
    episode: ReplayEpisodeRecord,
    refs: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
    explicit = metadata.get("future_training_artifacts", {})
    if not isinstance(explicit, Mapping):
        explicit = {}
    artifacts: Dict[str, Any] = {
        key: value
        for key, value in dict(explicit).items()
        if value not in (None, "", [], {})
    }
    for candidate in ("training_runtime_manifest", "promotion_ledger_ref"):
        value = refs.get(candidate) or metadata.get(candidate)
        if value not in (None, "", [], {}):
            artifacts[candidate] = value
    return dict(sorted(artifacts.items()))


def build_replay_execution_preconditions(
    episode: ReplayEpisodeRecord,
    *,
    steps: Optional[Iterable[ReplayStepRecord]] = None,
    windows: Optional[Iterable[ReplayWindowRecord]] = None,
) -> ExecutionPreconditionsReport:
    """Assess whether a replay episode is trace-complete enough for training."""

    step_rows = list(steps or [])
    window_rows = list(windows or [])
    refs = collect_replay_artifact_refs(episode, steps=step_rows, windows=window_rows)
    future_signals = _future_training_signals(episode, refs)
    future_artifacts = _future_training_artifact_refs(episode, refs)
    refs = {
        **dict(refs),
        **dict(future_artifacts),
    }
    signal_values = {
        "event_ref_count": len(list(episode.metadata.get("event_refs", []) or [])),
        "decision_ref_count": len(
            list(episode.metadata.get("decision_refs", []) or [])
        ),
        "window_count": len(window_rows),
        "step_count": len(step_rows) or int(episode.total_steps),
        **future_signals,
    }
    required_refs = list(REPLAY_REQUIRED_ARTIFACT_REFS)
    if refs.get("counterfactual_eval_ref") or refs.get("counterfactual_eval_path"):
        required_refs.append(
            "counterfactual_eval_path"
            if refs.get("counterfactual_eval_path")
            else "counterfactual_eval_ref"
        )
    if refs.get("value_target_pack_ref") or refs.get("value_target_pack_path"):
        required_refs.append(
            "value_target_pack_path"
            if refs.get("value_target_pack_path")
            else "value_target_pack_ref"
        )
    return build_execution_preconditions(
        subject_id=episode.episode_id,
        subject_kind="replay_episode",
        artifact_refs=refs,
        required_artifact_refs=required_refs,
        signal_values=signal_values,
        min_signal_thresholds={
            "event_ref_count": 1.0,
            "decision_ref_count": 1.0,
            "step_count": 1.0,
        },
        soft_required_artifact_refs=list(future_artifacts.keys()),
        soft_boolean_signals={key: True for key in future_signals},
        metadata={
            "run_id": episode.run_id,
            "source_domain": episode.source_domain,
            "future_training_signals": future_signals,
            "future_training_artifacts": future_artifacts,
        },
    )


def summarize_replay_execution_preconditions(
    reports: Iterable[ExecutionPreconditionsReport | Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = summarize_execution_preconditions(list(reports))
    summary["required_artifact_refs"] = list(REPLAY_REQUIRED_ARTIFACT_REFS)
    return summary


__all__ = [
    "REPLAY_REQUIRED_ARTIFACT_REFS",
    "build_replay_execution_preconditions",
    "collect_replay_artifact_refs",
    "summarize_replay_execution_preconditions",
]
