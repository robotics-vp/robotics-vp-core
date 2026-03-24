"""Replay-specific trace completeness and self-improvement readiness helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

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
            if key.endswith(("_ref", "_refs", "_id", "_ids")) and value not in (None, "", [], {}):
                refs[str(key)] = value
    return refs


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
    signal_values = {
        "event_ref_count": len(list(episode.metadata.get("event_refs", []) or [])),
        "decision_ref_count": len(list(episode.metadata.get("decision_refs", []) or [])),
        "window_count": len(window_rows),
        "step_count": len(step_rows) or int(episode.total_steps),
    }
    required_refs = list(REPLAY_REQUIRED_ARTIFACT_REFS)
    if refs.get("counterfactual_eval_ref") or refs.get("counterfactual_eval_path"):
        required_refs.append("counterfactual_eval_path" if refs.get("counterfactual_eval_path") else "counterfactual_eval_ref")
    if refs.get("value_target_pack_ref") or refs.get("value_target_pack_path"):
        required_refs.append("value_target_pack_path" if refs.get("value_target_pack_path") else "value_target_pack_ref")
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
        metadata={
            "run_id": episode.run_id,
            "source_domain": episode.source_domain,
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
