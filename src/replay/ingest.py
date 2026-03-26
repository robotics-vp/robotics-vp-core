"""Replay ingestion adapters for shadow artifacts and workcell episode logs."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.constraints.constraint_set import ConstraintSet
from src.economics.functor import ObjectiveEconFunctor
from src.economics.pricing_sentinel import PricingSentinel, PricingTickInput
from src.economics.value_ledger import summarize_econ_tensor
from src.evidence.scene_tracks_truth import normalize_scene_tracks_truth
from src.envs.workcell_env.base import EpisodeLog
from src.motor_backend.rollout_capture import finalize_rollout_bundle
from src.objectives.runtime_builder import (
    ObjectiveRuntimeBuilder,
    ObjectiveRuntimeRecord,
    ObjectiveRuntimeWindow,
    SourceDomain,
    summarize_objective_tensor,
)
from src.objectives.tensor import ObjectiveTensor
from src.observation.condition_vector import ConditionVector
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.replay.schema import (
    ReplayEpisodeRecord,
    ReplayStepRecord,
    ReplayWindowRecord,
)
from src.runtime.event_spine import DecisionLedgerEntry, RuntimeEvent
from src.runtime.packets import RuntimePacket
from src.utils.config_digest import sha256_json


REPLAY_SCHEMA_VERSION = "shadow_replay_dataset_v1"
_TARGET_RE = re.compile(r"slot_(\d+)")


class ReplayIngestionError(RuntimeError):
    """Raised when a replay source is incomplete or malformed."""


def ingest_shadow_run(
    run_dir: str | Path,
    *,
    pricing_policy_path: str | Path = "config/pricing/default.yaml",
) -> Tuple[List[ReplayEpisodeRecord], List[ReplayStepRecord], List[ReplayWindowRecord], Dict[str, Any]]:
    """Ingest a shadow control-plane run into canonical replay records."""

    root = Path(run_dir)
    traces_payload = _load_json(root / "shadow_episode_traces.json")
    if not isinstance(traces_payload, Mapping):
        raise ReplayIngestionError(f"Missing shadow episode trace sidecar under {root}")

    objective_payload = _load_json(root / "objective_tensor.json")
    econ_payload = _load_json(root / "econ_tensor.json")
    flags_payload = _load_json(root / "constraint_flags.json")
    datapack_payload = _load_json(root / "datapack_credit_update.json")
    regal_payload = _load_json(root / "regal_decisions.json")
    pricing_rows = _load_jsonl(root / "pricing_ticks.jsonl")
    ledger_rows = _load_jsonl(root / "value_ledger.jsonl")
    runtime_packet_path = root / "runtime_packets.json"
    event_spine_path = root / "event_spine.json"
    decision_ledger_path = root / "decision_ledger.json"
    runtime_packet_payload = _load_json(runtime_packet_path) if runtime_packet_path.exists() else {}
    event_spine_payload = _load_json(event_spine_path) if event_spine_path.exists() else {}
    decision_ledger_payload = _load_json(decision_ledger_path) if decision_ledger_path.exists() else {}

    objective_by_episode = {
        str(row.get("episode_id")): dict(row.get("objective_tensor", {}))
        for row in list(objective_payload.get("episodes", []) or [])
    }
    objective_windows_by_episode = {
        str(row.get("episode_id")): list(row.get("windows", []) or [])
        for row in list(objective_payload.get("windows", []) or [])
    }
    econ_by_episode = {
        str(row.get("episode_id")): dict(row.get("econ_tensor", {}))
        for row in list(econ_payload.get("episodes", []) or [])
    }
    econ_windows_by_episode = defaultdict(dict)
    for row in list(econ_payload.get("windows", []) or []):
        episode_id = str(row.get("episode_id"))
        window = dict(row.get("window", {}) or {})
        window_id = str(window.get("window_id", ""))
        econ_windows_by_episode[episode_id][window_id] = dict(row.get("econ_tensor", {}))
    flags_by_episode = {
        str(row.get("episode_id")): [dict(flag) for flag in list(row.get("flags", []) or [])]
        for row in list(flags_payload.get("episodes", []) or [])
    }
    datapack_by_episode = {
        str(row.get("episode_id")): dict(row)
        for row in list(datapack_payload.get("updates", []) or [])
    }
    regal_by_episode = {
        str(row.get("episode_id")): dict(row.get("regal_decision", {}))
        for row in list(regal_payload.get("episodes", []) or [])
    }
    pricing_by_episode = defaultdict(list)
    for row in pricing_rows:
        pricing_by_episode[str(row.get("episode_id"))].append(dict(row))
    ledger_by_episode = defaultdict(list)
    for row in ledger_rows:
        ledger_by_episode[str(row.get("episode_id"))].append(dict(row))
    runtime_packets_by_episode: Dict[str, RuntimePacket] = {}
    if isinstance(runtime_packet_payload, Mapping):
        for row in list(runtime_packet_payload.get("episodes", []) or []):
            packet_payload = row.get("runtime_packet", row)
            if not isinstance(packet_payload, Mapping):
                continue
            packet = RuntimePacket.from_dict(packet_payload)
            if packet.episode_id:
                runtime_packets_by_episode[packet.episode_id] = packet
    runtime_packet_ref = runtime_packet_path.name if runtime_packets_by_episode else None
    events_by_episode: Dict[str, List[RuntimeEvent]] = defaultdict(list)
    if isinstance(event_spine_payload, Mapping):
        for row in list(event_spine_payload.get("events", []) or []):
            if isinstance(row, Mapping):
                event = RuntimeEvent.from_dict(row)
                if event.episode_id:
                    events_by_episode[event.episode_id].append(event)
    decisions_by_episode: Dict[str, List[DecisionLedgerEntry]] = defaultdict(list)
    if isinstance(decision_ledger_payload, Mapping):
        for row in list(decision_ledger_payload.get("decisions", []) or []):
            if isinstance(row, Mapping):
                decision = DecisionLedgerEntry.from_dict(row)
                if decision.episode_id:
                    decisions_by_episode[decision.episode_id].append(decision)
    event_spine_ref = event_spine_path.name if events_by_episode else None
    decision_ledger_ref = decision_ledger_path.name if decisions_by_episode else None

    condition_builder = ConditionVectorBuilder()
    episodes: List[ReplayEpisodeRecord] = []
    steps: List[ReplayStepRecord] = []
    windows: List[ReplayWindowRecord] = []

    for trace_payload in list(traces_payload.get("episodes", []) or []):
        episode_id = str(trace_payload.get("episode_id", ""))
        runtime_record = dict(trace_payload.get("runtime_record", {}) or {})
        objective_tensor = ObjectiveTensor.from_dict(objective_by_episode[episode_id])
        objective_summary = summarize_objective_tensor(objective_tensor)
        econ_summary = summarize_econ_tensor(econ_by_episode[episode_id])
        constraint_flags = flags_by_episode.get(episode_id, [])
        pricing_ticks = sorted(
            pricing_by_episode.get(episode_id, []),
            key=lambda row: (str(row.get("mode", "")), str(row.get("tick_id", ""))),
        )
        episode_pricing = next((row for row in pricing_ticks if row.get("mode") == "episode"), {})
        datapack_summary = datapack_by_episode.get(episode_id, {})
        regal_summary = regal_by_episode.get(episode_id, {})
        step_traces = {
            int(row.get("step", index)): dict(row)
            for index, row in enumerate(list(trace_payload.get("step_traces", []) or []))
        }
        episode_log = dict(trace_payload.get("episode_log", {}) or {})
        trajectory = list(episode_log.get("trajectory", []) or [])
        runtime_packet = runtime_packets_by_episode.get(episode_id)
        episode_events = sorted(events_by_episode.get(episode_id, []), key=lambda row: (row.sequence_idx, row.event_id))
        episode_decisions = sorted(
            decisions_by_episode.get(episode_id, []),
            key=lambda row: (row.sequence_idx, row.decision_id),
        )
        seed = int(runtime_record.get("seed", 0))
        task_id = str(trace_payload.get("task_id", runtime_record.get("task_id", "")))
        env_id = str(trace_payload.get("env_id", runtime_record.get("env_id", "")))
        source_domain = str(trace_payload.get("source_domain", runtime_record.get("source_domain", SourceDomain.REPLAY.value)))
        started_at = str(trace_payload.get("started_at", runtime_record.get("timestamp", "")))
        ended_at = str(trace_payload.get("ended_at", runtime_record.get("timestamp", "")))

        episode_condition = build_shadow_condition_vector(
            condition_builder=condition_builder,
            task_id=task_id,
            env_id=env_id,
            source_domain=source_domain,
            objective_summary=objective_summary,
            econ_summary=econ_summary,
            constraint_flags=constraint_flags,
            semantic_tags=_extract_semantic_tags(runtime_record),
            objective_profile_id=str(episode_pricing.get("objective_profile_id", "")),
            uncertainty=float(episode_pricing.get("metadata", {}).get("uncertainty", runtime_record.get("telemetry", {}).get("uncertainty", 0.0))),
            trust_score=float(episode_pricing.get("confidence", runtime_record.get("telemetry", {}).get("trust_score", 1.0))),
            episode_step=0,
        )
        ledger_event_ids = [str(row.get("ledger_event_id")) for row in ledger_by_episode.get(episode_id, [])]

        episodes.append(
            ReplayEpisodeRecord(
                run_id=str(trace_payload.get("run_id", "")),
                episode_id=episode_id,
                task_id=task_id,
                env_id=env_id,
                source_domain=source_domain,
                seed=seed,
                status=str(trace_payload.get("status", "unknown")),
                started_at=started_at,
                ended_at=ended_at,
                total_steps=len(trajectory),
                total_reward=float(runtime_record.get("episode_metrics", {}).get("reward_total", 0.0)),
                skill_mode=episode_condition.skill_mode,
                condition_vector=episode_condition.to_dict(),
                condition_vector_values=[float(value) for value in episode_condition.to_vector().tolist()],
                objective_tensor_summary=objective_summary,
                objective_tensor_ref="objective_tensor.json",
                econ_tensor_summary=econ_summary,
                econ_tensor_ref="econ_tensor.json",
                pricing_summary=dict(episode_pricing),
                pricing_tick_refs=[str(row.get("tick_id")) for row in pricing_ticks],
                constraint_flags=[dict(flag) for flag in constraint_flags],
                regal_summary=dict(regal_summary),
                datapack_summary=dict(datapack_summary),
                ledger_event_ids=ledger_event_ids,
                metadata={
                    "objective_profile_id": episode_pricing.get("objective_profile_id"),
                    "runtime_record_hash": sha256_json(runtime_record),
                    "trace_hash": sha256_json(trace_payload),
                    "runtime_packet_id": runtime_packet.packet_id if runtime_packet else None,
                    "contract_id": runtime_packet.contract.contract_id if runtime_packet else None,
                    "event_refs": [event.event_id for event in episode_events],
                    "decision_refs": [decision.decision_id for decision in episode_decisions],
                    "event_kinds": [event.event_kind for event in episode_events],
                    "decision_kinds": [decision.decision_kind for decision in episode_decisions],
                },
                provenance={
                    "source_adapter": "shadow_control_plane_artifacts_v1",
                    "source_root": str(root),
                    "trace_sidecar": "shadow_episode_traces.json",
                    "objective_tensor_ref": "objective_tensor.json",
                    "econ_tensor_ref": "econ_tensor.json",
                    "runtime_packet_ref": runtime_packet_ref,
                    "runtime_packet_hash": sha256_json(runtime_packet.to_dict()) if runtime_packet else None,
                    "event_spine_ref": event_spine_ref,
                    "decision_ledger_ref": decision_ledger_ref,
                },
            )
        )

        for step in trajectory:
            step_idx = int(step.get("step", 0))
            step_trace = step_traces.get(step_idx, {})
            obs = dict(step.get("obs", {}) or {})
            action = dict(step.get("action", {}) or {})
            step_condition = build_shadow_condition_vector(
                condition_builder=condition_builder,
                task_id=task_id,
                env_id=env_id,
                source_domain=source_domain,
                objective_summary=objective_summary,
                econ_summary=econ_summary,
                constraint_flags=constraint_flags,
                semantic_tags=_extract_semantic_tags(runtime_record),
                objective_profile_id=str(episode_pricing.get("objective_profile_id", "")),
                uncertainty=float(step_trace.get("task_state", {}).get("constraint_error", episode_pricing.get("metadata", {}).get("uncertainty", 0.0))),
                trust_score=float(runtime_record.get("telemetry", {}).get("trust_score", 1.0)),
                episode_step=step_idx,
            )
            step_time = _timestamp_for_step(
                started_at,
                step_idx=step_idx,
                time_step_s=float(runtime_record.get("episode_metrics", {}).get("time_step_s", 1.0)),
            )
            price_ref = _pick_window_tick_id(pricing_ticks, step_idx=step_idx) or episode_pricing.get("tick_id")
            step_events = _events_for_step(episode_events, step_idx=step_idx)
            step_decisions = _decisions_for_step(episode_decisions, step_idx=step_idx)
            steps.append(
                ReplayStepRecord(
                    run_id=str(trace_payload.get("run_id", "")),
                    episode_id=episode_id,
                    step_idx=step_idx,
                    obs=obs,
                    obs_vector=_extract_obs_vector(obs),
                    action=action,
                    action_vector=_extract_action_vector(action),
                    reward=float(step_trace.get("reward", step.get("info", {}).get("reward", 0.0))),
                    reward_decomposition=dict(step_trace.get("reward_breakdown", {}) or {}),
                    done=bool(step.get("done", False)),
                    task_id=task_id,
                    env_id=env_id,
                    condition_vector=step_condition.to_dict(),
                    condition_vector_values=[float(value) for value in step_condition.to_vector().tolist()],
                    skill_mode=step_condition.skill_mode,
                    objective_tensor_summary=objective_summary,
                    objective_tensor_ref="objective_tensor.json",
                    econ_tensor_summary=econ_summary,
                    econ_tensor_ref="econ_tensor.json",
                    constraint_flags=[dict(flag) for flag in constraint_flags],
                    pricing_tick_ref=str(price_ref) if price_ref else None,
                    ledger_event_ref=ledger_event_ids[0] if ledger_event_ids else None,
                    source_domain=source_domain,
                    seed=seed,
                    timestamp=step_time,
                    metadata={
                        "success": bool(step.get("info", {}).get("success", False)),
                        "task_info": dict(step_trace.get("task_info", {}) or {}),
                        "runtime_packet_id": runtime_packet.packet_id if runtime_packet else None,
                        "event_refs": [event.event_id for event in step_events],
                        "decision_refs": [decision.decision_id for decision in step_decisions],
                    },
                    provenance={
                        "source_adapter": "shadow_control_plane_artifacts_v1",
                        "source_root": str(root),
                        "step_trace_hash": sha256_json(step_trace),
                        "runtime_packet_ref": runtime_packet_ref,
                        "contract_id": runtime_packet.contract.contract_id if runtime_packet else None,
                        "event_spine_ref": event_spine_ref,
                        "decision_ledger_ref": decision_ledger_ref,
                    },
                )
            )

        objective_windows = {
            str(row.get("window", {}).get("window_id", "")): dict(row)
            for row in objective_windows_by_episode.get(episode_id, [])
        }
        for window_payload in list(runtime_record.get("windows", []) or []):
            window = dict(window_payload or {})
            window_id = str(window.get("window_id", ""))
            start_step = int(window.get("start_step", 0))
            end_step = int(window.get("end_step", start_step))
            window_objective_payload = objective_windows.get(window_id, {}).get("objective_tensor")
            if window_objective_payload:
                window_objective = summarize_objective_tensor(ObjectiveTensor.from_dict(window_objective_payload))
            else:
                window_objective = objective_summary
            window_econ = summarize_econ_tensor(econ_windows_by_episode.get(episode_id, {}).get(window_id, {}))
            window_tick = next(
                (
                    row
                    for row in pricing_ticks
                    if row.get("mode") == "step_window" and str(row.get("metadata", {}).get("window_id", "")) == window_id
                ),
                {},
            )
            window_condition = build_shadow_condition_vector(
                condition_builder=condition_builder,
                task_id=task_id,
                env_id=env_id,
                source_domain=source_domain,
                objective_summary=window_objective,
                econ_summary=window_econ,
                constraint_flags=constraint_flags,
                semantic_tags=_extract_semantic_tags(runtime_record),
                objective_profile_id=str(window_tick.get("objective_profile_id", episode_pricing.get("objective_profile_id", ""))),
                uncertainty=float(window_tick.get("metadata", {}).get("uncertainty", window.get("telemetry", {}).get("uncertainty", 0.0))),
                trust_score=float(window.get("telemetry", {}).get("trust_score", episode_pricing.get("confidence", 1.0))),
                episode_step=start_step,
            )
            window_steps = [row for row in steps if row.episode_id == episode_id and start_step <= row.step_idx <= end_step]
            window_events = _events_for_window(
                episode_events,
                start_step=start_step,
                end_step=end_step,
            )
            window_decisions = _decisions_for_window(
                episode_decisions,
                start_step=start_step,
                end_step=end_step,
            )
            windows.append(
                ReplayWindowRecord(
                    run_id=str(trace_payload.get("run_id", "")),
                    episode_id=episode_id,
                    window_id=window_id,
                    start_step=start_step,
                    end_step=end_step,
                    task_id=task_id,
                    env_id=env_id,
                    source_domain=source_domain,
                    seed=seed,
                    timestamp=_timestamp_for_step(started_at, step_idx=start_step, time_step_s=float(runtime_record.get("episode_metrics", {}).get("time_step_s", 1.0))),
                    reward_sum=sum(row.reward for row in window_steps),
                    obs_vector_mean=_mean_vectors([row.obs_vector for row in window_steps]),
                    action_vector_mean=_mean_vectors([row.action_vector for row in window_steps]),
                    condition_vector=window_condition.to_dict(),
                    condition_vector_values=[float(value) for value in window_condition.to_vector().tolist()],
                    skill_mode=window_condition.skill_mode,
                    objective_tensor_summary=window_objective,
                    econ_tensor_summary=window_econ,
                    pricing_summary=dict(window_tick),
                    constraint_flags=[dict(flag) for flag in constraint_flags],
                    metadata={
                        "window_hash": sha256_json(window),
                        "event_refs": [event.event_id for event in window_events],
                        "decision_refs": [decision.decision_id for decision in window_decisions],
                    },
                    provenance={
                        "source_adapter": "shadow_control_plane_artifacts_v1",
                        "source_root": str(root),
                        "window_id": window_id,
                        "runtime_packet_ref": runtime_packet_ref,
                        "runtime_packet_id": runtime_packet.packet_id if runtime_packet else None,
                        "event_spine_ref": event_spine_ref,
                        "decision_ledger_ref": decision_ledger_ref,
                    },
                )
            )

    metadata = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_adapter": "shadow_control_plane_artifacts_v1",
        "source_root": str(root),
        "pricing_policy_path": str(pricing_policy_path),
        "runtime_packet_ref": runtime_packet_ref,
        "runtime_packet_count": len(runtime_packets_by_episode),
        "event_spine_ref": event_spine_ref,
        "event_count": sum(len(rows) for rows in events_by_episode.values()),
        "decision_ledger_ref": decision_ledger_ref,
        "decision_count": sum(len(rows) for rows in decisions_by_episode.values()),
        "provenance_digest": sha256_json(
            {
                "root": str(root),
                "schema_version": REPLAY_SCHEMA_VERSION,
                "runtime_packet_count": len(runtime_packets_by_episode),
                "event_count": sum(len(rows) for rows in events_by_episode.values()),
                "decision_count": sum(len(rows) for rows in decisions_by_episode.values()),
            }
        ),
    }
    return _sort_episode_records(episodes), _sort_step_records(steps), _sort_window_records(windows), metadata


def ingest_workcell_episode_log(
    episode_log_path: str | Path,
    *,
    run_id: Optional[str] = None,
    source_domain: str = SourceDomain.SYNTHETIC.value,
    objective_profile_id: str = "balanced_contract",
    pricing_policy_path: str | Path = "config/pricing/default.yaml",
) -> Tuple[List[ReplayEpisodeRecord], List[ReplayStepRecord], List[ReplayWindowRecord], Dict[str, Any]]:
    """Ingest an existing WorkcellEnv episode log into canonical replay records."""

    payload = _load_json(Path(episode_log_path))
    if isinstance(payload, Mapping) and "metadata" not in payload and "episode_log" in payload:
        payload = payload.get("episode_log")
    if not isinstance(payload, Mapping):
        raise ReplayIngestionError(f"Invalid workcell episode log at {episode_log_path}")
    episode_log = EpisodeLog.from_dict(dict(payload))
    metadata = episode_log.metadata
    metrics = dict(episode_log.metrics)
    if not metrics:
        metrics = {
            "reward_total": float(sum(float(step.get("info", {}).get("reward", 0.0)) for step in episode_log.trajectory)),
            "steps": len(episode_log.trajectory),
            "time_step_s": 1.0,
        }
    runtime_record = ObjectiveRuntimeRecord(
        task_id=metadata.task_id,
        episode_id=metadata.episode_id,
        env_id=str((metadata.env_params or {}).get("config", {}).get("topology_type", "workcell_env")),
        world_id="workcell_episode_log",
        robot_id=str(metadata.robot_family or "workcell_robot"),
        source_domain=source_domain,
        seed=int(metadata.seed or 0),
        run_id=run_id or f"workcell_replay_{sha256_json(payload)[:10]}",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        episode_metrics=metrics,
        reward_components={"scalar_reward": float(metrics.get("reward_total", 0.0))},
        telemetry={},
        windows=_build_episode_log_windows(episode_log.trajectory, time_step_s=float(metrics.get("time_step_s", 1.0))),
        context={"episode_log_source": str(episode_log_path)},
    )
    builder = ObjectiveRuntimeBuilder()
    objective_tensor = builder.build(runtime_record)
    objective_summary = summarize_objective_tensor(objective_tensor)
    constraint_set = ConstraintSet.from_runtime(
        hard_constraints={"throughput": {"min": 0.0}},
        soft_constraints={"energy": {"max": float(metrics.get("energy_wh_per_unit", 8.0) or 8.0)}},
        geometry_hints={"source": "workcell_episode_log"},
        trust_metadata={"trust_score": 0.75},
    )
    constraint_flags = constraint_set.flag_observations(metrics)
    functor = ObjectiveEconFunctor(base_price_per_unit=3.0)
    econ_tensor = functor.map(
        objective_tensor,
        constraint_flags=constraint_flags,
        uncertainty=0.15,
        context={"run_id": runtime_record.run_id, "episode_id": runtime_record.episode_id, "source_domain": source_domain},
    )
    econ_summary = summarize_econ_tensor(econ_tensor)
    pricing = PricingSentinel.from_path(pricing_policy_path).emit_tick(
        PricingTickInput(
            run_id=runtime_record.run_id,
            episode_id=runtime_record.episode_id,
            objective_profile_id=objective_profile_id,
            source_domain=source_domain,
            timestamp=runtime_record.timestamp,
            mode="episode",
            econ_tensor=econ_tensor,
            uncertainty=0.15,
            constraint_flags=constraint_flags,
            trust_score=0.75,
        )
    )
    condition_builder = ConditionVectorBuilder()
    episode_condition = build_shadow_condition_vector(
        condition_builder=condition_builder,
        task_id=runtime_record.task_id,
        env_id=runtime_record.env_id,
        source_domain=source_domain,
        objective_summary=objective_summary,
        econ_summary=econ_summary,
        constraint_flags=constraint_flags,
        semantic_tags=[],
        objective_profile_id=objective_profile_id,
        uncertainty=0.15,
        trust_score=0.75,
        episode_step=0,
    )
    episode_records = [
        ReplayEpisodeRecord(
            run_id=runtime_record.run_id,
            episode_id=runtime_record.episode_id,
            task_id=runtime_record.task_id,
            env_id=runtime_record.env_id,
            source_domain=source_domain,
            seed=runtime_record.seed,
            status="completed",
            started_at=runtime_record.timestamp,
            ended_at=runtime_record.timestamp,
            total_steps=len(episode_log.trajectory),
            total_reward=float(metrics.get("reward_total", 0.0)),
            skill_mode=episode_condition.skill_mode,
            condition_vector=episode_condition.to_dict(),
            condition_vector_values=[float(value) for value in episode_condition.to_vector().tolist()],
            objective_tensor_summary=objective_summary,
            objective_tensor_ref=str(episode_log_path),
            econ_tensor_summary=econ_summary,
            econ_tensor_ref=str(episode_log_path),
            pricing_summary=pricing.to_dict(),
            pricing_tick_refs=[pricing.tick_id],
            constraint_flags=constraint_flags,
            regal_summary={},
            datapack_summary={},
            ledger_event_ids=[],
            metadata={"episode_log_source": str(episode_log_path)},
            provenance={
                "source_adapter": "workcell_episode_log_v1",
                "source_path": str(episode_log_path),
            },
        )
    ]
    step_records: List[ReplayStepRecord] = []
    for step in list(episode_log.trajectory or []):
        step_idx = int(step.get("step", 0))
        obs = dict(step.get("obs", {}) or {})
        action = dict(step.get("action", {}) or {})
        condition = build_shadow_condition_vector(
            condition_builder=condition_builder,
            task_id=runtime_record.task_id,
            env_id=runtime_record.env_id,
            source_domain=source_domain,
            objective_summary=objective_summary,
            econ_summary=econ_summary,
            constraint_flags=constraint_flags,
            semantic_tags=[],
            objective_profile_id=objective_profile_id,
            uncertainty=0.15,
            trust_score=0.75,
            episode_step=step_idx,
        )
        step_records.append(
            ReplayStepRecord(
                run_id=runtime_record.run_id,
                episode_id=runtime_record.episode_id,
                step_idx=step_idx,
                obs=obs,
                obs_vector=_extract_obs_vector(obs),
                action=action,
                action_vector=_extract_action_vector(action),
                reward=float(step.get("info", {}).get("reward", 0.0)),
                reward_decomposition={},
                done=bool(step.get("done", False)),
                task_id=runtime_record.task_id,
                env_id=runtime_record.env_id,
                condition_vector=condition.to_dict(),
                condition_vector_values=[float(value) for value in condition.to_vector().tolist()],
                skill_mode=condition.skill_mode,
                objective_tensor_summary=objective_summary,
                objective_tensor_ref=str(episode_log_path),
                econ_tensor_summary=econ_summary,
                econ_tensor_ref=str(episode_log_path),
                constraint_flags=constraint_flags,
                pricing_tick_ref=pricing.tick_id,
                ledger_event_ref=None,
                source_domain=source_domain,
                seed=runtime_record.seed,
                timestamp=_timestamp_for_step(runtime_record.timestamp, step_idx=step_idx, time_step_s=float(metrics.get("time_step_s", 1.0))),
                metadata={},
                provenance={
                    "source_adapter": "workcell_episode_log_v1",
                    "source_path": str(episode_log_path),
                },
            )
        )
    window_records = _episode_log_window_records(
        runtime_record=runtime_record,
        step_records=step_records,
        objective_profile_id=objective_profile_id,
        pricing_policy_path=pricing_policy_path,
        condition_builder=condition_builder,
    )
    metadata_payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_adapter": "workcell_episode_log_v1",
        "source_path": str(episode_log_path),
        "provenance_digest": sha256_json({"source_path": str(episode_log_path), "schema_version": REPLAY_SCHEMA_VERSION}),
    }
    return episode_records, _sort_step_records(step_records), _sort_window_records(window_records), metadata_payload


def _resolve_rollout_artifact_path(episode_dir: Path, value: Any) -> Optional[str]:
    if value in (None, "", [], {}):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path.resolve())
    if path.exists():
        return str(path.resolve())
    for anchor in (episode_dir, *episode_dir.parents):
        candidate = anchor / path
        if candidate.exists():
            return str(candidate.resolve())
    return str((episode_dir / path).resolve())


def _register_rollout_artifact_ref(refs: Dict[str, Any], key: str, value: Optional[str]) -> None:
    if value in (None, "", [], {}):
        return
    normalized_key = str(key)
    refs[normalized_key] = value
    if normalized_key.endswith("_path"):
        refs[f"{normalized_key[:-5]}_ref"] = value
    elif normalized_key.endswith("_paths"):
        refs[f"{normalized_key[:-6]}_refs"] = value


def _load_rollout_metadata_payload(episode_dir: Path) -> Dict[str, Any]:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return _load_json(metadata_path)


def _discover_rollout_artifact_refs(episode_dir: Path, episode_id: str) -> Dict[str, Any]:
    metadata_payload = _load_rollout_metadata_payload(episode_dir)
    refs: Dict[str, Any] = {}

    explicit_path_keys = (
        "trajectory_path",
        "rgb_video_path",
        "depth_video_path",
        "scene_tracks_path",
        "semantic_world_model_path",
        "semantic_snapshot_path",
        "orchestrator_advisory_path",
        "teacher_trace_path",
        "vla_semantic_evidence_path",
        "semantic_fusion_path",
        "belief_state_path",
        "evidence_bus_path",
    )
    for key in explicit_path_keys:
        resolved = _resolve_rollout_artifact_path(episode_dir, metadata_payload.get(key))
        _register_rollout_artifact_ref(refs, key, resolved)

    if isinstance(metadata_payload.get("sensor_bundle"), Mapping):
        _register_rollout_artifact_ref(refs, "sensor_bundle_metadata_path", str((episode_dir / "metadata.json").resolve()))

    sidecar_patterns = {
        "scene_tracks_path": [
            f"{episode_id}_*_scene_tracks_v1.npz",
            "*_scene_tracks_v1.npz",
        ],
        "semantic_world_model_path": [
            f"{episode_id}_semantic_world_model_v1.json",
            "*_semantic_world_model_v1.json",
        ],
        "semantic_snapshot_path": [
            f"{episode_id}_semantic_snapshot_v1.json",
            "*_semantic_snapshot_v1.json",
        ],
        "orchestrator_advisory_path": [
            f"{episode_id}_orchestrator_advisory_v1.json",
            "*_orchestrator_advisory_v1.json",
        ],
        "teacher_trace_path": [
            f"{episode_id}_teacher_trace_v1.json",
            "*_teacher_trace_v1.json",
        ],
        "vla_semantic_evidence_path": [
            f"{episode_id}_vla_semantic_evidence_v1.npz",
            "*_vla_semantic_evidence_v1.npz",
        ],
        "semantic_fusion_path": [
            f"{episode_id}_semantic_fusion_v1.npz",
            "*_semantic_fusion_v1.npz",
        ],
        "belief_state_path": [
            f"{episode_id}_belief_state_v1.json",
            "*_belief_state_v1.json",
        ],
        "evidence_bus_path": [
            f"{episode_id}_evidence_bus_v1.json",
            "*_evidence_bus_v1.json",
        ],
    }
    for key, patterns in sidecar_patterns.items():
        if key in refs:
            continue
        for pattern in patterns:
            candidates = sorted(episode_dir.glob(pattern))
            if candidates:
                _register_rollout_artifact_ref(refs, key, str(candidates[0].resolve()))
                break
    return refs


def _scene_tracks_rollout_metadata(
    scene_tracks_path: Optional[str],
    *,
    metadata_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(metadata_payload or {})
    semantic_summary = payload.get("scene_tracks_semantic_summary", {})
    if not isinstance(semantic_summary, Mapping):
        semantic_summary = {}
    backend = ""
    training_eligible = bool(payload.get("scene_tracks_training_eligible", False))
    if scene_tracks_path:
        try:
            scene_tracks = dict(np.load(scene_tracks_path, allow_pickle=False))
            summary_json_raw = scene_tracks.get("scene_tracks_v1/summary_json")
            if isinstance(summary_json_raw, np.ndarray) and summary_json_raw.size > 0:
                summary_payload = json.loads(str(summary_json_raw.flat[0]))
                if isinstance(summary_payload, Mapping):
                    backend = str(
                        summary_payload.get("backend_selected")
                        or dict(summary_payload.get("adapter_status", {}) or {}).get("overall_mode", "")
                    )
                    training_eligible = bool(summary_payload.get("training_eligible", False))
            semantic_summary_raw = scene_tracks.get("scene_tracks_v1/semantic_summary_json")
            if isinstance(semantic_summary_raw, np.ndarray) and semantic_summary_raw.size > 0 and not semantic_summary:
                decoded_summary = json.loads(str(semantic_summary_raw.flat[0]))
                if isinstance(decoded_summary, Mapping):
                    semantic_summary = decoded_summary
        except Exception:
            backend = ""
    if not backend:
        backend = str(payload.get("scene_tracks_backend", ""))
    truth = normalize_scene_tracks_truth(
        backend=backend,
        explicit_non_stub=bool(payload.get("scene_tracks_non_stub", False)),
        semantic_grounding_ready=bool(semantic_summary.get("grounding_ready", False)),
        training_eligible=bool(training_eligible),
        explicit_non_heuristic=bool(payload.get("semantic_grounding_non_heuristic", False)),
    )
    return {
        "scene_tracks_backend": str(truth.get("scene_tracks_backend", "")),
        "scene_tracks_non_stub": bool(truth.get("scene_tracks_non_stub", False)),
        "scene_tracks_training_eligible": bool(truth.get("scene_tracks_training_eligible", False)),
        "semantic_grounding_ready": bool(truth.get("semantic_grounding_ready", False)),
        "semantic_grounding_non_heuristic": bool(
            truth.get("semantic_grounding_non_heuristic", False)
        ),
        "semantic_density_score": float(semantic_summary.get("semantic_density_score", 0.0) or 0.0),
    }


def _semantic_world_model_rollout_summary(semantic_world_model_path: Optional[str]) -> Dict[str, Any]:
    if not semantic_world_model_path:
        return {}
    try:
        payload = _load_json(Path(semantic_world_model_path))
    except Exception:
        return {}
    topology = dict(payload.get("topology", {}) or {}) if isinstance(payload, Mapping) else {}
    capability_scores = dict(payload.get("capability_scores", {}) or {}) if isinstance(payload, Mapping) else {}
    active_capabilities: List[str] = []
    for key, value in capability_scores.items():
        try:
            score = float(value)
        except Exception:
            continue
        if score >= 0.5:
            active_capabilities.append(str(key))
    active_capabilities.sort()
    return {
        "present": True,
        "world_model_id": str(payload.get("world_model_id", "")),
        "topology": topology,
        "capability_scores": capability_scores,
        "grounded_track_object_count": int(topology.get("grounded_track_object_count", 0) or 0),
        "active_capabilities": active_capabilities,
    }


def ingest_rollout_bundle(
    rollout_root: str | Path,
    *,
    scenario_id: Optional[str] = None,
    run_id: Optional[str] = None,
    source_domain: str = SourceDomain.SYNTHETIC.value,
    objective_profile_id: str = "balanced_contract",
    pricing_policy_path: str | Path = "config/pricing/default.yaml",
) -> Tuple[List[ReplayEpisodeRecord], List[ReplayStepRecord], List[ReplayWindowRecord], Dict[str, Any]]:
    """Ingest rollout-capture bundles into canonical replay records."""

    root = Path(rollout_root)
    resolved_scenario_id = scenario_id
    base_dir = root
    if resolved_scenario_id is None:
        if any(root.glob("episode_*")):
            resolved_scenario_id = root.name
            base_dir = root.parent
        else:
            scenario_dirs = [path for path in sorted(root.iterdir()) if path.is_dir() and any(path.glob("episode_*"))]
            if len(scenario_dirs) == 1:
                resolved_scenario_id = scenario_dirs[0].name
                base_dir = root
            else:
                raise ReplayIngestionError(
                    f"Could not infer rollout scenario under {root}; provide scenario_id explicitly"
                )
    bundle = finalize_rollout_bundle(resolved_scenario_id, base_dir)
    episodes: List[ReplayEpisodeRecord] = []
    steps: List[ReplayStepRecord] = []
    windows: List[ReplayWindowRecord] = []

    for rollout in bundle.episodes:
        episode_dir = rollout.trajectory_path.parent
        raw_rollout_metadata = _load_rollout_metadata_payload(episode_dir)
        artifact_refs = _discover_rollout_artifact_refs(episode_dir, rollout.metadata.episode_id)
        scene_tracks_metadata = _scene_tracks_rollout_metadata(
            artifact_refs.get("scene_tracks_path"),
            metadata_payload=raw_rollout_metadata,
        )
        semantic_world_model_summary = _semantic_world_model_rollout_summary(
            artifact_refs.get("semantic_world_model_path")
        )
        trajectory_rows = _load_rollout_trajectory(rollout.trajectory_path, metrics=rollout.metrics)
        episode_log = EpisodeLog(
            metadata=rollout.metadata,
            trajectory=trajectory_rows,
            info_history=[dict(row.get("info", {}) or {}) for row in trajectory_rows],
            metrics={
                "reward_total": float(rollout.metrics.get("reward", 0.0)),
                "steps": len(trajectory_rows),
                "time_step_s": 1.0,
                "throughput_units_per_hour": max(0.0, float(len(trajectory_rows)) * 12.0),
                "error_rate": float(rollout.metrics.get("error_rate", 0.0)),
                "safety_score": float(rollout.metrics.get("safety_score", 0.9)),
                "energy_wh_per_unit": float(rollout.metrics.get("energy_wh_per_unit", 2.0)),
            },
        )
        temp_payload = episode_log.to_dict()
        temp_payload["metadata"]["seed"] = rollout.metadata.seed
        temp_path = root / f".rollout_replay_{rollout.metadata.episode_id}.json"
        temp_path.write_text(json.dumps(temp_payload), encoding="utf-8")
        try:
            e_rows, s_rows, w_rows, _ = ingest_workcell_episode_log(
                temp_path,
                run_id=run_id or f"rollout_replay_{sha256_json({'scenario_id': resolved_scenario_id, 'episode_id': rollout.metadata.episode_id})[:10]}",
                source_domain=source_domain,
                objective_profile_id=objective_profile_id,
                pricing_policy_path=pricing_policy_path,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()
        for episode in e_rows:
            episode_payload = episode.to_dict()
            episode_payload["metadata"] = {
                **dict(episode_payload.get("metadata", {}) or {}),
                "rollout_episode_dir": str(episode_dir.resolve()),
                "sensor_bundle": dict(raw_rollout_metadata.get("sensor_bundle", {}) or {})
                if isinstance(raw_rollout_metadata.get("sensor_bundle"), Mapping)
                else {},
                "scene_tracks_non_stub": bool(scene_tracks_metadata.get("scene_tracks_non_stub", False)),
                "scene_tracks_backend": str(scene_tracks_metadata.get("scene_tracks_backend", "")),
                "scene_tracks_training_eligible": bool(scene_tracks_metadata.get("scene_tracks_training_eligible", False)),
                "semantic_memory_grounded": bool(
                    semantic_world_model_summary.get("topology", {}).get("grounded_track_object_count", 0)
                    or scene_tracks_metadata.get("semantic_grounding_ready", False)
                ),
                "semantic_grounding_non_heuristic": bool(
                    scene_tracks_metadata.get("semantic_grounding_non_heuristic", False)
                    or (
                        raw_rollout_metadata.get("semantic_grounding_non_heuristic", False)
                        and scene_tracks_metadata.get("scene_tracks_backend", "") not in {"passthrough", "stub", "auto"}
                    )
                ),
                "semantic_density_score": float(scene_tracks_metadata.get("semantic_density_score", 0.0)),
                "openvla_backend_selected": str(raw_rollout_metadata.get("openvla_backend_selected", "")),
                "openvla_vision_backbone_selected": str(raw_rollout_metadata.get("openvla_vision_backbone_selected", "")),
                "semantic_world_model_summary": semantic_world_model_summary,
            }
            episode_payload["provenance"] = {
                **dict(episode_payload.get("provenance", {}) or {}),
                "source_adapter": "rollout_capture_bundle_v1",
                "scenario_id": resolved_scenario_id,
                "trajectory_path": str(rollout.trajectory_path),
                **dict(artifact_refs),
            }
            episodes.append(
                ReplayEpisodeRecord.from_dict(episode_payload)
            )
        for step in s_rows:
            step_payload = step.to_dict()
            step_payload["metadata"] = {
                **dict(step_payload.get("metadata", {}) or {}),
                "scene_tracks_backend": str(scene_tracks_metadata.get("scene_tracks_backend", "")),
                "semantic_density_score": float(scene_tracks_metadata.get("semantic_density_score", 0.0)),
            }
            step_payload["provenance"] = {
                **dict(step_payload.get("provenance", {}) or {}),
                "source_adapter": "rollout_capture_bundle_v1",
                "scenario_id": resolved_scenario_id,
                "trajectory_path": str(rollout.trajectory_path),
                **dict(artifact_refs),
            }
            steps.append(
                ReplayStepRecord.from_dict(step_payload)
            )
        for window in w_rows:
            window_payload = window.to_dict()
            window_payload["metadata"] = {
                **dict(window_payload.get("metadata", {}) or {}),
                "scene_tracks_backend": str(scene_tracks_metadata.get("scene_tracks_backend", "")),
                "semantic_density_score": float(scene_tracks_metadata.get("semantic_density_score", 0.0)),
            }
            window_payload["provenance"] = {
                **dict(window_payload.get("provenance", {}) or {}),
                "source_adapter": "rollout_capture_bundle_v1",
                "scenario_id": resolved_scenario_id,
                "trajectory_path": str(rollout.trajectory_path),
                **dict(artifact_refs),
            }
            windows.append(
                ReplayWindowRecord.from_dict(window_payload)
            )

    metadata = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_adapter": "rollout_capture_bundle_v1",
        "scenario_id": resolved_scenario_id,
        "source_root": str(root),
        "provenance_digest": sha256_json({"source_root": str(root), "scenario_id": resolved_scenario_id}),
    }
    return _sort_episode_records(episodes), _sort_step_records(steps), _sort_window_records(windows), metadata


def build_shadow_condition_vector(
    *,
    condition_builder: ConditionVectorBuilder,
    task_id: str,
    env_id: str,
    source_domain: str,
    objective_summary: Mapping[str, Any],
    econ_summary: Mapping[str, Any],
    constraint_flags: Sequence[Mapping[str, Any]],
    semantic_tags: Sequence[str],
    objective_profile_id: str,
    uncertainty: float,
    trust_score: float,
    episode_step: int,
) -> ConditionVector:
    """Build a ConditionVector aligned to the repo's Hydra-style conditioning path."""

    objective_axes = dict(objective_summary.get("axes", {}) or {})
    normalized_axes = dict(objective_summary.get("normalized_axes", {}) or {})
    econ_axes = dict(econ_summary.get("axes", {}) or {})
    hard_flags = sum(1 for flag in constraint_flags if str(flag.get("severity", "")) == "hard")
    skill_mode = _resolve_skill_mode(
        constraint_flags=constraint_flags,
        semantic_tags=semantic_tags,
        frontier_gain=float(econ_axes.get("marginal_frontier_gain", 0.0)),
        uncertainty=float(uncertainty),
    )
    objective_vector = [
        float(normalized_axes.get("throughput", 0.0)),
        float(normalized_axes.get("error", 0.0)),
        float(normalized_axes.get("safety", 0.0)),
        float(normalized_axes.get("energy", 0.0)),
    ]
    return condition_builder.build(
        episode_config={
            "task_id": task_id,
            "env_id": env_id,
            "backend_id": source_domain,
            "objective_preset": objective_profile_id or "balanced_contract",
            "objective_vector": objective_vector,
        },
        econ_state={
            "target_mpl": float(objective_axes.get("throughput", 0.0)),
            "current_wage_parity": float(econ_axes.get("price_tick", 0.0)) / 28.0 if float(econ_axes.get("price_tick", 0.0)) else 0.0,
            "energy_budget_wh": max(1.0, float(objective_axes.get("energy", 0.0)) * 4.0),
        },
        curriculum_phase="shadow_replay",
        sima2_trust=float(trust_score),
        datapack_metadata={"tags": list(semantic_tags), "phase": "shadow_replay"},
        episode_step=episode_step,
        overrides={
            "skill_mode": skill_mode,
            "objective_preset": objective_profile_id or "balanced_contract",
            "novelty_tier": min(3, len(set(semantic_tags)) + hard_flags),
            "ood_risk_level": float(max(0.0, min(1.0, uncertainty))),
            "recovery_priority": 1.0 if hard_flags else float(max(0.0, min(1.0, uncertainty + 0.1))),
        },
        econ_slice={
            "mpl": float(objective_axes.get("throughput", 0.0)),
            "energy_wh": float(objective_axes.get("energy", 0.0)),
            "wage_parity": float(econ_axes.get("price_tick", 0.0)) / 28.0 if float(econ_axes.get("price_tick", 0.0)) else 0.0,
        },
        semantic_tags={str(tag): 1.0 for tag in semantic_tags},
        trust_summary={"shadow_replay": float(trust_score)},
        episode_metadata={"episode_id": f"{task_id}:{env_id}", "step": episode_step},
        advisory_context={"skill_mode": skill_mode, "frontier_score": float(econ_axes.get("marginal_frontier_gain", 0.0))},
    )


def _episode_log_window_records(
    *,
    runtime_record: ObjectiveRuntimeRecord,
    step_records: Sequence[ReplayStepRecord],
    objective_profile_id: str,
    pricing_policy_path: str | Path,
    condition_builder: ConditionVectorBuilder,
) -> List[ReplayWindowRecord]:
    functor = ObjectiveEconFunctor(base_price_per_unit=3.0)
    pricing = PricingSentinel.from_path(pricing_policy_path)
    builder = ObjectiveRuntimeBuilder()
    records: List[ReplayWindowRecord] = []
    for window in runtime_record.windows:
        metrics = dict(window.metrics)
        window_runtime = ObjectiveRuntimeRecord(
            task_id=runtime_record.task_id,
            episode_id=runtime_record.episode_id,
            env_id=runtime_record.env_id,
            world_id=runtime_record.world_id,
            robot_id=runtime_record.robot_id,
            source_domain=runtime_record.source_domain,
            seed=runtime_record.seed,
            run_id=runtime_record.run_id,
            timestamp=runtime_record.timestamp,
            episode_metrics=metrics,
            reward_components=dict(window.reward_components),
            telemetry=dict(window.telemetry),
        )
        objective_summary = summarize_objective_tensor(builder.build(window_runtime))
        constraint_flags: List[Dict[str, Any]] = []
        econ_summary = summarize_econ_tensor(
            functor.map(
                builder.build(window_runtime),
                constraint_flags=constraint_flags,
                uncertainty=float(window.telemetry.get("uncertainty", 0.15)),
                context={"window_id": window.window_id},
            )
        )
        tick = pricing.emit_tick(
            PricingTickInput(
                run_id=runtime_record.run_id,
                episode_id=runtime_record.episode_id,
                objective_profile_id=objective_profile_id,
                source_domain=str(runtime_record.source_domain),
                timestamp=runtime_record.timestamp,
                mode="step_window",
                econ_tensor=econ_summary,
                uncertainty=float(window.telemetry.get("uncertainty", 0.15)),
                constraint_flags=constraint_flags,
                trust_score=float(window.telemetry.get("trust_score", 0.75)),
                tick_id=f"{runtime_record.episode_id}_{window.window_id}",
                start_step=window.start_step,
                end_step=window.end_step,
                metadata={"window_id": window.window_id},
            )
        )
        condition = build_shadow_condition_vector(
            condition_builder=condition_builder,
            task_id=runtime_record.task_id,
            env_id=runtime_record.env_id,
            source_domain=str(runtime_record.source_domain),
            objective_summary=objective_summary,
            econ_summary=econ_summary,
            constraint_flags=constraint_flags,
            semantic_tags=[],
            objective_profile_id=objective_profile_id,
            uncertainty=float(window.telemetry.get("uncertainty", 0.15)),
            trust_score=float(window.telemetry.get("trust_score", 0.75)),
            episode_step=window.start_step,
        )
        window_steps = [row for row in step_records if window.start_step <= row.step_idx <= window.end_step]
        records.append(
            ReplayWindowRecord(
                run_id=runtime_record.run_id,
                episode_id=runtime_record.episode_id,
                window_id=window.window_id,
                start_step=window.start_step,
                end_step=window.end_step,
                task_id=runtime_record.task_id,
                env_id=runtime_record.env_id,
                source_domain=str(runtime_record.source_domain),
                seed=runtime_record.seed,
                timestamp=_timestamp_for_step(runtime_record.timestamp, step_idx=window.start_step, time_step_s=float(runtime_record.episode_metrics.get("time_step_s", 1.0))),
                reward_sum=sum(row.reward for row in window_steps),
                obs_vector_mean=_mean_vectors([row.obs_vector for row in window_steps]),
                action_vector_mean=_mean_vectors([row.action_vector for row in window_steps]),
                condition_vector=condition.to_dict(),
                condition_vector_values=[float(value) for value in condition.to_vector().tolist()],
                skill_mode=condition.skill_mode,
                objective_tensor_summary=objective_summary,
                econ_tensor_summary=econ_summary,
                pricing_summary=tick.to_dict(),
                constraint_flags=constraint_flags,
                metadata={"source": "workcell_episode_log_sliding_window"},
            )
        )
    return records


def _build_episode_log_windows(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    time_step_s: float,
    window_size: int = 2,
) -> List[ObjectiveRuntimeWindow]:
    windows: List[ObjectiveRuntimeWindow] = []
    for start in range(0, len(trajectory), max(1, window_size)):
        end = min(len(trajectory), start + max(1, window_size)) - 1
        window_steps = trajectory[start : end + 1]
        reward_sum = sum(float(step.get("info", {}).get("reward", 0.0)) for step in window_steps)
        windows.append(
            ObjectiveRuntimeWindow(
                window_id=f"window_{start:03d}_{end:03d}",
                start_step=start,
                end_step=end,
                metrics={
                    "steps": len(window_steps),
                    "duration_s": len(window_steps) * float(time_step_s),
                    "reward_total": reward_sum,
                    "throughput_units_per_hour": float(len(window_steps)) * 3600.0 / max(float(time_step_s) * len(window_steps), 1.0),
                    "energy_wh_per_unit": 1.0,
                    "error_rate": 0.0,
                    "safety_score": 1.0,
                },
                telemetry={},
                metadata={"source": "episode_log_window_builder"},
            )
        )
    return windows


def _extract_semantic_tags(runtime_record: Mapping[str, Any]) -> List[str]:
    telemetry = dict(runtime_record.get("telemetry", {}) or {})
    tags = telemetry.get("semantic_tags", []) or []
    return [str(tag) for tag in tags]


def _pick_window_tick_id(pricing_ticks: Sequence[Mapping[str, Any]], *, step_idx: int) -> Optional[str]:
    for row in pricing_ticks:
        if row.get("mode") != "step_window":
            continue
        start_step = row.get("start_step")
        end_step = row.get("end_step")
        if start_step is None or end_step is None:
            continue
        if int(start_step) <= step_idx <= int(end_step):
            return str(row.get("tick_id"))
    return None


def _mean_vectors(vectors: Iterable[Sequence[float]]) -> List[float]:
    vectors = [list(vector) for vector in vectors if vector]
    if not vectors:
        return []
    dim = max(len(vector) for vector in vectors)
    totals = [0.0] * dim
    for vector in vectors:
        padded = list(vector) + [0.0] * (dim - len(vector))
        for index, value in enumerate(padded):
            totals[index] += float(value)
    return [value / len(vectors) for value in totals]


def _extract_obs_vector(obs: Mapping[str, Any]) -> List[float]:
    if "state_vector" in obs and isinstance(obs.get("state_vector"), list):
        return [float(value) for value in list(obs.get("state_vector") or [])]
    return _flatten_numeric_payload(obs)


def _extract_action_vector(action: Mapping[str, Any]) -> List[float]:
    if not action:
        return [0.0]
    if "action_vector" in action and isinstance(action.get("action_vector"), list):
        return [float(value) for value in list(action.get("action_vector") or [])]
    task_state = dict(action.get("task_state", {}) or {})
    target = str(action.get("target", ""))
    slot_match = _TARGET_RE.search(target)
    slot_index = int(slot_match.group(1)) if slot_match else 0
    placement_count = len(list(task_state.get("placement_order", []) or []))
    return [
        _hash_to_unit(str(action.get("action_type", "unknown"))),
        float(slot_index),
        float(task_state.get("collision_count", 0.0) or 0.0),
        float(task_state.get("constraint_error", 0.0) or 0.0),
        float(task_state.get("constraint_violations", 0.0) or 0.0),
        float(placement_count),
        1.0 if bool(task_state.get("respect_fragility", False)) else 0.0,
    ]


def _flatten_numeric_payload(payload: Any) -> List[float]:
    values: List[float] = []
    if isinstance(payload, Mapping):
        for key, value in sorted(payload.items(), key=lambda item: str(item[0])):
            if isinstance(value, Mapping):
                values.extend(_flatten_numeric_payload(value))
            elif isinstance(value, (list, tuple)):
                values.extend(_flatten_numeric_payload(list(value)))
            elif isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                values.append(float(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            values.extend(_flatten_numeric_payload(value))
    elif isinstance(payload, bool):
        values.append(1.0 if payload else 0.0)
    elif isinstance(payload, (int, float)):
        values.append(float(payload))
    return values


def _resolve_skill_mode(
    *,
    constraint_flags: Sequence[Mapping[str, Any]],
    semantic_tags: Sequence[str],
    frontier_gain: float,
    uncertainty: float,
) -> str:
    semantic_tags = [str(tag) for tag in semantic_tags]
    if "fragile" in semantic_tags or any(str(flag.get("severity", "")) == "hard" for flag in constraint_flags):
        return "safety_critical"
    if frontier_gain > 0.20 or uncertainty > 0.35:
        return "frontier_exploration"
    if any(str(flag.get("axis", "")) == "error" for flag in constraint_flags):
        return "recovery_heavy"
    return "efficiency_throughput"


def _hash_to_unit(value: str) -> float:
    digest = sha256_json({"value": value})
    return int(digest[:12], 16) / float(16 ** 12)


def _timestamp_for_step(timestamp: str, *, step_idx: int, time_step_s: float) -> str:
    if not timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp
    return (dt + timedelta(seconds=float(time_step_s) * int(step_idx))).isoformat()


def _events_for_step(
    events: Sequence[RuntimeEvent],
    *,
    step_idx: int,
) -> List[RuntimeEvent]:
    return [event for event in events if _scope_applies_to_step(event.scope, step_idx=step_idx)]


def _decisions_for_step(
    decisions: Sequence[DecisionLedgerEntry],
    *,
    step_idx: int,
) -> List[DecisionLedgerEntry]:
    return [decision for decision in decisions if _scope_applies_to_step(decision.scope, step_idx=step_idx)]


def _events_for_window(
    events: Sequence[RuntimeEvent],
    *,
    start_step: int,
    end_step: int,
) -> List[RuntimeEvent]:
    return [
        event
        for event in events
        if _scope_applies_to_window(event.scope, start_step=start_step, end_step=end_step)
    ]


def _decisions_for_window(
    decisions: Sequence[DecisionLedgerEntry],
    *,
    start_step: int,
    end_step: int,
) -> List[DecisionLedgerEntry]:
    return [
        decision
        for decision in decisions
        if _scope_applies_to_window(decision.scope, start_step=start_step, end_step=end_step)
    ]


def _scope_applies_to_step(scope: Mapping[str, Any], *, step_idx: int) -> bool:
    scope_kind = str(scope.get("scope_kind", "episode"))
    if scope_kind == "episode":
        return True
    if scope_kind == "step":
        return int(scope.get("step_idx", -1)) == int(step_idx)
    start_step = scope.get("start_step")
    end_step = scope.get("end_step")
    if start_step is None or end_step is None:
        return scope_kind == "window"
    return int(start_step) <= int(step_idx) <= int(end_step)


def _scope_applies_to_window(
    scope: Mapping[str, Any],
    *,
    start_step: int,
    end_step: int,
) -> bool:
    scope_kind = str(scope.get("scope_kind", "episode"))
    if scope_kind == "episode":
        return True
    if scope_kind == "step":
        step_idx = int(scope.get("step_idx", -1))
        return int(start_step) <= step_idx <= int(end_step)
    event_start = scope.get("start_step")
    event_end = scope.get("end_step")
    if event_start is None or event_end is None:
        return scope_kind == "window"
    return not (int(event_end) < int(start_step) or int(event_start) > int(end_step))


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise ReplayIngestionError(f"Missing required replay artifact {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sort_episode_records(records: Sequence[ReplayEpisodeRecord]) -> List[ReplayEpisodeRecord]:
    return sorted(records, key=lambda record: (record.run_id, record.episode_id))


def _sort_step_records(records: Sequence[ReplayStepRecord]) -> List[ReplayStepRecord]:
    return sorted(records, key=lambda record: (record.run_id, record.episode_id, record.step_idx))


def _sort_window_records(records: Sequence[ReplayWindowRecord]) -> List[ReplayWindowRecord]:
    return sorted(records, key=lambda record: (record.run_id, record.episode_id, record.start_step, record.window_id))


def _load_rollout_trajectory(path: Path, *, metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = np.load(path, allow_pickle=True)
        trajectory = payload.get("trajectory")
    except Exception:
        trajectory = None
    if trajectory is None:
        return []
    if isinstance(trajectory, np.ndarray):
        trajectory = trajectory.tolist()
    if isinstance(trajectory, Mapping):
        if isinstance(trajectory.get("trajectory"), list):
            trajectory = list(trajectory.get("trajectory", []) or [])
        else:
            states = list(trajectory.get("states", []) or [])
            actions = list(trajectory.get("actions", []) or [])
            count = max(len(states), len(actions))
            trajectory = []
            for index in range(count):
                raw_state = states[index] if index < len(states) else {}
                raw_action = actions[index] if index < len(actions) else {}
                state = dict(raw_state) if isinstance(raw_state, Mapping) else {}
                action = dict(raw_action) if isinstance(raw_action, Mapping) else {}
                obs = dict(state.get("obs", {}) or {})
                if not obs:
                    obs = {"state_vector": _flatten_numeric_payload(state) or [float(index)]}
                info = dict(state.get("info", {}) or {})
                trajectory.append(
                    {
                        "step": int(state.get("step", index)),
                        "obs": obs,
                        "action": action or {"action_vector": [0.0]},
                        "done": bool(state.get("done", index == count - 1)),
                        "info": info,
                    }
                )
    if not isinstance(trajectory, list):
        trajectory = [trajectory]

    rows: List[Dict[str, Any]] = []
    per_step_reward = float(metrics.get("reward", 0.0)) / float(max(1, len(trajectory)))
    for index, entry in enumerate(trajectory):
        if isinstance(entry, Mapping):
            rows.append(
                {
                    "step": int(entry.get("step", index)),
                    "obs": dict(entry.get("obs", {}) or {"state_vector": _flatten_numeric_payload(entry)}),
                    "action": dict(entry.get("action", {}) or {"action_vector": [0.0]}),
                    "done": bool(entry.get("done", index == len(trajectory) - 1)),
                    "info": dict(entry.get("info", {}) or {"reward": per_step_reward}),
                }
            )
            continue
        values = _flatten_numeric_payload(entry)
        rows.append(
            {
                "step": index,
                "obs": {"state_vector": values or [float(index)]},
                "action": {"action_vector": [0.0]},
                "done": index == len(trajectory) - 1,
                "info": {"reward": per_step_reward},
            }
        )
    return rows
