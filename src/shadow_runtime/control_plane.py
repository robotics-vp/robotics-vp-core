"""End-to-end shadow economic control plane runner."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.constraints.constraint_set import ConstraintSet
from src.determinism.determinism_context import get_context_summary, set_determinism
from src.economics.functor import ObjectiveEconFunctor
from src.economics.pricing_sentinel import PricingSentinel, PricingTickInput
from src.economics.value_ledger import ValueLedger, summarize_econ_tensor
from src.objectives.profile_loader import ObjectiveContractProfile, load_contract_profile
from src.objectives.runtime_builder import ObjectiveRuntimeBuilder, summarize_objective_tensor
from src.objectives.tensor import ObjectiveTensor
from src.ontology.shadow_updates import ShadowDatapackCreditUpdate, persist_shadow_episode
from src.ontology.store import OntologyStore
from src.regality import MetaRegalController, ShadowRegalContext
from src.runtime.event_spine import (
    DecisionLedgerEntry,
    RuntimeEvent,
    decision_ledger_sidecar_payload,
    event_spine_sidecar_payload,
)
from src.runtime.packets import (
    RuntimePacket,
    SchemaRef,
    runtime_packet_from_record,
    runtime_packet_sidecar_payload,
)
from src.shadow_runtime.demo_source import ShadowEpisodeTrace, generate_workcell_shadow_batch
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


@dataclass(frozen=True)
class ShadowEpisodeArtifacts:
    """Per-episode outputs emitted by the shadow control plane."""

    episode_id: str
    objective_tensor: Dict[str, Any]
    constraint_set: Dict[str, Any]
    constraint_flags: list[Dict[str, Any]]
    objective_compile: Dict[str, Any]
    econ_tensor: Dict[str, Any]
    pricing_ticks: list[Dict[str, Any]]
    regal_decision: Dict[str, Any]
    datapack_credit_update: Dict[str, Any]
    ontology_refs: Dict[str, Any]
    runtime_packet: Dict[str, Any]
    event_refs: list[str]
    decision_refs: list[str]
    baseline_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "objective_tensor": dict(self.objective_tensor),
            "constraint_set": dict(self.constraint_set),
            "constraint_flags": list(self.constraint_flags),
            "objective_compile": dict(self.objective_compile),
            "econ_tensor": dict(self.econ_tensor),
            "pricing_ticks": list(self.pricing_ticks),
            "regal_decision": dict(self.regal_decision),
            "datapack_credit_update": dict(self.datapack_credit_update),
            "ontology_refs": dict(self.ontology_refs),
            "runtime_packet": dict(self.runtime_packet),
            "event_refs": list(self.event_refs),
            "decision_refs": list(self.decision_refs),
            "baseline_summary": dict(self.baseline_summary),
        }


@dataclass(frozen=True)
class ShadowRunResult:
    """Aggregate shadow run outputs used by scripts and tests."""

    run_id: str
    objective_profile_id: str
    include_regal: bool
    output_dir: str
    summary: Dict[str, Any]
    episode_artifacts: list[Dict[str, Any]]
    artifact_paths: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective_profile_id": self.objective_profile_id,
            "include_regal": self.include_regal,
            "output_dir": self.output_dir,
            "summary": dict(self.summary),
            "episode_artifacts": list(self.episode_artifacts),
            "artifact_paths": dict(self.artifact_paths),
        }


def run_shadow_control_plane(
    *,
    output_dir: str | Path,
    seed: int = 42,
    episodes: int = 2,
    objective_profile_id: str = "balanced_contract",
    pricing_policy_path: str | Path = "config/pricing/default.yaml",
    include_regal: bool = True,
    timestamp_base: Optional[str] = None,
    run_id: Optional[str] = None,
    episode_traces: Optional[Sequence[ShadowEpisodeTrace]] = None,
) -> ShadowRunResult:
    """Run the full shadow accounting, pricing, and governance loop."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    set_determinism(seed=seed)

    if episode_traces is not None:
        traces = list(episode_traces)
        run_id = run_id or traces[0].run_id
    else:
        run_id = run_id or f"shadow_{sha256_json({'seed': seed, 'episodes': episodes, 'profile': objective_profile_id})[:10]}"
        traces = list(generate_workcell_shadow_batch(
            run_id=run_id,
            seed=seed,
            episodes=episodes,
            timestamp_base=timestamp_base,
        ))
    if not traces:
        raise ValueError("No episode traces available for the shadow control plane")

    contract_profile = load_contract_profile(objective_profile_id)
    runtime_builder = ObjectiveRuntimeBuilder()
    functor = ObjectiveEconFunctor(base_price_per_unit=3.0)
    pricing_sentinel = PricingSentinel.from_path(pricing_policy_path)
    regality = MetaRegalController() if include_regal else None

    ontology_root = output_root / "ontology"
    sidecar_dir = output_root / "ontology_sidecars"
    ontology_store = OntologyStore(root_dir=str(ontology_root))

    artifact_paths = {
        "objective_tensor": str(output_root / "objective_tensor.json"),
        "constraint_set": str(output_root / "constraint_set.json"),
        "constraint_flags": str(output_root / "constraint_flags.json"),
        "objective_compile": str(output_root / "objective_compile.json"),
        "econ_tensor": str(output_root / "econ_tensor.json"),
        "pricing_ticks": str(output_root / "pricing_ticks.jsonl"),
        "regal_decisions": str(output_root / "regal_decisions.json"),
        "value_ledger": str(output_root / "value_ledger.jsonl"),
        "datapack_credit_update": str(output_root / "datapack_credit_update.json"),
        "runtime_packets": str(output_root / "runtime_packets.json"),
        "event_spine": str(output_root / "event_spine.json"),
        "decision_ledger": str(output_root / "decision_ledger.json"),
        "shadow_episode_traces": str(output_root / "shadow_episode_traces.json"),
        "summary_json": str(output_root / "summary.json"),
        "summary_md": str(output_root / "summary.md"),
    }
    _reset_output_files([artifact_paths["pricing_ticks"], artifact_paths["value_ledger"]])
    ledger = ValueLedger(artifact_paths["value_ledger"])

    objective_payload = {
        "run_id": run_id,
        "objective_profile_id": contract_profile.profile_id,
        "episodes": [],
        "windows": [],
    }
    constraint_payload = {"run_id": run_id, "episodes": []}
    constraint_flags_payload = {"run_id": run_id, "episodes": []}
    compile_payload = {
        "run_id": run_id,
        "objective_profile": contract_profile.to_dict(),
        "episodes": [],
    }
    econ_payload = {"run_id": run_id, "episodes": [], "windows": []}
    regal_payload = {"run_id": run_id, "enabled": include_regal, "episodes": []}
    datapack_payload = {"run_id": run_id, "updates": []}
    pricing_rows: list[Dict[str, Any]] = []
    episode_artifacts: list[ShadowEpisodeArtifacts] = []
    runtime_packets: list[RuntimePacket] = []
    runtime_events: list[RuntimeEvent] = []
    decision_entries: list[DecisionLedgerEntry] = []
    event_sequence_idx = 0
    decision_sequence_idx = 0

    for trace in traces:
        objective_tensor = runtime_builder.build(trace.runtime_record)
        objective_summary = summarize_objective_tensor(objective_tensor)
        window_objectives = runtime_builder.build_window_tensors(trace.runtime_record)
        constraint_set = _build_constraint_set(trace, contract_profile)
        runtime_flags = constraint_set.flag_observations(trace.constraint_observations)
        compile_result = runtime_builder.compile_contract(
            objective_tensor,
            contract_profile.profile,
            metadata={"contract_profile_hash": contract_profile.profile_hash},
        )
        combined_flags = _merge_constraint_flags(compile_result.constraint_flags, runtime_flags)
        uncertainty = float(trace.runtime_record.telemetry.get("uncertainty", trace.baseline_summary.get("uncertainty", 0.0)))
        trust_score = float(trace.runtime_record.telemetry.get("trust_score", trace.baseline_summary.get("trust_score", 1.0)))

        econ_tensor = functor.map(
            objective_tensor,
            constraint_flags=combined_flags,
            uncertainty=uncertainty,
            context={
                "run_id": run_id,
                "episode_id": trace.episode_id,
                "objective_profile_id": contract_profile.profile_id,
                "source_domain": trace.source_domain,
            },
        )
        econ_summary = summarize_econ_tensor(econ_tensor)
        runtime_packet = _build_runtime_packet(
            trace=trace,
            contract_profile=contract_profile,
            objective_tensor=objective_tensor,
            econ_tensor=econ_tensor,
            constraint_set=constraint_set,
            constraint_flags=combined_flags,
        )
        runtime_packets.append(runtime_packet)
        episode_tick = pricing_sentinel.emit_tick(
            PricingTickInput(
                run_id=run_id,
                episode_id=trace.episode_id,
                objective_profile_id=contract_profile.profile_id,
                source_domain=trace.source_domain,
                timestamp=trace.runtime_record.timestamp,
                mode="episode",
                econ_tensor=econ_tensor,
                uncertainty=uncertainty,
                constraint_flags=combined_flags,
                trust_score=trust_score,
                metadata={"uncertainty": uncertainty, "kind": "episode_aggregate"},
            )
        )
        pricing_rows.append(episode_tick.to_dict())

        window_ticks: list[Dict[str, Any]] = []
        for window_payload, window in zip(window_objectives, trace.runtime_record.windows):
            window_tensor = ObjectiveTensor.from_dict(window_payload["objective_tensor"])
            window_compile = runtime_builder.compile_contract(
                window_tensor,
                contract_profile.profile,
                metadata={"window_id": window.window_id},
            )
            window_observations = {
                **dict(window.metrics),
                "throughput": window.metrics.get("throughput_units_per_hour", 0.0),
                "error": window.metrics.get("error_rate", 0.0),
                "safety": window.metrics.get("safety_score", 0.0),
                "energy": window.metrics.get("energy_wh_per_unit", 0.0),
                "respect_fragility": trace.constraint_observations.get("respect_fragility", True),
                "contact_force_n": trace.constraint_observations.get("contact_force_n", 0.0),
            }
            window_flags = _merge_constraint_flags(
                window_compile.constraint_flags,
                constraint_set.flag_observations(window_observations),
            )
            window_uncertainty = float(window.telemetry.get("uncertainty", uncertainty))
            window_trust = float(window.telemetry.get("trust_score", trust_score))
            window_econ_tensor = functor.map(
                window_tensor,
                constraint_flags=window_flags,
                uncertainty=window_uncertainty,
                context={
                    "run_id": run_id,
                    "episode_id": trace.episode_id,
                    "window_id": window.window_id,
                    "source_domain": trace.source_domain,
                },
            )
            econ_payload["windows"].append(
                {
                    "episode_id": trace.episode_id,
                    "window": window.to_dict(),
                    "econ_tensor": window_econ_tensor.to_dict(),
                }
            )
            tick = pricing_sentinel.emit_tick(
                PricingTickInput(
                    run_id=run_id,
                    episode_id=trace.episode_id,
                    objective_profile_id=contract_profile.profile_id,
                    source_domain=trace.source_domain,
                    timestamp=trace.runtime_record.timestamp,
                    mode="step_window",
                    econ_tensor=window_econ_tensor,
                    uncertainty=window_uncertainty,
                    constraint_flags=window_flags,
                    trust_score=window_trust,
                    tick_id=f"{trace.episode_id}_{window.window_id}",
                    start_step=window.start_step,
                    end_step=window.end_step,
                    metadata={"uncertainty": window_uncertainty, "window_id": window.window_id},
                )
            )
            tick_dict = tick.to_dict()
            pricing_rows.append(tick_dict)
            window_ticks.append(tick_dict)

        pre_regal_datapack_update = {
            "datapack_id": trace.datapack_id,
            "episode_id": trace.episode_id,
            "run_id": run_id,
            "objective_profile_id": contract_profile.profile_id,
            "marginal_frontier_gain": float(econ_summary["axes"].get("marginal_frontier_gain", 0.0)),
            "data_share_credit": float(episode_tick.data_share_credit),
            "quality_score": float(trace.baseline_summary.get("quality_score", 0.0)),
            "recommendation": "keep",
        }
        regal_summary = _default_regal_summary()
        if regality is not None:
            regal_context = ShadowRegalContext(
                run_id=run_id,
                episode_id=trace.episode_id,
                source_domain=trace.source_domain,
                objective_tensor=objective_tensor.to_dict(),
                objective_profile=contract_profile.to_dict(),
                compile_artifact=compile_result.to_dict(),
                constraint_set=constraint_set.to_dict(),
                constraint_flags=combined_flags,
                econ_tensor=econ_summary,
                pricing_ticks=[episode_tick.to_dict(), *window_ticks],
                datapack_credit_update=pre_regal_datapack_update,
                episode_metrics=trace.runtime_record.episode_metrics,
                provenance={
                    "objective_tensor_hash": objective_summary["schema_hash"],
                    "econ_tensor_hash": econ_summary["schema_hash"],
                    "trace_hash": sha256_json(trace.to_dict()),
                },
                evidence_pointers={
                    "objective_tensor": "objective_tensor.json",
                    "pricing_ticks": "pricing_ticks.jsonl",
                    "value_ledger": "value_ledger.jsonl",
                    "runtime_packets": "runtime_packets.json",
                },
            )
            regal_summary = regality.evaluate(regal_context).to_dict()

        datapack_update = ShadowDatapackCreditUpdate(
            datapack_id=trace.datapack_id,
            episode_id=trace.episode_id,
            run_id=run_id,
            objective_profile_id=contract_profile.profile_id,
            marginal_frontier_gain=float(econ_summary["axes"].get("marginal_frontier_gain", 0.0)),
            data_share_credit=float(episode_tick.data_share_credit),
            quality_score=float(trace.baseline_summary.get("quality_score", 0.0)),
            recommendation=str(regal_summary.get("datapack_recommendation", "keep")),
            metadata={
                "pricing_confidence": float(episode_tick.confidence),
                "pricing_recommendation": regal_summary.get("pricing_recommendation", "publish"),
                "window_tick_count": len(window_ticks),
            },
        )

        ontology_refs = persist_shadow_episode(
            store=ontology_store,
            sidecar_dir=sidecar_dir,
            task_id=trace.task_id,
            task_name="Shadow Kitting",
            env_id=trace.env_id,
            robot_id=trace.robot_id,
            robot_name="Shadow Sim Arm",
            episode_id=trace.episode_id,
            run_id=run_id,
            source_domain=trace.source_domain,
            started_at=trace.started_at,
            ended_at=trace.ended_at,
            status=trace.status,
            objective_tensor=objective_tensor.to_dict(),
            econ_tensor=econ_tensor.to_dict(),
            pricing_summary=episode_tick.to_dict(),
            regal_summary=regal_summary,
            datapack_update=datapack_update,
            episode_metadata=trace.baseline_summary,
        )

        ledger_receipt = ledger.build_receipt(
            event_type="episode_shadow_receipt",
            run_id=run_id,
            episode_id=trace.episode_id,
            objective_profile_id=contract_profile.profile_id,
            objective_tensor=objective_tensor,
            econ_tensor=econ_tensor,
            pricing_tick=episode_tick,
            constraint_set=constraint_set.summary(trace.constraint_observations),
            regal_decision_summary=regal_summary,
            datapack_id=trace.datapack_id,
            source_domain=trace.source_domain,
            timestamp=trace.ended_at,
        )
        ledger.append(ledger_receipt)

        objective_payload["episodes"].append({"episode_id": trace.episode_id, "objective_tensor": objective_tensor.to_dict()})
        objective_payload["windows"].append({"episode_id": trace.episode_id, "windows": window_objectives})
        constraint_payload["episodes"].append({"episode_id": trace.episode_id, "constraint_set": constraint_set.to_dict()})
        constraint_flags_payload["episodes"].append(
            {
                "episode_id": trace.episode_id,
                "observations": dict(trace.constraint_observations),
                "flags": combined_flags,
            }
        )
        compile_payload["episodes"].append({"episode_id": trace.episode_id, "objective_compile": compile_result.to_dict()})
        econ_payload["episodes"].append({"episode_id": trace.episode_id, "econ_tensor": econ_tensor.to_dict()})
        regal_payload["episodes"].append({"episode_id": trace.episode_id, "regal_decision": regal_summary})
        datapack_payload["updates"].append(datapack_update.to_dict())

        episode_events, episode_decisions, event_sequence_idx, decision_sequence_idx = _build_episode_event_artifacts(
            trace=trace,
            runtime_packet=runtime_packet,
            episode_tick=episode_tick.to_dict(),
            window_ticks=window_ticks,
            regal_summary=regal_summary,
            datapack_update=datapack_update,
            artifact_refs=_episode_artifact_refs(),
            event_sequence_start=event_sequence_idx,
            decision_sequence_start=decision_sequence_idx,
        )
        runtime_events.extend(episode_events)
        decision_entries.extend(episode_decisions)

        episode_artifacts.append(
            ShadowEpisodeArtifacts(
                episode_id=trace.episode_id,
                objective_tensor=objective_tensor.to_dict(),
                constraint_set=constraint_set.to_dict(),
                constraint_flags=combined_flags,
                objective_compile=compile_result.to_dict(),
                econ_tensor=econ_tensor.to_dict(),
                pricing_ticks=[episode_tick.to_dict(), *window_ticks],
                regal_decision=regal_summary,
                datapack_credit_update=datapack_update.to_dict(),
                ontology_refs=ontology_refs,
                runtime_packet=runtime_packet.to_dict(),
                event_refs=[event.event_id for event in episode_events],
                decision_refs=[decision.decision_id for decision in episode_decisions],
                baseline_summary=trace.baseline_summary,
            )
        )

    _write_json(artifact_paths["objective_tensor"], objective_payload)
    _write_json(artifact_paths["constraint_set"], constraint_payload)
    _write_json(artifact_paths["constraint_flags"], constraint_flags_payload)
    _write_json(artifact_paths["objective_compile"], compile_payload)
    _write_json(artifact_paths["econ_tensor"], econ_payload)
    _write_jsonl(artifact_paths["pricing_ticks"], pricing_rows)
    _write_json(artifact_paths["regal_decisions"], regal_payload)
    _write_json(artifact_paths["datapack_credit_update"], datapack_payload)
    _write_json(
        artifact_paths["runtime_packets"],
        runtime_packet_sidecar_payload(run_id=run_id, packets=runtime_packets),
    )
    _write_json(
        artifact_paths["event_spine"],
        event_spine_sidecar_payload(run_id=run_id, events=runtime_events),
    )
    _write_json(
        artifact_paths["decision_ledger"],
        decision_ledger_sidecar_payload(run_id=run_id, decisions=decision_entries),
    )
    _write_json(
        artifact_paths["shadow_episode_traces"],
        {
            "run_id": run_id,
            "episodes": [trace.to_dict() for trace in traces],
        },
    )

    summary = _build_summary(
        run_id=run_id,
        objective_profile=contract_profile,
        include_regal=include_regal,
        traces=traces,
        episode_artifacts=episode_artifacts,
        runtime_events=runtime_events,
        decision_entries=decision_entries,
        pricing_rows=pricing_rows,
        artifact_paths=artifact_paths,
    )
    _write_json(artifact_paths["summary_json"], summary)
    Path(artifact_paths["summary_md"]).write_text(_summary_markdown(summary), encoding="utf-8")

    return ShadowRunResult(
        run_id=run_id,
        objective_profile_id=contract_profile.profile_id,
        include_regal=include_regal,
        output_dir=str(output_root),
        summary=summary,
        episode_artifacts=[artifact.to_dict() for artifact in episode_artifacts],
        artifact_paths=artifact_paths,
    )


def _build_constraint_set(
    trace: ShadowEpisodeTrace,
    contract_profile: ObjectiveContractProfile,
) -> ConstraintSet:
    telemetry = trace.runtime_record.telemetry
    hard_constraints = {
        "throughput": {"min": 4.0},
        "error": {"max": 0.35},
        "safety": {"min": 0.65},
        "energy": {"max": 5.5},
        "collision_rate": {"max": 0.05 if telemetry.get("fragile", False) else 0.15},
    }
    if telemetry.get("fragile", False):
        hard_constraints["contact_force_n"] = {"max": 6.0}
    soft_constraints = dict(contract_profile.soft_constraints)
    soft_constraints.setdefault("constraint_error_rate", {"max": 0.12})
    return ConstraintSet.from_runtime(
        hard_constraints=hard_constraints,
        soft_constraints=soft_constraints,
        geometry_hints={
            "manifold_family": "tray_kitting",
            "fixture_layout": "assembly_bench_simple",
            "occlusion_level": trace.runtime_record.telemetry.get("map_first_quality_score", 0.0),
        },
        semantic_evidence={
            "semantic_tags": list(telemetry.get("semantic_tags", []) or []),
            "fragile": bool(telemetry.get("fragile", False)),
            "vla_confidence": telemetry.get("vla_confidence", telemetry.get("trust_score", 0.0)),
            "source": "shadow_workcell_adapter",
        },
        uncertainty={
            "runtime_uncertainty": trace.runtime_record.telemetry.get("uncertainty", 0.0),
            "constraint_error_rate": trace.constraint_observations.get("constraint_error_rate", 0.0),
        },
        trust_metadata={
            "trust_score": trace.runtime_record.telemetry.get("trust_score", 0.0),
            "fusion_confidence": trace.runtime_record.telemetry.get("semantic_fusion_confidence_mean", 0.0),
        },
        kinematic_limits={
            "max_joint_velocity": trace.runtime_record.telemetry.get("max_joint_velocity", 1.0),
            "max_gripper_force": trace.runtime_record.telemetry.get("max_gripper_force", 1.0),
        },
        safety_invariants=(
            {"respect_fragility": True}
            if telemetry.get("fragile", False)
            else {}
        ),
        geometry_priors={
            "bench_reachability": 0.92,
            "tray_alignment": 0.88,
            "map_first_quality_score": trace.runtime_record.telemetry.get("map_first_quality_score", 0.0),
        },
        source_refs={
            "adapter": "workcell_shadow_source_v1",
            "trace_hash": sha256_json(trace.to_dict()),
        },
        metadata={
            "contract_profile_id": contract_profile.profile_id,
            "contract_profile_hash": contract_profile.profile_hash,
        },
    )


def _build_runtime_packet(
    *,
    trace: ShadowEpisodeTrace,
    contract_profile: ObjectiveContractProfile,
    objective_tensor: ObjectiveTensor,
    econ_tensor: Mapping[str, Any],
    constraint_set: ConstraintSet,
    constraint_flags: Sequence[Mapping[str, Any]],
) -> RuntimePacket:
    observation_schema, action_schema = _build_shadow_schema_refs(trace)
    telemetry = trace.runtime_record.telemetry
    return runtime_packet_from_record(
        record=trace.runtime_record,
        contract_id=_runtime_contract_id(
            task_id=trace.task_id,
            objective_profile_id=contract_profile.profile_id,
            embodiment_id=trace.robot_id,
        ),
        objective_profile_id=contract_profile.profile_id,
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint_set,
        observation_schema=observation_schema,
        action_schema=action_schema,
        semantic_evidence={
            "semantic_tags": list(telemetry.get("semantic_tags", []) or []),
            "fragile": bool(telemetry.get("fragile", False)),
            "vla_confidence": float(telemetry.get("vla_confidence", telemetry.get("trust_score", 0.0))),
            "map_first_quality_score": float(telemetry.get("map_first_quality_score", 0.0)),
            "semantic_fusion_confidence_mean": float(telemetry.get("semantic_fusion_confidence_mean", 0.0)),
            "semantic_disagreement_vla_vs_map": float(telemetry.get("semantic_disagreement_vla_vs_map", 0.0)),
            "source_adapter": "shadow_control_plane_artifacts_v1",
        },
        uncertainty={
            "runtime": float(telemetry.get("uncertainty", 0.0)),
            "constraint_error_rate": float(trace.constraint_observations.get("constraint_error_rate", 0.0)),
            "semantic_disagreement": float(telemetry.get("semantic_disagreement_vla_vs_map", 0.0)),
        },
        provenance={
            "source_adapter": "shadow_control_plane_artifacts_v1",
            "trace_hash": sha256_json(trace.to_dict()),
            "episode_log_hash": sha256_json(trace.episode_log),
        },
        metadata={
            "contract_profile_hash": contract_profile.profile_hash,
            "constraint_flag_count": len(list(constraint_flags)),
            "datapack_id": trace.datapack_id,
            "path": "shadow_control_plane",
        },
        semantic_schema_id="shadow_semantic_evidence_sidecar_v1",
    )


def _build_shadow_schema_refs(trace: ShadowEpisodeTrace) -> tuple[SchemaRef, SchemaRef]:
    trajectory = list(trace.episode_log.get("trajectory", []) or [])
    first_step = dict(trajectory[0] or {}) if trajectory else {}
    obs = dict(first_step.get("obs", {}) or {})
    action = dict(first_step.get("action", {}) or {})
    time_step_s = float(trace.runtime_record.episode_metrics.get("time_step_s", 1.0) or 1.0)
    sample_hz = (1.0 / time_step_s) if time_step_s > 0.0 else 0.0
    observation_schema = SchemaRef(
        schema_id="shadow_workcell_observation_v1",
        version="v1",
        shape={
            "obs_keys": sorted(obs.keys()),
            "state_vector_dim": len(list(obs.get("state_vector", []) or [])),
        },
        timing={"sample_hz": sample_hz, "time_step_s": time_step_s},
        provenance={"source_adapter": "shadow_episode_log_v1"},
        metadata={"env_id": trace.env_id},
    )
    action_schema = SchemaRef(
        schema_id="shadow_workcell_action_v1",
        version="v1",
        shape={
            "action_keys": sorted(action.keys()),
            "action_vector_dim": len(list(action.get("action_vector", []) or [])),
            "task_state_keys": sorted((action.get("task_state", {}) or {}).keys()),
        },
        timing={"apply_hz": sample_hz, "time_step_s": time_step_s},
        provenance={"source_adapter": "shadow_episode_log_v1"},
        metadata={"robot_id": trace.robot_id},
    )
    return observation_schema, action_schema


def _build_episode_event_artifacts(
    *,
    trace: ShadowEpisodeTrace,
    runtime_packet: RuntimePacket,
    episode_tick: Mapping[str, Any],
    window_ticks: Sequence[Mapping[str, Any]],
    regal_summary: Mapping[str, Any],
    datapack_update: ShadowDatapackCreditUpdate,
    artifact_refs: Mapping[str, Any],
    event_sequence_start: int,
    decision_sequence_start: int,
) -> tuple[list[RuntimeEvent], list[DecisionLedgerEntry], int, int]:
    events: list[RuntimeEvent] = []
    decisions: list[DecisionLedgerEntry] = []
    event_sequence_idx = int(event_sequence_start)
    decision_sequence_idx = int(decision_sequence_start)
    receipt_label_refs: list[str] = []
    governance_reasons = [str(reason) for reason in regal_summary.get("reasons", []) or []]
    queue_weight_multiplier = _queue_weight_multiplier(
        regal_summary=regal_summary,
        datapack_update=datapack_update,
    )
    queue_event = RuntimeEvent.from_components(
        run_id=trace.run_id,
        episode_id=trace.episode_id,
        timestamp=trace.ended_at,
        event_kind="queue_reweight",
        sequence_idx=event_sequence_idx,
        scope={"scope_kind": "episode"},
        runtime_packet_id=runtime_packet.packet_id,
        contract_id=runtime_packet.contract.contract_id,
        receipt_label_refs=receipt_label_refs,
        artifact_refs=artifact_refs,
        provenance={
            "advisor": {
                "component": "shadow_control_plane",
                "authority": "replay_queue_advisory",
            }
        },
        metadata={
            "queue_weight_multiplier": queue_weight_multiplier,
            "datapack_recommendation": regal_summary.get("datapack_recommendation", datapack_update.recommendation),
            "deploy_recommendation": regal_summary.get("deploy_recommendation", "allow_shadow"),
        },
    )
    events.append(queue_event)
    event_sequence_idx += 1
    decisions.append(
        DecisionLedgerEntry.from_components(
            run_id=trace.run_id,
            episode_id=trace.episode_id,
            timestamp=trace.ended_at,
            decision_kind="queue_reweight",
            outcome=_queue_reweight_outcome(queue_weight_multiplier),
            sequence_idx=decision_sequence_idx,
            scope={"scope_kind": "episode"},
            reasons=governance_reasons
            or [str(regal_summary.get("datapack_recommendation", datapack_update.recommendation))],
            source_event_ids=[queue_event.event_id],
            runtime_packet_id=runtime_packet.packet_id,
            contract_id=runtime_packet.contract.contract_id,
            receipt_label_refs=receipt_label_refs,
            artifact_refs=artifact_refs,
            provenance=queue_event.provenance,
            metadata={"queue_weight_multiplier": queue_weight_multiplier},
        )
    )
    decision_sequence_idx += 1

    pricing_event_kind = (
        "pricing_tick_suppressed"
        if str(regal_summary.get("pricing_recommendation", "publish")) == "suppress"
        else "pricing_tick_published"
    )
    tick_rows = [dict(episode_tick), *[dict(row) for row in window_ticks]]
    window_index = {
        str(window.window_id): window
        for window in trace.runtime_record.windows
    }
    for tick in tick_rows:
        scope = _event_scope_for_tick(tick=tick, window_index=window_index)
        tick_event = RuntimeEvent.from_components(
            run_id=trace.run_id,
            episode_id=trace.episode_id,
            timestamp=str(tick.get("timestamp", trace.ended_at)),
            event_kind=pricing_event_kind,
            sequence_idx=event_sequence_idx,
            scope=scope,
            runtime_packet_id=runtime_packet.packet_id,
            contract_id=runtime_packet.contract.contract_id,
            receipt_label_refs=receipt_label_refs,
            artifact_refs=artifact_refs,
            provenance={
                "actor": {
                    "component": "pricing_sentinel",
                    "authority": "pricing_policy",
                }
            },
            metadata={
                "tick_id": tick.get("tick_id"),
                "mode": tick.get("mode"),
                "net_customer_rate": float(tick.get("net_customer_rate", 0.0)),
                "confidence": float(tick.get("confidence", 0.0)),
                "pricing_recommendation": regal_summary.get("pricing_recommendation", "publish"),
            },
        )
        events.append(tick_event)
        event_sequence_idx += 1
        decisions.append(
            DecisionLedgerEntry.from_components(
                run_id=trace.run_id,
                episode_id=trace.episode_id,
                timestamp=str(tick.get("timestamp", trace.ended_at)),
                decision_kind=pricing_event_kind,
                outcome=str(regal_summary.get("pricing_recommendation", "publish")),
                sequence_idx=decision_sequence_idx,
                scope=scope,
                reasons=governance_reasons or [str(regal_summary.get("pricing_recommendation", "publish"))],
                source_event_ids=[tick_event.event_id],
                runtime_packet_id=runtime_packet.packet_id,
                contract_id=runtime_packet.contract.contract_id,
                receipt_label_refs=receipt_label_refs,
                artifact_refs=artifact_refs,
                provenance=tick_event.provenance,
                metadata={"tick_id": tick.get("tick_id"), "mode": tick.get("mode")},
            )
        )
        decision_sequence_idx += 1

    regal_event_kind = _regal_event_kind(regal_summary)
    if regal_event_kind is not None:
        regal_event = RuntimeEvent.from_components(
            run_id=trace.run_id,
            episode_id=trace.episode_id,
            timestamp=trace.ended_at,
            event_kind=regal_event_kind,
            sequence_idx=event_sequence_idx,
            scope={"scope_kind": "episode"},
            runtime_packet_id=runtime_packet.packet_id,
            contract_id=runtime_packet.contract.contract_id,
            receipt_label_refs=receipt_label_refs,
            artifact_refs=artifact_refs,
            provenance={
                "critic": {
                    "component": "meta_regal_controller",
                    "authority": "governance_gate",
                }
            },
            metadata={
                "overall_status": regal_summary.get("overall_status", "pass"),
                "deploy_recommendation": regal_summary.get("deploy_recommendation", "allow_shadow"),
                "node_count": len(list(regal_summary.get("node_decisions", []) or [])),
            },
        )
        events.append(regal_event)
        event_sequence_idx += 1
        decisions.append(
            DecisionLedgerEntry.from_components(
                run_id=trace.run_id,
                episode_id=trace.episode_id,
                timestamp=trace.ended_at,
                decision_kind=regal_event_kind,
                outcome=str(regal_summary.get("deploy_recommendation", "allow_shadow")),
                sequence_idx=decision_sequence_idx,
                scope={"scope_kind": "episode"},
                reasons=governance_reasons or [str(regal_summary.get("overall_status", "pass"))],
                source_event_ids=[regal_event.event_id],
                runtime_packet_id=runtime_packet.packet_id,
                contract_id=runtime_packet.contract.contract_id,
                receipt_label_refs=receipt_label_refs,
                artifact_refs=artifact_refs,
                provenance=regal_event.provenance,
                metadata={"pricing_recommendation": regal_summary.get("pricing_recommendation", "publish")},
            )
        )
        decision_sequence_idx += 1

    adaptation_event_kind = _adaptation_event_kind(regal_summary)
    adaptation_event = RuntimeEvent.from_components(
        run_id=trace.run_id,
        episode_id=trace.episode_id,
        timestamp=trace.ended_at,
        event_kind=adaptation_event_kind,
        sequence_idx=event_sequence_idx,
        scope={"scope_kind": "episode"},
        runtime_packet_id=runtime_packet.packet_id,
        contract_id=runtime_packet.contract.contract_id,
        receipt_label_refs=receipt_label_refs,
        artifact_refs=artifact_refs,
        provenance={
            "advisor": {
                "component": "meta_regal_controller",
                "authority": "adaptation_policy",
            }
        },
        metadata={
            "adaptation_recommendation": regal_summary.get("adaptation_recommendation", "no_op"),
            "marginal_frontier_gain": float(datapack_update.marginal_frontier_gain),
        },
    )
    events.append(adaptation_event)
    event_sequence_idx += 1
    decisions.append(
        DecisionLedgerEntry.from_components(
            run_id=trace.run_id,
            episode_id=trace.episode_id,
            timestamp=trace.ended_at,
            decision_kind=adaptation_event_kind,
            outcome=str(regal_summary.get("adaptation_recommendation", "no_op")),
            sequence_idx=decision_sequence_idx,
            scope={"scope_kind": "episode"},
            reasons=governance_reasons or [str(regal_summary.get("adaptation_recommendation", "no_op"))],
            source_event_ids=[adaptation_event.event_id],
            runtime_packet_id=runtime_packet.packet_id,
            contract_id=runtime_packet.contract.contract_id,
            receipt_label_refs=receipt_label_refs,
            artifact_refs=artifact_refs,
            provenance=adaptation_event.provenance,
            metadata={"data_share_credit": float(datapack_update.data_share_credit)},
        )
    )
    decision_sequence_idx += 1

    credit_event = RuntimeEvent.from_components(
        run_id=trace.run_id,
        episode_id=trace.episode_id,
        timestamp=trace.ended_at,
        event_kind="datapack_credit_assigned",
        sequence_idx=event_sequence_idx,
        scope={"scope_kind": "episode"},
        runtime_packet_id=runtime_packet.packet_id,
        contract_id=runtime_packet.contract.contract_id,
        receipt_label_refs=receipt_label_refs,
        artifact_refs=artifact_refs,
        provenance={
            "actor": {
                "component": "value_ledger",
                "authority": "economic_credit",
            }
        },
        metadata={
            "datapack_id": datapack_update.datapack_id,
            "data_share_credit": float(datapack_update.data_share_credit),
            "recommendation": datapack_update.recommendation,
        },
    )
    events.append(credit_event)
    event_sequence_idx += 1
    decisions.append(
        DecisionLedgerEntry.from_components(
            run_id=trace.run_id,
            episode_id=trace.episode_id,
            timestamp=trace.ended_at,
            decision_kind="datapack_credit_assigned",
            outcome=str(datapack_update.recommendation),
            sequence_idx=decision_sequence_idx,
            scope={"scope_kind": "episode"},
            reasons=[str(datapack_update.recommendation)],
            source_event_ids=[credit_event.event_id],
            runtime_packet_id=runtime_packet.packet_id,
            contract_id=runtime_packet.contract.contract_id,
            receipt_label_refs=receipt_label_refs,
            artifact_refs=artifact_refs,
            provenance=credit_event.provenance,
            metadata={
                "datapack_id": datapack_update.datapack_id,
                "data_share_credit": float(datapack_update.data_share_credit),
            },
        )
    )
    decision_sequence_idx += 1

    promotion_event_kind = (
        "promotion_recommend_promote"
        if str(regal_summary.get("deploy_recommendation", "allow_shadow")) in {"recommend_promote", "promote"}
        else "promotion_hold"
    )
    promotion_event = RuntimeEvent.from_components(
        run_id=trace.run_id,
        episode_id=trace.episode_id,
        timestamp=trace.ended_at,
        event_kind=promotion_event_kind,
        sequence_idx=event_sequence_idx,
        scope={"scope_kind": "episode"},
        runtime_packet_id=runtime_packet.packet_id,
        contract_id=runtime_packet.contract.contract_id,
        receipt_label_refs=receipt_label_refs,
        artifact_refs=artifact_refs,
        provenance={
            "advisor": {
                "component": "shadow_control_plane",
                "authority": "promotion_readiness",
            }
        },
        metadata={
            "deploy_recommendation": regal_summary.get("deploy_recommendation", "allow_shadow"),
            "overall_status": regal_summary.get("overall_status", "pass"),
        },
    )
    events.append(promotion_event)
    event_sequence_idx += 1
    decisions.append(
        DecisionLedgerEntry.from_components(
            run_id=trace.run_id,
            episode_id=trace.episode_id,
            timestamp=trace.ended_at,
            decision_kind=promotion_event_kind,
            outcome=str(regal_summary.get("deploy_recommendation", "allow_shadow")),
            sequence_idx=decision_sequence_idx,
            scope={"scope_kind": "episode"},
            reasons=governance_reasons or [str(regal_summary.get("deploy_recommendation", "allow_shadow"))],
            source_event_ids=[promotion_event.event_id],
            runtime_packet_id=runtime_packet.packet_id,
            contract_id=runtime_packet.contract.contract_id,
            receipt_label_refs=receipt_label_refs,
            artifact_refs=artifact_refs,
            provenance=promotion_event.provenance,
            metadata={"shadow_only": True},
        )
    )
    decision_sequence_idx += 1
    return events, decisions, event_sequence_idx, decision_sequence_idx


def _episode_artifact_refs() -> Dict[str, str]:
    return {
        "runtime_packets": "runtime_packets.json",
        "objective_tensor": "objective_tensor.json",
        "objective_compile": "objective_compile.json",
        "constraint_flags": "constraint_flags.json",
        "econ_tensor": "econ_tensor.json",
        "pricing_ticks": "pricing_ticks.jsonl",
        "regal_decisions": "regal_decisions.json",
        "datapack_credit_update": "datapack_credit_update.json",
        "value_ledger": "value_ledger.jsonl",
    }


def _event_scope_for_tick(
    *,
    tick: Mapping[str, Any],
    window_index: Mapping[str, Any],
) -> Dict[str, Any]:
    if str(tick.get("mode", "")) != "step_window":
        return {"scope_kind": "episode"}
    metadata = dict(tick.get("metadata", {}) or {})
    window_id = str(metadata.get("window_id", ""))
    window = window_index.get(window_id)
    if window is None:
        return {"scope_kind": "window", "window_id": window_id}
    return {
        "scope_kind": "window",
        "window_id": window_id,
        "start_step": int(getattr(window, "start_step", 0)),
        "end_step": int(getattr(window, "end_step", 0)),
    }


def _queue_weight_multiplier(
    *,
    regal_summary: Mapping[str, Any],
    datapack_update: ShadowDatapackCreditUpdate,
) -> float:
    deploy_recommendation = str(regal_summary.get("deploy_recommendation", "allow_shadow"))
    datapack_recommendation = str(regal_summary.get("datapack_recommendation", datapack_update.recommendation))
    if deploy_recommendation == "deny_shadow" or datapack_recommendation == "downweight":
        return 0.5
    if deploy_recommendation == "require_review" or datapack_recommendation == "review":
        return 0.75
    if datapack_recommendation == "reward_credit" or float(datapack_update.data_share_credit) > 0.0:
        return 1.25
    return 1.0


def _queue_reweight_outcome(multiplier: float) -> str:
    if multiplier > 1.0:
        return "upweight"
    if multiplier < 1.0:
        return "downweight"
    return "hold"


def _regal_event_kind(regal_summary: Mapping[str, Any]) -> Optional[str]:
    deploy_recommendation = str(regal_summary.get("deploy_recommendation", "allow_shadow"))
    overall_status = str(regal_summary.get("overall_status", "pass"))
    if deploy_recommendation == "deny_shadow":
        return "regal_veto"
    if overall_status != "pass" or deploy_recommendation == "require_review":
        return "regal_warn"
    return None


def _adaptation_event_kind(regal_summary: Mapping[str, Any]) -> str:
    recommendation = str(regal_summary.get("adaptation_recommendation", "no_op"))
    if recommendation == "adapt":
        return "adaptation_admitted"
    if recommendation == "collect_data":
        return "collect_more_data"
    return "adaptation_denied"


def _runtime_contract_id(*, task_id: str, objective_profile_id: str, embodiment_id: str) -> str:
    return "contract.{task}.{profile}.{embodiment}.v1".format(
        task=_contract_fragment(task_id),
        profile=_contract_fragment(objective_profile_id),
        embodiment=_contract_fragment(embodiment_id),
    )


def _contract_fragment(value: str) -> str:
    fragment = "".join(character.lower() if character.isalnum() else "_" for character in str(value))
    return fragment.strip("_") or "unknown"


def _merge_constraint_flags(
    objective_flags: Sequence[Mapping[str, Any]],
    runtime_flags: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    for flag in objective_flags:
        merged.append(
            {
                "constraint_id": f"objective:{flag.get('axis')}:{flag.get('flag')}",
                "severity": "hard",
                "axis": flag.get("axis"),
                "flag": flag.get("flag"),
                "threshold": float(flag.get("threshold", 0.0)),
                "observed": float(flag.get("observed", 0.0)),
            }
        )
    for flag in runtime_flags:
        merged.append(dict(flag))
    return sorted(merged, key=lambda item: str(item.get("constraint_id", "")))


def _default_regal_summary() -> Dict[str, Any]:
    return {
        "overall_status": "pass",
        "deploy_recommendation": "allow_shadow",
        "adaptation_recommendation": "no_op",
        "pricing_recommendation": "publish",
        "datapack_recommendation": "keep",
        "reasons": ["regal_disabled"],
        "node_decisions": [],
    }


def _build_summary(
    *,
    run_id: str,
    objective_profile: ObjectiveContractProfile,
    include_regal: bool,
    traces: Sequence[ShadowEpisodeTrace],
    episode_artifacts: Sequence[ShadowEpisodeArtifacts],
    runtime_events: Sequence[RuntimeEvent],
    decision_entries: Sequence[DecisionLedgerEntry],
    pricing_rows: Sequence[Mapping[str, Any]],
    artifact_paths: Mapping[str, str],
) -> Dict[str, Any]:
    episode_summaries = []
    deploy_counter: Counter[str] = Counter()
    pricing_counter: Counter[str] = Counter()
    datapack_counter: Counter[str] = Counter()
    event_kind_counter: Counter[str] = Counter()
    decision_kind_counter: Counter[str] = Counter()
    total_credit = 0.0
    episode_ticks = [row for row in pricing_rows if row.get("mode") == "episode"]
    for artifact in episode_artifacts:
        regal = artifact.regal_decision
        deploy_counter[str(regal.get("deploy_recommendation", "allow_shadow"))] += 1
        pricing_counter[str(regal.get("pricing_recommendation", "publish"))] += 1
        datapack_counter[str(regal.get("datapack_recommendation", artifact.datapack_credit_update.get("recommendation", "keep")))] += 1
        total_credit += float(artifact.datapack_credit_update.get("data_share_credit", 0.0))
        episode_tick = next((row for row in artifact.pricing_ticks if row.get("mode") == "episode"), {})
        episode_summaries.append(
            {
                "episode_id": artifact.episode_id,
                "reward_total": artifact.baseline_summary.get("reward_total", 0.0),
                "success": artifact.baseline_summary.get("success", False),
                "scalar_reward": artifact.objective_compile.get("scalar_reward", 0.0),
                "net_customer_rate": episode_tick.get("net_customer_rate", 0.0),
                "pricing_confidence": episode_tick.get("confidence", 0.0),
                "deploy_recommendation": regal.get("deploy_recommendation", "allow_shadow"),
                "pricing_recommendation": regal.get("pricing_recommendation", "publish"),
                "datapack_recommendation": regal.get("datapack_recommendation", artifact.datapack_credit_update.get("recommendation", "keep")),
                "data_share_credit": artifact.datapack_credit_update.get("data_share_credit", 0.0),
            }
        )

    success_rate = sum(1 for trace in traces if trace.baseline_summary.get("success")) / max(len(traces), 1)
    mean_reward = sum(float(trace.baseline_summary.get("reward_total", 0.0)) for trace in traces) / max(len(traces), 1)
    mean_net_rate = sum(float(row.get("net_customer_rate", 0.0)) for row in episode_ticks) / max(len(episode_ticks), 1)
    for event in runtime_events:
        event_kind_counter[event.event_kind] += 1
    for decision in decision_entries:
        decision_kind_counter[decision.decision_kind] += 1
    return {
        "run_id": run_id,
        "objective_profile_id": objective_profile.profile_id,
        "objective_profile_hash": objective_profile.profile_hash,
        "include_regal": include_regal,
        "episodes": len(traces),
        "success_rate": success_rate,
        "mean_reward_total": mean_reward,
        "mean_net_customer_rate": mean_net_rate,
        "total_data_share_credit": total_credit,
        "deploy_recommendations": dict(deploy_counter),
        "pricing_recommendations": dict(pricing_counter),
        "datapack_recommendations": dict(datapack_counter),
        "event_count": len(runtime_events),
        "decision_count": len(decision_entries),
        "event_kind_counts": dict(event_kind_counter),
        "decision_kind_counts": dict(decision_kind_counter),
        "determinism": get_context_summary(),
        "artifact_paths": dict(artifact_paths),
        "episode_summaries": episode_summaries,
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow Economic Control Plane",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Objective profile: `{summary['objective_profile_id']}`",
        f"- Episodes: `{summary['episodes']}`",
        f"- Success rate: `{summary['success_rate']:.2%}`",
        f"- Mean net customer rate: `${summary['mean_net_customer_rate']:.2f}/hr`",
        f"- Total data-share credit: `${summary['total_data_share_credit']:.2f}`",
        "",
        "## Episode Summary",
        "",
        "| Episode | Reward | Net Rate | Deploy | Pricing | Datapack | Credit |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for episode in summary.get("episode_summaries", []):
        lines.append(
            "| {episode_id} | {reward_total:.2f} | {net_customer_rate:.2f} | {deploy_recommendation} | {pricing_recommendation} | {datapack_recommendation} | {data_share_credit:.2f} |".format(
                **episode
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for label, path in summary.get("artifact_paths", {}).items():
        lines.append(f"- `{label}`: `{path}`")
    return "\n".join(lines) + "\n"


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(to_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_safe(row), sort_keys=True) + "\n")


def _reset_output_files(paths: Sequence[str | Path]) -> None:
    for path in paths:
        file_path = Path(path)
        if file_path.exists():
            file_path.unlink()
