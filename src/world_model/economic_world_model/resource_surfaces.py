"""Phase-5 local resource and compute surfaces for the Economic WM.

These artifacts define receipt schemas and ingestion slots for capacity,
latency, thermal, battery, companion-compute, degraded-mode, and queueing
telemetry. They are local scaffold artifacts only: no GPU training, provider
bring-up, live control, promotion, or reward-math mutation is claimed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.training_rows import (
    EconomicWMReplayFeatureRow,
    EconomicWMTrainingCorpusManifest,
    load_economic_wm_replay_feature_rows,
    load_economic_wm_training_corpus_manifest,
)

ECONOMIC_WM_RESOURCE_RECEIPT_VERSION = "economic_wm_resource_receipt_v1"
ECONOMIC_WM_COMPANION_COMPUTE_CONTRACT_VERSION = (
    "economic_wm_companion_compute_contract_v1"
)
ECONOMIC_WM_DEGRADED_MODE_RUNBOOK_VERSION = "economic_wm_degraded_mode_runbook_v1"
ECONOMIC_WM_QUEUE_TELEMETRY_SURFACE_VERSION = "economic_wm_queue_telemetry_surface_v1"
ECONOMIC_WM_RESOURCE_INGESTION_MANIFEST_VERSION = (
    "economic_wm_resource_ingestion_manifest_v1"
)

ECONOMIC_WM_INGESTION_SLOTS = (
    "capacity_receipts",
    "latency_receipts",
    "thermal_receipts",
    "battery_receipts",
    "companion_compute_contracts",
    "degraded_mode_runbooks",
    "queue_telemetry_surfaces",
)

ALLOCATABLE_BUDGET_OBJECTS = (
    "inference_spend",
    "routing_spend",
    "simulation_spend",
    "diffusion_spend",
    "data_collection_spend",
    "conservation_reserve",
    "inferential_work_order_budget",
)

LOCAL_PHASE5_BLOCKERS = (
    "gpu_training_not_run",
    "provider_bringup_not_run",
    "promotion_grade_shadow_benchmarks_missing",
    "non_stub_teacher_runtime_not_verified",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_dict(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class EconomicWMResourceReceipt:
    """Capacity/latency/thermal/battery receipt schema for one row."""

    receipt_id: str
    source_row_id: str
    source_episode_id: str
    receipt_kind: str
    capacity_units: Dict[str, float] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    thermal_headroom: Dict[str, float] = field(default_factory=dict)
    battery_reserve: Dict[str, float] = field(default_factory=dict)
    queue_depth: Dict[str, float] = field(default_factory=dict)
    telemetry_quality: Dict[str, float] = field(default_factory=dict)
    placement_class: str = "local_shadow_only"
    allocatable_budget_objects: list[str] = field(
        default_factory=lambda: list(ALLOCATABLE_BUDGET_OBJECTS)
    )
    authority_class: str = "resource_receipt_schema_only"
    ready_for_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_RESOURCE_RECEIPT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "receipt_kind": self.receipt_kind,
            "capacity_units": _float_dict(self.capacity_units),
            "latency_ms": _float_dict(self.latency_ms),
            "thermal_headroom": _float_dict(self.thermal_headroom),
            "battery_reserve": _float_dict(self.battery_reserve),
            "queue_depth": _float_dict(self.queue_depth),
            "telemetry_quality": _float_dict(self.telemetry_quality),
            "placement_class": self.placement_class,
            "allocatable_budget_objects": list(self.allocatable_budget_objects),
            "authority_class": self.authority_class,
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMResourceReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            receipt_kind=str(payload.get("receipt_kind", "resource_envelope")),
            capacity_units=_float_dict(payload.get("capacity_units", {})),
            latency_ms=_float_dict(payload.get("latency_ms", {})),
            thermal_headroom=_float_dict(payload.get("thermal_headroom", {})),
            battery_reserve=_float_dict(payload.get("battery_reserve", {})),
            queue_depth=_float_dict(payload.get("queue_depth", {})),
            telemetry_quality=_float_dict(payload.get("telemetry_quality", {})),
            placement_class=str(payload.get("placement_class", "local_shadow_only")),
            allocatable_budget_objects=[
                str(item)
                for item in list(payload.get("allocatable_budget_objects", []) or [])
            ],
            authority_class=str(
                payload.get("authority_class", "resource_receipt_schema_only")
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ECONOMIC_WM_RESOURCE_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class EconomicWMCompanionComputeContract:
    """Contract for future companion-compute execution without live authority."""

    contract_id: str
    source_row_id: str
    source_episode_id: str
    compute_plane: str
    control_split: Dict[str, str] = field(default_factory=dict)
    planner_rate_hz: float = 2.0
    servo_rate_hz: float = 0.0
    max_planner_latency_ms: float = 100.0
    communication_plane: str = "local_loopback_contract"
    degraded_mode_allowed: bool = True
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_COMPANION_COMPUTE_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "compute_plane": self.compute_plane,
            "control_split": {str(k): str(v) for k, v in self.control_split.items()},
            "planner_rate_hz": float(self.planner_rate_hz),
            "servo_rate_hz": float(self.servo_rate_hz),
            "max_planner_latency_ms": float(self.max_planner_latency_ms),
            "communication_plane": self.communication_plane,
            "degraded_mode_allowed": bool(self.degraded_mode_allowed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMCompanionComputeContract":
        return cls(
            contract_id=str(payload.get("contract_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            compute_plane=str(payload.get("compute_plane", "local_cpu_contract")),
            control_split={
                str(k): str(v)
                for k, v in dict(payload.get("control_split", {})).items()
            },
            planner_rate_hz=float(payload.get("planner_rate_hz", 2.0)),
            servo_rate_hz=float(payload.get("servo_rate_hz", 0.0)),
            max_planner_latency_ms=float(payload.get("max_planner_latency_ms", 100.0)),
            communication_plane=str(
                payload.get("communication_plane", "local_loopback_contract")
            ),
            degraded_mode_allowed=bool(payload.get("degraded_mode_allowed", True)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_COMPANION_COMPUTE_CONTRACT_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMDegradedModeRunbook:
    """Runbook for resource-degraded Economic WM shadow operation."""

    runbook_id: str
    source_row_id: str
    source_episode_id: str
    trigger_conditions: list[str] = field(default_factory=list)
    allowed_modes: list[str] = field(default_factory=list)
    denied_modes: list[str] = field(default_factory=list)
    recovery_actions: list[str] = field(default_factory=list)
    authority_class: str = "degraded_mode_runbook_only"
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_DEGRADED_MODE_RUNBOOK_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runbook_id": self.runbook_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "trigger_conditions": list(self.trigger_conditions),
            "allowed_modes": list(self.allowed_modes),
            "denied_modes": list(self.denied_modes),
            "recovery_actions": list(self.recovery_actions),
            "authority_class": self.authority_class,
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMDegradedModeRunbook":
        return cls(
            runbook_id=str(payload.get("runbook_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            trigger_conditions=[
                str(item) for item in list(payload.get("trigger_conditions", []) or [])
            ],
            allowed_modes=[
                str(item) for item in list(payload.get("allowed_modes", []) or [])
            ],
            denied_modes=[
                str(item) for item in list(payload.get("denied_modes", []) or [])
            ],
            recovery_actions=[
                str(item) for item in list(payload.get("recovery_actions", []) or [])
            ],
            authority_class=str(
                payload.get("authority_class", "degraded_mode_runbook_only")
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_DEGRADED_MODE_RUNBOOK_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMQueueTelemetrySurface:
    """Queue and telemetry surface for local shadow allocation harnesses."""

    surface_id: str
    source_row_id: str
    source_episode_id: str
    queue_depth: Dict[str, float] = field(default_factory=dict)
    work_order_backlog: Dict[str, float] = field(default_factory=dict)
    budget_pressure: Dict[str, float] = field(default_factory=dict)
    telemetry_quality: Dict[str, float] = field(default_factory=dict)
    ready_for_shadow_execution: bool = True
    authority_class: str = "queue_telemetry_surface_only"
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_QUEUE_TELEMETRY_SURFACE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "queue_depth": _float_dict(self.queue_depth),
            "work_order_backlog": _float_dict(self.work_order_backlog),
            "budget_pressure": _float_dict(self.budget_pressure),
            "telemetry_quality": _float_dict(self.telemetry_quality),
            "ready_for_shadow_execution": bool(self.ready_for_shadow_execution),
            "authority_class": self.authority_class,
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMQueueTelemetrySurface":
        return cls(
            surface_id=str(payload.get("surface_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            queue_depth=_float_dict(payload.get("queue_depth", {})),
            work_order_backlog=_float_dict(payload.get("work_order_backlog", {})),
            budget_pressure=_float_dict(payload.get("budget_pressure", {})),
            telemetry_quality=_float_dict(payload.get("telemetry_quality", {})),
            ready_for_shadow_execution=bool(
                payload.get("ready_for_shadow_execution", True)
            ),
            authority_class=str(
                payload.get("authority_class", "queue_telemetry_surface_only")
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_QUEUE_TELEMETRY_SURFACE_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMResourceIngestionManifest:
    """Manifest proving Phase-5 resource receipt slots exist locally."""

    manifest_id: str
    corpus_id: str
    row_count: int
    receipt_count: int
    contract_count: int
    runbook_count: int
    telemetry_surface_count: int
    receipts_path: str
    contracts_path: str
    degraded_runbooks_path: str
    telemetry_surfaces_path: str
    status: str
    economic_wm_ingestion_slots: list[str] = field(
        default_factory=lambda: list(ECONOMIC_WM_INGESTION_SLOTS)
    )
    allocatable_budget_objects: list[str] = field(
        default_factory=lambda: list(ALLOCATABLE_BUDGET_OBJECTS)
    )
    resource_receipts_defined: bool = False
    companion_compute_contracts_defined: bool = False
    degraded_mode_runbooks_defined: bool = False
    queue_telemetry_surfaces_defined: bool = False
    ready_for_phase5_local_prep: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_RESOURCE_INGESTION_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "corpus_id": self.corpus_id,
            "row_count": int(self.row_count),
            "receipt_count": int(self.receipt_count),
            "contract_count": int(self.contract_count),
            "runbook_count": int(self.runbook_count),
            "telemetry_surface_count": int(self.telemetry_surface_count),
            "receipts_path": self.receipts_path,
            "contracts_path": self.contracts_path,
            "degraded_runbooks_path": self.degraded_runbooks_path,
            "telemetry_surfaces_path": self.telemetry_surfaces_path,
            "status": self.status,
            "economic_wm_ingestion_slots": list(self.economic_wm_ingestion_slots),
            "allocatable_budget_objects": list(self.allocatable_budget_objects),
            "resource_receipts_defined": bool(self.resource_receipts_defined),
            "companion_compute_contracts_defined": bool(
                self.companion_compute_contracts_defined
            ),
            "degraded_mode_runbooks_defined": bool(self.degraded_mode_runbooks_defined),
            "queue_telemetry_surfaces_defined": bool(
                self.queue_telemetry_surfaces_defined
            ),
            "ready_for_phase5_local_prep": bool(self.ready_for_phase5_local_prep),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMResourceIngestionManifest":
        return cls(
            manifest_id=str(payload.get("manifest_id", "")),
            corpus_id=str(payload.get("corpus_id", "")),
            row_count=int(payload.get("row_count", 0) or 0),
            receipt_count=int(payload.get("receipt_count", 0) or 0),
            contract_count=int(payload.get("contract_count", 0) or 0),
            runbook_count=int(payload.get("runbook_count", 0) or 0),
            telemetry_surface_count=int(payload.get("telemetry_surface_count", 0) or 0),
            receipts_path=str(payload.get("receipts_path", "")),
            contracts_path=str(payload.get("contracts_path", "")),
            degraded_runbooks_path=str(payload.get("degraded_runbooks_path", "")),
            telemetry_surfaces_path=str(payload.get("telemetry_surfaces_path", "")),
            status=str(payload.get("status", "blocked")),
            economic_wm_ingestion_slots=[
                str(item)
                for item in list(payload.get("economic_wm_ingestion_slots", []) or [])
            ],
            allocatable_budget_objects=[
                str(item)
                for item in list(payload.get("allocatable_budget_objects", []) or [])
            ],
            resource_receipts_defined=bool(
                payload.get("resource_receipts_defined", False)
            ),
            companion_compute_contracts_defined=bool(
                payload.get("companion_compute_contracts_defined", False)
            ),
            degraded_mode_runbooks_defined=bool(
                payload.get("degraded_mode_runbooks_defined", False)
            ),
            queue_telemetry_surfaces_defined=bool(
                payload.get("queue_telemetry_surfaces_defined", False)
            ),
            ready_for_phase5_local_prep=bool(
                payload.get("ready_for_phase5_local_prep", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_RESOURCE_INGESTION_MANIFEST_VERSION)
            ),
        )


def _build_resource_receipt(
    row: EconomicWMReplayFeatureRow,
) -> EconomicWMResourceReceipt:
    provider_gap = float(row.target_vector.get("provider_bringup_gap_weight", 1.0))
    gpu_gap = float(row.target_vector.get("gpu_training_deferred_weight", 1.0))
    shadow_gap = float(row.target_vector.get("shadow_gap_weight", 0.0))
    benchmark_weight = float(row.target_vector.get("benchmark_training_weight", 0.0))
    local_capacity = 1.0 if row.local_materialization_eligible else 0.5
    placement_class = (
        "local_benchmark_fixture" if row.benchmark_ready else "local_shadow_gap_fixture"
    )
    capacity_units = {
        "local_cpu_budget": local_capacity,
        "companion_compute_budget": 0.25 + 0.25 * benchmark_weight,
        "gpu_training_budget": 0.0,
        "provider_runtime_budget": 0.0,
        "shadow_planning_budget": 1.0,
    }
    latency_ms = {
        "canonical_state_ingest_ms": 5.0 + 10.0 * shadow_gap,
        "shadow_planner_cycle_ms": 40.0 + 40.0 * provider_gap,
        "companion_round_trip_ms": 25.0 + 25.0 * shadow_gap,
        "gpu_training_latency_ms": 0.0,
    }
    thermal_headroom = {
        "local_cpu_headroom": 0.75,
        "companion_compute_headroom": 0.55 + 0.1 * benchmark_weight,
        "gpu_training_headroom": 0.0,
    }
    battery_reserve = {
        "minimum_reserve_fraction": 0.25,
        "shadow_execution_reserve_fraction": 0.5 + 0.1 * benchmark_weight,
        "provider_bringup_reserve_fraction": 0.0,
    }
    queue_depth = {
        "shadow_gap_queue": shadow_gap,
        "provider_evidence_queue": provider_gap,
        "gpu_training_queue": gpu_gap,
        "benchmark_fixture_queue": benchmark_weight,
    }
    telemetry_quality = {
        "replay_row_contract": 1.0,
        "lower_wm_reference_presence": 1.0
        if row.source_refs.get("canonical_lower_wm_refs")
        else 0.0,
        "teacher_runtime_truth": 0.0 if provider_gap > 0.0 else 1.0,
        "gpu_runtime_truth": 0.0,
    }
    blockers = _unique([*row.denied_promotion_reasons, *LOCAL_PHASE5_BLOCKERS])
    payload = {
        "source_row_id": row.row_id,
        "source_episode_id": row.source_episode_id,
        "receipt_kind": "capacity_latency_thermal_battery_queue",
        "capacity_units": capacity_units,
        "latency_ms": latency_ms,
        "thermal_headroom": thermal_headroom,
        "battery_reserve": battery_reserve,
        "queue_depth": queue_depth,
        "telemetry_quality": telemetry_quality,
        "placement_class": placement_class,
        "blockers": blockers,
        "source_refs": row.source_refs,
    }
    return EconomicWMResourceReceipt(
        receipt_id=f"ewm_resource_receipt_{sha256_json(payload)[:16]}",
        source_row_id=row.row_id,
        source_episode_id=row.source_episode_id,
        receipt_kind=str(payload["receipt_kind"]),
        capacity_units=_float_dict(capacity_units),
        latency_ms=_float_dict(latency_ms),
        thermal_headroom=_float_dict(thermal_headroom),
        battery_reserve=_float_dict(battery_reserve),
        queue_depth=_float_dict(queue_depth),
        telemetry_quality=_float_dict(telemetry_quality),
        placement_class=placement_class,
        blockers=list(blockers),
        source_refs=_mapping(row.source_refs),
        metadata={
            "boundary": "local Phase-5 resource schema only",
            "gpu_training_budget_is_zero_because": "gpu/provider execution not run",
        },
    )


def _build_companion_contract(
    row: EconomicWMReplayFeatureRow,
    receipt: EconomicWMResourceReceipt,
) -> EconomicWMCompanionComputeContract:
    payload = {
        "source_row_id": row.row_id,
        "source_episode_id": row.source_episode_id,
        "receipt_id": receipt.receipt_id,
        "compute_plane": "local_cpu_with_future_companion_slot",
    }
    return EconomicWMCompanionComputeContract(
        contract_id=f"ewm_companion_compute_{sha256_json(payload)[:16]}",
        source_row_id=row.row_id,
        source_episode_id=row.source_episode_id,
        compute_plane="local_cpu_with_future_companion_slot",
        control_split={
            "economic_wm": "advisory_shadow_work_order_planner",
            "lower_wms": "canonical_state_and_receipt_owners",
            "policy_runtime": "no_live_policy_control",
            "reward_runtime": "frozen_no_mutation",
        },
        planner_rate_hz=2.0 if row.benchmark_ready else 1.0,
        servo_rate_hz=0.0,
        max_planner_latency_ms=max(receipt.latency_ms.values() or [100.0]),
        communication_plane="typed_receipt_queue_contract",
        degraded_mode_allowed=True,
        blockers=list(receipt.blockers),
        source_refs={"resource_receipt_id": receipt.receipt_id, **row.source_refs},
        metadata={
            "companion_compute_contract_scope": "shape and authority contract only",
        },
    )


def _build_degraded_runbook(
    row: EconomicWMReplayFeatureRow,
    receipt: EconomicWMResourceReceipt,
    contract: EconomicWMCompanionComputeContract,
) -> EconomicWMDegradedModeRunbook:
    payload = {
        "source_row_id": row.row_id,
        "receipt_id": receipt.receipt_id,
        "contract_id": contract.contract_id,
    }
    return EconomicWMDegradedModeRunbook(
        runbook_id=f"ewm_degraded_mode_{sha256_json(payload)[:16]}",
        source_row_id=row.row_id,
        source_episode_id=row.source_episode_id,
        trigger_conditions=[
            "battery_reserve_below_minimum",
            "thermal_headroom_below_contract",
            "queue_backlog_above_shadow_budget",
            "provider_runtime_unavailable",
            "gpu_capacity_unavailable",
            "lower_wm_receipt_stale_or_missing",
        ],
        allowed_modes=[
            "shadow_work_order_only",
            "conserve_compute",
            "defer_provider_request",
            "queue_for_later_gpu_training",
            "request_lower_wm_receipt_refresh",
        ],
        denied_modes=[
            "live_policy_control",
            "reward_math_mutation",
            "promotion_without_benchmark_evidence",
            "gpu_training_without_runtime_receipt",
            "provider_truth_substitution",
        ],
        recovery_actions=[
            "emit_denied_promotion_gate",
            "refresh_canonical_lower_wm_receipts",
            "rerun_shadow_allocation_eval",
            "record_queue_telemetry_delta",
        ],
        blockers=list(receipt.blockers),
        source_refs={
            "resource_receipt_id": receipt.receipt_id,
            "companion_compute_contract_id": contract.contract_id,
            **row.source_refs,
        },
        metadata={"runbook_scope": "local degraded-mode planning contract"},
    )


def _build_queue_surface(
    row: EconomicWMReplayFeatureRow,
    receipt: EconomicWMResourceReceipt,
) -> EconomicWMQueueTelemetrySurface:
    payload = {"source_row_id": row.row_id, "receipt_id": receipt.receipt_id}
    return EconomicWMQueueTelemetrySurface(
        surface_id=f"ewm_queue_telemetry_{sha256_json(payload)[:16]}",
        source_row_id=row.row_id,
        source_episode_id=row.source_episode_id,
        queue_depth={
            "shadow_gap_queue": receipt.queue_depth.get("shadow_gap_queue", 0.0),
            "provider_evidence_queue": receipt.queue_depth.get(
                "provider_evidence_queue", 0.0
            ),
            "gpu_training_queue": receipt.queue_depth.get("gpu_training_queue", 0.0),
            "lower_wm_refresh_queue": 0.0
            if receipt.telemetry_quality.get("lower_wm_reference_presence", 0.0) > 0.0
            else 1.0,
        },
        work_order_backlog={
            "curate_replay_fixture": 1.0 if row.benchmark_ready else 0.0,
            "close_shadow_gap": 1.0 if row.shadow_only else 0.0,
            "prepare_provider_contract": 1.0,
            "train_gpu_model": 0.0,
        },
        budget_pressure={
            "inference_spend_pressure": 0.25,
            "routing_spend_pressure": 0.25 + 0.25 * float(row.shadow_only),
            "simulation_spend_pressure": 0.5,
            "data_collection_spend_pressure": 0.5 + 0.25 * float(row.shadow_only),
            "conservation_reserve_pressure": 0.25,
        },
        telemetry_quality=receipt.telemetry_quality,
        blockers=list(receipt.blockers),
        source_refs={"resource_receipt_id": receipt.receipt_id, **row.source_refs},
        metadata={"telemetry_scope": "shadow queue surface only"},
    )


def build_economic_wm_resource_surfaces(
    *,
    corpus_manifest: EconomicWMTrainingCorpusManifest,
    rows: Iterable[EconomicWMReplayFeatureRow],
    receipts_path: str | Path,
    contracts_path: str | Path,
    degraded_runbooks_path: str | Path,
    telemetry_surfaces_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[
    EconomicWMResourceIngestionManifest,
    list[EconomicWMResourceReceipt],
    list[EconomicWMCompanionComputeContract],
    list[EconomicWMDegradedModeRunbook],
    list[EconomicWMQueueTelemetrySurface],
]:
    """Build local Phase-5 resource surfaces from replay feature rows."""

    row_items = list(rows)
    receipts: list[EconomicWMResourceReceipt] = []
    contracts: list[EconomicWMCompanionComputeContract] = []
    runbooks: list[EconomicWMDegradedModeRunbook] = []
    telemetry_surfaces: list[EconomicWMQueueTelemetrySurface] = []
    for row in row_items:
        receipt = _build_resource_receipt(row)
        contract = _build_companion_contract(row, receipt)
        runbook = _build_degraded_runbook(row, receipt, contract)
        telemetry = _build_queue_surface(row, receipt)
        receipts.append(receipt)
        contracts.append(contract)
        runbooks.append(runbook)
        telemetry_surfaces.append(telemetry)

    row_count = len(row_items)
    status = "ok" if row_count and len(receipts) == row_count else "blocked"
    ready_for_phase5_local_prep = status == "ok"
    payload = {
        "corpus_id": corpus_manifest.corpus_id,
        "row_count": row_count,
        "receipt_count": len(receipts),
        "contract_count": len(contracts),
        "runbook_count": len(runbooks),
        "telemetry_surface_count": len(telemetry_surfaces),
        "receipts_path": str(receipts_path),
        "contracts_path": str(contracts_path),
        "degraded_runbooks_path": str(degraded_runbooks_path),
        "telemetry_surfaces_path": str(telemetry_surfaces_path),
        "status": status,
        "slots": list(ECONOMIC_WM_INGESTION_SLOTS),
        "budget_objects": list(ALLOCATABLE_BUDGET_OBJECTS),
    }
    aggregate_counts = {
        "row_count": float(row_count),
        "receipt_count": float(len(receipts)),
        "contract_count": float(len(contracts)),
        "runbook_count": float(len(runbooks)),
        "telemetry_surface_count": float(len(telemetry_surfaces)),
        "ingestion_slot_count": float(len(ECONOMIC_WM_INGESTION_SLOTS)),
        "allocatable_budget_object_count": float(len(ALLOCATABLE_BUDGET_OBJECTS)),
        "gpu_training_budget_total": sum(
            item.capacity_units.get("gpu_training_budget", 0.0) for item in receipts
        ),
        "live_policy_control_count": float(
            sum(1 for item in contracts if item.live_policy_control)
        ),
    }
    manifest = EconomicWMResourceIngestionManifest(
        manifest_id=f"ewm_resource_ingestion_{sha256_json(payload)[:16]}",
        corpus_id=corpus_manifest.corpus_id,
        row_count=row_count,
        receipt_count=len(receipts),
        contract_count=len(contracts),
        runbook_count=len(runbooks),
        telemetry_surface_count=len(telemetry_surfaces),
        receipts_path=str(receipts_path),
        contracts_path=str(contracts_path),
        degraded_runbooks_path=str(degraded_runbooks_path),
        telemetry_surfaces_path=str(telemetry_surfaces_path),
        status=status,
        resource_receipts_defined=bool(receipts),
        companion_compute_contracts_defined=bool(contracts),
        degraded_mode_runbooks_defined=bool(runbooks),
        queue_telemetry_surfaces_defined=bool(telemetry_surfaces),
        ready_for_phase5_local_prep=ready_for_phase5_local_prep,
        blockers=list(LOCAL_PHASE5_BLOCKERS),
        aggregate_counts=aggregate_counts,
        artifact_refs={
            **_mapping(artifact_refs),
            "receipts_path": str(receipts_path),
            "contracts_path": str(contracts_path),
            "degraded_runbooks_path": str(degraded_runbooks_path),
            "telemetry_surfaces_path": str(telemetry_surfaces_path),
        },
        metadata={
            **_mapping(metadata),
            "boundary": "Phase-5 local resource ingestion slots only",
        },
    )
    return manifest, receipts, contracts, runbooks, telemetry_surfaces


def save_economic_wm_resource_surfaces(
    *,
    manifest_path: str | Path,
    manifest: EconomicWMResourceIngestionManifest,
    receipts: Iterable[EconomicWMResourceReceipt],
    contracts: Iterable[EconomicWMCompanionComputeContract],
    runbooks: Iterable[EconomicWMDegradedModeRunbook],
    telemetry_surfaces: Iterable[EconomicWMQueueTelemetrySurface],
) -> None:
    """Persist the resource-surface manifest and row families."""

    payload = manifest.to_dict()
    _write_json(manifest_path, payload)
    _write_jsonl(manifest.receipts_path, [item.to_dict() for item in receipts])
    _write_jsonl(manifest.contracts_path, [item.to_dict() for item in contracts])
    _write_jsonl(manifest.degraded_runbooks_path, [item.to_dict() for item in runbooks])
    _write_jsonl(
        manifest.telemetry_surfaces_path,
        [item.to_dict() for item in telemetry_surfaces],
    )


def load_economic_wm_resource_ingestion_manifest(
    path: str | Path,
) -> EconomicWMResourceIngestionManifest:
    return EconomicWMResourceIngestionManifest.from_dict(_load_json(path))


def load_economic_wm_resource_receipts(
    path: str | Path,
) -> list[EconomicWMResourceReceipt]:
    return [EconomicWMResourceReceipt.from_dict(row) for row in _load_jsonl(path)]


def load_economic_wm_companion_compute_contracts(
    path: str | Path,
) -> list[EconomicWMCompanionComputeContract]:
    return [
        EconomicWMCompanionComputeContract.from_dict(row) for row in _load_jsonl(path)
    ]


def load_economic_wm_degraded_mode_runbooks(
    path: str | Path,
) -> list[EconomicWMDegradedModeRunbook]:
    return [EconomicWMDegradedModeRunbook.from_dict(row) for row in _load_jsonl(path)]


def load_economic_wm_queue_telemetry_surfaces(
    path: str | Path,
) -> list[EconomicWMQueueTelemetrySurface]:
    return [EconomicWMQueueTelemetrySurface.from_dict(row) for row in _load_jsonl(path)]


def build_economic_wm_resource_surfaces_from_paths(
    *,
    corpus_manifest_path: str | Path,
    rows_path: str | Path,
    manifest_path: str | Path,
    receipts_path: str | Path,
    contracts_path: str | Path,
    degraded_runbooks_path: str | Path,
    telemetry_surfaces_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMResourceIngestionManifest:
    corpus_manifest = load_economic_wm_training_corpus_manifest(corpus_manifest_path)
    rows = load_economic_wm_replay_feature_rows(rows_path)
    manifest, receipts, contracts, runbooks, telemetry_surfaces = (
        build_economic_wm_resource_surfaces(
            corpus_manifest=corpus_manifest,
            rows=rows,
            receipts_path=receipts_path,
            contracts_path=contracts_path,
            degraded_runbooks_path=degraded_runbooks_path,
            telemetry_surfaces_path=telemetry_surfaces_path,
            artifact_refs={
                "corpus_manifest_path": str(corpus_manifest_path),
                "rows_path": str(rows_path),
                "manifest_path": str(manifest_path),
            },
            metadata=metadata,
        )
    )
    save_economic_wm_resource_surfaces(
        manifest_path=manifest_path,
        manifest=manifest,
        receipts=receipts,
        contracts=contracts,
        runbooks=runbooks,
        telemetry_surfaces=telemetry_surfaces,
    )
    return manifest
