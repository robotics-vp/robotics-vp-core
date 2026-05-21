"""Phase-5 local prep rows for the Economic WM.

This module deepens local Economic WM ingestion beyond Stage-1 by deriving
mereotopological datapack-composition rows, temporal replay windows, and
counterfactual/value-target joins over canonical lower-WM refs and resource
surfaces. It is scaffold-only local prep: no training, provider bring-up,
promotion, live control, or reward-math mutation is claimed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.lower_wm_consumption import (
    EconomicWMCanonicalConsumptionRow,
    EconomicWMLowerWMConsumptionPreflight,
    load_economic_wm_canonical_consumption_rows,
    load_economic_wm_lower_wm_consumption_preflight,
)
from src.world_model.economic_world_model.resource_surfaces import (
    EconomicWMResourceIngestionManifest,
    EconomicWMResourceReceipt,
    EconomicWMQueueTelemetrySurface,
    load_economic_wm_queue_telemetry_surfaces,
    load_economic_wm_resource_ingestion_manifest,
    load_economic_wm_resource_receipts,
)
from src.world_model.economic_world_model.training_rows import (
    EconomicWMReplayFeatureRow,
    EconomicWMTrainingCorpusManifest,
    load_economic_wm_replay_feature_rows,
    load_economic_wm_training_corpus_manifest,
)

ECONOMIC_WM_DATAPACK_COMPOSITION_ROW_VERSION = "economic_wm_datapack_composition_row_v1"
ECONOMIC_WM_COUNTERFACTUAL_VALUE_JOIN_ROW_VERSION = (
    "economic_wm_counterfactual_value_join_row_v1"
)
ECONOMIC_WM_TEMPORAL_WINDOW_ROW_VERSION = "economic_wm_temporal_window_row_v1"
ECONOMIC_WM_PHASE5_LOCAL_PREP_MANIFEST_VERSION = (
    "economic_wm_phase5_local_prep_manifest_v1"
)

PHASE5_LOCAL_BLOCKERS = (
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


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


@dataclass(frozen=True)
class EconomicWMCounterfactualValueJoinRow:
    """Structural join between counterfactual evals and value target packs."""

    join_row_id: str
    source_row_id: str
    source_episode_id: str
    counterfactual_eval_ref: str
    value_target_pack_ref: str
    value_ledger_ref: str = ""
    join_status: str = "partial_structural_join"
    feature_keys: list[str] = field(default_factory=list)
    target_keys: list[str] = field(default_factory=list)
    authority_class: str = "counterfactual_value_join_only"
    ready_for_trainer_scaffold: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_COUNTERFACTUAL_VALUE_JOIN_ROW_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "join_row_id": self.join_row_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "counterfactual_eval_ref": self.counterfactual_eval_ref,
            "value_target_pack_ref": self.value_target_pack_ref,
            "value_ledger_ref": self.value_ledger_ref,
            "join_status": self.join_status,
            "feature_keys": list(self.feature_keys),
            "target_keys": list(self.target_keys),
            "authority_class": self.authority_class,
            "ready_for_trainer_scaffold": bool(self.ready_for_trainer_scaffold),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMCounterfactualValueJoinRow":
        return cls(
            join_row_id=str(payload.get("join_row_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            counterfactual_eval_ref=str(payload.get("counterfactual_eval_ref", "")),
            value_target_pack_ref=str(payload.get("value_target_pack_ref", "")),
            value_ledger_ref=str(payload.get("value_ledger_ref", "")),
            join_status=str(payload.get("join_status", "partial_structural_join")),
            feature_keys=[
                str(item) for item in list(payload.get("feature_keys", []) or [])
            ],
            target_keys=[
                str(item) for item in list(payload.get("target_keys", []) or [])
            ],
            authority_class=str(
                payload.get("authority_class", "counterfactual_value_join_only")
            ),
            ready_for_trainer_scaffold=bool(
                payload.get("ready_for_trainer_scaffold", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get(
                    "version", ECONOMIC_WM_COUNTERFACTUAL_VALUE_JOIN_ROW_VERSION
                )
            ),
        )


@dataclass(frozen=True)
class EconomicWMDatapackCompositionRow:
    """Mereotopological datapack-composition surface for one Economic WM row."""

    composition_row_id: str
    source_row_id: str
    source_episode_id: str
    material_provenance_composition: Dict[str, float] = field(default_factory=dict)
    functional_contribution_composition: Dict[str, float] = field(default_factory=dict)
    lower_wm_refs: Dict[str, Any] = field(default_factory=dict)
    resource_receipt_ref: str = ""
    queue_telemetry_ref: str = ""
    counterfactual_value_join_ref: str = ""
    feature_vector: Dict[str, float] = field(default_factory=dict)
    target_vector: Dict[str, float] = field(default_factory=dict)
    authority_class: str = "datapack_composition_row_only"
    ready_for_trainer_scaffold: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_DATAPACK_COMPOSITION_ROW_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composition_row_id": self.composition_row_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "material_provenance_composition": _float_dict(
                self.material_provenance_composition
            ),
            "functional_contribution_composition": _float_dict(
                self.functional_contribution_composition
            ),
            "lower_wm_refs": _mapping(self.lower_wm_refs),
            "resource_receipt_ref": self.resource_receipt_ref,
            "queue_telemetry_ref": self.queue_telemetry_ref,
            "counterfactual_value_join_ref": self.counterfactual_value_join_ref,
            "feature_vector": _float_dict(self.feature_vector),
            "target_vector": _float_dict(self.target_vector),
            "authority_class": self.authority_class,
            "ready_for_trainer_scaffold": bool(self.ready_for_trainer_scaffold),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMDatapackCompositionRow":
        return cls(
            composition_row_id=str(payload.get("composition_row_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            material_provenance_composition=_float_dict(
                payload.get("material_provenance_composition", {})
            ),
            functional_contribution_composition=_float_dict(
                payload.get("functional_contribution_composition", {})
            ),
            lower_wm_refs=_mapping(payload.get("lower_wm_refs")),
            resource_receipt_ref=str(payload.get("resource_receipt_ref", "")),
            queue_telemetry_ref=str(payload.get("queue_telemetry_ref", "")),
            counterfactual_value_join_ref=str(
                payload.get("counterfactual_value_join_ref", "")
            ),
            feature_vector=_float_dict(payload.get("feature_vector", {})),
            target_vector=_float_dict(payload.get("target_vector", {})),
            authority_class=str(
                payload.get("authority_class", "datapack_composition_row_only")
            ),
            ready_for_trainer_scaffold=bool(
                payload.get("ready_for_trainer_scaffold", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_DATAPACK_COMPOSITION_ROW_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMTemporalWindowRow:
    """Local temporal replay window over composition rows."""

    window_id: str
    window_index: int
    source_row_ids: list[str] = field(default_factory=list)
    source_episode_ids: list[str] = field(default_factory=list)
    datapack_composition_row_ids: list[str] = field(default_factory=list)
    counterfactual_value_join_row_ids: list[str] = field(default_factory=list)
    resource_receipt_refs: list[str] = field(default_factory=list)
    benchmark_ready_count: int = 0
    shadow_only_count: int = 0
    aggregate_feature_vector: Dict[str, float] = field(default_factory=dict)
    aggregate_target_vector: Dict[str, float] = field(default_factory=dict)
    authority_class: str = "temporal_window_row_only"
    ready_for_trainer_scaffold: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_TEMPORAL_WINDOW_ROW_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "version": self.version,
            "window_index": int(self.window_index),
            "source_row_ids": list(self.source_row_ids),
            "source_episode_ids": list(self.source_episode_ids),
            "datapack_composition_row_ids": list(self.datapack_composition_row_ids),
            "counterfactual_value_join_row_ids": list(
                self.counterfactual_value_join_row_ids
            ),
            "resource_receipt_refs": list(self.resource_receipt_refs),
            "benchmark_ready_count": int(self.benchmark_ready_count),
            "shadow_only_count": int(self.shadow_only_count),
            "aggregate_feature_vector": _float_dict(self.aggregate_feature_vector),
            "aggregate_target_vector": _float_dict(self.aggregate_target_vector),
            "authority_class": self.authority_class,
            "ready_for_trainer_scaffold": bool(self.ready_for_trainer_scaffold),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMTemporalWindowRow":
        return cls(
            window_id=str(payload.get("window_id", "")),
            window_index=int(payload.get("window_index", 0) or 0),
            source_row_ids=[
                str(item) for item in list(payload.get("source_row_ids", []) or [])
            ],
            source_episode_ids=[
                str(item) for item in list(payload.get("source_episode_ids", []) or [])
            ],
            datapack_composition_row_ids=[
                str(item)
                for item in list(payload.get("datapack_composition_row_ids", []) or [])
            ],
            counterfactual_value_join_row_ids=[
                str(item)
                for item in list(
                    payload.get("counterfactual_value_join_row_ids", []) or []
                )
            ],
            resource_receipt_refs=[
                str(item)
                for item in list(payload.get("resource_receipt_refs", []) or [])
            ],
            benchmark_ready_count=int(payload.get("benchmark_ready_count", 0) or 0),
            shadow_only_count=int(payload.get("shadow_only_count", 0) or 0),
            aggregate_feature_vector=_float_dict(
                payload.get("aggregate_feature_vector", {})
            ),
            aggregate_target_vector=_float_dict(
                payload.get("aggregate_target_vector", {})
            ),
            authority_class=str(
                payload.get("authority_class", "temporal_window_row_only")
            ),
            ready_for_trainer_scaffold=bool(
                payload.get("ready_for_trainer_scaffold", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_TEMPORAL_WINDOW_ROW_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMPhase5LocalPrepManifest:
    """Manifest for all local Phase-5 row families."""

    manifest_id: str
    corpus_id: str
    lower_wm_preflight_id: str
    resource_ingestion_manifest_id: str
    row_count: int
    composition_row_count: int
    counterfactual_value_join_count: int
    temporal_window_count: int
    composition_rows_path: str
    counterfactual_value_joins_path: str
    temporal_windows_path: str
    status: str
    authority_class: str = "phase5_local_prep_only"
    ready_for_trainer_scaffold: bool = False
    ready_for_gpu_training: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_PHASE5_LOCAL_PREP_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "corpus_id": self.corpus_id,
            "lower_wm_preflight_id": self.lower_wm_preflight_id,
            "resource_ingestion_manifest_id": self.resource_ingestion_manifest_id,
            "row_count": int(self.row_count),
            "composition_row_count": int(self.composition_row_count),
            "counterfactual_value_join_count": int(
                self.counterfactual_value_join_count
            ),
            "temporal_window_count": int(self.temporal_window_count),
            "composition_rows_path": self.composition_rows_path,
            "counterfactual_value_joins_path": self.counterfactual_value_joins_path,
            "temporal_windows_path": self.temporal_windows_path,
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_trainer_scaffold": bool(self.ready_for_trainer_scaffold),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
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
    ) -> "EconomicWMPhase5LocalPrepManifest":
        return cls(
            manifest_id=str(payload.get("manifest_id", "")),
            corpus_id=str(payload.get("corpus_id", "")),
            lower_wm_preflight_id=str(payload.get("lower_wm_preflight_id", "")),
            resource_ingestion_manifest_id=str(
                payload.get("resource_ingestion_manifest_id", "")
            ),
            row_count=int(payload.get("row_count", 0) or 0),
            composition_row_count=int(payload.get("composition_row_count", 0) or 0),
            counterfactual_value_join_count=int(
                payload.get("counterfactual_value_join_count", 0) or 0
            ),
            temporal_window_count=int(payload.get("temporal_window_count", 0) or 0),
            composition_rows_path=str(payload.get("composition_rows_path", "")),
            counterfactual_value_joins_path=str(
                payload.get("counterfactual_value_joins_path", "")
            ),
            temporal_windows_path=str(payload.get("temporal_windows_path", "")),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "phase5_local_prep_only")
            ),
            ready_for_trainer_scaffold=bool(
                payload.get("ready_for_trainer_scaffold", False)
            ),
            ready_for_gpu_training=bool(payload.get("ready_for_gpu_training", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_PHASE5_LOCAL_PREP_MANIFEST_VERSION)
            ),
        )


def _lower_refs_for_row(
    row: EconomicWMReplayFeatureRow,
    consumption_by_row: Mapping[str, EconomicWMCanonicalConsumptionRow],
) -> Dict[str, Any]:
    consumption = consumption_by_row.get(row.row_id)
    if consumption:
        return {
            ref.wm_key: {
                "artifact_path": ref.artifact_path,
                "state_id": ref.state_id,
                "observed_version": ref.observed_version,
                "reference_status": ref.reference_status,
                "direct_reference": ref.direct_reference,
                "satisfied": ref.satisfied,
            }
            for ref in consumption.canonical_refs
        }
    return _mapping(row.source_refs.get("canonical_lower_wm_refs"))


def _join_row(row: EconomicWMReplayFeatureRow) -> EconomicWMCounterfactualValueJoinRow:
    counterfactual_ref = str(row.source_refs.get("counterfactual_eval_path", ""))
    value_ref = str(row.source_refs.get("value_target_pack_path", ""))
    ledger_ref = str(row.source_refs.get("value_ledger_receipt_path", ""))
    missing = []
    if not counterfactual_ref:
        missing.append("counterfactual_eval_ref_missing")
    if not value_ref:
        missing.append("value_target_pack_ref_missing")
    status = "structural_join_ready" if not missing else "partial_structural_join"
    payload = {
        "source_row_id": row.row_id,
        "counterfactual_eval_ref": counterfactual_ref,
        "value_target_pack_ref": value_ref,
        "value_ledger_ref": ledger_ref,
        "status": status,
    }
    return EconomicWMCounterfactualValueJoinRow(
        join_row_id=f"ewm_cf_value_join_{sha256_json(payload)[:16]}",
        source_row_id=row.row_id,
        source_episode_id=row.source_episode_id,
        counterfactual_eval_ref=counterfactual_ref,
        value_target_pack_ref=value_ref,
        value_ledger_ref=ledger_ref,
        join_status=status,
        feature_keys=sorted(row.feature_vector.keys()),
        target_keys=sorted(row.target_vector.keys()),
        ready_for_trainer_scaffold=not missing,
        blockers=_unique([*missing, *PHASE5_LOCAL_BLOCKERS]),
        source_refs=row.source_refs,
        metadata={"join_scope": "structural counterfactual/value target join"},
    )


def _composition_row(
    row: EconomicWMReplayFeatureRow,
    lower_refs: Mapping[str, Any],
    receipt: Optional[EconomicWMResourceReceipt],
    telemetry: Optional[EconomicWMQueueTelemetrySurface],
    join: EconomicWMCounterfactualValueJoinRow,
) -> EconomicWMDatapackCompositionRow:
    satisfied_lower_ref_count = sum(
        1.0 for ref in lower_refs.values() if bool(dict(ref).get("satisfied", True))
    )
    material = {
        "stage1_replay_row": 1.0,
        "perception_grounding_state": 1.0
        if "perception_grounding" in lower_refs
        else 0.0,
        "sim_synth_physics_state": 1.0 if "sim_synth_physics" in lower_refs else 0.0,
        "embodiment_actuation_state": 1.0
        if "embodiment_actuation" in lower_refs
        else 0.0,
        "resource_budget_receipt": 1.0 if receipt else 0.0,
        "queue_telemetry_surface": 1.0 if telemetry else 0.0,
    }
    functional = {
        "benchmark_evaluator_fixture": 1.0 if row.benchmark_ready else 0.0,
        "shadow_gap_closure_fixture": 1.0 if row.shadow_only else 0.0,
        "counterfactual_value_join": 1.0
        if join.join_status == "structural_join_ready"
        else 0.0,
        "lower_wm_state_conditioning": satisfied_lower_ref_count / 3.0,
        "resource_budget_conditioning": 1.0 if receipt else 0.0,
        "allocation_shadow_feedback": 1.0,
    }
    feature_vector = {
        **row.feature_vector,
        "composition_lower_wm_ref_fraction": functional["lower_wm_state_conditioning"],
        "composition_resource_receipt_present": material["resource_budget_receipt"],
        "composition_counterfactual_value_join_ready": functional[
            "counterfactual_value_join"
        ],
    }
    if receipt:
        feature_vector.update(
            {
                "resource_local_cpu_budget": receipt.capacity_units.get(
                    "local_cpu_budget", 0.0
                ),
                "resource_shadow_planning_budget": receipt.capacity_units.get(
                    "shadow_planning_budget", 0.0
                ),
                "resource_provider_evidence_queue": receipt.queue_depth.get(
                    "provider_evidence_queue", 0.0
                ),
                "resource_gpu_training_budget": receipt.capacity_units.get(
                    "gpu_training_budget", 0.0
                ),
            }
        )
    target_vector = {
        **row.target_vector,
        "target_counterfactual_value_join_weight": functional[
            "counterfactual_value_join"
        ],
        "target_resource_budget_weight": material["resource_budget_receipt"],
    }
    blockers = _unique([*row.denied_promotion_reasons, *PHASE5_LOCAL_BLOCKERS])
    if not receipt:
        blockers.append("resource_receipt_missing")
    payload = {
        "source_row_id": row.row_id,
        "material": material,
        "functional": functional,
        "receipt": receipt.receipt_id if receipt else "",
        "telemetry": telemetry.surface_id if telemetry else "",
        "join": join.join_row_id,
    }
    return EconomicWMDatapackCompositionRow(
        composition_row_id=f"ewm_datapack_composition_{sha256_json(payload)[:16]}",
        source_row_id=row.row_id,
        source_episode_id=row.source_episode_id,
        material_provenance_composition=material,
        functional_contribution_composition=functional,
        lower_wm_refs=_mapping(lower_refs),
        resource_receipt_ref=receipt.receipt_id if receipt else "",
        queue_telemetry_ref=telemetry.surface_id if telemetry else "",
        counterfactual_value_join_ref=join.join_row_id,
        feature_vector=feature_vector,
        target_vector=target_vector,
        ready_for_trainer_scaffold=bool(receipt and lower_refs),
        blockers=blockers,
        source_refs=row.source_refs,
        metadata={"composition_scope": "mereotopological datapack mixture row"},
    )


def _aggregate_vector(
    rows: Iterable[Mapping[str, float]],
) -> Dict[str, float]:
    items = [dict(row) for row in rows]
    keys = sorted({key for row in items for key in row})
    return {key: _mean(row.get(key, 0.0) for row in items) for key in keys}


def _temporal_windows(
    *,
    rows: list[EconomicWMReplayFeatureRow],
    compositions: list[EconomicWMDatapackCompositionRow],
    joins: list[EconomicWMCounterfactualValueJoinRow],
    window_size: int,
) -> list[EconomicWMTemporalWindowRow]:
    by_row = {row.row_id: row for row in rows}
    joins_by_row = {join.source_row_id: join for join in joins}
    ordered = sorted(
        compositions, key=lambda item: (item.source_episode_id, item.source_row_id)
    )
    size = max(1, int(window_size))
    windows: list[EconomicWMTemporalWindowRow] = []
    for index, start in enumerate(range(0, len(ordered), size)):
        chunk = ordered[start : start + size]
        source_rows = [by_row[item.source_row_id] for item in chunk]
        join_chunk = [joins_by_row[item.source_row_id] for item in chunk]
        payload = {
            "index": index,
            "composition_ids": [item.composition_row_id for item in chunk],
            "source_row_ids": [item.source_row_id for item in chunk],
        }
        windows.append(
            EconomicWMTemporalWindowRow(
                window_id=f"ewm_temporal_window_{sha256_json(payload)[:16]}",
                window_index=index,
                source_row_ids=[item.source_row_id for item in chunk],
                source_episode_ids=[item.source_episode_id for item in chunk],
                datapack_composition_row_ids=[
                    item.composition_row_id for item in chunk
                ],
                counterfactual_value_join_row_ids=[
                    item.join_row_id for item in join_chunk
                ],
                resource_receipt_refs=[
                    item.resource_receipt_ref
                    for item in chunk
                    if item.resource_receipt_ref
                ],
                benchmark_ready_count=sum(
                    1 for row in source_rows if row.benchmark_ready
                ),
                shadow_only_count=sum(1 for row in source_rows if row.shadow_only),
                aggregate_feature_vector=_aggregate_vector(
                    item.feature_vector for item in chunk
                ),
                aggregate_target_vector=_aggregate_vector(
                    item.target_vector for item in chunk
                ),
                ready_for_trainer_scaffold=all(
                    item.ready_for_trainer_scaffold for item in chunk
                ),
                blockers=_unique(
                    blocker for item in chunk for blocker in item.blockers
                ),
                metadata={
                    "window_family": "lexical_replay_window",
                    "window_size": size,
                },
            )
        )
    return windows


def build_economic_wm_phase5_local_prep(
    *,
    corpus_manifest: EconomicWMTrainingCorpusManifest,
    rows: Iterable[EconomicWMReplayFeatureRow],
    lower_wm_preflight: EconomicWMLowerWMConsumptionPreflight,
    canonical_consumption_rows: Iterable[EconomicWMCanonicalConsumptionRow],
    resource_manifest: EconomicWMResourceIngestionManifest,
    resource_receipts: Iterable[EconomicWMResourceReceipt],
    queue_telemetry_surfaces: Iterable[EconomicWMQueueTelemetrySurface],
    composition_rows_path: str | Path,
    counterfactual_value_joins_path: str | Path,
    temporal_windows_path: str | Path,
    window_size: int = 2,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[
    EconomicWMPhase5LocalPrepManifest,
    list[EconomicWMDatapackCompositionRow],
    list[EconomicWMCounterfactualValueJoinRow],
    list[EconomicWMTemporalWindowRow],
]:
    row_items = list(rows)
    consumption_by_row = {
        item.source_row_id: item for item in canonical_consumption_rows
    }
    receipt_by_row = {item.source_row_id: item for item in resource_receipts}
    telemetry_by_row = {item.source_row_id: item for item in queue_telemetry_surfaces}

    joins: list[EconomicWMCounterfactualValueJoinRow] = []
    compositions: list[EconomicWMDatapackCompositionRow] = []
    for row in row_items:
        lower_refs = _lower_refs_for_row(row, consumption_by_row)
        join = _join_row(row)
        composition = _composition_row(
            row,
            lower_refs,
            receipt_by_row.get(row.row_id),
            telemetry_by_row.get(row.row_id),
            join,
        )
        joins.append(join)
        compositions.append(composition)
    windows = _temporal_windows(
        rows=row_items,
        compositions=compositions,
        joins=joins,
        window_size=window_size,
    )
    ready_for_trainer_scaffold = bool(
        compositions
        and windows
        and lower_wm_preflight.ready_for_neural_manifest
        and resource_manifest.ready_for_phase5_local_prep
        and all(item.ready_for_trainer_scaffold for item in compositions)
    )
    status = "ok" if ready_for_trainer_scaffold else "blocked"
    payload = {
        "corpus_id": corpus_manifest.corpus_id,
        "lower_wm_preflight_id": lower_wm_preflight.preflight_id,
        "resource_ingestion_manifest_id": resource_manifest.manifest_id,
        "composition_ids": [item.composition_row_id for item in compositions],
        "join_ids": [item.join_row_id for item in joins],
        "window_ids": [item.window_id for item in windows],
    }
    aggregate_counts = {
        "row_count": float(len(row_items)),
        "composition_row_count": float(len(compositions)),
        "counterfactual_value_join_count": float(len(joins)),
        "temporal_window_count": float(len(windows)),
        "benchmark_ready_count": float(corpus_manifest.benchmark_ready_count),
        "shadow_only_count": float(corpus_manifest.shadow_only_count),
        "resource_receipt_count": float(resource_manifest.receipt_count),
        "canonical_lower_wm_direct_reference_count": float(
            lower_wm_preflight.direct_reference_count
        ),
        "canonical_lower_wm_compiled_reference_count": float(
            lower_wm_preflight.compiled_reference_count
        ),
        "structural_join_ready_count": float(
            sum(1 for item in joins if item.ready_for_trainer_scaffold)
        ),
    }
    manifest = EconomicWMPhase5LocalPrepManifest(
        manifest_id=f"ewm_phase5_local_prep_{sha256_json(payload)[:16]}",
        corpus_id=corpus_manifest.corpus_id,
        lower_wm_preflight_id=lower_wm_preflight.preflight_id,
        resource_ingestion_manifest_id=resource_manifest.manifest_id,
        row_count=len(row_items),
        composition_row_count=len(compositions),
        counterfactual_value_join_count=len(joins),
        temporal_window_count=len(windows),
        composition_rows_path=str(composition_rows_path),
        counterfactual_value_joins_path=str(counterfactual_value_joins_path),
        temporal_windows_path=str(temporal_windows_path),
        status=status,
        ready_for_trainer_scaffold=ready_for_trainer_scaffold,
        blockers=list(PHASE5_LOCAL_BLOCKERS),
        aggregate_counts=aggregate_counts,
        artifact_refs={
            **_mapping(artifact_refs),
            "composition_rows_path": str(composition_rows_path),
            "counterfactual_value_joins_path": str(counterfactual_value_joins_path),
            "temporal_windows_path": str(temporal_windows_path),
        },
        metadata={
            **_mapping(metadata),
            "boundary": "Phase-5 local prep only; no training or promotion",
            "window_size": int(window_size),
        },
    )
    return manifest, compositions, joins, windows


def save_economic_wm_phase5_local_prep(
    *,
    manifest_path: str | Path,
    manifest: EconomicWMPhase5LocalPrepManifest,
    composition_rows: Iterable[EconomicWMDatapackCompositionRow],
    counterfactual_value_joins: Iterable[EconomicWMCounterfactualValueJoinRow],
    temporal_windows: Iterable[EconomicWMTemporalWindowRow],
) -> None:
    _write_json(manifest_path, manifest.to_dict())
    _write_jsonl(
        manifest.composition_rows_path, [item.to_dict() for item in composition_rows]
    )
    _write_jsonl(
        manifest.counterfactual_value_joins_path,
        [item.to_dict() for item in counterfactual_value_joins],
    )
    _write_jsonl(
        manifest.temporal_windows_path, [item.to_dict() for item in temporal_windows]
    )


def load_economic_wm_phase5_local_prep_manifest(
    path: str | Path,
) -> EconomicWMPhase5LocalPrepManifest:
    return EconomicWMPhase5LocalPrepManifest.from_dict(_load_json(path))


def load_economic_wm_datapack_composition_rows(
    path: str | Path,
) -> list[EconomicWMDatapackCompositionRow]:
    return [
        EconomicWMDatapackCompositionRow.from_dict(row) for row in _load_jsonl(path)
    ]


def load_economic_wm_counterfactual_value_join_rows(
    path: str | Path,
) -> list[EconomicWMCounterfactualValueJoinRow]:
    return [
        EconomicWMCounterfactualValueJoinRow.from_dict(row) for row in _load_jsonl(path)
    ]


def load_economic_wm_temporal_window_rows(
    path: str | Path,
) -> list[EconomicWMTemporalWindowRow]:
    return [EconomicWMTemporalWindowRow.from_dict(row) for row in _load_jsonl(path)]


def build_economic_wm_phase5_local_prep_from_paths(
    *,
    corpus_manifest_path: str | Path,
    rows_path: str | Path,
    lower_wm_preflight_path: str | Path,
    canonical_consumption_rows_path: str | Path,
    resource_manifest_path: str | Path,
    resource_receipts_path: str | Path,
    queue_telemetry_surfaces_path: str | Path,
    manifest_path: str | Path,
    composition_rows_path: str | Path,
    counterfactual_value_joins_path: str | Path,
    temporal_windows_path: str | Path,
    window_size: int = 2,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMPhase5LocalPrepManifest:
    corpus_manifest = load_economic_wm_training_corpus_manifest(corpus_manifest_path)
    rows = load_economic_wm_replay_feature_rows(rows_path)
    lower_wm_preflight = load_economic_wm_lower_wm_consumption_preflight(
        lower_wm_preflight_path
    )
    canonical_rows = load_economic_wm_canonical_consumption_rows(
        canonical_consumption_rows_path
    )
    resource_manifest = load_economic_wm_resource_ingestion_manifest(
        resource_manifest_path
    )
    resource_receipts = load_economic_wm_resource_receipts(resource_receipts_path)
    telemetry_surfaces = load_economic_wm_queue_telemetry_surfaces(
        queue_telemetry_surfaces_path
    )
    manifest, compositions, joins, windows = build_economic_wm_phase5_local_prep(
        corpus_manifest=corpus_manifest,
        rows=rows,
        lower_wm_preflight=lower_wm_preflight,
        canonical_consumption_rows=canonical_rows,
        resource_manifest=resource_manifest,
        resource_receipts=resource_receipts,
        queue_telemetry_surfaces=telemetry_surfaces,
        composition_rows_path=composition_rows_path,
        counterfactual_value_joins_path=counterfactual_value_joins_path,
        temporal_windows_path=temporal_windows_path,
        window_size=window_size,
        artifact_refs={
            "corpus_manifest_path": str(corpus_manifest_path),
            "rows_path": str(rows_path),
            "lower_wm_preflight_path": str(lower_wm_preflight_path),
            "canonical_consumption_rows_path": str(canonical_consumption_rows_path),
            "resource_manifest_path": str(resource_manifest_path),
            "resource_receipts_path": str(resource_receipts_path),
            "queue_telemetry_surfaces_path": str(queue_telemetry_surfaces_path),
            "manifest_path": str(manifest_path),
        },
        metadata=metadata,
    )
    save_economic_wm_phase5_local_prep(
        manifest_path=manifest_path,
        manifest=manifest,
        composition_rows=compositions,
        counterfactual_value_joins=joins,
        temporal_windows=windows,
    )
    return manifest
