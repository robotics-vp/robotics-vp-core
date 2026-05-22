"""Economic WM supervision substrate materialization.

Phase-5 rows already carry refs to counterfactual evals, value target packs,
and value-ledger receipts. This module proves those refs can be loaded as typed
supervision records rather than treated as summary strings. It remains local
prep only: no model training, provider bring-up, promotion, or reward-math
mutation is claimed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.economics.counterfactual_eval import CounterfactualEval
from src.economics.value_targets import ValueTargetPack
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.phase5_local_prep import (
    EconomicWMCounterfactualValueJoinRow,
    EconomicWMPhase5LocalPrepManifest,
    load_economic_wm_counterfactual_value_join_rows,
    load_economic_wm_phase5_local_prep_manifest,
)

ECONOMIC_WM_SUPERVISION_RECORD_VERSION = "economic_wm_supervision_record_v1"
ECONOMIC_WM_SUPERVISION_MANIFEST_VERSION = "economic_wm_supervision_manifest_v1"

SUPERVISION_BLOCKERS = (
    "gpu_training_not_run",
    "provider_bringup_not_run",
    "promotion_grade_outcome_evidence_missing",
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


def _safe_load(path: str) -> Dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        return _load_json(candidate)
    except Exception:
        return None


@dataclass(frozen=True)
class EconomicWMSupervisionRecord:
    """Typed supervision record over one counterfactual/value join row."""

    supervision_record_id: str
    source_row_id: str
    source_episode_id: str
    join_row_id: str
    counterfactual_eval_id: str = ""
    value_target_pack_id: str = ""
    value_ledger_receipt_id: str = ""
    recommended_action: str = "noop"
    candidate_count: int = 0
    value_target_count: int = 0
    target_kind_counts: Dict[str, float] = field(default_factory=dict)
    target_value_summary: Dict[str, float] = field(default_factory=dict)
    counterfactual_delta_summary: Dict[str, float] = field(default_factory=dict)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    authority_class: str = "supervision_substrate_only"
    ready_for_shadow_outcome_loop: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SUPERVISION_RECORD_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supervision_record_id": self.supervision_record_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "join_row_id": self.join_row_id,
            "counterfactual_eval_id": self.counterfactual_eval_id,
            "value_target_pack_id": self.value_target_pack_id,
            "value_ledger_receipt_id": self.value_ledger_receipt_id,
            "recommended_action": self.recommended_action,
            "candidate_count": int(self.candidate_count),
            "value_target_count": int(self.value_target_count),
            "target_kind_counts": _float_dict(self.target_kind_counts),
            "target_value_summary": _float_dict(self.target_value_summary),
            "counterfactual_delta_summary": _float_dict(
                self.counterfactual_delta_summary
            ),
            "source_refs": _mapping(self.source_refs),
            "authority_class": self.authority_class,
            "ready_for_shadow_outcome_loop": bool(self.ready_for_shadow_outcome_loop),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMSupervisionRecord":
        return cls(
            supervision_record_id=str(payload.get("supervision_record_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            join_row_id=str(payload.get("join_row_id", "")),
            counterfactual_eval_id=str(payload.get("counterfactual_eval_id", "")),
            value_target_pack_id=str(payload.get("value_target_pack_id", "")),
            value_ledger_receipt_id=str(payload.get("value_ledger_receipt_id", "")),
            recommended_action=str(payload.get("recommended_action", "noop")),
            candidate_count=int(payload.get("candidate_count", 0) or 0),
            value_target_count=int(payload.get("value_target_count", 0) or 0),
            target_kind_counts=_float_dict(payload.get("target_kind_counts", {})),
            target_value_summary=_float_dict(payload.get("target_value_summary", {})),
            counterfactual_delta_summary=_float_dict(
                payload.get("counterfactual_delta_summary", {})
            ),
            source_refs=_mapping(payload.get("source_refs")),
            authority_class=str(
                payload.get("authority_class", "supervision_substrate_only")
            ),
            ready_for_shadow_outcome_loop=bool(
                payload.get("ready_for_shadow_outcome_loop", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ECONOMIC_WM_SUPERVISION_RECORD_VERSION)),
        )


@dataclass(frozen=True)
class EconomicWMSupervisionManifest:
    """Manifest proving concrete counterfactual/value supervision is loadable."""

    manifest_id: str
    phase5_manifest_id: str
    record_count: int
    ready_record_count: int
    counterfactual_eval_count: int
    value_target_pack_count: int
    value_ledger_receipt_count: int
    records_path: str
    status: str
    authority_class: str = "supervision_substrate_manifest_only"
    ready_for_shadow_outcome_loop: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SUPERVISION_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "phase5_manifest_id": self.phase5_manifest_id,
            "record_count": int(self.record_count),
            "ready_record_count": int(self.ready_record_count),
            "counterfactual_eval_count": int(self.counterfactual_eval_count),
            "value_target_pack_count": int(self.value_target_pack_count),
            "value_ledger_receipt_count": int(self.value_ledger_receipt_count),
            "records_path": self.records_path,
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_shadow_outcome_loop": bool(self.ready_for_shadow_outcome_loop),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMSupervisionManifest":
        return cls(
            manifest_id=str(payload.get("manifest_id", "")),
            phase5_manifest_id=str(payload.get("phase5_manifest_id", "")),
            record_count=int(payload.get("record_count", 0) or 0),
            ready_record_count=int(payload.get("ready_record_count", 0) or 0),
            counterfactual_eval_count=int(
                payload.get("counterfactual_eval_count", 0) or 0
            ),
            value_target_pack_count=int(payload.get("value_target_pack_count", 0) or 0),
            value_ledger_receipt_count=int(
                payload.get("value_ledger_receipt_count", 0) or 0
            ),
            records_path=str(payload.get("records_path", "")),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "supervision_substrate_manifest_only")
            ),
            ready_for_shadow_outcome_loop=bool(
                payload.get("ready_for_shadow_outcome_loop", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_SUPERVISION_MANIFEST_VERSION)
            ),
        )


def _counterfactual_summary(
    eval_payload: Mapping[str, Any],
) -> tuple[str, int, Dict[str, float]]:
    eval_obj = CounterfactualEval.from_dict(eval_payload)
    deltas = [
        float(candidate.deltas.get("delta_value_vs_noop", 0.0))
        for candidate in eval_obj.candidates
    ]
    return (
        eval_obj.recommended_action,
        len(eval_obj.candidates),
        {
            "max_delta_value_vs_noop": max(deltas) if deltas else 0.0,
            "min_delta_value_vs_noop": min(deltas) if deltas else 0.0,
            "mean_delta_value_vs_noop": sum(deltas) / len(deltas) if deltas else 0.0,
        },
    )


def _value_target_summary(
    pack_payload: Mapping[str, Any],
) -> tuple[int, Dict[str, float], Dict[str, float]]:
    pack = ValueTargetPack.from_dict(pack_payload)
    counts: Dict[str, float] = {}
    totals: Dict[str, float] = {}
    confidence_totals: Dict[str, float] = {}
    for target in pack.targets:
        kind = str(target.target_kind)
        counts[kind] = counts.get(kind, 0.0) + 1.0
        totals[kind] = totals.get(kind, 0.0) + float(target.target_value)
        confidence_totals[kind] = confidence_totals.get(kind, 0.0) + float(
            target.confidence
        )
    summary: Dict[str, float] = {}
    for kind, total in totals.items():
        summary[f"{kind}_target_value_total"] = total
        summary[f"{kind}_confidence_mean"] = confidence_totals[kind] / max(
            1.0, counts[kind]
        )
    summary["target_value_total"] = sum(totals.values())
    return len(pack.targets), counts, summary


def _value_ledger_id(payload: Mapping[str, Any] | None, path: str) -> str:
    if not payload:
        return ""
    return str(
        payload.get("receipt_id")
        or payload.get("ledger_id")
        or payload.get("value_ledger_receipt_id")
        or Path(path).stem
    )


def _record_from_join(
    join: EconomicWMCounterfactualValueJoinRow,
) -> EconomicWMSupervisionRecord:
    counterfactual_payload = _safe_load(join.counterfactual_eval_ref)
    value_pack_payload = _safe_load(join.value_target_pack_ref)
    value_ledger_payload = _safe_load(join.value_ledger_ref)
    missing: list[str] = []
    if counterfactual_payload is None:
        missing.append("counterfactual_eval_missing_or_unreadable")
    if value_pack_payload is None:
        missing.append("value_target_pack_missing_or_unreadable")
    if value_ledger_payload is None:
        missing.append("value_ledger_receipt_missing_or_unreadable")

    recommended_action = "noop"
    candidate_count = 0
    delta_summary: Dict[str, float] = {}
    if counterfactual_payload is not None:
        recommended_action, candidate_count, delta_summary = _counterfactual_summary(
            counterfactual_payload
        )

    target_count = 0
    target_kind_counts: Dict[str, float] = {}
    target_value_summary: Dict[str, float] = {}
    if value_pack_payload is not None:
        target_count, target_kind_counts, target_value_summary = _value_target_summary(
            value_pack_payload
        )

    counterfactual_eval_id = ""
    if counterfactual_payload is not None:
        counterfactual_eval_id = str(counterfactual_payload.get("eval_id", ""))
    value_target_pack_id = ""
    if value_pack_payload is not None:
        value_target_pack_id = str(value_pack_payload.get("pack_id", ""))
    value_ledger_receipt_id = _value_ledger_id(
        value_ledger_payload, join.value_ledger_ref
    )
    ready = not missing and candidate_count > 0 and target_count > 0
    blockers = list(SUPERVISION_BLOCKERS)
    blockers.extend(missing)
    payload = {
        "source_row_id": join.source_row_id,
        "join_row_id": join.join_row_id,
        "counterfactual_eval_id": counterfactual_eval_id,
        "value_target_pack_id": value_target_pack_id,
        "value_ledger_receipt_id": value_ledger_receipt_id,
        "candidate_count": candidate_count,
        "target_count": target_count,
    }
    return EconomicWMSupervisionRecord(
        supervision_record_id=f"ewm_supervision_{sha256_json(payload)[:16]}",
        source_row_id=join.source_row_id,
        source_episode_id=join.source_episode_id,
        join_row_id=join.join_row_id,
        counterfactual_eval_id=counterfactual_eval_id,
        value_target_pack_id=value_target_pack_id,
        value_ledger_receipt_id=value_ledger_receipt_id,
        recommended_action=recommended_action,
        candidate_count=candidate_count,
        value_target_count=target_count,
        target_kind_counts=target_kind_counts,
        target_value_summary=target_value_summary,
        counterfactual_delta_summary=delta_summary,
        source_refs={
            "counterfactual_eval_ref": join.counterfactual_eval_ref,
            "value_target_pack_ref": join.value_target_pack_ref,
            "value_ledger_ref": join.value_ledger_ref,
        },
        ready_for_shadow_outcome_loop=ready,
        blockers=sorted(set(blockers)),
        metadata={
            "boundary": "typed supervision substrate only",
            "join_status": join.join_status,
        },
    )


def build_economic_wm_supervision_substrate(
    *,
    phase5_manifest: EconomicWMPhase5LocalPrepManifest,
    join_rows: Iterable[EconomicWMCounterfactualValueJoinRow],
    records_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[EconomicWMSupervisionManifest, list[EconomicWMSupervisionRecord]]:
    records = [_record_from_join(join) for join in join_rows]
    ready_count = sum(1 for record in records if record.ready_for_shadow_outcome_loop)
    counterfactual_count = sum(1 for record in records if record.counterfactual_eval_id)
    value_target_count = sum(1 for record in records if record.value_target_pack_id)
    ledger_count = sum(1 for record in records if record.value_ledger_receipt_id)
    ready = bool(records) and ready_count == len(records)
    status = "ok" if ready else "blocked"
    payload = {
        "phase5_manifest_id": phase5_manifest.manifest_id,
        "record_ids": [record.supervision_record_id for record in records],
        "ready_count": ready_count,
        "counterfactual_count": counterfactual_count,
        "value_target_count": value_target_count,
        "ledger_count": ledger_count,
    }
    aggregate_counts = {
        "record_count": float(len(records)),
        "ready_record_count": float(ready_count),
        "counterfactual_eval_count": float(counterfactual_count),
        "value_target_pack_count": float(value_target_count),
        "value_ledger_receipt_count": float(ledger_count),
        "candidate_count": float(sum(record.candidate_count for record in records)),
        "value_target_count": float(
            sum(record.value_target_count for record in records)
        ),
    }
    return (
        EconomicWMSupervisionManifest(
            manifest_id=f"ewm_supervision_manifest_{sha256_json(payload)[:16]}",
            phase5_manifest_id=phase5_manifest.manifest_id,
            record_count=len(records),
            ready_record_count=ready_count,
            counterfactual_eval_count=counterfactual_count,
            value_target_pack_count=value_target_count,
            value_ledger_receipt_count=ledger_count,
            records_path=str(records_path),
            status=status,
            ready_for_shadow_outcome_loop=ready,
            blockers=list(SUPERVISION_BLOCKERS),
            aggregate_counts=aggregate_counts,
            artifact_refs={
                **_mapping(artifact_refs),
                "records_path": str(records_path),
            },
            metadata={
                **_mapping(metadata),
                "boundary": "local supervision substrate only",
            },
        ),
        records,
    )


def save_economic_wm_supervision_substrate(
    *,
    manifest_path: str | Path,
    manifest: EconomicWMSupervisionManifest,
    records: Iterable[EconomicWMSupervisionRecord],
) -> None:
    _write_json(manifest_path, manifest.to_dict())
    _write_jsonl(manifest.records_path, [record.to_dict() for record in records])


def load_economic_wm_supervision_manifest(
    path: str | Path,
) -> EconomicWMSupervisionManifest:
    return EconomicWMSupervisionManifest.from_dict(_load_json(path))


def load_economic_wm_supervision_records(
    path: str | Path,
) -> list[EconomicWMSupervisionRecord]:
    return [EconomicWMSupervisionRecord.from_dict(row) for row in _load_jsonl(path)]


def build_economic_wm_supervision_substrate_from_paths(
    *,
    phase5_prep_path: str | Path,
    manifest_path: str | Path,
    records_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMSupervisionManifest:
    phase5_manifest = load_economic_wm_phase5_local_prep_manifest(phase5_prep_path)
    join_rows = load_economic_wm_counterfactual_value_join_rows(
        phase5_manifest.counterfactual_value_joins_path
    )
    manifest, records = build_economic_wm_supervision_substrate(
        phase5_manifest=phase5_manifest,
        join_rows=join_rows,
        records_path=records_path,
        artifact_refs={
            "phase5_prep_path": str(phase5_prep_path),
            "counterfactual_value_joins_path": phase5_manifest.counterfactual_value_joins_path,
            "manifest_path": str(manifest_path),
        },
        metadata=metadata,
    )
    save_economic_wm_supervision_substrate(
        manifest_path=manifest_path,
        manifest=manifest,
        records=records,
    )
    return manifest
