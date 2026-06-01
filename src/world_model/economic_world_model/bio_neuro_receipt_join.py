"""Local bio/neuro receipt joins for Economic WM consumption surfaces.

The join produced here makes bio/neuro substrate receipts queryable beside the
normal lower-WM Economic WM receipt surfaces. It is structural evidence only:
no provider execution, hardware proof, training, weight writes, reward-math
mutation, or promotion authority is granted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

BIO_NEURO_RECEIPT_JOIN_ROW_VERSION = "bio_neuro_receipt_join_row_v1"
BIO_NEURO_RECEIPT_JOIN_REPORT_VERSION = "bio_neuro_receipt_join_report_v1"

BIO_NEURO_MISSING_EXTERNAL_PROOF = (
    "provider_execution",
    "gpu_training",
    "unitree_sim_runtime",
    "unitree_hardware_runtime",
    "promotion_grade_benchmarks",
)

BIO_NEURO_DENIED_AUTHORITIES = (
    "training_executed",
    "weights_written",
    "provider_executed",
    "hardware_executed",
    "live_policy_control",
    "reward_math_mutation",
    "promotion_eligible",
    "phase7_abstraction_expansion",
)

_VERSION_TO_OWNER = {
    "embodiment_bio_neuro_substrate_receipt_v1": "embodiment_actuation",
    "self_motion_expectation_v1": "embodiment_actuation",
    "active_sensing_proposal_v1": "embodiment_actuation",
    "synergy_codebook_entry_v1": "embodiment_actuation",
    "interoceptive_state_v1": "embodiment_actuation",
    "self_disturbance_receipt_v1": "perception_grounding",
    "active_sensing_receipt_v1": "perception_grounding",
    "regime_broadcast_v1": "economic_world_model",
    "regime_acknowledgment_receipt_v1": "lower_wm_acknowledgment",
    "anomaly_suspicion_receipt_v1": "regal_anomaly_governance",
    "governance_escalation_event_v1": "regal_anomaly_governance",
}

_VERSION_TO_CONSUMPTION_SLOTS = {
    "embodiment_bio_neuro_substrate_receipt_v1": (
        "bio_neuro_substrate_materialization",
    ),
    "self_motion_expectation_v1": (
        "efference_copy",
        "self_disturbance_comparison",
    ),
    "active_sensing_proposal_v1": (
        "active_sensing_value_of_information",
    ),
    "synergy_codebook_entry_v1": (
        "motor_synergy_prior",
        "body_control_placeholder",
    ),
    "interoceptive_state_v1": (
        "interoceptive_resource_state",
        "economic_resource_context",
    ),
    "self_disturbance_receipt_v1": (
        "efference_copy",
        "self_disturbance_comparison",
        "perception_grounding_receipt",
    ),
    "active_sensing_receipt_v1": (
        "active_sensing_value_of_information",
        "value_of_information_outcome",
        "perception_grounding_receipt",
    ),
    "regime_broadcast_v1": (
        "regime_broadcast_conditioning",
        "economic_posture_signal",
    ),
    "regime_acknowledgment_receipt_v1": (
        "regime_broadcast_acknowledgment",
    ),
    "anomaly_suspicion_receipt_v1": (
        "anomaly_governance",
        "suspicion_signal",
    ),
    "governance_escalation_event_v1": (
        "anomaly_governance",
        "operator_review_signal",
    ),
}


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Iterable[Any]) -> list[str]:
    return [str(value) for value in values if str(value)]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_mapping(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(_mapping(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


def _receipt_identity(receipt: Mapping[str, Any]) -> str:
    for key in (
        "receipt_id",
        "expectation_id",
        "proposal_id",
        "synergy_id",
        "interoceptive_state_id",
        "broadcast_id",
        "event_id",
    ):
        value = str(receipt.get(key, ""))
        if value:
            return value
    return _stable_id(
        "bio_neuro_receipt",
        {
            "version": str(receipt.get("version", "")),
            "payload": _mapping(receipt),
        },
    )


def _owner_wm(receipt: Mapping[str, Any]) -> str:
    version = str(receipt.get("version", ""))
    if version == "regime_acknowledgment_receipt_v1":
        wm_id = str(receipt.get("wm_id", ""))
        return wm_id or _VERSION_TO_OWNER[version]
    if version == "anomaly_suspicion_receipt_v1":
        domain = str(receipt.get("domain", ""))
        return domain or _VERSION_TO_OWNER[version]
    return _VERSION_TO_OWNER.get(version, "unknown_bio_neuro_surface")


def _consumption_slots(receipt: Mapping[str, Any]) -> list[str]:
    return list(_VERSION_TO_CONSUMPTION_SLOTS.get(str(receipt.get("version", "")), ()))


def _receipt_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "receipt_id": _receipt_identity(receipt),
        "version": str(receipt.get("version", "")),
        "authority_level": str(
            receipt.get("authority_level", receipt.get("authority_class", "none"))
        ),
        "authority_class": str(receipt.get("authority_class", "receipt_only")),
        "promotion_eligible": bool(receipt.get("promotion_eligible", False)),
        "provider_or_hardware_proof": bool(
            receipt.get("provider_or_hardware_proof", False)
        ),
        "trained_model_proof": bool(receipt.get("trained_model_proof", False)),
    }


@dataclass(frozen=True)
class BioNeuroReceiptJoinRow:
    """A local bio/neuro receipt mapped into Economic WM consumption slots."""

    row_id: str
    receipt_id: str
    receipt_version: str
    owner_wm: str
    economic_consumption_slots: list[str] = field(default_factory=list)
    receipt_payload: dict[str, Any] = field(default_factory=dict)
    receipt_summary: dict[str, Any] = field(default_factory=dict)
    join_status: str = "joined_local_bio_neuro_receipt"
    authority_class: str = "bio_neuro_receipt_join_only"
    promotion_eligible: bool = False
    provider_or_hardware_proof: bool = False
    trained_model_proof: bool = False
    ready_for_training: bool = False
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = BIO_NEURO_RECEIPT_JOIN_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "version": self.version,
            "receipt_id": self.receipt_id,
            "receipt_version": self.receipt_version,
            "owner_wm": self.owner_wm,
            "economic_consumption_slots": list(self.economic_consumption_slots),
            "receipt_payload": _mapping(self.receipt_payload),
            "receipt_summary": _mapping(self.receipt_summary),
            "join_status": self.join_status,
            "authority_class": self.authority_class,
            "promotion_eligible": bool(self.promotion_eligible),
            "provider_or_hardware_proof": bool(self.provider_or_hardware_proof),
            "trained_model_proof": bool(self.trained_model_proof),
            "ready_for_training": bool(self.ready_for_training),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BioNeuroReceiptJoinRow":
        return cls(
            row_id=str(payload.get("row_id", "")),
            receipt_id=str(payload.get("receipt_id", "")),
            receipt_version=str(payload.get("receipt_version", "")),
            owner_wm=str(payload.get("owner_wm", "")),
            economic_consumption_slots=_strings(
                list(payload.get("economic_consumption_slots", []) or [])
            ),
            receipt_payload=_mapping(payload.get("receipt_payload")),
            receipt_summary=_mapping(payload.get("receipt_summary")),
            join_status=str(
                payload.get("join_status", "joined_local_bio_neuro_receipt")
            ),
            authority_class=str(
                payload.get("authority_class", "bio_neuro_receipt_join_only")
            ),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            provider_or_hardware_proof=bool(
                payload.get("provider_or_hardware_proof", False)
            ),
            trained_model_proof=bool(payload.get("trained_model_proof", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", BIO_NEURO_RECEIPT_JOIN_ROW_VERSION)),
        )


@dataclass(frozen=True)
class BioNeuroReceiptJoinReport:
    """Aggregate report for local bio/neuro receipt joins."""

    join_id: str
    status: str
    row_count: int
    join_rows_path: str
    source_receipts_path: str = ""
    source_report_path: str = ""
    receipt_ids: list[str] = field(default_factory=list)
    surface_versions: list[str] = field(default_factory=list)
    owner_wm_counts: dict[str, int] = field(default_factory=dict)
    economic_consumption_slots: list[str] = field(default_factory=list)
    authority_class: str = "bio_neuro_receipt_join_only"
    promotion_eligible: bool = False
    provider_or_hardware_proof: bool = False
    trained_model_proof: bool = False
    ready_for_training: bool = False
    lower_wm_truth_redefined: bool = False
    reward_math_mutation: bool = False
    phase7_abstraction_expanded: bool = False
    denied_authorities: list[str] = field(
        default_factory=lambda: list(BIO_NEURO_DENIED_AUTHORITIES)
    )
    missing_external_proof: list[str] = field(
        default_factory=lambda: list(BIO_NEURO_MISSING_EXTERNAL_PROOF)
    )
    blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = BIO_NEURO_RECEIPT_JOIN_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "join_id": self.join_id,
            "version": self.version,
            "status": self.status,
            "row_count": int(self.row_count),
            "join_rows_path": self.join_rows_path,
            "source_receipts_path": self.source_receipts_path,
            "source_report_path": self.source_report_path,
            "receipt_ids": list(self.receipt_ids),
            "surface_versions": list(self.surface_versions),
            "owner_wm_counts": {
                str(key): int(value) for key, value in self.owner_wm_counts.items()
            },
            "economic_consumption_slots": list(self.economic_consumption_slots),
            "authority_class": self.authority_class,
            "promotion_eligible": bool(self.promotion_eligible),
            "provider_or_hardware_proof": bool(self.provider_or_hardware_proof),
            "trained_model_proof": bool(self.trained_model_proof),
            "ready_for_training": bool(self.ready_for_training),
            "lower_wm_truth_redefined": bool(self.lower_wm_truth_redefined),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "phase7_abstraction_expanded": bool(self.phase7_abstraction_expanded),
            "denied_authorities": list(self.denied_authorities),
            "missing_external_proof": list(self.missing_external_proof),
            "blockers": list(self.blockers),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BioNeuroReceiptJoinReport":
        return cls(
            join_id=str(payload.get("join_id", "")),
            status=str(payload.get("status", "failed")),
            row_count=int(payload.get("row_count", 0) or 0),
            join_rows_path=str(payload.get("join_rows_path", "")),
            source_receipts_path=str(payload.get("source_receipts_path", "")),
            source_report_path=str(payload.get("source_report_path", "")),
            receipt_ids=_strings(list(payload.get("receipt_ids", []) or [])),
            surface_versions=_strings(list(payload.get("surface_versions", []) or [])),
            owner_wm_counts={
                str(key): int(value)
                for key, value in dict(payload.get("owner_wm_counts", {}) or {}).items()
            },
            economic_consumption_slots=_strings(
                list(payload.get("economic_consumption_slots", []) or [])
            ),
            authority_class=str(
                payload.get("authority_class", "bio_neuro_receipt_join_only")
            ),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            provider_or_hardware_proof=bool(
                payload.get("provider_or_hardware_proof", False)
            ),
            trained_model_proof=bool(payload.get("trained_model_proof", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            lower_wm_truth_redefined=bool(
                payload.get("lower_wm_truth_redefined", False)
            ),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            phase7_abstraction_expanded=bool(
                payload.get("phase7_abstraction_expanded", False)
            ),
            denied_authorities=_strings(
                list(payload.get("denied_authorities", []) or [])
            ),
            missing_external_proof=_strings(
                list(payload.get("missing_external_proof", []) or [])
            ),
            blockers=_strings(list(payload.get("blockers", []) or [])),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", BIO_NEURO_RECEIPT_JOIN_REPORT_VERSION)),
        )


def load_bio_neuro_receipts_jsonl(path: str | Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            receipts.append(_mapping(json.loads(line)))
    return receipts


def build_bio_neuro_receipt_join(
    *,
    receipts: Iterable[Mapping[str, Any]],
    join_rows_path: str | Path,
    source_receipts_path: str | Path = "",
    source_report_path: str | Path = "",
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[BioNeuroReceiptJoinReport, list[BioNeuroReceiptJoinRow]]:
    receipt_rows = [_mapping(receipt) for receipt in receipts]
    join_rows: list[BioNeuroReceiptJoinRow] = []
    blockers: list[str] = []
    owner_counts: dict[str, int] = {}
    all_slots: set[str] = set()

    for index, receipt in enumerate(receipt_rows):
        receipt_version = str(receipt.get("version", ""))
        receipt_id = _receipt_identity(receipt)
        owner_wm = _owner_wm(receipt)
        slots = _consumption_slots(receipt)
        if not receipt_version:
            blockers.append("bio_neuro_receipt_version_missing")
        if not slots:
            blockers.append(f"bio_neuro_receipt_slots_missing::{receipt_version}")
        owner_counts[owner_wm] = owner_counts.get(owner_wm, 0) + 1
        all_slots.update(slots)
        join_payload = {
            "receipt_id": receipt_id,
            "receipt_version": receipt_version,
            "owner_wm": owner_wm,
            "slots": slots,
            "index": index,
        }
        join_rows.append(
            BioNeuroReceiptJoinRow(
                row_id=_stable_id("bio_neuro_receipt_join_row", join_payload),
                receipt_id=receipt_id,
                receipt_version=receipt_version,
                owner_wm=owner_wm,
                economic_consumption_slots=slots,
                receipt_payload=receipt,
                receipt_summary=_receipt_summary(receipt),
                promotion_eligible=False,
                provider_or_hardware_proof=False,
                trained_model_proof=False,
                ready_for_training=False,
                source_refs={
                    "source_receipts_path": str(source_receipts_path),
                    "source_report_path": str(source_report_path),
                    "receipt_index": index,
                },
                metadata={
                    "boundary": "local receipt join only; not promotion evidence",
                    "lower_wm_truth_preserved": True,
                    "phase7_abstraction_expansion": False,
                },
            )
        )

    if not receipt_rows:
        blockers.append("bio_neuro_receipts_missing")
    source_report_status = ""
    if source_report_path:
        report_path = Path(source_report_path)
        if report_path.exists():
            source_report_status = str(_load_json(report_path).get("status", ""))
        else:
            blockers.append("bio_neuro_source_report_missing")

    status = (
        "ok_bio_neuro_receipts_joined"
        if join_rows and not blockers
        else "blocked_bio_neuro_receipt_join"
    )
    report_payload = {
        "row_count": len(join_rows),
        "receipt_ids": [row.receipt_id for row in join_rows],
        "surface_versions": sorted({row.receipt_version for row in join_rows}),
        "slots": sorted(all_slots),
        "blockers": sorted(set(blockers)),
        "source_report_status": source_report_status,
        "version": BIO_NEURO_RECEIPT_JOIN_REPORT_VERSION,
    }
    report = BioNeuroReceiptJoinReport(
        join_id=_stable_id("bio_neuro_receipt_join", report_payload),
        status=status,
        row_count=len(join_rows),
        join_rows_path=str(join_rows_path),
        source_receipts_path=str(source_receipts_path),
        source_report_path=str(source_report_path),
        receipt_ids=[row.receipt_id for row in join_rows],
        surface_versions=sorted({row.receipt_version for row in join_rows}),
        owner_wm_counts=dict(sorted(owner_counts.items())),
        economic_consumption_slots=sorted(all_slots),
        promotion_eligible=False,
        provider_or_hardware_proof=False,
        trained_model_proof=False,
        ready_for_training=False,
        lower_wm_truth_redefined=False,
        reward_math_mutation=False,
        phase7_abstraction_expanded=False,
        blockers=sorted(set(blockers)),
        artifact_refs={
            "join_rows_path": str(join_rows_path),
            "source_receipts_path": str(source_receipts_path),
            "source_report_path": str(source_report_path),
            **_mapping(artifact_refs),
        },
        metadata={
            "boundary": "bio/neuro receipt join only; no training, provider, hardware, promotion, or Phase 7 expansion claim",
            "source_report_status": source_report_status,
            **_mapping(metadata),
        },
    )
    return report, join_rows


def save_bio_neuro_receipt_join(
    *,
    report_path: str | Path,
    join_rows_path: str | Path,
    report: BioNeuroReceiptJoinReport,
    join_rows: Iterable[BioNeuroReceiptJoinRow],
) -> None:
    _write_jsonl(join_rows_path, (row.to_dict() for row in join_rows))
    _write_json(report_path, report.to_dict())


def load_bio_neuro_receipt_join_rows(
    path: str | Path,
) -> list[BioNeuroReceiptJoinRow]:
    rows: list[BioNeuroReceiptJoinRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(BioNeuroReceiptJoinRow.from_dict(json.loads(line)))
    return rows


def load_bio_neuro_receipt_join_report(
    path: str | Path,
) -> BioNeuroReceiptJoinReport:
    return BioNeuroReceiptJoinReport.from_dict(_load_json(path))


def build_bio_neuro_receipt_join_from_paths(
    *,
    receipts_path: str | Path,
    output_dir: str | Path,
    source_report_path: str | Path = "",
    report_path: str | Path | None = None,
    join_rows_path: str | Path | None = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> BioNeuroReceiptJoinReport:
    output_root = Path(output_dir)
    resolved_rows_path = (
        Path(join_rows_path)
        if join_rows_path is not None
        else output_root / "bio_neuro_receipt_join_rows_v1.jsonl"
    )
    resolved_report_path = (
        Path(report_path)
        if report_path is not None
        else output_root / "bio_neuro_receipt_join_report_v1.json"
    )
    report, rows = build_bio_neuro_receipt_join(
        receipts=load_bio_neuro_receipts_jsonl(receipts_path),
        join_rows_path=resolved_rows_path,
        source_receipts_path=receipts_path,
        source_report_path=source_report_path,
        artifact_refs={"report_path": str(resolved_report_path)},
        metadata=metadata,
    )
    save_bio_neuro_receipt_join(
        report_path=resolved_report_path,
        join_rows_path=resolved_rows_path,
        report=report,
        join_rows=rows,
    )
    return report


__all__ = [
    "BIO_NEURO_DENIED_AUTHORITIES",
    "BIO_NEURO_MISSING_EXTERNAL_PROOF",
    "BIO_NEURO_RECEIPT_JOIN_REPORT_VERSION",
    "BIO_NEURO_RECEIPT_JOIN_ROW_VERSION",
    "BioNeuroReceiptJoinReport",
    "BioNeuroReceiptJoinRow",
    "build_bio_neuro_receipt_join",
    "build_bio_neuro_receipt_join_from_paths",
    "load_bio_neuro_receipt_join_report",
    "load_bio_neuro_receipt_join_rows",
    "load_bio_neuro_receipts_jsonl",
    "save_bio_neuro_receipt_join",
]
