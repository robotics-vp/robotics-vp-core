"""Training-row builders for Phase 3.4 Embodiment / Actuation seams."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import clip01, mapping, safe_float, stable_id, strings
from .neural_seams import encode_state_features
from .receipts import EmbodimentReceipt
from .state import EmbodimentActuationWorldState

PHASE34_TRAINING_SCHEMA_VERSION = "phase3_4_embodiment_training_rows_v1"


@dataclass(frozen=True)
class EmbodimentSeamTrainingRow:
    row_id: str
    seam_id: str
    row_kind: str
    feature_vector: list[float]
    target_vector: list[float]
    target_names: list[str]
    admissibility: str = "diagnostic_only"
    blocker_reasons: list[str] = field(default_factory=list)
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE34_TRAINING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "seam_id": self.seam_id,
            "row_kind": self.row_kind,
            "feature_vector": [float(value) for value in self.feature_vector],
            "target_vector": [float(value) for value in self.target_vector],
            "target_names": strings(self.target_names),
            "admissibility": self.admissibility,
            "blocker_reasons": strings(self.blocker_reasons),
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentTrainingManifest:
    manifest_id: str
    row_count: int
    row_counts_by_seam: dict[str, int]
    admissibility_counts: dict[str, int]
    promotion_eligible: bool = False
    blocker_reasons: list[str] = field(default_factory=list)
    source_refs: dict[str, Any] = field(default_factory=dict)
    version: str = "phase3_4_embodiment_training_manifest_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "row_count": int(self.row_count),
            "row_counts_by_seam": {str(k): int(v) for k, v in self.row_counts_by_seam.items()},
            "admissibility_counts": {str(k): int(v) for k, v in self.admissibility_counts.items()},
            "promotion_eligible": bool(self.promotion_eligible),
            "blocker_reasons": strings(self.blocker_reasons),
            "source_refs": mapping(self.source_refs),
            "version": self.version,
        }


def _features(state: EmbodimentActuationWorldState, dim: int) -> list[float]:
    return [float(value) for value in encode_state_features(state, dim).tolist()]


def _receipt_versions(receipts: Iterable[EmbodimentReceipt]) -> list[str]:
    return [receipt.version for receipt in receipts]


def build_phase34_training_rows_from_state(
    state: EmbodimentActuationWorldState,
    receipts: Iterable[EmbodimentReceipt] = (),
) -> list[EmbodimentSeamTrainingRow]:
    receipt_versions = _receipt_versions(receipts)
    source_refs = {"state_id": state.state_id, "receipt_versions": receipt_versions}
    external_blockers = sorted(
        set(
            strings(state.safety_envelope.missing_evidence)
            + strings(state.calibration_targets.missing_evidence)
        )
    )
    rows = [
        EmbodimentSeamTrainingRow(
            row_id=stable_id("embodiment_training_row", {"state_id": state.state_id, "seam": "local_contact_dynamics"}),
            seam_id="local_contact_dynamics",
            row_kind="shadow_contact_forecast",
            feature_vector=_features(state, 16),
            target_vector=[
                clip01(state.contact_state.contact_confidence_mean),
                clip01(state.local_dynamics_forecast.contact_transition_risk),
                clip01(state.local_dynamics_forecast.confidence),
            ],
            target_names=["next_contact_probability", "transition_risk", "forecast_confidence"],
            admissibility="positive_training" if state.contact_state.contact_event_count > 0 else "diagnostic_only",
            blocker_reasons=[] if state.contact_state.contact_event_count > 0 else ["no_contact_events"],
            source_refs=source_refs,
        ),
        EmbodimentSeamTrainingRow(
            row_id=stable_id("embodiment_training_row", {"state_id": state.state_id, "seam": "inverse_retargeting"}),
            seam_id="inverse_retargeting",
            row_kind="adapter_retarget_shadow",
            feature_vector=_features(state, 32),
            target_vector=[clip01(state.inverse_retarget_trace.readiness_score)],
            target_names=["readiness_score"],
            admissibility=(
                "positive_training"
                if state.inverse_retarget_trace.readiness_score > 0.0
                else "negative_supervision"
            ),
            blocker_reasons=list(state.inverse_retarget_trace.missing_evidence),
            source_refs=source_refs,
        ),
        EmbodimentSeamTrainingRow(
            row_id=stable_id("embodiment_training_row", {"state_id": state.state_id, "seam": "action_proposal"}),
            seam_id="action_proposal",
            row_kind="shadow_action_feasibility",
            feature_vector=_features(state, 32),
            target_vector=[
                clip01(state.action_proposal_bundle.action_feasibility_score),
                safe_float(state.action_proposal_bundle.proposal_count) / max(
                    safe_float(state.contact_affordance_graph.node_count, 1.0), 1.0
                ),
            ],
            target_names=["feasibility_score", "proposal_density"],
            admissibility="positive_training" if state.action_proposal_bundle.proposal_count > 0 else "diagnostic_only",
            blocker_reasons=[] if state.action_proposal_bundle.proposal_count > 0 else ["no_actionable_objects"],
            source_refs=source_refs,
        ),
        EmbodimentSeamTrainingRow(
            row_id=stable_id("embodiment_training_row", {"state_id": state.state_id, "seam": "drift_calibration"}),
            seam_id="drift_calibration",
            row_kind="shadow_calibration_priority",
            feature_vector=_features(state, 20),
            target_vector=[
                clip01(state.drift_summary.drift_score),
                clip01(state.calibration_targets.priority_score),
                clip01(state.safety_envelope.margin_fraction),
            ],
            target_names=["drift_score", "calibration_priority", "safety_margin_estimate"],
            admissibility="negative_supervision" if external_blockers else "positive_training",
            blocker_reasons=external_blockers,
            source_refs=source_refs,
        ),
    ]
    return rows


def build_phase34_training_manifest(
    rows: Iterable[EmbodimentSeamTrainingRow],
    *,
    source_refs: Mapping[str, Any] | None = None,
) -> EmbodimentTrainingManifest:
    row_list = list(rows)
    by_seam: dict[str, int] = {}
    by_admissibility: dict[str, int] = {}
    blockers: list[str] = []
    for row in row_list:
        by_seam[row.seam_id] = by_seam.get(row.seam_id, 0) + 1
        by_admissibility[row.admissibility] = by_admissibility.get(row.admissibility, 0) + 1
        blockers.extend(row.blocker_reasons)
    blockers.extend(["no_gpu_training_run", "no_benchmark_promotion_evidence"])
    return EmbodimentTrainingManifest(
        manifest_id=stable_id(
            "embodiment_training_manifest",
            {"row_ids": [row.row_id for row in row_list], "source_refs": mapping(source_refs)},
        ),
        row_count=len(row_list),
        row_counts_by_seam=by_seam,
        admissibility_counts=by_admissibility,
        promotion_eligible=False,
        blocker_reasons=sorted(set(strings(blockers))),
        source_refs=mapping(source_refs),
    )


def write_phase34_training_rows_jsonl(
    rows: Iterable[EmbodimentSeamTrainingRow],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    return output


def load_phase34_training_rows_jsonl(path: str | Path) -> list[EmbodimentSeamTrainingRow]:
    rows: list[EmbodimentSeamTrainingRow] = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                EmbodimentSeamTrainingRow(
                    row_id=str(payload["row_id"]),
                    seam_id=str(payload["seam_id"]),
                    row_kind=str(payload["row_kind"]),
                    feature_vector=[float(v) for v in payload.get("feature_vector", [])],
                    target_vector=[float(v) for v in payload.get("target_vector", [])],
                    target_names=strings(payload.get("target_names")),
                    admissibility=str(payload.get("admissibility", "diagnostic_only")),
                    blocker_reasons=strings(payload.get("blocker_reasons")),
                    source_refs=mapping(payload.get("source_refs")),
                    metadata=mapping(payload.get("metadata")),
                )
            )
    return rows
