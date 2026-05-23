"""Phase-6 transport local closure audit.

This audit confirms whether the local Phase-6 transport scaffold is
structurally closed. It deliberately separates local contract/runtime closure
from future training, corpus-density, provider, hardware, latency, and
promotion evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.advisory_runtime import (
    WMTransportAdvisoryRuntimeReport,
)
from src.world_model.transport.losses import WMTransportLossLedger
from src.world_model.transport.neural_manifest import (
    WMTransportNeuralArchitectureManifest,
)
from src.world_model.transport.runtime import WMTransportPhase6ScaffoldReport
from src.world_model.transport.training import WMTransportTrainerScaffoldManifest

WM_TRANSPORT_PHASE6_CLOSURE_AUDIT_VERSION = "wm_transport_phase6_closure_audit_v1"

PHASE6_REMAINING_EVIDENCE_BLOCKERS = (
    "cross_wm_corpus_density_not_proven",
    "gpu_bridge_receiver_training_not_run",
    "topology_latency_benchmarks_not_run",
    "provider_or_hardware_transport_evidence_missing",
    "promotion_grade_downstream_benchmark_missing",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class WMTransportPhase6ClosureAuditReport:
    """Audit report for local Phase-6 transport structural closure."""

    audit_id: str
    scaffold_report_id: str
    neural_manifest_id: str
    loss_ledger_id: str
    trainer_scaffold_id: str
    advisory_runtime_report_id: str
    status: str
    local_phase6_structurally_closed: bool = False
    missing_local_runtime_contracts: list[str] = field(default_factory=list)
    remaining_evidence_blockers: list[str] = field(default_factory=list)
    closed_local_surfaces: list[str] = field(default_factory=list)
    contract_count: int = 0
    transformer_count: int = 0
    training_row_count: int = 0
    roundtrip_receipt_count: int = 0
    neural_component_count: int = 0
    loss_count: int = 0
    advisory_proposal_count: int = 0
    advisory_receipt_count: int = 0
    decomposed_eval_report_count: int = 0
    joined_shadow_outcome_count: int = 0
    authority_class: str = "transport_phase6_closure_audit_only"
    ready_for_training: bool = False
    ready_for_gpu_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_PHASE6_CLOSURE_AUDIT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "version": self.version,
            "scaffold_report_id": self.scaffold_report_id,
            "neural_manifest_id": self.neural_manifest_id,
            "loss_ledger_id": self.loss_ledger_id,
            "trainer_scaffold_id": self.trainer_scaffold_id,
            "advisory_runtime_report_id": self.advisory_runtime_report_id,
            "status": self.status,
            "local_phase6_structurally_closed": bool(
                self.local_phase6_structurally_closed
            ),
            "missing_local_runtime_contracts": list(
                self.missing_local_runtime_contracts
            ),
            "remaining_evidence_blockers": list(self.remaining_evidence_blockers),
            "closed_local_surfaces": list(self.closed_local_surfaces),
            "contract_count": int(self.contract_count),
            "transformer_count": int(self.transformer_count),
            "training_row_count": int(self.training_row_count),
            "roundtrip_receipt_count": int(self.roundtrip_receipt_count),
            "neural_component_count": int(self.neural_component_count),
            "loss_count": int(self.loss_count),
            "advisory_proposal_count": int(self.advisory_proposal_count),
            "advisory_receipt_count": int(self.advisory_receipt_count),
            "decomposed_eval_report_count": int(self.decomposed_eval_report_count),
            "joined_shadow_outcome_count": int(self.joined_shadow_outcome_count),
            "authority_class": self.authority_class,
            "ready_for_training": bool(self.ready_for_training),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportPhase6ClosureAuditReport":
        return cls(
            audit_id=str(payload.get("audit_id", "")),
            scaffold_report_id=str(payload.get("scaffold_report_id", "")),
            neural_manifest_id=str(payload.get("neural_manifest_id", "")),
            loss_ledger_id=str(payload.get("loss_ledger_id", "")),
            trainer_scaffold_id=str(payload.get("trainer_scaffold_id", "")),
            advisory_runtime_report_id=str(
                payload.get("advisory_runtime_report_id", "")
            ),
            status=str(payload.get("status", "blocked")),
            local_phase6_structurally_closed=bool(
                payload.get("local_phase6_structurally_closed", False)
            ),
            missing_local_runtime_contracts=[
                str(item)
                for item in list(
                    payload.get("missing_local_runtime_contracts", []) or []
                )
            ],
            remaining_evidence_blockers=[
                str(item)
                for item in list(payload.get("remaining_evidence_blockers", []) or [])
            ],
            closed_local_surfaces=[
                str(item)
                for item in list(payload.get("closed_local_surfaces", []) or [])
            ],
            contract_count=int(payload.get("contract_count", 0) or 0),
            transformer_count=int(payload.get("transformer_count", 0) or 0),
            training_row_count=int(payload.get("training_row_count", 0) or 0),
            roundtrip_receipt_count=int(
                payload.get("roundtrip_receipt_count", 0) or 0
            ),
            neural_component_count=int(
                payload.get("neural_component_count", 0) or 0
            ),
            loss_count=int(payload.get("loss_count", 0) or 0),
            advisory_proposal_count=int(
                payload.get("advisory_proposal_count", 0) or 0
            ),
            advisory_receipt_count=int(
                payload.get("advisory_receipt_count", 0) or 0
            ),
            decomposed_eval_report_count=int(
                payload.get("decomposed_eval_report_count", 0) or 0
            ),
            joined_shadow_outcome_count=int(
                payload.get("joined_shadow_outcome_count", 0) or 0
            ),
            authority_class=str(
                payload.get("authority_class", "transport_phase6_closure_audit_only")
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            ready_for_gpu_training=bool(
                payload.get("ready_for_gpu_training", False)
            ),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_PHASE6_CLOSURE_AUDIT_VERSION)
            ),
        )


def _missing_local_surfaces(
    *,
    scaffold_report: WMTransportPhase6ScaffoldReport,
    neural_manifest: WMTransportNeuralArchitectureManifest,
    loss_ledger: WMTransportLossLedger,
    trainer_manifest: WMTransportTrainerScaffoldManifest,
    advisory_runtime_report: WMTransportAdvisoryRuntimeReport,
) -> list[str]:
    missing: list[str] = []
    if scaffold_report.status != "ok" or scaffold_report.contract_count <= 0:
        missing.append("adjacent_transport_contracts")
    if scaffold_report.transformer_count <= 0:
        missing.append("per_wm_exporter_receiver_posture")
    if scaffold_report.training_row_count <= 0:
        missing.append("transport_training_rows")
    if scaffold_report.roundtrip_receipt_count != scaffold_report.contract_count:
        missing.append("roundtrip_topology_uncertainty_receipts")
    if not neural_manifest.ready_for_trainer_scaffold or not neural_manifest.components:
        missing.append("phase6_3_neural_manifest")
    if loss_ledger.status != "ok" or loss_ledger.loss_count <= 0:
        missing.append("phase6_3_loss_ledger")
    if (
        not trainer_manifest.dataset_contract_ready
        or not trainer_manifest.losses_defined
        or not trainer_manifest.cpu_smoke_forward_passed
    ):
        missing.append("non_training_trainer_scaffold")
    if (
        advisory_runtime_report.status != "ok"
        or not advisory_runtime_report.ready_for_decomposed_eval
        or advisory_runtime_report.proposal_count <= 0
        or advisory_runtime_report.eval_report_count <= 0
    ):
        missing.append("phase6_4_advisory_runtime")
    return missing


def build_wm_transport_phase6_closure_audit(
    *,
    scaffold_report: WMTransportPhase6ScaffoldReport,
    neural_manifest: WMTransportNeuralArchitectureManifest,
    loss_ledger: WMTransportLossLedger,
    trainer_manifest: WMTransportTrainerScaffoldManifest,
    advisory_runtime_report: WMTransportAdvisoryRuntimeReport,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportPhase6ClosureAuditReport:
    missing = _missing_local_surfaces(
        scaffold_report=scaffold_report,
        neural_manifest=neural_manifest,
        loss_ledger=loss_ledger,
        trainer_manifest=trainer_manifest,
        advisory_runtime_report=advisory_runtime_report,
    )
    denied_runtime = (
        not scaffold_report.training_executed
        and not neural_manifest.training_executed
        and not neural_manifest.weights_written
        and not trainer_manifest.training_executed
        and not trainer_manifest.weights_written
        and not trainer_manifest.provider_executed
        and not trainer_manifest.hardware_executed
        and not trainer_manifest.live_policy_control
        and not trainer_manifest.reward_math_mutation
        and not trainer_manifest.promotion_eligible
        and not advisory_runtime_report.training_executed
        and not advisory_runtime_report.weights_written
        and not advisory_runtime_report.provider_executed
        and not advisory_runtime_report.hardware_executed
        and not advisory_runtime_report.live_policy_control
        and not advisory_runtime_report.reward_math_mutation
        and not advisory_runtime_report.promotion_eligible
    )
    if not denied_runtime:
        missing.append("denied_authority_gates")
    status = "ok" if not missing else "blocked"
    payload = {
        "scaffold_report_id": scaffold_report.report_id,
        "neural_manifest_id": neural_manifest.manifest_id,
        "loss_ledger_id": loss_ledger.ledger_id,
        "trainer_scaffold_id": trainer_manifest.trainer_scaffold_id,
        "advisory_runtime_report_id": advisory_runtime_report.report_id,
        "missing": missing,
    }
    return WMTransportPhase6ClosureAuditReport(
        audit_id=f"wm_transport_phase6_closure_{sha256_json(payload)[:16]}",
        scaffold_report_id=scaffold_report.report_id,
        neural_manifest_id=neural_manifest.manifest_id,
        loss_ledger_id=loss_ledger.ledger_id,
        trainer_scaffold_id=trainer_manifest.trainer_scaffold_id,
        advisory_runtime_report_id=advisory_runtime_report.report_id,
        status=status,
        local_phase6_structurally_closed=status == "ok",
        missing_local_runtime_contracts=missing,
        remaining_evidence_blockers=list(PHASE6_REMAINING_EVIDENCE_BLOCKERS),
        closed_local_surfaces=[
            "adjacent_transport_contracts",
            "per_wm_exporter_receiver_posture",
            "transport_training_rows",
            "roundtrip_topology_uncertainty_governance_receiver_receipts",
            "neural_manifest_loss_ledger",
            "non_training_trainer_scaffold",
            "advisory_runtime_proposals_invocations_receipts",
            "decomposed_bridge_receiver_downstream_eval_reports",
            "shadow_outcome_join_slots",
            "phase5_followup_ledger_preserved",
        ]
        if status == "ok"
        else [],
        contract_count=scaffold_report.contract_count,
        transformer_count=scaffold_report.transformer_count,
        training_row_count=scaffold_report.training_row_count,
        roundtrip_receipt_count=scaffold_report.roundtrip_receipt_count,
        neural_component_count=len(neural_manifest.components),
        loss_count=loss_ledger.loss_count,
        advisory_proposal_count=advisory_runtime_report.proposal_count,
        advisory_receipt_count=advisory_runtime_report.receipt_count,
        decomposed_eval_report_count=advisory_runtime_report.eval_report_count,
        joined_shadow_outcome_count=advisory_runtime_report.joined_shadow_outcome_count,
        blockers=list(PHASE6_REMAINING_EVIDENCE_BLOCKERS),
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "phase": "6_transport_local_closure_audit",
            "closure_claim": "local_structural_only",
            "not_claimed": [
                "gpu_training",
                "provider_execution",
                "hardware_execution",
                "latency_benchmark",
                "promotion",
                "live_authority",
            ],
            **_mapping(metadata),
        },
    )


def save_wm_transport_phase6_closure_audit(
    path: str | Path, report: WMTransportPhase6ClosureAuditReport
) -> None:
    _write_json(path, report.to_dict())


def load_wm_transport_phase6_closure_audit(
    path: str | Path,
) -> WMTransportPhase6ClosureAuditReport:
    return WMTransportPhase6ClosureAuditReport.from_dict(_load_json(path))


__all__ = [
    "PHASE6_REMAINING_EVIDENCE_BLOCKERS",
    "WM_TRANSPORT_PHASE6_CLOSURE_AUDIT_VERSION",
    "WMTransportPhase6ClosureAuditReport",
    "build_wm_transport_phase6_closure_audit",
    "load_wm_transport_phase6_closure_audit",
    "save_wm_transport_phase6_closure_audit",
]
