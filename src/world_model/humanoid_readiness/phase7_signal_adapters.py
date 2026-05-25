"""Lower-WM signal adapters for Phase 7 governance-node surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.world_model.humanoid_readiness.common import (
    float_mapping,
    load_json,
    load_jsonl,
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.phase7 import (
    DENIED_PHASE7_AUTHORITIES,
    PHASE7_REMAINING_BLOCKERS,
    Phase7GovernanceNodeSurface,
    load_phase7_governance_node_surfaces,
)

PHASE7_GOVERNANCE_SIGNAL_ADAPTER_REPORT_VERSION = (
    "phase7_governance_signal_adapter_report_v1"
)
PHASE7_GOVERNANCE_NODE_SIGNAL_ADAPTER_VERSION = (
    "phase7_governance_node_signal_adapter_v1"
)
PHASE7_GOVERNANCE_NODE_SIGNAL_RECEIPT_VERSION = (
    "phase7_governance_node_signal_receipt_v1"
)

PHASE7_SIGNAL_ADAPTER_REMAINING_BLOCKERS = (
    *PHASE7_REMAINING_BLOCKERS,
    "live_lower_wm_runtime_streams_missing",
    "labeled_governance_signal_outcomes_missing",
    "trained_governance_signal_weights_missing",
)

EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS = (
    "economic_allocation_governance",
    "reward_integrity_governance",
    "plausibility_geometry_governance",
    "deployment_truth_governance",
    "safety_constraint_governance",
    "data_value_governance",
    "embodiment_limit_governance",
    "coordination_operator_governance",
)


def _signal_denied_gates(extra: Mapping[str, Any] | None = None) -> dict[str, bool]:
    gates = {
        "training_executed": False,
        "weights_written": False,
        "provider_executed": False,
        "hardware_executed": False,
        "unitree_sim_runtime_executed": False,
        "live_policy_control": False,
        "reward_math_mutation": False,
        "promotion_eligible": False,
        "phase7_authority_granted": False,
        "live_dispatch_allowed": False,
        "hard_veto_dispatch": False,
        "lower_wm_replacement": False,
        "scalar_governance_collapse": False,
    }
    gates.update({str(key): bool(value) for key, value in dict(extra or {}).items()})
    return gates


@dataclass(frozen=True)
class Phase7GovernanceNodeSignalAdapter:
    adapter_id: str
    surface_id: str
    node_key: str
    domain_key: str
    source_artifact_refs: dict[str, str] = field(default_factory=dict)
    source_receipt_ids: list[str] = field(default_factory=list)
    source_receipt_families: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    signal_slots: dict[str, Any] = field(default_factory=dict)
    adapter_status: str = "ok"
    evidence_class: str = "lower_wm_receipt_backed_shadow_signal"
    lower_wm_receipt_backed: bool = True
    shadow_only: bool = True
    advisory_only: bool = True
    training_aware: bool = True
    promotion_eligible: bool = False
    authority_class: str = "phase7_governance_signal_adapter_only"
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_GOVERNANCE_NODE_SIGNAL_ADAPTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "version": self.version,
            "surface_id": self.surface_id,
            "node_key": self.node_key,
            "domain_key": self.domain_key,
            "source_artifact_refs": dict(self.source_artifact_refs),
            "source_receipt_ids": list(self.source_receipt_ids),
            "source_receipt_families": list(self.source_receipt_families),
            "metrics": float_mapping(self.metrics),
            "signal_slots": mapping(self.signal_slots),
            "adapter_status": self.adapter_status,
            "evidence_class": self.evidence_class,
            "lower_wm_receipt_backed": bool(self.lower_wm_receipt_backed),
            "shadow_only": bool(self.shadow_only),
            "advisory_only": bool(self.advisory_only),
            "training_aware": bool(self.training_aware),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7GovernanceNodeSignalAdapter":
        return cls(
            adapter_id=str(payload.get("adapter_id", "")),
            surface_id=str(payload.get("surface_id", "")),
            node_key=str(payload.get("node_key", "")),
            domain_key=str(payload.get("domain_key", "")),
            source_artifact_refs={
                str(key): str(value)
                for key, value in dict(payload.get("source_artifact_refs") or {}).items()
            },
            source_receipt_ids=strings(payload.get("source_receipt_ids")),
            source_receipt_families=strings(payload.get("source_receipt_families")),
            metrics=float_mapping(payload.get("metrics")),
            signal_slots=mapping(payload.get("signal_slots")),
            adapter_status=str(payload.get("adapter_status", "blocked")),
            evidence_class=str(
                payload.get(
                    "evidence_class", "lower_wm_receipt_backed_shadow_signal"
                )
            ),
            lower_wm_receipt_backed=bool(
                payload.get("lower_wm_receipt_backed", True)
            ),
            shadow_only=bool(payload.get("shadow_only", True)),
            advisory_only=bool(payload.get("advisory_only", True)),
            training_aware=bool(payload.get("training_aware", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get(
                    "authority_class", "phase7_governance_signal_adapter_only"
                )
            ),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(
                payload.get(
                    "version", PHASE7_GOVERNANCE_NODE_SIGNAL_ADAPTER_VERSION
                )
            ),
        )


@dataclass(frozen=True)
class Phase7GovernanceNodeSignalReceipt:
    signal_id: str
    adapter_id: str
    surface_id: str
    node_key: str
    domain_key: str
    signal_key: str
    source_receipt_ids: list[str] = field(default_factory=list)
    source_artifact_refs: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    candidate_outputs: dict[str, Any] = field(default_factory=dict)
    hard_constraint_candidate: bool = False
    shadow_only: bool = True
    advisory_only: bool = True
    lower_wm_receipt_backed: bool = True
    training_aware: bool = True
    live_dispatch_allowed: bool = False
    hard_veto_dispatch: bool = False
    reward_math_mutation: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    authority_class: str = "phase7_governance_node_signal_receipt_only"
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_GOVERNANCE_NODE_SIGNAL_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "version": self.version,
            "adapter_id": self.adapter_id,
            "surface_id": self.surface_id,
            "node_key": self.node_key,
            "domain_key": self.domain_key,
            "signal_key": self.signal_key,
            "source_receipt_ids": list(self.source_receipt_ids),
            "source_artifact_refs": dict(self.source_artifact_refs),
            "confidence": float(self.confidence),
            "candidate_outputs": mapping(self.candidate_outputs),
            "hard_constraint_candidate": bool(self.hard_constraint_candidate),
            "shadow_only": bool(self.shadow_only),
            "advisory_only": bool(self.advisory_only),
            "lower_wm_receipt_backed": bool(self.lower_wm_receipt_backed),
            "training_aware": bool(self.training_aware),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "hard_veto_dispatch": bool(self.hard_veto_dispatch),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7GovernanceNodeSignalReceipt":
        return cls(
            signal_id=str(payload.get("signal_id", "")),
            adapter_id=str(payload.get("adapter_id", "")),
            surface_id=str(payload.get("surface_id", "")),
            node_key=str(payload.get("node_key", "")),
            domain_key=str(payload.get("domain_key", "")),
            signal_key=str(payload.get("signal_key", "")),
            source_receipt_ids=strings(payload.get("source_receipt_ids")),
            source_artifact_refs={
                str(key): str(value)
                for key, value in dict(payload.get("source_artifact_refs") or {}).items()
            },
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            candidate_outputs=mapping(payload.get("candidate_outputs")),
            hard_constraint_candidate=bool(
                payload.get("hard_constraint_candidate", False)
            ),
            shadow_only=bool(payload.get("shadow_only", True)),
            advisory_only=bool(payload.get("advisory_only", True)),
            lower_wm_receipt_backed=bool(
                payload.get("lower_wm_receipt_backed", True)
            ),
            training_aware=bool(payload.get("training_aware", True)),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            hard_veto_dispatch=bool(payload.get("hard_veto_dispatch", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get(
                    "authority_class", "phase7_governance_node_signal_receipt_only"
                )
            ),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(
                payload.get(
                    "version", PHASE7_GOVERNANCE_NODE_SIGNAL_RECEIPT_VERSION
                )
            ),
        )


@dataclass(frozen=True)
class Phase7GovernanceSignalAdapterReport:
    report_id: str
    phase7_scaffold_report_id: str
    status: str
    governance_node_surface_count: int
    adapter_count: int
    signal_receipt_count: int
    source_artifact_count: int
    missing_source_artifact_count: int
    lower_wm_receipt_backed_node_count: int
    all_eight_nodes_signal_backed: bool
    shadow_runtime_feed_ready: bool
    local_signal_adapter_complete: bool
    phase7_authority_granted: bool = False
    live_dispatch_allowed: bool = False
    hard_veto_dispatch: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_signal_denied_gates)
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_GOVERNANCE_SIGNAL_ADAPTER_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase7_scaffold_report_id": self.phase7_scaffold_report_id,
            "status": self.status,
            "governance_node_surface_count": int(self.governance_node_surface_count),
            "adapter_count": int(self.adapter_count),
            "signal_receipt_count": int(self.signal_receipt_count),
            "source_artifact_count": int(self.source_artifact_count),
            "missing_source_artifact_count": int(
                self.missing_source_artifact_count
            ),
            "lower_wm_receipt_backed_node_count": int(
                self.lower_wm_receipt_backed_node_count
            ),
            "all_eight_nodes_signal_backed": bool(
                self.all_eight_nodes_signal_backed
            ),
            "shadow_runtime_feed_ready": bool(self.shadow_runtime_feed_ready),
            "local_signal_adapter_complete": bool(
                self.local_signal_adapter_complete
            ),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "hard_veto_dispatch": bool(self.hard_veto_dispatch),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": _signal_denied_gates(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7GovernanceSignalAdapterReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase7_scaffold_report_id=str(
                payload.get("phase7_scaffold_report_id", "")
            ),
            status=str(payload.get("status", "blocked")),
            governance_node_surface_count=int(
                payload.get("governance_node_surface_count", 0) or 0
            ),
            adapter_count=int(payload.get("adapter_count", 0) or 0),
            signal_receipt_count=int(payload.get("signal_receipt_count", 0) or 0),
            source_artifact_count=int(payload.get("source_artifact_count", 0) or 0),
            missing_source_artifact_count=int(
                payload.get("missing_source_artifact_count", 0) or 0
            ),
            lower_wm_receipt_backed_node_count=int(
                payload.get("lower_wm_receipt_backed_node_count", 0) or 0
            ),
            all_eight_nodes_signal_backed=bool(
                payload.get("all_eight_nodes_signal_backed", False)
            ),
            shadow_runtime_feed_ready=bool(
                payload.get("shadow_runtime_feed_ready", False)
            ),
            local_signal_adapter_complete=bool(
                payload.get("local_signal_adapter_complete", False)
            ),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            hard_veto_dispatch=bool(payload.get("hard_veto_dispatch", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            unitree_sim_runtime_executed=bool(
                payload.get("unitree_sim_runtime_executed", False)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates=_signal_denied_gates(payload.get("denied_gates")),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get(
                    "version", PHASE7_GOVERNANCE_SIGNAL_ADAPTER_REPORT_VERSION
                )
            ),
        )


@dataclass(frozen=True)
class _EvidenceBundle:
    refs: dict[str, str]
    payloads: dict[str, Any]
    rows: dict[str, list[dict[str, Any]]]
    missing_refs: list[str]


def build_phase7_governance_signal_adapters(
    *,
    phase7_scaffold_report_id: str,
    governance_node_surfaces: Sequence[Phase7GovernanceNodeSurface],
    lower_artifact_root: str | Path,
    artifact_refs: Mapping[str, Any] | None = None,
) -> tuple[
    Phase7GovernanceSignalAdapterReport,
    list[Phase7GovernanceNodeSignalAdapter],
    list[Phase7GovernanceNodeSignalReceipt],
]:
    root = Path(lower_artifact_root)
    bundle = _load_lower_wm_evidence(root)
    surface_by_key = {surface.node_key: surface for surface in governance_node_surfaces}
    adapters: list[Phase7GovernanceNodeSignalAdapter] = []
    receipts: list[Phase7GovernanceNodeSignalReceipt] = []
    for node_key in EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS:
        surface = surface_by_key.get(node_key)
        if surface is None:
            continue
        adapter, receipt = _build_node_signal(surface=surface, bundle=bundle)
        adapters.append(adapter)
        receipts.append(receipt)

    complete = (
        len(adapters) == len(EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS)
        and len(receipts) == len(adapters)
        and all(item.adapter_status == "ok" for item in adapters)
        and all(item.lower_wm_receipt_backed for item in receipts)
        and not bundle.missing_refs
    )
    backed_count = sum(1 for item in receipts if item.lower_wm_receipt_backed)
    report_payload = {
        "phase7_scaffold_report_id": phase7_scaffold_report_id,
        "adapter_count": len(adapters),
        "signal_receipt_count": len(receipts),
        "artifact_refs": mapping(artifact_refs),
    }
    report = Phase7GovernanceSignalAdapterReport(
        report_id=stable_id("phase7_signal_adapter", report_payload),
        phase7_scaffold_report_id=phase7_scaffold_report_id,
        status="ok" if complete else "blocked",
        governance_node_surface_count=len(governance_node_surfaces),
        adapter_count=len(adapters),
        signal_receipt_count=len(receipts),
        source_artifact_count=len(bundle.refs),
        missing_source_artifact_count=len(bundle.missing_refs),
        lower_wm_receipt_backed_node_count=backed_count,
        all_eight_nodes_signal_backed=backed_count
        == len(EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS),
        shadow_runtime_feed_ready=complete,
        local_signal_adapter_complete=complete,
        denied_gates=_signal_denied_gates(),
        remaining_blockers=list(PHASE7_SIGNAL_ADAPTER_REMAINING_BLOCKERS),
        artifact_refs={
            **mapping(artifact_refs),
            "lower_artifact_root": str(root),
            "missing_source_artifacts": list(bundle.missing_refs),
        },
    )
    return report, adapters, receipts


def _load_lower_wm_evidence(root: Path) -> _EvidenceBundle:
    json_specs = {
        "phase35_refit_report": (
            "phase35_humanoid_capacity_env_refit/"
            "humanoid_phase35_refit_report_v1.json"
        ),
        "phase35_bipedal_readiness_audit": (
            "phase35_bipedal_readiness_audit/"
            "phase35_bipedal_readiness_audit_v1.json"
        ),
        "phase4_downstream_controller_report": (
            "phase4_downstream_controller_scaffold/"
            "phase4_downstream_controller_scaffold_report_v1.json"
        ),
        "phase4_bringup_readiness_report": (
            "phase4_unitree_bringup_readiness/"
            "phase4_unitree_bringup_readiness_report_v1.json"
        ),
        "phase4_local_harness_report": (
            "phase4_unitree_local_harnesses/"
            "phase4_unitree_local_harness_report_v1.json"
        ),
        "phase4_runtime_bridge_report": (
            "phase4_unitree_runtime_evidence_bridge/"
            "phase4_unitree_runtime_evidence_bridge_report_v1.json"
        ),
        "phase4_blocker_probe_report": (
            "phase4_unitree_blocker_stress_probes/"
            "phase4_unitree_blocker_stress_probe_report_v1.json"
        ),
        "phase6_advisory_runtime_report": (
            "phase6_transport_advisory_runtime/"
            "wm_transport_advisory_runtime_report_v1.json"
        ),
        "phase6_closure_audit": (
            "phase6_transport_closure_audit/"
            "wm_transport_phase6_closure_audit_v1.json"
        ),
        "phase65_report": (
            "phase65_meta_node_neuralization/"
            "phase65_meta_node_neuralization_report_v1.json"
        ),
        "phase7_shadow_summary": (
            "phase7_meta_regal_shadow_runtime/summary.json"
        ),
        "phase7_eval_report": (
            "phase7_meta_governance_eval/"
            "phase7_meta_governance_evaluation_report_v1.json"
        ),
    }
    jsonl_specs = {
        "phase35_balance_geometry_reports": (
            "phase35_bipedal_readiness_audit/"
            "balance_geometry_reports_v1.jsonl"
        ),
        "phase35_joint_vector_receipts": (
            "phase35_bipedal_readiness_audit/"
            "joint_vector_validation_receipts_v1.jsonl"
        ),
        "phase35_whole_body_replay_rows": (
            "phase35_bipedal_readiness_audit/"
            "whole_body_replay_rows_v1.jsonl"
        ),
        "phase4_controller_safety_receipts": (
            "phase4_downstream_controller_scaffold/"
            "controller_safety_receipts_v1.jsonl"
        ),
        "phase4_low_level_command_frames": (
            "phase4_downstream_controller_scaffold/"
            "low_level_command_frames_v1.jsonl"
        ),
        "phase4_safety_preflight_receipts": (
            "phase4_unitree_bringup_readiness/"
            "unitree_safety_preflight_receipts_v1.jsonl"
        ),
        "phase4_operator_runbooks": (
            "phase4_unitree_bringup_readiness/"
            "unitree_operator_recovery_runbooks_v1.jsonl"
        ),
        "phase4_mock_receivers": (
            "phase4_unitree_local_harnesses/"
            "unitree_mock_receiver_receipts_v1.jsonl"
        ),
        "phase4_stale_validations": (
            "phase4_unitree_local_harnesses/"
            "unitree_stale_data_validation_receipts_v1.jsonl"
        ),
        "phase4_watchdog_demotions": (
            "phase4_unitree_local_harnesses/"
            "unitree_watchdog_demotion_receipts_v1.jsonl"
        ),
        "phase4_safety_transitions": (
            "phase4_unitree_local_harnesses/"
            "unitree_safety_state_transitions_v1.jsonl"
        ),
        "phase4_trace_replay_receipts": (
            "phase4_unitree_local_harnesses/"
            "unitree_trace_replay_receipts_v1.jsonl"
        ),
        "phase4_ros2_readiness_receipts": (
            "phase4_unitree_runtime_evidence_bridge/"
            "unitree_ros2_runtime_readiness_receipts_v1.jsonl"
        ),
        "phase4_operator_drill_receipts": (
            "phase4_unitree_runtime_evidence_bridge/"
            "unitree_operator_recovery_drill_receipts_v1.jsonl"
        ),
        "phase4_operator_drill_transitions": (
            "phase4_unitree_runtime_evidence_bridge/"
            "unitree_operator_recovery_drill_transitions_v1.jsonl"
        ),
        "phase4_safety_expansion_receipts": (
            "phase4_unitree_runtime_evidence_bridge/"
            "unitree_safety_envelope_expansion_receipts_v1.jsonl"
        ),
        "phase4_blocker_probe_receipts": (
            "phase4_unitree_blocker_stress_probes/"
            "unitree_blocker_stress_probe_receipts_v1.jsonl"
        ),
        "phase4_mujoco_model_stress_receipts": (
            "phase4_unitree_blocker_stress_probes/"
            "unitree_mujoco_model_stress_receipts_v1.jsonl"
        ),
        "phase6_transport_eval_reports": (
            "phase6_transport_advisory_runtime/"
            "wm_transport_decomposed_eval_reports_v1.jsonl"
        ),
        "phase65_meta_node_trajectory_receipts": (
            "phase65_meta_node_neuralization/"
            "meta_node_trajectory_receipts_v1.jsonl"
        ),
        "phase65_meta_node_robustness_reports": (
            "phase65_meta_node_neuralization/"
            "meta_node_robustness_reports_v1.jsonl"
        ),
        "phase7_outcome_join_rows": (
            "phase7_meta_governance_eval/phase7_outcome_join_rows_v1.jsonl"
        ),
        "phase7_control_field_evals": (
            "phase7_meta_governance_eval/"
            "phase7_control_field_eval_reports_v1.jsonl"
        ),
    }
    refs: dict[str, str] = {}
    payloads: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for key, relative in json_specs.items():
        path = root / relative
        refs[key] = str(path)
        if path.exists():
            payloads[key] = load_json(path)
        else:
            missing.append(str(path))
            payloads[key] = {}
    for key, relative in jsonl_specs.items():
        path = root / relative
        refs[key] = str(path)
        if path.exists():
            rows[key] = load_jsonl(path)
        else:
            missing.append(str(path))
            rows[key] = []
    return _EvidenceBundle(
        refs=refs,
        payloads=payloads,
        rows=rows,
        missing_refs=missing,
    )


def _build_node_signal(
    *,
    surface: Phase7GovernanceNodeSurface,
    bundle: _EvidenceBundle,
) -> tuple[Phase7GovernanceNodeSignalAdapter, Phase7GovernanceNodeSignalReceipt]:
    refs, families, metrics, slots = _node_evidence(surface.node_key, bundle)
    source_receipt_ids = _source_ids(bundle=bundle, families=families)
    source_artifact_refs = {family: bundle.refs[family] for family in families}
    lower_backed = bool(source_receipt_ids) and all(refs.get(family, False) for family in families)
    status = "ok" if lower_backed else "blocked"
    adapter_payload = {
        "surface_id": surface.surface_id,
        "node_key": surface.node_key,
        "families": families,
        "source_receipt_ids": source_receipt_ids,
    }
    adapter_id = stable_id("phase7_signal_adapter_node", adapter_payload)
    signal_key = f"{surface.node_key}_lower_wm_signal"
    confidence = _bounded_confidence(
        float(surface.confidence_prior)
        + (0.04 * min(len(source_receipt_ids), 5))
        + (0.02 * min(len(families), 5))
    )
    candidate_outputs = {
        "surface_output_refs": list(surface.output_refs),
        "signal_slots": mapping(slots),
        "hard_constraint_capable": bool(surface.hard_constraint_capable),
        "runtime_action": "shadow_feed_only_no_dispatch",
    }
    adapter = Phase7GovernanceNodeSignalAdapter(
        adapter_id=adapter_id,
        surface_id=surface.surface_id,
        node_key=surface.node_key,
        domain_key=surface.domain_key,
        source_artifact_refs=source_artifact_refs,
        source_receipt_ids=source_receipt_ids,
        source_receipt_families=families,
        metrics={
            **metrics,
            "source_artifact_count": float(len(families)),
            "source_receipt_count": float(len(source_receipt_ids)),
            "confidence": confidence,
            "live_dispatch_denied": 1.0,
            "hard_veto_dispatch_denied": 1.0,
            "training_denied": 1.0,
            "weights_denied": 1.0,
            "reward_mutation_denied": 1.0,
            "promotion_denied": 1.0,
        },
        signal_slots=slots,
        adapter_status=status,
        lower_wm_receipt_backed=lower_backed,
        denied_authority=list(DENIED_PHASE7_AUTHORITIES),
    )
    receipt = Phase7GovernanceNodeSignalReceipt(
        signal_id=stable_id(
            "phase7_governance_node_signal",
            {
                "adapter_id": adapter_id,
                "surface_id": surface.surface_id,
                "node_key": surface.node_key,
                "families": families,
            },
        ),
        adapter_id=adapter_id,
        surface_id=surface.surface_id,
        node_key=surface.node_key,
        domain_key=surface.domain_key,
        signal_key=signal_key,
        source_receipt_ids=source_receipt_ids,
        source_artifact_refs=source_artifact_refs,
        confidence=confidence,
        candidate_outputs=candidate_outputs,
        hard_constraint_candidate=bool(surface.hard_constraint_capable),
        lower_wm_receipt_backed=lower_backed,
        denied_authority=list(DENIED_PHASE7_AUTHORITIES),
    )
    return adapter, receipt


def _node_evidence(
    node_key: str,
    bundle: _EvidenceBundle,
) -> tuple[dict[str, bool], list[str], dict[str, float], dict[str, Any]]:
    p = bundle.payloads
    r = bundle.rows
    if node_key == "economic_allocation_governance":
        families = [
            "phase7_shadow_summary",
            "phase6_advisory_runtime_report",
            "phase6_closure_audit",
            "phase6_transport_eval_reports",
        ]
        summary = p["phase7_shadow_summary"]
        advisory = p["phase6_advisory_runtime_report"]
        closure = p["phase6_closure_audit"]
        metrics = {
            "mean_net_customer_rate": _number(summary, "mean_net_customer_rate"),
            "total_data_share_credit": _number(summary, "total_data_share_credit"),
            "transport_eval_report_count": _number(advisory, "eval_report_count"),
            "phase6_training_denied": float(not bool(closure.get("training_executed"))),
        }
        slots = {
            "allocation_signal": "shadow_econ_tensor_and_transport_quality",
            "budget_evidence": _first_id(advisory),
            "opportunity_cost_join": "phase7_shadow_runtime_summary",
        }
    elif node_key == "reward_integrity_governance":
        families = [
            "phase7_shadow_summary",
            "phase7_control_field_evals",
            "phase7_outcome_join_rows",
            "phase6_advisory_runtime_report",
        ]
        summary = p["phase7_shadow_summary"]
        eval_report = p["phase7_eval_report"]
        metrics = {
            "mean_reward_total": _number(summary, "mean_reward_total"),
            "control_field_eval_count": _number(eval_report, "control_field_eval_count"),
            "reward_mutation_denied": float(
                not bool(eval_report.get("reward_math_mutation"))
            ),
            "promotion_denied": float(not bool(eval_report.get("promotion_eligible"))),
        }
        slots = {
            "reward_integrity_signal": "reward_mutation_denied_with_outcome_slots",
            "exploit_suspicion_status": "awaiting_labeled_reward_hack_corpus",
            "reward_math_mutation": False,
        }
    elif node_key == "plausibility_geometry_governance":
        families = [
            "phase35_bipedal_readiness_audit",
            "phase35_balance_geometry_reports",
            "phase35_joint_vector_receipts",
            "phase4_mujoco_model_stress_receipts",
            "phase4_blocker_probe_report",
        ]
        audit = p["phase35_bipedal_readiness_audit"]
        blocker = p["phase4_blocker_probe_report"]
        metrics = {
            "balance_geometry_report_count": _number(
                audit, "balance_geometry_report_count"
            ),
            "joint_vector_validation_count": _number(
                audit, "joint_vector_validation_receipt_count"
            ),
            "mujoco_model_stress_success_count": _number(
                blocker, "mujoco_model_stress_success_count"
            ),
            "hardware_evidence_present": float(bool(blocker.get("hardware_executed"))),
        }
        slots = {
            "plausibility_signal": "bipedal_geometry_and_local_mujoco_model_probe",
            "geometry_consistency_ref": _first_row_id(
                r["phase35_balance_geometry_reports"]
            ),
            "sim_real_delta_status": "hardware_or_honest_sim_delta_missing",
        }
    elif node_key == "deployment_truth_governance":
        families = [
            "phase4_bringup_readiness_report",
            "phase4_runtime_bridge_report",
            "phase4_blocker_probe_report",
            "phase4_ros2_readiness_receipts",
            "phase4_blocker_probe_receipts",
        ]
        bringup = p["phase4_bringup_readiness_report"]
        runtime = p["phase4_runtime_bridge_report"]
        blocker = p["phase4_blocker_probe_report"]
        metrics = {
            "dependency_verified_count": _number(bringup, "dependency_verified_count"),
            "ros2_runtime_readiness_receipt_count": _number(
                runtime, "ros2_runtime_readiness_receipt_count"
            ),
            "succeeded_probe_count": _number(blocker, "succeeded_probe_count"),
            "hardware_evidence_present": float(bool(runtime.get("hardware_executed"))),
            "live_stream_observed": float(bool(runtime.get("live_stream_observed"))),
        }
        slots = {
            "deployment_truth_signal": "local_preflight_receipt_backed_no_live_claim",
            "provider_truth": False,
            "hardware_runtime_evidence": False,
            "runtime_readiness_hint": "local_preflight_ready_live_runtime_missing",
        }
    elif node_key == "safety_constraint_governance":
        families = [
            "phase4_downstream_controller_report",
            "phase4_controller_safety_receipts",
            "phase4_safety_preflight_receipts",
            "phase4_stale_validations",
            "phase4_watchdog_demotions",
            "phase4_safety_transitions",
            "phase4_safety_expansion_receipts",
        ]
        downstream = p["phase4_downstream_controller_report"]
        local = p["phase4_local_harness_report"]
        runtime = p["phase4_runtime_bridge_report"]
        metrics = {
            "controller_safety_receipt_count": _number(
                downstream, "safety_receipt_count"
            ),
            "stale_validation_receipt_count": _number(
                local, "stale_validation_receipt_count"
            ),
            "watchdog_demotion_receipt_count": _number(
                local, "watchdog_demotion_receipt_count"
            ),
            "safety_expansion_receipt_count": _number(
                runtime, "safety_envelope_expansion_receipt_count"
            ),
        }
        slots = {
            "safety_signal": "local_safety_preflight_stale_watchdog_receipts",
            "estop_state": "synthetic_latched_clear_drills_only",
            "degraded_mode_hint": "stable_base_mobile_manipulator_fallback_only",
        }
    elif node_key == "data_value_governance":
        families = [
            "phase7_shadow_summary",
            "phase6_advisory_runtime_report",
            "phase6_transport_eval_reports",
            "phase7_outcome_join_rows",
            "phase65_meta_node_trajectory_receipts",
        ]
        summary = p["phase7_shadow_summary"]
        advisory = p["phase6_advisory_runtime_report"]
        metrics = {
            "total_data_share_credit": _number(summary, "total_data_share_credit"),
            "joined_shadow_outcome_count": _number(
                advisory, "joined_shadow_outcome_count"
            ),
            "shadow_join_slot_count": _number(advisory, "shadow_join_slot_count"),
            "outcome_join_row_count": float(len(r["phase7_outcome_join_rows"])),
        }
        slots = {
            "data_value_signal": "datapack_credit_transport_eval_outcome_join",
            "collection_priority": "awaiting_counterfactual_value_label",
            "training_dispatch": "denied",
        }
    elif node_key == "embodiment_limit_governance":
        families = [
            "phase35_refit_report",
            "phase35_bipedal_readiness_audit",
            "phase35_joint_vector_receipts",
            "phase4_downstream_controller_report",
            "phase4_low_level_command_frames",
        ]
        refit = p["phase35_refit_report"]
        audit = p["phase35_bipedal_readiness_audit"]
        downstream = p["phase4_downstream_controller_report"]
        metrics = {
            "bipedal_chassis_joint_count": _number(refit, "bipedal_chassis_joint_count"),
            "joint_limit_envelope_count": _number(
                refit, "bipedal_chassis_joint_limit_envelope_count"
            ),
            "whole_body_replay_row_count": _number(audit, "whole_body_replay_row_count"),
            "command_frame_count": _number(downstream, "command_frame_count"),
        }
        slots = {
            "embodiment_signal": "canonical_bipedal_joint_frame_capacity_receipts",
            "primary_posture": "bipedal_whole_body",
            "fallback_posture": "stable_base_mobile_manipulator",
            "fixed_base_tabletop": "curriculum_regression_only",
        }
    elif node_key == "coordination_operator_governance":
        families = [
            "phase4_bringup_readiness_report",
            "phase4_operator_runbooks",
            "phase4_operator_drill_receipts",
            "phase4_operator_drill_transitions",
            "phase4_runtime_bridge_report",
        ]
        bringup = p["phase4_bringup_readiness_report"]
        runtime = p["phase4_runtime_bridge_report"]
        metrics = {
            "operator_recovery_runbook_count": _number(
                bringup, "operator_recovery_runbook_count"
            ),
            "operator_recovery_drill_receipt_count": _number(
                runtime, "operator_recovery_drill_receipt_count"
            ),
            "operator_recovery_scenario_count": _number(
                runtime, "operator_recovery_scenario_count"
            ),
            "live_policy_control": float(bool(runtime.get("live_policy_control"))),
        }
        slots = {
            "operator_signal": "runbook_and_synthetic_recovery_drill_receipts",
            "operator_handoff_candidate": "shadow_only_no_autonomy_preemption",
            "comms_qos": "local_contract_only_live_qos_missing",
        }
    else:
        families = []
        metrics = {}
        slots = {"signal_status": "unknown_node_key"}

    refs = {
        family: bool(bundle.refs.get(family))
        and bundle.refs[family] not in set(bundle.missing_refs)
        for family in families
    }
    return refs, families, metrics, slots


def _source_ids(*, bundle: _EvidenceBundle, families: Sequence[str]) -> list[str]:
    ids: list[str] = []
    for family in families:
        payload = bundle.payloads.get(family)
        if isinstance(payload, Mapping):
            ids.extend(_payload_ids(payload))
        for row in bundle.rows.get(family, []):
            ids.extend(_payload_ids(row))
    return sorted(dict.fromkeys(ids))


def _payload_ids(payload: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in (
        "report_id",
        "audit_id",
        "receipt_id",
        "row_id",
        "target_id",
        "signal_id",
        "eval_report_id",
        "contract_id",
        "node_id",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            ids.append(str(value))
    return ids


def _first_id(payload: Mapping[str, Any]) -> str:
    ids = _payload_ids(payload)
    return ids[0] if ids else ""


def _first_row_id(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        ids = _payload_ids(row)
        if ids:
            return ids[0]
    return ""


def _number(payload: Mapping[str, Any], key: str) -> float:
    try:
        return float(payload.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _bounded_confidence(value: float) -> float:
    return max(0.0, min(0.99, float(value)))


def save_phase7_governance_signal_adapters(
    output_dir: str | Path,
    report: Phase7GovernanceSignalAdapterReport,
    adapters: Sequence[Phase7GovernanceNodeSignalAdapter],
    signal_receipts: Sequence[Phase7GovernanceNodeSignalReceipt],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase7_governance_signal_adapter_report_v1.json",
        "adapters_path": output / "phase7_governance_node_signal_adapters_v1.jsonl",
        "signal_receipts_path": output
        / "phase7_governance_node_signal_receipts_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(paths["adapters_path"], [adapter.to_dict() for adapter in adapters])
    write_jsonl(
        paths["signal_receipts_path"],
        [receipt.to_dict() for receipt in signal_receipts],
    )
    return {key: str(value) for key, value in paths.items()}


def load_phase7_governance_signal_adapter_report(
    path: str | Path,
) -> Phase7GovernanceSignalAdapterReport:
    return Phase7GovernanceSignalAdapterReport.from_dict(load_json(path))


def load_phase7_governance_node_signal_adapters(
    path: str | Path,
) -> list[Phase7GovernanceNodeSignalAdapter]:
    return [
        Phase7GovernanceNodeSignalAdapter.from_dict(row) for row in load_jsonl(path)
    ]


def load_phase7_governance_node_signal_receipts(
    path: str | Path,
) -> list[Phase7GovernanceNodeSignalReceipt]:
    return [
        Phase7GovernanceNodeSignalReceipt.from_dict(row) for row in load_jsonl(path)
    ]


def load_phase7_signal_adapter_scaffold_surfaces(
    phase7_scaffold_dir: str | Path,
) -> list[Phase7GovernanceNodeSurface]:
    return load_phase7_governance_node_surfaces(
        Path(phase7_scaffold_dir) / "phase7_governance_node_surfaces_v1.jsonl"
    )
