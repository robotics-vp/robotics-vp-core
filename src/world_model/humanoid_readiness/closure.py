"""Integrated local closure audit for Phase 3.5, Phase 4, and Phase 6.5."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.world_model.humanoid_readiness.common import (
    denied_gate_map,
    load_json,
    mapping,
    stable_id,
    strings,
    write_json,
)
from src.world_model.humanoid_readiness.phase35 import HumanoidPhase35RefitReport
from src.world_model.humanoid_readiness.phase4 import (
    Phase4DeploymentEnablerSweepReport,
)
from src.world_model.humanoid_readiness.downstream_controller import (
    Phase4DownstreamControllerScaffoldReport,
)
from src.world_model.humanoid_readiness.phase65 import (
    Phase65MetaNodeNeuralizationReport,
)
from src.world_model.embodiment_actuation.bipedal_readiness import (
    Phase35BipedalReadinessAudit,
)

PHASE35465_CLOSURE_AUDIT_VERSION = "phase35_4_65_local_closure_audit_v1"

PHASE35465_REMAINING_BLOCKERS = (
    "unitree_assets_backend_runtime_and_calibration_missing",
    "live_streams_control_interfaces_and_timing_jitter_missing",
    "companion_middleware_and_operator_recovery_runtime_missing",
    "counterfactual_meta_node_corpus_density_missing",
    "gpu_training_provider_hardware_evidence_missing",
    "promotion_grade_humanoid_governance_benchmarks_missing",
)


@dataclass(frozen=True)
class Phase35465LocalClosureAudit:
    """Audit that the three local phases are structurally closed."""

    audit_id: str
    phase35_report_id: str
    phase35_bipedal_readiness_audit_id: str
    phase4_report_id: str
    phase4_downstream_controller_report_id: str
    phase65_report_id: str
    status: str
    local_phase35_complete: bool
    local_phase35_bipedal_readiness_complete: bool
    local_phase4_complete: bool
    local_phase4_downstream_controller_complete: bool
    local_phase65_complete: bool
    all_local_structures_complete: bool
    ready_for_phase7_scaffold: bool
    phase7_authority_granted: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=denied_gate_map)
    closed_local_surfaces: list[str] = field(default_factory=list)
    remaining_evidence_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE35465_CLOSURE_AUDIT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "version": self.version,
            "phase35_report_id": self.phase35_report_id,
            "phase35_bipedal_readiness_audit_id": (
                self.phase35_bipedal_readiness_audit_id
            ),
            "phase4_report_id": self.phase4_report_id,
            "phase4_downstream_controller_report_id": (
                self.phase4_downstream_controller_report_id
            ),
            "phase65_report_id": self.phase65_report_id,
            "status": self.status,
            "local_phase35_complete": bool(self.local_phase35_complete),
            "local_phase35_bipedal_readiness_complete": bool(
                self.local_phase35_bipedal_readiness_complete
            ),
            "local_phase4_complete": bool(self.local_phase4_complete),
            "local_phase4_downstream_controller_complete": bool(
                self.local_phase4_downstream_controller_complete
            ),
            "local_phase65_complete": bool(self.local_phase65_complete),
            "all_local_structures_complete": bool(
                self.all_local_structures_complete
            ),
            "ready_for_phase7_scaffold": bool(self.ready_for_phase7_scaffold),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": denied_gate_map(self.denied_gates),
            "closed_local_surfaces": list(self.closed_local_surfaces),
            "remaining_evidence_blockers": list(self.remaining_evidence_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase35465LocalClosureAudit":
        return cls(
            audit_id=str(payload.get("audit_id", "")),
            phase35_report_id=str(payload.get("phase35_report_id", "")),
            phase35_bipedal_readiness_audit_id=str(
                payload.get("phase35_bipedal_readiness_audit_id", "")
            ),
            phase4_report_id=str(payload.get("phase4_report_id", "")),
            phase4_downstream_controller_report_id=str(
                payload.get("phase4_downstream_controller_report_id", "")
            ),
            phase65_report_id=str(payload.get("phase65_report_id", "")),
            status=str(payload.get("status", "blocked")),
            local_phase35_complete=bool(payload.get("local_phase35_complete", False)),
            local_phase35_bipedal_readiness_complete=bool(
                payload.get("local_phase35_bipedal_readiness_complete", False)
            ),
            local_phase4_complete=bool(payload.get("local_phase4_complete", False)),
            local_phase4_downstream_controller_complete=bool(
                payload.get("local_phase4_downstream_controller_complete", False)
            ),
            local_phase65_complete=bool(payload.get("local_phase65_complete", False)),
            all_local_structures_complete=bool(
                payload.get("all_local_structures_complete", False)
            ),
            ready_for_phase7_scaffold=bool(
                payload.get("ready_for_phase7_scaffold", False)
            ),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
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
            denied_gates=denied_gate_map(payload.get("denied_gates")),
            closed_local_surfaces=strings(payload.get("closed_local_surfaces")),
            remaining_evidence_blockers=strings(
                payload.get("remaining_evidence_blockers")
            ),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(payload.get("version", PHASE35465_CLOSURE_AUDIT_VERSION)),
        )


def build_phase35465_local_closure_audit(
    *,
    phase35_report: HumanoidPhase35RefitReport,
    phase35_bipedal_readiness_audit: Phase35BipedalReadinessAudit,
    phase4_report: Phase4DeploymentEnablerSweepReport,
    phase4_downstream_controller_report: Phase4DownstreamControllerScaffoldReport,
    phase65_report: Phase65MetaNodeNeuralizationReport,
    artifact_refs: Mapping[str, Any] | None = None,
) -> Phase35465LocalClosureAudit:
    phase35_complete = phase35_report.local_structural_refit_complete
    phase35_bipedal_readiness_complete = (
        phase35_bipedal_readiness_audit.status == "ok"
        and phase35_bipedal_readiness_audit.phase35_no_gpu_no_hardware_prepared
        and phase35_bipedal_readiness_audit.local_asset_ingestion_contract_present
        and phase35_bipedal_readiness_audit.kinematic_validators_present
        and phase35_bipedal_readiness_audit.whole_body_replay_row_count >= 1
        and not phase35_bipedal_readiness_audit.ready_for_unitree_runtime
        and not phase35_bipedal_readiness_audit.ready_for_training
        and not phase35_bipedal_readiness_audit.hardware_calibrated_limits
        and not phase35_bipedal_readiness_audit.unitree_sim_runtime_executed
        and not phase35_bipedal_readiness_audit.training_executed
        and not phase35_bipedal_readiness_audit.promotion_eligible
    )
    phase4_complete = phase4_report.local_non_hardware_scaffold_complete
    phase4_downstream_controller_complete = (
        phase4_downstream_controller_report.status == "ok"
        and phase4_downstream_controller_report.local_downstream_controller_scaffold_complete
        and phase4_downstream_controller_report.unitree_bridge_contract_present
        and phase4_downstream_controller_report.g1pilot_fallback_contract_present
        and phase4_downstream_controller_report.dry_run_controller_present
        and not phase4_downstream_controller_report.hardware_dispatch_enabled
        and not phase4_downstream_controller_report.ros2_publish_attempted
        and not phase4_downstream_controller_report.unitree_sdk2_write_enabled
        and not phase4_downstream_controller_report.g1pilot_runtime_invoked
        and not phase4_downstream_controller_report.live_policy_control
        and not phase4_downstream_controller_report.promotion_eligible
    )
    phase65_complete = phase65_report.local_meta_node_scaffold_complete
    all_complete = (
        phase35_complete
        and phase35_bipedal_readiness_complete
        and phase4_complete
        and phase4_downstream_controller_complete
        and phase65_complete
    )
    closed_surfaces = [
        "phase35_capacity_bands",
        "phase35_humanoid_observation_action_schema_deltas",
        "phase35_posture_tagged_env_taxonomy",
        "phase35_unitree_sim_target_contract",
        "phase35_humanoid_benchmark_taxonomy",
        "phase35_bipedal_asset_intake_and_parse_receipts",
        "phase35_bipedal_kinematic_consistency_validation",
        "phase35_bipedal_joint_vector_validation_receipts",
        "phase35_bipedal_balance_geometry_reports",
        "phase35_whole_body_replay_row_slots",
        "phase4a_control_loop_separation_contracts",
        "phase4_downstream_controller_bridge_targets",
        "phase4_downstream_controller_modes",
        "phase4_dry_run_command_frames",
        "phase4_controller_safety_and_dispatch_receipts",
        "phase4e_companion_compute_comms_contracts",
        "phase4f_operator_teleop_recovery_contracts",
        "phase4b_4c_4d_explicit_stubs",
        "phase65_meta_node_state",
        "phase65_trajectory_and_intervention_receipts",
        "phase65_counterfactual_target_rows",
        "phase65_robustness_reports",
        "phase65_denied_promotion_gates",
    ]
    audit_payload = {
        "phase35_report_id": phase35_report.report_id,
        "phase35_bipedal_readiness_audit_id": (
            phase35_bipedal_readiness_audit.audit_id
        ),
        "phase4_report_id": phase4_report.report_id,
        "phase4_downstream_controller_report_id": (
            phase4_downstream_controller_report.report_id
        ),
        "phase65_report_id": phase65_report.report_id,
        "all_local_structures_complete": all_complete,
        "artifact_refs": mapping(artifact_refs),
    }
    return Phase35465LocalClosureAudit(
        audit_id=stable_id("phase35465_closure", audit_payload),
        phase35_report_id=phase35_report.report_id,
        phase35_bipedal_readiness_audit_id=(
            phase35_bipedal_readiness_audit.audit_id
        ),
        phase4_report_id=phase4_report.report_id,
        phase4_downstream_controller_report_id=(
            phase4_downstream_controller_report.report_id
        ),
        phase65_report_id=phase65_report.report_id,
        status="ok" if all_complete else "blocked",
        local_phase35_complete=phase35_complete,
        local_phase35_bipedal_readiness_complete=(
            phase35_bipedal_readiness_complete
        ),
        local_phase4_complete=phase4_complete,
        local_phase4_downstream_controller_complete=(
            phase4_downstream_controller_complete
        ),
        local_phase65_complete=phase65_complete,
        all_local_structures_complete=all_complete,
        ready_for_phase7_scaffold=all_complete,
        denied_gates=denied_gate_map(),
        closed_local_surfaces=closed_surfaces if all_complete else [],
        remaining_evidence_blockers=list(PHASE35465_REMAINING_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )


def save_phase35465_local_closure_audit(
    path: str | Path,
    audit: Phase35465LocalClosureAudit,
) -> None:
    write_json(path, audit.to_dict())


def load_phase35465_local_closure_audit(
    path: str | Path,
) -> Phase35465LocalClosureAudit:
    return Phase35465LocalClosureAudit.from_dict(load_json(path))
