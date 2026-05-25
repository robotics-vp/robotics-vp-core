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
from src.world_model.humanoid_readiness.unitree_bringup_readiness import (
    Phase4UnitreeBringupReadinessReport,
)
from src.world_model.humanoid_readiness.unitree_local_harness import (
    Phase4UnitreeLocalHarnessReport,
)
from src.world_model.humanoid_readiness.unitree_runtime_bridge import (
    Phase4UnitreeRuntimeEvidenceBridgeReport,
)
from src.world_model.humanoid_readiness.unitree_blocker_probes import (
    Phase4UnitreeBlockerStressProbeReport,
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
    "unitree_runtime_dependency_build_and_interface_verification_missing",
    "real_unitree_stream_command_timing_and_recovery_runtime_missing",
    "unitree_physical_safety_and_estop_recovery_drills_missing",
    "deployment_grade_unitree_sim_or_hardware_runtime_evidence_missing",
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
    phase4_unitree_bringup_readiness_report_id: str
    phase4_unitree_local_harness_report_id: str
    phase4_unitree_runtime_bridge_report_id: str
    phase4_unitree_blocker_stress_probe_report_id: str
    phase65_report_id: str
    status: str
    local_phase35_complete: bool
    local_phase35_bipedal_readiness_complete: bool
    local_phase4_complete: bool
    local_phase4_downstream_controller_complete: bool
    local_phase4_unitree_bringup_readiness_complete: bool
    local_phase4_unitree_local_harness_complete: bool
    local_phase4_unitree_runtime_bridge_complete: bool
    local_phase4_unitree_blocker_stress_probe_complete: bool
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
            "phase4_unitree_bringup_readiness_report_id": (
                self.phase4_unitree_bringup_readiness_report_id
            ),
            "phase4_unitree_local_harness_report_id": (
                self.phase4_unitree_local_harness_report_id
            ),
            "phase4_unitree_runtime_bridge_report_id": (
                self.phase4_unitree_runtime_bridge_report_id
            ),
            "phase4_unitree_blocker_stress_probe_report_id": (
                self.phase4_unitree_blocker_stress_probe_report_id
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
            "local_phase4_unitree_bringup_readiness_complete": bool(
                self.local_phase4_unitree_bringup_readiness_complete
            ),
            "local_phase4_unitree_local_harness_complete": bool(
                self.local_phase4_unitree_local_harness_complete
            ),
            "local_phase4_unitree_runtime_bridge_complete": bool(
                self.local_phase4_unitree_runtime_bridge_complete
            ),
            "local_phase4_unitree_blocker_stress_probe_complete": bool(
                self.local_phase4_unitree_blocker_stress_probe_complete
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
            phase4_unitree_bringup_readiness_report_id=str(
                payload.get("phase4_unitree_bringup_readiness_report_id", "")
            ),
            phase4_unitree_local_harness_report_id=str(
                payload.get("phase4_unitree_local_harness_report_id", "")
            ),
            phase4_unitree_runtime_bridge_report_id=str(
                payload.get("phase4_unitree_runtime_bridge_report_id", "")
            ),
            phase4_unitree_blocker_stress_probe_report_id=str(
                payload.get("phase4_unitree_blocker_stress_probe_report_id", "")
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
            local_phase4_unitree_bringup_readiness_complete=bool(
                payload.get("local_phase4_unitree_bringup_readiness_complete", False)
            ),
            local_phase4_unitree_local_harness_complete=bool(
                payload.get("local_phase4_unitree_local_harness_complete", False)
            ),
            local_phase4_unitree_runtime_bridge_complete=bool(
                payload.get("local_phase4_unitree_runtime_bridge_complete", False)
            ),
            local_phase4_unitree_blocker_stress_probe_complete=bool(
                payload.get(
                    "local_phase4_unitree_blocker_stress_probe_complete", False
                )
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
    phase4_unitree_bringup_readiness_report: Phase4UnitreeBringupReadinessReport,
    phase4_unitree_local_harness_report: Phase4UnitreeLocalHarnessReport,
    phase4_unitree_runtime_bridge_report: Phase4UnitreeRuntimeEvidenceBridgeReport,
    phase4_unitree_blocker_stress_probe_report: Phase4UnitreeBlockerStressProbeReport,
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
    phase4_unitree_bringup_readiness_complete = (
        phase4_unitree_bringup_readiness_report.status == "ok"
        and phase4_unitree_bringup_readiness_report.local_pre_purchase_prepared
        and phase4_unitree_bringup_readiness_report.all_block_receipts_emitted
        and phase4_unitree_bringup_readiness_report.dependency_discovery_complete
        and phase4_unitree_bringup_readiness_report.asset_joint_subset_aligned
        and phase4_unitree_bringup_readiness_report.stream_contracts_present
        and phase4_unitree_bringup_readiness_report.command_conformance_dry_run_ready
        and phase4_unitree_bringup_readiness_report.local_timing_probe_present
        and phase4_unitree_bringup_readiness_report.physical_safety_preflight_present
        and phase4_unitree_bringup_readiness_report.operator_recovery_runbook_present
        and not phase4_unitree_bringup_readiness_report.honest_sim_or_hardware_evidence_present
        and not phase4_unitree_bringup_readiness_report.hardware_dispatch_enabled
        and not phase4_unitree_bringup_readiness_report.ros2_publish_attempted
        and not phase4_unitree_bringup_readiness_report.unitree_sdk2_write_enabled
        and not phase4_unitree_bringup_readiness_report.g1pilot_runtime_invoked
        and not phase4_unitree_bringup_readiness_report.honest_sim_executed
        and not phase4_unitree_bringup_readiness_report.hardware_executed
        and not phase4_unitree_bringup_readiness_report.live_policy_control
        and not phase4_unitree_bringup_readiness_report.training_executed
        and not phase4_unitree_bringup_readiness_report.weights_written
        and not phase4_unitree_bringup_readiness_report.provider_executed
        and not phase4_unitree_bringup_readiness_report.reward_math_mutation
        and not phase4_unitree_bringup_readiness_report.promotion_eligible
    )
    phase4_unitree_local_harness_complete = (
        phase4_unitree_local_harness_report.status == "ok"
        and phase4_unitree_local_harness_report.local_harnesses_complete
        and phase4_unitree_local_harness_report.trace_stream_harness_complete
        and phase4_unitree_local_harness_report.command_shape_harness_complete
        and phase4_unitree_local_harness_report.mock_timing_watchdog_harness_complete
        and phase4_unitree_local_harness_report.safety_recovery_harness_complete
        and phase4_unitree_local_harness_report.runtime_preflight_harness_complete
        and not phase4_unitree_local_harness_report.live_stream_observed
        and not phase4_unitree_local_harness_report.ros2_publish_attempted
        and not phase4_unitree_local_harness_report.unitree_sdk2_write_enabled
        and not phase4_unitree_local_harness_report.g1pilot_runtime_invoked
        and not phase4_unitree_local_harness_report.mujoco_launch_executed
        and not phase4_unitree_local_harness_report.ros2_launch_executed
        and not phase4_unitree_local_harness_report.hardware_executed
        and not phase4_unitree_local_harness_report.training_executed
        and not phase4_unitree_local_harness_report.weights_written
        and not phase4_unitree_local_harness_report.reward_math_mutation
        and not phase4_unitree_local_harness_report.promotion_eligible
    )
    phase4_unitree_runtime_bridge_complete = (
        phase4_unitree_runtime_bridge_report.status == "ok"
        and phase4_unitree_runtime_bridge_report.local_runtime_evidence_bridge_complete
        and phase4_unitree_runtime_bridge_report.ros2_runtime_preflight_complete
        and phase4_unitree_runtime_bridge_report.mujoco_headless_trace_attempt_complete
        and phase4_unitree_runtime_bridge_report.trace_ingestion_adapters_complete
        and phase4_unitree_runtime_bridge_report.safety_envelope_expansion_complete
        and phase4_unitree_runtime_bridge_report.operator_drill_runner_complete
        and not phase4_unitree_runtime_bridge_report.live_stream_observed
        and not phase4_unitree_runtime_bridge_report.ros2_publish_attempted
        and not phase4_unitree_runtime_bridge_report.unitree_sdk2_write_enabled
        and not phase4_unitree_runtime_bridge_report.g1pilot_runtime_invoked
        and not phase4_unitree_runtime_bridge_report.hardware_executed
        and not phase4_unitree_runtime_bridge_report.live_policy_control
        and not phase4_unitree_runtime_bridge_report.training_executed
        and not phase4_unitree_runtime_bridge_report.weights_written
        and not phase4_unitree_runtime_bridge_report.reward_math_mutation
        and not phase4_unitree_runtime_bridge_report.promotion_eligible
    )
    phase4_unitree_blocker_stress_probe_complete = (
        phase4_unitree_blocker_stress_probe_report.status == "ok"
        and phase4_unitree_blocker_stress_probe_report.local_phase4_probe_expansion_complete
        and phase4_unitree_blocker_stress_probe_report.all_local_probe_attempts_complete
        and phase4_unitree_blocker_stress_probe_report.probe_receipt_count >= 1
        and phase4_unitree_blocker_stress_probe_report.mujoco_model_stress_receipt_count >= 1
        and not phase4_unitree_blocker_stress_probe_report.live_stream_observed
        and not phase4_unitree_blocker_stress_probe_report.ros2_publish_attempted
        and not phase4_unitree_blocker_stress_probe_report.unitree_sdk2_write_enabled
        and not phase4_unitree_blocker_stress_probe_report.g1pilot_runtime_invoked
        and not phase4_unitree_blocker_stress_probe_report.hardware_executed
        and not phase4_unitree_blocker_stress_probe_report.live_policy_control
        and not phase4_unitree_blocker_stress_probe_report.training_executed
        and not phase4_unitree_blocker_stress_probe_report.weights_written
        and not phase4_unitree_blocker_stress_probe_report.reward_math_mutation
        and not phase4_unitree_blocker_stress_probe_report.promotion_eligible
    )
    phase65_complete = phase65_report.local_meta_node_scaffold_complete
    all_complete = (
        phase35_complete
        and phase35_bipedal_readiness_complete
        and phase4_complete
        and phase4_downstream_controller_complete
        and phase4_unitree_bringup_readiness_complete
        and phase4_unitree_local_harness_complete
        and phase4_unitree_runtime_bridge_complete
        and phase4_unitree_blocker_stress_probe_complete
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
        "phase4_unitree_dependency_inventory_receipts",
        "phase4_unitree_asset_joint_conformance_receipts",
        "phase4_unitree_stream_and_command_contracts",
        "phase4_unitree_local_timing_probe_receipts",
        "phase4_unitree_physical_safety_preflight_receipts",
        "phase4_unitree_operator_estop_recovery_runbooks",
        "phase4_unitree_sim_hardware_evidence_ledger",
        "phase4_unitree_lowstate_imu_estop_contact_trace_harness",
        "phase4_unitree_command_shape_validation_harness",
        "phase4_unitree_mock_timing_watchdog_harness",
        "phase4_unitree_safety_recovery_state_machine_harness",
        "phase4_unitree_mujoco_ros2_preflight_receipts",
        "phase4_unitree_ros2_colcon_runtime_readiness_receipts",
        "phase4_unitree_mujoco_headless_step_trace",
        "phase4_unitree_rosbag2_mcap_trace_ingestion_adapters",
        "phase4_unitree_expanded_safety_envelope_receipts",
        "phase4_unitree_scripted_operator_recovery_drills",
        "phase4_unitree_blocker_stress_probe_receipts",
        "phase4_unitree_multi_model_mujoco_stress_receipts",
        "phase4_unitree_static_g1pilot_policy_isaac_lerobot_probe_receipts",
        "phase4_unitree_compile_only_dds_sdk2_probe_receipts",
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
        "phase4_unitree_bringup_readiness_report_id": (
            phase4_unitree_bringup_readiness_report.report_id
        ),
        "phase4_unitree_local_harness_report_id": (
            phase4_unitree_local_harness_report.report_id
        ),
        "phase4_unitree_runtime_bridge_report_id": (
            phase4_unitree_runtime_bridge_report.report_id
        ),
        "phase4_unitree_blocker_stress_probe_report_id": (
            phase4_unitree_blocker_stress_probe_report.report_id
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
        phase4_unitree_bringup_readiness_report_id=(
            phase4_unitree_bringup_readiness_report.report_id
        ),
        phase4_unitree_local_harness_report_id=(
            phase4_unitree_local_harness_report.report_id
        ),
        phase4_unitree_runtime_bridge_report_id=(
            phase4_unitree_runtime_bridge_report.report_id
        ),
        phase4_unitree_blocker_stress_probe_report_id=(
            phase4_unitree_blocker_stress_probe_report.report_id
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
        local_phase4_unitree_bringup_readiness_complete=(
            phase4_unitree_bringup_readiness_complete
        ),
        local_phase4_unitree_local_harness_complete=(
            phase4_unitree_local_harness_complete
        ),
        local_phase4_unitree_runtime_bridge_complete=(
            phase4_unitree_runtime_bridge_complete
        ),
        local_phase4_unitree_blocker_stress_probe_complete=(
            phase4_unitree_blocker_stress_probe_complete
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
