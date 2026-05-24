"""Phase 4 local non-hardware deployment-enabler sweep surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.world_model.humanoid_readiness.common import (
    denied_gate_map,
    load_json,
    load_jsonl,
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.phase35 import HumanoidPhase35RefitReport

PHASE4_SWEEP_REPORT_VERSION = "humanoid_phase4_deployment_enabler_sweep_report_v1"
PHASE4_CONTRACT_SURFACE_VERSION = "humanoid_phase4_contract_surface_v1"
PHASE4_STUB_SURFACE_VERSION = "humanoid_phase4_stub_surface_v1"

PHASE4_REMAINING_BLOCKERS = (
    "live_streams_missing",
    "actual_control_interfaces_missing",
    "timing_jitter_traces_missing",
    "companion_compute_middleware_not_measured",
    "operator_teleop_recovery_traces_missing",
    "hardware_or_honest_sim_runtime_evidence_missing",
)


@dataclass(frozen=True)
class Phase4ContractSurface:
    """Replay/training-aware Phase 4 contract surface."""

    surface_id: str
    phase_key: str
    contract_name: str
    purpose: str
    posture_tag: str
    later_evidence_required: list[str] = field(default_factory=list)
    denied_authority: list[str] = field(default_factory=list)
    robot_asset_refs: list[str] = field(default_factory=list)
    observation_schema_refs: list[str] = field(default_factory=list)
    action_schema_refs: list[str] = field(default_factory=list)
    timing_refs: list[str] = field(default_factory=list)
    placement_refs: list[str] = field(default_factory=list)
    degraded_mode_reason: str = ""
    event_spine_ref: str = "event_spine_ref_required"
    governance_trace_ref: str = "governance_trace_ref_required"
    replay_export_posture: str = "sidecar_planning_only"
    promotion_posture: str = "denied_pending_runtime_evidence"
    authority_class: str = "phase4_contract_surface_only"
    version: str = PHASE4_CONTRACT_SURFACE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "version": self.version,
            "phase_key": self.phase_key,
            "contract_name": self.contract_name,
            "purpose": self.purpose,
            "posture_tag": self.posture_tag,
            "later_evidence_required": list(self.later_evidence_required),
            "denied_authority": list(self.denied_authority),
            "robot_asset_refs": list(self.robot_asset_refs),
            "observation_schema_refs": list(self.observation_schema_refs),
            "action_schema_refs": list(self.action_schema_refs),
            "timing_refs": list(self.timing_refs),
            "placement_refs": list(self.placement_refs),
            "degraded_mode_reason": self.degraded_mode_reason,
            "event_spine_ref": self.event_spine_ref,
            "governance_trace_ref": self.governance_trace_ref,
            "replay_export_posture": self.replay_export_posture,
            "promotion_posture": self.promotion_posture,
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase4ContractSurface":
        return cls(
            surface_id=str(payload.get("surface_id", "")),
            phase_key=str(payload.get("phase_key", "")),
            contract_name=str(payload.get("contract_name", "")),
            purpose=str(payload.get("purpose", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            later_evidence_required=strings(payload.get("later_evidence_required")),
            denied_authority=strings(payload.get("denied_authority")),
            robot_asset_refs=strings(payload.get("robot_asset_refs")),
            observation_schema_refs=strings(payload.get("observation_schema_refs")),
            action_schema_refs=strings(payload.get("action_schema_refs")),
            timing_refs=strings(payload.get("timing_refs")),
            placement_refs=strings(payload.get("placement_refs")),
            degraded_mode_reason=str(payload.get("degraded_mode_reason", "")),
            event_spine_ref=str(
                payload.get("event_spine_ref", "event_spine_ref_required")
            ),
            governance_trace_ref=str(
                payload.get("governance_trace_ref", "governance_trace_ref_required")
            ),
            replay_export_posture=str(
                payload.get("replay_export_posture", "sidecar_planning_only")
            ),
            promotion_posture=str(
                payload.get("promotion_posture", "denied_pending_runtime_evidence")
            ),
            authority_class=str(
                payload.get("authority_class", "phase4_contract_surface_only")
            ),
            version=str(payload.get("version", PHASE4_CONTRACT_SURFACE_VERSION)),
        )


@dataclass(frozen=True)
class Phase4StubSurface:
    """Explicit planning-only stub for Phase 4B/4C/4D local work."""

    stub_id: str
    phase_key: str
    stub_name: str
    purpose: str
    full_closure_waits_for: list[str] = field(default_factory=list)
    denied_authority: list[str] = field(default_factory=list)
    explicit_stub: bool = True
    planning_only: bool = True
    promotion_posture: str = "stub_not_promotable"
    version: str = PHASE4_STUB_SURFACE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "stub_id": self.stub_id,
            "version": self.version,
            "phase_key": self.phase_key,
            "stub_name": self.stub_name,
            "purpose": self.purpose,
            "full_closure_waits_for": list(self.full_closure_waits_for),
            "denied_authority": list(self.denied_authority),
            "explicit_stub": bool(self.explicit_stub),
            "planning_only": bool(self.planning_only),
            "promotion_posture": self.promotion_posture,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase4StubSurface":
        return cls(
            stub_id=str(payload.get("stub_id", "")),
            phase_key=str(payload.get("phase_key", "")),
            stub_name=str(payload.get("stub_name", "")),
            purpose=str(payload.get("purpose", "")),
            full_closure_waits_for=strings(payload.get("full_closure_waits_for")),
            denied_authority=strings(payload.get("denied_authority")),
            explicit_stub=bool(payload.get("explicit_stub", True)),
            planning_only=bool(payload.get("planning_only", True)),
            promotion_posture=str(payload.get("promotion_posture", "stub_not_promotable")),
            version=str(payload.get("version", PHASE4_STUB_SURFACE_VERSION)),
        )


@dataclass(frozen=True)
class Phase4DeploymentEnablerSweepReport:
    """Top-level local Phase 4 sweep report."""

    report_id: str
    phase35_report_id: str
    status: str
    contract_surface_count: int
    stub_surface_count: int
    local_non_hardware_scaffold_complete: bool
    ready_for_phase65_local_meta_nodes: bool
    phase_counts: dict[str, int] = field(default_factory=dict)
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=denied_gate_map)
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE4_SWEEP_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase35_report_id": self.phase35_report_id,
            "status": self.status,
            "contract_surface_count": int(self.contract_surface_count),
            "stub_surface_count": int(self.stub_surface_count),
            "local_non_hardware_scaffold_complete": bool(
                self.local_non_hardware_scaffold_complete
            ),
            "ready_for_phase65_local_meta_nodes": bool(
                self.ready_for_phase65_local_meta_nodes
            ),
            "phase_counts": {str(k): int(v) for k, v in self.phase_counts.items()},
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": denied_gate_map(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase4DeploymentEnablerSweepReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase35_report_id=str(payload.get("phase35_report_id", "")),
            status=str(payload.get("status", "blocked")),
            contract_surface_count=int(payload.get("contract_surface_count", 0) or 0),
            stub_surface_count=int(payload.get("stub_surface_count", 0) or 0),
            local_non_hardware_scaffold_complete=bool(
                payload.get("local_non_hardware_scaffold_complete", False)
            ),
            ready_for_phase65_local_meta_nodes=bool(
                payload.get("ready_for_phase65_local_meta_nodes", False)
            ),
            phase_counts={
                str(k): int(v)
                for k, v in dict(payload.get("phase_counts", {}) or {}).items()
            },
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
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(payload.get("version", PHASE4_SWEEP_REPORT_VERSION)),
        )


def build_phase4_deployment_enabler_sweep(
    phase35_report: HumanoidPhase35RefitReport,
    artifact_refs: Mapping[str, Any] | None = None,
) -> tuple[
    Phase4DeploymentEnablerSweepReport,
    list[Phase4ContractSurface],
    list[Phase4StubSurface],
]:
    denied = [
        "live_actuator_authority",
        "companion_control_authority",
        "operator_loop_runtime_claim",
        "reward_math_mutation",
        "promotion",
    ]
    contract_specs = [
        (
            "4A",
            "ControlLoopRateContract",
            "Names servo/reflex, whole-body fast, WM-slow, and offline rates.",
            ["measured_loop_timing", "jitter_histogram"],
        ),
        (
            "4A",
            "ServoReflexBoundaryReceipt",
            "Separates robot-local reflex work from slow governance/economics.",
            ["runtime_control_interface_evidence"],
        ),
        (
            "4A",
            "SlowLoopCommandEnvelope",
            "Bounds slow WM/economic/meta requests to non-servo envelopes.",
            ["actual_action_interface_integration"],
        ),
        (
            "4A",
            "WatchdogDegradationReceipt",
            "Records stale data, battery, thermal, and comms degradation.",
            ["live_watchdog_trace"],
        ),
        (
            "4A",
            "AuthoritySplitManifest",
            "Denies slow-loop live authority until safety/timing gates pass.",
            ["rollback_and_demotion_tests"],
        ),
        (
            "4E",
            "ComputePlacementContract",
            "Names onboard, companion, cloud, and offline placement classes.",
            ["measured_runtime_placement_trace"],
        ),
        (
            "4E",
            "CommsQoSReceipt",
            "Records latency, stale-data age, packet loss, and freshness.",
            ["ros2_dds_or_unitree_middleware_evidence"],
        ),
        (
            "4E",
            "CompanionOffloadEnvelope",
            "Bounds offload work without hard real-time authority.",
            ["timing_and_failure_mode_tests"],
        ),
        (
            "4E",
            "BatteryThermalComputeReceipt",
            "Joins compute spend to reserve and thermal headroom.",
            ["live_battery_thermal_telemetry"],
        ),
        (
            "4E",
            "DegradedLinkRunbook",
            "Names fallback behavior under stale or lost companion links.",
            ["operator_recovery_and_watchdog_trace"],
        ),
        (
            "4F",
            "OperatorHandoffContract",
            "Names when operator intervention is requested or required.",
            ["teleop_runtime_trace"],
        ),
        (
            "4F",
            "TeleopSessionReceipt",
            "Captures command timing, authority, and replay refs.",
            ["real_or_sim_teleop_session"],
        ),
        (
            "4F",
            "RecoveryTraceReceipt",
            "Records recovery cause, action, posture, and outcome.",
            ["sim_or_hardware_recovery_drill"],
        ),
        (
            "4F",
            "FallbackAuthorityGate",
            "Separates stable-base fallback from bipedal promotion.",
            ["safety_benchmark_evidence"],
        ),
        (
            "4F",
            "PostmortemReplayExport",
            "Makes recovery traces replay/training-aware.",
            ["replay_export_validation"],
        ),
    ]
    contracts = [
        Phase4ContractSurface(
            surface_id=f"phase4_{phase}_{name}",
            phase_key=phase,
            contract_name=name,
            purpose=purpose,
            posture_tag="bipedal_whole_body",
            later_evidence_required=evidence,
            denied_authority=denied,
            robot_asset_refs=["unitree_robot_asset_ref_required"],
            observation_schema_refs=["humanoid_observation_schema_ref_required"],
            action_schema_refs=["humanoid_action_schema_ref_required"],
            timing_refs=["control_loop_timing_ref_required"],
            placement_refs=["compute_placement_ref_required"],
            degraded_mode_reason="must_be_explicit_when_active",
        )
        for phase, name, purpose, evidence in contract_specs
    ]
    stub_specs = [
        (
            "4B",
            "SensorFusionInputSchemaStub",
            "Names camera/depth/IMU/proprio/force streams and timestamp expectations.",
            ["live_streams", "calibration", "timestamp_sync_evidence"],
        ),
        (
            "4C",
            "PhysicalSafetyEnvelopeStub",
            "Names joint limits, self-collision, e-stop, and fall protection.",
            ["actual_safety_interface", "hardware_or_sim_safety_tests"],
        ),
        (
            "4D",
            "SpatialStateInterfaceStub",
            "Names localization, map, nav state, and degraded spatial state.",
            ["slam_backend_runtime", "mobile_sim_or_hardware_evidence"],
        ),
    ]
    stubs = [
        Phase4StubSurface(
            stub_id=f"phase4_{phase}_{name}",
            phase_key=phase,
            stub_name=name,
            purpose=purpose,
            full_closure_waits_for=evidence,
            denied_authority=denied,
        )
        for phase, name, purpose, evidence in stub_specs
    ]
    phase_counts: dict[str, int] = {}
    for surface in contracts:
        phase_counts[surface.phase_key] = phase_counts.get(surface.phase_key, 0) + 1
    for stub in stubs:
        phase_counts[stub.phase_key] = phase_counts.get(stub.phase_key, 0) + 1
    complete = (
        phase35_report.local_structural_refit_complete
        and phase_counts.get("4A", 0) >= 5
        and phase_counts.get("4E", 0) >= 5
        and phase_counts.get("4F", 0) >= 5
        and phase_counts.get("4B", 0) >= 1
        and phase_counts.get("4C", 0) >= 1
        and phase_counts.get("4D", 0) >= 1
    )
    report_payload = {
        "phase35_report_id": phase35_report.report_id,
        "contract_surface_count": len(contracts),
        "stub_surface_count": len(stubs),
        "phase_counts": phase_counts,
        "artifact_refs": mapping(artifact_refs),
    }
    report = Phase4DeploymentEnablerSweepReport(
        report_id=stable_id("phase4_sweep", report_payload),
        phase35_report_id=phase35_report.report_id,
        status="ok" if complete else "blocked",
        contract_surface_count=len(contracts),
        stub_surface_count=len(stubs),
        local_non_hardware_scaffold_complete=complete,
        ready_for_phase65_local_meta_nodes=complete,
        phase_counts=phase_counts,
        denied_gates=denied_gate_map(),
        remaining_blockers=list(PHASE4_REMAINING_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return report, contracts, stubs


def save_phase4_deployment_enabler_sweep(
    output_dir: str | Path,
    report: Phase4DeploymentEnablerSweepReport,
    contracts: list[Phase4ContractSurface],
    stubs: list[Phase4StubSurface],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "humanoid_phase4_deployment_enabler_sweep_report_v1.json",
        "contracts_path": output / "humanoid_phase4_contract_surfaces_v1.jsonl",
        "stubs_path": output / "humanoid_phase4_stub_surfaces_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(paths["contracts_path"], [item.to_dict() for item in contracts])
    write_jsonl(paths["stubs_path"], [item.to_dict() for item in stubs])
    return {key: str(value) for key, value in paths.items()}


def load_phase4_deployment_enabler_sweep_report(
    path: str | Path,
) -> Phase4DeploymentEnablerSweepReport:
    return Phase4DeploymentEnablerSweepReport.from_dict(load_json(path))


def load_phase4_contract_surfaces(path: str | Path) -> list[Phase4ContractSurface]:
    return [Phase4ContractSurface.from_dict(row) for row in load_jsonl(path)]


def load_phase4_stub_surfaces(path: str | Path) -> list[Phase4StubSurface]:
    return [Phase4StubSurface.from_dict(row) for row in load_jsonl(path)]
