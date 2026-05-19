"""Receipt contracts for the sim/synth/physics WM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .common import clip01, mapping, strings


@dataclass(frozen=True)
class PhysicsCalibrationReceipt:
    """Calibration/system-ID summary for one WM planning window or backend run."""

    receipt_id: str
    backend: str
    fidelity_tier: str
    calibration_profile: str
    quality_score: float
    domain_randomization_regime: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "physics_calibration_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "backend": self.backend,
            "fidelity_tier": self.fidelity_tier,
            "calibration_profile": self.calibration_profile,
            "quality_score": clip01(self.quality_score),
            "domain_randomization_regime": self.domain_randomization_regime,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class PhysicsAdaptationReceipt:
    """Randomization and system-ID receipt for one WM planning window."""

    receipt_id: str
    policy_id: str
    backend: str
    target_hardware_class: str
    domain_randomization_profile: str
    system_identification_profile: str
    readiness_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "physics_adaptation_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "policy_id": self.policy_id,
            "backend": self.backend,
            "target_hardware_class": self.target_hardware_class,
            "domain_randomization_profile": self.domain_randomization_profile,
            "system_identification_profile": self.system_identification_profile,
            "readiness_score": clip01(self.readiness_score),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendExecutionBindingReceipt:
    """Backend execution-binding receipt for one WM planning window."""

    receipt_id: str
    binding_id: str
    backend: str
    binding_status: str
    executor_entrypoint: str
    asset_profile: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_execution_binding_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "binding_id": self.binding_id,
            "backend": self.backend,
            "binding_status": self.binding_status,
            "executor_entrypoint": self.executor_entrypoint,
            "asset_profile": self.asset_profile,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendShadowExecutionReceipt:
    """Receipt for WM-owned backend shadow execution/materialization."""

    receipt_id: str
    backend: str
    execution_mode: str
    execution_status: str
    episode_ids: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_shadow_execution_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "execution_status": self.execution_status,
            "episode_ids": strings(self.episode_ids),
            "artifact_refs": strings(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendRuntimeExecutionReceipt:
    """Receipt for WM-owned concrete backend runtime execution or bound request."""

    receipt_id: str
    backend: str
    execution_mode: str
    execution_status: str
    policy_id: str = ""
    artifact_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_runtime_execution_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "execution_status": self.execution_status,
            "policy_id": self.policy_id,
            "artifact_refs": strings(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendRuntimeLaunchReceipt:
    """Receipt for WM-owned external runtime launch preparation or execution."""

    receipt_id: str
    backend: str
    launch_profile: str
    launch_status: str
    executed: bool = False
    command: str = ""
    cwd: str = ""
    artifact_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_runtime_launch_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "backend": self.backend,
            "launch_profile": self.launch_profile,
            "launch_status": self.launch_status,
            "executed": bool(self.executed),
            "command": self.command,
            "cwd": self.cwd,
            "artifact_refs": strings(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendRuntimeAdapterReceipt:
    """Receipt for executable-adapter mediation over a backend runtime lane."""

    receipt_id: str
    backend: str
    adapter_family: str
    adapter_entrypoint: str
    consumer_mode: str
    adapter_status: str
    execution_path: str
    executed: bool = False
    artifact_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_runtime_adapter_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "backend": self.backend,
            "adapter_family": self.adapter_family,
            "adapter_entrypoint": self.adapter_entrypoint,
            "consumer_mode": self.consumer_mode,
            "adapter_status": self.adapter_status,
            "execution_path": self.execution_path,
            "executed": bool(self.executed),
            "artifact_refs": strings(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendRuntimeOutcomeReceipt:
    """Receipt for harvested outputs from an external backend runtime launch."""

    receipt_id: str
    backend: str
    outcome_profile: str
    outcome_status: str
    executed: bool = False
    harvested_output_count: int = 0
    artifact_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_runtime_outcome_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "backend": self.backend,
            "outcome_profile": self.outcome_profile,
            "outcome_status": self.outcome_status,
            "executed": bool(self.executed),
            "harvested_output_count": int(self.harvested_output_count),
            "artifact_refs": strings(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendRuntimeBridgeReceipt:
    """Receipt for one WM-owned backend runtime bridge contract."""

    receipt_id: str
    bridge_id: str
    backend: str
    bridge_status: str
    execution_authority: str
    transport_profile: str
    planner_rate_hz: float
    control_rate_hz: float
    observation_rate_hz: float
    action_decimation: int
    latency_budget_ms: float
    bridge_readiness_score: float
    action_contracts: list[str] = field(default_factory=list)
    observation_contracts: list[str] = field(default_factory=list)
    telemetry_contracts: list[str] = field(default_factory=list)
    safety_channels: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_runtime_bridge_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "bridge_id": self.bridge_id,
            "backend": self.backend,
            "bridge_status": self.bridge_status,
            "execution_authority": self.execution_authority,
            "transport_profile": self.transport_profile,
            "planner_rate_hz": float(self.planner_rate_hz),
            "control_rate_hz": float(self.control_rate_hz),
            "observation_rate_hz": float(self.observation_rate_hz),
            "action_decimation": int(self.action_decimation),
            "latency_budget_ms": float(self.latency_budget_ms),
            "bridge_readiness_score": clip01(self.bridge_readiness_score),
            "action_contracts": strings(self.action_contracts),
            "observation_contracts": strings(self.observation_contracts),
            "telemetry_contracts": strings(self.telemetry_contracts),
            "safety_channels": strings(self.safety_channels),
            "artifact_refs": strings(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendRuntimeWorkOrderReceipt:
    """Work order for a concrete backend runtime bring-up or validation pass."""

    receipt_id: str
    backend: str
    bridge_id: str
    work_order_kind: str
    status: str
    linked_backlog_ids: list[str] = field(default_factory=list)
    command_hints: list[str] = field(default_factory=list)
    missing_runtime_targets: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    missing_preconditions: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_runtime_work_order_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "backend": self.backend,
            "bridge_id": self.bridge_id,
            "work_order_kind": self.work_order_kind,
            "status": self.status,
            "linked_backlog_ids": strings(self.linked_backlog_ids),
            "command_hints": strings(self.command_hints),
            "missing_runtime_targets": strings(self.missing_runtime_targets),
            "missing_assets": strings(self.missing_assets),
            "missing_preconditions": strings(self.missing_preconditions),
            "artifact_refs": strings(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class RobotAssetContractReceipt:
    """Receipt for one WM-owned robot-asset contract."""

    receipt_id: str
    contract_id: str
    asset_profile: str
    target_hardware_class: str
    readiness_score: float
    required_assets: list[str] = field(default_factory=list)
    available_assets: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    calibration_contracts: list[str] = field(default_factory=list)
    observation_contracts: list[str] = field(default_factory=list)
    action_contracts: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "robot_asset_contract_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "contract_id": self.contract_id,
            "asset_profile": self.asset_profile,
            "target_hardware_class": self.target_hardware_class,
            "readiness_score": clip01(self.readiness_score),
            "required_assets": strings(self.required_assets),
            "available_assets": strings(self.available_assets),
            "missing_assets": strings(self.missing_assets),
            "calibration_contracts": strings(self.calibration_contracts),
            "observation_contracts": strings(self.observation_contracts),
            "action_contracts": strings(self.action_contracts),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SimRealGapReceipt:
    """Estimated sim-real transfer gap for the current planning window."""

    receipt_id: str
    source_backend: str
    target_hardware_class: str
    comparison_scope: str
    gap_score: float
    realism_confidence: float
    status: str
    branch_plan_ids: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "sim_real_gap_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "source_backend": self.source_backend,
            "target_hardware_class": self.target_hardware_class,
            "comparison_scope": self.comparison_scope,
            "gap_score": clip01(self.gap_score),
            "realism_confidence": clip01(self.realism_confidence),
            "status": self.status,
            "branch_plan_ids": strings(self.branch_plan_ids),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class TaskMeasurementReceipt:
    """Live measurement receipt emitted from the task-definition contract."""

    receipt_id: str
    surface_id: str
    task_definition_contract_id: str
    task_family: str
    benchmark_gate_ready: bool
    measurement_values: Dict[str, float] = field(default_factory=dict)
    measurement_status: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "task_measurement_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "surface_id": self.surface_id,
            "task_definition_contract_id": self.task_definition_contract_id,
            "task_family": self.task_family,
            "benchmark_gate_ready": bool(self.benchmark_gate_ready),
            "measurement_values": {
                str(key): float(value) for key, value in self.measurement_values.items()
            },
            "measurement_status": {
                str(key): str(value) for key, value in self.measurement_status.items()
            },
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendMismatchReceipt:
    """Backend-to-backend mismatch estimate for the active runtime route."""

    receipt_id: str
    reference_backend: str
    candidate_backend: str
    mismatch_score: float
    calibration_staleness_score: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_mismatch_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "mismatch_score": clip01(self.mismatch_score),
            "calibration_staleness_score": clip01(self.calibration_staleness_score),
            "status": self.status,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SurrogatePhysicsReceipt:
    """Receipt for the advisory surrogate-physics lane."""

    receipt_id: str
    provider_id: str
    forecast_scope: str
    forecast_status: str
    surrogate_confidence: float
    branch_plan_ids: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "surrogate_physics_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "provider_id": self.provider_id,
            "forecast_scope": self.forecast_scope,
            "forecast_status": self.forecast_status,
            "surrogate_confidence": clip01(self.surrogate_confidence),
            "branch_plan_ids": strings(self.branch_plan_ids),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SurrogateCalibrationReceipt:
    """Receipt for surrogate-vs-backend calibration posture."""

    receipt_id: str
    provider_id: str
    reference_backend: str
    calibration_status: str
    calibration_score: float
    staleness_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "surrogate_calibration_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "provider_id": self.provider_id,
            "reference_backend": self.reference_backend,
            "calibration_status": self.calibration_status,
            "calibration_score": clip01(self.calibration_score),
            "staleness_score": clip01(self.staleness_score),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BranchValidityReceipt:
    """Per-branch validity / reject-filter receipt for training admission."""

    receipt_id: str
    branch_plan_id: str
    job_id: str
    validity_score: float
    admission_score: float
    admissible: bool
    evidence_status: str
    reject_reasons: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "branch_validity_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "branch_plan_id": self.branch_plan_id,
            "job_id": self.job_id,
            "validity_score": clip01(self.validity_score),
            "admission_score": clip01(self.admission_score),
            "admissible": bool(self.admissible),
            "evidence_status": self.evidence_status,
            "reject_reasons": strings(self.reject_reasons),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SensorAlignmentReceipt:
    """CPU-local camera geometry / sensor-alignment receipt."""

    receipt_id: str
    scene_hierarchy_id: str
    sensor_profile: str
    alignment_score: float
    status: str
    checks: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "sensor_alignment_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "scene_hierarchy_id": self.scene_hierarchy_id,
            "sensor_profile": self.sensor_profile,
            "alignment_score": clip01(self.alignment_score),
            "status": self.status,
            "checks": {str(key): str(value) for key, value in self.checks.items()},
            "metrics": {str(key): float(value) for key, value in self.metrics.items()},
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class ReplayValidityReceipt:
    """Per-branch replay/task-consistency receipt for training admission."""

    receipt_id: str
    branch_plan_id: str
    outcome_receipt_id: str
    validity_score: float
    task_consistency_score: float
    transfer_consistency_score: float
    status: str
    reject_reasons: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "replay_validity_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "branch_plan_id": self.branch_plan_id,
            "outcome_receipt_id": self.outcome_receipt_id,
            "validity_score": clip01(self.validity_score),
            "task_consistency_score": clip01(self.task_consistency_score),
            "transfer_consistency_score": clip01(self.transfer_consistency_score),
            "status": self.status,
            "reject_reasons": strings(self.reject_reasons),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class Gen2SimAdmissionReceipt:
    """Receipt for one WM-owned gen2sim admission decision surface."""

    receipt_id: str
    admission_id: str
    benchmark_gate_ready: bool
    admissible_branch_ids: list[str] = field(default_factory=list)
    blocked_branch_ids: list[str] = field(default_factory=list)
    selection_policy: str = "receipt_gated"
    rationale: str = ""
    inferential_learnability_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "gen2sim_admission_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "admission_id": self.admission_id,
            "benchmark_gate_ready": bool(self.benchmark_gate_ready),
            "admissible_branch_ids": strings(self.admissible_branch_ids),
            "blocked_branch_ids": strings(self.blocked_branch_ids),
            "selection_policy": self.selection_policy,
            "rationale": self.rationale,
            "inferential_learnability_summary": mapping(
                self.inferential_learnability_summary
            ),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class RenderProviderReceipt:
    """Materialization receipt for one WM-owned branch/render provider selection."""

    receipt_id: str
    branch_plan_id: str
    provider_id: str
    provider_kind: str
    provider_status: str
    render_mode: str
    counterfactual_mode: str
    materialization_status: str = ""
    materialization_mode: str = ""
    materialization_entrypoint: str = ""
    artifact_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "render_provider_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "branch_plan_id": self.branch_plan_id,
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "provider_status": self.provider_status,
            "render_mode": self.render_mode,
            "counterfactual_mode": self.counterfactual_mode,
            "materialization_status": self.materialization_status,
            "materialization_mode": self.materialization_mode,
            "materialization_entrypoint": self.materialization_entrypoint,
            "artifact_refs": strings(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SimulationOutcomeReceipt:
    """Canonical outcome receipt for one executed simulation or synth branch."""

    receipt_id: str
    job_id: str
    branch_plan_id: str
    status: str
    replay_refs: list[str] = field(default_factory=list)
    event_refs: list[str] = field(default_factory=list)
    governance_refs: list[str] = field(default_factory=list)
    training_feedback_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "simulation_outcome_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "job_id": self.job_id,
            "branch_plan_id": self.branch_plan_id,
            "status": self.status,
            "replay_refs": strings(self.replay_refs),
            "event_refs": strings(self.event_refs),
            "governance_refs": strings(self.governance_refs),
            "training_feedback_refs": strings(self.training_feedback_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }
