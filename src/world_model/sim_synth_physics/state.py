"""Typed state objects for the sim/synth/physics world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .agenda import SimulationAgenda
from .common import clip01, mapping, strings


@dataclass(frozen=True)
class PhysicsContextState:
    """Canonical backend/fidelity state for the current planning window."""

    context_id: str
    backend: str
    fidelity_tier: str
    timestep_ms: float
    domain_randomization_regime: str
    calibration_profile: str
    selection_policy: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "physics_context_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "backend": self.backend,
            "fidelity_tier": self.fidelity_tier,
            "timestep_ms": float(self.timestep_ms),
            "domain_randomization_regime": self.domain_randomization_regime,
            "calibration_profile": self.calibration_profile,
            "selection_policy": self.selection_policy,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class PhysicsAdaptationPolicyState:
    """Canonical domain-randomization and system-ID policy for the current window."""

    policy_id: str
    backend: str
    simulator_family: str
    target_hardware_class: str
    robot_asset_profile: str
    domain_randomization_profile: str
    system_identification_profile: str
    randomization_axes: list[str] = field(default_factory=list)
    calibration_targets: list[str] = field(default_factory=list)
    selection_policy: str = "heuristic_only"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "physics_adaptation_policy_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "backend": self.backend,
            "simulator_family": self.simulator_family,
            "target_hardware_class": self.target_hardware_class,
            "robot_asset_profile": self.robot_asset_profile,
            "domain_randomization_profile": self.domain_randomization_profile,
            "system_identification_profile": self.system_identification_profile,
            "randomization_axes": strings(self.randomization_axes),
            "calibration_targets": strings(self.calibration_targets),
            "selection_policy": self.selection_policy,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BackendExecutionBindingState:
    """Concrete execution-binding state for the selected backend."""

    binding_id: str
    backend: str
    binding_name: str
    binding_status: str
    executor_entrypoint: str
    executor_kind: str
    observation_adapter_entrypoint: str
    target_runtime_stack: list[str] = field(default_factory=list)
    asset_profile: str = ""
    required_assets: list[str] = field(default_factory=list)
    available_assets: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_execution_binding_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "backend": self.backend,
            "binding_name": self.binding_name,
            "binding_status": self.binding_status,
            "executor_entrypoint": self.executor_entrypoint,
            "executor_kind": self.executor_kind,
            "observation_adapter_entrypoint": self.observation_adapter_entrypoint,
            "target_runtime_stack": strings(self.target_runtime_stack),
            "asset_profile": self.asset_profile,
            "required_assets": strings(self.required_assets),
            "available_assets": strings(self.available_assets),
            "missing_assets": strings(self.missing_assets),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class RobotAssetContractState:
    """Canonical robot-asset contract for backend/runtime execution."""

    contract_id: str
    asset_profile: str
    target_hardware_class: str
    required_assets: list[str] = field(default_factory=list)
    available_assets: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    calibration_contracts: list[str] = field(default_factory=list)
    observation_contracts: list[str] = field(default_factory=list)
    action_contracts: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "robot_asset_contract_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "asset_profile": self.asset_profile,
            "target_hardware_class": self.target_hardware_class,
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
class BackendRuntimeBridgeState:
    """Canonical slow-loop to backend-runtime bridge contract."""

    bridge_id: str
    backend: str
    bridge_status: str
    transport_profile: str
    transport_stack: list[str] = field(default_factory=list)
    required_runtime_targets: list[str] = field(default_factory=list)
    ready_runtime_targets: list[str] = field(default_factory=list)
    missing_runtime_targets: list[str] = field(default_factory=list)
    planner_rate_hz: float = 0.0
    control_rate_hz: float = 0.0
    observation_rate_hz: float = 0.0
    action_decimation: int = 1
    latency_budget_ms: float = 0.0
    bridge_readiness_score: float = 0.0
    action_contracts: list[str] = field(default_factory=list)
    observation_contracts: list[str] = field(default_factory=list)
    telemetry_contracts: list[str] = field(default_factory=list)
    safety_channels: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "backend_runtime_bridge_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "backend": self.backend,
            "bridge_status": self.bridge_status,
            "transport_profile": self.transport_profile,
            "transport_stack": strings(self.transport_stack),
            "required_runtime_targets": strings(self.required_runtime_targets),
            "ready_runtime_targets": strings(self.ready_runtime_targets),
            "missing_runtime_targets": strings(self.missing_runtime_targets),
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
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class DiffusionConditioningState:
    """Governed diffusion/render-conditioning state derived from WM state."""

    conditioning_id: str
    objective_preset: str
    env_backend: str
    semantic_tags: list[str] = field(default_factory=list)
    branch_job_ids: list[str] = field(default_factory=list)
    admissible_branch_ids: list[str] = field(default_factory=list)
    blocked_branch_ids: list[str] = field(default_factory=list)
    governed_modes: list[str] = field(default_factory=list)
    render_budget: int = 0
    prompt_hints: Dict[str, Any] = field(default_factory=dict)
    routing_context: Dict[str, Any] = field(default_factory=dict)
    inferential_learnability_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "diffusion_conditioning_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conditioning_id": self.conditioning_id,
            "objective_preset": self.objective_preset,
            "env_backend": self.env_backend,
            "semantic_tags": strings(self.semantic_tags),
            "branch_job_ids": strings(self.branch_job_ids),
            "admissible_branch_ids": strings(self.admissible_branch_ids),
            "blocked_branch_ids": strings(self.blocked_branch_ids),
            "governed_modes": strings(self.governed_modes),
            "render_budget": int(self.render_budget),
            "prompt_hints": mapping(self.prompt_hints),
            "routing_context": mapping(self.routing_context),
            "inferential_learnability_summary": mapping(self.inferential_learnability_summary),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class BranchRenderProviderState:
    """Typed provider contract for materializing one synthetic branch."""

    provider_id: str
    provider_kind: str
    provider_status: str
    render_mode: str
    counterfactual_mode: str
    ggds_mode: str
    materialization_status: str = ""
    materialization_entrypoint: str = ""
    provider_config: Dict[str, Any] = field(default_factory=dict)
    fallback_provider: str = ""
    fallback_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "branch_render_provider_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "provider_status": self.provider_status,
            "render_mode": self.render_mode,
            "counterfactual_mode": self.counterfactual_mode,
            "ggds_mode": self.ggds_mode,
            "materialization_status": self.materialization_status,
            "materialization_entrypoint": self.materialization_entrypoint,
            "provider_config": mapping(self.provider_config),
            "fallback_provider": self.fallback_provider,
            "fallback_reason": self.fallback_reason,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SyntheticBranchPlan:
    """Typed synthetic-branch plan compiled from the WM agenda."""

    plan_id: str
    source_job_id: str
    branch_family: str
    generation_mode: str
    render_backend: str
    gap_target_refs: list[Dict[str, Any]] = field(default_factory=list)
    admission_preconditions: Dict[str, Any] = field(default_factory=dict)
    expected_yield_score: float = 0.0
    selection_policy: str = "heuristic_only"
    render_provider: Optional[BranchRenderProviderState] = None
    inferential_learnability_contract: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "synthetic_branch_plan_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source_job_id": self.source_job_id,
            "branch_family": self.branch_family,
            "generation_mode": self.generation_mode,
            "render_backend": self.render_backend,
            "gap_target_refs": [mapping(item) for item in self.gap_target_refs],
            "admission_preconditions": mapping(self.admission_preconditions),
            "expected_yield_score": clip01(self.expected_yield_score),
            "selection_policy": self.selection_policy,
            "render_provider": (
                self.render_provider.to_dict() if self.render_provider is not None else None
            ),
            "inferential_learnability_contract": mapping(
                self.inferential_learnability_contract
            ),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class Gen2SimAdmissionState:
    """Admission summary for WM-owned synthetic plans."""

    admission_id: str
    benchmark_gate_ready: bool
    admissible_branch_ids: list[str] = field(default_factory=list)
    blocked_branch_ids: list[str] = field(default_factory=list)
    selection_policy: str = "receipt_gated"
    rationale: str = ""
    inferential_learnability_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "gen2sim_admission_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
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
class SimSynthPhysicsWorldState:
    """Top-level canonical state for the sim/synth/physics WM."""

    state_id: str
    simulation_agenda: SimulationAgenda
    physics_context: PhysicsContextState
    physics_adaptation_policy: Optional[PhysicsAdaptationPolicyState] = None
    backend_execution_binding: Optional[BackendExecutionBindingState] = None
    robot_asset_contract: Optional[RobotAssetContractState] = None
    backend_runtime_bridge: Optional[BackendRuntimeBridgeState] = None
    synthetic_branch_plans: list[SyntheticBranchPlan] = field(default_factory=list)
    gen2sim_admission: Optional[Gen2SimAdmissionState] = None
    diffusion_conditioning: Optional[DiffusionConditioningState] = None
    input_context: Dict[str, Any] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "sim_synth_physics_world_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "simulation_agenda": self.simulation_agenda.to_dict(),
            "physics_context": self.physics_context.to_dict(),
            "physics_adaptation_policy": (
                self.physics_adaptation_policy.to_dict()
                if self.physics_adaptation_policy is not None
                else None
            ),
            "backend_execution_binding": (
                self.backend_execution_binding.to_dict()
                if self.backend_execution_binding is not None
                else None
            ),
            "robot_asset_contract": (
                self.robot_asset_contract.to_dict()
                if self.robot_asset_contract is not None
                else None
            ),
            "backend_runtime_bridge": (
                self.backend_runtime_bridge.to_dict()
                if self.backend_runtime_bridge is not None
                else None
            ),
            "synthetic_branch_plans": [plan.to_dict() for plan in self.synthetic_branch_plans],
            "gen2sim_admission": (
                self.gen2sim_admission.to_dict() if self.gen2sim_admission is not None else None
            ),
            "diffusion_conditioning": (
                self.diffusion_conditioning.to_dict() if self.diffusion_conditioning is not None else None
            ),
            "input_context": mapping(self.input_context),
            "artifact_refs": mapping(self.artifact_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }
