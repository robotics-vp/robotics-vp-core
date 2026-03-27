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
