"""Phase 6.5 local meta-node trainer and loss scaffolds.

This module materializes dataset, loss, model-config, and CPU smoke-forward
contracts for the Phase 6.5 meta-node surfaces. It intentionally does not
instantiate trainable weights, run training, mutate reward math, or grant Phase
7 authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.world_model.humanoid_readiness.common import (
    denied_gate_map,
    float_mapping,
    load_json,
    mapping,
    stable_id,
    strings,
    write_json,
)
from src.world_model.humanoid_readiness.phase65 import (
    PHASE65_REMAINING_BLOCKERS,
    MetaNodeCounterfactualTarget,
    MetaNodeInterventionReceipt,
    MetaNodePromotionGate,
    MetaNodeRobustnessReport,
    MetaNodeState,
    MetaNodeTrajectoryReceipt,
    Phase65MetaNodeNeuralizationReport,
)

PHASE65_TRAINER_DATASET_CONTRACT_VERSION = (
    "phase65_meta_node_trainer_dataset_contract_v1"
)
PHASE65_LOSS_DEFINITION_VERSION = "phase65_meta_node_loss_definition_v1"
PHASE65_MODEL_COMPONENT_CONFIG_VERSION = "phase65_meta_node_model_component_config_v1"
PHASE65_CPU_SMOKE_FORWARD_VERSION = "phase65_meta_node_cpu_smoke_forward_v1"
PHASE65_TRAINER_SCAFFOLD_MANIFEST_VERSION = (
    "phase65_meta_node_trainer_scaffold_manifest_v1"
)

PHASE65_TRAINER_BLOCKERS = (
    *PHASE65_REMAINING_BLOCKERS,
    "gpu_meta_node_training_not_run",
    "meta_node_weights_not_written",
    "phase7_control_wm_authority_not_granted",
    "promotion_grade_governance_benchmark_missing",
)

PHASE65_FEATURE_KEYS = (
    "confidence_prior",
    "activation_strength_prior",
    "neighbor_count",
    "input_ref_count",
    "target_ref_count",
    "trajectory_event_count",
    "intervention_kind_id",
    "robustness_surface_completeness",
    "activation_calibration_evidence",
    "neighbor_consistency_benchmark_evidence",
    "deployment_robustness_evidence",
    "promotion_denied",
)

PHASE65_TARGET_KEYS = (
    "activation_timing",
    "activation_strength",
    "intervention_kind",
    "target_selection",
    "counterfactual_downstream_improvement",
    "governance_satisfaction",
    "rollback_sensitivity",
    "neighbor_consistency",
    "robustness_replay_shift",
)


def _phase65_trainer_denied_gates(
    extra: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    gates = denied_gate_map(
        {
            "phase7_authority_granted": False,
            "phase7_control_wm_authority": False,
            "live_dispatch_allowed": False,
            "hard_veto_dispatch": False,
            "meta_node_weights_initialized": False,
            **dict(extra or {}),
        }
    )
    return gates


@dataclass(frozen=True)
class Phase65MetaNodeTrainerDatasetContract:
    dataset_contract_id: str
    phase65_report_id: str
    row_count: int
    feature_keys: list[str]
    target_keys: list[str]
    feature_dim: int
    target_dim: int
    row_family_counts: dict[str, int] = field(default_factory=dict)
    source_artifact_refs: dict[str, Any] = field(default_factory=dict)
    authority_class: str = "phase65_meta_node_dataset_contract_only"
    ready_for_cpu_smoke_forward: bool = False
    ready_for_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE65_TRAINER_DATASET_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_contract_id": self.dataset_contract_id,
            "version": self.version,
            "phase65_report_id": self.phase65_report_id,
            "row_count": int(self.row_count),
            "row_family_counts": {
                str(key): int(value) for key, value in self.row_family_counts.items()
            },
            "feature_keys": list(self.feature_keys),
            "target_keys": list(self.target_keys),
            "shape_contracts": {
                "feature_dim": int(self.feature_dim),
                "target_dim": int(self.target_dim),
                "row_count": int(self.row_count),
            },
            "source_artifact_refs": mapping(self.source_artifact_refs),
            "authority_class": self.authority_class,
            "ready_for_cpu_smoke_forward": bool(self.ready_for_cpu_smoke_forward),
            "ready_for_training": bool(self.ready_for_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase65MetaNodeTrainerDatasetContract":
        shapes = mapping(payload.get("shape_contracts"))
        return cls(
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            phase65_report_id=str(payload.get("phase65_report_id", "")),
            row_count=int(payload.get("row_count", shapes.get("row_count", 0)) or 0),
            row_family_counts={
                str(key): int(value)
                for key, value in dict(payload.get("row_family_counts") or {}).items()
            },
            feature_keys=strings(payload.get("feature_keys")),
            target_keys=strings(payload.get("target_keys")),
            feature_dim=int(shapes.get("feature_dim", 0) or 0),
            target_dim=int(shapes.get("target_dim", 0) or 0),
            source_artifact_refs=mapping(payload.get("source_artifact_refs")),
            authority_class=str(
                payload.get(
                    "authority_class", "phase65_meta_node_dataset_contract_only"
                )
            ),
            ready_for_cpu_smoke_forward=bool(
                payload.get("ready_for_cpu_smoke_forward", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=strings(payload.get("blockers")),
            metadata=mapping(payload.get("metadata")),
            version=str(
                payload.get("version", PHASE65_TRAINER_DATASET_CONTRACT_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase65MetaNodeLossDefinition:
    loss_id: str
    loss_key: str
    role: str
    row_families: list[str] = field(default_factory=list)
    target_keys: list[str] = field(default_factory=list)
    formula: str = "defined_not_optimized"
    default_weight: float = 1.0
    optimization_status: str = "defined_not_optimized"
    uses_rl_style_signal: bool = False
    direct_policy_rl: bool = False
    authority_class: str = "phase65_meta_node_loss_definition_only"
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE65_LOSS_DEFINITION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss_id": self.loss_id,
            "version": self.version,
            "loss_key": self.loss_key,
            "role": self.role,
            "row_families": list(self.row_families),
            "target_keys": list(self.target_keys),
            "formula": self.formula,
            "default_weight": float(self.default_weight),
            "optimization_status": self.optimization_status,
            "uses_rl_style_signal": bool(self.uses_rl_style_signal),
            "direct_policy_rl": bool(self.direct_policy_rl),
            "authority_class": self.authority_class,
            "blockers": list(self.blockers),
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase65MetaNodeLossDefinition":
        return cls(
            loss_id=str(payload.get("loss_id", "")),
            loss_key=str(payload.get("loss_key", "")),
            role=str(payload.get("role", "")),
            row_families=strings(payload.get("row_families")),
            target_keys=strings(payload.get("target_keys")),
            formula=str(payload.get("formula", "defined_not_optimized")),
            default_weight=float(payload.get("default_weight", 1.0) or 1.0),
            optimization_status=str(
                payload.get("optimization_status", "defined_not_optimized")
            ),
            uses_rl_style_signal=bool(payload.get("uses_rl_style_signal", False)),
            direct_policy_rl=bool(payload.get("direct_policy_rl", False)),
            authority_class=str(
                payload.get(
                    "authority_class", "phase65_meta_node_loss_definition_only"
                )
            ),
            blockers=strings(payload.get("blockers")),
            metadata=mapping(payload.get("metadata")),
            version=str(payload.get("version", PHASE65_LOSS_DEFINITION_VERSION)),
        )


@dataclass(frozen=True)
class Phase65MetaNodeModelComponentConfig:
    model_config_id: str
    dataset_contract_id: str
    components: list[dict[str, Any]] = field(default_factory=list)
    component_count: int = 0
    training_executed: bool = False
    weights_initialized: bool = False
    weights_written: bool = False
    ready_for_gpu_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    authority_class: str = "phase65_meta_node_model_config_only"
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE65_MODEL_COMPONENT_CONFIG_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config_id": self.model_config_id,
            "version": self.version,
            "dataset_contract_id": self.dataset_contract_id,
            "component_count": int(self.component_count),
            "components": [mapping(component) for component in self.components],
            "training_executed": bool(self.training_executed),
            "weights_initialized": bool(self.weights_initialized),
            "weights_written": bool(self.weights_written),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "authority_class": self.authority_class,
            "blockers": list(self.blockers),
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase65MetaNodeModelComponentConfig":
        return cls(
            model_config_id=str(payload.get("model_config_id", "")),
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            components=[
                mapping(item) for item in list(payload.get("components") or [])
            ],
            component_count=int(payload.get("component_count", 0) or 0),
            training_executed=bool(payload.get("training_executed", False)),
            weights_initialized=bool(payload.get("weights_initialized", False)),
            weights_written=bool(payload.get("weights_written", False)),
            ready_for_gpu_training=bool(payload.get("ready_for_gpu_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            authority_class=str(
                payload.get("authority_class", "phase65_meta_node_model_config_only")
            ),
            blockers=strings(payload.get("blockers")),
            metadata=mapping(payload.get("metadata")),
            version=str(payload.get("version", PHASE65_MODEL_COMPONENT_CONFIG_VERSION)),
        )


@dataclass(frozen=True)
class Phase65MetaNodeCPUSmokeForwardReceipt:
    smoke_forward_id: str
    dataset_contract_id: str
    model_config_id: str
    input_shape: list[int]
    hidden_shape: list[int]
    output_shapes: dict[str, list[int]]
    loss_keys_checked: list[str]
    smoke_forward_passed: bool
    authority_class: str = "phase65_cpu_smoke_forward_shape_check_only"
    training_executed: bool = False
    weights_initialized: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    live_policy_control: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE65_CPU_SMOKE_FORWARD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoke_forward_id": self.smoke_forward_id,
            "version": self.version,
            "dataset_contract_id": self.dataset_contract_id,
            "model_config_id": self.model_config_id,
            "input_shape": [int(value) for value in self.input_shape],
            "hidden_shape": [int(value) for value in self.hidden_shape],
            "output_shapes": {
                str(key): [int(value) for value in values]
                for key, values in self.output_shapes.items()
            },
            "loss_keys_checked": list(self.loss_keys_checked),
            "smoke_forward_passed": bool(self.smoke_forward_passed),
            "authority_class": self.authority_class,
            "training_executed": bool(self.training_executed),
            "weights_initialized": bool(self.weights_initialized),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "live_policy_control": bool(self.live_policy_control),
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase65MetaNodeCPUSmokeForwardReceipt":
        return cls(
            smoke_forward_id=str(payload.get("smoke_forward_id", "")),
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            model_config_id=str(payload.get("model_config_id", "")),
            input_shape=[
                int(value) for value in list(payload.get("input_shape") or [])
            ],
            hidden_shape=[
                int(value) for value in list(payload.get("hidden_shape") or [])
            ],
            output_shapes={
                str(key): [int(value) for value in list(values or [])]
                for key, values in dict(payload.get("output_shapes") or {}).items()
            },
            loss_keys_checked=strings(payload.get("loss_keys_checked")),
            smoke_forward_passed=bool(payload.get("smoke_forward_passed", False)),
            authority_class=str(
                payload.get(
                    "authority_class", "phase65_cpu_smoke_forward_shape_check_only"
                )
            ),
            training_executed=bool(payload.get("training_executed", False)),
            weights_initialized=bool(payload.get("weights_initialized", False)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            metadata=mapping(payload.get("metadata")),
            version=str(payload.get("version", PHASE65_CPU_SMOKE_FORWARD_VERSION)),
        )


@dataclass(frozen=True)
class Phase65MetaNodeTrainerScaffoldManifest:
    trainer_scaffold_id: str
    phase65_report_id: str
    dataset_contract_id: str
    model_config_id: str
    smoke_forward_id: str
    loss_count: int
    status: str
    dataset_contract_ready: bool
    losses_defined: bool
    model_config_ready: bool
    cpu_smoke_forward_passed: bool
    ready_for_training: bool = False
    ready_for_gpu_training: bool = False
    phase7_authority_granted: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_phase65_trainer_denied_gates)
    remaining_blockers: list[str] = field(default_factory=list)
    aggregate_counts: dict[str, float] = field(default_factory=dict)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE65_TRAINER_SCAFFOLD_MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainer_scaffold_id": self.trainer_scaffold_id,
            "version": self.version,
            "phase65_report_id": self.phase65_report_id,
            "dataset_contract_id": self.dataset_contract_id,
            "model_config_id": self.model_config_id,
            "smoke_forward_id": self.smoke_forward_id,
            "loss_count": int(self.loss_count),
            "status": self.status,
            "dataset_contract_ready": bool(self.dataset_contract_ready),
            "losses_defined": bool(self.losses_defined),
            "model_config_ready": bool(self.model_config_ready),
            "cpu_smoke_forward_passed": bool(self.cpu_smoke_forward_passed),
            "ready_for_training": bool(self.ready_for_training),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": _phase65_trainer_denied_gates(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "aggregate_counts": float_mapping(self.aggregate_counts),
            "artifact_refs": mapping(self.artifact_refs),
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase65MetaNodeTrainerScaffoldManifest":
        return cls(
            trainer_scaffold_id=str(payload.get("trainer_scaffold_id", "")),
            phase65_report_id=str(payload.get("phase65_report_id", "")),
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            model_config_id=str(payload.get("model_config_id", "")),
            smoke_forward_id=str(payload.get("smoke_forward_id", "")),
            loss_count=int(payload.get("loss_count", 0) or 0),
            status=str(payload.get("status", "blocked")),
            dataset_contract_ready=bool(payload.get("dataset_contract_ready", False)),
            losses_defined=bool(payload.get("losses_defined", False)),
            model_config_ready=bool(payload.get("model_config_ready", False)),
            cpu_smoke_forward_passed=bool(
                payload.get("cpu_smoke_forward_passed", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            ready_for_gpu_training=bool(payload.get("ready_for_gpu_training", False)),
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
            denied_gates=_phase65_trainer_denied_gates(
                payload.get("denied_gates")
            ),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            aggregate_counts=float_mapping(payload.get("aggregate_counts")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            metadata=mapping(payload.get("metadata")),
            version=str(
                payload.get("version", PHASE65_TRAINER_SCAFFOLD_MANIFEST_VERSION)
            ),
        )


def _loss(
    *,
    loss_key: str,
    role: str,
    target_keys: Sequence[str],
    formula: str,
    default_weight: float,
) -> Phase65MetaNodeLossDefinition:
    payload = {
        "loss_key": loss_key,
        "target_keys": list(target_keys),
        "version": PHASE65_LOSS_DEFINITION_VERSION,
    }
    return Phase65MetaNodeLossDefinition(
        loss_id=stable_id("phase65_meta_node_loss", payload),
        loss_key=loss_key,
        role=role,
        row_families=["phase65_meta_node_training_row_v1"],
        target_keys=list(target_keys),
        formula=formula,
        default_weight=default_weight,
        blockers=list(PHASE65_TRAINER_BLOCKERS),
        metadata={"training_claim": False, "weights_claim": False},
    )


def build_phase65_meta_node_loss_definitions() -> list[Phase65MetaNodeLossDefinition]:
    return [
        _loss(
            loss_key="activation_timing_loss",
            role="calibrate when a meta-node should activate in a shadow trace",
            target_keys=["activation_timing"],
            formula="masked_cross_entropy_or_l1(activation_timing_label, predicted_timing)",
            default_weight=0.8,
        ),
        _loss(
            loss_key="activation_strength_loss",
            role="calibrate shadow activation strength without dispatch authority",
            target_keys=["activation_strength"],
            formula="masked_l1(activation_strength_label, predicted_strength)",
            default_weight=0.8,
        ),
        _loss(
            loss_key="intervention_kind_loss",
            role="classify the advisory intervention family for a node",
            target_keys=["intervention_kind"],
            formula="cross_entropy(intervention_kind_label, intervention_logits)",
            default_weight=1.0,
        ),
        _loss(
            loss_key="target_selection_loss",
            role="score which target surfaces the node should condition on",
            target_keys=["target_selection"],
            formula="multi_label_bce(target_selection_mask, target_logits)",
            default_weight=0.9,
        ),
        _loss(
            loss_key="counterfactual_downstream_effect_loss",
            role="learn signed downstream effect slots once counterfactual corpus exists",
            target_keys=[
                "counterfactual_downstream_improvement",
                "governance_satisfaction",
                "rollback_sensitivity",
            ],
            formula="masked_huber(counterfactual_effect_labels, predicted_effects)",
            default_weight=1.0,
        ),
        _loss(
            loss_key="neighbor_consistency_loss",
            role="keep related meta-node activations coherent across the graph",
            target_keys=["neighbor_consistency"],
            formula="graph_smoothness_penalty(neighbor_embeddings, activation_logits)",
            default_weight=0.5,
        ),
        _loss(
            loss_key="robustness_shift_loss",
            role="penalize brittle activations under replay and deployment-shift probes",
            target_keys=["robustness_replay_shift"],
            formula="masked_l1(robustness_shift_label, predicted_shift)",
            default_weight=0.6,
        ),
        _loss(
            loss_key="promotion_denial_regularizer",
            role="preserve denial of authority unless promotion evidence is explicit",
            target_keys=["promotion_denied"],
            formula="bce(always_deny_without_evidence, authority_logits)",
            default_weight=1.0,
        ),
    ]


def _intervention_kind_id(kind: str) -> float:
    kinds = {
        "shape": 0.1,
        "defer": 0.2,
        "fallback": 0.3,
        "veto": 0.4,
        "operator_handoff": 0.5,
    }
    return kinds.get(kind, 0.0)


def _build_dataset_contract(
    *,
    phase65_report: Phase65MetaNodeNeuralizationReport,
    states: Sequence[MetaNodeState],
    trajectories: Sequence[MetaNodeTrajectoryReceipt],
    interventions: Sequence[MetaNodeInterventionReceipt],
    robustness_reports: Sequence[MetaNodeRobustnessReport],
    gates: Sequence[MetaNodePromotionGate],
    artifact_refs: Mapping[str, Any] | None,
) -> Phase65MetaNodeTrainerDatasetContract:
    row_count = len(states)
    payload = {
        "phase65_report_id": phase65_report.report_id,
        "row_count": row_count,
        "feature_keys": list(PHASE65_FEATURE_KEYS),
        "target_keys": list(PHASE65_TARGET_KEYS),
    }
    return Phase65MetaNodeTrainerDatasetContract(
        dataset_contract_id=stable_id("phase65_meta_node_dataset", payload),
        phase65_report_id=phase65_report.report_id,
        row_count=row_count,
        row_family_counts={"phase65_meta_node_training_row_v1": row_count},
        feature_keys=list(PHASE65_FEATURE_KEYS),
        target_keys=list(PHASE65_TARGET_KEYS),
        feature_dim=len(PHASE65_FEATURE_KEYS),
        target_dim=len(PHASE65_TARGET_KEYS),
        source_artifact_refs=mapping(artifact_refs),
        ready_for_cpu_smoke_forward=bool(
            row_count
            and trajectories
            and interventions
            and robustness_reports
            and gates
        ),
        blockers=list(PHASE65_TRAINER_BLOCKERS),
        metadata={
            "trajectory_receipt_count": len(trajectories),
            "intervention_receipt_count": len(interventions),
            "robustness_report_count": len(robustness_reports),
            "promotion_gate_count": len(gates),
            "training_claim": False,
            "weights_claim": False,
        },
    )


def _build_model_config(
    dataset_contract: Phase65MetaNodeTrainerDatasetContract,
) -> Phase65MetaNodeModelComponentConfig:
    hidden_dim = min(256, max(32, dataset_contract.feature_dim * 4))
    components = [
        {
            "component_key": "meta_node_state_encoder",
            "model_family": "mlp_or_small_transformer_encoder",
            "architecture_pattern": "typed_feature_encoder",
            "input_dim": dataset_contract.feature_dim,
            "hidden_dims": [hidden_dim, max(16, hidden_dim // 2)],
            "output_dim": hidden_dim,
            "training_enabled": False,
            "weights_initialized": False,
            "weights_written": False,
            "promotion_eligible": False,
            "authority_class": "phase65_model_component_config_only",
        },
        {
            "component_key": "trajectory_context_encoder",
            "model_family": "event_sequence_encoder",
            "architecture_pattern": "receipt_sequence_conditioner",
            "input_dim": dataset_contract.feature_dim,
            "hidden_dims": [hidden_dim],
            "output_dim": hidden_dim,
            "training_enabled": False,
            "weights_initialized": False,
            "weights_written": False,
            "promotion_eligible": False,
            "authority_class": "phase65_model_component_config_only",
        },
        {
            "component_key": "meta_node_activation_heads",
            "model_family": "multi_head_predictor",
            "architecture_pattern": "activation_intervention_counterfactual_heads",
            "input_dim": hidden_dim,
            "hidden_dims": [max(16, hidden_dim // 2)],
            "output_dim": dataset_contract.target_dim,
            "training_enabled": False,
            "weights_initialized": False,
            "weights_written": False,
            "promotion_eligible": False,
            "authority_class": "phase65_model_component_config_only",
        },
        {
            "component_key": "robustness_and_denial_calibrator",
            "model_family": "calibration_head",
            "architecture_pattern": "robustness_uncertainty_authority_denial",
            "input_dim": hidden_dim,
            "hidden_dims": [max(16, hidden_dim // 2)],
            "output_dim": 3,
            "training_enabled": False,
            "weights_initialized": False,
            "weights_written": False,
            "promotion_eligible": False,
            "authority_class": "phase65_model_component_config_only",
        },
    ]
    payload = {
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "components": [component["component_key"] for component in components],
    }
    return Phase65MetaNodeModelComponentConfig(
        model_config_id=stable_id("phase65_meta_node_model_config", payload),
        dataset_contract_id=dataset_contract.dataset_contract_id,
        components=components,
        component_count=len(components),
        blockers=list(PHASE65_TRAINER_BLOCKERS),
        metadata={
            "training_claim": False,
            "weights_claim": False,
            "live_authority_claim": False,
        },
    )


def _build_cpu_smoke_forward(
    *,
    dataset_contract: Phase65MetaNodeTrainerDatasetContract,
    model_config: Phase65MetaNodeModelComponentConfig,
    losses: Sequence[Phase65MetaNodeLossDefinition],
) -> Phase65MetaNodeCPUSmokeForwardReceipt:
    hidden_dim = 32
    input_shape = [max(1, dataset_contract.row_count), dataset_contract.feature_dim]
    hidden_shape = [input_shape[0], hidden_dim]
    output_shapes = {
        "activation_timing_logits": [input_shape[0], 4],
        "activation_strength": [input_shape[0], 1],
        "intervention_kind_logits": [input_shape[0], 5],
        "target_selection_logits": [input_shape[0], max(1, dataset_contract.target_dim)],
        "counterfactual_effect": [input_shape[0], 3],
        "robustness_shift": [input_shape[0], 1],
        "authority_denial_logits": [input_shape[0], 1],
    }
    passed = (
        dataset_contract.ready_for_cpu_smoke_forward
        and dataset_contract.feature_dim > 0
        and dataset_contract.target_dim > 0
        and model_config.component_count >= 4
        and bool(losses)
    )
    payload = {
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "model_config_id": model_config.model_config_id,
        "input_shape": input_shape,
        "output_shapes": output_shapes,
    }
    return Phase65MetaNodeCPUSmokeForwardReceipt(
        smoke_forward_id=stable_id("phase65_meta_node_cpu_smoke", payload),
        dataset_contract_id=dataset_contract.dataset_contract_id,
        model_config_id=model_config.model_config_id,
        input_shape=input_shape,
        hidden_shape=hidden_shape,
        output_shapes=output_shapes,
        loss_keys_checked=[loss.loss_key for loss in losses],
        smoke_forward_passed=passed,
        metadata={
            "shape_check_only": True,
            "framework": "pure_python_no_weight_allocation",
            "training_claim": False,
            "weights_claim": False,
        },
    )


def build_phase65_meta_node_trainer_scaffold(
    *,
    phase65_report: Phase65MetaNodeNeuralizationReport,
    states: Sequence[MetaNodeState],
    trajectories: Sequence[MetaNodeTrajectoryReceipt],
    interventions: Sequence[MetaNodeInterventionReceipt],
    targets: Sequence[MetaNodeCounterfactualTarget],
    robustness_reports: Sequence[MetaNodeRobustnessReport],
    gates: Sequence[MetaNodePromotionGate],
    artifact_refs: Mapping[str, Any] | None = None,
) -> tuple[
    Phase65MetaNodeTrainerScaffoldManifest,
    Phase65MetaNodeTrainerDatasetContract,
    list[Phase65MetaNodeLossDefinition],
    Phase65MetaNodeModelComponentConfig,
    Phase65MetaNodeCPUSmokeForwardReceipt,
]:
    dataset_contract = _build_dataset_contract(
        phase65_report=phase65_report,
        states=states,
        trajectories=trajectories,
        interventions=interventions,
        robustness_reports=robustness_reports,
        gates=gates,
        artifact_refs=artifact_refs,
    )
    losses = build_phase65_meta_node_loss_definitions()
    model_config = _build_model_config(dataset_contract)
    smoke_forward = _build_cpu_smoke_forward(
        dataset_contract=dataset_contract,
        model_config=model_config,
        losses=losses,
    )
    dataset_ready = dataset_contract.ready_for_cpu_smoke_forward
    losses_defined = bool(losses) and all(
        loss.optimization_status == "defined_not_optimized"
        and not loss.direct_policy_rl
        for loss in losses
    )
    model_ready = model_config.component_count >= 4
    complete = dataset_ready and losses_defined and model_ready and (
        smoke_forward.smoke_forward_passed
    )
    payload = {
        "phase65_report_id": phase65_report.report_id,
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "model_config_id": model_config.model_config_id,
        "smoke_forward_id": smoke_forward.smoke_forward_id,
        "loss_count": len(losses),
    }
    manifest = Phase65MetaNodeTrainerScaffoldManifest(
        trainer_scaffold_id=stable_id("phase65_meta_node_trainer", payload),
        phase65_report_id=phase65_report.report_id,
        dataset_contract_id=dataset_contract.dataset_contract_id,
        model_config_id=model_config.model_config_id,
        smoke_forward_id=smoke_forward.smoke_forward_id,
        loss_count=len(losses),
        status="ok" if complete else "blocked",
        dataset_contract_ready=dataset_ready,
        losses_defined=losses_defined,
        model_config_ready=model_ready,
        cpu_smoke_forward_passed=smoke_forward.smoke_forward_passed,
        denied_gates=_phase65_trainer_denied_gates(),
        remaining_blockers=list(PHASE65_TRAINER_BLOCKERS),
        aggregate_counts={
            "node_state_count": float(len(states)),
            "trajectory_receipt_count": float(len(trajectories)),
            "intervention_receipt_count": float(len(interventions)),
            "counterfactual_target_count": float(len(targets)),
            "robustness_report_count": float(len(robustness_reports)),
            "promotion_gate_count": float(len(gates)),
            "loss_count": float(len(losses)),
        },
        artifact_refs=mapping(artifact_refs),
        metadata={
            "trainer_scaffold_only": True,
            "training_claim": False,
            "weights_claim": False,
            "promotion_claim": False,
        },
    )
    return manifest, dataset_contract, losses, model_config, smoke_forward


def save_phase65_meta_node_trainer_scaffold(
    output_dir: str | Path,
    manifest: Phase65MetaNodeTrainerScaffoldManifest,
    dataset_contract: Phase65MetaNodeTrainerDatasetContract,
    losses: Sequence[Phase65MetaNodeLossDefinition],
    model_config: Phase65MetaNodeModelComponentConfig,
    smoke_forward: Phase65MetaNodeCPUSmokeForwardReceipt,
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "manifest_path": output / "phase65_meta_node_trainer_scaffold_manifest_v1.json",
        "dataset_contract_path": output
        / "phase65_meta_node_trainer_dataset_contract_v1.json",
        "loss_definitions_path": output / "phase65_meta_node_loss_definitions_v1.json",
        "model_config_path": output / "phase65_meta_node_model_component_config_v1.json",
        "cpu_smoke_forward_path": output / "phase65_meta_node_cpu_smoke_forward_v1.json",
    }
    write_json(paths["manifest_path"], manifest.to_dict())
    write_json(paths["dataset_contract_path"], dataset_contract.to_dict())
    write_json(
        paths["loss_definitions_path"],
        {
            "version": PHASE65_LOSS_DEFINITION_VERSION,
            "loss_count": len(losses),
            "definitions": [loss.to_dict() for loss in losses],
            "training_executed": False,
            "weights_written": False,
            "promotion_eligible": False,
            "reward_math_mutation": False,
        },
    )
    write_json(paths["model_config_path"], model_config.to_dict())
    write_json(paths["cpu_smoke_forward_path"], smoke_forward.to_dict())
    return {key: str(value) for key, value in paths.items()}


def load_phase65_meta_node_trainer_scaffold_manifest(
    path: str | Path,
) -> Phase65MetaNodeTrainerScaffoldManifest:
    return Phase65MetaNodeTrainerScaffoldManifest.from_dict(load_json(path))


def load_phase65_meta_node_trainer_dataset_contract(
    path: str | Path,
) -> Phase65MetaNodeTrainerDatasetContract:
    return Phase65MetaNodeTrainerDatasetContract.from_dict(load_json(path))


def load_phase65_meta_node_loss_definitions(
    path: str | Path,
) -> list[Phase65MetaNodeLossDefinition]:
    payload = load_json(path)
    return [
        Phase65MetaNodeLossDefinition.from_dict(item)
        for item in list(payload.get("definitions") or [])
    ]


def load_phase65_meta_node_model_component_config(
    path: str | Path,
) -> Phase65MetaNodeModelComponentConfig:
    return Phase65MetaNodeModelComponentConfig.from_dict(load_json(path))


def load_phase65_meta_node_cpu_smoke_forward(
    path: str | Path,
) -> Phase65MetaNodeCPUSmokeForwardReceipt:
    return Phase65MetaNodeCPUSmokeForwardReceipt.from_dict(load_json(path))
