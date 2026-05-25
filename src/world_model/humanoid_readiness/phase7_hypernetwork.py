"""Phase 7 meta-composition hypernetwork scaffold.

The scaffold makes the future learned composition path explicit without
training it. Current Phase 7 receipts are wired into conditioning specs and a
CPU shape-check contract only. Outputs remain shadow/advisory and cannot
dispatch, veto, mutate reward math, write weights, or promote.
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
    write_jsonl,
)
from src.world_model.humanoid_readiness.phase7 import (
    PHASE7_REMAINING_BLOCKERS,
    Phase7CompositionModeSpec,
    Phase7ConflictOverrideReceipt,
    Phase7ControlFieldSlot,
    Phase7GovernanceNodeSurface,
    Phase7MetaRegalControlScaffoldReport,
    Phase7PromotionGate,
    Phase7TrainingRowSlot,
)
from src.world_model.humanoid_readiness.phase7_eval import (
    PHASE7_EVAL_REMAINING_BLOCKERS,
    Phase7ConflictJoinEvalReport,
    Phase7ControlFieldEvalReport,
    Phase7MetaGovernanceEvaluationReport,
    Phase7OutcomeJoinRow,
    Phase7ParetoRegimeEvalReport,
)
from src.world_model.humanoid_readiness.phase7_signal_adapters import (
    PHASE7_SIGNAL_ADAPTER_REMAINING_BLOCKERS,
    Phase7GovernanceNodeSignalAdapter,
    Phase7GovernanceNodeSignalReceipt,
    Phase7GovernanceSignalAdapterReport,
)

PHASE7_HYPERNETWORK_CONDITIONING_SPEC_VERSION = (
    "phase7_hypernetwork_conditioning_spec_v1"
)
PHASE7_HYPERNETWORK_OUTPUT_HEAD_VERSION = "phase7_hypernetwork_output_head_v1"
PHASE7_META_COMPOSITION_LOSS_VERSION = "phase7_meta_composition_loss_v1"
PHASE7_HYPERNETWORK_DATASET_CONTRACT_VERSION = (
    "phase7_hypernetwork_dataset_contract_v1"
)
PHASE7_HYPERNETWORK_MODEL_CONFIG_VERSION = "phase7_hypernetwork_model_config_v1"
PHASE7_HYPERNETWORK_CPU_SMOKE_FORWARD_VERSION = (
    "phase7_hypernetwork_cpu_smoke_forward_v1"
)
PHASE7_META_COMPOSITION_HYPERNETWORK_REPORT_VERSION = (
    "phase7_meta_composition_hypernetwork_scaffold_report_v1"
)

PHASE7_HYPERNETWORK_BLOCKERS = (
    *PHASE7_REMAINING_BLOCKERS,
    *[
        blocker
        for blocker in PHASE7_SIGNAL_ADAPTER_REMAINING_BLOCKERS
        if blocker not in PHASE7_REMAINING_BLOCKERS
    ],
    *[
        blocker
        for blocker in PHASE7_EVAL_REMAINING_BLOCKERS
        if blocker
        not in (*PHASE7_REMAINING_BLOCKERS, *PHASE7_SIGNAL_ADAPTER_REMAINING_BLOCKERS)
    ],
    "gpu_hypernetwork_training_not_run",
    "hypernetwork_weights_not_written",
    "labeled_meta_composition_outcomes_missing",
    "promotion_grade_meta_composition_benchmark_missing",
)

PHASE7_HYPERNETWORK_FEATURE_KEYS = (
    "node_confidence_prior",
    "node_signal_confidence",
    "hard_constraint_capable",
    "source_receipt_count",
    "lower_wm_receipt_backed",
    "composition_mode_id",
    "conflict_severity_prior",
    "control_field_eval_status",
    "conflict_eval_status",
    "pareto_regime_present",
    "false_veto_label_available",
    "false_allow_label_available",
    "policy_regret_label_available",
    "runtime_denial_mask",
    "promotion_denied",
)

PHASE7_HYPERNETWORK_TARGET_KEYS = (
    "node_gate_logits",
    "node_activation_strength",
    "composition_mode_logits",
    "conflict_override_weight_delta",
    "pareto_regime_parameters",
    "advisory_control_field_decoder",
    "veto_candidate_calibration",
    "uncertainty_calibration",
)


def _phase7_hypernetwork_denied_gates(
    extra: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    return denied_gate_map(
        {
            "phase7_authority_granted": False,
            "phase7_runtime_authority": False,
            "live_dispatch_allowed": False,
            "hard_veto_dispatch": False,
            "live_cross_wm_control": False,
            "lower_wm_replacement": False,
            "scalar_governance_collapse": False,
            "hypernetwork_weights_initialized": False,
            **dict(extra or {}),
        }
    )


@dataclass(frozen=True)
class Phase7HypernetworkConditioningSpec:
    conditioning_id: str
    conditioning_key: str
    role: str
    source_families: list[str] = field(default_factory=list)
    feature_keys: list[str] = field(default_factory=list)
    tensor_shape: list[int] = field(default_factory=list)
    now_wired_artifact_refs: dict[str, Any] = field(default_factory=dict)
    now_wired_receipt_ids: list[str] = field(default_factory=list)
    future_runtime_binding: str = "future_training_and_shadow_inference_only"
    meta_composition_semantics: str = ""
    training_target_only: bool = True
    shadow_only: bool = True
    live_dispatch_allowed: bool = False
    hard_veto_dispatch: bool = False
    promotion_eligible: bool = False
    authority_class: str = "phase7_hypernetwork_conditioning_spec_only"
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_HYPERNETWORK_CONDITIONING_SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditioning_id": self.conditioning_id,
            "version": self.version,
            "conditioning_key": self.conditioning_key,
            "role": self.role,
            "source_families": list(self.source_families),
            "feature_keys": list(self.feature_keys),
            "tensor_shape": [int(value) for value in self.tensor_shape],
            "now_wired_artifact_refs": mapping(self.now_wired_artifact_refs),
            "now_wired_receipt_ids": list(self.now_wired_receipt_ids),
            "future_runtime_binding": self.future_runtime_binding,
            "meta_composition_semantics": self.meta_composition_semantics,
            "training_target_only": bool(self.training_target_only),
            "shadow_only": bool(self.shadow_only),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "hard_veto_dispatch": bool(self.hard_veto_dispatch),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7HypernetworkConditioningSpec":
        return cls(
            conditioning_id=str(payload.get("conditioning_id", "")),
            conditioning_key=str(payload.get("conditioning_key", "")),
            role=str(payload.get("role", "")),
            source_families=strings(payload.get("source_families")),
            feature_keys=strings(payload.get("feature_keys")),
            tensor_shape=[
                int(value) for value in list(payload.get("tensor_shape") or [])
            ],
            now_wired_artifact_refs=mapping(payload.get("now_wired_artifact_refs")),
            now_wired_receipt_ids=strings(payload.get("now_wired_receipt_ids")),
            future_runtime_binding=str(
                payload.get(
                    "future_runtime_binding", "future_training_and_shadow_inference_only"
                )
            ),
            meta_composition_semantics=str(
                payload.get("meta_composition_semantics", "")
            ),
            training_target_only=bool(payload.get("training_target_only", True)),
            shadow_only=bool(payload.get("shadow_only", True)),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            hard_veto_dispatch=bool(payload.get("hard_veto_dispatch", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get(
                    "authority_class", "phase7_hypernetwork_conditioning_spec_only"
                )
            ),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(
                payload.get("version", PHASE7_HYPERNETWORK_CONDITIONING_SPEC_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase7HypernetworkOutputHeadSpec:
    head_id: str
    head_key: str
    role: str
    output_shape: list[int]
    conditioned_by: list[str] = field(default_factory=list)
    output_authority: str = "shadow_advisory_only"
    writes_runtime_policy: bool = False
    writes_weights: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    authority_class: str = "phase7_hypernetwork_output_head_spec_only"
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_HYPERNETWORK_OUTPUT_HEAD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_id": self.head_id,
            "version": self.version,
            "head_key": self.head_key,
            "role": self.role,
            "output_shape": [int(value) for value in self.output_shape],
            "conditioned_by": list(self.conditioned_by),
            "output_authority": self.output_authority,
            "writes_runtime_policy": bool(self.writes_runtime_policy),
            "writes_weights": bool(self.writes_weights),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7HypernetworkOutputHeadSpec":
        return cls(
            head_id=str(payload.get("head_id", "")),
            head_key=str(payload.get("head_key", "")),
            role=str(payload.get("role", "")),
            output_shape=[
                int(value) for value in list(payload.get("output_shape") or [])
            ],
            conditioned_by=strings(payload.get("conditioned_by")),
            output_authority=str(payload.get("output_authority", "shadow_advisory_only")),
            writes_runtime_policy=bool(payload.get("writes_runtime_policy", False)),
            writes_weights=bool(payload.get("writes_weights", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get(
                    "authority_class", "phase7_hypernetwork_output_head_spec_only"
                )
            ),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(payload.get("version", PHASE7_HYPERNETWORK_OUTPUT_HEAD_VERSION)),
        )


@dataclass(frozen=True)
class Phase7MetaCompositionLossDefinition:
    loss_id: str
    loss_key: str
    role: str
    conditioned_by: list[str] = field(default_factory=list)
    target_heads: list[str] = field(default_factory=list)
    formula: str = "defined_not_optimized"
    default_weight: float = 1.0
    optimization_status: str = "defined_not_optimized"
    uses_rl_style_signal: bool = False
    direct_policy_rl: bool = False
    authority_class: str = "phase7_meta_composition_loss_definition_only"
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_META_COMPOSITION_LOSS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss_id": self.loss_id,
            "version": self.version,
            "loss_key": self.loss_key,
            "role": self.role,
            "conditioned_by": list(self.conditioned_by),
            "target_heads": list(self.target_heads),
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
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7MetaCompositionLossDefinition":
        return cls(
            loss_id=str(payload.get("loss_id", "")),
            loss_key=str(payload.get("loss_key", "")),
            role=str(payload.get("role", "")),
            conditioned_by=strings(payload.get("conditioned_by")),
            target_heads=strings(payload.get("target_heads")),
            formula=str(payload.get("formula", "defined_not_optimized")),
            default_weight=float(payload.get("default_weight", 1.0) or 1.0),
            optimization_status=str(
                payload.get("optimization_status", "defined_not_optimized")
            ),
            uses_rl_style_signal=bool(payload.get("uses_rl_style_signal", False)),
            direct_policy_rl=bool(payload.get("direct_policy_rl", False)),
            authority_class=str(
                payload.get(
                    "authority_class", "phase7_meta_composition_loss_definition_only"
                )
            ),
            blockers=strings(payload.get("blockers")),
            metadata=mapping(payload.get("metadata")),
            version=str(payload.get("version", PHASE7_META_COMPOSITION_LOSS_VERSION)),
        )


@dataclass(frozen=True)
class Phase7HypernetworkDatasetContract:
    dataset_contract_id: str
    phase7_report_id: str
    signal_adapter_report_id: str
    eval_report_id: str
    row_count: int
    feature_keys: list[str]
    target_keys: list[str]
    feature_dim: int
    target_dim: int
    row_family_counts: dict[str, int] = field(default_factory=dict)
    conditioning_keys: list[str] = field(default_factory=list)
    shadow_outcome_join_slot_count: int = 0
    authority_class: str = "phase7_hypernetwork_dataset_contract_only"
    ready_for_cpu_smoke_forward: bool = False
    ready_for_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_HYPERNETWORK_DATASET_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_contract_id": self.dataset_contract_id,
            "version": self.version,
            "phase7_report_id": self.phase7_report_id,
            "signal_adapter_report_id": self.signal_adapter_report_id,
            "eval_report_id": self.eval_report_id,
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
            "conditioning_keys": list(self.conditioning_keys),
            "shadow_outcome_join_slot_count": int(self.shadow_outcome_join_slot_count),
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7HypernetworkDatasetContract":
        shapes = mapping(payload.get("shape_contracts"))
        return cls(
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            phase7_report_id=str(payload.get("phase7_report_id", "")),
            signal_adapter_report_id=str(
                payload.get("signal_adapter_report_id", "")
            ),
            eval_report_id=str(payload.get("eval_report_id", "")),
            row_count=int(payload.get("row_count", shapes.get("row_count", 0)) or 0),
            row_family_counts={
                str(key): int(value)
                for key, value in dict(payload.get("row_family_counts") or {}).items()
            },
            feature_keys=strings(payload.get("feature_keys")),
            target_keys=strings(payload.get("target_keys")),
            feature_dim=int(shapes.get("feature_dim", 0) or 0),
            target_dim=int(shapes.get("target_dim", 0) or 0),
            conditioning_keys=strings(payload.get("conditioning_keys")),
            shadow_outcome_join_slot_count=int(
                payload.get("shadow_outcome_join_slot_count", 0) or 0
            ),
            authority_class=str(
                payload.get(
                    "authority_class", "phase7_hypernetwork_dataset_contract_only"
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
                payload.get("version", PHASE7_HYPERNETWORK_DATASET_CONTRACT_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase7HypernetworkModelComponentConfig:
    model_config_id: str
    dataset_contract_id: str
    components: list[dict[str, Any]] = field(default_factory=list)
    component_count: int = 0
    future_meta_composition_wiring: dict[str, Any] = field(default_factory=dict)
    training_executed: bool = False
    weights_initialized: bool = False
    weights_written: bool = False
    ready_for_gpu_training: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    live_policy_control: bool = False
    authority_class: str = "phase7_hypernetwork_model_config_only"
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_HYPERNETWORK_MODEL_CONFIG_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config_id": self.model_config_id,
            "version": self.version,
            "dataset_contract_id": self.dataset_contract_id,
            "component_count": int(self.component_count),
            "components": [mapping(component) for component in self.components],
            "future_meta_composition_wiring": mapping(
                self.future_meta_composition_wiring
            ),
            "training_executed": bool(self.training_executed),
            "weights_initialized": bool(self.weights_initialized),
            "weights_written": bool(self.weights_written),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "live_policy_control": bool(self.live_policy_control),
            "authority_class": self.authority_class,
            "blockers": list(self.blockers),
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7HypernetworkModelComponentConfig":
        return cls(
            model_config_id=str(payload.get("model_config_id", "")),
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            components=[
                mapping(item) for item in list(payload.get("components") or [])
            ],
            component_count=int(payload.get("component_count", 0) or 0),
            future_meta_composition_wiring=mapping(
                payload.get("future_meta_composition_wiring")
            ),
            training_executed=bool(payload.get("training_executed", False)),
            weights_initialized=bool(payload.get("weights_initialized", False)),
            weights_written=bool(payload.get("weights_written", False)),
            ready_for_gpu_training=bool(payload.get("ready_for_gpu_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            authority_class=str(
                payload.get("authority_class", "phase7_hypernetwork_model_config_only")
            ),
            blockers=strings(payload.get("blockers")),
            metadata=mapping(payload.get("metadata")),
            version=str(payload.get("version", PHASE7_HYPERNETWORK_MODEL_CONFIG_VERSION)),
        )


@dataclass(frozen=True)
class Phase7HypernetworkCPUSmokeForwardReceipt:
    smoke_forward_id: str
    dataset_contract_id: str
    model_config_id: str
    conditioning_tensor_shape: list[int]
    context_token_shape: list[int]
    output_shapes: dict[str, list[int]]
    smoke_forward_passed: bool
    shape_check_only: bool = True
    training_executed: bool = False
    weights_initialized: bool = False
    weights_written: bool = False
    live_dispatch_allowed: bool = False
    hard_veto_dispatch: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    authority_class: str = "phase7_hypernetwork_cpu_smoke_shape_check_only"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_HYPERNETWORK_CPU_SMOKE_FORWARD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoke_forward_id": self.smoke_forward_id,
            "version": self.version,
            "dataset_contract_id": self.dataset_contract_id,
            "model_config_id": self.model_config_id,
            "conditioning_tensor_shape": [
                int(value) for value in self.conditioning_tensor_shape
            ],
            "context_token_shape": [int(value) for value in self.context_token_shape],
            "output_shapes": {
                str(key): [int(value) for value in values]
                for key, values in self.output_shapes.items()
            },
            "smoke_forward_passed": bool(self.smoke_forward_passed),
            "shape_check_only": bool(self.shape_check_only),
            "training_executed": bool(self.training_executed),
            "weights_initialized": bool(self.weights_initialized),
            "weights_written": bool(self.weights_written),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "hard_veto_dispatch": bool(self.hard_veto_dispatch),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "authority_class": self.authority_class,
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7HypernetworkCPUSmokeForwardReceipt":
        return cls(
            smoke_forward_id=str(payload.get("smoke_forward_id", "")),
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            model_config_id=str(payload.get("model_config_id", "")),
            conditioning_tensor_shape=[
                int(value)
                for value in list(payload.get("conditioning_tensor_shape") or [])
            ],
            context_token_shape=[
                int(value) for value in list(payload.get("context_token_shape") or [])
            ],
            output_shapes={
                str(key): [int(value) for value in list(values or [])]
                for key, values in dict(payload.get("output_shapes") or {}).items()
            },
            smoke_forward_passed=bool(payload.get("smoke_forward_passed", False)),
            shape_check_only=bool(payload.get("shape_check_only", True)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_initialized=bool(payload.get("weights_initialized", False)),
            weights_written=bool(payload.get("weights_written", False)),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            hard_veto_dispatch=bool(payload.get("hard_veto_dispatch", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            authority_class=str(
                payload.get(
                    "authority_class",
                    "phase7_hypernetwork_cpu_smoke_shape_check_only",
                )
            ),
            metadata=mapping(payload.get("metadata")),
            version=str(
                payload.get("version", PHASE7_HYPERNETWORK_CPU_SMOKE_FORWARD_VERSION)
            ),
        )


@dataclass(frozen=True)
class Phase7MetaCompositionHypernetworkScaffoldReport:
    report_id: str
    phase7_report_id: str
    signal_adapter_report_id: str
    eval_report_id: str
    status: str
    conditioning_spec_count: int
    output_head_count: int
    loss_count: int
    dataset_contract_id: str
    model_config_id: str
    smoke_forward_id: str
    local_hypernetwork_scaffold_complete: bool
    conditioning_wiring_complete: bool
    future_meta_composition_explicit: bool
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
    live_dispatch_allowed: bool = False
    hard_veto_dispatch: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_phase7_hypernetwork_denied_gates)
    remaining_blockers: list[str] = field(default_factory=list)
    aggregate_counts: dict[str, float] = field(default_factory=dict)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_META_COMPOSITION_HYPERNETWORK_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase7_report_id": self.phase7_report_id,
            "signal_adapter_report_id": self.signal_adapter_report_id,
            "eval_report_id": self.eval_report_id,
            "status": self.status,
            "conditioning_spec_count": int(self.conditioning_spec_count),
            "output_head_count": int(self.output_head_count),
            "loss_count": int(self.loss_count),
            "dataset_contract_id": self.dataset_contract_id,
            "model_config_id": self.model_config_id,
            "smoke_forward_id": self.smoke_forward_id,
            "local_hypernetwork_scaffold_complete": bool(
                self.local_hypernetwork_scaffold_complete
            ),
            "conditioning_wiring_complete": bool(self.conditioning_wiring_complete),
            "future_meta_composition_explicit": bool(
                self.future_meta_composition_explicit
            ),
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
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "hard_veto_dispatch": bool(self.hard_veto_dispatch),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": _phase7_hypernetwork_denied_gates(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "aggregate_counts": float_mapping(self.aggregate_counts),
            "artifact_refs": mapping(self.artifact_refs),
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7MetaCompositionHypernetworkScaffoldReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase7_report_id=str(payload.get("phase7_report_id", "")),
            signal_adapter_report_id=str(
                payload.get("signal_adapter_report_id", "")
            ),
            eval_report_id=str(payload.get("eval_report_id", "")),
            status=str(payload.get("status", "blocked")),
            conditioning_spec_count=int(payload.get("conditioning_spec_count", 0) or 0),
            output_head_count=int(payload.get("output_head_count", 0) or 0),
            loss_count=int(payload.get("loss_count", 0) or 0),
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            model_config_id=str(payload.get("model_config_id", "")),
            smoke_forward_id=str(payload.get("smoke_forward_id", "")),
            local_hypernetwork_scaffold_complete=bool(
                payload.get("local_hypernetwork_scaffold_complete", False)
            ),
            conditioning_wiring_complete=bool(
                payload.get("conditioning_wiring_complete", False)
            ),
            future_meta_composition_explicit=bool(
                payload.get("future_meta_composition_explicit", False)
            ),
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
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            hard_veto_dispatch=bool(payload.get("hard_veto_dispatch", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates=_phase7_hypernetwork_denied_gates(
                payload.get("denied_gates")
            ),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            aggregate_counts=float_mapping(payload.get("aggregate_counts")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            metadata=mapping(payload.get("metadata")),
            version=str(
                payload.get(
                    "version", PHASE7_META_COMPOSITION_HYPERNETWORK_REPORT_VERSION
                )
            ),
        )


def _conditioning(
    *,
    conditioning_key: str,
    role: str,
    source_families: Sequence[str],
    feature_keys: Sequence[str],
    tensor_shape: Sequence[int],
    receipt_ids: Sequence[str],
    artifact_refs: Mapping[str, Any],
    semantics: str,
) -> Phase7HypernetworkConditioningSpec:
    payload = {
        "conditioning_key": conditioning_key,
        "feature_keys": list(feature_keys),
        "tensor_shape": list(tensor_shape),
    }
    return Phase7HypernetworkConditioningSpec(
        conditioning_id=stable_id("phase7_hyper_conditioning", payload),
        conditioning_key=conditioning_key,
        role=role,
        source_families=list(source_families),
        feature_keys=list(feature_keys),
        tensor_shape=[int(value) for value in tensor_shape],
        now_wired_artifact_refs=mapping(artifact_refs),
        now_wired_receipt_ids=list(receipt_ids),
        future_runtime_binding=(
            "future hypernetwork consumes these tensors to generate advisory "
            "composition parameters; runtime dispatch remains separately gated"
        ),
        meta_composition_semantics=semantics,
        denied_authority=list(PHASE7_HYPERNETWORK_BLOCKERS),
    )


def _loss(
    *,
    loss_key: str,
    role: str,
    conditioned_by: Sequence[str],
    target_heads: Sequence[str],
    formula: str,
    default_weight: float,
) -> Phase7MetaCompositionLossDefinition:
    payload = {
        "loss_key": loss_key,
        "conditioned_by": list(conditioned_by),
        "target_heads": list(target_heads),
    }
    return Phase7MetaCompositionLossDefinition(
        loss_id=stable_id("phase7_meta_composition_loss", payload),
        loss_key=loss_key,
        role=role,
        conditioned_by=list(conditioned_by),
        target_heads=list(target_heads),
        formula=formula,
        default_weight=default_weight,
        blockers=list(PHASE7_HYPERNETWORK_BLOCKERS),
        metadata={"training_claim": False, "weights_claim": False},
    )


def _mode_index(mode_key: str, modes: Sequence[Phase7CompositionModeSpec]) -> int:
    for index, mode in enumerate(modes):
        if mode.mode_key == mode_key:
            return index
    return 0


def _build_conditioning_specs(
    *,
    surfaces: Sequence[Phase7GovernanceNodeSurface],
    modes: Sequence[Phase7CompositionModeSpec],
    conflicts: Sequence[Phase7ConflictOverrideReceipt],
    field_evals: Sequence[Phase7ControlFieldEvalReport],
    conflict_evals: Sequence[Phase7ConflictJoinEvalReport],
    regime_evals: Sequence[Phase7ParetoRegimeEvalReport],
    outcome_rows: Sequence[Phase7OutcomeJoinRow],
    signal_receipts: Sequence[Phase7GovernanceNodeSignalReceipt],
    runtime_summary: Mapping[str, Any],
    artifact_refs: Mapping[str, Any],
) -> list[Phase7HypernetworkConditioningSpec]:
    node_receipt_ids = [receipt.signal_id for receipt in signal_receipts]
    conflict_receipt_ids = [receipt.receipt_id for receipt in conflicts]
    eval_receipt_ids = [
        *(report.report_id for report in field_evals),
        *(report.report_id for report in conflict_evals),
        *(report.report_id for report in regime_evals),
    ]
    outcome_receipt_ids = [row.row_id for row in outcome_rows]
    return [
        _conditioning(
            conditioning_key="node_signal_conditioning",
            role=(
                "condition hypernetwork node gates on the eight governance-node "
                "signals, lower-WM provenance, confidence priors, and hard "
                "constraint masks"
            ),
            source_families=[
                "phase7_governance_node_surface",
                "phase7_governance_node_signal_receipt",
            ],
            feature_keys=[
                "node_key_embedding",
                "domain_key_embedding",
                "composition_role_embedding",
                "confidence_prior",
                "signal_confidence",
                "hard_constraint_capable",
                "source_receipt_count",
                "lower_wm_receipt_backed",
            ],
            tensor_shape=[max(1, len(surfaces)), 8],
            receipt_ids=node_receipt_ids,
            artifact_refs=artifact_refs,
            semantics=(
                "future meta-composition starts from node-local typed signals; "
                "this conditions generated node gate logits and activation strengths"
            ),
        ),
        _conditioning(
            conditioning_key="conflict_context_conditioning",
            role=(
                "condition generated override parameters on pairwise or multi-node "
                "conflicts, severity priors, composition modes, and related fields"
            ),
            source_families=[
                "phase7_conflict_override_receipt",
                "phase7_conflict_join_eval_report",
            ],
            feature_keys=[
                "conflict_key_embedding",
                "source_node_pair_embedding",
                "composition_mode_id",
                "severity_prior",
                "related_control_field_count",
                "hard_veto_dispatch_denied",
                "live_dispatch_denied",
            ],
            tensor_shape=[max(1, len(conflicts)), 7],
            receipt_ids=conflict_receipt_ids,
            artifact_refs=artifact_refs,
            semantics=(
                "future hypernetwork emits conflict override deltas while keeping "
                "hard-veto dispatch separately denied until promotion evidence exists"
            ),
        ),
        _conditioning(
            conditioning_key="pareto_regime_conditioning",
            role=(
                "condition generated Pareto/regime parameters on observed modes, "
                "active conflicts, economic/safety/deployment dimensions, and "
                "regime labels"
            ),
            source_families=[
                "phase7_pareto_regime_eval_report",
                "phase7_composition_mode_spec",
            ],
            feature_keys=[
                "regime_key_embedding",
                "composition_mode_histogram",
                "active_conflict_count",
                "pareto_dimension_mask",
                "deploy_recommendation_label_slot",
                "pricing_recommendation_label_slot",
            ],
            tensor_shape=[max(1, len(regime_evals)), max(1, len(modes) + 4)],
            receipt_ids=[report.report_id for report in regime_evals],
            artifact_refs=artifact_refs,
            semantics=(
                "future meta-composition learns non-scalar Pareto tradeoff surfaces; "
                "economic WM remains one governance voice inside the frontier, not "
                "the sole objective"
            ),
        ),
        _conditioning(
            conditioning_key="shadow_outcome_conditioning",
            role=(
                "join shadow control-field, conflict, and Pareto rows to future "
                "false-veto, false-allow, policy-regret, and downstream-effect labels"
            ),
            source_families=["phase7_outcome_join_row"],
            feature_keys=[
                "row_family_embedding",
                "false_veto_label_available",
                "false_allow_label_available",
                "policy_regret_label_available",
                "operator_recovery_delta_available",
                "ground_truth_join_status_embedding",
            ],
            tensor_shape=[max(1, len(outcome_rows)), 6],
            receipt_ids=outcome_receipt_ids,
            artifact_refs=artifact_refs,
            semantics=(
                "future trainer uses this as the supervised meta-composition target "
                "surface once labeled runtime or benchmark outcomes exist"
            ),
        ),
        _conditioning(
            conditioning_key="runtime_truth_and_denial_conditioning",
            role=(
                "condition every future output on runtime truth, provider/hardware "
                "evidence, and explicit denial masks so absence of evidence cannot "
                "be converted into authority"
            ),
            source_families=[
                "phase7_shadow_runtime_summary",
                "phase7_meta_governance_evaluation_report",
            ],
            feature_keys=[
                "runtime_event_count",
                "decision_count",
                "training_denied",
                "weights_denied",
                "provider_denied",
                "hardware_denied",
                "live_dispatch_denied",
                "promotion_denied",
            ],
            tensor_shape=[1, 8],
            receipt_ids=[
                str(runtime_summary.get("run_id", "")),
                *eval_receipt_ids,
            ],
            artifact_refs=artifact_refs,
            semantics=(
                "future hypernetwork conditioning includes denial masks as first-class "
                "inputs; generated parameters never bypass promotion gates"
            ),
        ),
    ]


def _build_output_heads(
    *,
    surfaces: Sequence[Phase7GovernanceNodeSurface],
    modes: Sequence[Phase7CompositionModeSpec],
    conflicts: Sequence[Phase7ConflictOverrideReceipt],
    fields: Sequence[Phase7ControlFieldSlot],
    regimes: Sequence[Phase7ParetoRegimeEvalReport],
    conditioning_specs: Sequence[Phase7HypernetworkConditioningSpec],
) -> list[Phase7HypernetworkOutputHeadSpec]:
    conditioning_keys = [spec.conditioning_key for spec in conditioning_specs]
    specs = [
        (
            "node_gate_logits",
            "score which governance nodes should influence a shadow composition",
            [max(1, len(surfaces)), 2],
        ),
        (
            "node_activation_strength",
            "emit advisory node influence strengths for replay/training rows",
            [max(1, len(surfaces)), 1],
        ),
        (
            "composition_mode_logits",
            "choose among typed composition modes for future supervised labels",
            [max(1, len(modes))],
        ),
        (
            "conflict_override_weight_delta",
            "emit non-dispatching conflict override deltas for shadow receipts",
            [max(1, len(conflicts)), 1],
        ),
        (
            "pareto_regime_parameters",
            "parameterize future Pareto frontier and admissible-region surfaces",
            [max(1, len(regimes)), 6],
        ),
        (
            "advisory_control_field_decoder",
            "decode generated parameters into advisory control-field slots only",
            [max(1, len(fields)), 4],
        ),
        (
            "veto_candidate_calibration",
            "calibrate hard-constraint candidates without allowing hard-veto dispatch",
            [max(1, len(conflicts)), 2],
        ),
        (
            "uncertainty_calibration",
            "calibrate abstention and confidence for meta-composition outputs",
            [max(1, len(surfaces)), 3],
        ),
    ]
    return [
        Phase7HypernetworkOutputHeadSpec(
            head_id=stable_id(
                "phase7_hyper_head",
                {"head_key": head_key, "output_shape": output_shape},
            ),
            head_key=head_key,
            role=role,
            output_shape=output_shape,
            conditioned_by=conditioning_keys,
            denied_authority=list(PHASE7_HYPERNETWORK_BLOCKERS),
        )
        for head_key, role, output_shape in specs
    ]


def _build_losses(
    conditioning_specs: Sequence[Phase7HypernetworkConditioningSpec],
    output_heads: Sequence[Phase7HypernetworkOutputHeadSpec],
) -> list[Phase7MetaCompositionLossDefinition]:
    conditioning_keys = [spec.conditioning_key for spec in conditioning_specs]
    head_keys = [head.head_key for head in output_heads]
    return [
        _loss(
            loss_key="signal_reconstruction_conditioning_coverage_loss",
            role="preserve node signal/provenance coverage in the conditioning state",
            conditioned_by=["node_signal_conditioning"],
            target_heads=["node_gate_logits", "node_activation_strength"],
            formula="masked_l1(signal_features, reconstructed_signal_features) + coverage_penalty",
            default_weight=0.7,
        ),
        _loss(
            loss_key="node_gate_calibration_loss",
            role="calibrate governance-node gate logits against labeled outcomes",
            conditioned_by=["node_signal_conditioning", "shadow_outcome_conditioning"],
            target_heads=["node_gate_logits"],
            formula="cross_entropy_or_brier(node_gate_label, node_gate_logits)",
            default_weight=1.0,
        ),
        _loss(
            loss_key="composition_mode_classification_loss",
            role="learn which composition mode fits a regime and conflict context",
            conditioned_by=["conflict_context_conditioning", "pareto_regime_conditioning"],
            target_heads=["composition_mode_logits"],
            formula="cross_entropy(composition_mode_label, composition_mode_logits)",
            default_weight=1.0,
        ),
        _loss(
            loss_key="conflict_override_correctness_loss",
            role="learn whether conflict overrides reduced false allows/vetoes",
            conditioned_by=["conflict_context_conditioning", "shadow_outcome_conditioning"],
            target_heads=["conflict_override_weight_delta", "veto_candidate_calibration"],
            formula="masked_bce(override_correctness_label, override_logits)",
            default_weight=1.0,
        ),
        _loss(
            loss_key="pareto_frontier_regime_loss",
            role="fit non-dominated economic, safety, deployment, data, and embodiment tradeoffs",
            conditioned_by=["pareto_regime_conditioning"],
            target_heads=["pareto_regime_parameters"],
            formula="frontier_distance_loss(observed_labels, generated_pareto_params)",
            default_weight=1.0,
        ),
        _loss(
            loss_key="false_veto_false_allow_loss",
            role="penalize erroneous blocking or allowing once governance labels exist",
            conditioned_by=["shadow_outcome_conditioning"],
            target_heads=["veto_candidate_calibration", "advisory_control_field_decoder"],
            formula="weighted_bce(false_veto_label, false_allow_label)",
            default_weight=1.0,
        ),
        _loss(
            loss_key="policy_regret_downstream_effect_loss",
            role="learn downstream effect and regret labels without touching live policy",
            conditioned_by=["shadow_outcome_conditioning", "runtime_truth_and_denial_conditioning"],
            target_heads=["advisory_control_field_decoder"],
            formula="masked_huber(policy_regret_delta, predicted_shadow_effect)",
            default_weight=0.8,
        ),
        _loss(
            loss_key="uncertainty_calibration_loss",
            role="calibrate abstention and confidence for generated composition parameters",
            conditioned_by=conditioning_keys,
            target_heads=["uncertainty_calibration"],
            formula="nll + brier + ece_proxy",
            default_weight=0.6,
        ),
        _loss(
            loss_key="promotion_denial_regularizer",
            role="make denial masks explicit so generated parameters cannot imply authority",
            conditioned_by=["runtime_truth_and_denial_conditioning"],
            target_heads=head_keys,
            formula="bce(always_deny_without_evidence, authority_logits)",
            default_weight=1.0,
        ),
    ]


def _build_dataset_contract(
    *,
    phase7_report: Phase7MetaRegalControlScaffoldReport,
    signal_report: Phase7GovernanceSignalAdapterReport,
    eval_report: Phase7MetaGovernanceEvaluationReport,
    conditioning_specs: Sequence[Phase7HypernetworkConditioningSpec],
    outcome_rows: Sequence[Phase7OutcomeJoinRow],
) -> Phase7HypernetworkDatasetContract:
    row_count = max(1, len(outcome_rows))
    payload = {
        "phase7_report_id": phase7_report.report_id,
        "signal_adapter_report_id": signal_report.report_id,
        "eval_report_id": eval_report.report_id,
        "conditioning_keys": [spec.conditioning_key for spec in conditioning_specs],
        "row_count": row_count,
    }
    return Phase7HypernetworkDatasetContract(
        dataset_contract_id=stable_id("phase7_hyper_dataset", payload),
        phase7_report_id=phase7_report.report_id,
        signal_adapter_report_id=signal_report.report_id,
        eval_report_id=eval_report.report_id,
        row_count=row_count,
        row_family_counts={
            "phase7_outcome_join_row_v1": len(outcome_rows),
            "phase7_conditioning_spec_v1": len(conditioning_specs),
        },
        feature_keys=list(PHASE7_HYPERNETWORK_FEATURE_KEYS),
        target_keys=list(PHASE7_HYPERNETWORK_TARGET_KEYS),
        feature_dim=len(PHASE7_HYPERNETWORK_FEATURE_KEYS),
        target_dim=len(PHASE7_HYPERNETWORK_TARGET_KEYS),
        conditioning_keys=[spec.conditioning_key for spec in conditioning_specs],
        shadow_outcome_join_slot_count=len(outcome_rows),
        ready_for_cpu_smoke_forward=bool(
            conditioning_specs and outcome_rows and signal_report.signal_receipt_count
        ),
        blockers=list(PHASE7_HYPERNETWORK_BLOCKERS),
        metadata={
            "future_meta_composition_conditioning": True,
            "training_claim": False,
            "weights_claim": False,
            "authority_claim": False,
        },
    )


def _build_model_config(
    *,
    dataset_contract: Phase7HypernetworkDatasetContract,
    output_heads: Sequence[Phase7HypernetworkOutputHeadSpec],
    conditioning_specs: Sequence[Phase7HypernetworkConditioningSpec],
) -> Phase7HypernetworkModelComponentConfig:
    hidden_dim = min(512, max(64, dataset_contract.feature_dim * 8))
    components = [
        {
            "component_key": "governance_conditioning_encoder",
            "model_family": "typed_transformer_or_set_encoder",
            "architecture_pattern": "multi_source_conditioning_tokens",
            "input_dim": dataset_contract.feature_dim,
            "hidden_dims": [hidden_dim, hidden_dim],
            "output_dim": hidden_dim,
            "conditioning_keys": dataset_contract.conditioning_keys,
            "training_enabled": False,
            "weights_initialized": False,
            "weights_written": False,
            "promotion_eligible": False,
            "authority_class": "phase7_hypernetwork_model_config_only",
        },
        {
            "component_key": "meta_composition_hypernetwork",
            "model_family": "hypernetwork_parameter_generator",
            "architecture_pattern": (
                "conditioning_encoder_generates_shadow_composition_head_parameters"
            ),
            "input_dim": hidden_dim,
            "hidden_dims": [hidden_dim, max(32, hidden_dim // 2)],
            "output_heads": [head.head_key for head in output_heads],
            "training_enabled": False,
            "weights_initialized": False,
            "weights_written": False,
            "promotion_eligible": False,
            "authority_class": "phase7_hypernetwork_model_config_only",
        },
        {
            "component_key": "pareto_frontier_generator",
            "model_family": "frontier_parameter_head",
            "architecture_pattern": "non_scalar_pareto_tradeoff_surface",
            "input_dim": hidden_dim,
            "hidden_dims": [max(32, hidden_dim // 2)],
            "output_dim": 6,
            "economic_wm_role": "one_conditioning_voice_not_global_reward",
            "training_enabled": False,
            "weights_initialized": False,
            "weights_written": False,
            "promotion_eligible": False,
            "authority_class": "phase7_hypernetwork_model_config_only",
        },
        {
            "component_key": "authority_denial_calibrator",
            "model_family": "calibration_and_abstention_head",
            "architecture_pattern": "denial_mask_conditioned_uncertainty",
            "input_dim": hidden_dim,
            "hidden_dims": [max(32, hidden_dim // 2)],
            "output_dim": 3,
            "training_enabled": False,
            "weights_initialized": False,
            "weights_written": False,
            "promotion_eligible": False,
            "authority_class": "phase7_hypernetwork_model_config_only",
        },
    ]
    wiring = {
        "current_wiring": (
            "conditioning specs reference current Phase 7 scaffold, signal adapter, "
            "runtime/eval, and outcome artifacts"
        ),
        "future_training_wiring": (
            "GPU trainer may consume conditioning tensors and labels to fit a "
            "hypernetwork that generates shadow composition head parameters"
        ),
        "future_runtime_wiring": (
            "shadow runtime may read generated advisory parameters only after "
            "separate promotion evidence; live dispatch remains denied here"
        ),
        "conditioning_keys": [spec.conditioning_key for spec in conditioning_specs],
        "output_heads": [head.head_key for head in output_heads],
        "explicit_denials": [
            "no live dispatch",
            "no hard veto execution",
            "no reward math mutation",
            "no lower-WM replacement",
            "no scalar governance collapse",
            "no promotion",
        ],
    }
    payload = {
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "component_keys": [component["component_key"] for component in components],
        "output_heads": [head.head_key for head in output_heads],
    }
    return Phase7HypernetworkModelComponentConfig(
        model_config_id=stable_id("phase7_hyper_model_config", payload),
        dataset_contract_id=dataset_contract.dataset_contract_id,
        components=components,
        component_count=len(components),
        future_meta_composition_wiring=wiring,
        blockers=list(PHASE7_HYPERNETWORK_BLOCKERS),
        metadata={
            "hypernetwork_conditioning_explicit": True,
            "training_claim": False,
            "weights_claim": False,
            "live_authority_claim": False,
        },
    )


def _build_cpu_smoke_forward(
    *,
    dataset_contract: Phase7HypernetworkDatasetContract,
    model_config: Phase7HypernetworkModelComponentConfig,
    output_heads: Sequence[Phase7HypernetworkOutputHeadSpec],
    surfaces: Sequence[Phase7GovernanceNodeSurface],
    conditioning_specs: Sequence[Phase7HypernetworkConditioningSpec],
) -> Phase7HypernetworkCPUSmokeForwardReceipt:
    conditioning_shape = [
        max(1, len(conditioning_specs)),
        max(1, len(surfaces)),
        dataset_contract.feature_dim,
    ]
    context_shape = [
        max(1, dataset_contract.row_count + len(conditioning_specs)),
        64,
    ]
    output_shapes = {head.head_key: head.output_shape for head in output_heads}
    passed = (
        dataset_contract.ready_for_cpu_smoke_forward
        and dataset_contract.feature_dim > 0
        and dataset_contract.target_dim > 0
        and model_config.component_count >= 4
        and len(output_heads) >= 8
        and all(not head.writes_runtime_policy for head in output_heads)
        and all(not head.writes_weights for head in output_heads)
    )
    payload = {
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "model_config_id": model_config.model_config_id,
        "conditioning_shape": conditioning_shape,
        "output_shapes": output_shapes,
    }
    return Phase7HypernetworkCPUSmokeForwardReceipt(
        smoke_forward_id=stable_id("phase7_hyper_cpu_smoke", payload),
        dataset_contract_id=dataset_contract.dataset_contract_id,
        model_config_id=model_config.model_config_id,
        conditioning_tensor_shape=conditioning_shape,
        context_token_shape=context_shape,
        output_shapes=output_shapes,
        smoke_forward_passed=passed,
        metadata={
            "shape_check_only": True,
            "framework": "pure_python_no_weight_allocation",
            "training_claim": False,
            "weights_claim": False,
            "live_authority_claim": False,
        },
    )


def build_phase7_meta_composition_hypernetwork_scaffold(
    *,
    phase7_report: Phase7MetaRegalControlScaffoldReport,
    surfaces: Sequence[Phase7GovernanceNodeSurface],
    modes: Sequence[Phase7CompositionModeSpec],
    conflicts: Sequence[Phase7ConflictOverrideReceipt],
    control_fields: Sequence[Phase7ControlFieldSlot],
    training_rows: Sequence[Phase7TrainingRowSlot],
    promotion_gates: Sequence[Phase7PromotionGate],
    signal_report: Phase7GovernanceSignalAdapterReport,
    signal_adapters: Sequence[Phase7GovernanceNodeSignalAdapter],
    signal_receipts: Sequence[Phase7GovernanceNodeSignalReceipt],
    eval_report: Phase7MetaGovernanceEvaluationReport,
    field_evals: Sequence[Phase7ControlFieldEvalReport],
    conflict_evals: Sequence[Phase7ConflictJoinEvalReport],
    regime_evals: Sequence[Phase7ParetoRegimeEvalReport],
    outcome_rows: Sequence[Phase7OutcomeJoinRow],
    runtime_summary: Mapping[str, Any] | None = None,
    artifact_refs: Mapping[str, Any] | None = None,
) -> tuple[
    Phase7MetaCompositionHypernetworkScaffoldReport,
    list[Phase7HypernetworkConditioningSpec],
    list[Phase7HypernetworkOutputHeadSpec],
    list[Phase7MetaCompositionLossDefinition],
    Phase7HypernetworkDatasetContract,
    Phase7HypernetworkModelComponentConfig,
    Phase7HypernetworkCPUSmokeForwardReceipt,
]:
    refs = mapping(artifact_refs)
    conditioning_specs = _build_conditioning_specs(
        surfaces=surfaces,
        modes=modes,
        conflicts=conflicts,
        field_evals=field_evals,
        conflict_evals=conflict_evals,
        regime_evals=regime_evals,
        outcome_rows=outcome_rows,
        signal_receipts=signal_receipts,
        runtime_summary=runtime_summary or {},
        artifact_refs=refs,
    )
    output_heads = _build_output_heads(
        surfaces=surfaces,
        modes=modes,
        conflicts=conflicts,
        fields=control_fields,
        regimes=regime_evals,
        conditioning_specs=conditioning_specs,
    )
    losses = _build_losses(conditioning_specs, output_heads)
    dataset_contract = _build_dataset_contract(
        phase7_report=phase7_report,
        signal_report=signal_report,
        eval_report=eval_report,
        conditioning_specs=conditioning_specs,
        outcome_rows=outcome_rows,
    )
    model_config = _build_model_config(
        dataset_contract=dataset_contract,
        output_heads=output_heads,
        conditioning_specs=conditioning_specs,
    )
    smoke_forward = _build_cpu_smoke_forward(
        dataset_contract=dataset_contract,
        model_config=model_config,
        output_heads=output_heads,
        surfaces=surfaces,
        conditioning_specs=conditioning_specs,
    )
    conditioning_complete = (
        len(conditioning_specs) >= 5
        and signal_report.shadow_runtime_feed_ready
        and signal_report.all_eight_nodes_signal_backed
        and eval_report.local_meta_governance_eval_complete
    )
    future_explicit = (
        "meta_composition_hypernetwork"
        in {
            str(component.get("component_key", ""))
            for component in model_config.components
        }
        and bool(model_config.future_meta_composition_wiring)
        and len(output_heads) >= 8
    )
    complete = (
        conditioning_complete
        and future_explicit
        and bool(losses)
        and dataset_contract.ready_for_cpu_smoke_forward
        and smoke_forward.smoke_forward_passed
        and all(not gate.authority_granted for gate in promotion_gates)
    )
    payload = {
        "phase7_report_id": phase7_report.report_id,
        "signal_adapter_report_id": signal_report.report_id,
        "eval_report_id": eval_report.report_id,
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "model_config_id": model_config.model_config_id,
        "smoke_forward_id": smoke_forward.smoke_forward_id,
    }
    report = Phase7MetaCompositionHypernetworkScaffoldReport(
        report_id=stable_id("phase7_hypernetwork_scaffold", payload),
        phase7_report_id=phase7_report.report_id,
        signal_adapter_report_id=signal_report.report_id,
        eval_report_id=eval_report.report_id,
        status="ok" if complete else "blocked",
        conditioning_spec_count=len(conditioning_specs),
        output_head_count=len(output_heads),
        loss_count=len(losses),
        dataset_contract_id=dataset_contract.dataset_contract_id,
        model_config_id=model_config.model_config_id,
        smoke_forward_id=smoke_forward.smoke_forward_id,
        local_hypernetwork_scaffold_complete=complete,
        conditioning_wiring_complete=conditioning_complete,
        future_meta_composition_explicit=future_explicit,
        cpu_smoke_forward_passed=smoke_forward.smoke_forward_passed,
        denied_gates=_phase7_hypernetwork_denied_gates(),
        remaining_blockers=list(PHASE7_HYPERNETWORK_BLOCKERS),
        aggregate_counts={
            "governance_node_surface_count": float(len(surfaces)),
            "composition_mode_count": float(len(modes)),
            "conflict_override_receipt_count": float(len(conflicts)),
            "control_field_slot_count": float(len(control_fields)),
            "training_row_slot_count": float(len(training_rows)),
            "promotion_gate_count": float(len(promotion_gates)),
            "signal_adapter_count": float(len(signal_adapters)),
            "signal_receipt_count": float(len(signal_receipts)),
            "control_field_eval_count": float(len(field_evals)),
            "conflict_join_eval_count": float(len(conflict_evals)),
            "pareto_regime_eval_count": float(len(regime_evals)),
            "outcome_join_row_count": float(len(outcome_rows)),
        },
        artifact_refs=refs,
        metadata={
            "hypernetwork_conditioning_explicit": True,
            "future_meta_composition_wiring_explicit": True,
            "economic_wm_role": "conditioned_governance_voice_inside_pareto_composition",
            "training_claim": False,
            "weights_claim": False,
            "promotion_claim": False,
            "live_authority_claim": False,
        },
    )
    return (
        report,
        conditioning_specs,
        output_heads,
        losses,
        dataset_contract,
        model_config,
        smoke_forward,
    )


def save_phase7_meta_composition_hypernetwork_scaffold(
    output_dir: str | Path,
    report: Phase7MetaCompositionHypernetworkScaffoldReport,
    conditioning_specs: Sequence[Phase7HypernetworkConditioningSpec],
    output_heads: Sequence[Phase7HypernetworkOutputHeadSpec],
    losses: Sequence[Phase7MetaCompositionLossDefinition],
    dataset_contract: Phase7HypernetworkDatasetContract,
    model_config: Phase7HypernetworkModelComponentConfig,
    smoke_forward: Phase7HypernetworkCPUSmokeForwardReceipt,
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output
        / "phase7_meta_composition_hypernetwork_scaffold_report_v1.json",
        "conditioning_specs_path": output
        / "phase7_hypernetwork_conditioning_specs_v1.jsonl",
        "output_heads_path": output / "phase7_hypernetwork_output_heads_v1.jsonl",
        "loss_definitions_path": output / "phase7_meta_composition_losses_v1.json",
        "dataset_contract_path": output
        / "phase7_hypernetwork_dataset_contract_v1.json",
        "model_config_path": output / "phase7_hypernetwork_model_config_v1.json",
        "cpu_smoke_forward_path": output
        / "phase7_hypernetwork_cpu_smoke_forward_v1.json",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(
        paths["conditioning_specs_path"],
        [spec.to_dict() for spec in conditioning_specs],
    )
    write_jsonl(
        paths["output_heads_path"],
        [head.to_dict() for head in output_heads],
    )
    write_json(
        paths["loss_definitions_path"],
        {
            "version": PHASE7_META_COMPOSITION_LOSS_VERSION,
            "loss_count": len(losses),
            "definitions": [loss.to_dict() for loss in losses],
            "training_executed": False,
            "weights_written": False,
            "promotion_eligible": False,
            "reward_math_mutation": False,
        },
    )
    write_json(paths["dataset_contract_path"], dataset_contract.to_dict())
    write_json(paths["model_config_path"], model_config.to_dict())
    write_json(paths["cpu_smoke_forward_path"], smoke_forward.to_dict())
    return {key: str(value) for key, value in paths.items()}


def load_phase7_meta_composition_hypernetwork_scaffold_report(
    path: str | Path,
) -> Phase7MetaCompositionHypernetworkScaffoldReport:
    return Phase7MetaCompositionHypernetworkScaffoldReport.from_dict(load_json(path))


def load_phase7_hypernetwork_conditioning_specs(
    path: str | Path,
) -> list[Phase7HypernetworkConditioningSpec]:
    from src.world_model.humanoid_readiness.common import load_jsonl

    return [
        Phase7HypernetworkConditioningSpec.from_dict(row)
        for row in load_jsonl(path)
    ]


def load_phase7_hypernetwork_output_heads(
    path: str | Path,
) -> list[Phase7HypernetworkOutputHeadSpec]:
    from src.world_model.humanoid_readiness.common import load_jsonl

    return [Phase7HypernetworkOutputHeadSpec.from_dict(row) for row in load_jsonl(path)]


def load_phase7_meta_composition_losses(
    path: str | Path,
) -> list[Phase7MetaCompositionLossDefinition]:
    payload = load_json(path)
    return [
        Phase7MetaCompositionLossDefinition.from_dict(item)
        for item in list(payload.get("definitions") or [])
    ]


def load_phase7_hypernetwork_dataset_contract(
    path: str | Path,
) -> Phase7HypernetworkDatasetContract:
    return Phase7HypernetworkDatasetContract.from_dict(load_json(path))


def load_phase7_hypernetwork_model_config(
    path: str | Path,
) -> Phase7HypernetworkModelComponentConfig:
    return Phase7HypernetworkModelComponentConfig.from_dict(load_json(path))


def load_phase7_hypernetwork_cpu_smoke_forward(
    path: str | Path,
) -> Phase7HypernetworkCPUSmokeForwardReceipt:
    return Phase7HypernetworkCPUSmokeForwardReceipt.from_dict(load_json(path))
