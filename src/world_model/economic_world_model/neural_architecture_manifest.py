"""Economic WM neural architecture manifest scaffolding.

The manifest names the future Economic WM estimator, dynamics, allocator, and
governance neural surfaces after canonical lower-WM consumption is available.
It is a topology/training-contract artifact only. It does not instantiate model
weights or claim training/promotion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.lower_wm_consumption import (
    EconomicWMLowerWMConsumptionPreflight,
    load_economic_wm_lower_wm_consumption_preflight,
)

ECONOMIC_WM_NEURAL_COMPONENT_SPEC_VERSION = "economic_wm_neural_component_spec_v1"
ECONOMIC_WM_NEURAL_ARCHITECTURE_MANIFEST_VERSION = (
    "economic_wm_neural_architecture_manifest_v1"
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


@dataclass(frozen=True)
class EconomicWMNeuralComponentSpec:
    """One future learned component in the Economic WM."""

    component_id: str
    component_key: str
    role: str
    model_family: str
    architecture_pattern: str
    input_surfaces: list[str] = field(default_factory=list)
    output_surfaces: list[str] = field(default_factory=list)
    training_rows: list[str] = field(default_factory=list)
    training_signals: list[str] = field(default_factory=list)
    loss_families: list[str] = field(default_factory=list)
    promotion_gates: list[str] = field(default_factory=list)
    authority_boundary: str = "shadow_only_until_promoted"
    authority_class: str = "neural_scaffold_only"
    runtime_plane: str = "gpu_train_required"
    training_ready: bool = False
    promotion_eligible: bool = False
    estimated_parameter_band: str = "unknown"
    timescale: str = "meso"
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_NEURAL_COMPONENT_SPEC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "version": self.version,
            "component_key": self.component_key,
            "role": self.role,
            "component_role": self.role,
            "model_family": self.model_family,
            "architecture_pattern": self.architecture_pattern,
            "input_surfaces": list(self.input_surfaces),
            "output_surfaces": list(self.output_surfaces),
            "training_rows": list(self.training_rows),
            "training_signals": list(self.training_signals),
            "loss_families": list(self.loss_families),
            "promotion_gates": list(self.promotion_gates),
            "authority_boundary": self.authority_boundary,
            "authority_class": self.authority_class,
            "runtime_plane": self.runtime_plane,
            "training_ready": bool(self.training_ready),
            "promotion_eligible": bool(self.promotion_eligible),
            "estimated_parameter_band": self.estimated_parameter_band,
            "timescale": self.timescale,
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMNeuralComponentSpec":
        return cls(
            component_id=str(payload.get("component_id", "")),
            component_key=str(payload.get("component_key", "")),
            role=str(payload.get("role", payload.get("component_role", ""))),
            model_family=str(payload.get("model_family", "")),
            architecture_pattern=str(payload.get("architecture_pattern", "")),
            input_surfaces=[
                str(item) for item in list(payload.get("input_surfaces", []) or [])
            ],
            output_surfaces=[
                str(item) for item in list(payload.get("output_surfaces", []) or [])
            ],
            training_rows=[
                str(item) for item in list(payload.get("training_rows", []) or [])
            ],
            training_signals=[
                str(item) for item in list(payload.get("training_signals", []) or [])
            ],
            loss_families=[
                str(item) for item in list(payload.get("loss_families", []) or [])
            ],
            promotion_gates=[
                str(item) for item in list(payload.get("promotion_gates", []) or [])
            ],
            authority_boundary=str(
                payload.get("authority_boundary", "shadow_only_until_promoted")
            ),
            authority_class=str(payload.get("authority_class", "neural_scaffold_only")),
            runtime_plane=str(payload.get("runtime_plane", "gpu_train_required")),
            training_ready=bool(payload.get("training_ready", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            estimated_parameter_band=str(
                payload.get("estimated_parameter_band", "unknown")
            ),
            timescale=str(payload.get("timescale", "meso")),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_NEURAL_COMPONENT_SPEC_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMNeuralArchitectureManifest:
    """Topology manifest for future Economic WM learned modules."""

    manifest_id: str
    lower_wm_preflight_id: str
    architecture_stage: str
    components: list[EconomicWMNeuralComponentSpec] = field(default_factory=list)
    input_contracts: list[str] = field(default_factory=list)
    output_contracts: list[str] = field(default_factory=list)
    training_blockers: list[str] = field(default_factory=list)
    provider_blockers: list[str] = field(default_factory=list)
    gpu_training_ready: bool = False
    provider_bringup_ready: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    authority_class: str = "neural_manifest_only"
    ready_for_training_scaffold: bool = False
    ready_for_gpu_training: bool = False
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_NEURAL_ARCHITECTURE_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "lower_wm_preflight_id": self.lower_wm_preflight_id,
            "architecture_stage": self.architecture_stage,
            "components": [component.to_dict() for component in self.components],
            "input_contracts": list(self.input_contracts),
            "output_contracts": list(self.output_contracts),
            "training_blockers": list(self.training_blockers),
            "provider_blockers": list(self.provider_blockers),
            "gpu_training_ready": bool(self.gpu_training_ready),
            "provider_bringup_ready": bool(self.provider_bringup_ready),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "authority_class": self.authority_class,
            "ready_for_training_scaffold": bool(self.ready_for_training_scaffold),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
            "aggregate_counts": {
                str(key): float(value) for key, value in self.aggregate_counts.items()
            },
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMNeuralArchitectureManifest":
        return cls(
            manifest_id=str(payload.get("manifest_id", "")),
            lower_wm_preflight_id=str(payload.get("lower_wm_preflight_id", "")),
            architecture_stage=str(payload.get("architecture_stage", "planned")),
            components=[
                EconomicWMNeuralComponentSpec.from_dict(item)
                for item in list(payload.get("components", []) or [])
            ],
            input_contracts=[
                str(item) for item in list(payload.get("input_contracts", []) or [])
            ],
            output_contracts=[
                str(item) for item in list(payload.get("output_contracts", []) or [])
            ],
            training_blockers=[
                str(item) for item in list(payload.get("training_blockers", []) or [])
            ],
            provider_blockers=[
                str(item) for item in list(payload.get("provider_blockers", []) or [])
            ],
            gpu_training_ready=bool(payload.get("gpu_training_ready", False)),
            provider_bringup_ready=bool(payload.get("provider_bringup_ready", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            authority_class=str(payload.get("authority_class", "neural_manifest_only")),
            ready_for_training_scaffold=bool(
                payload.get("ready_for_training_scaffold", False)
            ),
            ready_for_gpu_training=bool(payload.get("ready_for_gpu_training", False)),
            aggregate_counts={
                str(key): float(value)
                for key, value in dict(
                    payload.get("aggregate_counts", {}) or {}
                ).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_NEURAL_ARCHITECTURE_MANIFEST_VERSION)
            ),
        )


def _component(
    *,
    component_key: str,
    role: str,
    model_family: str,
    architecture_pattern: str,
    input_surfaces: Iterable[str],
    output_surfaces: Iterable[str],
    training_rows: Iterable[str],
    training_signals: Iterable[str],
    loss_families: Iterable[str],
    promotion_gates: Iterable[str],
    estimated_parameter_band: str,
    timescale: str,
    blockers: Iterable[str],
    runtime_plane: str = "gpu_train_required",
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMNeuralComponentSpec:
    payload = {
        "component_key": component_key,
        "role": role,
        "model_family": model_family,
        "architecture_pattern": architecture_pattern,
        "input_surfaces": list(input_surfaces),
        "output_surfaces": list(output_surfaces),
        "version": ECONOMIC_WM_NEURAL_COMPONENT_SPEC_VERSION,
    }
    return EconomicWMNeuralComponentSpec(
        component_id=f"ewm_neural_component_{sha256_json(payload)[:16]}",
        component_key=component_key,
        role=role,
        model_family=model_family,
        architecture_pattern=architecture_pattern,
        input_surfaces=list(input_surfaces),
        output_surfaces=list(output_surfaces),
        training_rows=list(training_rows),
        training_signals=list(training_signals),
        loss_families=list(loss_families),
        promotion_gates=list(promotion_gates),
        authority_boundary="shadow_only_until_promoted",
        authority_class="neural_scaffold_only",
        runtime_plane=runtime_plane,
        training_ready=False,
        promotion_eligible=False,
        estimated_parameter_band=estimated_parameter_band,
        timescale=timescale,
        blockers=list(blockers),
        metadata={"training_claim": False, **_mapping(metadata)},
    )


def _planned_components(
    preflight: EconomicWMLowerWMConsumptionPreflight,
) -> list[EconomicWMNeuralComponentSpec]:
    base_rows = [
        "economic_wm_replay_feature_row_v1",
        "economic_wm_canonical_consumption_row_v1",
    ]
    common_gates = [
        "lower_wm_canonical_consumption_preflight_ok",
        "non_stub_teacher_runtime_invocation_receipt",
        "provider_runtime_truth_receipts",
        "gpu_training_runtime_receipt",
        "promotion_grade_benchmark_evidence",
        "no_reward_math_mutation",
    ]
    common_blockers = [
        "gpu_training_not_run",
        "provider_bringup_not_run",
        "promotion_grade_benchmark_evidence_missing",
        "non_stub_teacher_runtime_not_verified",
    ]
    return [
        _component(
            component_key="datapack_composition_network",
            role="Encode cross-WM source composition, lineage, validation, and functional contribution.",
            model_family="heterogeneous_graph_set_temporal_encoder",
            architecture_pattern="Mereotopological datapack encoder with graph lineage edges, set pooling, temporal positions, and objective conditioning.",
            input_surfaces=[
                "economic_wm_canonical_consumption_row_v1",
                "PerceptionGroundingWorldState",
                "SimSynthPhysicsWorldState",
                "EmbodimentActuationWorldState",
                "runtime_packet_v1",
                "value_target_pack_v1",
                "governance_trace_v1",
            ],
            output_surfaces=[
                "DatapackCompositionEmbedding",
                "FunctionalContributionComposition",
                "PredictedMarginalUtilitySeed",
                "RecommendedUseClass",
            ],
            training_rows=base_rows,
            training_signals=[
                "benchmark_training_weight",
                "shadow_gap_weight",
                "reconstruction_training_weight",
                "downstream_training_improvement_receipts",
            ],
            loss_families=[
                "supervised_marginal_utility_regression",
                "contrastive_use_class_separation",
                "lineage_topology_preservation",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band="2M-20M",
            timescale="per-datapack/per-corpus-window",
            blockers=common_blockers,
            metadata={"preflight_id": preflight.preflight_id},
        ),
        _component(
            component_key="economic_state_estimator",
            role="Estimate EconomicState, EconomicRegime, bottlenecks, slow manifold state, and shadow-price seeds from lower-WM receipts.",
            model_family="regime_switching_state_space_model",
            architecture_pattern="DS3M / RED-SDS-style discrete regime latents plus continuous economic state, with explicit-duration regime persistence.",
            input_surfaces=[
                "EconomicState",
                "economic_wm_canonical_consumption_row_v1",
                "PerceptionGroundingWorldState",
                "SimSynthPhysicsWorldState",
                "EmbodimentActuationWorldState",
                "provider_invocation_receipt_v1",
                "training_runtime_manifest_v1",
            ],
            output_surfaces=[
                "EconomicRegime",
                "SlowManifoldProjection",
                "BottleneckMap",
                "OpportunityField",
                "ShadowPriceFieldSeed",
            ],
            training_rows=base_rows,
            training_signals=[
                "regime_labels_from_receipt_windows",
                "bottleneck_recurrence",
                "provider_gap_weight",
                "gpu_training_deferred_weight",
            ],
            loss_families=[
                "sequence_likelihood",
                "regime_duration_prediction",
                "bottleneck_multilabel_prediction",
                "slow_fast_consistency_regularization",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band="5M-50M",
            timescale="fast_to_slow_projection",
            blockers=common_blockers,
            metadata={
                "candidate_families": ["DS3M", "RED-SDS", "regime-aware sequence model"]
            },
        ),
        _component(
            component_key="economic_dynamics_model",
            role="Forecast economic transitions and counterfactual outcomes under candidate allocations.",
            model_family="regime_conditioned_counterfactual_sequence_model",
            architecture_pattern="Typed receipt sequence model conditioned on SlowManifoldProjection and candidate AllocationEnvelope deltas.",
            input_surfaces=[
                "EconomicState",
                "SlowManifoldProjection",
                "AllocationEnvelope",
                "EconomicCounterfactual",
                "lower_wm_receipt_windows",
            ],
            output_surfaces=[
                "EconomicTransition",
                "EconomicCounterfactualForecast",
                "ResourceDissipationForecast",
                "FutureBottleneckTrajectory",
            ],
            training_rows=base_rows,
            training_signals=[
                "counterfactual_eval_v1",
                "value_ledger_receipt_v1",
                "realized_training_improvement",
                "provider_runtime_outcome",
            ],
            loss_families=[
                "counterfactual_forecast_error",
                "resource_trajectory_prediction",
                "uncertainty_calibration",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band="5M-100M",
            timescale="per-episode/per-shift forecast",
            blockers=common_blockers,
        ),
        _component(
            component_key="distributional_pareto_allocator",
            role="Compile estimated state and forecasts into distributional Pareto allocation surfaces.",
            model_family="distributional_multi_objective_allocator",
            architecture_pattern="DPMORL/PGMORL-inspired frontier model with coherent-risk filtering and augmented-Lagrangian shadow prices.",
            input_surfaces=[
                "EconomicState",
                "EconomicRegime",
                "EconomicCounterfactualForecast",
                "DatapackCompositionEmbedding",
                "resource_constraints",
            ],
            output_surfaces=[
                "ParetoFrontierSlice",
                "ShadowPriceField",
                "RiskField",
                "AllocationEnvelope",
                "TrainingSlicePriorityField",
            ],
            training_rows=base_rows,
            training_signals=[
                "multi_objective_return_distribution",
                "throughput_energy_wear_compute_error_tradeoffs",
                "chosen_vs_unchosen_counterfactuals",
            ],
            loss_families=[
                "distributional_pareto_frontier_loss",
                "cvar_tail_risk_loss",
                "lagrangian_constraint_violation_loss",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band="2M-30M",
            timescale="meso allocation window",
            blockers=common_blockers,
        ),
        _component(
            component_key="discrete_receding_horizon_allocator",
            role="Solve bounded finite-set routing choices for compute, sim budget, replay slices, and queue dispatch.",
            model_family="finite_set_receding_horizon_solver_with_amortized_warm_start",
            architecture_pattern="QUBO/Ising-inspired compile lane wrapped by typed EconomicState and Pareto context; conventional solver first.",
            input_surfaces=[
                "EconomicState",
                "ParetoFrontierSlice",
                "ShadowPriceField",
                "candidate_work_orders",
                "finite_resource_action_set",
            ],
            output_surfaces=[
                "DiscreteAllocationPlan",
                "SolverTraceReceipt",
                "AllocationEnvelopeDelta",
            ],
            training_rows=base_rows,
            training_signals=[
                "work_order_completion_outcomes",
                "queue_relief",
                "resource_cost_realization",
            ],
            loss_families=[
                "warm_start_plan_imitation",
                "constraint_satisfaction",
                "receding_horizon_regret",
            ],
            promotion_gates=common_gates,
            estimated_parameter_band="solver + optional 100K-2M warm-start proposer",
            timescale="short-horizon discrete allocation",
            blockers=common_blockers,
            runtime_plane="local_solver_scaffold_gpu_training_optional",
            metadata={
                "optional_acceleration": "Ising/QUBO hardware is future optional, not required"
            },
        ),
        _component(
            component_key="governance_reciprocity_compiler",
            role="Convert economic allocation outputs into typed downward shaping and admissible-region transport without becoming sovereign.",
            model_family="typed_governance_field_compiler",
            architecture_pattern="Small calibrated compiler from Pareto/risk/shadow-price surfaces to lower-WM budget envelopes, persistence annotations, and admissible slices.",
            input_surfaces=[
                "AllocationEnvelope",
                "ParetoFrontierSlice",
                "RiskField",
                "ShadowPriceField",
                "lower_wm_readiness_receipts",
            ],
            output_surfaces=[
                "ShapingField",
                "BudgetEnvelope",
                "PersistenceAnnotation",
                "AdmissibleParetoSlice",
                "EconomicRunReceipt",
            ],
            training_rows=base_rows,
            training_signals=[
                "lower_wm_response_receipts",
                "governance_satisfaction",
                "anti_reward_hacking_flags",
                "deployment_truth_feedback",
            ],
            loss_families=[
                "governance_satisfaction_loss",
                "persistence_hysteresis_loss",
                "downstream_response_calibration",
            ],
            promotion_gates=[
                *common_gates,
                "meta_regal_override_trace_available_before_authority",
            ],
            estimated_parameter_band="500K-10M",
            timescale="downward transport / meso governance",
            blockers=[*common_blockers, "meta_regal_governance_not_built"],
        ),
    ]


def build_economic_wm_neural_architecture_manifest(
    *,
    lower_wm_preflight: EconomicWMLowerWMConsumptionPreflight,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMNeuralArchitectureManifest:
    components = _planned_components(lower_wm_preflight)
    component_blockers = _unique(
        blocker for component in components for blocker in component.blockers
    )
    training_blockers = _unique(
        [
            *component_blockers,
            "gpu_training_not_run",
            "provider_bringup_not_run",
            "promotion_grade_benchmark_evidence_missing",
            "non_stub_teacher_runtime_not_verified",
        ]
    )
    provider_blockers = [
        blocker
        for blocker in training_blockers
        if "provider" in blocker or "teacher" in blocker
    ]
    input_contracts = _unique(
        surface for component in components for surface in component.input_surfaces
    )
    output_contracts = _unique(
        surface for component in components for surface in component.output_surfaces
    )
    gpu_required_count = sum(
        1 for component in components if component.runtime_plane == "gpu_train_required"
    )
    aggregate_counts = {
        "component_count": float(len(components)),
        "gpu_train_required_count": float(gpu_required_count),
        "local_scaffold_component_count": float(len(components)),
        "input_contract_count": float(len(input_contracts)),
        "output_contract_count": float(len(output_contracts)),
        "training_blocker_count": float(len(training_blockers)),
        "lower_wm_reference_row_count": float(lower_wm_preflight.row_count),
    }
    payload = {
        "lower_wm_preflight_id": lower_wm_preflight.preflight_id,
        "component_ids": [component.component_id for component in components],
        "version": ECONOMIC_WM_NEURAL_ARCHITECTURE_MANIFEST_VERSION,
    }
    return EconomicWMNeuralArchitectureManifest(
        manifest_id=f"ewm_neural_arch_manifest_{sha256_json(payload)[:16]}",
        lower_wm_preflight_id=lower_wm_preflight.preflight_id,
        architecture_stage="pretraining_topology_scaffold",
        components=components,
        input_contracts=input_contracts,
        output_contracts=output_contracts,
        training_blockers=training_blockers,
        provider_blockers=provider_blockers,
        gpu_training_ready=False,
        provider_bringup_ready=False,
        promotion_eligible=False,
        reward_math_mutation=False,
        authority_class="neural_manifest_only",
        ready_for_training_scaffold=bool(lower_wm_preflight.ready_for_neural_manifest),
        ready_for_gpu_training=False,
        aggregate_counts=aggregate_counts,
        artifact_refs={
            "lower_wm_preflight_id": lower_wm_preflight.preflight_id,
            **_mapping(lower_wm_preflight.artifact_refs),
            **_mapping(artifact_refs),
        },
        metadata={
            "boundary": "neural topology manifest only; no weights, training, provider bring-up, or promotion claim",
            "source_lower_wm_preflight_status": lower_wm_preflight.status,
            "economic_wm_is_not_sovereign": True,
            **_mapping(metadata),
        },
    )


def save_economic_wm_neural_architecture_manifest(
    path: str | Path, manifest: EconomicWMNeuralArchitectureManifest
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_economic_wm_neural_architecture_manifest(
    path: str | Path,
) -> EconomicWMNeuralArchitectureManifest:
    return EconomicWMNeuralArchitectureManifest.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def build_economic_wm_neural_architecture_manifest_from_path(
    *,
    lower_wm_preflight_path: str | Path,
    output_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMNeuralArchitectureManifest:
    preflight = load_economic_wm_lower_wm_consumption_preflight(lower_wm_preflight_path)
    manifest = build_economic_wm_neural_architecture_manifest(
        lower_wm_preflight=preflight,
        artifact_refs={"lower_wm_preflight_path": str(lower_wm_preflight_path)},
        metadata=metadata,
    )
    save_economic_wm_neural_architecture_manifest(output_path, manifest)
    return manifest


__all__ = [
    "ECONOMIC_WM_NEURAL_ARCHITECTURE_MANIFEST_VERSION",
    "ECONOMIC_WM_NEURAL_COMPONENT_SPEC_VERSION",
    "EconomicWMNeuralArchitectureManifest",
    "EconomicWMNeuralComponentSpec",
    "build_economic_wm_neural_architecture_manifest",
    "build_economic_wm_neural_architecture_manifest_from_path",
    "load_economic_wm_neural_architecture_manifest",
    "save_economic_wm_neural_architecture_manifest",
]
