"""
Meta-Transformer scaffold that arbitrates between semantic (DINO) and affordance (OpenVLA) features.
No training or heavy logic; provides placeholder methods and typed dataclasses.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional
import numpy as np
from src.config.objective_profile import ObjectiveVector
from src.evidence.preconditions import build_execution_preconditions, build_execution_work_order
from src.orchestrator.context import OrchestratorContext
from src.orchestrator.semantic_transformer_bridge import (
    build_semantic_orchestration_plan,
    build_semantic_world_model_summary,
    coerce_semantic_world_model,
    derive_backend,
    derive_data_mix_weights,
    derive_energy_profile_mix,
    derive_objective_preset,
    encode_semantic_world_model_features,
    estimate_expected_deltas,
    semantic_tokens,
)
from src.valuation.reward_builder import build_reward_terms, combine_reward, default_objective_vector
from src.config.econ_params import EconParams


@dataclass
class MetaTransformerOutputs:
    """
    Canonical meta-transformer output that includes policy/diffusion embeddings
    and advisory profile suggestions (objective preset, energy/data mixes).
    """

    objective_preset: str = "balanced"
    energy_profile_weights: Dict[str, float] = field(default_factory=dict)
    data_mix_weights: Dict[str, float] = field(default_factory=dict)
    chosen_backend: str = "pybullet"
    expected_delta_mpl: float = 0.0
    expected_delta_error: float = 0.0
    expected_delta_energy_Wh: float = 0.0
    orchestration_plan: List[Any] = field(default_factory=list)

    # Embedding outputs (optional; downstream may ignore)
    shared_policy_state: Optional[np.ndarray] = None
    diffusion_conditioning: Optional[np.ndarray] = None
    ontology_tokens: List[str] = field(default_factory=list)
    affordance_summary: Dict[str, Any] = field(default_factory=dict)
    authority: str = "dino"  # "dino" or "vla"
    execution_mode: str = "advisory"
    bounded_actions: List[str] = field(default_factory=list)
    execution_preconditions: Dict[str, Any] = field(default_factory=dict)
    execution_work_order: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Backward-compatible alias
MetaTransformerOutput = MetaTransformerOutputs


class MetaTransformer:
    def __init__(self, d_shared: int = 32):
        self.d_shared = d_shared

    def integrate_embeddings(
        self,
        dino_features: np.ndarray,
        vla_features: np.ndarray,
        semantic_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        streams = [np.asarray(dino_features, dtype=np.float32), np.asarray(vla_features, dtype=np.float32)]
        if semantic_features is not None:
            streams.append(np.asarray(semantic_features, dtype=np.float32))
        combined = np.concatenate(streams)
        if combined.size < self.d_shared:
            combined = np.pad(combined, (0, self.d_shared - combined.size))
        return combined[: self.d_shared]

    def select_authority(self, dino_conf: float, vla_conf: float) -> str:
        return "dino" if dino_conf >= vla_conf else "vla"

    def produce_policy_state(self, shared: np.ndarray) -> np.ndarray:
        return shared

    def produce_diffusion_conditioning(self, shared: np.ndarray) -> np.ndarray:
        return shared

    def produce_ontology_tokens(self, shared: np.ndarray) -> List[str]:
        return ["meta_token"]

    def produce_affordance_summary(self, vla_features: np.ndarray) -> Dict[str, Any]:
        return {"affordance_norm": float(np.linalg.norm(vla_features))}

    def forward(
        self,
        dino_features: np.ndarray,
        vla_features: np.ndarray,
        dino_conf: float = 0.5,
        vla_conf: float = 0.5,
        semantic_features: Optional[np.ndarray] = None,
    ) -> MetaTransformerOutputs:
        shared = self.integrate_embeddings(dino_features, vla_features, semantic_features=semantic_features)
        authority = self.select_authority(dino_conf, vla_conf)
        return MetaTransformerOutputs(
            shared_policy_state=self.produce_policy_state(shared),
            diffusion_conditioning=self.produce_diffusion_conditioning(shared),
            ontology_tokens=self.produce_ontology_tokens(shared),
            affordance_summary=self.produce_affordance_summary(vla_features),
            authority=authority,
            objective_preset="balanced",
            energy_profile_weights={},
            data_mix_weights={},
            chosen_backend="pybullet",
            expected_delta_mpl=0.0,
            expected_delta_error=0.0,
            expected_delta_energy_Wh=0.0,
            orchestration_plan=[],
        )

    def _mapping(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                payload = to_dict()
                if isinstance(payload, Mapping):
                    return dict(payload)
            except Exception:
                return {}
        return {}

    def _build_feature_streams(
        self,
        *,
        econ_signals: Mapping[str, Any],
        datapack_signals: Mapping[str, Any],
        semantic_summary: Mapping[str, Any],
        perception_embeddings: Optional[Any] = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        semantic_features = encode_semantic_world_model_features(semantic_summary)
        if isinstance(perception_embeddings, Mapping):
            dino_payload = np.asarray(perception_embeddings.get("dino_features", []), dtype=np.float32)
            vla_payload = np.asarray(perception_embeddings.get("vla_features", []), dtype=np.float32)
            dino_conf = float(perception_embeddings.get("dino_confidence", 0.5))
            vla_conf = float(perception_embeddings.get("vla_confidence", 0.5))
            if dino_payload.size > 0 and vla_payload.size > 0:
                return dino_payload, vla_payload, dino_conf, vla_conf
        dino_features = np.array(
            [
                float(econ_signals.get("mpl_urgency", 0.0)),
                float(econ_signals.get("error_urgency", 0.0)),
                float(econ_signals.get("energy_urgency", 0.0)),
                float(datapack_signals.get("data_coverage_score", 0.0)),
                float(datapack_signals.get("semantic_tag_diversity", 0.0)) / 32.0,
                float(semantic_summary.get("capability_mean", 0.0)),
                float(semantic_summary.get("risk_object_fraction", 0.0)),
                float(semantic_summary.get("recovery_router_score", 0.0)),
            ],
            dtype=np.float32,
        )
        vla_features = np.array(
            [
                float(datapack_signals.get("vla_annotation_fraction", 0.0)),
                float(datapack_signals.get("guidance_annotation_fraction", 0.0)),
                float(datapack_signals.get("embedding_diversity", 0.0)),
                float(semantic_summary.get("affordance_density", 0.0)),
                float(semantic_summary.get("fusion_bridge", 0.0)),
                float(semantic_summary.get("stage2_bridge", 0.0)),
                float(semantic_summary.get("efficiency_router_score", 0.0)),
                float(semantic_summary.get("object_memory", 0.0)),
            ],
            dtype=np.float32,
        )
        return dino_features, vla_features, float(max(0.35, semantic_features[10] if semantic_features.size > 10 else 0.5)), float(
            max(0.35, semantic_features[12] if semantic_features.size > 12 else 0.5)
        )

    def propose_plan(
        self,
        *,
        econ_signals: Any,
        datapack_signals: Any,
        perception_embeddings: Optional[Any] = None,
        semantic_world_model: Optional[Any] = None,
        semantic_snapshot: Optional[Any] = None,
        orchestrator_context: Optional[OrchestratorContext] = None,
    ) -> MetaTransformerOutputs:
        econ_payload = self._mapping(econ_signals)
        datapack_payload = self._mapping(datapack_signals)
        if isinstance(perception_embeddings, Mapping):
            semantic_world_model = semantic_world_model or perception_embeddings.get("semantic_world_model")
            semantic_snapshot = semantic_snapshot or perception_embeddings.get("semantic_snapshot")
        world_model = coerce_semantic_world_model(
            semantic_world_model,
            semantic_snapshot=semantic_snapshot,
            context=orchestrator_context,
        )
        semantic_summary = build_semantic_world_model_summary(
            world_model,
            semantic_snapshot=semantic_snapshot,
            context=orchestrator_context,
        )
        semantic_features = encode_semantic_world_model_features(semantic_summary)
        dino_features, vla_features, dino_conf, vla_conf = self._build_feature_streams(
            econ_signals=econ_payload,
            datapack_signals=datapack_payload,
            semantic_summary=semantic_summary,
            perception_embeddings=perception_embeddings,
        )
        output = self.forward(
            dino_features=dino_features,
            vla_features=vla_features,
            dino_conf=dino_conf,
            vla_conf=vla_conf,
            semantic_features=semantic_features,
        )
        objective_preset = derive_objective_preset(
            semantic_summary,
            econ_signals=econ_payload,
            datapack_signals=datapack_payload,
        )
        energy_profile_weights = derive_energy_profile_mix(
            semantic_summary,
            econ_signals=econ_payload,
            objective_preset=objective_preset,
        )
        data_mix_weights = derive_data_mix_weights(
            semantic_summary,
            datapack_signals=datapack_payload,
        )
        chosen_backend = derive_backend(
            semantic_summary,
            econ_signals=econ_payload,
            current_backend=str(getattr(orchestrator_context, "engine_type", "pybullet")),
        )
        deltas = estimate_expected_deltas(
            semantic_summary,
            econ_signals=econ_payload,
            datapack_signals=datapack_payload,
        )
        plan = build_semantic_orchestration_plan(
            semantic_summary,
            objective_preset=objective_preset,
            data_mix_weights=data_mix_weights,
            energy_profile_weights=energy_profile_weights,
            datapack_signals=datapack_payload,
        )
        readiness = build_execution_preconditions(
            subject_id=str(
                semantic_summary.get("task_id")
                or getattr(orchestrator_context, "task_type", "")
                or "meta_transformer"
            ),
            subject_kind="meta_transformer",
            artifact_refs={
                "semantic_world_model_id": semantic_summary.get("world_model_id"),
            },
            required_artifact_refs=["semantic_world_model_id"],
            signal_values={
                "semantic_present": 1.0 if semantic_summary.get("present") else 0.0,
                "object_count": semantic_summary.get("object_count", 0.0),
                "grounded_track_object_count": semantic_summary.get("grounded_track_object_count", 0.0),
                "capability_mean": semantic_summary.get("capability_mean", 0.0),
                "meta_node_orchestration": semantic_summary.get("meta_node_orchestration", 0.0),
                "data_coverage_score": datapack_payload.get("data_coverage_score", 0.0),
            },
            min_signal_thresholds={
                "object_count": 1.0,
                "capability_mean": 0.15,
            },
            required_boolean_signals={"semantic_present": True},
            soft_min_signal_thresholds={
                "grounded_track_object_count": 1.0,
                "data_coverage_score": 0.1,
                "meta_node_orchestration": 0.2,
            },
            metadata={
                "semantic_world_model_summary": semantic_summary,
                "objective_preset": objective_preset,
            },
        )
        bounded_actions = [
            "set_objective_preset",
            "set_energy_profile",
            "set_data_mix",
            "set_backend",
            "route_meta_nodes",
        ]
        if float(semantic_summary.get("wm_validation_error_rate", 0.0)) >= 0.2:
            bounded_actions.append("request_wm_state_validation")
        if float(semantic_summary.get("graph_mutation_pressure", 0.0)) >= 1.0:
            bounded_actions.append("queue_graph_mutation_review")
        if float(semantic_summary.get("trust_overlay_mean", 0.0)) < 0.45:
            bounded_actions.append("route_trust_recalibration")
        if float(semantic_summary.get("econ_overlay_mean", 0.0)) >= 0.5:
            bounded_actions.append("prioritize_gap_fill")
        execution_mode = "bounded_execution" if readiness.ready else "advisory"
        work_order = build_execution_work_order(
            order_type="transformer_routing",
            subject_id=str(
                semantic_summary.get("task_id")
                or getattr(orchestrator_context, "task_type", "")
                or "meta_transformer"
            ),
            subject_kind="meta_transformer",
            decision="activate_meta_transformer_routing",
            priority=float(max(semantic_summary.get("capability_mean", 0.0), 0.1)),
            recommended_mode=execution_mode,
            readiness=readiness,
            reasons=bounded_actions,
            artifact_refs={"semantic_world_model_id": semantic_summary.get("world_model_id")},
            metadata={"semantic_world_model_summary": semantic_summary},
        ).to_dict()
        output.objective_preset = objective_preset
        output.energy_profile_weights = energy_profile_weights
        output.data_mix_weights = data_mix_weights
        output.chosen_backend = chosen_backend
        output.expected_delta_mpl = deltas["expected_delta_mpl"]
        output.expected_delta_error = deltas["expected_delta_error"]
        output.expected_delta_energy_Wh = deltas["expected_delta_energy_Wh"]
        output.orchestration_plan = plan
        output.ontology_tokens = semantic_tokens(semantic_summary) or output.ontology_tokens
        output.affordance_summary = {
            **dict(output.affordance_summary or {}),
            "semantic_top_objects": list(semantic_summary.get("top_object_labels", []) or []),
            "semantic_top_meta_nodes": list(semantic_summary.get("top_meta_nodes", []) or []),
            "semantic_affordance_density": float(semantic_summary.get("affordance_density", 0.0)),
        }
        output.execution_mode = execution_mode
        output.bounded_actions = bounded_actions
        output.execution_preconditions = readiness.to_dict()
        output.execution_work_order = work_order
        output.metadata = {
            "semantic_world_model_summary": semantic_summary,
            "semantic_world_model_id": semantic_summary.get("world_model_id"),
            "active_capabilities": list(semantic_summary.get("active_capabilities", []) or []),
            "top_meta_nodes": list(semantic_summary.get("top_meta_nodes", []) or []),
            "coverage_feedback_summary": dict(semantic_summary.get("coverage_feedback_summary", {}) or {}),
            "wm_validation_summary": dict(semantic_summary.get("wm_validation_summary", {}) or {}),
        }
        return output

    # ---- Helper utilities for downstream components ----
    def derive_expected_objectives(self, meta_out: MetaTransformerOutputs, ctx: OrchestratorContext) -> List[float]:
        """
        Map meta-transformer advisory output to a numeric objective vector.

        Does NOT change any reward path; purely advisory for context updates.
        """
        if meta_out.objective_preset and meta_out.objective_preset != "balanced":
            try:
                return ObjectiveVector.from_preset(meta_out.objective_preset).to_list()
            except Exception:
                pass
        return ctx.objective_vector if hasattr(ctx, "objective_vector") else default_objective_vector()

    def predict_backend(self, meta_out: MetaTransformerOutputs) -> str:
        """Return backend suggestion from meta output (default pybullet)."""
        return meta_out.chosen_backend or "pybullet"

    def predict_expected_delta(self, meta_out: MetaTransformerOutputs) -> Dict[str, float]:
        """Extract expected deltas for logging/attribution."""
        return {
            "expected_delta_mpl": meta_out.expected_delta_mpl,
            "expected_delta_error": meta_out.expected_delta_error,
            "expected_delta_energy_Wh": meta_out.expected_delta_energy_Wh,
        }

    def validate_reward_round_trip(
        self,
        meta_out: MetaTransformerOutputs,
        ctx: OrchestratorContext,
        econ_params: Optional[EconParams] = None,
    ) -> Dict[str, Any]:
        """
        Lightweight round-trip validation: derive objective vector, build dummy reward terms,
        combine reward (advisory only). Does NOT alter live reward paths.
        """
        objective_vector = self.derive_expected_objectives(meta_out, ctx)
        # Minimal summary for reward terms
        dummy_summary = type("DummySummary", (), {})()
        dummy_summary.mpl_episode = getattr(ctx, "mean_delta_mpl", 0.0)
        dummy_summary.error_rate_episode = getattr(ctx, "mean_delta_error", 0.0)
        dummy_summary.energy_Wh = getattr(ctx, "mean_delta_j", 0.0)
        dummy_summary.energy_Wh_per_unit = getattr(ctx, "mean_delta_j", 0.0)
        dummy_summary.energy_Wh_per_hour = getattr(ctx, "mean_delta_j", 0.0)
        dummy_summary.ep_episode = getattr(ctx, "mean_delta_j", 0.0)
        dummy_summary.throughput_units_per_hour = getattr(ctx, "mean_delta_mpl", 0.0)
        dummy_summary.wage_parity = 1.0
        dummy_summary.limb_energy_Wh = {}
        dummy_summary.skill_energy_Wh = {}
        dummy_summary.energy_per_limb = {}
        dummy_summary.energy_per_skill = {}
        dummy_summary.energy_per_joint = {}
        dummy_summary.energy_per_effector = {}
        dummy_summary.coordination_metrics = {}
        dummy_summary.episode_id = ""
        dummy_summary.media_refs = {}
        terms = build_reward_terms(dummy_summary, econ_params or EconParams(
            price_per_unit=0.3,
            damage_cost=1.0,
            energy_Wh_per_attempt=0.1,
            time_step_s=60.0,
            base_rate=1.0,
            p_min=0.02,
            k_err=0.1,
            q_speed=1.0,
            q_care=1.0,
            care_cost=0.1,
            max_steps=100,
            max_catastrophic_errors=3,
            max_error_rate_sla=0.5,
            min_steps_for_sla=1,
            zero_throughput_patience=5,
            preset="meta_dummy",
        ))
        combined = combine_reward(objective_vector, terms)
        return {
            "objective_vector": objective_vector,
            "reward_terms": terms,
            "combined_reward": combined,
        }
