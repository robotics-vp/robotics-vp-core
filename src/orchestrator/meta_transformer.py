"""Meta-transformer runtime with bounded learned-helper support."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional
import numpy as np
from src.config.objective_profile import ObjectiveVector
from src.evidence.preconditions import build_execution_preconditions, build_execution_work_order
from src.orchestrator.context import OrchestratorContext
from src.orchestrator.meta_transformer_planning import (
    META_DATA_MIX_LABELS,
    META_ENERGY_PROFILE_LABELS,
    META_EXPECTED_DELTA_LABELS,
    build_meta_planning_context_vector,
    normalize_named_weights,
)
from src.orchestrator.meta_transformer_runtime import (
    LoadedMetaTransformerRuntime,
    MetaTransformerRuntimeInference,
    load_meta_transformer_runtime_package,
)
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
    def __init__(
        self,
        d_shared: int = 32,
        *,
        helper_package_path: Optional[str] = None,
        helper_mode: Literal["disabled", "auto", "required"] = "auto",
    ):
        if helper_mode not in {"disabled", "auto", "required"}:
            raise ValueError(f"Unsupported meta-transformer helper mode: {helper_mode}")
        self.d_shared = d_shared
        self.helper_package_path = helper_package_path
        self.helper_mode = helper_mode
        self._loaded_helper: Optional[LoadedMetaTransformerRuntime] = None
        self._helper_load_error: Optional[str] = None

    def _coerce_vector(self, value: np.ndarray, dim: int) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size < dim:
            vector = np.pad(vector, (0, dim - vector.size))
        return vector[:dim]

    def _blend_vectors(self, base: np.ndarray, learned: np.ndarray, *, weight: float) -> np.ndarray:
        base_vec = self._coerce_vector(base, max(len(base), len(learned)))
        learned_vec = self._coerce_vector(learned, max(len(base), len(learned)))
        return ((1.0 - weight) * base_vec) + (weight * learned_vec)

    def _blend_named_weights(
        self,
        base: Mapping[str, Any],
        learned: Mapping[str, Any],
        *,
        weight: float,
        labels: list[str],
    ) -> Dict[str, float]:
        base_normalized = normalize_named_weights(base, labels)
        learned_normalized = normalize_named_weights(learned, labels)
        blended = {
            str(label): ((1.0 - weight) * float(base_normalized.get(label, 0.0)))
            + (weight * float(learned_normalized.get(label, 0.0)))
            for label in labels
        }
        return normalize_named_weights(blended, labels)

    def _blend_expected_deltas(
        self,
        base: Mapping[str, Any],
        learned: Mapping[str, Any],
        *,
        weight: float,
    ) -> Dict[str, float]:
        base_payload = dict(base or {})
        learned_payload = dict(learned or {})
        return {
            label: float(
                ((1.0 - weight) * float(base_payload.get(label, 0.0)))
                + (weight * float(learned_payload.get(label, 0.0)))
            )
            for label in META_EXPECTED_DELTA_LABELS
        }

    def _normalized_helper_path(self) -> Optional[Path]:
        if not self.helper_package_path:
            return None
        candidate = Path(self.helper_package_path)
        if candidate.is_dir():
            candidate = candidate / "meta_transformer_package.json"
        return candidate

    def _resolve_helper_runtime(self) -> tuple[Optional[LoadedMetaTransformerRuntime], Dict[str, Any]]:
        if self.helper_mode == "disabled":
            return None, {
                "mode": self.helper_mode,
                "status": "disabled",
                "promotion_stage": "disabled",
                "benchmark_gate_ready": False,
            }
        if self._loaded_helper is not None:
            package = self._loaded_helper.package
            benchmark_gate_ready = bool(package.benchmark_gate.get("ready", False))
            if self.helper_mode == "required" and not benchmark_gate_ready:
                raise ValueError(
                    "meta-transformer helper mode 'required' requires a benchmark-gated package"
                )
            return self._loaded_helper, {
                "mode": self.helper_mode,
                "status": "loaded",
                "package_id": package.package_id,
                "package_path": package.package_path,
                "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
                "benchmark_gate_ready": benchmark_gate_ready,
                "unsatisfied_preconditions": list(
                    package.execution_preconditions.get("unsatisfied_preconditions", []) or []
                ),
            }
        if self._helper_load_error is not None:
            if self.helper_mode == "required":
                raise ValueError(self._helper_load_error)
            return None, {
                "mode": self.helper_mode,
                "status": "load_failed",
                "promotion_stage": "heuristic_fallback",
                "benchmark_gate_ready": False,
                "error": self._helper_load_error,
            }

        package_path = self._normalized_helper_path()
        if package_path is None:
            message = "meta-transformer helper package path was not provided"
            if self.helper_mode == "required":
                raise ValueError(message)
            return None, {
                "mode": self.helper_mode,
                "status": "package_missing",
                "promotion_stage": "heuristic_fallback",
                "benchmark_gate_ready": False,
            }
        if not package_path.exists():
            message = f"meta-transformer helper package not found: {package_path}"
            self._helper_load_error = message
            if self.helper_mode == "required":
                raise ValueError(message)
            return None, {
                "mode": self.helper_mode,
                "status": "package_missing",
                "promotion_stage": "heuristic_fallback",
                "benchmark_gate_ready": False,
                "package_path": str(package_path),
            }

        try:
            package = load_meta_transformer_runtime_package(package_path)
            self._loaded_helper = LoadedMetaTransformerRuntime(package)
        except Exception as exc:
            self._helper_load_error = str(exc)
            if self.helper_mode == "required":
                raise ValueError(str(exc)) from exc
            return None, {
                "mode": self.helper_mode,
                "status": "load_failed",
                "promotion_stage": "heuristic_fallback",
                "benchmark_gate_ready": False,
                "package_path": str(package_path),
                "error": str(exc),
            }

        benchmark_gate_ready = bool(self._loaded_helper.package.benchmark_gate.get("ready", False))
        if self.helper_mode == "required" and not benchmark_gate_ready:
            raise ValueError(
                "meta-transformer helper mode 'required' requires a benchmark-gated package"
            )
        return self._loaded_helper, {
            "mode": self.helper_mode,
            "status": "loaded",
            "package_id": self._loaded_helper.package.package_id,
            "package_path": self._loaded_helper.package.package_path,
            "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
            "benchmark_gate_ready": benchmark_gate_ready,
            "unsatisfied_preconditions": list(
                self._loaded_helper.package.execution_preconditions.get(
                    "unsatisfied_preconditions", []
                )
                or []
            ),
        }

    def _apply_learned_helper(
        self,
        output: MetaTransformerOutputs,
        *,
        inference: MetaTransformerRuntimeInference,
        helper_status: Mapping[str, Any],
    ) -> MetaTransformerOutputs:
        helper_weight = 0.55 if inference.benchmark_gate_ready else 0.2
        authority_margin = float(
            inference.authority_confidence - inference.alternate_authority_confidence
        )
        authority = output.authority
        authority_source = "heuristic"
        if inference.benchmark_gate_ready or authority_margin >= 0.2:
            authority = inference.authority
            authority_source = "learned_helper"

        output.shared_policy_state = self._blend_vectors(
            np.asarray(output.shared_policy_state, dtype=np.float32),
            inference.policy_state,
            weight=helper_weight,
        )
        output.diffusion_conditioning = self._blend_vectors(
            np.asarray(output.diffusion_conditioning, dtype=np.float32),
            inference.diffusion_conditioning,
            weight=helper_weight,
        )
        output.ontology_tokens = list(
            dict.fromkeys(list(output.ontology_tokens or []) + list(inference.ontology_tokens or []))
        )[:16]
        output.authority = authority
        planning_trace = dict(inference.planning_trace or {})
        planning_trace.setdefault("objective_distribution", {})
        planning_trace.setdefault("backend_distribution", {})
        output.metadata = {
            **dict(output.metadata or {}),
            "learned_helper": {
                **dict(helper_status),
                "helper_weight": helper_weight,
                "authority": inference.authority,
                "authority_confidence": float(inference.authority_confidence),
                "alternate_authority_confidence": float(
                    inference.alternate_authority_confidence
                ),
                "authority_margin": authority_margin,
                "authority_source": authority_source,
                "predicted_ontology_tokens": list(inference.ontology_tokens),
                "planning_heads_available": bool(inference.planning_heads_available),
                "objective_preset": inference.objective_preset,
                "objective_confidence": float(inference.objective_confidence),
                "objective_alternate_confidence": float(
                    inference.objective_alternate_confidence
                ),
                "chosen_backend": inference.chosen_backend,
                "backend_confidence": float(inference.backend_confidence),
                "backend_alternate_confidence": float(
                    inference.backend_alternate_confidence
                ),
                "energy_profile_weights": dict(inference.energy_profile_weights),
                "data_mix_weights": dict(inference.data_mix_weights),
                "expected_deltas": dict(inference.expected_deltas),
                "planning_trace": planning_trace,
            },
        }
        return output

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
        planning_context: Optional[np.ndarray] = None,
    ) -> MetaTransformerOutputs:
        shared = self.integrate_embeddings(dino_features, vla_features, semantic_features=semantic_features)
        authority = self.select_authority(dino_conf, vla_conf)
        output = MetaTransformerOutputs(
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
        helper_runtime, helper_status = self._resolve_helper_runtime()
        if helper_runtime is None:
            output.metadata = {**dict(output.metadata or {}), "learned_helper": dict(helper_status)}
            return output
        inference = helper_runtime.infer(
            dino_features=np.asarray(dino_features, dtype=np.float32),
            vla_features=np.asarray(vla_features, dtype=np.float32),
            planning_context=planning_context,
        )
        return self._apply_learned_helper(
            output,
            inference=inference,
            helper_status=helper_status,
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
        return dino_features, vla_features, float(
            max(0.35, semantic_features[10] if semantic_features.size > 10 else 0.5)
        ), float(max(0.35, semantic_features[12] if semantic_features.size > 12 else 0.5))

    def _resolve_selection_summary(
        self,
        *,
        selection_summary: Optional[Any] = None,
        perception_embeddings: Optional[Any] = None,
        orchestrator_context: Optional[OrchestratorContext] = None,
    ) -> Dict[str, Any]:
        if isinstance(selection_summary, Mapping):
            return dict(selection_summary)
        if isinstance(perception_embeddings, Mapping):
            candidate = perception_embeddings.get("selection_summary")
            if isinstance(candidate, Mapping):
                return dict(candidate)
        semantic_metadata = getattr(orchestrator_context, "semantic_metadata", None)
        if isinstance(semantic_metadata, Mapping):
            candidate = semantic_metadata.get("selection_summary")
            if isinstance(candidate, Mapping):
                return dict(candidate)
        return {}

    def _apply_planning_helper(
        self,
        *,
        objective_preset: str,
        energy_profile_weights: Mapping[str, Any],
        data_mix_weights: Mapping[str, Any],
        chosen_backend: str,
        expected_deltas: Mapping[str, Any],
        helper_metadata: Mapping[str, Any],
    ) -> tuple[str, Dict[str, float], Dict[str, float], str, Dict[str, float], Dict[str, Any]]:
        if not helper_metadata or not bool(helper_metadata.get("planning_heads_available", False)):
            return (
                objective_preset,
                normalize_named_weights(energy_profile_weights, META_ENERGY_PROFILE_LABELS),
                normalize_named_weights(data_mix_weights, META_DATA_MIX_LABELS),
                chosen_backend,
                {label: float(expected_deltas.get(label, 0.0)) for label in META_EXPECTED_DELTA_LABELS},
                {
                    "planning_available": False,
                    "objective_source": "heuristic_prior",
                    "backend_source": "heuristic_prior",
                    "energy_profile_source": "heuristic_prior",
                    "data_mix_source": "heuristic_prior",
                    "expected_delta_source": "heuristic_prior",
                },
            )

        helper_weight = float(helper_metadata.get("helper_weight", 0.0))
        benchmark_gate_ready = bool(helper_metadata.get("benchmark_gate_ready", False))
        objective_candidate = str(helper_metadata.get("objective_preset", objective_preset) or objective_preset)
        objective_confidence = float(helper_metadata.get("objective_confidence", 0.0))
        objective_alternate_confidence = float(
            helper_metadata.get("objective_alternate_confidence", 0.0)
        )
        objective_margin = objective_confidence - objective_alternate_confidence
        objective_override = benchmark_gate_ready or (
            objective_confidence >= 0.58 and objective_margin >= 0.12
        )
        final_objective = objective_candidate if objective_override else objective_preset
        objective_source = (
            "learned_helper"
            if objective_override
            else ("agreement" if objective_candidate == objective_preset else "heuristic_prior")
        )

        backend_candidate = str(helper_metadata.get("chosen_backend", chosen_backend) or chosen_backend)
        backend_confidence = float(helper_metadata.get("backend_confidence", 0.0))
        backend_alternate_confidence = float(
            helper_metadata.get("backend_alternate_confidence", 0.0)
        )
        backend_margin = backend_confidence - backend_alternate_confidence
        backend_override = benchmark_gate_ready or (
            backend_confidence >= 0.65 and backend_margin >= 0.18
        )
        final_backend = backend_candidate if backend_override else chosen_backend
        backend_source = (
            "learned_helper"
            if backend_override
            else ("agreement" if backend_candidate == chosen_backend else "heuristic_prior")
        )

        learned_energy_mix = dict(helper_metadata.get("energy_profile_weights", {}) or {})
        learned_data_mix = dict(helper_metadata.get("data_mix_weights", {}) or {})
        learned_expected_deltas = dict(helper_metadata.get("expected_deltas", {}) or {})
        final_energy_mix = self._blend_named_weights(
            energy_profile_weights,
            learned_energy_mix,
            weight=helper_weight,
            labels=META_ENERGY_PROFILE_LABELS,
        )
        final_data_mix = self._blend_named_weights(
            data_mix_weights,
            learned_data_mix,
            weight=helper_weight,
            labels=META_DATA_MIX_LABELS,
        )
        final_expected_deltas = self._blend_expected_deltas(
            expected_deltas,
            learned_expected_deltas,
            weight=helper_weight,
        )
        return (
            final_objective,
            final_energy_mix,
            final_data_mix,
            final_backend,
            final_expected_deltas,
            {
                "planning_available": True,
                "objective_prior": objective_preset,
                "objective_candidate": objective_candidate,
                "objective_source": objective_source,
                "objective_confidence": objective_confidence,
                "objective_margin": objective_margin,
                "backend_prior": chosen_backend,
                "backend_candidate": backend_candidate,
                "backend_source": backend_source,
                "backend_confidence": backend_confidence,
                "backend_margin": backend_margin,
                "energy_profile_source": "blended",
                "data_mix_source": "blended",
                "expected_delta_source": "blended",
                "energy_profile_prior": normalize_named_weights(
                    energy_profile_weights,
                    META_ENERGY_PROFILE_LABELS,
                ),
                "energy_profile_candidate": normalize_named_weights(
                    learned_energy_mix,
                    META_ENERGY_PROFILE_LABELS,
                ),
                "data_mix_prior": normalize_named_weights(
                    data_mix_weights,
                    META_DATA_MIX_LABELS,
                ),
                "data_mix_candidate": normalize_named_weights(
                    learned_data_mix,
                    META_DATA_MIX_LABELS,
                ),
                "expected_delta_prior": {
                    label: float(expected_deltas.get(label, 0.0))
                    for label in META_EXPECTED_DELTA_LABELS
                },
                "expected_delta_candidate": {
                    label: float(learned_expected_deltas.get(label, 0.0))
                    for label in META_EXPECTED_DELTA_LABELS
                },
            },
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
        selection_summary: Optional[Any] = None,
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
        selection_payload = self._resolve_selection_summary(
            selection_summary=selection_summary,
            perception_embeddings=perception_embeddings,
            orchestrator_context=orchestrator_context,
        )
        semantic_features = encode_semantic_world_model_features(semantic_summary)
        dino_features, vla_features, dino_conf, vla_conf = self._build_feature_streams(
            econ_signals=econ_payload,
            datapack_signals=datapack_payload,
            semantic_summary=semantic_summary,
            perception_embeddings=perception_embeddings,
        )
        planning_context = build_meta_planning_context_vector(
            semantic_summary=semantic_summary,
            econ_signals=econ_payload,
            datapack_signals=datapack_payload,
            selection_summary=selection_payload,
        )
        output = self.forward(
            dino_features=dino_features,
            vla_features=vla_features,
            dino_conf=dino_conf,
            vla_conf=vla_conf,
            semantic_features=semantic_features,
            planning_context=planning_context,
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
        (
            objective_preset,
            energy_profile_weights,
            data_mix_weights,
            chosen_backend,
            deltas,
            planning_application,
        ) = self._apply_planning_helper(
            objective_preset=objective_preset,
            energy_profile_weights=energy_profile_weights,
            data_mix_weights=data_mix_weights,
            chosen_backend=chosen_backend,
            expected_deltas=deltas,
            helper_metadata=dict(output.metadata.get("learned_helper", {}) or {}),
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
        output.ontology_tokens = list(
            dict.fromkeys(
                list(semantic_tokens(semantic_summary) or [])
                + list(output.ontology_tokens or [])
            )
        )[:16]
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
            **dict(output.metadata or {}),
            "semantic_world_model_summary": semantic_summary,
            "semantic_world_model_id": semantic_summary.get("world_model_id"),
            "active_capabilities": list(semantic_summary.get("active_capabilities", []) or []),
            "top_meta_nodes": list(semantic_summary.get("top_meta_nodes", []) or []),
            "coverage_feedback_summary": dict(semantic_summary.get("coverage_feedback_summary", {}) or {}),
            "wm_validation_summary": dict(semantic_summary.get("wm_validation_summary", {}) or {}),
            "selection_summary": dict(selection_payload),
            "planning_context": {
                "dim": int(planning_context.shape[0]),
                "l2_norm": float(np.linalg.norm(planning_context)),
                "selection_summary_available": bool(selection_payload),
            },
        }
        if "learned_helper" in output.metadata:
            helper_metadata = dict(output.metadata.get("learned_helper", {}) or {})
            helper_metadata["planning_application"] = planning_application
            output.metadata["learned_helper"] = helper_metadata
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
