"""
Meta-Transformer scaffold that arbitrates between semantic (DINO) and affordance (OpenVLA) features.
No training or heavy logic; provides placeholder methods and typed dataclasses.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from src.config.objective_profile import ObjectiveVector
from src.orchestrator.context import OrchestratorContext
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


# Backward-compatible alias
MetaTransformerOutput = MetaTransformerOutputs


class MetaTransformer:
    def __init__(self, d_shared: int = 32):
        self.d_shared = d_shared

    def integrate_embeddings(self, dino_features: np.ndarray, vla_features: np.ndarray) -> np.ndarray:
        # Simple concat + trim as placeholder
        combined = np.concatenate([dino_features, vla_features])
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
    ) -> MetaTransformerOutputs:
        shared = self.integrate_embeddings(dino_features, vla_features)
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
