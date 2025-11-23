"""
ObservationAdapter: unified observation construction + policy feature flattening.

Deterministic, JSON-safe, and flag-gated for Phase G observation unification.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.observation.condition_vector import ConditionVector
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.observation.models import Observation, VisionSlice, SemanticSlice, EconSlice, RecapSlice, ControlSlice
from src.utils.json_safe import to_json_safe
from src.vla.recap_features import summarize_vision_features

try:
    from src.vision.interfaces import VisionFrame, VisionLatent
except Exception:  # pragma: no cover - optional dependency for tests
    VisionFrame = Any  # type: ignore
    VisionLatent = Any  # type: ignore

try:
    from src.ontology.models import EconVector
except Exception:  # pragma: no cover
    EconVector = Any  # type: ignore

try:
    from src.semantic.models import SemanticSnapshot
except Exception:  # pragma: no cover
    SemanticSnapshot = Any  # type: ignore

try:
    from src.vla.recap_inference import RecapEpisodeScores
except Exception:  # pragma: no cover
    RecapEpisodeScores = Any  # type: ignore


def _hash_to_unit(val: str) -> float:
    """Convert an arbitrary string into a stable float in [0,1]."""
    if not val:
        return 0.0
    digest = hashlib.sha256(str(val).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pad_list(values: List[float], target_len: int) -> List[float]:
    if target_len <= 0:
        return []
    padded = list(values[:target_len])
    if len(padded) < target_len:
        padded.extend([0.0] * (target_len - len(padded)))
    return padded


def _one_hot(value: Optional[str], order: List[str]) -> List[float]:
    if not order:
        return []
    vec = [0.0] * len(order)
    if value is None:
        return vec
    value = str(value)
    if value in order:
        vec[order.index(value)] = 1.0
    return vec


class ObservationAdapter:
    """
    Builds canonical Observation objects and deterministic policy tensors.
    """

    DEFAULT_REWARD_COMPONENT_ORDER = ["mpl_component", "ep_component", "error_penalty", "wage_penalty"]
    DEFAULT_CONTROL_PHASE_ORDER = ["warmup", "skill_building", "frontier", "fine_tuning"]
    DEFAULT_SAMPLER_STRATEGY_ORDER = ["balanced", "frontier_prioritized", "econ_urgency"]
    DEFAULT_OBJECTIVE_PRESET_ORDER = ["throughput", "energy_saver", "balanced", "safety_first", "custom"]

    def __init__(
        self,
        policy_registry,
        trust_matrix_loader=None,
        recap_loader=None,
        config: Optional[Dict[str, Any]] = None,
        condition_builder: Optional[ConditionVectorBuilder] = None,
    ):
        self.policies = policy_registry
        self.trust_matrix = trust_matrix_loader() if callable(trust_matrix_loader) else {}
        self.recap_bundle = recap_loader() if callable(recap_loader) else None
        self.condition_builder = condition_builder

        cfg = config or {}
        self.semantic_tag_order: List[str] = list(cfg.get("semantic_tag_order") or sorted(self.trust_matrix.keys()))
        self.econ_component_order: List[str] = list(cfg.get("econ_component_order") or self.DEFAULT_REWARD_COMPONENT_ORDER)
        self.recap_metric_order: List[str] = list(cfg.get("recap_metric_order") or [])
        self.recap_adv_bins: int = int(cfg.get("recap_advantage_bins", 0))
        self.control_phase_order: List[str] = list(cfg.get("control_phase_order") or self.DEFAULT_CONTROL_PHASE_ORDER)
        self.sampler_strategy_order: List[str] = list(cfg.get("sampler_strategy_order") or self.DEFAULT_SAMPLER_STRATEGY_ORDER)
        self.objective_preset_order: List[str] = list(cfg.get("objective_preset_order") or self.DEFAULT_OBJECTIVE_PRESET_ORDER)
        self._vision_latent_dim: Optional[int] = cfg.get("vision_latent_dim")
        self._feature_dim: Optional[int] = None
        if self.condition_builder is None:
            self.condition_builder = ConditionVectorBuilder(cfg.get("condition_vector"))

    def build_observation(
        self,
        *,
        vision_frame: Optional[VisionFrame],
        vision_latent: Optional[VisionLatent],
        reward_scalar: float,
        reward_components: Dict[str, float],
        econ_vector: Optional[EconVector],
        semantic_snapshot: Optional[SemanticSnapshot],
        recap_scores: Optional[RecapEpisodeScores],
        descriptor: Optional[Dict[str, Any]],
        episode_metadata: Dict[str, Any],
        raw_env_obs: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        """
        Main constructor: tolerant to missing inputs and deterministic.
        """
        vision_slice = self._build_vision_slice(vision_frame, vision_latent)
        semantic_slice = self._build_semantic_slice(semantic_snapshot)
        econ_slice = self._build_econ_slice(reward_scalar, reward_components, econ_vector)
        recap_slice = self._build_recap_slice(recap_scores)
        control_slice = self._build_control_slice(descriptor, episode_metadata)

        return Observation(
            vision=vision_slice,
            semantics=semantic_slice,
            econ=econ_slice,
            recap=recap_slice,
            control=control_slice,
            raw_env_obs=raw_env_obs or {},
        )

    def build_condition_vector(
        self,
        *,
        episode_config: Any,
        econ_state: Any,
        curriculum_phase: str,
        sima2_trust: Optional[Any],
        datapack_metadata: Optional[Dict[str, Any]] = None,
        episode_step: int = 0,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConditionVector]:
        """
        Public helper: fuse task/econ/semantic metadata into a ConditionVector.
        """
        if self.condition_builder is None:
            return None
        return self.condition_builder.build(
            episode_config=episode_config,
            econ_state=econ_state,
            curriculum_phase=curriculum_phase,
            sima2_trust=sima2_trust,
            datapack_metadata=datapack_metadata,
            episode_step=episode_step,
            overrides=overrides,
        )

    def build_observation_and_condition(
        self,
        *,
        vision_frame: Optional[VisionFrame],
        vision_latent: Optional[VisionLatent],
        reward_scalar: float,
        reward_components: Dict[str, float],
        econ_vector: Optional[EconVector],
        semantic_snapshot: Optional[SemanticSnapshot],
        recap_scores: Optional[RecapEpisodeScores],
        descriptor: Optional[Dict[str, Any]],
        episode_metadata: Dict[str, Any],
        raw_env_obs: Optional[Dict[str, Any]] = None,
        condition_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Observation, Optional[ConditionVector]]:
        """
        Build Observation and (optionally) ConditionVector in one call.
        """
        observation = self.build_observation(
            vision_frame=vision_frame,
            vision_latent=vision_latent,
            reward_scalar=reward_scalar,
            reward_components=reward_components,
            econ_vector=econ_vector,
            semantic_snapshot=semantic_snapshot,
            recap_scores=recap_scores,
            descriptor=descriptor,
            episode_metadata=episode_metadata,
            raw_env_obs=raw_env_obs,
        )
        condition_kwargs = condition_kwargs or {}
        condition = None
        if condition_kwargs:
            condition = self.build_condition_vector(**condition_kwargs)
        return observation, condition

    def to_policy_tensor(
        self,
        obs: Observation,
        condition: Optional[ConditionVector] = None,
        include_condition: bool = False,
    ) -> np.ndarray:
        """
        Canonical flattening for RL policies and future NN policies.

        Ordering (stable once initialized):
        - Vision summary: backend hash, state_digest hash, latent mean/min/max
        - Raw env telemetry: t, completed, attempts, errors
        - Econ base scalars: reward_scalar, mpl, wage_parity, energy_wh, damage_cost
        - Econ reward components (sorted/ordered)
        - Semantic tags (ordered), trust scores (aligned), ood/recovery
        - Recap: advantage bin probs (padded), recap_goodness_score, metric expectations
        - Control: one-hot curriculum_phase, sampler_strategy, objective_preset
        - Condition vector (optional, gated via include_condition)
        """
        self._maybe_initialize_orders(obs)

        features: List[float] = []
        features.extend(self._vision_features(obs))
        features.extend(self._raw_env_features(obs))
        features.extend(self._econ_features(obs))
        features.extend(self._semantic_features(obs))
        features.extend(self._recap_features(obs))
        features.extend(self._control_features(obs))
        if include_condition and condition is not None:
            features.extend(self._condition_features(condition))

        tensor = np.array(features, dtype=np.float32)
        self._feature_dim = tensor.shape[-1]
        return tensor

    # --- slice builders -------------------------------------------------
    def _build_vision_slice(
        self,
        vision_frame: Optional[VisionFrame],
        vision_latent: Optional[VisionLatent],
    ) -> Optional[VisionSlice]:
        if vision_frame is None and vision_latent is None:
            return None
        frame_md = getattr(vision_frame, "metadata", {}) if vision_frame else {}
        backend_id = getattr(vision_frame, "backend_id", None) or getattr(vision_frame, "backend", None) or ""
        state_digest = getattr(vision_frame, "state_digest", None) or frame_md.get("state_digest") or ""
        intrinsics = getattr(vision_frame, "camera_intrinsics", {}) if vision_frame else {}
        extrinsics = getattr(vision_frame, "camera_extrinsics", {}) if vision_frame else {}
        latent = None
        if vision_latent is not None:
            latent = list(getattr(vision_latent, "latent", []) or [])
        elif frame_md.get("latent") is not None:
            try:
                latent = list(frame_md.get("latent") or [])
            except Exception:
                latent = None
        metadata = {
            "backend": getattr(vision_frame, "backend", None),
            "camera_name": getattr(vision_frame, "camera_name", None),
            "metadata": frame_md,
        }
        return VisionSlice(
            backend_id=str(backend_id),
            state_digest=str(state_digest),
            intrinsics={k: _safe_float(v) for k, v in (intrinsics or {}).items()},
            extrinsics={k: _safe_float(v) for k, v in (extrinsics or {}).items()},
            latent=latent,
            metadata=metadata,
        )

    def _build_semantic_slice(self, snapshot: Optional[SemanticSnapshot]) -> Optional[SemanticSlice]:
        if snapshot is None:
            return None
        tags = self._extract_semantic_tags(snapshot)
        trust_scores = {k: float(self.trust_matrix.get(k, {}).get("trust_score", 0.0)) for k in tags.keys()}
        md = getattr(snapshot, "metadata", {}) if snapshot else {}
        return SemanticSlice(
            tags=tags,
            ood_score=_safe_float(md.get("ood_score", md.get("recap", {}).get("mean_goodness"))),
            recovery_score=_safe_float(md.get("recovery_score")),
            trust_scores=trust_scores,
            metadata=md,
        )

    def _build_econ_slice(
        self,
        reward_scalar: float,
        reward_components: Dict[str, float],
        econ_vector: Optional[EconVector],
    ) -> EconSlice:
        # EconVector provides calibrated metrics if available
        mpl = _safe_float(getattr(econ_vector, "mpl_units_per_hour", None), _safe_float(reward_components.get("mpl_component")))
        wage_parity = _safe_float(getattr(econ_vector, "wage_parity", None))
        energy_wh = _safe_float(getattr(econ_vector, "energy_cost", None))
        damage_cost = _safe_float(getattr(econ_vector, "damage_cost", None))
        reward_scalar_val = _safe_float(reward_scalar)
        components = {k: _safe_float(v) for k, v in (reward_components or {}).items()}
        domain = getattr(econ_vector, "source_domain", None) if econ_vector else None
        metadata = getattr(econ_vector, "metadata", {}) if econ_vector else {}
        return EconSlice(
            mpl=mpl,
            wage_parity=wage_parity,
            energy_wh=energy_wh,
            damage_cost=damage_cost,
            reward_scalar=reward_scalar_val,
            components=components,
            domain_name=domain,
            metadata=metadata,
        )

    def _build_recap_slice(self, recap_scores: Optional[RecapEpisodeScores]) -> Optional[RecapSlice]:
        if recap_scores is None:
            return None
        metric_expectations: Dict[str, float] = {}
        metric_dists = getattr(recap_scores, "metric_distributions", {}) or {}
        value_supports = getattr(recap_scores, "metadata", {}).get("value_supports", {}) if hasattr(recap_scores, "metadata") else {}
        for metric, dist in metric_dists.items():
            try:
                support = value_supports.get(metric, (0.0, 1.0))
                lo, hi = float(support[0]), float(support[1])
                vals = list(dist)
                if not vals:
                    metric_expectations[metric] = 0.0
                    continue
                lin = np.linspace(lo, hi, num=len(vals))
                metric_expectations[metric] = float(np.dot(lin, np.array(vals) / max(1e-12, np.sum(vals))))
            except Exception:
                metric_expectations[metric] = 0.0
        return RecapSlice(
            advantage_bin_probs=list(getattr(recap_scores, "advantage_bin_probs_mean", []) or []),
            metric_expectations=metric_expectations,
            recap_goodness_score=_safe_float(getattr(recap_scores, "recap_goodness_score", None)),
            metadata=getattr(recap_scores, "metadata", {}) if hasattr(recap_scores, "metadata") else {},
        )

    def _build_control_slice(self, descriptor: Optional[Dict[str, Any]], episode_metadata: Dict[str, Any]) -> Optional[ControlSlice]:
        if descriptor is None and not episode_metadata:
            return None
        desc = descriptor or {}
        sampling_md = desc.get("sampling_metadata", {}) if isinstance(desc, dict) else {}
        md = dict(episode_metadata or {})
        md.update({k: v for k, v in sampling_md.items() if k not in md})
        return ControlSlice(
            curriculum_phase=sampling_md.get("phase") or episode_metadata.get("curriculum_phase"),
            sampler_strategy=sampling_md.get("strategy") or episode_metadata.get("sampler_strategy"),
            objective_preset=desc.get("objective_preset") or episode_metadata.get("objective_preset"),
            task_id=desc.get("task_id") or episode_metadata.get("task_id"),
            episode_id=desc.get("episode_id") or episode_metadata.get("episode_id"),
            pack_id=desc.get("pack_id") or episode_metadata.get("pack_id"),
            metadata=md,
        )

    # --- feature helpers ------------------------------------------------
    def _maybe_initialize_orders(self, obs: Observation) -> None:
        if self._vision_latent_dim is None and obs.vision and obs.vision.latent:
            self._vision_latent_dim = len(obs.vision.latent)
        if not self.semantic_tag_order and obs.semantics:
            self.semantic_tag_order = sorted(obs.semantics.tags.keys())
        if not self.econ_component_order and obs.econ:
            self.econ_component_order = sorted(obs.econ.components.keys())
        if not self.recap_metric_order and obs.recap:
            self.recap_metric_order = sorted(obs.recap.metric_expectations.keys())
        if self.recap_adv_bins <= 0 and obs.recap and obs.recap.advantage_bin_probs:
            self.recap_adv_bins = len(obs.recap.advantage_bin_probs)

    def _vision_features(self, obs: Observation) -> List[float]:
        vs = obs.vision
        if vs is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        pf = {
            "vision_latent": {"latent": vs.latent or []},
            "backend": vs.backend_id,
            "state_digest": vs.state_digest,
        }
        summary = summarize_vision_features(pf)
        latent_mean = _safe_float(summary.get("vision_latent_mean"))
        latent_min = _safe_float(summary.get("vision_latent_min"))
        latent_max = _safe_float(summary.get("vision_latent_max"))
        return [
            _hash_to_unit(vs.backend_id),
            _hash_to_unit(vs.state_digest),
            latent_mean,
            latent_min,
            latent_max,
        ]

    def _raw_env_features(self, obs: Observation) -> List[float]:
        raw = obs.raw_env_obs or {}
        t = _safe_float(raw.get("t"))
        completed = _safe_float(raw.get("completed"))
        attempts = _safe_float(raw.get("attempts"))
        errors = _safe_float(raw.get("errors"))
        return [t, completed, attempts, errors]

    def _econ_features(self, obs: Observation) -> List[float]:
        econ = obs.econ
        base = [0.0, 0.0, 0.0, 0.0, 0.0]
        if econ:
            base = [
                _safe_float(econ.reward_scalar),
                _safe_float(econ.mpl),
                _safe_float(econ.wage_parity),
                _safe_float(econ.energy_wh),
                _safe_float(econ.damage_cost),
            ]
        comps: List[float] = []
        for key in self.econ_component_order:
            comps.append(_safe_float(econ.components.get(key) if econ else None))
        return base + comps

    def _semantic_features(self, obs: Observation) -> List[float]:
        sem = obs.semantics
        if sem is None:
            return [0.0] * (2 * len(self.semantic_tag_order) + 2)
        tag_vals = [ _safe_float(sem.tags.get(tag)) for tag in self.semantic_tag_order ]
        trust_vals = [_safe_float(sem.trust_scores.get(tag)) for tag in self.semantic_tag_order]
        extras = [
            _safe_float(sem.ood_score),
            _safe_float(sem.recovery_score),
        ]
        return tag_vals + trust_vals + extras

    def _recap_features(self, obs: Observation) -> List[float]:
        recap = obs.recap
        if recap is None:
            num_bins = max(self.recap_adv_bins, 0)
            return [0.0] * (num_bins + len(self.recap_metric_order) + 1)
        bins = recap.advantage_bin_probs or []
        num_bins = self.recap_adv_bins or len(bins)
        bin_features = _pad_list([_safe_float(v) for v in bins], num_bins)
        metric_feats = []
        for metric in self.recap_metric_order:
            metric_feats.append(_safe_float(recap.metric_expectations.get(metric)))
        return bin_features + metric_feats + [_safe_float(recap.recap_goodness_score)]

    def _control_features(self, obs: Observation) -> List[float]:
        ctrl = obs.control
        phase_vec = _one_hot(getattr(ctrl, "curriculum_phase", None) if ctrl else None, self.control_phase_order)
        sampler_vec = _one_hot(getattr(ctrl, "sampler_strategy", None) if ctrl else None, self.sampler_strategy_order)
        preset_vec = _one_hot(getattr(ctrl, "objective_preset", None) if ctrl else None, self.objective_preset_order)
        return phase_vec + sampler_vec + preset_vec

    def _condition_features(self, condition: ConditionVector) -> List[float]:
        """Stable numeric vector for ConditionVector appended to policy tensor."""
        try:
            vec = condition.to_vector()
            return [float(x) for x in vec.tolist()]
        except Exception:
            return []

    # --- semantic helpers ----------------------------------------------
    def _extract_semantic_tags(self, snapshot: SemanticSnapshot) -> Dict[str, float]:
        """
        Deterministically flatten semantic tags into a score map.
        """
        tags: Dict[str, float] = {}
        raw_tags = getattr(snapshot, "semantic_tags", []) or []
        for tag in raw_tags:
            tag_dict = to_json_safe(tag)
            if isinstance(tag_dict, dict):
                for key, val in sorted(tag_dict.items(), key=lambda kv: str(kv[0])):
                    if isinstance(val, (int, float)):
                        tags[str(key)] = tags.get(str(key), 0.0) + float(val)
                    elif isinstance(val, str):
                        compound = f"{key}:{val}"
                        tags[compound] = tags.get(compound, 0.0) + 1.0
            else:
                labels = str(tag_dict)
                tags[labels] = tags.get(labels, 0.0) + 1.0
        return dict(sorted(tags.items(), key=lambda kv: kv[0]))
