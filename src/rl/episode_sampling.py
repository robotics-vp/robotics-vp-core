"""
Episode sampling scaffolding (advisory-only).

Stage 3 extends the simple Stage 1 → RL descriptor conversion with an
advisory sampler that can balance tiers, prioritize frontier datapacks,
and weight by economic urgency without modifying reward math or training
algorithms.
"""
import copy
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable, Tuple, TYPE_CHECKING
import hashlib

from src.valuation.datapack_schema import DataPackMeta
from src.rl.episode_descriptor_validator import (
    normalize_episode_descriptor,
    validate_episode_descriptor,
    normalize_and_validate,
)
from src.utils.json_safe import to_json_safe
from src.rl.skill_mode_resolver import SkillModeResolver
from src.orchestrator.semantic_orchestrator_v2 import OrchestratorAdvisory

if TYPE_CHECKING:
    from src.policies.registry import PolicyBundle


def _load_recap_scores(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    scores: Dict[str, float] = {}
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ep = rec.get("episode_id") or rec.get("pack_id")
                if ep is not None:
                    scores[str(ep)] = float(rec.get("recap_goodness_score", 0.0))
            except Exception:
                continue
    return scores


def _episode_key(episode: Dict[str, Any]) -> str:
    desc = episode.get("descriptor", {})
    return str(desc.get("pack_id") or desc.get("episode_id") or desc.get("id") or "")


def _embodiment_metric(episode: Dict[str, Any], key: str, default: float) -> float:
    if key in episode:
        try:
            return float(episode.get(key, default))
        except Exception:
            return float(default)
    desc = episode.get("descriptor", {}) if isinstance(episode.get("descriptor"), dict) else {}
    if key in desc:
        try:
            return float(desc.get(key, default))
        except Exception:
            return float(default)
    return float(default)


def _summarize_condition_metadata(skill_mode: str, tags: Dict[str, float], phase: str) -> Dict[str, Any]:
    """Compact, JSON-safe summary of condition inputs for logging."""
    tag_items = [f"{str(k)}:{float(v):.4f}" for k, v in sorted(tags.items(), key=lambda kv: str(kv[0]))]
    payload = f"{skill_mode}|{phase}|" + "|".join(tag_items)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return {
        "skill_mode": skill_mode,
        "curriculum_phase": phase,
        "tag_fingerprint": digest,
        "tag_count": len(tags),
    }


def summarize_condition_metadata(skill_mode: str, tags: Dict[str, float], phase: str) -> Dict[str, Any]:
    """Public wrapper for condition metadata summaries."""
    return _summarize_condition_metadata(skill_mode, tags, phase)


def datapack_to_rl_episode_descriptor(datapack: DataPackMeta) -> Dict[str, Any]:
    """
    Convert a datapack into a lightweight descriptor suitable for an RL sampler.

    Reads:
    - objective vector (what to optimize for)
    - env_name, task_type (which environment to use)
    - engine/backend (PyBullet, Isaac, etc.)
    - guidance tags (semantic focus areas)
    - tier, trust_score (for prioritization)

    Returns:
    - Minimal episode descriptor that can be passed to RL training loop
    """
    # Extract objective vector
    objective_vector = []
    if datapack.objective_profile:
        objective_vector = datapack.objective_profile.objective_vector
    elif datapack.condition:
        # Fallback to condition profile if objective_profile is missing
        objective_vector = datapack.condition.objective_vector + [0.0, 0.0]  # Pad to 5 dimensions

    # Extract environment info
    env_name = datapack.task_name
    engine_type = datapack.env_type
    backend = "pybullet"  # Default backend

    if datapack.condition:
        env_name = datapack.condition.task_name or env_name
        backend = datapack.condition.engine_type or backend

    # Extract guidance tags
    semantic_tags = []
    focus_areas = []
    priority = "medium"

    if datapack.guidance_profile:
        semantic_tags = datapack.guidance_profile.semantic_tags or []
        focus_areas = [datapack.guidance_profile.main_driver]
        priority = "high" if datapack.guidance_profile.is_good else "medium"

    # Extract tier and trust for sampling weight
    tier = 1
    trust_score = 0.5
    delta_J = 0.0
    w_embodiment = None
    embodiment_drift_score = None
    embodiment_impossible_contacts = None
    embodiment_trust_override = None

    if datapack.attribution:
        tier = datapack.attribution.tier
        trust_score = datapack.attribution.trust_score
        delta_J = datapack.attribution.delta_J
    if datapack.embodiment_profile is not None:
        emb = datapack.embodiment_profile
        w_embodiment = emb.w_embodiment
        embodiment_drift_score = emb.drift_score
        embodiment_impossible_contacts = emb.physically_impossible_contacts
        embodiment_trust_override = emb.trust_override_candidate
    elif datapack.episode_metrics:
        w_embodiment = datapack.episode_metrics.get("w_embodiment")
        embodiment_drift_score = datapack.episode_metrics.get("embodiment_drift_score")
        embodiment_impossible_contacts = datapack.episode_metrics.get("embodiment_physically_impossible_contacts")
        embodiment_trust_override = datapack.episode_metrics.get("embodiment_trust_override_candidate")

    # Compute sampling weight (higher for higher-tier, higher-trust datapacks)
    sampling_weight = trust_score * (1.0 + 0.5 * tier)  # Tier 2 gets 1.5x boost

    # Episode length heuristic (can be overridden by env defaults)
    episode_length = 1000  # Default

    descriptor = {
        # Identification
        "pack_id": datapack.pack_id,
        "datapack_type": "stage1" if "stage1" in datapack.pack_id else "runtime",

        # Environment configuration
        "env_name": env_name,
        "task_type": env_name,
        "backend": backend,
        "engine_type": engine_type,

        # Objective and reward
        "objective_vector": objective_vector,
        "objective_preset": _infer_objective_preset(objective_vector),

        # Guidance
        "semantic_tags": semantic_tags,
        "focus_areas": focus_areas,
        "priority": priority,

        # Quality/sampling signals
        "tier": tier,
        "trust_score": trust_score,
        "delta_J": delta_J,
        "sampling_weight": sampling_weight,
        "w_embodiment": w_embodiment if w_embodiment is not None else 1.0,
        "embodiment_drift_score": embodiment_drift_score if embodiment_drift_score is not None else 0.0,
        "embodiment_physically_impossible_contacts": embodiment_impossible_contacts or 0,
        "embodiment_trust_override_candidate": bool(embodiment_trust_override)
        if embodiment_trust_override is not None
        else False,

        # Episode parameters
        "episode_length": episode_length,

        # Logging/tracking
        "tags": {
            "is_good": datapack.guidance_profile.is_good if datapack.guidance_profile else False,
            "main_driver": focus_areas[0] if focus_areas else "unknown",
            "source": "stage1_diffusion_vla" if "stage1" in datapack.pack_id else "runtime",
        }
    }
    unified_weights = None
    pr_profile = datapack.process_reward_profile
    if pr_profile is not None and getattr(pr_profile, "has_data", lambda: True)():
        descriptor["process_reward_profile"] = pr_profile.to_dict()
        try:
            from src.policies.unified_quality import UnifiedQualityPolicy

            unified_weights = UnifiedQualityPolicy().compute_from_datapack(datapack)
        except Exception:
            unified_weights = None
    if unified_weights is not None:
        descriptor["unified_quality_weight"] = max(0.0, float(unified_weights.w_combined))
        descriptor["unified_quality_eligible"] = bool(unified_weights.is_eligible)
        descriptor["unified_quality_reason"] = unified_weights.eligibility_reason
        descriptor["unified_quality"] = unified_weights.to_dict()
    descriptor = normalize_episode_descriptor(descriptor)
    errors = validate_episode_descriptor(descriptor)
    if errors:
        raise ValueError(f"Episode descriptor validation failed: {errors}")
    return descriptor


def _infer_objective_preset(objective_vector: List[float]) -> str:
    """
    Infer objective preset from objective vector.

    Standard presets:
    - throughput: [2.0, 1.0, 0.5, 1.0, 0.0]
    - safety: [1.0, 1.0, 0.5, 3.0, 0.0]
    - energy_saver: [1.0, 1.0, 2.0, 1.0, 0.0]
    - balanced: [1.0, 1.0, 1.0, 1.0, 0.0]
    """
    if len(objective_vector) < 4:
        return "balanced"

    # Check for throughput (high MPL weight)
    if objective_vector[0] > 1.5:
        return "throughput"

    # Check for safety (high safety weight)
    if len(objective_vector) >= 4 and objective_vector[3] > 2.0:
        return "safety"

    # Check for energy_saver (high energy weight)
    if len(objective_vector) >= 3 and objective_vector[2] > 1.5:
        return "energy_saver"

    return "balanced"


def sampler_stub(datapacks: List[DataPackMeta]) -> Dict[str, Any]:
    """
    Placeholder sampler that returns descriptors only.
    Real sampling logic is intentionally omitted.
    """
    return {dp.pack_id: datapack_to_rl_episode_descriptor(dp) for dp in datapacks}


class DataPackRLSampler:
    """
    Advisory RL sampler over Stage 1 datapack descriptors + Stage 2 enrichments.

    Strategies (flagged at call time):
    - balanced: tier-aware, lightly trust-weighted stratified sampling
    - frontier_prioritized: weight by ΔMPL/ΔJ + novelty/expected MPL gains
    - econ_urgency: weight by economic urgency and supervision hints
    - process_reward_conf: weight by process reward confidence (if present)
    - process_reward_progress: weight by process reward progress (delta)
    - process_reward_quality: weight by combined process reward signals
    - embodiment_quality: weight by embodiment confidence (if present)
    - embodiment_drift_penalty: penalize high embodiment drift (if present)
    - embodiment_quality_drift: combine embodiment weight with drift penalty

    Deterministic for a given seed and configuration; returns JSON-safe
    descriptors without mutating objectives or reward math. Embodiment-based
    strategies only affect sampling weights.
    """

    def __init__(
        self,
        datapacks: Optional[List[DataPackMeta]] = None,
        enrichments: Optional[Iterable[Dict[str, Any]]] = None,
        existing_descriptors: Optional[Iterable[Dict[str, Any]]] = None,
        default_strategy: str = "balanced",
        tier_ratios: Optional[Dict[int, float]] = None,
        advisory: Optional[OrchestratorAdvisory] = None,
        use_recap_weights: bool = False,
        recap_scores: Optional[Dict[str, float]] = None,
        recap_scores_path: Optional[str] = None,
        policies: Optional["PolicyBundle"] = None,
        use_datapack_auditor: bool = False,
        trust_matrix: Optional[Dict[str, Any]] = None,
        use_condition_vector: bool = False,
        use_unified_quality: bool = True,
    ) -> None:
        self.default_strategy = default_strategy
        self.tier_ratios = tier_ratios or {0: 0.2, 1: 0.5, 2: 0.3}
        self.enrichment_map = self._build_enrichment_map(enrichments)
        self._episodes: List[Dict[str, Any]] = []
        self.advisory = advisory
        self.use_recap_weights = bool(use_recap_weights)
        self.use_datapack_auditor = bool(use_datapack_auditor)
        self.trust_matrix = trust_matrix or {}
        self.use_condition_vector = bool(use_condition_vector)
        self.use_unified_quality = bool(use_unified_quality)
        self.skill_resolver = SkillModeResolver(
            default_mode="efficiency_throughput",
            mode_order=[
                "frontier_exploration",
                "safety_critical",
                "efficiency_throughput",
                "recovery_heavy",
                "default",
            ],
        )
        from src.policies.registry import build_all_policies

        self.policies = policies or build_all_policies()
        provided_scores = recap_scores or {}
        path_scores = _load_recap_scores(recap_scores_path)
        self.recap_scores = {**path_scores, **provided_scores}

        if datapacks:
            for dp in datapacks:
                desc = datapack_to_rl_episode_descriptor(dp)
                self._add_episode(desc, source="stage1_datapack")

        if existing_descriptors:
            for desc in existing_descriptors:
                self._add_episode(desc, source="existing_descriptor")

        self._episodes.sort(key=lambda e: e["descriptor"].get("pack_id", ""))

        if not self._episodes:
            raise ValueError("DataPackRLSampler requires at least one episode descriptor")

    def sample_batch(
        self,
        batch_size: int,
        seed: int = 0,
        strategy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample a batch of episode descriptors using the requested strategy.

        Deterministic for a given seed, strategy, and input pool. Returns
        JSON-safe descriptors that include enrichment metadata for inspection.
        """
        if batch_size <= 0:
            return []

        strategy_name = self._select_strategy(strategy)
        rng = random.Random(seed)

        if strategy_name in {
            "balanced",
            "process_reward_conf",
            "process_reward_progress",
            "process_reward_quality",
            "embodiment_quality",
            "embodiment_drift_penalty",
            "embodiment_quality_drift",
        }:
            weight_strategy = "balanced" if strategy_name == "balanced" else strategy_name
            selected = self._sample_balanced(batch_size, rng, weight_strategy=weight_strategy)
        elif strategy_name == "frontier_prioritized":
            selected = self._sample_frontier_prioritized(batch_size, rng)
        elif strategy_name == "econ_urgency":
            selected = self._sample_econ_urgency(batch_size, rng)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy_name}")

        rng.shuffle(selected)  # Deterministic shuffle for batch decorrelation
        return [self._format_output(ep, strategy_name) for ep in selected]

    def pool_summary(self) -> Dict[str, Any]:
        """Lightweight summary of the sampler pool for debugging/preview."""
        tiers: Dict[int, int] = {}
        presets: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        for ep in self._episodes:
            desc = ep["descriptor"]
            tier = desc.get("tier", 0)
            preset = desc.get("objective_preset", "balanced")
            tiers[tier] = tiers.get(tier, 0) + 1
            presets[preset] = presets.get(preset, 0) + 1
            sources[ep["source"]] = sources.get(ep["source"], 0) + 1
        return {
            "num_episodes": len(self._episodes),
            "tiers": tiers,
            "objective_presets": presets,
            "sources": sources,
        }

    def _select_strategy(self, strategy: Optional[str]) -> str:
        if strategy:
            return strategy.lower()
        if self.advisory and self.advisory.sampler_strategy_overrides:
            weights = self.advisory.sampler_strategy_overrides
            # Deterministic choice: highest weight then lexicographic
            ordered = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
            return ordered[0][0]
        return self.default_strategy.lower()

    def _add_episode(self, descriptor: Dict[str, Any], source: str) -> None:
        normalized, errors = normalize_and_validate(descriptor)
        if errors:
            raise ValueError(f"Episode descriptor validation failed: {errors}")
        pack_id = normalized.get("pack_id") or normalized.get("episode_id") or f"desc_{len(self._episodes)}"
        enrichment = self.enrichment_map.get(pack_id, _normalize_enrichment(descriptor.get("enrichment")))

        episode = {
            "descriptor": normalized,
            "enrichment": enrichment,
            "source": source,
            "advisory": self.advisory,
            "recap_goodness_score": self.recap_scores.get(pack_id),
        }
        episode["novelty_score"] = _extract_max_novelty(enrichment)
        episode["expected_mpl_gain"] = _extract_expected_mpl_gain(enrichment)
        episode["frontier_score"] = self._compute_frontier_score(episode)
        episode["econ_urgency_score"] = self._compute_econ_urgency_score(episode)
        episode["recap_weight_multiplier"] = self._recap_weight_multiplier(episode)
        episode["unified_quality_weight"] = max(0.0, float(descriptor.get("unified_quality_weight", 1.0)))
        episode["unified_quality_eligible"] = bool(descriptor.get("unified_quality_eligible", True))
        episode["unified_quality_reason"] = descriptor.get("unified_quality_reason")
        
        # Auditor Integration
        episode["auditor_result"] = None
        episode["auditor_weight_multiplier"] = 1.0
        if self.use_datapack_auditor and getattr(self.policies, "datapack_auditor", None):
            try:
                # Build features
                auditor_features = self.policies.datapack_auditor.build_features(
                    datapack=descriptor, # passing dict as datapack proxy
                    semantic_tags=enrichment.get("novelty_tags", []) + enrichment.get("fragility_tags", []) + enrichment.get("risk_tags", []), # Flatten tags roughly
                    econ_slice={"expected_mpl_gain": episode["expected_mpl_gain"], "novelty_score": episode["novelty_score"]},
                    recap_scores={"quality_score": episode.get("recap_goodness_score", 0.5)}
                )
                # Evaluate
                audit = self.policies.datapack_auditor.evaluate(auditor_features)
                episode["auditor_result"] = audit
                
                # Compute mild weight multiplier based on rating
                rating = audit.get("rating", "BBB")
                # AAA -> 1.2, AA -> 1.1, A -> 1.0, BBB -> 1.0, JUNK -> 0.8
                rating_mult = {"AAA": 1.2, "AA": 1.1, "A": 1.0, "BBB": 1.0, "JUNK": 0.8}
                episode["auditor_weight_multiplier"] = rating_mult.get(rating, 1.0)
            except Exception:
                # Fail gracefully to identity
                episode["auditor_weight_multiplier"] = 1.0

        self._episodes.append(episode)

    def _build_enrichment_map(
        self, enrichments: Optional[Iterable[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        if not enrichments:
            return {}
        mapping: Dict[str, Dict[str, Any]] = {}
        for record in enrichments:
            if not isinstance(record, dict):
                continue
            key = record.get("episode_id") or record.get("pack_id") or record.get("id")
            enrichment_payload = record.get("enrichment", record)
            if key:
                mapping[str(key)] = _normalize_enrichment(enrichment_payload)
        return mapping

    def _compute_weights(self, episodes: List[Dict[str, Any]], strategy: str) -> List[float]:
        if not episodes:
            return []
        policy = getattr(self, "policies", None)
        if policy and getattr(policy, "sampler_weights", None):
            features = policy.sampler_weights.build_features(episodes)
            weight_map = policy.sampler_weights.evaluate(features, strategy=strategy)
            weights = [float(weight_map.get(_episode_key(ep), 0.0)) for ep in episodes]
        elif strategy == "frontier_prioritized":
            weights = [max(ep["frontier_score"], 1e-3) * ep.get("recap_weight_multiplier", 1.0) for ep in episodes]
        elif strategy == "econ_urgency":
            weights = [max(ep["econ_urgency_score"], 1e-3) * ep.get("recap_weight_multiplier", 1.0) for ep in episodes]
        elif strategy == "embodiment_quality":
            weights = [
                max(_embodiment_metric(ep, "w_embodiment", 1.0), 0.1) * ep.get("recap_weight_multiplier", 1.0)
                for ep in episodes
            ]
        elif strategy == "embodiment_drift_penalty":
            weights = [
                max(1.0 - _embodiment_metric(ep, "embodiment_drift_score", 0.0), 0.1)
                * ep.get("recap_weight_multiplier", 1.0)
                for ep in episodes
            ]
        elif strategy == "embodiment_quality_drift":
            weights = [
                max(
                    _embodiment_metric(ep, "w_embodiment", 1.0)
                    * (1.0 - _embodiment_metric(ep, "embodiment_drift_score", 0.0)),
                    0.1,
                )
                * ep.get("recap_weight_multiplier", 1.0)
                for ep in episodes
            ]
        else:
            weights = [_balanced_weight(ep) for ep in episodes]
        if not self.use_unified_quality:
            clamped = [max(0.0, float(w)) for w in weights]
        else:
            clamped = [
                max(0.0, float(w * max(0.0, ep.get("unified_quality_weight", 1.0))))
                for w, ep in zip(weights, episodes)
            ]
        if clamped and max(clamped) <= 0.0:
            return [1.0] * len(clamped)
        return clamped

    def _eligible_pool(self) -> List[Dict[str, Any]]:
        if not self.use_unified_quality:
            return list(self._episodes)
        eligible = [ep for ep in self._episodes if ep.get("unified_quality_eligible", True)]
        return eligible if eligible else list(self._episodes)

    def _sample_balanced(self, batch_size: int, rng: random.Random, weight_strategy: str = "balanced") -> List[Dict[str, Any]]:
        """Tier-aware sampling that lightly respects trust_score without over-concentrating."""
        episodes = self._eligible_pool()
        tier_groups: Dict[int, List[Dict[str, Any]]] = {}
        for ep in episodes:
            tier = ep["descriptor"].get("tier", 0)
            tier_groups.setdefault(tier, []).append(ep)

        counts: Dict[int, int] = {}
        total_assigned = 0
        for tier, ratio in self.tier_ratios.items():
            count = int(round(batch_size * ratio))
            counts[tier] = min(count, len(tier_groups.get(tier, [])))
            total_assigned += counts[tier]

        # Allocate remaining slots deterministically to tiers with headroom
        remaining = batch_size - total_assigned
        tier_order = sorted(self.tier_ratios.keys(), key=lambda t: (-self.tier_ratios[t], t))
        while remaining > 0:
            allocated = False
            for tier in tier_order:
                if len(tier_groups.get(tier, [])) > counts.get(tier, 0):
                    counts[tier] = counts.get(tier, 0) + 1
                    remaining -= 1
                    allocated = True
                    if remaining == 0:
                        break
            if not allocated:
                break

        selected: List[Dict[str, Any]] = []
        for tier in sorted(tier_groups.keys()):
            pool = tier_groups[tier]
            need = counts.get(tier, 0)
            weights = self._compute_weights(pool, strategy=weight_strategy)
            selected.extend(_weighted_sample_without_replacement(pool, weights, need, rng))

        # Fill any shortfall from the remaining pool with uniform coverage
        if len(selected) < batch_size:
            remaining_pool = [ep for ep in episodes if ep not in selected]
            weights = self._compute_weights(remaining_pool, strategy=weight_strategy)
            selected.extend(
                _weighted_sample_without_replacement(
                    remaining_pool, weights, batch_size - len(selected), rng
                )
            )
        return selected[:batch_size]

    def _sample_frontier_prioritized(self, batch_size: int, rng: random.Random) -> List[Dict[str, Any]]:
        """Bias sampling toward high ΔMPL/ΔJ datapacks while keeping diversity."""
        episodes = self._eligible_pool()
        scores = [ep["frontier_score"] for ep in episodes]
        threshold = _percentile(scores, 0.65)
        urgent = [ep for ep in episodes if ep["frontier_score"] >= threshold]
        non_urgent = [ep for ep in episodes if ep["frontier_score"] < threshold]

        urgent_count = min(max(int(batch_size * 0.7), 1), len(urgent))
        weights_urgent = self._compute_weights(urgent, strategy="frontier_prioritized")
        selected = _weighted_sample_without_replacement(urgent, weights_urgent, urgent_count, rng)

        remaining = batch_size - len(selected)
        if remaining > 0:
            fallback_pool = non_urgent if non_urgent else [ep for ep in urgent if ep not in selected]
            weights_fallback = self._compute_weights(fallback_pool, strategy="balanced")
            selected.extend(_weighted_sample_without_replacement(fallback_pool, weights_fallback, remaining, rng))
        return selected[:batch_size]

    def _sample_econ_urgency(self, batch_size: int, rng: random.Random) -> List[Dict[str, Any]]:
        """Weight by economic urgency and novelty, with a diversity buffer."""
        episodes = self._eligible_pool()
        scores = [ep["econ_urgency_score"] for ep in episodes]
        threshold = _percentile(scores, 0.6)
        urgent = [ep for ep in episodes if ep["econ_urgency_score"] >= threshold]
        baseline = [ep for ep in episodes if ep["econ_urgency_score"] < threshold]

        urgent_count = min(max(int(batch_size * 0.65), 1), len(urgent))
        critical = [ep for ep in urgent if (ep["enrichment"].get("supervision_hints", {}) or {}).get("priority_level", "").lower() == "critical"]
        critical_sorted = sorted(critical, key=lambda ep: ep["econ_urgency_score"], reverse=True)
        critical_selection = critical_sorted[: min(len(critical_sorted), urgent_count)]
        selected = list(critical_selection)

        remaining_urgent = [ep for ep in urgent if ep not in selected]
        weights_urgent = self._compute_weights(remaining_urgent, strategy="econ_urgency")
        selected.extend(_weighted_sample_without_replacement(remaining_urgent, weights_urgent, urgent_count - len(selected), rng))

        remaining = batch_size - len(selected)
        if remaining > 0:
            pool = baseline if baseline else [ep for ep in urgent if ep not in selected]
            weights = self._compute_weights(pool, strategy="balanced")
            selected.extend(_weighted_sample_without_replacement(pool, weights, remaining, rng))
        return selected[:batch_size]

    def _compute_frontier_score(self, episode: Dict[str, Any]) -> float:
        desc = episode["descriptor"]
        enrichment = episode["enrichment"]
        tier_weight = {0: 0.8, 1: 1.0, 2: 1.3}.get(desc.get("tier", 1), 1.0)
        trust = float(desc.get("trust_score", 0.5))
        delta_mpl = _safe_float(desc.get("delta_mpl"), 0.0)
        delta_J = _safe_float(desc.get("delta_J"), 0.0)
        novelty = _extract_max_novelty(enrichment)
        expected_gain = _extract_expected_mpl_gain(enrichment)

        base = max(delta_mpl, 0.0) + 0.7 * max(delta_J, 0.0) + 0.5 * expected_gain
        novelty_boost = 1.0 + 0.5 * novelty
        trust_boost = 0.7 + 0.6 * trust
        return max(base, 0.0) * tier_weight * novelty_boost * trust_boost

    def _compute_econ_urgency_score(self, episode: Dict[str, Any]) -> float:
        desc = episode["descriptor"]
        enrichment = episode["enrichment"]
        novelty = _extract_max_novelty(enrichment)
        expected_gain = _extract_expected_mpl_gain(enrichment)
        coherence = _safe_float(enrichment.get("coherence_score"), 0.0)
        hints = enrichment.get("supervision_hints", {}) or {}
        priority_level = (hints.get("priority_level") or "medium").lower()
        priority_boost = {"low": 0.8, "medium": 1.0, "high": 1.1, "critical": 1.2}.get(priority_level, 1.0)
        weight_mult = _safe_float(hints.get("suggested_weight_multiplier"), 1.0)
        trust = float(desc.get("trust_score", 0.5))
        tier_weight = {0: 0.8, 1: 1.0, 2: 1.15}.get(desc.get("tier", 1), 1.0)

        econ_signal = 0.45 * novelty + 0.45 * min(expected_gain / 10.0, 1.0) + 0.1 * coherence
        econ_signal *= priority_boost * weight_mult
        econ_signal *= (0.6 + 0.4 * trust) * tier_weight
        if trust > 0.8:
            econ_signal *= 5.0
        elif trust > 0.5:
            econ_signal *= 1.5
        return max(econ_signal, 0.0)

    def _recap_weight_multiplier(self, episode: Dict[str, Any]) -> float:
        if not self.use_recap_weights:
            return 1.0
        score = episode.get("recap_goodness_score")
        if score is None:
            desc = episode.get("descriptor", {})
            pid = desc.get("pack_id") or desc.get("episode_id")
            score = self.recap_scores.get(pid)
        if score is None:
            return 1.0
        norm = max(-1.0, min(1.0, float(score) / 5.0))
        return max(0.8, min(1.2, 1.0 + 0.2 * norm))

    def _format_output(self, episode: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        descriptor = copy.deepcopy(episode["descriptor"])
        descriptor["enrichment"] = copy.deepcopy(episode["enrichment"])
        descriptor["sampling_metadata"] = {
            "strategy": strategy,
            "source": episode["source"],
            "frontier_score": episode["frontier_score"],
            "econ_urgency_score": episode["econ_urgency_score"],
            "novelty_score": episode["novelty_score"],
            "expected_mpl_gain": episode["expected_mpl_gain"],
        }
        if self.use_condition_vector:
            tags = descriptor.get("semantic_tags") or {}
            tag_map = {str(t): 1.0 for t in tags} if isinstance(tags, list) else dict(tags)
            skill_mode = self.skill_resolver.resolve(
                tags=tag_map,
                trust_matrix=self.trust_matrix,
                curriculum_phase=descriptor["sampling_metadata"].get("phase", "warmup"),
                advisory=descriptor["sampling_metadata"],
                econ_slice=None,
                recap_bucket=None,
                strategy=strategy,
                use_condition_vector=self.use_condition_vector,
            )
            descriptor["sampling_metadata"]["skill_mode"] = skill_mode
            descriptor["sampling_metadata"]["condition_metadata"] = _summarize_condition_metadata(
                skill_mode, tag_map, descriptor["sampling_metadata"].get("phase", "warmup")
            )
            descriptor["condition_metadata"] = descriptor["sampling_metadata"]["condition_metadata"]
        if episode.get("auditor_result"):
            descriptor["sampling_metadata"]["auditor_rating"] = episode["auditor_result"].get("rating")
            descriptor["sampling_metadata"]["auditor_predicted_econ"] = episode["auditor_result"].get("predicted_econ")
            meta = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), dict) else {}
            meta = copy.deepcopy(meta)
            meta["auditor_rating"] = episode["auditor_result"].get("rating")
            meta["auditor_score"] = episode["auditor_result"].get("score")
            meta["auditor_predicted_econ"] = episode["auditor_result"].get("predicted_econ")
            descriptor["metadata"] = meta
        return to_json_safe(descriptor)


def _normalize_enrichment(enrichment: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure enrichment payload from Stage 2 is JSON-safe and keyed."""
    base = {
        "novelty_tags": [],
        "fragility_tags": [],
        "risk_tags": [],
        "affordance_tags": [],
        "efficiency_tags": [],
        "intervention_tags": [],
        "semantic_conflicts": [],
        "coherence_score": 0.0,
        "supervision_hints": {
            "prioritize_for_training": False,
            "priority_level": "medium",
            "suggested_weight_multiplier": 1.0,
            "suggested_replay_frequency": "standard",
            "requires_human_review": False,
            "safety_critical": False,
            "curriculum_stage": "mid",
            "prerequisite_tags": [],
        },
        "confidence": 0.0,
    }
    if enrichment is None:
        return copy.deepcopy(base)

    merged = copy.deepcopy(base)
    payload = enrichment.get("enrichment", enrichment) if isinstance(enrichment, dict) else {}
    for key, value in payload.items():
        if key == "supervision_hints" and isinstance(value, dict):
            merged["supervision_hints"].update(value)
        elif key in merged:
            merged[key] = value if value is not None else merged[key]
    return to_json_safe(merged)


def _balanced_weight(episode: Dict[str, Any]) -> float:
    desc = episode["descriptor"]
    trust = float(desc.get("trust_score", 0.5))
    sampling_weight = float(desc.get("sampling_weight", 1.0))
    novelty = float(episode.get("novelty_score", 0.0))
    base = max(0.1, 0.4 * sampling_weight + 0.4 * trust + 0.2 * novelty)
    advisory = episode.get("advisory")
    if advisory and getattr(advisory, "datapack_priority_tags", None):
        tags = desc.get("semantic_tags", []) or []
        if any(tag in advisory.datapack_priority_tags for tag in tags):
            base *= 1.2
    base *= episode.get("recap_weight_multiplier", 1.0)
    base *= episode.get("auditor_weight_multiplier", 1.0)
    return base


def _weighted_sample_without_replacement(
    items: List[Dict[str, Any]],
    weights: List[float],
    count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if not items or count <= 0:
        return []
    available: List[Tuple[Dict[str, Any], float]] = []
    for item, w in zip(items, weights):
        available.append((item, max(float(w), 0.0)))
    selected: List[Dict[str, Any]] = []
    count = min(count, len(available))
    for _ in range(count):
        total = sum(w if w > 0 else 1e-6 for _, w in available)
        r = rng.random() * total
        cumulative = 0.0
        choice_idx = 0
        for idx, (item, weight) in enumerate(available):
            cumulative += (weight if weight > 0 else 1e-6)
            if cumulative >= r:
                choice_idx = idx
                break
        item, _ = available.pop(choice_idx)
        selected.append(item)
    return selected


def _extract_max_novelty(enrichment: Dict[str, Any]) -> float:
    max_score = 0.0
    for tag in enrichment.get("novelty_tags", []) or []:
        try:
            max_score = max(max_score, float(tag.get("novelty_score", 0.0)))
        except Exception:
            continue
    return max_score


def _extract_expected_mpl_gain(enrichment: Dict[str, Any]) -> float:
    total_gain = 0.0
    for tag in enrichment.get("novelty_tags", []) or []:
        try:
            total_gain += float(tag.get("expected_mpl_gain", 0.0))
        except Exception:
            continue
    return total_gain


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(max(0, min(len(values_sorted) - 1, quantile * (len(values_sorted) - 1))))
    return float(values_sorted[idx])


def load_episode_descriptors_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL descriptors (Stage 1 or existing) with validation."""
    descriptors: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            normalized, errors = normalize_and_validate(record)
            if errors:
                raise ValueError(f"Invalid descriptor in {path}: {errors}")
            descriptors.append(normalized)
    return descriptors


def load_enrichments_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load Stage 2 semantic enrichments from JSONL."""
    enrichments: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            enrichments.append(json.loads(line))
    return enrichments
