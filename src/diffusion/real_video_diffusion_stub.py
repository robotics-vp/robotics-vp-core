"""
Real Video Diffusion Stub for Stage 1/4 integration.

This is a placeholder for actual diffusion model integration.
It provides structured interfaces for proposing augmented video clips
based on semantic tags and economic context.

No GPU, no actual generation - just shimming in the interfaces.
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.utils.json_safe import to_json_safe


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class DiffusionProposal:
    """
    Proposed augmented video clip from diffusion model.

    This is what the diffusion model would generate if we had one.
    For now, it's a structured placeholder that the orchestrator can consume.
    """
    proposal_id: str
    episode_id: str
    media_refs: List[str]  # Original video references
    augmentation_type: str  # "failed_grasp", "high_speed", "edge_case", etc.
    semantic_tags: List[str]  # Tags from VLA + SemanticOrchestrator
    objective_preset: str  # "throughput", "safety", "energy_saver", etc.
    energy_profile: str  # "BASE", "BOOST", "SAVER", "SAFE"
    econ_context: Dict[str, float]  # Wage, energy price, customer segment
    confidence: float  # Model's confidence in this proposal
    estimated_novelty: float  # Expected novelty score of generated clip
    rationale: str  # Why this clip was proposed
    timestamp: float
    constraint_set: Optional[Dict[str, Any]] = None
    routing_source: str = "semantic_tag_fallback"
    routing_score: float = 0.0
    benchmark_gate_ready: bool = False
    benchmark_signals: Dict[str, Any] = field(default_factory=dict)
    source_hypothesis_id: Optional[str] = None


@dataclass
class SyntheticEpisodeProposal:
    """
    Proposed synthetic episode (datapack-like) from diffusion stub.

    This would be used to bootstrap training from video demonstrations.
    """
    episode_id: str
    source_type: str  # "synthetic_diffusion"
    diffusion_proposals: List[DiffusionProposal]
    objective_preset: str
    energy_profile: str
    semantic_tags: List[str]
    econ_context: Dict[str, float]
    expected_mpl_gain: float  # Expected learning gain from this episode
    estimated_tier: int  # 0=redundant, 1=context-novel, 2=frontier


class VideoDiffusionStub:
    """
    Stub class for video diffusion model integration.

    This is a placeholder that will be replaced with actual diffusion model
    when we have GPU resources and cloud pipeline.

    For now, it provides structured sampling logic and proposals.
    """

    def __init__(self):
        self.model_version = "stub-v1.0"
        self.proposal_counter = 0

    def _routing_score(
        self,
        hypothesis: Dict[str, Any],
        routing_context: Optional[Dict[str, Any]],
    ) -> float:
        context = dict(routing_context or {})
        scores = hypothesis.get("scores", {}) if isinstance(hypothesis, dict) else {}
        mode = str(hypothesis.get("mode", "")) if isinstance(hypothesis, dict) else ""
        render_priority = _safe_float(scores.get("render_priority", 0.0), 0.0)
        plausibility = _safe_float(scores.get("plausibility", 0.0), 0.0)
        novelty = _safe_float(scores.get("novelty", 0.0), 0.0)
        economic_priority = max(
            _safe_float(scores.get("economic_priority", 0.0), 0.0),
            _safe_float(context.get("economic_priority_score", 0.0), 0.0),
        )
        trust_priority = max(
            _safe_float(scores.get("trust_priority", 0.0), 0.0),
            _safe_float(context.get("trust_priority_score", 0.0), 0.0),
        )
        coverage_gap = _safe_float(context.get("coverage_gap_score", 0.0), 0.0)
        evidence_coverage = _safe_float(context.get("evidence_coverage", 0.0), 0.0)
        benchmark_ready = bool(context.get("benchmark_gate_ready", False))
        grounding_mode = str(context.get("semantic_grounding_mode", "") or "")
        heuristic_penalty = 0.0
        if grounding_mode in {"heuristic", "heuristic_fallback", "keyword_tags", "coverage_gap_pending"}:
            heuristic_penalty = 0.12
        mode_bonus = 0.0
        risk_targets = list(context.get("risk_family_targets", []) or [])
        affordance_targets = list(context.get("affordance_family_targets", []) or [])
        missing_env = list(context.get("missing_env_primitives", []) or [])
        if risk_targets and "fragile" in mode:
            mode_bonus += 0.12
        if affordance_targets and "disambiguation" in mode:
            mode_bonus += 0.1
        if missing_env and "geometry" in mode:
            mode_bonus += 0.08
        if context.get("objective_preset") == "throughput" and "throughput" in mode:
            mode_bonus += 0.1
        if context.get("objective_preset") == "energy_saver" and "energy" in mode:
            mode_bonus += 0.1
        score = (
            0.3 * render_priority
            + 0.25 * plausibility
            + 0.15 * novelty
            + 0.15 * economic_priority
            + 0.1 * trust_priority
            + 0.05 * coverage_gap
            + 0.05 * evidence_coverage
            + mode_bonus
        )
        if not benchmark_ready:
            score *= 0.75
        score -= heuristic_penalty
        return _clip01(score)

    def _sorted_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        routing_context: Optional[Dict[str, Any]],
        num_proposals: int,
    ) -> List[Dict[str, Any]]:
        ranked = []
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            scored = dict(hypothesis)
            scored["routing_score"] = self._routing_score(scored, routing_context)
            render_intent = dict(scored.get("render_intent", {}) or {})
            if render_intent.get("should_render", True) is False:
                continue
            ranked.append(scored)
        ranked.sort(
            key=lambda item: (
                _safe_float(item.get("routing_score", 0.0), 0.0),
                _safe_float(dict(item.get("scores", {}) or {}).get("render_priority", 0.0), 0.0),
                _safe_float(dict(item.get("scores", {}) or {}).get("plausibility", 0.0), 0.0),
            ),
            reverse=True,
        )
        return ranked[: max(int(num_proposals), 1)]

    def _fallback_candidates(
        self,
        semantic_tags: List[str],
        objective_preset: str,
        energy_profile: str,
        routing_context: Optional[Dict[str, Any]],
        num_proposals: int,
    ) -> List[Dict[str, Any]]:
        context = dict(routing_context or {})
        tags = " ".join(semantic_tags)
        candidates: List[Dict[str, Any]] = []
        risk_targets = list(context.get("risk_family_targets", []) or [])
        affordance_targets = list(context.get("affordance_family_targets", []) or [])
        missing_env = list(context.get("missing_env_primitives", []) or [])
        if risk_targets or "fragile" in semantic_tags or "safety" in tags:
            candidates.append(
                {
                    "mode": "fragile_object_preservation",
                    "confidence": 0.55,
                    "novelty": 0.45,
                    "rationale": "Fallback preserved risk-targeted routing because fragile or safety pressure is present.",
                }
            )
        if affordance_targets or missing_env:
            candidates.append(
                {
                    "mode": "semantic_disambiguation",
                    "confidence": 0.5,
                    "novelty": 0.42,
                    "rationale": "Fallback preserved gap-targeted routing because affordance or env-primitive evidence is incomplete.",
                }
            )
        if "error" in tags or "recover" in tags:
            candidates.append(
                {
                    "mode": "recovery_branch",
                    "confidence": 0.58,
                    "novelty": 0.48,
                    "rationale": "Fallback preserved recovery routing because error signatures are present.",
                }
            )
        if objective_preset == "throughput" or "high_speed" in semantic_tags:
            candidates.append(
                {
                    "mode": "throughput_push",
                    "confidence": 0.52,
                    "novelty": 0.4,
                    "rationale": "Fallback preserved throughput routing because the objective emphasizes throughput.",
                }
            )
        if objective_preset == "energy_saver" or energy_profile == "SAVER":
            candidates.append(
                {
                    "mode": "energy_saver_retiming",
                    "confidence": 0.5,
                    "novelty": 0.38,
                    "rationale": "Fallback preserved energy routing because the objective emphasizes efficiency.",
                }
            )
        if not candidates:
            candidates.append(
                {
                    "mode": "geometry_guarded_continuation",
                    "confidence": 0.45,
                    "novelty": 0.3,
                    "rationale": "Fallback used guarded continuation because no governed routing payload was available.",
                }
            )
        benchmark_ready = bool(context.get("benchmark_gate_ready", False))
        grounding_mode = str(context.get("semantic_grounding_mode", "") or "")
        if not benchmark_ready or grounding_mode in {"heuristic", "heuristic_fallback", "keyword_tags", "coverage_gap_pending"}:
            for candidate in candidates:
                candidate["confidence"] = min(candidate["confidence"], 0.45)
                candidate["novelty"] = min(candidate["novelty"], 0.35)
                candidate["rationale"] += " Benchmark gate is not ready, so novelty/confidence are clamped."
        return candidates[: max(1, min(num_proposals, len(candidates)))]

    def render_hypotheses(
        self,
        *,
        episode_id: str,
        media_refs: List[str],
        semantic_tags: List[str],
        objective_preset: str,
        hypotheses: List[Dict[str, Any]],
        energy_profile: str = "BASE",
        econ_context: Optional[Dict[str, float]] = None,
        constraint_set: Optional[Dict[str, Any]] = None,
        routing_context: Optional[Dict[str, Any]] = None,
        num_proposals: Optional[int] = None,
    ) -> List[DiffusionProposal]:
        """Render already-scored hypotheses into diffusion-style proposals."""
        if econ_context is None:
            econ_context = {
                "wage_human": 18.0,
                "energy_price_kWh": 0.12,
                "customer_segment": "balanced",
            }

        proposals: List[DiffusionProposal] = []
        benchmark_signals = dict((routing_context or {}).get("benchmark_signals", {}) or {})
        routing_source = str((routing_context or {}).get("routing_source", "governed_hypotheses"))
        hypothesis_limit = len(hypotheses) if num_proposals is None else max(int(num_proposals), 1)
        for hypothesis in self._sorted_hypotheses(hypotheses, routing_context, hypothesis_limit):
            self.proposal_counter += 1
            scores = hypothesis.get("scores", {}) if isinstance(hypothesis, dict) else {}
            render_intent = hypothesis.get("render_intent", {}) if isinstance(hypothesis, dict) else {}
            rationale = str(hypothesis.get("rationale", "Governed geometry-first hypothesis"))
            action_conditioning = hypothesis.get("action_conditioning", {}) if isinstance(hypothesis, dict) else {}
            mode = str(hypothesis.get("mode", "geometry_guarded_continuation"))
            routing_score = _safe_float(hypothesis.get("routing_score", 0.0), 0.0)
            proposal = DiffusionProposal(
                proposal_id=f"diff_prop_{self.proposal_counter}_{int(time.time())}",
                episode_id=episode_id,
                media_refs=media_refs,
                augmentation_type=mode,
                semantic_tags=semantic_tags,
                objective_preset=objective_preset,
                energy_profile=energy_profile,
                econ_context=econ_context,
                confidence=max(float(scores.get("plausibility", 0.5)), routing_score),
                estimated_novelty=max(float(scores.get("novelty", 0.4)), min(routing_score + 0.1, 1.0)),
                rationale=f"{rationale} [routing_source={routing_source}]",
                timestamp=time.time(),
                constraint_set={
                    **(constraint_set or {}),
                    "routing_context": dict(routing_context or {}),
                    "render_intent": render_intent,
                    "action_conditioning": action_conditioning,
                    "hypothesis_id": hypothesis.get("hypothesis_id") if isinstance(hypothesis, dict) else None,
                },
                routing_source=routing_source,
                routing_score=routing_score,
                benchmark_gate_ready=bool((routing_context or {}).get("benchmark_gate_ready", False)),
                benchmark_signals=benchmark_signals,
                source_hypothesis_id=(
                    str(hypothesis.get("hypothesis_id"))
                    if isinstance(hypothesis, dict) and hypothesis.get("hypothesis_id") is not None
                    else None
                ),
            )
            proposals.append(proposal)
        return proposals

    def propose_augmented_clips(
        self,
        episode_id: str,
        media_refs: List[str],
        semantic_tags: List[str],
        objective_preset: str = "balanced",
        energy_profile: str = "BASE",
        econ_context: Optional[Dict[str, float]] = None,
        constraint_set: Optional[Dict[str, Any]] = None,
        hypotheses: Optional[List[Dict[str, Any]]] = None,
        routing_context: Optional[Dict[str, Any]] = None,
        num_proposals: int = 3,
    ) -> List[DiffusionProposal]:
        """
        Propose augmented video clips based on semantic tags and econ context.

        This is where the actual diffusion model would generate proposals.
        For now, returns structured placeholders based on semantic analysis.

        Args:
            episode_id: ID of source episode
            media_refs: References to original video files
            semantic_tags: Tags from VLA + SemanticOrchestrator
            objective_preset: Current objective preset
            energy_profile: Current energy profile
            econ_context: Economic context (wage, energy price, etc.)
            num_proposals: Number of proposals to generate

        Returns:
            List of DiffusionProposal objects
        """
        if econ_context is None:
            econ_context = {
                "wage_human": 18.0,
                "energy_price_kWh": 0.12,
                "customer_segment": "balanced",
            }

        governed_hypotheses = list(hypotheses or list((routing_context or {}).get("governed_hypotheses", []) or []))
        if governed_hypotheses:
            return self.render_hypotheses(
                episode_id=episode_id,
                media_refs=media_refs,
                semantic_tags=semantic_tags,
                objective_preset=objective_preset,
                hypotheses=governed_hypotheses,
                num_proposals=num_proposals,
                energy_profile=energy_profile,
                econ_context=econ_context,
                constraint_set=constraint_set,
                routing_context=routing_context,
            )

        proposals = []
        fallback_candidates = self._fallback_candidates(
            semantic_tags=semantic_tags,
            objective_preset=objective_preset,
            energy_profile=energy_profile,
            routing_context=routing_context,
            num_proposals=num_proposals,
        )
        benchmark_signals = dict((routing_context or {}).get("benchmark_signals", {}) or {})
        routing_source = str((routing_context or {}).get("routing_source", "semantic_tag_fallback"))
        for candidate in fallback_candidates:
            self.proposal_counter += 1
            proposal_id = f"diff_prop_{self.proposal_counter}_{int(time.time())}"
            aug_type = str(candidate.get("mode", "geometry_guarded_continuation"))
            rationale = str(candidate.get("rationale", "Fallback variation"))
            confidence = _safe_float(candidate.get("confidence", 0.4), 0.4)
            novelty = _safe_float(candidate.get("novelty", 0.3), 0.3)
            if constraint_set:
                constrained_axes = sorted(list((constraint_set.get("hard_bounds") or {}).keys()))
                if constrained_axes:
                    rationale += f"; constrained_by={','.join(constrained_axes[:4])}"

            # Add some randomness to confidence and novelty
            confidence = max(0.1, min(1.0, confidence + random.uniform(-0.1, 0.1)))
            novelty = max(0.1, min(1.0, novelty + random.uniform(-0.15, 0.15)))

            proposal = DiffusionProposal(
                proposal_id=proposal_id,
                episode_id=episode_id,
                media_refs=media_refs,
                augmentation_type=aug_type,
                semantic_tags=semantic_tags,
                objective_preset=objective_preset,
                energy_profile=energy_profile,
                econ_context=econ_context,
                confidence=confidence,
                estimated_novelty=novelty,
                rationale=f"{rationale} [routing_source={routing_source}]",
                timestamp=time.time(),
                constraint_set={
                    **(constraint_set or {}),
                    "routing_context": dict(routing_context or {}),
                },
                routing_source=routing_source,
                routing_score=_clip01(0.5 * confidence + 0.5 * novelty),
                benchmark_gate_ready=bool((routing_context or {}).get("benchmark_gate_ready", False)),
                benchmark_signals=benchmark_signals,
            )
            proposals.append(proposal)

        return proposals

    def propose_synthetic_episode(
        self,
        source_episode_id: str,
        semantic_tags: List[str],
        objective_preset: str = "balanced",
        energy_profile: str = "BASE",
        econ_context: Optional[Dict[str, float]] = None,
        constraint_set: Optional[Dict[str, Any]] = None,
    ) -> SyntheticEpisodeProposal:
        """
        Propose a synthetic episode based on existing episode.

        This would generate a complete synthetic episode for training.
        For now, returns structured placeholder.
        """
        if econ_context is None:
            econ_context = {
                "wage_human": 18.0,
                "energy_price_kWh": 0.12,
                "customer_segment": "balanced",
            }

        # Generate clip proposals for this episode
        proposals = self.propose_augmented_clips(
            episode_id=source_episode_id,
            media_refs=[f"synthetic_{source_episode_id}"],
            semantic_tags=semantic_tags,
            objective_preset=objective_preset,
            energy_profile=energy_profile,
            econ_context=econ_context,
            constraint_set=constraint_set,
            num_proposals=2,
        )

        # Estimate tier based on novelty
        max_novelty = max(p.estimated_novelty for p in proposals) if proposals else 0.5
        if max_novelty > 0.7:
            tier = 2  # Frontier
        elif max_novelty > 0.4:
            tier = 1  # Context-novel
        else:
            tier = 0  # Redundant

        # Expected MPL gain correlates with novelty and tier
        expected_mpl_gain = max_novelty * (tier + 1) * 2.0  # Simplified formula

        return SyntheticEpisodeProposal(
            episode_id=f"synthetic_{source_episode_id}_{int(time.time())}",
            source_type="synthetic_diffusion",
            diffusion_proposals=proposals,
            objective_preset=objective_preset,
            energy_profile=energy_profile,
            semantic_tags=semantic_tags,
            econ_context=econ_context,
            expected_mpl_gain=expected_mpl_gain,
            estimated_tier=tier,
        )


def proposal_to_dict(proposal: DiffusionProposal) -> Dict[str, Any]:
    """Convert DiffusionProposal to JSON-serializable dict."""
    return to_json_safe(proposal)


def synthetic_episode_to_dict(episode: SyntheticEpisodeProposal) -> Dict[str, Any]:
    """Convert SyntheticEpisodeProposal to JSON-serializable dict."""
    d = to_json_safe(episode)
    if isinstance(d, dict):
        # Ensure nested proposals are properly serialized
        d["diffusion_proposals"] = [proposal_to_dict(p) for p in episode.diffusion_proposals]
    return d
