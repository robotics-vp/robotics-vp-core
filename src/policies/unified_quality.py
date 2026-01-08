"""
Unified Quality Gating.

Combines quality signals into a single weight used for:
- Dataset sampling
- Datapack valuation
- Training eligibility

Signals:
1. MHN motion quality (plausibility, structural difficulty)
2. SceneIR quality (convergence, visibility)
3. ProcessReward quality (confidence, progress, disagreement)
4. Map-First quality (static map coverage, dynamics stability)
5. Embodiment quality (contacts/affordances, drift diagnostics; optional)

Formula:
    w = w_mhn * w_scene_ir * w_process_reward * w_map_first
    (optionally include w_embodiment when enabled in config)

Each component is in [0, 1], so combined weight is also in [0, 1].
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.utils.json_safe import to_json_safe


@dataclass
class UnifiedQualityWeights:
    """Container for individual and combined quality weights."""
    # Individual weights (each in [0, 1])
    w_mhn: float = 1.0
    w_scene_ir: float = 1.0
    w_process_reward: float = 1.0
    w_map_first: float = 1.0
    w_embodiment: float = 1.0

    # Combined weight
    w_combined: float = 1.0

    # Eligibility flags
    is_eligible: bool = True
    eligibility_reason: str = "passed"

    # Debug info
    components: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_mhn": self.w_mhn,
            "w_scene_ir": self.w_scene_ir,
            "w_process_reward": self.w_process_reward,
            "w_map_first": self.w_map_first,
            "w_embodiment": self.w_embodiment,
            "w_combined": self.w_combined,
            "is_eligible": self.is_eligible,
            "eligibility_reason": self.eligibility_reason,
            "components": to_json_safe(self.components),
        }


@dataclass
class UnifiedQualityConfig:
    """Configuration for unified quality computation."""
    # Eligibility thresholds
    min_combined_weight: float = 0.1  # Below this = not eligible
    min_mhn_plausibility: float = 0.2  # Below this = immediate reject
    min_process_reward_conf: float = 0.2  # Below this = immediate reject
    min_scene_tracks_quality: float = 0.2  # Below this = immediate reject
    min_embodiment_weight: float = 0.0  # Only used if include_embodiment_weight is True

    # Weight computation parameters
    mhn_plausibility_weight: float = 0.7  # Weight of plausibility in MHN score
    mhn_difficulty_penalty: float = 0.3  # Penalty for high difficulty

    scene_ir_convergence_weight: float = 0.6  # Weight of convergence
    scene_ir_visibility_weight: float = 0.4  # Weight of visibility

    # Stagnation detection
    detect_stagnation: bool = True
    stagnation_conf_min: float = 0.4
    stagnation_delta_max: float = 0.05

    # Embodiment integration
    include_embodiment_weight: bool = False  # Off by default to avoid reward coupling


class UnifiedQualityPolicy:
    """Computes unified quality weights from multiple signals.

    Usage:
        policy = UnifiedQualityPolicy()
        weights = policy.compute(
            mhn_plausibility=0.8,
            mhn_difficulty=0.3,
            scene_ir_convergence=0.9,
            scene_ir_visibility=0.85,
            process_reward_conf=0.7,
            process_reward_delta=0.2,
            process_reward_disagreement=0.1,
        )
        if weights.is_eligible:
            sampler.add_episode(episode, weight=weights.w_combined)
    """

    def __init__(self, config: Optional[UnifiedQualityConfig] = None):
        self.config = config or UnifiedQualityConfig()

    def compute(
        self,
        # MHN signals
        mhn_plausibility: float = 1.0,
        mhn_difficulty: float = 0.0,
        # SceneIR signals
        scene_ir_convergence: float = 1.0,
        scene_ir_visibility: float = 1.0,
        scene_ir_quality: Optional[float] = None,  # If pre-computed
        scene_tracks_quality: Optional[Any] = None,
        # ProcessReward signals
        process_reward_conf: float = 0.5,
        process_reward_conf_p10: float = 0.5,
        process_reward_delta: float = 0.0,
        process_reward_disagreement: float = 0.0,
        process_reward_quality: Optional[float] = None,  # If pre-computed
        # Map-First signals
        map_first_quality: Optional[float] = None,
        # Embodiment signals
        embodiment_weight: Optional[float] = None,
        embodiment_drift_score: Optional[float] = None,
        embodiment_impossible_contacts: Optional[int] = None,
        # Semantic fusion diagnostics (observability only)
        semantic_fusion_confidence_mean: Optional[float] = None,
        semantic_disagreement_vla_vs_map: Optional[float] = None,
        # Episode info
        num_frames: int = 0,
    ) -> UnifiedQualityWeights:
        """Compute unified quality weights.

        Args:
            mhn_plausibility: MHN plausibility score in [0, 1].
            mhn_difficulty: MHN structural difficulty in [0, 1].
            scene_ir_convergence: SceneIR convergence rate in [0, 1].
            scene_ir_visibility: SceneIR visibility rate in [0, 1].
            scene_ir_quality: Pre-computed SceneIR quality (overrides above).
            process_reward_conf: Mean process reward confidence.
            process_reward_conf_p10: 10th percentile confidence.
            process_reward_delta: Phi_star progress (final - initial).
            process_reward_disagreement: Mean perspective disagreement.
            process_reward_quality: Pre-computed PR quality (overrides above).
            map_first_quality: Optional Map-First quality score in [0, 1].
            embodiment_weight: Optional embodiment weight in [0, 1].
            embodiment_drift_score: Optional drift score in [0, 1].
            embodiment_impossible_contacts: Optional count of impossible contacts.
            num_frames: Number of frames (for stagnation detection).

        Returns:
            UnifiedQualityWeights with combined weight and eligibility.
        """
        cfg = self.config
        components = {}

        # --- MHN weight ---
        # w_mhn = plausibility * (1 - difficulty_penalty * difficulty)
        difficulty_factor = 1.0 - cfg.mhn_difficulty_penalty * min(1.0, mhn_difficulty)
        w_mhn = mhn_plausibility * difficulty_factor
        w_mhn = max(0.0, min(1.0, w_mhn))
        components["mhn"] = {
            "plausibility": mhn_plausibility,
            "difficulty": mhn_difficulty,
            "difficulty_factor": difficulty_factor,
        }

        # --- SceneIR weight ---
        if scene_ir_quality is not None:
            w_scene_ir = scene_ir_quality
        else:
            w_scene_ir = (
                cfg.scene_ir_convergence_weight * scene_ir_convergence +
                cfg.scene_ir_visibility_weight * scene_ir_visibility
            )
        scene_tracks_quality_val = _extract_scene_tracks_quality(scene_tracks_quality)
        if scene_tracks_quality_val is not None:
            w_scene_ir = min(w_scene_ir, scene_tracks_quality_val)
        w_scene_ir = max(0.0, min(1.0, w_scene_ir))
        components["scene_ir"] = {
            "convergence": scene_ir_convergence,
            "visibility": scene_ir_visibility,
            "quality": w_scene_ir,
        }
        if scene_tracks_quality_val is not None:
            components["scene_tracks"] = {"quality": scene_tracks_quality_val}

        # --- ProcessReward weight ---
        if process_reward_quality is not None:
            w_process_reward = process_reward_quality
        else:
            # Formula: conf * (1 + max(0, delta)) * (1 - 0.3 * disagreement)
            progress_factor = max(0.0, process_reward_delta)
            disagreement_penalty = 0.3 * min(1.0, process_reward_disagreement)
            w_process_reward = (
                process_reward_conf *
                (1.0 + progress_factor) *
                (1.0 - disagreement_penalty)
            )
            # Normalize to [0, 1] (max is ~2 when conf=1, delta=1, disagreement=0)
            w_process_reward = min(1.0, w_process_reward / 2.0)
        w_process_reward = max(0.0, min(1.0, w_process_reward))
        components["process_reward"] = {
            "conf_mean": process_reward_conf,
            "conf_p10": process_reward_conf_p10,
            "delta": process_reward_delta,
            "disagreement": process_reward_disagreement,
            "quality": w_process_reward,
        }

        # --- Map-First weight ---
        if map_first_quality is not None:
            w_map_first = max(0.0, min(1.0, map_first_quality))
        else:
            w_map_first = 1.0
        components["map_first"] = {
            "quality": w_map_first,
        }
        if semantic_fusion_confidence_mean is not None or semantic_disagreement_vla_vs_map is not None:
            components["semantic_fusion"] = {
                "confidence_mean": semantic_fusion_confidence_mean,
                "disagreement_vla_vs_map": semantic_disagreement_vla_vs_map,
            }

        # --- Embodiment weight ---
        w_embodiment = 1.0
        if embodiment_weight is not None:
            w_embodiment = max(0.0, min(1.0, float(embodiment_weight)))
        components["embodiment"] = {
            "w_embodiment": w_embodiment,
            "drift_score": embodiment_drift_score,
            "physically_impossible_contacts": embodiment_impossible_contacts,
        }

        # --- Combined weight ---
        w_combined = w_mhn * w_scene_ir * w_process_reward * w_map_first
        if cfg.include_embodiment_weight:
            w_combined *= w_embodiment

        # --- Eligibility checks ---
        is_eligible = True
        eligibility_reason = "passed"

        # Hard gates
        if mhn_plausibility < cfg.min_mhn_plausibility:
            is_eligible = False
            eligibility_reason = f"mhn_plausibility={mhn_plausibility:.2f} < {cfg.min_mhn_plausibility}"
        elif scene_tracks_quality_val is not None and scene_tracks_quality_val < cfg.min_scene_tracks_quality:
            is_eligible = False
            eligibility_reason = (
                f"scene_tracks_quality={scene_tracks_quality_val:.2f} < {cfg.min_scene_tracks_quality}"
            )
        elif process_reward_conf_p10 < cfg.min_process_reward_conf:
            is_eligible = False
            eligibility_reason = f"conf_p10={process_reward_conf_p10:.2f} < {cfg.min_process_reward_conf}"
        elif cfg.include_embodiment_weight and w_embodiment < cfg.min_embodiment_weight:
            is_eligible = False
            eligibility_reason = f"w_embodiment={w_embodiment:.2f} < {cfg.min_embodiment_weight}"
        elif w_combined < cfg.min_combined_weight:
            is_eligible = False
            eligibility_reason = f"w_combined={w_combined:.3f} < {cfg.min_combined_weight}"

        # Stagnation detection
        is_stagnant = False
        if cfg.detect_stagnation and is_eligible:
            is_confident = process_reward_conf_p10 >= cfg.stagnation_conf_min
            is_stuck = abs(process_reward_delta) <= cfg.stagnation_delta_max
            is_long = num_frames >= 10 if num_frames > 0 else False
            is_stagnant = is_confident and is_stuck and is_long
            if is_stagnant:
                components["stagnation"] = {
                    "detected": True,
                    "conf_p10": process_reward_conf_p10,
                    "delta": process_reward_delta,
                    "num_frames": num_frames,
                }

        return UnifiedQualityWeights(
            w_mhn=w_mhn,
            w_scene_ir=w_scene_ir,
            w_process_reward=w_process_reward,
            w_map_first=w_map_first,
            w_embodiment=w_embodiment,
            w_combined=w_combined,
            is_eligible=is_eligible,
            eligibility_reason=eligibility_reason,
            components=components,
        )

    def compute_from_datapack(
        self,
        datapack: "DataPackMeta",
        mhn_summary: Optional[Any] = None,
        scene_ir_quality: Optional[float] = None,
        scene_tracks_quality: Optional[Any] = None,
    ) -> UnifiedQualityWeights:
        """Compute weights from a DataPackMeta object.

        Args:
            datapack: DataPackMeta with optional process_reward_profile.
            mhn_summary: Optional MHNSummary object.
            scene_ir_quality: Optional pre-computed SceneIR quality.

        Returns:
            UnifiedQualityWeights.
        """
        # Extract MHN signals
        mhn_plausibility = 1.0
        mhn_difficulty = 0.0
        if mhn_summary is not None:
            mhn_plausibility = getattr(mhn_summary, "plausibility_score", 1.0)
            mhn_difficulty = getattr(mhn_summary, "structural_difficulty", 0.0)

        # Extract ProcessReward signals
        pr_conf = 0.5
        pr_conf_p10 = 0.5
        pr_delta = 0.0
        pr_disagreement = 0.0
        pr_quality = None
        num_frames = 0
        w_embodiment = None
        embodiment_drift = None
        embodiment_impossible = None

        if datapack.process_reward_profile is not None:
            prp = datapack.process_reward_profile
            pr_conf = prp.conf_mean
            pr_conf_p10 = prp.conf_p10
            pr_delta = prp.phi_star_delta
            pr_disagreement = prp.disagreement_mean
            pr_quality = max(0.0, prp.quality_score())

        # Get num_frames from episode_metrics if available
        if datapack.episode_metrics:
            num_frames = datapack.episode_metrics.get("num_frames", 0)
        map_first_quality = None
        if datapack.episode_metrics:
            map_first_quality = datapack.episode_metrics.get("map_first_quality_score")
            if map_first_quality is None:
                summary = datapack.episode_metrics.get("map_first_summary", {})
                if isinstance(summary, dict):
                    map_first_quality = summary.get("map_first_quality_score")
        if map_first_quality is not None:
            try:
                map_first_quality = float(map_first_quality)
            except Exception:
                map_first_quality = None

        sem_conf = None
        sem_disagreement = None
        if datapack.episode_metrics:
            sem_conf = datapack.episode_metrics.get("semantic_fusion_confidence_mean")
            sem_disagreement = datapack.episode_metrics.get("semantic_disagreement_vla_vs_map")
            if w_embodiment is None:
                w_embodiment = datapack.episode_metrics.get("w_embodiment")
            if embodiment_drift is None:
                embodiment_drift = datapack.episode_metrics.get("embodiment_drift_score")
            if embodiment_impossible is None:
                embodiment_impossible = datapack.episode_metrics.get("embodiment_physically_impossible_contacts")
        if sem_conf is not None:
            try:
                sem_conf = float(sem_conf)
            except Exception:
                sem_conf = None
        if sem_disagreement is not None:
            try:
                sem_disagreement = float(sem_disagreement)
            except Exception:
                sem_disagreement = None

        if datapack.embodiment_profile is not None:
            emb = datapack.embodiment_profile
            w_embodiment = emb.w_embodiment
            embodiment_drift = emb.drift_score
            embodiment_impossible = emb.physically_impossible_contacts

        return self.compute(
            mhn_plausibility=mhn_plausibility,
            mhn_difficulty=mhn_difficulty,
            scene_ir_quality=scene_ir_quality,
            scene_tracks_quality=scene_tracks_quality,
            process_reward_conf=pr_conf,
            process_reward_conf_p10=pr_conf_p10,
            process_reward_delta=pr_delta,
            process_reward_disagreement=pr_disagreement,
            process_reward_quality=pr_quality,
            map_first_quality=map_first_quality if map_first_quality is not None else None,
            embodiment_weight=w_embodiment,
            embodiment_drift_score=embodiment_drift,
            embodiment_impossible_contacts=embodiment_impossible,
            semantic_fusion_confidence_mean=sem_conf,
            semantic_disagreement_vla_vs_map=sem_disagreement,
            num_frames=num_frames,
        )

    def filter_eligible(
        self,
        datapacks: list,
        mhn_summaries: Optional[Dict[str, Any]] = None,
        scene_ir_qualities: Optional[Dict[str, float]] = None,
        scene_tracks_qualities: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Filter datapacks by eligibility.

        Args:
            datapacks: List of DataPackMeta.
            mhn_summaries: Optional dict of pack_id -> MHNSummary.
            scene_ir_qualities: Optional dict of pack_id -> quality score.

        Returns:
            (eligible, ineligible, stagnant) lists.
        """
        eligible = []
        ineligible = []
        stagnant = []

        mhn_summaries = mhn_summaries or {}
        scene_ir_qualities = scene_ir_qualities or {}
        scene_tracks_qualities = scene_tracks_qualities or {}

        for dp in datapacks:
            pack_id = dp.pack_id
            weights = self.compute_from_datapack(
                dp,
                mhn_summary=mhn_summaries.get(pack_id),
                scene_ir_quality=scene_ir_qualities.get(pack_id),
                scene_tracks_quality=scene_tracks_qualities.get(pack_id),
            )

            if not weights.is_eligible:
                ineligible.append((dp, weights))
            elif weights.components.get("stagnation", {}).get("detected", False):
                stagnant.append((dp, weights))
                eligible.append((dp, weights))  # Stagnant is still eligible
            else:
                eligible.append((dp, weights))

        return eligible, ineligible, stagnant


def _extract_scene_tracks_quality(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        score = value.get("quality_score") or value.get("scene_tracks_quality")
        try:
            return float(score)
        except (TypeError, ValueError):
            return None
    if hasattr(value, "quality_score"):
        try:
            return float(getattr(value, "quality_score"))
        except (TypeError, ValueError):
            return None
    return None
