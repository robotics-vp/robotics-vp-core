"""
Unified Quality Gating.

Combines three quality signals into a single weight used for:
- Dataset sampling
- Datapack valuation
- Training eligibility

Signals:
1. MHN motion quality (plausibility, structural difficulty)
2. SceneIR quality (convergence, visibility)
3. ProcessReward quality (confidence, progress, disagreement)

Formula:
    w = w_mhn * w_scene_ir * w_process_reward

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

    # Weight computation parameters
    mhn_plausibility_weight: float = 0.7  # Weight of plausibility in MHN score
    mhn_difficulty_penalty: float = 0.3  # Penalty for high difficulty

    scene_ir_convergence_weight: float = 0.6  # Weight of convergence
    scene_ir_visibility_weight: float = 0.4  # Weight of visibility

    # Stagnation detection
    detect_stagnation: bool = True
    stagnation_conf_min: float = 0.4
    stagnation_delta_max: float = 0.05


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
        # ProcessReward signals
        process_reward_conf: float = 0.5,
        process_reward_conf_p10: float = 0.5,
        process_reward_delta: float = 0.0,
        process_reward_disagreement: float = 0.0,
        process_reward_quality: Optional[float] = None,  # If pre-computed
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
        w_scene_ir = max(0.0, min(1.0, w_scene_ir))
        components["scene_ir"] = {
            "convergence": scene_ir_convergence,
            "visibility": scene_ir_visibility,
            "quality": w_scene_ir,
        }

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

        # --- Combined weight ---
        w_combined = w_mhn * w_scene_ir * w_process_reward

        # --- Eligibility checks ---
        is_eligible = True
        eligibility_reason = "passed"

        # Hard gates
        if mhn_plausibility < cfg.min_mhn_plausibility:
            is_eligible = False
            eligibility_reason = f"mhn_plausibility={mhn_plausibility:.2f} < {cfg.min_mhn_plausibility}"
        elif process_reward_conf_p10 < cfg.min_process_reward_conf:
            is_eligible = False
            eligibility_reason = f"conf_p10={process_reward_conf_p10:.2f} < {cfg.min_process_reward_conf}"
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

        return self.compute(
            mhn_plausibility=mhn_plausibility,
            mhn_difficulty=mhn_difficulty,
            scene_ir_quality=scene_ir_quality,
            process_reward_conf=pr_conf,
            process_reward_conf_p10=pr_conf_p10,
            process_reward_delta=pr_delta,
            process_reward_disagreement=pr_disagreement,
            process_reward_quality=pr_quality,
            num_frames=num_frames,
        )

    def filter_eligible(
        self,
        datapacks: list,
        mhn_summaries: Optional[Dict[str, Any]] = None,
        scene_ir_qualities: Optional[Dict[str, float]] = None,
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

        for dp in datapacks:
            pack_id = dp.pack_id
            weights = self.compute_from_datapack(
                dp,
                mhn_summary=mhn_summaries.get(pack_id),
                scene_ir_quality=scene_ir_qualities.get(pack_id),
            )

            if not weights.is_eligible:
                ineligible.append((dp, weights))
            elif weights.components.get("stagnation", {}).get("detected", False):
                stagnant.append((dp, weights))
                eligible.append((dp, weights))  # Stagnant is still eligible
            else:
                eligible.append((dp, weights))

        return eligible, ineligible, stagnant
