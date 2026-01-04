"""
Logging utilities for Process Reward integration with training/rollout pipelines.

Provides structured logging for:
- Rollout labeling / training logs
- Economic reports (correlations with MPL uplift)
- Orchestrator override tracking
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np

from src.process_reward.schemas import (
    ProcessRewardConfig,
    ProcessRewardEpisodeOutput,
    FusionOverride,
    MHNSummary,
)


@dataclass
class ProcessRewardLogEntry:
    """Structured log entry for process reward output.

    Designed to be appended to rollout logs alongside MHN/SceneIR summaries.
    All fields are JSON-serializable.
    """
    # Episode identification
    episode_id: Optional[str] = None
    timestamp: Optional[str] = None

    # Core metrics
    phi_star_mean: float = 0.0
    phi_star_final: float = 0.0
    phi_star_delta: float = 0.0  # final - initial
    conf_mean: float = 0.0
    conf_p10: float = 0.0  # 10th percentile
    conf_min: float = 0.0

    # Shaped reward
    r_shape_sum: float = 0.0
    r_shape_mean: float = 0.0
    r_shape_std: float = 0.0

    # Fusion diagnostics
    disagreement_mean: float = 0.0
    disagreement_max: float = 0.0
    entropy_mean: float = 0.0
    entropy_max: float = 0.0

    # Perspective usage
    weight_I_mean: float = 0.0  # Mean weight for Φ_I
    weight_F_mean: float = 0.0  # Mean weight for Φ_F
    weight_B_mean: float = 0.0  # Mean weight for Φ_B
    phi_B_disabled: bool = False  # Was Φ_B disabled due to hindsight?
    pct_masked_candidates: float = 0.0  # % of candidates masked

    # Episode info
    num_frames: int = 0
    goal_is_hindsight: bool = False

    # Orchestrator settings (for reproducibility)
    fusion_override: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


def extract_log_entry(
    result: ProcessRewardEpisodeOutput,
    episode_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    fusion_override: Optional[FusionOverride] = None,
) -> ProcessRewardLogEntry:
    """Extract a structured log entry from ProcessRewardEpisodeOutput.

    Args:
        result: The episode output from process_reward_episode.
        episode_id: Optional episode identifier.
        timestamp: Optional timestamp string.
        fusion_override: Optional FusionOverride used for this run.

    Returns:
        ProcessRewardLogEntry ready for logging.
    """
    # Core metrics
    phi_star = result.phi_star
    conf = result.conf
    r_shape = result.r_shape
    weights = result.diagnostics.weights

    # Compute percentiles
    conf_p10 = float(np.percentile(conf, 10)) if len(conf) > 0 else 0.0

    # Compute masked candidate percentage
    if fusion_override is not None:
        num_masked = sum(1 for m in fusion_override.candidate_mask if not m)
        pct_masked = num_masked / 3.0
    else:
        pct_masked = 0.0

    return ProcessRewardLogEntry(
        episode_id=episode_id or result.episode_id,
        timestamp=timestamp,
        # Core
        phi_star_mean=float(np.mean(phi_star)),
        phi_star_final=float(phi_star[-1]) if len(phi_star) > 0 else 0.0,
        phi_star_delta=float(phi_star[-1] - phi_star[0]) if len(phi_star) > 0 else 0.0,
        conf_mean=float(np.mean(conf)),
        conf_p10=conf_p10,
        conf_min=float(np.min(conf)) if len(conf) > 0 else 0.0,
        # Shaped reward
        r_shape_sum=float(np.sum(r_shape)),
        r_shape_mean=float(np.mean(r_shape)) if len(r_shape) > 0 else 0.0,
        r_shape_std=float(np.std(r_shape)) if len(r_shape) > 0 else 0.0,
        # Fusion diagnostics
        disagreement_mean=float(np.mean(result.diagnostics.disagreement)),
        disagreement_max=float(np.max(result.diagnostics.disagreement)),
        entropy_mean=float(np.mean(result.diagnostics.entropy)),
        entropy_max=float(np.max(result.diagnostics.entropy)),
        # Perspective usage
        weight_I_mean=float(np.mean(weights[:, 0])),
        weight_F_mean=float(np.mean(weights[:, 1])),
        weight_B_mean=float(np.mean(weights[:, 2])),
        phi_B_disabled=result.metadata.get("phi_B_disabled", False),
        pct_masked_candidates=pct_masked,
        # Episode info
        num_frames=len(phi_star),
        goal_is_hindsight=result.metadata.get("goal_is_hindsight", False),
        # Orchestrator settings
        fusion_override=fusion_override.to_dict() if fusion_override else None,
    )


@dataclass
class ProcessRewardCorrelationReport:
    """Report for correlating process reward metrics with MPL/success.

    Used for data valuation / gating analysis.
    """
    # Correlation coefficients
    conf_vs_success: float = 0.0
    conf_vs_mpl_uplift: float = 0.0
    disagreement_vs_success: float = 0.0
    entropy_vs_success: float = 0.0
    phi_delta_vs_success: float = 0.0
    r_shape_sum_vs_mpl_uplift: float = 0.0

    # Segmentation analysis
    low_conf_success_rate: float = 0.0  # Success rate when conf < threshold
    high_conf_success_rate: float = 0.0  # Success rate when conf > threshold
    conf_threshold: float = 0.5

    # Quality gating stats
    pct_episodes_gated: float = 0.0  # % with conf below threshold
    gated_avg_mpl: float = 0.0
    ungated_avg_mpl: float = 0.0

    # Sample sizes
    num_episodes: int = 0
    num_gated: int = 0
    num_ungated: int = 0


def compute_correlation_report(
    log_entries: List[ProcessRewardLogEntry],
    success_labels: List[bool],
    mpl_uplifts: Optional[List[float]] = None,
    conf_threshold: float = 0.5,
) -> ProcessRewardCorrelationReport:
    """Compute correlation report from process reward logs and outcome labels.

    Args:
        log_entries: List of ProcessRewardLogEntry from multiple episodes.
        success_labels: Binary success label for each episode.
        mpl_uplifts: Optional MPL improvement for each episode.
        conf_threshold: Threshold for low/high confidence segmentation.

    Returns:
        ProcessRewardCorrelationReport with correlation analysis.
    """
    n = len(log_entries)
    if n == 0:
        return ProcessRewardCorrelationReport()

    # Extract arrays
    conf_means = np.array([e.conf_mean for e in log_entries])
    disagreements = np.array([e.disagreement_mean for e in log_entries])
    entropies = np.array([e.entropy_mean for e in log_entries])
    phi_deltas = np.array([e.phi_star_delta for e in log_entries])
    r_shape_sums = np.array([e.r_shape_sum for e in log_entries])
    successes = np.array(success_labels, dtype=np.float32)

    # Compute correlations
    def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    report = ProcessRewardCorrelationReport(
        conf_vs_success=safe_corr(conf_means, successes),
        disagreement_vs_success=safe_corr(disagreements, successes),
        entropy_vs_success=safe_corr(entropies, successes),
        phi_delta_vs_success=safe_corr(phi_deltas, successes),
        conf_threshold=conf_threshold,
        num_episodes=n,
    )

    # MPL correlations if available
    if mpl_uplifts is not None:
        mpl = np.array(mpl_uplifts)
        report.conf_vs_mpl_uplift = safe_corr(conf_means, mpl)
        report.r_shape_sum_vs_mpl_uplift = safe_corr(r_shape_sums, mpl)

    # Segmentation by confidence
    low_conf_mask = conf_means < conf_threshold
    high_conf_mask = conf_means >= conf_threshold

    num_low = int(np.sum(low_conf_mask))
    num_high = int(np.sum(high_conf_mask))

    if num_low > 0:
        report.low_conf_success_rate = float(np.mean(successes[low_conf_mask]))
    if num_high > 0:
        report.high_conf_success_rate = float(np.mean(successes[high_conf_mask]))

    report.num_gated = num_low
    report.num_ungated = num_high
    report.pct_episodes_gated = num_low / n if n > 0 else 0.0

    # MPL by confidence segment
    if mpl_uplifts is not None:
        mpl = np.array(mpl_uplifts)
        if num_low > 0:
            report.gated_avg_mpl = float(np.mean(mpl[low_conf_mask]))
        if num_high > 0:
            report.ungated_avg_mpl = float(np.mean(mpl[high_conf_mask]))

    return report


@dataclass
class OrchestratorPolicy:
    """Adaptive policy for orchestrator-controlled fusion.

    Defines rules for adjusting FusionOverride based on episode characteristics.
    All adjustments are applied (not advisory-only) to the returned FusionOverride.

    Policy rules (in priority order):
    1. Very low plausibility → effectively disable shaping (conf_cap ~0)
    2. High occlusion OR IR diverged → smooth fusion (higher temperature)
    3. Low plausibility → cap confidence moderately
    4. Sustained low confidence → trigger data quality flag
    """
    # Occlusion-based adjustments
    high_occlusion_threshold: float = 0.5
    high_occlusion_temperature_boost: float = 0.5  # Add to temperature
    high_occlusion_risk_tolerance_boost: float = 0.1

    # IR divergence adjustments
    ir_diverged_threshold: float = 0.1
    ir_diverged_temperature_boost: float = 0.3

    # MHN plausibility gating (tiered)
    low_plausibility_threshold: float = 0.5
    low_plausibility_conf_cap: float = 0.3  # Cap confidence when plausibility is low
    very_low_plausibility_threshold: float = 0.2
    very_low_plausibility_conf_cap: float = 0.05  # Effectively disable shaping

    # Sustained low confidence detection
    low_conf_threshold: float = 0.3
    low_conf_trigger_enabled: bool = True

    # Combined degradation handling
    combined_degradation_threshold: int = 2  # Number of issues to trigger extra smoothing
    combined_degradation_temp_boost: float = 0.3

    def apply(
        self,
        base_override: FusionOverride,
        mean_occlusion: float = 0.0,
        ir_loss_mean: float = 0.0,
        mhn_plausibility: float = 1.0,
        mean_confidence: Optional[float] = None,
    ) -> Tuple[FusionOverride, Dict[str, Any]]:
        """Apply policy to adjust fusion override based on episode characteristics.

        All adjustments are APPLIED to the returned FusionOverride - not advisory.
        The returned override should be passed to process_reward_episode().

        Args:
            base_override: Base FusionOverride to adjust.
            mean_occlusion: Mean occlusion rate across episode.
            ir_loss_mean: Mean IR loss across episode.
            mhn_plausibility: MHN plausibility score.
            mean_confidence: Optional mean confidence from prior episodes.

        Returns:
            (adjusted_override, adjustments_made) tuple.
            adjustments_made may include "triggers" for actionable alerts.

        Example:
            >>> policy = OrchestratorPolicy()
            >>> base = FusionOverride()
            >>> adjusted, adjustments = policy.apply(
            ...     base, mean_occlusion=0.6, ir_loss_mean=0.05, mhn_plausibility=0.4
            ... )
            >>> result = process_reward_episode(
            ...     scene_tracks, instruction, orchestrator_overrides=adjusted
            ... )
            >>> if "triggers" in adjustments:
            ...     # Handle data quality issues
            ...     for trigger in adjustments["triggers"]:
            ...         log_data_quality_issue(trigger)
        """
        adjustments = {}
        triggers = []
        new_temp = base_override.temperature
        new_risk = base_override.risk_tolerance
        new_conf_cap = base_override.confidence_cap
        issue_count = 0

        # Rule 1: Very low MHN plausibility → effectively disable shaping
        if mhn_plausibility < self.very_low_plausibility_threshold:
            new_conf_cap = min(new_conf_cap, self.very_low_plausibility_conf_cap)
            adjustments["very_low_plausibility"] = {
                "plausibility": mhn_plausibility,
                "conf_cap_applied": new_conf_cap,
                "action": "shaping_disabled",
            }
            triggers.append({
                "type": "shaping_disabled",
                "reason": "very_low_mhn_plausibility",
                "plausibility": mhn_plausibility,
                "recommendation": "Review MHN model or scene tracks quality",
            })
            issue_count += 1

        # Rule 2a: High occlusion → smooth fusion (higher temperature = softer weights)
        elif mean_occlusion > self.high_occlusion_threshold:
            new_temp += self.high_occlusion_temperature_boost
            new_risk += self.high_occlusion_risk_tolerance_boost
            adjustments["high_occlusion"] = {
                "occlusion": mean_occlusion,
                "temp_boost": self.high_occlusion_temperature_boost,
                "risk_boost": self.high_occlusion_risk_tolerance_boost,
            }
            issue_count += 1

        # Rule 2b: IR diverged → smooth fusion
        if ir_loss_mean > self.ir_diverged_threshold:
            new_temp += self.ir_diverged_temperature_boost
            adjustments["ir_diverged"] = {
                "ir_loss": ir_loss_mean,
                "temp_boost": self.ir_diverged_temperature_boost,
            }
            issue_count += 1

        # Rule 3: Low MHN plausibility (but not very low) → cap confidence
        if (self.very_low_plausibility_threshold <= mhn_plausibility < self.low_plausibility_threshold):
            new_conf_cap = min(new_conf_cap, self.low_plausibility_conf_cap)
            adjustments["low_plausibility"] = {
                "plausibility": mhn_plausibility,
                "conf_cap_applied": new_conf_cap,
            }
            issue_count += 1

        # Rule 4: Sustained low confidence → trigger data quality flag
        if self.low_conf_trigger_enabled and mean_confidence is not None:
            if mean_confidence < self.low_conf_threshold:
                triggers.append({
                    "type": "sustained_low_confidence",
                    "mean_confidence": mean_confidence,
                    "threshold": self.low_conf_threshold,
                    "recommendation": "Request higher-fidelity data or re-render scene",
                })
                adjustments["low_confidence"] = {
                    "mean_confidence": mean_confidence,
                    "action": "data_quality_flag",
                }

        # Rule 5: Combined degradation → extra smoothing
        if issue_count >= self.combined_degradation_threshold:
            new_temp += self.combined_degradation_temp_boost
            adjustments["combined_degradation"] = {
                "issue_count": issue_count,
                "extra_temp_boost": self.combined_degradation_temp_boost,
            }

        # Add triggers if any
        if triggers:
            adjustments["triggers"] = triggers

        adjusted = FusionOverride(
            temperature=new_temp,
            candidate_mask=base_override.candidate_mask,
            risk_tolerance=min(1.0, new_risk),
            entropy_penalty=base_override.entropy_penalty,
            weight_smoothing=base_override.weight_smoothing,
            min_weight_floor=base_override.min_weight_floor,
            confidence_cap=new_conf_cap,
        )

        return adjusted, adjustments

    def check_triggers(self, adjustments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actionable triggers from adjustments.

        Args:
            adjustments: The adjustments dict from apply().

        Returns:
            List of trigger dicts that require action (logging, alerts, etc.)
        """
        return adjustments.get("triggers", [])


def format_log_for_training(
    result: ProcessRewardEpisodeOutput,
    scene_ir_quality: Optional[float] = None,
    motion_quality: Optional[float] = None,
    mhn_summary: Optional[MHNSummary] = None,
    fusion_override: Optional[FusionOverride] = None,
) -> Dict[str, Any]:
    """Format process reward output for training log integration.

    This produces a flat dictionary suitable for CSV/JSON logging
    alongside existing MHN/SceneIR summaries.

    Args:
        result: ProcessRewardEpisodeOutput
        scene_ir_quality: Optional scene IR quality score
        motion_quality: Optional motion quality score
        mhn_summary: Optional MHN summary
        fusion_override: Optional FusionOverride used

    Returns:
        Flat dictionary for logging.
    """
    entry = extract_log_entry(result, fusion_override=fusion_override)
    log = entry.to_dict()

    # Add upstream quality scores
    if scene_ir_quality is not None:
        log["scene_ir_quality"] = scene_ir_quality
    if motion_quality is not None:
        log["motion_quality"] = motion_quality

    # Add MHN summary
    if mhn_summary is not None:
        log["mhn_plausibility"] = mhn_summary.plausibility_score
        log["mhn_difficulty"] = mhn_summary.structural_difficulty

    return log
