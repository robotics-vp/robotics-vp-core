"""
Process Reward Schemas.

Dataclass definitions for configuration, inputs, and outputs of the process reward module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class FusionOverride:
    """Orchestrator-controllable fusion parameters.

    These can be passed at runtime to control fusion behavior without retraining.

    Attributes:
        temperature: Softmax temperature for fusion weights. Lower = sharper weighting.
        candidate_mask: Boolean mask [I, F, B] to enable/disable perspectives.
            True = enabled, False = disabled. Disabled candidates get weight ~0.
        risk_tolerance: Minimum confidence threshold. Below this, r_shape is scaled down.
        entropy_penalty: Penalty for high-entropy (uncertain) fusion weights.
        weight_smoothing: Temporal smoothing factor for fusion weights [0, 1].
            0 = no smoothing, 1 = full smoothing (frozen weights).
        min_weight_floor: Minimum weight for any enabled candidate (prevents collapse).
        confidence_cap: Maximum confidence value (1.0 = no cap). Set lower to discount
            unreliable episodes (e.g., low MHN plausibility).
    """
    temperature: float = 1.0
    candidate_mask: Tuple[bool, bool, bool] = (True, True, True)  # [Phi_I, Phi_F, Phi_B]
    risk_tolerance: float = 0.3
    entropy_penalty: float = 0.1
    weight_smoothing: float = 0.0
    min_weight_floor: float = 0.01
    confidence_cap: float = 1.0  # Max confidence value (1.0 = no cap)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "temperature": self.temperature,
            "candidate_mask": list(self.candidate_mask),
            "risk_tolerance": self.risk_tolerance,
            "entropy_penalty": self.entropy_penalty,
            "weight_smoothing": self.weight_smoothing,
            "min_weight_floor": self.min_weight_floor,
            "confidence_cap": self.confidence_cap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionOverride":
        """Deserialize from dictionary."""
        mask = data.get("candidate_mask", [True, True, True])
        return cls(
            temperature=data.get("temperature", 1.0),
            candidate_mask=tuple(mask) if isinstance(mask, list) else mask,
            risk_tolerance=data.get("risk_tolerance", 0.3),
            entropy_penalty=data.get("entropy_penalty", 0.1),
            weight_smoothing=data.get("weight_smoothing", 0.0),
            min_weight_floor=data.get("min_weight_floor", 0.01),
            confidence_cap=data.get("confidence_cap", 1.0),
        )


@dataclass
class ProcessRewardConfig:
    """Configuration for process reward computation.

    Attributes:
        gamma: Discount factor for PBRS. Must be in (0, 1].
        use_confidence_gating: If True, scale r_shape by confidence.
        online_mode: If True, forbid any hindsight constructs (goal from last frame,
            future window stats, Φ_B without explicit goal). Fails loudly if violated.
        hop_model_path: Path to pretrained HopNet checkpoint (optional).
        fusion_hidden_dim: Hidden dimension for FusionNet.
        fusion_num_layers: Number of layers in FusionNet.
        feature_dim: Dimension of extracted features per track.
        use_latents: If True, use z_shape/z_tex if available.
        use_mhn_features: If True, use MHN summary features if available.
        phi_clip_min: Minimum value for Phi (for stability).
        phi_clip_max: Maximum value for Phi (for stability).
        hop_uncertainty_method: Method for hop uncertainty estimation.
        instruction_embedding_dim: Dimension of instruction embeddings.
        device: Device for PyTorch models ("cpu", "cuda", etc.).
    """
    gamma: float = 0.99
    use_confidence_gating: bool = True
    online_mode: bool = True  # Default True for RL training safety; forbid hindsight constructs
    hop_model_path: Optional[str] = None
    fusion_hidden_dim: int = 64
    fusion_num_layers: int = 2
    feature_dim: int = 32
    use_latents: bool = True
    use_mhn_features: bool = True
    phi_clip_min: float = 0.0
    phi_clip_max: float = 1.0
    hop_uncertainty_method: Literal["ensemble", "dropout", "single"] = "single"
    instruction_embedding_dim: int = 64
    device: str = "cpu"

    # Feature extraction settings
    pool_method: Literal["mean", "max", "attention"] = "mean"
    include_velocity_features: bool = True
    include_visibility_features: bool = True
    include_ir_features: bool = True

    # Default fusion override (can be overridden at runtime)
    default_fusion_override: FusionOverride = field(default_factory=FusionOverride)


# -----------------------------------------------------------------------------
# MHN Summary (from src.vision.motion_hierarchy.metrics)
# -----------------------------------------------------------------------------

@dataclass
class MHNSummary:
    """Summary of Motion Hierarchy Network output.

    This mirrors MotionHierarchySummary from src.vision.motion_hierarchy.metrics
    but is defined here for API clarity.

    Attributes:
        mean_tree_depth: Average depth of hierarchy tree.
        mean_branch_factor: Average branching factor.
        residual_mean: Mean reconstruction residual.
        residual_std: Std of reconstruction residual.
        structural_difficulty: Difficulty score based on structure.
        plausibility_score: Plausibility in [0, 1] (higher = more plausible).
        hierarchy: Optional (N, N) parent adjacency matrix.
    """
    mean_tree_depth: float = 0.0
    mean_branch_factor: float = 0.0
    residual_mean: float = 0.0
    residual_std: float = 0.0
    structural_difficulty: float = 0.0
    plausibility_score: float = 1.0
    hierarchy: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "mean_tree_depth": self.mean_tree_depth,
            "mean_branch_factor": self.mean_branch_factor,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "structural_difficulty": self.structural_difficulty,
            "plausibility_score": self.plausibility_score,
        }
        if self.hierarchy is not None:
            result["hierarchy"] = self.hierarchy.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MHNSummary":
        """Deserialize from dictionary."""
        hierarchy = data.get("hierarchy")
        if hierarchy is not None and not isinstance(hierarchy, np.ndarray):
            hierarchy = np.array(hierarchy, dtype=np.float32)
        return cls(
            mean_tree_depth=data.get("mean_tree_depth", 0.0),
            mean_branch_factor=data.get("mean_branch_factor", 0.0),
            residual_mean=data.get("residual_mean", 0.0),
            residual_std=data.get("residual_std", 0.0),
            structural_difficulty=data.get("structural_difficulty", 0.0),
            plausibility_score=data.get("plausibility_score", 1.0),
            hierarchy=hierarchy,
        )


# -----------------------------------------------------------------------------
# Output Types
# -----------------------------------------------------------------------------

@dataclass
class ProgressPerspectives:
    """Three progress potential candidates from different perspectives.

    Attributes:
        phi_I: Incremental accumulation. phi_I[t] = clipped cumsum of hop predictions.
        phi_F: Forward anchored. phi_F[t] = progress from init to t.
        phi_B: Backward anchored. phi_B[t] = 1 - distance from t to goal.
        conf_I: Confidence for Phi_I at each timestep.
        conf_F: Confidence for Phi_F at each timestep.
        conf_B: Confidence for Phi_B at each timestep.
    """
    phi_I: np.ndarray  # (T,)
    phi_F: np.ndarray  # (T,)
    phi_B: np.ndarray  # (T,)
    conf_I: np.ndarray  # (T,)
    conf_F: np.ndarray  # (T,)
    conf_B: np.ndarray  # (T,)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "phi_I": self.phi_I.tolist(),
            "phi_F": self.phi_F.tolist(),
            "phi_B": self.phi_B.tolist(),
            "conf_I": self.conf_I.tolist(),
            "conf_F": self.conf_F.tolist(),
            "conf_B": self.conf_B.tolist(),
        }


@dataclass
class FusionDiagnostics:
    """Diagnostics from the fusion process.

    Attributes:
        weights: Fusion weights [w_I, w_F, w_B] at each timestep. Shape (T, 3).
        entropy: Entropy of fusion weights at each timestep. Shape (T,).
        disagreement: Max absolute difference between perspectives. Shape (T,).
        gating_factor: Confidence-based gating factor applied to r_shape. Shape (T,).
        raw_fusion_input: Raw input features to FusionNet (for debugging).
    """
    weights: np.ndarray  # (T, 3)
    entropy: np.ndarray  # (T,)
    disagreement: np.ndarray  # (T,)
    gating_factor: np.ndarray  # (T,)
    raw_fusion_input: Optional[np.ndarray] = None


@dataclass
class ProcessRewardStepOutput:
    """Output from process_reward_step for a single timestep.

    Attributes:
        phi_t: Fused potential at time t.
        phi_t1: Fused potential at time t+1 (for PBRS).
        conf_t: Confidence at time t.
        conf_t1: Confidence at time t+1.
        r_shape: Shaped reward = gamma * phi_t1 - phi_t.
        perspectives: Individual perspective values at t.
        weights: Fusion weights [w_I, w_F, w_B] at t.
        debug: Additional debug info.
    """
    phi_t: float
    phi_t1: float
    conf_t: float
    conf_t1: float
    r_shape: float
    perspectives: Tuple[float, float, float]  # (phi_I, phi_F, phi_B) at t
    weights: Tuple[float, float, float]  # (w_I, w_F, w_B)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "phi_t": self.phi_t,
            "phi_t1": self.phi_t1,
            "conf_t": self.conf_t,
            "conf_t1": self.conf_t1,
            "r_shape": self.r_shape,
            "perspectives": list(self.perspectives),
            "weights": list(self.weights),
            "debug": self.debug,
        }


@dataclass
class ProcessRewardEpisodeOutput:
    """Output from process_reward_episode for a full episode.

    Attributes:
        phi_star: Fused potential at each timestep. Shape (T,).
        conf: Confidence at each timestep. Shape (T,).
        r_shape: Shaped reward at each timestep. Shape (T-1,).
            r_shape[t] = gamma * phi_star[t+1] - phi_star[t]
        perspectives: Three perspective potentials.
        diagnostics: Fusion diagnostics.
        episode_id: Optional episode identifier.
        metadata: Additional metadata.
    """
    phi_star: np.ndarray  # (T,)
    conf: np.ndarray  # (T,)
    r_shape: np.ndarray  # (T-1,) PBRS: gamma * phi[t+1] - phi[t]
    perspectives: ProgressPerspectives
    diagnostics: FusionDiagnostics
    episode_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "phi_star": self.phi_star.tolist(),
            "conf": self.conf.tolist(),
            "r_shape": self.r_shape.tolist(),
            "perspectives": self.perspectives.to_dict(),
            "diagnostics": {
                "weights": self.diagnostics.weights.tolist(),
                "entropy": self.diagnostics.entropy.tolist(),
                "disagreement": self.diagnostics.disagreement.tolist(),
                "gating_factor": self.diagnostics.gating_factor.tolist(),
            },
            "episode_id": self.episode_id,
            "metadata": self.metadata,
        }

    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "phi_star_mean": float(np.mean(self.phi_star)),
            "phi_star_final": float(self.phi_star[-1]) if len(self.phi_star) > 0 else 0.0,
            "conf_mean": float(np.mean(self.conf)),
            "conf_min": float(np.min(self.conf)),
            "r_shape_sum": float(np.sum(self.r_shape)),
            "r_shape_mean": float(np.mean(self.r_shape)) if len(self.r_shape) > 0 else 0.0,
            "disagreement_mean": float(np.mean(self.diagnostics.disagreement)),
            "entropy_mean": float(np.mean(self.diagnostics.entropy)),
            "num_frames": len(self.phi_star),
        }


# -----------------------------------------------------------------------------
# Feature Types
# -----------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    """Extracted features for a single frame.

    Attributes:
        pooled: Pooled features across all tracks. Shape (feature_dim,).
        per_track: Per-track features. Shape (K, feature_dim).
        visibility_stats: Visibility statistics (num_visible, pct_occluded, etc.).
        ir_stats: IR loss statistics (mean, std, etc.).
        mhn_features: Optional MHN-derived features.
    """
    pooled: np.ndarray  # (feature_dim,)
    per_track: np.ndarray  # (K, feature_dim)
    visibility_stats: Dict[str, float] = field(default_factory=dict)
    ir_stats: Dict[str, float] = field(default_factory=dict)
    mhn_features: Optional[np.ndarray] = None  # (mhn_feature_dim,)


@dataclass
class TransitionFeatures:
    """Features describing the transition from frame t to t+1.

    Attributes:
        delta_pos: Position changes per track. Shape (K,).
        delta_rot: Rotation changes per track (angle). Shape (K,).
        delta_scale: Scale changes per track. Shape (K,).
        visibility_transitions: Visibility change events.
        ir_delta: IR loss changes. Shape (K,).
        latent_sim_delta: Latent cosine similarity changes (if available). Shape (K,).
        pooled: Pooled transition features. Shape (trans_feature_dim,).
    """
    delta_pos: np.ndarray  # (K,)
    delta_rot: np.ndarray  # (K,)
    delta_scale: np.ndarray  # (K,)
    visibility_transitions: Dict[str, int] = field(default_factory=dict)
    ir_delta: np.ndarray = field(default_factory=lambda: np.array([]))  # (K,)
    latent_sim_delta: Optional[np.ndarray] = None  # (K,)
    pooled: np.ndarray = field(default_factory=lambda: np.array([]))  # (trans_feature_dim,)


@dataclass
class EpisodeFeatures:
    """Aggregated features for an entire episode.

    Attributes:
        frame_features: List of per-frame features. Length T.
        transition_features: List of transition features. Length T-1.
        init_features: Features at the initial frame.
        goal_features: Features at the goal frame (if known).
        global_stats: Global episode statistics.
    """
    frame_features: List[FrameFeatures]
    transition_features: List[TransitionFeatures]
    init_features: FrameFeatures
    goal_features: Optional[FrameFeatures] = None
    global_stats: Dict[str, float] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frame_features)


# -----------------------------------------------------------------------------
# Hop Labels
# -----------------------------------------------------------------------------

@dataclass
class HopLabel:
    """Label for a hop prediction (before -> after transition).

    Attributes:
        hop_value: Ground truth hop in [-1, 1]. Positive = progress, negative = regress.
        source: Label source ("oracle", "human", "llm", "task_success", "proxy").
        confidence: Confidence in the label [0, 1].
        before_idx: Frame index of "before" state.
        after_idx: Frame index of "after" state.
        metadata: Additional metadata about the label.
    """
    hop_value: float
    source: Literal["oracle", "human", "llm", "task_success", "proxy"]
    confidence: float = 1.0
    before_idx: int = 0
    after_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
