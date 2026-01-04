"""
Core Process Reward API.

Primary entry points for process reward computation:
- process_reward_episode: Full episode processing
- process_reward_step: Per-step processing for RL rollouts
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.process_reward.schemas import (
    ProcessRewardConfig,
    FusionOverride,
    MHNSummary,
    ProcessRewardStepOutput,
    ProcessRewardEpisodeOutput,
    EpisodeFeatures,
)
from src.process_reward.features import (
    FeatureExtractor,
    extract_features_from_scene_tracks_lite,
)
from src.process_reward.hop_model import (
    create_hop_predictor,
    HeuristicHopPredictor,
)
from src.process_reward.progress_perspectives import compute_all_perspectives
from src.process_reward.fusion import (
    create_fusion,
    build_context_features,
    HeuristicFusion,
)
from src.process_reward.shaping import create_pbrs_wrapper


def process_reward_episode(
    scene_tracks: Any,  # SceneTracksLite or compatible
    instruction: Union[str, np.ndarray],
    goal_frame_idx: Optional[int] = None,
    cfg: Optional[ProcessRewardConfig] = None,
    mhn: Optional[MHNSummary] = None,
    orchestrator_overrides: Optional[FusionOverride] = None,
    episode_id: Optional[str] = None,
) -> ProcessRewardEpisodeOutput:
    """Process reward for an entire episode.

    Primary entry point for full episode processing.

    Args:
        scene_tracks: SceneTracksLite from deserialized scene_tracks_v1.
            Must have: poses_R, poses_t, scales, visibility, occlusion,
            ir_loss, converged, entity_types. Optionally: z_shape, z_tex.
        instruction: Task instruction as string or embedding array.
        goal_frame_idx: Optional goal frame index. If None, uses last frame.
        cfg: Process reward configuration. Uses defaults if None.
        mhn: Optional MHN summary with motion hierarchy features.
        orchestrator_overrides: Optional fusion parameters from orchestrator.
        episode_id: Optional episode identifier for tracking.

    Returns:
        ProcessRewardEpisodeOutput with phi_star, conf, r_shape, perspectives,
        diagnostics, and metadata.

    Example:
        >>> from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
        >>> data = dict(np.load("episode.npz", allow_pickle=False))
        >>> scene_tracks = deserialize_scene_tracks_v1(data)
        >>> cfg = ProcessRewardConfig()
        >>> result = process_reward_episode(scene_tracks, "pick up the box", cfg=cfg)
        >>> print(f"Shaped reward sum: {result.r_shape.sum():.3f}")
    """
    cfg = cfg or ProcessRewardConfig()
    override = orchestrator_overrides or cfg.default_fusion_override

    # CRITICAL: If goal_frame_idx is not explicitly provided, we're using hindsight
    # which leaks future information.
    goal_is_hindsight = goal_frame_idx is None

    # Online mode: fail loudly if any hindsight construct is used
    if cfg.online_mode and goal_is_hindsight:
        raise ValueError(
            "online_mode=True but goal_frame_idx=None. "
            "In online mode, you must provide an explicit goal_frame_idx "
            "(e.g., from env spec) to avoid future leakage. "
            "If you're doing offline evaluation, set online_mode=False."
        )

    if goal_is_hindsight:
        # Auto-mask Φ_B to prevent future leakage
        # User can override by explicitly providing goal_frame_idx
        if override is None:
            override = FusionOverride(candidate_mask=(True, True, False))
        elif override.candidate_mask[2]:  # Φ_B was enabled
            import warnings
            warnings.warn(
                "goal_frame_idx=None uses hindsight (last frame as goal). "
                "Φ_B is disabled to prevent future leakage. "
                "Provide explicit goal_frame_idx to enable Φ_B."
            )
            override = FusionOverride(
                temperature=override.temperature,
                candidate_mask=(override.candidate_mask[0], override.candidate_mask[1], False),
                risk_tolerance=override.risk_tolerance,
                entropy_penalty=override.entropy_penalty,
                weight_smoothing=override.weight_smoothing,
                min_weight_floor=override.min_weight_floor,
                confidence_cap=override.confidence_cap,
            )

    # Get instruction embedding
    if isinstance(instruction, str):
        instruction_embedding = _embed_instruction(instruction, cfg.instruction_embedding_dim)
    else:
        instruction_embedding = instruction

    # Extract features
    episode_features = extract_features_from_scene_tracks_lite(
        scene_tracks,
        cfg,
        mhn_summary=mhn,
        goal_frame_idx=goal_frame_idx,
    )

    # Predict hops
    hop_predictor = create_hop_predictor(cfg)
    if isinstance(hop_predictor, HeuristicHopPredictor):
        hops, uncertainties = hop_predictor.predict_episode_hops(
            episode_features,
            goal_frame_idx=goal_frame_idx,
        )
    else:
        hops, uncertainties = hop_predictor.predict_episode_hops(
            episode_features,
            instruction_embedding,
            goal_frame_idx=goal_frame_idx,
        )

    # Compute perspectives
    perspectives = compute_all_perspectives(
        episode_features,
        hops,
        uncertainties,
        goal_frame_idx=goal_frame_idx,
        config=cfg,
    )

    # Build context features
    context_features = build_context_features(episode_features, mhn)

    # Fuse perspectives
    fusion = create_fusion(cfg)
    phi_star, conf, diagnostics = fusion.fuse_perspectives(
        perspectives,
        context_features,
        mhn_summary=mhn,
        override=override,
    )

    # Compute PBRS
    pbrs = create_pbrs_wrapper(cfg)
    r_shape, shaping_stats = pbrs.compute(phi_star, conf, diagnostics)

    # Build output
    metadata = {
        "instruction": instruction if isinstance(instruction, str) else "embedding_provided",
        "goal_frame_idx": goal_frame_idx,
        "goal_is_hindsight": goal_is_hindsight,
        "phi_B_disabled": goal_is_hindsight,  # Flag for diagnostics
        "num_frames": len(phi_star),
        "num_tracks": episode_features.global_stats.get("num_tracks", 0),
        "shaping_stats": shaping_stats,
        "global_stats": episode_features.global_stats,
    }

    if mhn is not None:
        metadata["mhn_summary"] = mhn.to_dict()

    return ProcessRewardEpisodeOutput(
        phi_star=phi_star,
        conf=conf,
        r_shape=r_shape,
        perspectives=perspectives,
        diagnostics=diagnostics,
        episode_id=episode_id,
        metadata=metadata,
    )


def process_reward_step(
    scene_tracks_t: Any,
    t: int,
    goal_frame_idx: Optional[int],
    cfg: Optional[ProcessRewardConfig] = None,
    orchestrator_overrides: Optional[FusionOverride] = None,
    mhn_features_t: Optional[np.ndarray] = None,
    instruction_embedding: Optional[np.ndarray] = None,
    # State from previous call
    prev_phi: Optional[float] = None,
    prev_conf: Optional[float] = None,
    init_features: Optional[np.ndarray] = None,
    goal_features: Optional[np.ndarray] = None,
    cumulative_hop: float = 0.0,
) -> ProcessRewardStepOutput:
    """Process reward for a single step (for online RL rollouts).

    This is a stateful API designed for use during RL training where
    we receive one frame at a time.

    Args:
        scene_tracks_t: Frame slice of scene tracks at time t.
            Can be a dict with poses_R[t], poses_t[t], etc.
        t: Current timestep index.
        goal_frame_idx: Goal frame index (None if unknown).
        cfg: Process reward configuration.
        orchestrator_overrides: Optional fusion parameters.
        mhn_features_t: Optional MHN features at time t.
        instruction_embedding: Instruction embedding.
        prev_phi: Previous fused potential (from t-1).
        prev_conf: Previous confidence (from t-1).
        init_features: Features at initial frame (for anchoring).
        goal_features: Features at goal frame (for backward perspective).
        cumulative_hop: Cumulative hop sum from t=0 to t-1.

    Returns:
        ProcessRewardStepOutput with phi_t, phi_t1, r_shape, etc.
    """
    cfg = cfg or ProcessRewardConfig()
    override = orchestrator_overrides or cfg.default_fusion_override

    # Extract current frame features
    extractor = FeatureExtractor(cfg)

    # Handle different input formats
    if hasattr(scene_tracks_t, "poses_R"):
        # SceneTracksLite-like object
        poses_R = scene_tracks_t.poses_R
        poses_t_arr = scene_tracks_t.poses_t
        scales = scene_tracks_t.scales
        visibility = scene_tracks_t.visibility
        occlusion = scene_tracks_t.occlusion
        ir_loss = scene_tracks_t.ir_loss
        converged = scene_tracks_t.converged
        entity_types = scene_tracks_t.entity_types
        z_shape = getattr(scene_tracks_t, "z_shape", None)
        z_tex = getattr(scene_tracks_t, "z_tex", None)
    elif isinstance(scene_tracks_t, dict):
        poses_R = scene_tracks_t["poses_R"]
        poses_t_arr = scene_tracks_t["poses_t"]
        scales = scene_tracks_t["scales"]
        visibility = scene_tracks_t["visibility"]
        occlusion = scene_tracks_t["occlusion"]
        ir_loss = scene_tracks_t["ir_loss"]
        converged = scene_tracks_t["converged"]
        entity_types = scene_tracks_t["entity_types"]
        z_shape = scene_tracks_t.get("z_shape")
        z_tex = scene_tracks_t.get("z_tex")
    else:
        raise ValueError(f"Unsupported scene_tracks_t type: {type(scene_tracks_t)}")

    # Extract single frame features
    frame_features = extractor._extract_frame_features(
        poses_R, poses_t_arr, scales, visibility, occlusion, ir_loss, converged,
        entity_types, z_shape, z_tex, None,
    )

    current_features = frame_features.pooled

    # Initialize if first step
    if init_features is None:
        init_features = current_features
    if goal_features is None:
        goal_features = current_features  # Will be updated when goal is known

    # Compute hop prediction (if we have previous features)
    hop_predictor = create_hop_predictor(cfg)
    if prev_phi is not None:
        # We need the previous frame's features for hop
        # For now, use a simple distance-based estimate
        if isinstance(hop_predictor, HeuristicHopPredictor):
            # Estimate hop from feature distance
            hop, hop_unc = hop_predictor.predict_hop(
                before_features=init_features,  # Approximate
                after_features=current_features,
                init_features=init_features,
                goal_features=goal_features,
            )
        else:
            hop, hop_unc = 0.0, 0.5  # Default for learned model
    else:
        hop, hop_unc = 0.0, 0.0

    # Update cumulative hop
    new_cumulative_hop = cumulative_hop + hop

    # Compute perspectives at current timestep
    # Incremental: based on cumulative hop
    phi_I = np.clip(new_cumulative_hop, cfg.phi_clip_min, cfg.phi_clip_max)

    # Forward: distance from init
    dist_init = np.linalg.norm(current_features - init_features)
    dist_init_goal = np.linalg.norm(goal_features - init_features)
    phi_F = dist_init / (dist_init_goal + 1e-6) if dist_init_goal > 0 else 0.0
    phi_F = np.clip(phi_F, cfg.phi_clip_min, cfg.phi_clip_max)

    # Backward: distance to goal
    dist_goal = np.linalg.norm(current_features - goal_features)
    phi_B = 1.0 - dist_goal / (dist_init_goal + 1e-6) if dist_init_goal > 0 else 1.0
    phi_B = np.clip(phi_B, cfg.phi_clip_min, cfg.phi_clip_max)

    # Simple confidence estimate
    vis_conf = float(np.mean(visibility))
    ir_conf = float(np.exp(-np.mean(ir_loss)))
    conf_I = conf_F = conf_B = 0.5 * vis_conf + 0.5 * ir_conf

    # Fuse perspectives (simplified for single-step)
    # Use confidence-weighted average (heuristic for online use)
    mask = np.array(override.candidate_mask, dtype=np.float32)
    phis = np.array([phi_I, phi_F, phi_B])
    confs = np.array([conf_I, conf_F, conf_B]) * mask

    weights = confs / (confs.sum() + 1e-8)
    weights = np.maximum(weights, override.min_weight_floor * mask)
    weights = weights / (weights.sum() + 1e-8)

    phi_t = float(np.sum(weights * phis))
    conf_t = float(np.sum(weights * confs))

    # Compute r_shape if we have previous phi
    if prev_phi is not None:
        r_shape = cfg.gamma * phi_t - prev_phi
        if cfg.use_confidence_gating and prev_conf is not None:
            gate = 0.5 * (prev_conf + conf_t)
            r_shape = r_shape * gate
    else:
        r_shape = 0.0

    return ProcessRewardStepOutput(
        phi_t=phi_t,
        phi_t1=phi_t,  # Current becomes "t+1" for next iteration
        conf_t=conf_t,
        conf_t1=conf_t,
        r_shape=float(r_shape),
        perspectives=(float(phi_I), float(phi_F), float(phi_B)),
        weights=(float(weights[0]), float(weights[1]), float(weights[2])),
        debug={
            "hop": hop,
            "hop_uncertainty": hop_unc,
            "cumulative_hop": new_cumulative_hop,
            "dist_init": float(dist_init),
            "dist_goal": float(dist_goal),
        },
    )


def _embed_instruction(instruction: str, dim: int) -> np.ndarray:
    """Create a simple instruction embedding.

    This is a placeholder. In production, use a proper text encoder.

    Args:
        instruction: Instruction string.
        dim: Embedding dimension.

    Returns:
        (dim,) embedding array.
    """
    # Simple hash-based embedding
    import hashlib

    # Hash the instruction
    hash_bytes = hashlib.sha256(instruction.encode()).digest()

    # Convert to floats
    embedding = np.array([
        float(b) / 255.0 for b in hash_bytes[:dim]
    ], dtype=np.float32)

    # Pad if needed
    if len(embedding) < dim:
        embedding = np.pad(embedding, (0, dim - len(embedding)))

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding
