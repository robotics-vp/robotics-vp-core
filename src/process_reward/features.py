"""
Feature Extraction for Process Reward.

Extracts features from SceneTracks_v1 (numpy-only format) and optional MHN outputs.
Operates on latents and kinematics, NOT pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.process_reward.schemas import (
    ProcessRewardConfig,
    MHNSummary,
    FrameFeatures,
    TransitionFeatures,
    EpisodeFeatures,
)


def rotation_matrix_to_angle(R: np.ndarray) -> float:
    """Convert 3x3 rotation matrix to angle (axis-angle representation).

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Rotation angle in radians.
    """
    # Angle from trace: tr(R) = 1 + 2*cos(theta)
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FeatureExtractor:
    """Extracts features from SceneTracksLite for process reward computation.

    Supports operation with and without optional latents (z_shape, z_tex).
    """

    def __init__(self, config: ProcessRewardConfig):
        """Initialize feature extractor.

        Args:
            config: Process reward configuration.
        """
        self.config = config
        self.feature_dim = config.feature_dim
        self.use_latents = config.use_latents
        self.use_mhn_features = config.use_mhn_features
        self.pool_method = config.pool_method

    def extract_episode_features(
        self,
        poses_R: np.ndarray,
        poses_t: np.ndarray,
        scales: np.ndarray,
        visibility: np.ndarray,
        occlusion: np.ndarray,
        ir_loss: np.ndarray,
        converged: np.ndarray,
        entity_types: np.ndarray,
        z_shape: Optional[np.ndarray] = None,
        z_tex: Optional[np.ndarray] = None,
        mhn_summary: Optional[MHNSummary] = None,
        goal_frame_idx: Optional[int] = None,
    ) -> EpisodeFeatures:
        """Extract features for an entire episode.

        Args:
            poses_R: (T, K, 3, 3) rotation matrices.
            poses_t: (T, K, 3) translations.
            scales: (T, K) scale factors.
            visibility: (T, K) visibility scores.
            occlusion: (T, K) occlusion scores.
            ir_loss: (T, K) IR loss per entity.
            converged: (T, K) convergence flags.
            entity_types: (K,) entity types (0=object, 1=body).
            z_shape: Optional (T, K, Zs) shape latents.
            z_tex: Optional (T, K, Zt) texture latents.
            mhn_summary: Optional MHN summary.
            goal_frame_idx: Optional goal frame index.

        Returns:
            EpisodeFeatures with per-frame and transition features.
        """
        T, K = poses_t.shape[:2]

        # Extract frame features
        frame_features = []
        for t in range(T):
            ff = self._extract_frame_features(
                poses_R[t], poses_t[t], scales[t],
                visibility[t], occlusion[t], ir_loss[t], converged[t],
                entity_types,
                z_shape[t] if z_shape is not None else None,
                z_tex[t] if z_tex is not None else None,
                mhn_summary,
            )
            frame_features.append(ff)

        # Extract transition features
        transition_features = []
        for t in range(T - 1):
            tf = self._extract_transition_features(
                poses_R[t], poses_R[t + 1],
                poses_t[t], poses_t[t + 1],
                scales[t], scales[t + 1],
                visibility[t], visibility[t + 1],
                ir_loss[t], ir_loss[t + 1],
                entity_types,
                z_shape[t] if z_shape is not None else None,
                z_shape[t + 1] if z_shape is not None else None,
            )
            transition_features.append(tf)

        # Init and goal features
        init_features = frame_features[0] if frame_features else self._empty_frame_features(K)
        goal_features = None
        if goal_frame_idx is not None and 0 <= goal_frame_idx < T:
            goal_features = frame_features[goal_frame_idx]

        # Global stats
        global_stats = self._compute_global_stats(
            visibility, occlusion, ir_loss, converged, entity_types
        )

        return EpisodeFeatures(
            frame_features=frame_features,
            transition_features=transition_features,
            init_features=init_features,
            goal_features=goal_features,
            global_stats=global_stats,
        )

    def _extract_frame_features(
        self,
        R: np.ndarray,  # (K, 3, 3)
        t: np.ndarray,  # (K, 3)
        scale: np.ndarray,  # (K,)
        visibility: np.ndarray,  # (K,)
        occlusion: np.ndarray,  # (K,)
        ir_loss: np.ndarray,  # (K,)
        converged: np.ndarray,  # (K,)
        entity_types: np.ndarray,  # (K,)
        z_shape: Optional[np.ndarray] = None,  # (K, Zs)
        z_tex: Optional[np.ndarray] = None,  # (K, Zt)
        mhn_summary: Optional[MHNSummary] = None,
    ) -> FrameFeatures:
        """Extract features for a single frame.

        Args:
            R: (K, 3, 3) rotation matrices.
            t: (K, 3) translations.
            scale: (K,) scale factors.
            visibility: (K,) visibility scores.
            occlusion: (K,) occlusion scores.
            ir_loss: (K,) IR loss.
            converged: (K,) convergence flags.
            entity_types: (K,) entity types.
            z_shape: Optional (K, Zs) shape latents.
            z_tex: Optional (K, Zt) texture latents.
            mhn_summary: Optional MHN summary.

        Returns:
            FrameFeatures for this frame.
        """
        K = t.shape[0]

        # Per-track features
        per_track_list = []
        for k in range(K):
            features = self._build_track_features(
                R[k], t[k], scale[k],
                visibility[k], occlusion[k], ir_loss[k],
                entity_types[k],
                z_shape[k] if z_shape is not None else None,
                z_tex[k] if z_tex is not None else None,
            )
            per_track_list.append(features)

        per_track = np.array(per_track_list, dtype=np.float32)  # (K, feature_dim)

        # Pool across tracks
        pooled = self._pool_features(per_track, visibility)

        # Visibility stats
        visibility_stats = {
            "num_visible": int(np.sum(visibility > 0.5)),
            "num_tracks": K,
            "pct_visible": float(np.mean(visibility > 0.5)) if K > 0 else 0.0,
            "pct_occluded": float(np.mean(occlusion > 0.5)) if K > 0 else 0.0,
            "mean_visibility": float(np.mean(visibility)) if K > 0 else 0.0,
            "num_bodies": int(np.sum(entity_types == 1)),
            "num_objects": int(np.sum(entity_types == 0)),
        }

        # IR stats
        ir_stats = {
            "mean_ir_loss": float(np.mean(ir_loss)) if K > 0 else 0.0,
            "max_ir_loss": float(np.max(ir_loss)) if K > 0 else 0.0,
            "pct_converged": float(np.mean(converged)) if K > 0 else 1.0,
        }

        # MHN features
        mhn_features = None
        if self.use_mhn_features and mhn_summary is not None:
            mhn_features = np.array([
                mhn_summary.mean_tree_depth,
                mhn_summary.mean_branch_factor,
                mhn_summary.residual_mean,
                mhn_summary.structural_difficulty,
                mhn_summary.plausibility_score,
            ], dtype=np.float32)

        return FrameFeatures(
            pooled=pooled,
            per_track=per_track,
            visibility_stats=visibility_stats,
            ir_stats=ir_stats,
            mhn_features=mhn_features,
        )

    def _build_track_features(
        self,
        R: np.ndarray,  # (3, 3)
        t: np.ndarray,  # (3,)
        scale: float,
        visibility: float,
        occlusion: float,
        ir_loss: float,
        entity_type: int,
        z_shape: Optional[np.ndarray] = None,  # (Zs,)
        z_tex: Optional[np.ndarray] = None,  # (Zt,)
    ) -> np.ndarray:
        """Build feature vector for a single track at one frame.

        Args:
            R: (3, 3) rotation matrix.
            t: (3,) translation.
            scale: Scale factor.
            visibility: Visibility score.
            occlusion: Occlusion score.
            ir_loss: IR loss.
            entity_type: 0=object, 1=body.
            z_shape: Optional shape latent.
            z_tex: Optional texture latent.

        Returns:
            Feature vector of dimension feature_dim.
        """
        features = []

        # Position features (normalized)
        pos_norm = t / (np.linalg.norm(t) + 1e-6)
        features.extend(pos_norm.tolist())
        features.append(np.linalg.norm(t))  # distance from origin

        # Rotation features (first column of R as direction + angle)
        direction = R[:, 0]  # x-axis direction
        features.extend(direction.tolist())
        angle = rotation_matrix_to_angle(R)
        features.append(angle / np.pi)  # normalized angle

        # Scale
        features.append(np.log1p(scale))

        # Visibility/occlusion
        if self.config.include_visibility_features:
            features.append(visibility)
            features.append(occlusion)

        # IR loss
        if self.config.include_ir_features:
            features.append(ir_loss)

        # Entity type (one-hot)
        features.append(float(entity_type == 0))  # is_object
        features.append(float(entity_type == 1))  # is_body

        # Latent features (if available)
        if self.use_latents:
            if z_shape is not None:
                # Use first few PCA-like components
                z_norm = z_shape[:min(4, len(z_shape))]
                z_norm = z_norm / (np.linalg.norm(z_norm) + 1e-6)
                features.extend(z_norm.tolist())
                # Pad if needed
                for _ in range(4 - len(z_norm)):
                    features.append(0.0)
            else:
                features.extend([0.0] * 4)

            if z_tex is not None:
                z_norm = z_tex[:min(4, len(z_tex))]
                z_norm = z_norm / (np.linalg.norm(z_norm) + 1e-6)
                features.extend(z_norm.tolist())
                for _ in range(4 - len(z_norm)):
                    features.append(0.0)
            else:
                features.extend([0.0] * 4)

        # Pad or truncate to feature_dim
        features = np.array(features, dtype=np.float32)
        if len(features) >= self.feature_dim:
            return features[:self.feature_dim]
        else:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:len(features)] = features
            return padded

    def _pool_features(
        self,
        per_track: np.ndarray,  # (K, feature_dim)
        visibility: np.ndarray,  # (K,)
    ) -> np.ndarray:
        """Pool features across tracks.

        Args:
            per_track: (K, feature_dim) per-track features.
            visibility: (K,) visibility weights.

        Returns:
            (feature_dim,) pooled features.
        """
        K = per_track.shape[0]
        if K == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Weight by visibility
        weights = np.clip(visibility, 0.1, 1.0)  # floor to avoid zeros
        weights = weights / weights.sum()

        if self.pool_method == "mean":
            return np.sum(per_track * weights[:, None], axis=0)
        elif self.pool_method == "max":
            return np.max(per_track, axis=0)
        else:
            # Default to weighted mean
            return np.sum(per_track * weights[:, None], axis=0)

    def _extract_transition_features(
        self,
        R_t: np.ndarray,  # (K, 3, 3)
        R_t1: np.ndarray,  # (K, 3, 3)
        t_t: np.ndarray,  # (K, 3)
        t_t1: np.ndarray,  # (K, 3)
        scale_t: np.ndarray,  # (K,)
        scale_t1: np.ndarray,  # (K,)
        vis_t: np.ndarray,  # (K,)
        vis_t1: np.ndarray,  # (K,)
        ir_t: np.ndarray,  # (K,)
        ir_t1: np.ndarray,  # (K,)
        entity_types: np.ndarray,  # (K,)
        z_shape_t: Optional[np.ndarray] = None,  # (K, Zs)
        z_shape_t1: Optional[np.ndarray] = None,  # (K, Zs)
    ) -> TransitionFeatures:
        """Extract transition features between two frames.

        Args:
            R_t, R_t1: Rotation matrices at t and t+1.
            t_t, t_t1: Translations at t and t+1.
            scale_t, scale_t1: Scales at t and t+1.
            vis_t, vis_t1: Visibility at t and t+1.
            ir_t, ir_t1: IR loss at t and t+1.
            entity_types: Entity types.
            z_shape_t, z_shape_t1: Optional shape latents.

        Returns:
            TransitionFeatures.
        """
        K = t_t.shape[0]

        # Position delta
        delta_pos = np.linalg.norm(t_t1 - t_t, axis=1)

        # Rotation delta (angle between rotations)
        delta_rot = np.zeros(K, dtype=np.float32)
        for k in range(K):
            # R_delta = R_t1 @ R_t.T
            R_delta = R_t1[k] @ R_t[k].T
            delta_rot[k] = rotation_matrix_to_angle(R_delta)

        # Scale delta
        delta_scale = np.abs(scale_t1 - scale_t)

        # Visibility transitions
        became_visible = int(np.sum((vis_t < 0.5) & (vis_t1 >= 0.5)))
        became_occluded = int(np.sum((vis_t >= 0.5) & (vis_t1 < 0.5)))
        visibility_transitions = {
            "became_visible": became_visible,
            "became_occluded": became_occluded,
        }

        # IR delta
        ir_delta = ir_t1 - ir_t

        # Latent similarity delta
        latent_sim_delta = None
        if z_shape_t is not None and z_shape_t1 is not None:
            latent_sim_delta = np.zeros(K, dtype=np.float32)
            for k in range(K):
                sim = cosine_similarity(z_shape_t[k], z_shape_t1[k])
                latent_sim_delta[k] = 1.0 - sim  # distance = 1 - similarity

        # Pooled transition features
        pooled_list = [
            np.mean(delta_pos),
            np.max(delta_pos),
            np.mean(delta_rot),
            np.max(delta_rot),
            np.mean(delta_scale),
            float(became_visible),
            float(became_occluded),
            np.mean(ir_delta),
        ]
        if latent_sim_delta is not None:
            pooled_list.append(np.mean(latent_sim_delta))
        else:
            pooled_list.append(0.0)

        pooled = np.array(pooled_list, dtype=np.float32)

        return TransitionFeatures(
            delta_pos=delta_pos.astype(np.float32),
            delta_rot=delta_rot.astype(np.float32),
            delta_scale=delta_scale.astype(np.float32),
            visibility_transitions=visibility_transitions,
            ir_delta=ir_delta.astype(np.float32),
            latent_sim_delta=latent_sim_delta,
            pooled=pooled,
        )

    def _compute_global_stats(
        self,
        visibility: np.ndarray,  # (T, K)
        occlusion: np.ndarray,  # (T, K)
        ir_loss: np.ndarray,  # (T, K)
        converged: np.ndarray,  # (T, K)
        entity_types: np.ndarray,  # (K,)
    ) -> Dict[str, float]:
        """Compute global episode statistics.

        Args:
            visibility: (T, K) visibility over episode.
            occlusion: (T, K) occlusion over episode.
            ir_loss: (T, K) IR loss over episode.
            converged: (T, K) convergence over episode.
            entity_types: (K,) entity types.

        Returns:
            Dictionary of global statistics.
        """
        T, K = visibility.shape

        return {
            "num_frames": T,
            "num_tracks": K,
            "num_bodies": int(np.sum(entity_types == 1)),
            "num_objects": int(np.sum(entity_types == 0)),
            "mean_visibility": float(np.mean(visibility)),
            "mean_occlusion": float(np.mean(occlusion)),
            "mean_ir_loss": float(np.mean(ir_loss)),
            "pct_converged": float(np.mean(converged)),
            "ir_loss_trend": float(np.mean(ir_loss[-1]) - np.mean(ir_loss[0])) if T > 1 else 0.0,
        }

    def _empty_frame_features(self, K: int) -> FrameFeatures:
        """Create empty frame features for edge cases."""
        return FrameFeatures(
            pooled=np.zeros(self.feature_dim, dtype=np.float32),
            per_track=np.zeros((K, self.feature_dim), dtype=np.float32),
            visibility_stats={},
            ir_stats={},
            mhn_features=None,
        )


def extract_features_from_scene_tracks_lite(
    scene_tracks_lite: Any,  # SceneTracksLite
    config: ProcessRewardConfig,
    mhn_summary: Optional[MHNSummary] = None,
    goal_frame_idx: Optional[int] = None,
) -> EpisodeFeatures:
    """Convenience function to extract features from SceneTracksLite.

    Args:
        scene_tracks_lite: SceneTracksLite instance from deserialization.
        config: Process reward configuration.
        mhn_summary: Optional MHN summary.
        goal_frame_idx: Optional goal frame index.

    Returns:
        EpisodeFeatures.
    """
    extractor = FeatureExtractor(config)

    return extractor.extract_episode_features(
        poses_R=scene_tracks_lite.poses_R,
        poses_t=scene_tracks_lite.poses_t,
        scales=scene_tracks_lite.scales,
        visibility=scene_tracks_lite.visibility,
        occlusion=scene_tracks_lite.occlusion,
        ir_loss=scene_tracks_lite.ir_loss,
        converged=scene_tracks_lite.converged,
        entity_types=scene_tracks_lite.entity_types,
        z_shape=scene_tracks_lite.z_shape,
        z_tex=scene_tracks_lite.z_tex,
        mhn_summary=mhn_summary,
        goal_frame_idx=goal_frame_idx,
    )
