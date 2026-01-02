"""
Kalman Track Manager.

Implements multi-object/body tracking with Kalman filtering and association.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np

from src.vision.scene_ir_tracker.config import TrackingConfig
from src.vision.scene_ir_tracker.types import SceneEntity3D

logger = logging.getLogger(__name__)


@dataclass
class KalmanTrack:
    """Internal track state for Kalman filter.

    Attributes:
        track_id: Unique track identifier.
        entity_type: "body" or "object".
        class_name: Class name for objects.
        state: State vector [x, y, z, vx, vy, vz, scale, yaw].
        covariance: State covariance matrix.
        z_shape_ema: EMA shape latent.
        z_tex_ema: EMA texture latent.
        age: Frames since track creation.
        hits: Number of successful associations.
        time_since_update: Frames since last update.
        history: List of past entity states.
    """

    track_id: str
    entity_type: str
    class_name: Optional[str]
    state: np.ndarray  # (8,) [x, y, z, vx, vy, vz, scale, yaw]
    covariance: np.ndarray  # (8, 8)
    z_shape_ema: Optional[np.ndarray] = None
    z_tex_ema: Optional[np.ndarray] = None
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    history: List[SceneEntity3D] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        """Get current position from state."""
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity from state."""
        return self.state[3:6].copy()

    @property
    def scale(self) -> float:
        """Get current scale from state."""
        return float(self.state[6])


class KalmanTrackManager:
    """Multi-object/body tracking manager with Kalman filtering.

    Maintains tracks across frames using Kalman prediction/update
    and Hungarian algorithm for association.
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        """Initialize track manager.

        Args:
            config: Tracking configuration.
        """
        self.config = config or TrackingConfig()
        self.tracks: List[KalmanTrack] = []
        self._next_id = 0
        self._id_switch_count = 0

        # Kalman filter matrices
        self._setup_kalman_matrices()

    def _setup_kalman_matrices(self) -> None:
        """Setup Kalman filter transition and observation matrices."""
        # State: [x, y, z, vx, vy, vz, scale, yaw]
        # State transition (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 3] = 1.0  # x += vx * dt
        self.F[1, 4] = 1.0  # y += vy * dt
        self.F[2, 5] = 1.0  # z += vz * dt

        # Observation matrix (we observe position and scale)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z
        self.H[3, 6] = 1.0  # scale

        # Process noise
        q = self.config.kalman_process_noise
        self.Q = np.diag([q, q, q, q * 2, q * 2, q * 2, q * 0.5, q * 0.5]).astype(np.float32)

        # Observation noise
        r = self.config.kalman_observation_noise
        self.R = np.diag([r, r, r, r * 0.5]).astype(np.float32)

    def _generate_track_id(self) -> str:
        """Generate unique track ID."""
        self._next_id += 1
        return f"track_{self._next_id:04d}"

    def update(
        self,
        frame_entities: List[SceneEntity3D],
    ) -> List[SceneEntity3D]:
        """Update tracks with new frame observations.

        Args:
            frame_entities: Detected entities in current frame.

        Returns:
            Updated entities with stable track IDs.
        """
        # Predict existing tracks
        for track in self.tracks:
            self._predict(track)

        # Associate detections to tracks
        matches, unmatched_dets, unmatched_tracks = self._associate(frame_entities)

        # Update matched tracks
        for det_idx, track_idx in matches:
            entity = frame_entities[det_idx]
            track = self.tracks[track_idx]
            self._update_track(track, entity)

        # Handle unmatched tracks (mark as missed)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            entity = frame_entities[det_idx]
            new_track = self._create_track(entity)
            self.tracks.append(new_track)

        # Build output entities with stable IDs
        output_entities = []
        for det_idx, track_idx in matches:
            entity = frame_entities[det_idx]
            track = self.tracks[track_idx]
            updated = self._entity_from_track(entity, track)
            output_entities.append(updated)

        for det_idx in unmatched_dets:
            entity = frame_entities[det_idx]
            # Find the newly created track
            for track in self.tracks:
                if track.age == 0 and np.allclose(track.position, entity.position, atol=0.01):
                    updated = self._entity_from_track(entity, track)
                    output_entities.append(updated)
                    break

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.config.max_age
        ]

        return output_entities

    def _predict(self, track: KalmanTrack) -> None:
        """Kalman predict step."""
        track.state = self.F @ track.state
        track.covariance = self.F @ track.covariance @ self.F.T + self.Q
        track.age += 1

    def _update_track(
        self,
        track: KalmanTrack,
        entity: SceneEntity3D,
    ) -> None:
        """Kalman update step with observation."""
        # Observation: [x, y, z, scale]
        z = np.array([
            entity.position[0],
            entity.position[1],
            entity.position[2],
            entity.scale,
        ], dtype=np.float32)

        # Innovation
        y = z - self.H @ track.state

        # Innovation covariance
        S = self.H @ track.covariance @ self.H.T + self.R

        # Kalman gain
        K = track.covariance @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        track.state = track.state + K @ y
        track.covariance = (np.eye(8) - K @ self.H) @ track.covariance

        # Update EMA latents
        alpha = self.config.ema_alpha
        if entity.z_shape is not None:
            if track.z_shape_ema is None:
                track.z_shape_ema = entity.z_shape.copy()
            else:
                track.z_shape_ema = alpha * track.z_shape_ema + (1 - alpha) * entity.z_shape

        if entity.z_tex is not None:
            if track.z_tex_ema is None:
                track.z_tex_ema = entity.z_tex.copy()
            else:
                track.z_tex_ema = alpha * track.z_tex_ema + (1 - alpha) * entity.z_tex

        track.hits += 1
        track.time_since_update = 0
        track.history.append(entity)

    def _create_track(self, entity: SceneEntity3D) -> KalmanTrack:
        """Create new track from entity."""
        state = np.zeros(8, dtype=np.float32)
        state[:3] = entity.position
        state[6] = entity.scale

        covariance = np.eye(8, dtype=np.float32) * 10.0

        return KalmanTrack(
            track_id=self._generate_track_id(),
            entity_type=entity.entity_type,
            class_name=entity.class_name,
            state=state,
            covariance=covariance,
            z_shape_ema=entity.z_shape.copy() if entity.z_shape is not None else None,
            z_tex_ema=entity.z_tex.copy() if entity.z_tex is not None else None,
            age=0,
            hits=1,
            time_since_update=0,
            history=[entity],
        )

    def _entity_from_track(
        self,
        entity: SceneEntity3D,
        track: KalmanTrack,
    ) -> SceneEntity3D:
        """Create entity with track ID and EMA latents."""
        # Update pose with Kalman-filtered position
        pose = entity.pose.copy()
        pose[:3, 3] = track.position

        return SceneEntity3D(
            entity_type=entity.entity_type,
            track_id=track.track_id,
            pose=pose,
            scale=track.scale,
            class_name=entity.class_name,
            mask_2d=entity.mask_2d,
            mask_logits=entity.mask_logits,
            geometry_handle=entity.geometry_handle,
            z_shape=entity.z_shape,
            z_tex=entity.z_tex,
            z_shape_ema=track.z_shape_ema,
            z_tex_ema=track.z_tex_ema,
            visibility=entity.visibility,
            occlusion_score=entity.occlusion_score,
            ir_loss=entity.ir_loss,
            joints_3d=entity.joints_3d,
        )

    def _associate(
        self,
        detections: List[SceneEntity3D],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to existing tracks.

        Returns:
            Tuple of (matches, unmatched_dets, unmatched_tracks)
            where matches is list of (detection_idx, track_idx) pairs.
        """
        if not self.tracks or not detections:
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
            return [], unmatched_dets, unmatched_tracks

        # Build cost matrix
        cost_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                cost_matrix[d, t] = self._compute_cost(det, track)

        # Hungarian algorithm (greedy approximation for simplicity)
        matches, unmatched_dets, unmatched_tracks = self._greedy_assignment(
            cost_matrix,
            threshold=self.config.association_distance_threshold,
        )

        # Detect ID switches
        for det_idx, track_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_idx]
            if det.entity_type != track.entity_type:
                self._id_switch_count += 1

        return matches, unmatched_dets, unmatched_tracks

    def _compute_cost(
        self,
        detection: SceneEntity3D,
        track: KalmanTrack,
    ) -> float:
        """Compute association cost between detection and track."""
        # Centroid distance
        dist = np.linalg.norm(detection.position - track.position)
        dist_cost = self.config.centroid_distance_weight * dist

        # 3D IoU for objects (simplified as scale overlap)
        if detection.entity_type == "object":
            s1, s2 = detection.scale, track.scale
            overlap = min(s1, s2) / (max(s1, s2) + 1e-8)
            iou_cost = self.config.iou_weight * (1 - overlap)
        else:
            iou_cost = 0.0

        # Latent similarity (only if dimensions match)
        latent_cost = 0.0
        if (detection.z_shape is not None and track.z_shape_ema is not None
                and detection.z_shape.shape == track.z_shape_ema.shape):
            sim = np.dot(detection.z_shape, track.z_shape_ema)
            sim /= (np.linalg.norm(detection.z_shape) * np.linalg.norm(track.z_shape_ema) + 1e-8)
            latent_cost = self.config.latent_similarity_weight * (1 - sim)

        # Type mismatch penalty
        type_penalty = 0.0 if detection.entity_type == track.entity_type else 100.0

        return dist_cost + iou_cost + latent_cost + type_penalty

    def _greedy_assignment(
        self,
        cost_matrix: np.ndarray,
        threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy assignment for association.

        Args:
            cost_matrix: (M, N) cost matrix.
            threshold: Max cost for valid assignment.

        Returns:
            (matches, unmatched_rows, unmatched_cols)
        """
        M, N = cost_matrix.shape
        matches = []
        matched_rows = set()
        matched_cols = set()

        # Sort all costs
        costs = []
        for i in range(M):
            for j in range(N):
                if cost_matrix[i, j] < threshold:
                    costs.append((cost_matrix[i, j], i, j))
        costs.sort()

        # Greedy matching
        for cost, i, j in costs:
            if i not in matched_rows and j not in matched_cols:
                matches.append((i, j))
                matched_rows.add(i)
                matched_cols.add(j)

        unmatched_rows = [i for i in range(M) if i not in matched_rows]
        unmatched_cols = [j for j in range(N) if j not in matched_cols]

        return matches, unmatched_rows, unmatched_cols

    def get_tracks_dict(self) -> Dict[str, List[SceneEntity3D]]:
        """Get all tracks as dict of track_id -> history."""
        return {t.track_id: t.history.copy() for t in self.tracks}

    @property
    def id_switch_count(self) -> int:
        """Get total ID switch count."""
        return self._id_switch_count

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks = []
        self._next_id = 0
        self._id_switch_count = 0


def create_kalman_track_manager(
    config: Optional[Dict[str, Any]] = None,
) -> KalmanTrackManager:
    """Factory function to create Kalman track manager.

    Args:
        config: Configuration dict.

    Returns:
        Configured KalmanTrackManager.
    """
    if config:
        cfg = TrackingConfig.from_dict(config)
    else:
        cfg = TrackingConfig()
    return KalmanTrackManager(config=cfg)
