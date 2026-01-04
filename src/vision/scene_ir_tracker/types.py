"""
Scene IR Tracker Types.

Core types for the Scene IR Tracker module including entity and track dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np


@dataclass
class SceneEntity3D:
    """A tracked 3D entity in the scene.

    Represents either a body (person) or object with its state at a single frame.
    All coordinates are in world frame.

    Attributes:
        entity_type: Type of entity ("body" or "object").
        class_name: Semantic class name (e.g., "forklift", "box") for objects.
        track_id: Unique track identifier.
        mask_2d: (H, W) boolean mask in image space.
        mask_logits: Optional (H, W) soft mask logits.
        pose: (4, 4) world_from_entity homogeneous transform.
        scale: Uniform scale factor.
        geometry_handle: Reference to mesh or gaussian splats representation.
        z_shape: Shape latent embedding.
        z_tex: Texture latent embedding.
        z_shape_ema: EMA-smoothed shape latent.
        z_tex_ema: EMA-smoothed texture latent.
        visibility: Visibility fraction [0, 1].
        occlusion_score: Occlusion score [0, 1] (1=fully occluded).
        ir_loss: Per-entity inverse rendering loss.
        joints_3d: For bodies, dict of joint_name -> (3,) world position.
    """

    entity_type: Literal["body", "object"]
    track_id: str
    pose: np.ndarray  # (4, 4) world_from_entity
    scale: float = 1.0
    class_name: Optional[str] = None
    mask_2d: Optional[np.ndarray] = None
    mask_logits: Optional[np.ndarray] = None
    geometry_handle: Any = None
    z_shape: Optional[np.ndarray] = None
    z_tex: Optional[np.ndarray] = None
    z_shape_ema: Optional[np.ndarray] = None
    z_tex_ema: Optional[np.ndarray] = None
    visibility: float = 1.0
    occlusion_score: float = 0.0
    ir_loss: float = 0.0
    joints_3d: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self) -> None:
        self.pose = np.asarray(self.pose, dtype=np.float32)
        if self.pose.shape != (4, 4):
            raise ValueError(f"pose must be (4, 4), got {self.pose.shape}")
        self.scale = float(self.scale)
        self.visibility = float(max(0.0, min(1.0, self.visibility)))
        self.occlusion_score = float(max(0.0, min(1.0, self.occlusion_score)))
        self.ir_loss = float(self.ir_loss)

        if self.mask_2d is not None:
            self.mask_2d = np.asarray(self.mask_2d, dtype=bool)
        if self.mask_logits is not None:
            self.mask_logits = np.asarray(self.mask_logits, dtype=np.float32)
        if self.z_shape is not None:
            self.z_shape = np.asarray(self.z_shape, dtype=np.float32)
        if self.z_tex is not None:
            self.z_tex = np.asarray(self.z_tex, dtype=np.float32)
        if self.z_shape_ema is not None:
            self.z_shape_ema = np.asarray(self.z_shape_ema, dtype=np.float32)
        if self.z_tex_ema is not None:
            self.z_tex_ema = np.asarray(self.z_tex_ema, dtype=np.float32)

    @property
    def position(self) -> np.ndarray:
        """Get entity position (translation from pose)."""
        return self.pose[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        """Get entity rotation matrix (3x3)."""
        return self.pose[:3, :3].copy()

    @property
    def centroid(self) -> np.ndarray:
        """Get entity centroid (same as position for objects, pelvis for bodies)."""
        if self.entity_type == "body" and self.joints_3d and "pelvis" in self.joints_3d:
            return self.joints_3d["pelvis"].copy()
        return self.position

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (without large arrays)."""
        result = {
            "entity_type": self.entity_type,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "pose": self.pose.tolist(),
            "scale": self.scale,
            "visibility": self.visibility,
            "occlusion_score": self.occlusion_score,
            "ir_loss": self.ir_loss,
        }
        if self.joints_3d is not None:
            result["joints_3d"] = {k: v.tolist() for k, v in self.joints_3d.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneEntity3D":
        """Create from dictionary."""
        joints_3d = data.get("joints_3d")
        if joints_3d is not None:
            joints_3d = {k: np.array(v, dtype=np.float32) for k, v in joints_3d.items()}

        return cls(
            entity_type=data["entity_type"],
            track_id=data["track_id"],
            pose=np.array(data["pose"], dtype=np.float32),
            scale=data.get("scale", 1.0),
            class_name=data.get("class_name"),
            visibility=data.get("visibility", 1.0),
            occlusion_score=data.get("occlusion_score", 0.0),
            ir_loss=data.get("ir_loss", 0.0),
            joints_3d=joints_3d,
        )


@dataclass
class SceneTrackerMetrics:
    """Metrics from a tracking run.

    Attributes:
        ir_loss_per_frame: IR loss at each frame.
        id_switch_count: Number of track ID switches detected.
        occlusion_rate: Fraction of entity-frames with occlusion > 0.5.
        mean_ir_loss: Mean IR loss across all frames.
        converged_count: Number of frames where IR refinement converged.
        total_frames: Total number of frames processed.
        total_tracks: Total unique track IDs.
        track_lengths: List of track lengths (frames per track).
    """

    ir_loss_per_frame: List[float] = field(default_factory=list)
    id_switch_count: int = 0
    occlusion_rate: float = 0.0
    mean_ir_loss: float = 0.0
    converged_count: int = 0
    total_frames: int = 0
    total_tracks: int = 0
    track_lengths: List[int] = field(default_factory=list)
    diverged_count: int = 0  # Number of frames where refinement diverged
    pct_diverged: float = 0.0  # Percentage of diverged frames
    pct_converged: float = 0.0  # Percentage of converged frames

    def __post_init__(self) -> None:
        self.id_switch_count = int(self.id_switch_count)
        self.occlusion_rate = float(self.occlusion_rate)
        self.mean_ir_loss = float(self.mean_ir_loss)
        self.converged_count = int(self.converged_count)
        self.total_frames = int(self.total_frames)
        self.total_tracks = int(self.total_tracks)
        self.diverged_count = int(self.diverged_count)
        
        # Compute percentages
        if self.total_frames > 0:
            self.pct_diverged = float(self.diverged_count / self.total_frames * 100)
            self.pct_converged = float(self.converged_count / self.total_frames * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ir_loss_per_frame": self.ir_loss_per_frame,
            "id_switch_count": self.id_switch_count,
            "occlusion_rate": self.occlusion_rate,
            "mean_ir_loss": self.mean_ir_loss,
            "converged_count": self.converged_count,
            "total_frames": self.total_frames,
            "total_tracks": self.total_tracks,
            "track_lengths": self.track_lengths,
            "diverged_count": self.diverged_count,
            "pct_diverged": self.pct_diverged,
            "pct_converged": self.pct_converged,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneTrackerMetrics":
        """Create from dictionary."""
        return cls(
            ir_loss_per_frame=data.get("ir_loss_per_frame", []),
            id_switch_count=data.get("id_switch_count", 0),
            occlusion_rate=data.get("occlusion_rate", 0.0),
            mean_ir_loss=data.get("mean_ir_loss", 0.0),
            converged_count=data.get("converged_count", 0),
            total_frames=data.get("total_frames", 0),
            total_tracks=data.get("total_tracks", 0),
            track_lengths=data.get("track_lengths", []),
            diverged_count=data.get("diverged_count", 0),
        )


@dataclass
class SceneTracks:
    """Container for tracked entities across an episode.

    Attributes:
        frames: List of per-frame entity states. frames[t] is list of entities at time t.
        tracks: Dict mapping track_id to list of entities across frames for that track.
        metrics: Aggregated tracking metrics.
        config_used: Configuration used for tracking (for reproducibility).
    """

    frames: List[List[SceneEntity3D]] = field(default_factory=list)
    tracks: Dict[str, List[SceneEntity3D]] = field(default_factory=dict)
    metrics: SceneTrackerMetrics = field(default_factory=SceneTrackerMetrics)
    config_used: Optional[Dict[str, Any]] = None

    @property
    def num_frames(self) -> int:
        """Number of frames."""
        return len(self.frames)

    @property
    def track_ids(self) -> List[str]:
        """List of all track IDs."""
        return list(self.tracks.keys())

    def get_positions_for_mhn(
        self,
        body_joints: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Extract positions array suitable for Motion Hierarchy Node.

        Returns positions as (T, N, 3) where N is the total number of entities
        (objects + body joints or body centroids).

        Args:
            body_joints: For bodies, which joints to extract. If None, uses centroid.

        Returns:
            (T, N, 3) array of world-space positions.
        """
        if not self.frames:
            return np.zeros((0, 0, 3), dtype=np.float32)

        # Collect all positions per frame
        all_positions = []
        for frame_entities in self.frames:
            frame_positions = []
            for entity in frame_entities:
                if entity.entity_type == "object":
                    frame_positions.append(entity.centroid)
                elif entity.entity_type == "body":
                    if body_joints and entity.joints_3d:
                        for joint_name in body_joints:
                            if joint_name in entity.joints_3d:
                                frame_positions.append(entity.joints_3d[joint_name])
                    else:
                        frame_positions.append(entity.centroid)
            all_positions.append(frame_positions)

        # Pad to uniform length
        max_n = max(len(fp) for fp in all_positions) if all_positions else 0
        if max_n == 0:
            return np.zeros((len(self.frames), 0, 3), dtype=np.float32)

        result = np.zeros((len(self.frames), max_n, 3), dtype=np.float32)
        for t, frame_positions in enumerate(all_positions):
            for i, pos in enumerate(frame_positions):
                result[t, i] = pos

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "frames": [[e.to_dict() for e in frame] for frame in self.frames],
            "tracks": {
                track_id: [e.to_dict() for e in entities]
                for track_id, entities in self.tracks.items()
            },
            "metrics": self.metrics.to_dict(),
            "config_used": self.config_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneTracks":
        """Create from dictionary."""
        frames = [
            [SceneEntity3D.from_dict(e) for e in frame]
            for frame in data.get("frames", [])
        ]
        tracks = {
            track_id: [SceneEntity3D.from_dict(e) for e in entities]
            for track_id, entities in data.get("tracks", {}).items()
        }
        metrics = SceneTrackerMetrics.from_dict(data.get("metrics", {}))

        return cls(
            frames=frames,
            tracks=tracks,
            metrics=metrics,
            config_used=data.get("config_used"),
        )

    def summary(self) -> Dict[str, Any]:
        """Get a concise summary for logging/metadata."""
        return {
            "num_frames": self.num_frames,
            "num_tracks": len(self.tracks),
            "mean_ir_loss": self.metrics.mean_ir_loss,
            "id_switch_count": self.metrics.id_switch_count,
            "occlusion_rate": self.metrics.occlusion_rate,
        }
