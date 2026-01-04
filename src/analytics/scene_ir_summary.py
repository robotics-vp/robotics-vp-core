"""
Scene IR Summary.

Provides summary metrics for scene IR tracker quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SceneIRSummary:
    """Summary metrics for scene IR tracker quality.
    
    Attributes:
        ir_loss_mean: Mean inverse rendering loss across frames.
        ir_loss_std: Standard deviation of IR loss.
        ir_loss_min: Minimum IR loss (best frame).
        ir_loss_max: Maximum IR loss (worst frame).
        convergence_rate: Fraction of frames where IR optimization converged.
        id_switch_rate: ID switches per 100 frames.
        occlusion_rate: Fraction of entity-frames with significant occlusion.
        num_tracks: Total number of unique tracks.
        num_bodies: Number of body tracks.
        num_objects: Number of object tracks.
        mean_track_length: Average track length in frames.
        quality_score: Scalar quality score [0, 1].
    """
    
    ir_loss_mean: float = 0.0
    ir_loss_std: float = 0.0
    ir_loss_min: float = 0.0
    ir_loss_max: float = 0.0
    convergence_rate: float = 1.0
    id_switch_rate: float = 0.0
    occlusion_rate: float = 0.0
    num_tracks: int = 0
    num_bodies: int = 0
    num_objects: int = 0
    mean_track_length: float = 0.0
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ir_loss_mean": self.ir_loss_mean,
            "ir_loss_std": self.ir_loss_std,
            "ir_loss_min": self.ir_loss_min,
            "ir_loss_max": self.ir_loss_max,
            "convergence_rate": self.convergence_rate,
            "id_switch_rate": self.id_switch_rate,
            "occlusion_rate": self.occlusion_rate,
            "num_tracks": self.num_tracks,
            "num_bodies": self.num_bodies,
            "num_objects": self.num_objects,
            "mean_track_length": self.mean_track_length,
            "quality_score": self.quality_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneIRSummary":
        """Deserialize from dictionary."""
        return cls(
            ir_loss_mean=data.get("ir_loss_mean", 0.0),
            ir_loss_std=data.get("ir_loss_std", 0.0),
            ir_loss_min=data.get("ir_loss_min", 0.0),
            ir_loss_max=data.get("ir_loss_max", 0.0),
            convergence_rate=data.get("convergence_rate", 1.0),
            id_switch_rate=data.get("id_switch_rate", 0.0),
            occlusion_rate=data.get("occlusion_rate", 0.0),
            num_tracks=data.get("num_tracks", 0),
            num_bodies=data.get("num_bodies", 0),
            num_objects=data.get("num_objects", 0),
            mean_track_length=data.get("mean_track_length", 0.0),
            quality_score=data.get("quality_score", 1.0),
        )


def compute_scene_ir_summary(scene_tracks: Dict[str, Any]) -> SceneIRSummary:
    """Compute summary metrics from scene tracks data.
    
    Args:
        scene_tracks: Scene tracks dictionary from SceneTracks.to_dict().
    
    Returns:
        SceneIRSummary with computed metrics.
    """
    # Extract metrics from scene_tracks
    metrics = scene_tracks.get("metrics", {})
    tracks = scene_tracks.get("tracks", {})
    frames = scene_tracks.get("frames", [])
    
    # IR loss stats
    ir_losses = metrics.get("ir_loss_per_frame", [])
    if ir_losses:
        ir_loss_mean = float(np.mean(ir_losses))
        ir_loss_std = float(np.std(ir_losses))
        ir_loss_min = float(np.min(ir_losses))
        ir_loss_max = float(np.max(ir_losses))
    else:
        ir_loss_mean = ir_loss_std = ir_loss_min = ir_loss_max = 0.0
    
    # Count tracks by type
    num_bodies = 0
    num_objects = 0
    track_lengths = []
    
    for track_id, track_history in tracks.items():
        track_lengths.append(len(track_history))
        if track_history:
            first_entity = track_history[0]
            if isinstance(first_entity, dict):
                entity_type = first_entity.get("entity_type", "object")
            else:
                entity_type = getattr(first_entity, "entity_type", "object")
            
            if entity_type == "body":
                num_bodies += 1
            else:
                num_objects += 1
    
    num_tracks = len(tracks)
    mean_track_length = float(np.mean(track_lengths)) if track_lengths else 0.0
    
    # Convergence and ID switches
    total_frames = metrics.get("total_frames", len(frames))
    converged_count = metrics.get("converged_count", total_frames)
    convergence_rate = converged_count / max(1, total_frames)
    
    id_switch_count = metrics.get("id_switch_count", 0)
    id_switch_rate = id_switch_count / max(1, total_frames) * 100  # per 100 frames
    
    occlusion_rate = metrics.get("occlusion_rate", 0.0)
    
    # Compute quality score
    quality_score = compute_quality_score(
        ir_loss_mean=ir_loss_mean,
        convergence_rate=convergence_rate,
        id_switch_rate=id_switch_rate,
        occlusion_rate=occlusion_rate,
    )
    
    return SceneIRSummary(
        ir_loss_mean=ir_loss_mean,
        ir_loss_std=ir_loss_std,
        ir_loss_min=ir_loss_min,
        ir_loss_max=ir_loss_max,
        convergence_rate=convergence_rate,
        id_switch_rate=id_switch_rate,
        occlusion_rate=occlusion_rate,
        num_tracks=num_tracks,
        num_bodies=num_bodies,
        num_objects=num_objects,
        mean_track_length=mean_track_length,
        quality_score=quality_score,
    )


def compute_quality_score(
    ir_loss_mean: float,
    convergence_rate: float,
    id_switch_rate: float,
    occlusion_rate: float,
    ir_loss_threshold: float = 0.5,
) -> float:
    """Compute scalar quality score from metrics.
    
    Args:
        ir_loss_mean: Mean IR loss (lower is better).
        convergence_rate: Convergence rate [0, 1] (higher is better).
        id_switch_rate: ID switches per 100 frames (lower is better).
        occlusion_rate: Occlusion rate [0, 1] (lower is better).
        ir_loss_threshold: Threshold for normalizing IR loss.
    
    Returns:
        Quality score in [0, 1].
    """
    # Normalize each component
    ir_quality = max(0, 1 - ir_loss_mean / ir_loss_threshold)
    convergence_quality = convergence_rate
    id_quality = max(0, 1 - id_switch_rate / 10)  # 10 switches per 100 frames is bad
    occlusion_quality = 1 - occlusion_rate
    
    # Weighted average
    quality = (
        0.4 * ir_quality +
        0.2 * convergence_quality +
        0.2 * id_quality +
        0.2 * occlusion_quality
    )
    
    return float(np.clip(quality, 0, 1))


def aggregate_scene_ir_summaries(summaries: List[SceneIRSummary]) -> SceneIRSummary:
    """Aggregate multiple summaries into one.
    
    Args:
        summaries: List of SceneIRSummary to aggregate.
    
    Returns:
        Aggregated SceneIRSummary.
    """
    if not summaries:
        return SceneIRSummary()
    
    return SceneIRSummary(
        ir_loss_mean=float(np.mean([s.ir_loss_mean for s in summaries])),
        ir_loss_std=float(np.mean([s.ir_loss_std for s in summaries])),
        ir_loss_min=float(np.min([s.ir_loss_min for s in summaries])),
        ir_loss_max=float(np.max([s.ir_loss_max for s in summaries])),
        convergence_rate=float(np.mean([s.convergence_rate for s in summaries])),
        id_switch_rate=float(np.mean([s.id_switch_rate for s in summaries])),
        occlusion_rate=float(np.mean([s.occlusion_rate for s in summaries])),
        num_tracks=sum(s.num_tracks for s in summaries),
        num_bodies=sum(s.num_bodies for s in summaries),
        num_objects=sum(s.num_objects for s in summaries),
        mean_track_length=float(np.mean([s.mean_track_length for s in summaries])),
        quality_score=float(np.mean([s.quality_score for s in summaries])),
    )
