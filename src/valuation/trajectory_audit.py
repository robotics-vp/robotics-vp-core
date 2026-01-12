"""TrajectoryAuditV1 producer for training loops.

Provides utility functions to create real TrajectoryAuditV1 records
from episode rollout data. This is the P0 implementation for regal
grounding on actual physics/reward data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.contracts.schemas import TrajectoryAuditV1
from src.utils.config_digest import sha256_json


def create_trajectory_audit(
    episode_id: str,
    num_steps: int,
    actions: Optional[List[List[float]]] = None,
    rewards: Optional[List[float]] = None,
    reward_components: Optional[Dict[str, List[float]]] = None,
    events: Optional[List[str]] = None,
    penetrations: Optional[List[float]] = None,
    velocities: Optional[List[List[float]]] = None,
    velocity_threshold: float = 10.0,
    scene_tracks_sha: Optional[str] = None,
    bev_summary_sha: Optional[str] = None,
) -> TrajectoryAuditV1:
    """Create a TrajectoryAuditV1 from episode rollout data.
    
    This is the canonical producer for training loop integration.
    
    Args:
        episode_id: Unique episode identifier
        num_steps: Number of timesteps in episode
        actions: List of action vectors per timestep
        rewards: List of scalar rewards per timestep
        reward_components: Dict of component_name -> list of per-step values
        events: List of discrete events that occurred
        penetrations: Collision penetration depths per timestep
        velocities: Velocity vectors per timestep (for spike detection)
        velocity_threshold: Threshold above which velocity is a "spike"
        scene_tracks_sha: SHA of associated scene tracks
        bev_summary_sha: SHA of associated BEV summary
        
    Returns:
        TrajectoryAuditV1 ready for regal evaluation
    """
    import math
    
    # Compute action statistics if provided
    action_mean: Optional[List[float]] = None
    action_std: Optional[List[float]] = None
    
    if actions and len(actions) > 0:
        action_dim = len(actions[0])
        action_mean = [0.0] * action_dim
        action_std = [0.0] * action_dim
        
        # Compute mean
        for action in actions:
            for i, val in enumerate(action):
                action_mean[i] += val / len(actions)
        
        # Compute std
        if len(actions) > 1:
            for action in actions:
                for i, val in enumerate(action):
                    action_std[i] += (val - action_mean[i]) ** 2
            action_std = [math.sqrt(v / (len(actions) - 1)) for v in action_std]
        else:
            action_std = [0.0] * action_dim
    
    # Compute total return and component totals
    total_return = sum(rewards) if rewards else 0.0
    
    component_totals: Optional[Dict[str, float]] = None
    if reward_components:
        component_totals = {}
        for comp_name, values in reward_components.items():
            component_totals[comp_name] = sum(values)
    
    # Count events
    event_counts: Optional[Dict[str, int]] = None
    if events:
        event_counts = {}
        for event in events:
            event_counts[event] = event_counts.get(event, 0) + 1
    
    # Compute physics anomaly stats
    penetration_max: Optional[float] = None
    if penetrations:
        penetration_max = max(penetrations)
    
    velocity_spike_count = 0
    if velocities:
        for vel in velocities:
            speed = math.sqrt(sum(v ** 2 for v in vel))
            if speed > velocity_threshold:
                velocity_spike_count += 1
    
    return TrajectoryAuditV1(
        episode_id=episode_id,
        num_steps=num_steps,
        action_mean=action_mean,
        action_std=action_std,
        total_return=total_return,
        reward_components=component_totals,
        events=events or [],
        event_counts=event_counts,
        penetration_max=penetration_max,
        velocity_spike_count=velocity_spike_count,
        scene_tracks_sha=scene_tracks_sha,
        bev_summary_sha=bev_summary_sha,
    )


def aggregate_trajectory_audits(audits: List[TrajectoryAuditV1]) -> str:
    """Compute aggregate SHA for a list of trajectory audits.
    
    Deterministic: sorted by episode_id before hashing.
    
    Args:
        audits: List of TrajectoryAuditV1 to aggregate
        
    Returns:
        SHA-256 of sorted audit SHAs
    """
    if not audits:
        return sha256_json([])
    
    sorted_shas = sorted(audit.sha256() for audit in audits)
    return sha256_json(sorted_shas)


__all__ = [
    "create_trajectory_audit",
    "aggregate_trajectory_audits",
]
