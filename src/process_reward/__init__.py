"""
Process Reward Module (Robo-Dopamine-style).

Provides potential-based reward shaping (PBRS) downstream of SceneTracks_v1 and MHN.
Consumes latent features (not pixels), produces progress potential Phi and confidence,
and computes policy-invariant shaped rewards.

Key Components:
    - ProcessRewardConfig: Configuration dataclass
    - FusionOverride: Orchestrator-controllable fusion parameters
    - process_reward_episode: Full episode processing
    - process_reward_step: Per-step processing for RL rollouts

Example:
    >>> from src.process_reward import process_reward_episode, ProcessRewardConfig
    >>> from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
    >>>
    >>> # Load scene tracks
    >>> data = dict(np.load("episode.npz", allow_pickle=False))
    >>> scene_tracks = deserialize_scene_tracks_v1(data)
    >>>
    >>> # Compute process reward
    >>> cfg = ProcessRewardConfig()
    >>> result = process_reward_episode(scene_tracks, instruction="pick up the box", cfg=cfg)
    >>> print(f"Phi_star: {result.phi_star}")
    >>> print(f"Shaped reward sum: {result.r_shape.sum()}")
"""
from __future__ import annotations

from src.process_reward.schemas import (
    ProcessRewardConfig,
    FusionOverride,
    ProcessRewardStepOutput,
    ProcessRewardEpisodeOutput,
    ProgressPerspectives,
    FusionDiagnostics,
    MHNSummary,
)
from src.process_reward.core import (
    process_reward_episode,
    process_reward_step,
)
from src.process_reward.logging_utils import (
    ProcessRewardLogEntry,
    ProcessRewardCorrelationReport,
    OrchestratorPolicy,
    extract_log_entry,
    compute_correlation_report,
    format_log_for_training,
)

__all__ = [
    # Config
    "ProcessRewardConfig",
    "FusionOverride",
    "MHNSummary",
    # Outputs
    "ProcessRewardStepOutput",
    "ProcessRewardEpisodeOutput",
    "ProgressPerspectives",
    "FusionDiagnostics",
    # Core API
    "process_reward_episode",
    "process_reward_step",
    # Logging / Integration
    "ProcessRewardLogEntry",
    "ProcessRewardCorrelationReport",
    "OrchestratorPolicy",
    "extract_log_entry",
    "compute_correlation_report",
    "format_log_for_training",
]
