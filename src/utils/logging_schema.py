"""
Canonical logging field names used across samplers/curricula/ontology logs.
Also includes standardized schemas for training and demo logs.

Integration with Process Reward:
    Use make_demo_episode_log_entry() with extra=format_log_for_training(pr_result)
    to include process reward metrics alongside MHN/SceneIR summaries.
"""
from typing import Any, Dict, Optional
from datetime import datetime
import json
import os


LOG_FIELDS = {
    "task_id": "task_id",
    "episode_id": "episode_id",
    "sampler_strategy": "sampler_strategy",
    "curriculum_phase": "curriculum_phase",
    "objective_preset": "objective_preset",
    "objective_vector": "objective_vector",
    "pack_id": "pack_id",
    "backend": "backend",
}

POLICY_LOG_FIELDS = {
    "policy_name": "policy",
    "input_features": "features",
    "output": "target",
    "meta": "meta",
    "timestamp": "timestamp",
    "task_id": "task_id",
    "episode_id": "episode_id",
    "datapack_id": "datapack_id",
}


# ============================================================================
# Training Log Schema (for GPU training jobs)
# ============================================================================

TRAINING_LOG_FIELDS = {
    "timestamp": str,       # ISO 8601 format
    "run_name": str,        # Unique identifier for this training run
    "step": int,            # Global training step
    "epoch": int,           # Current epoch
    "phase": str,           # "train" or "val"
    "loss": float,          # Primary loss value
    "lr": float,            # Learning rate
    "amp_enabled": bool,    # Whether AMP was enabled
    "checkpointing_enabled": bool, # Whether activation checkpointing was enabled
    "gpu_mem_mb": Optional[int],    # GPU memory used in MB
    "gpu_util_pct": Optional[int],  # GPU utilization percentage
    "task_id": str,         # Task identifier (e.g., "drawer_open")
    "seed": int,            # Random seed
    "config_digest": str,   # Short hash of config
    "extra": dict,          # Model-specific metrics
}


def make_training_log_entry(
    run_name: str,
    step: int,
    epoch: int,
    phase: str,
    loss: float,
    lr: float,
    task_id: str,
    seed: int,
    config_digest: str,
    amp_enabled: bool = False,
    checkpointing_enabled: bool = False,
    gpu_mem_mb: Optional[int] = None,
    gpu_util_pct: Optional[int] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Create a standardized training log entry.

    Args:
        run_name: Unique identifier for this training run
        step: Global training step
        epoch: Current epoch
        phase: "train" or "val"
        loss: Primary loss value
        lr: Learning rate
        task_id: Task identifier
        seed: Random seed
        config_digest: Short hash of config
        amp_enabled: Whether AMP was enabled
        checkpointing_enabled: Whether activation checkpointing was enabled
        gpu_mem_mb: GPU memory used in MB (optional)
        gpu_util_pct: GPU utilization percentage (optional)
        extra: Model-specific metrics (optional)

    Returns:
        Dictionary with standardized training log fields
    """
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_name": run_name,
        "step": step,
        "epoch": epoch,
        "phase": phase,
        "loss": float(loss),
        "lr": float(lr),
        "amp_enabled": amp_enabled,
        "checkpointing_enabled": checkpointing_enabled,
        "gpu_mem_mb": gpu_mem_mb,
        "gpu_util_pct": gpu_util_pct,
        "task_id": task_id,
        "seed": seed,
        "config_digest": config_digest,
        "extra": extra or {},
    }


def write_training_log_entry(
    filepath: str,
    entry: dict,
) -> None:
    """
    Append a training log entry to a JSONL file.
    Creates parent directories if needed.

    Args:
        filepath: Path to JSONL log file
        entry: Training log entry dictionary
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# Demo Simulation Log Schema
# ============================================================================

DEMO_EPISODE_LOG_FIELDS = {
    "episode_id": int,
    "success": bool,
    "total_reward": float,
    "mpl_estimate": Optional[float],
    "energy_wh": Optional[float],
    "ood_events": int,
    "recovery_events": int,
    "backend": str,
    "seed": int,
    # Process Reward fields (optional, from format_log_for_training)
    "phi_star_mean": Optional[float],
    "phi_star_final": Optional[float],
    "phi_star_delta": Optional[float],
    "conf_mean": Optional[float],
    "conf_p10": Optional[float],
    "r_shape_sum": Optional[float],
    "disagreement_mean": Optional[float],
    "entropy_mean": Optional[float],
    "phi_B_disabled": Optional[bool],
    # Upstream quality scores (optional)
    "scene_ir_quality": Optional[float],
    "motion_quality": Optional[float],
    "mhn_plausibility": Optional[float],
    "mhn_difficulty": Optional[float],
}

DEMO_STEP_LOG_FIELDS = {
    "episode_id": int,
    "t": int,
    "action": list,  # JSON-serializable action
    "reward": float,
    "done": bool,
}


def make_demo_episode_log_entry(
    episode_id: int,
    success: bool,
    total_reward: float,
    backend: str,
    seed: int,
    mpl_estimate: Optional[float] = None,
    energy_wh: Optional[float] = None,
    ood_events: int = 0,
    recovery_events: int = 0,
    extra: Optional[dict] = None,
) -> dict:
    """
    Create a standardized demo episode log entry.

    Args:
        episode_id: Episode identifier
        success: Whether episode succeeded
        total_reward: Total reward accumulated
        backend: Environment backend used
        seed: Random seed
        mpl_estimate: Marginal product estimate (optional)
        energy_wh: Energy consumption in watt-hours (optional)
        ood_events: Number of out-of-distribution events
        recovery_events: Number of recovery events
        extra: Additional fields (optional)

    Returns:
        Dictionary with standardized demo episode log fields
    """
    entry = {
        "episode_id": episode_id,
        "success": success,
        "total_reward": float(total_reward),
        "mpl_estimate": float(mpl_estimate) if mpl_estimate is not None else None,
        "energy_wh": float(energy_wh) if energy_wh is not None else None,
        "ood_events": ood_events,
        "recovery_events": recovery_events,
        "backend": backend,
        "seed": seed,
    }
    if extra:
        entry.update(extra)
    return entry


def make_demo_step_log_entry(
    episode_id: int,
    t: int,
    action: list,
    reward: float,
    done: bool,
    extra: Optional[dict] = None,
) -> dict:
    """
    Create a standardized demo step log entry.

    Args:
        episode_id: Episode identifier
        t: Timestep within episode
        action: Action taken (JSON-serializable list)
        reward: Reward received
        done: Whether episode is done
        extra: Additional fields (optional)

    Returns:
        Dictionary with standardized demo step log fields
    """
    entry = {
        "episode_id": episode_id,
        "t": t,
        "action": action,
        "reward": float(reward),
        "done": done,
    }
    if extra:
        entry.update(extra)
    return entry


def write_demo_log_entry(
    filepath: str,
    entry: dict,
) -> None:
    """
    Append a demo log entry to a JSONL file.
    Creates parent directories if needed.

    Args:
        filepath: Path to JSONL log file
        entry: Demo log entry dictionary
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# Process Reward Integration
# ============================================================================

def make_demo_episode_log_with_process_reward(
    episode_id: int,
    success: bool,
    total_reward: float,
    backend: str,
    seed: int,
    process_reward_result: Optional[Any] = None,
    scene_ir_quality: Optional[float] = None,
    motion_quality: Optional[float] = None,
    mhn_summary: Optional[Any] = None,
    fusion_override: Optional[Any] = None,
    mpl_estimate: Optional[float] = None,
    energy_wh: Optional[float] = None,
    ood_events: int = 0,
    recovery_events: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a demo episode log entry with integrated process reward metrics.

    This is the recommended way to log episodes with process reward data.
    It merges process reward metrics with the standard demo log schema.

    Args:
        episode_id: Episode identifier
        success: Whether episode succeeded
        total_reward: Total reward accumulated
        backend: Environment backend used
        seed: Random seed
        process_reward_result: ProcessRewardEpisodeOutput from process_reward_episode()
        scene_ir_quality: Optional scene IR quality score
        motion_quality: Optional motion quality score
        mhn_summary: Optional MHNSummary object
        fusion_override: Optional FusionOverride used
        mpl_estimate: Marginal product estimate (optional)
        energy_wh: Energy consumption in watt-hours (optional)
        ood_events: Number of out-of-distribution events
        recovery_events: Number of recovery events
        extra: Additional fields (optional)

    Returns:
        Dictionary with all demo log fields plus process reward metrics.

    Example:
        >>> from src.process_reward import process_reward_episode, ProcessRewardConfig
        >>> cfg = ProcessRewardConfig(online_mode=False)  # Offline eval
        >>> pr_result = process_reward_episode(scene_tracks, "task", cfg=cfg)
        >>> log_entry = make_demo_episode_log_with_process_reward(
        ...     episode_id=1,
        ...     success=True,
        ...     total_reward=10.5,
        ...     backend="mujoco",
        ...     seed=42,
        ...     process_reward_result=pr_result,
        ...     scene_ir_quality=0.85,
        ... )
        >>> write_demo_log_entry("logs/episodes.jsonl", log_entry)
    """
    # Start with base demo log entry
    entry = make_demo_episode_log_entry(
        episode_id=episode_id,
        success=success,
        total_reward=total_reward,
        backend=backend,
        seed=seed,
        mpl_estimate=mpl_estimate,
        energy_wh=energy_wh,
        ood_events=ood_events,
        recovery_events=recovery_events,
    )

    # Add process reward metrics if available
    if process_reward_result is not None:
        try:
            from src.process_reward.logging_utils import format_log_for_training
            pr_log = format_log_for_training(
                result=process_reward_result,
                scene_ir_quality=scene_ir_quality,
                motion_quality=motion_quality,
                mhn_summary=mhn_summary,
                fusion_override=fusion_override,
            )
            entry.update(pr_log)
        except ImportError:
            # Process reward module not available; skip
            pass

    # Add upstream quality scores directly if PR not provided
    if process_reward_result is None:
        if scene_ir_quality is not None:
            entry["scene_ir_quality"] = scene_ir_quality
        if motion_quality is not None:
            entry["motion_quality"] = motion_quality
        if mhn_summary is not None:
            entry["mhn_plausibility"] = getattr(mhn_summary, "plausibility_score", None)
            entry["mhn_difficulty"] = getattr(mhn_summary, "structural_difficulty", None)

    # Add any extra fields
    if extra:
        entry.update(extra)

    return entry
