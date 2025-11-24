"""
Canonical logging field names used across samplers/curricula/ontology logs.
Also includes standardized schemas for training and demo logs.
"""
from typing import Optional
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
