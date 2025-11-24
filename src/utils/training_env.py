"""
Training environment helpers.
Provides utilities for configuring AMP, activation checkpointing, and device management
based on the pipeline configuration.
"""
import torch
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def should_use_amp(cfg: Dict[str, Any]) -> bool:
    """
    Check if AMP (Automatic Mixed Precision) should be enabled.
    
    Args:
        cfg: Pipeline configuration dictionary
        
    Returns:
        True if AMP is enabled in config and CUDA is available
    """
    if not torch.cuda.is_available():
        return False
        
    return cfg.get("training", {}).get("amp", False)

def should_checkpoint(cfg: Dict[str, Any]) -> bool:
    """
    Check if activation checkpointing should be enabled.
    
    Args:
        cfg: Pipeline configuration dictionary
        
    Returns:
        True if activation checkpointing is enabled in config
    """
    return cfg.get("training", {}).get("activation_checkpointing", False)

def get_device(cfg: Dict[str, Any]) -> torch.device:
    """
    Get the target device for training.
    
    Args:
        cfg: Pipeline configuration dictionary
        
    Returns:
        torch.device: The target device (cuda or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def device_info() -> Dict[str, Any]:
    """
    Get information about the current device environment.
    
    Returns:
        Dictionary with device info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        
    return info

def configure_training_env(cfg: Dict[str, Any]) -> None:
    """
    Configure the training environment based on config.
    Sets float32 matmul precision, cudnn benchmark, etc.
    
    Args:
        cfg: Pipeline configuration dictionary
    """
    if torch.cuda.is_available():
        # Set float32 matmul precision for A100/H100 if needed
        # torch.set_float32_matmul_precision('high')
        
        if cfg.get("determinism", {}).get("enforce_cuda_determinism", False):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True

def checkpoint_if_enabled(fn, *args, enabled=False, **kwargs):
    """
    Apply activation checkpointing if enabled.
    
    Args:
        fn: Function or module to checkpoint
        *args: Arguments to fn
        enabled: Whether checkpointing is enabled
        **kwargs: Keyword arguments to fn (note: checkpoint doesn't support kwargs directly, 
                 so this might need wrapping if kwargs are used)
                 
    Returns:
        Output of fn(*args, **kwargs)
    """
    if enabled:
        from torch.utils.checkpoint import checkpoint
        # checkpoint only accepts positional args
        if kwargs:
            def wrapper(*args):
                return fn(*args, **kwargs)
            return checkpoint(wrapper, *args, use_reentrant=False)
        return checkpoint(fn, *args, use_reentrant=False)
    else:
        return fn(*args, **kwargs)

def check_vram_safety(threshold_mb: int = 1000) -> bool:
    """Check if available VRAM is above threshold."""
    if not torch.cuda.is_available():
        return True
    try:
        free_mem = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        return free_mem > threshold_mb
    except Exception:
        return True

def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_with_oom_recovery(train_step_fn, *args, **kwargs):
    """
    Run a training step with OOM recovery.
    
    Args:
        train_step_fn: Function to execute (e.g., train_epoch or single step)
        *args: Arguments to fn
        **kwargs: Keyword arguments to fn
        
    Returns:
        Result of fn
    """
    try:
        return train_step_fn(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("[WARN] CUDA OOM detected. Clearing cache and retrying...")
            clear_gpu_cache()
            # In a real implementation, we would dynamically reduce batch size here.
            # However, batch size is usually set in DataLoader which is passed in.
            # We can't easily change DataLoader batch size on the fly without recreating it.
            # For Phase I, we'll just clear cache and try once more.
            # If it fails again, we let it crash or skip.
            try:
                return train_step_fn(*args, **kwargs)
            except RuntimeError as e2:
                if "out of memory" in str(e2):
                    print("[ERROR] CUDA OOM persistent. Skipping step/epoch.")
                    # Return None or raise? Raising is safer to stop training loop if needed,
                    # but skipping might be better for robustness.
                    # Let's raise to let the outer loop handle or crash.
                    raise e2
                raise e2
        raise e
