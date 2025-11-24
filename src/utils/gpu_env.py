"""
GPU environment detection and logging utilities.
Simple helper to detect GPU availability and log environment details.
"""
import os
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


def get_gpu_env_summary() -> dict:
    """
    Get a summary of the GPU environment.

    Returns:
        Dictionary with GPU environment information:
        - cuda_available: Whether CUDA is available
        - device_count: Number of CUDA devices
        - device_name_0: Name of first GPU (if available)
        - visible_devices: CUDA_VISIBLE_DEVICES env var (if set)
    """
    cuda_available = bool(torch and torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name_0 = (
        torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None
    )

    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name_0": device_name_0,
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", None),
    }


def get_gpu_memory_info(device_idx: int = 0) -> Optional[dict]:
    """
    Get GPU memory information for a specific device.

    Args:
        device_idx: GPU device index (default 0)

    Returns:
        Dictionary with memory info (MB) or None if unavailable:
        - total_mb: Total GPU memory
        - allocated_mb: Currently allocated memory
        - reserved_mb: Currently reserved memory
        - free_mb: Free memory
    """
    if not torch or not torch.cuda.is_available():
        return None

    if device_idx >= torch.cuda.device_count():
        return None

    try:
        total = torch.cuda.get_device_properties(device_idx).total_memory
        allocated = torch.cuda.memory_allocated(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)

        return {
            "total_mb": int(total / (1024 ** 2)),
            "allocated_mb": int(allocated / (1024 ** 2)),
            "reserved_mb": int(reserved / (1024 ** 2)),
            "free_mb": int((total - reserved) / (1024 ** 2)),
        }
    except Exception:
        return None


def get_gpu_utilization(device_idx: int = 0) -> Optional[int]:
    """
    Get GPU utilization percentage for a specific device.
    Note: This requires nvidia-ml-py3 package for accurate results.
    Falls back to None if unavailable.

    Args:
        device_idx: GPU device index (default 0)

    Returns:
        GPU utilization percentage (0-100) or None if unavailable
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return int(util.gpu)
    except Exception:
        return None
