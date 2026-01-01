"""
Performance guardrails for Scene IR Tracker.

Provides timeout, fast mode, and batching utilities.
"""
from __future__ import annotations

import functools
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class TimeoutError(Exception):
    """Raised when operation exceeds time limit."""
    pass


class FrameTimeoutError(TimeoutError):
    """Raised when a single frame exceeds processing time limit."""
    pass


def timeout_guard(
    max_seconds: float,
    on_timeout: str = "raise",
) -> Callable[[F], F]:
    """Decorator to enforce timeout on function execution.
    
    Args:
        max_seconds: Maximum allowed execution time.
        on_timeout: Action on timeout: "raise" or "warn".
    
    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            if elapsed > max_seconds:
                msg = f"{func.__name__} took {elapsed:.2f}s, limit {max_seconds:.2f}s"
                if on_timeout == "raise":
                    raise FrameTimeoutError(msg)
                else:
                    logger.warning(msg)
            
            return result
        return wrapper  # type: ignore
    return decorator


@dataclass
class FastModePreset:
    """Preset configuration for fast mode processing.
    
    Reduces quality for speed.
    """
    # IR Refiner settings
    num_pose_iters: int = 5  # vs default 20
    num_shape_iters: int = 3  # vs default 10
    num_texture_iters: int = 2  # vs default 5
    
    # LPIPS settings
    lpips_every_n_frames: int = 5  # Skip LPIPS on most frames
    lpips_weight: float = 0.05  # Reduced weight
    
    # Early stopping
    early_stop_patience: int = 3  # vs default 10
    convergence_threshold: float = 1e-3  # vs default 1e-4
    
    @classmethod
    def standard(cls) -> "FastModePreset":
        """Standard fast mode preset."""
        return cls()
    
    @classmethod
    def ultra_fast(cls) -> "FastModePreset":
        """Ultra-fast preset for real-time."""
        return cls(
            num_pose_iters=2,
            num_shape_iters=1,
            num_texture_iters=1,
            lpips_every_n_frames=10,
            lpips_weight=0.0,
            early_stop_patience=2,
        )


@dataclass
class PerformanceStats:
    """Performance statistics for a processing run."""
    total_frames: int = 0
    total_wall_time_sec: float = 0.0
    max_frame_time_ms: float = 0.0
    min_frame_time_ms: float = float("inf")
    timeouts: int = 0
    skipped_lpips: int = 0
    
    @property
    def avg_frame_time_ms(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.total_wall_time_sec * 1000 / self.total_frames
    
    @property
    def fps(self) -> float:
        if self.total_wall_time_sec == 0:
            return 0.0
        return self.total_frames / self.total_wall_time_sec


class PerformanceMonitor:
    """Monitor and enforce performance constraints."""
    
    def __init__(
        self,
        max_frame_time_ms: float = 500.0,
        on_timeout: str = "warn",
    ):
        self.max_frame_time_ms = max_frame_time_ms
        self.on_timeout = on_timeout
        self.stats = PerformanceStats()
        self._frame_start: Optional[float] = None
    
    def start_frame(self) -> None:
        """Mark start of frame processing."""
        self._frame_start = time.perf_counter()
    
    def end_frame(self) -> float:
        """Mark end of frame processing.
        
        Returns:
            Frame time in milliseconds.
        """
        if self._frame_start is None:
            return 0.0
        
        elapsed_sec = time.perf_counter() - self._frame_start
        elapsed_ms = elapsed_sec * 1000
        
        self.stats.total_frames += 1
        self.stats.total_wall_time_sec += elapsed_sec
        self.stats.max_frame_time_ms = max(self.stats.max_frame_time_ms, elapsed_ms)
        self.stats.min_frame_time_ms = min(self.stats.min_frame_time_ms, elapsed_ms)
        
        if elapsed_ms > self.max_frame_time_ms:
            self.stats.timeouts += 1
            msg = f"Frame {self.stats.total_frames} took {elapsed_ms:.0f}ms (limit: {self.max_frame_time_ms:.0f}ms)"
            if self.on_timeout == "raise":
                raise FrameTimeoutError(msg)
            else:
                logger.warning(msg)
        
        self._frame_start = None
        return elapsed_ms
    
    def reset(self) -> None:
        """Reset statistics."""
        self.stats = PerformanceStats()
        self._frame_start = None


def apply_fast_mode_to_config(
    config: Any,
    preset: FastModePreset,
) -> Any:
    """Apply fast mode preset to IRRefinerConfig.
    
    Args:
        config: IRRefinerConfig instance.
        preset: FastModePreset to apply.
    
    Returns:
        Modified config.
    """
    if hasattr(config, "num_pose_iters"):
        config.num_pose_iters = preset.num_pose_iters
    if hasattr(config, "num_shape_iters"):
        config.num_shape_iters = preset.num_shape_iters
    if hasattr(config, "num_texture_iters"):
        config.num_texture_iters = preset.num_texture_iters
    if hasattr(config, "lpips_weight"):
        config.lpips_weight = preset.lpips_weight
    if hasattr(config, "early_stop_patience"):
        config.early_stop_patience = preset.early_stop_patience
    if hasattr(config, "convergence_threshold"):
        config.convergence_threshold = preset.convergence_threshold
    
    return config


def estimate_processing_time(
    num_frames: int,
    num_tracks: int,
    fast_mode: bool = False,
) -> float:
    """Estimate processing time in seconds.
    
    Args:
        num_frames: Number of frames.
        num_tracks: Average number of tracks.
        fast_mode: Using fast mode.
    
    Returns:
        Estimated time in seconds.
    """
    # Empirical estimates (tune based on real runs)
    base_time_per_frame_ms = 50.0 if fast_mode else 200.0
    per_track_overhead_ms = 10.0 if fast_mode else 50.0
    
    total_ms = num_frames * (base_time_per_frame_ms + num_tracks * per_track_overhead_ms)
    return total_ms / 1000.0
