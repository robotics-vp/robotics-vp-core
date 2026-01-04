"""
Scene IR Tracker Evaluation Report Generator.

Computes production-ready metrics from scene tracks.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EvalMetrics:
    """Evaluation metrics for Scene IR Tracker.
    
    Attributes:
        id_switch_rate_per_min: ID switches per minute of video.
        ir_loss_median: Median IR loss across all entities.
        ir_loss_p90: 90th percentile IR loss.
        ir_loss_mean: Mean IR loss.
        pct_diverged: Percentage of frames where refinement diverged.
        pct_converged: Percentage of frames converged.
        pct_frame_mismatch_fixed: Percentage of frame mismatches auto-fixed.
        runtime_ms_per_frame: Wall-clock time per frame in milliseconds.
        gpu_mem_peak_mb: Peak GPU memory usage in MB.
        export_size_mb_per_min: Export file size per minute of video.
        total_frames: Total frames processed.
        total_tracks: Total tracks detected.
        fps: Frames per second (for rate calculations).
    """
    id_switch_rate_per_min: float = 0.0
    ir_loss_median: float = 0.0
    ir_loss_p90: float = 0.0
    ir_loss_mean: float = 0.0
    pct_diverged: float = 0.0
    pct_converged: float = 0.0
    pct_frame_mismatch_fixed: float = 0.0
    runtime_ms_per_frame: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    export_size_mb_per_min: float = 0.0
    total_frames: int = 0
    total_tracks: int = 0
    fps: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_eval_metrics(
    scene_tracks_data: Dict[str, np.ndarray],
    wall_time_sec: float = 0.0,
    file_size_bytes: int = 0,
    fps: float = 30.0,
    gpu_mem_bytes: int = 0,
    frame_mismatch_fixes: int = 0,
) -> EvalMetrics:
    """Compute evaluation metrics from serialized scene tracks.
    
    Args:
        scene_tracks_data: Dict from scene_tracks_v1 serialization.
        wall_time_sec: Total wall-clock time for processing.
        file_size_bytes: Size of exported file.
        fps: Video frame rate.
        gpu_mem_bytes: Peak GPU memory usage.
        frame_mismatch_fixes: Number of frame mismatches auto-fixed.
    
    Returns:
        EvalMetrics with computed values.
    """
    prefix = "scene_tracks_v1/"
    
    # Extract arrays
    ir_loss = scene_tracks_data.get(f"{prefix}ir_loss", np.array([]))
    converged = scene_tracks_data.get(f"{prefix}converged", np.array([]))
    visibility = scene_tracks_data.get(f"{prefix}visibility", np.array([]))
    
    # Get dimensions
    if ir_loss.ndim == 2:
        T, K = ir_loss.shape
    else:
        T, K = 0, 0
    
    # ID switches - from summary if available
    summary_json = scene_tracks_data.get(f"{prefix}summary_json", np.array(["{}"]))[0]
    try:
        summary = json.loads(str(summary_json))
    except:
        summary = {}
    
    id_switches = summary.get("id_switch_count", 0)
    duration_min = T / fps / 60.0 if fps > 0 else 1.0
    id_switch_rate = id_switches / duration_min if duration_min > 0 else 0.0
    
    # IR loss stats
    ir_flat = ir_loss.flatten()
    ir_flat = ir_flat[np.isfinite(ir_flat)]
    
    ir_median = float(np.median(ir_flat)) if len(ir_flat) > 0 else 0.0
    ir_p90 = float(np.percentile(ir_flat, 90)) if len(ir_flat) > 0 else 0.0
    ir_mean = float(np.mean(ir_flat)) if len(ir_flat) > 0 else 0.0
    
    # Convergence stats
    if converged.size > 0:
        pct_converged = float(np.mean(converged.astype(float)) * 100)
    else:
        pct_converged = 0.0
    
    # Diverged - infer from non-convergence with high loss
    pct_diverged = summary.get("pct_diverged", 0.0)
    
    # Frame mismatch fixes
    pct_mismatch_fixed = (frame_mismatch_fixes / T * 100) if T > 0 else 0.0
    
    # Performance
    runtime_ms = (wall_time_sec * 1000 / T) if T > 0 else 0.0
    gpu_mb = gpu_mem_bytes / 1e6
    
    # Export size
    size_mb_per_min = (file_size_bytes / 1e6) / duration_min if duration_min > 0 else 0.0
    
    return EvalMetrics(
        id_switch_rate_per_min=id_switch_rate,
        ir_loss_median=ir_median,
        ir_loss_p90=ir_p90,
        ir_loss_mean=ir_mean,
        pct_diverged=pct_diverged,
        pct_converged=pct_converged,
        pct_frame_mismatch_fixed=pct_mismatch_fixed,
        runtime_ms_per_frame=runtime_ms,
        gpu_mem_peak_mb=gpu_mb,
        export_size_mb_per_min=size_mb_per_min,
        total_frames=T,
        total_tracks=K,
        fps=fps,
    )


def generate_eval_report_md(
    metrics: EvalMetrics,
    clip_metrics: Optional[List[EvalMetrics]] = None,
    title: str = "Scene IR Tracker Evaluation Report",
) -> str:
    """Generate markdown evaluation report.
    
    Args:
        metrics: Aggregate metrics.
        clip_metrics: Optional per-clip metrics.
        title: Report title.
    
    Returns:
        Markdown string.
    """
    lines = [
        f"# {title}",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| ID Switch Rate | {metrics.id_switch_rate_per_min:.2f} / min |",
        f"| IR Loss (median) | {metrics.ir_loss_median:.4f} |",
        f"| IR Loss (p90) | {metrics.ir_loss_p90:.4f} |",
        f"| IR Loss (mean) | {metrics.ir_loss_mean:.4f} |",
        f"| % Converged | {metrics.pct_converged:.1f}% |",
        f"| % Diverged | {metrics.pct_diverged:.1f}% |",
        f"| % Frame Mismatch Fixed | {metrics.pct_frame_mismatch_fixed:.1f}% |",
        f"| Runtime | {metrics.runtime_ms_per_frame:.1f} ms/frame |",
        f"| GPU Memory Peak | {metrics.gpu_mem_peak_mb:.1f} MB |",
        f"| Export Size | {metrics.export_size_mb_per_min:.2f} MB/min |",
        f"| Total Frames | {metrics.total_frames} |",
        f"| Total Tracks | {metrics.total_tracks} |",
        "",
    ]
    
    # Quality assessment
    lines.append("## Quality Assessment")
    lines.append("")
    
    if metrics.pct_diverged > 10:
        lines.append("> [!WARNING]")
        lines.append(f"> High divergence rate: {metrics.pct_diverged:.1f}%")
        lines.append("")
    
    if metrics.ir_loss_p90 > 0.5:
        lines.append("> [!CAUTION]")
        lines.append(f"> High IR loss p90: {metrics.ir_loss_p90:.4f}")
        lines.append("")
    
    if metrics.id_switch_rate_per_min > 5:
        lines.append("> [!WARNING]")
        lines.append(f"> High ID switch rate: {metrics.id_switch_rate_per_min:.2f}/min")
        lines.append("")
    
    if metrics.pct_diverged <= 5 and metrics.ir_loss_p90 <= 0.3:
        lines.append("> [!NOTE]")
        lines.append("> Quality metrics look healthy ✅")
        lines.append("")
    
    # Per-clip breakdown if provided
    if clip_metrics:
        lines.append("## Per-Clip Breakdown")
        lines.append("")
        lines.append("| Clip | Frames | IR Loss Med | % Diverged | Runtime |")
        lines.append("|------|--------|-------------|------------|---------|")
        for i, cm in enumerate(clip_metrics):
            lines.append(
                f"| {i+1} | {cm.total_frames} | {cm.ir_loss_median:.4f} | "
                f"{cm.pct_diverged:.1f}% | {cm.runtime_ms_per_frame:.0f}ms |"
            )
        lines.append("")
    
    return "\n".join(lines)


def save_eval_report(
    report_md: str,
    output_path: Path,
) -> None:
    """Save evaluation report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md)
