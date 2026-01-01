#!/usr/bin/env python3
"""
Run Scene IR Tracker Evaluation on golden clips.

Usage:
    python scripts/run_scene_ir_eval.py --input-dir data/golden_clips --output eval_report.md
    python scripts/run_scene_ir_eval.py --input-npz trajectory.npz --output eval_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# GPU memory tracking
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def load_scene_tracks_from_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load scene_tracks_v1 arrays from npz file."""
    data = dict(np.load(npz_path, allow_pickle=False))
    
    # Filter to scene_tracks_v1 keys
    prefix = "scene_tracks_v1/"
    filtered = {k: v for k, v in data.items() if k.startswith(prefix)}
    
    if not filtered:
        # Maybe keys don't have prefix
        filtered = data
    
    return filtered


def get_gpu_memory_peak() -> int:
    """Get peak GPU memory in bytes."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def reset_gpu_memory_stats() -> None:
    """Reset GPU memory stats."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def run_evaluation(
    input_paths: List[Path],
    fps: float = 30.0,
    verbose: bool = True,
) -> "EvalMetrics":
    """Run evaluation on input files.
    
    Args:
        input_paths: List of npz files to evaluate.
        fps: Frame rate for rate calculations.
        verbose: Print progress.
    
    Returns:
        Aggregate EvalMetrics.
    """
    from src.analytics.scene_ir_eval_report import (
        compute_eval_metrics,
        EvalMetrics,
    )
    
    all_ir_loss = []
    all_converged = []
    total_id_switches = 0
    total_frames = 0
    total_tracks = 0
    total_diverged_frames = 0
    total_file_bytes = 0
    total_wall_time = 0.0
    
    reset_gpu_memory_stats()
    
    for path in input_paths:
        if verbose:
            print(f"Processing: {path}")
        
        start = time.perf_counter()
        
        try:
            data = load_scene_tracks_from_npz(path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        elapsed = time.perf_counter() - start
        total_wall_time += elapsed
        total_file_bytes += path.stat().st_size
        
        # Extract arrays
        prefix = "scene_tracks_v1/"
        ir_loss = data.get(f"{prefix}ir_loss", data.get("ir_loss", np.array([])))
        converged = data.get(f"{prefix}converged", data.get("converged", np.array([])))
        
        if ir_loss.size > 0:
            all_ir_loss.append(ir_loss.flatten())
            if ir_loss.ndim == 2:
                T, K = ir_loss.shape
                total_frames += T
                total_tracks = max(total_tracks, K)
        
        if converged.size > 0:
            all_converged.append(converged.flatten())
        
        # Get summary
        summary_json = data.get(f"{prefix}summary_json", data.get("summary_json", np.array(["{}"])))
        if summary_json.size > 0:
            try:
                summary = json.loads(str(summary_json[0]))
            except:
                summary = {}
            total_id_switches += summary.get("id_switch_count", 0)
            
            # Diverged frames
            pct_div = summary.get("pct_diverged", 0.0)
            if ir_loss.ndim == 2:
                total_diverged_frames += int(ir_loss.shape[0] * pct_div / 100)
    
    # Aggregate
    if all_ir_loss:
        ir_concat = np.concatenate(all_ir_loss)
        ir_concat = ir_concat[np.isfinite(ir_concat)]
        ir_median = float(np.median(ir_concat)) if len(ir_concat) > 0 else 0.0
        ir_p90 = float(np.percentile(ir_concat, 90)) if len(ir_concat) > 0 else 0.0
        ir_mean = float(np.mean(ir_concat)) if len(ir_concat) > 0 else 0.0
    else:
        ir_median = ir_p90 = ir_mean = 0.0
    
    if all_converged:
        conv_concat = np.concatenate(all_converged)
        pct_converged = float(np.mean(conv_concat.astype(float)) * 100)
    else:
        pct_converged = 0.0
    
    duration_min = total_frames / fps / 60.0 if fps > 0 else 1.0
    id_switch_rate = total_id_switches / duration_min if duration_min > 0 else 0.0
    pct_diverged = (total_diverged_frames / total_frames * 100) if total_frames > 0 else 0.0
    runtime_ms = (total_wall_time * 1000 / total_frames) if total_frames > 0 else 0.0
    size_mb_per_min = (total_file_bytes / 1e6) / duration_min if duration_min > 0 else 0.0
    gpu_mb = get_gpu_memory_peak() / 1e6
    
    return EvalMetrics(
        id_switch_rate_per_min=id_switch_rate,
        ir_loss_median=ir_median,
        ir_loss_p90=ir_p90,
        ir_loss_mean=ir_mean,
        pct_converged=pct_converged,
        pct_diverged=pct_diverged,
        pct_frame_mismatch_fixed=0.0,  # TODO: track this
        runtime_ms_per_frame=runtime_ms,
        gpu_mem_peak_mb=gpu_mb,
        export_size_mb_per_min=size_mb_per_min,
        total_frames=total_frames,
        total_tracks=total_tracks,
        fps=fps,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Scene IR Tracker Evaluation"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing npz files to evaluate",
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        action="append",
        help="Single npz file to evaluate (can be repeated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate for rate calculations",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    # Collect input files
    input_paths: List[Path] = []
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if input_dir.is_dir():
            input_paths.extend(input_dir.glob("**/*.npz"))
    
    if args.input_npz:
        for p in args.input_npz:
            path = Path(p)
            if path.exists():
                input_paths.append(path)
    
    if not input_paths:
        print("ERROR: No input files found", file=sys.stderr)
        print("Use --input-dir or --input-npz to specify inputs", file=sys.stderr)
        return 1
    
    if not args.quiet:
        print(f"Found {len(input_paths)} files to evaluate")
        print()
    
    # Run evaluation
    from src.analytics.scene_ir_eval_report import (
        generate_eval_report_md,
        save_eval_report,
    )
    
    metrics = run_evaluation(input_paths, fps=args.fps, verbose=not args.quiet)
    
    # Generate report
    report = generate_eval_report_md(metrics)
    
    # Save
    output_path = Path(args.output)
    save_eval_report(report, output_path)
    
    if not args.quiet:
        print()
        print("=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Report saved to: {output_path}")
        print()
        print("Summary:")
        print(f"  ID Switch Rate: {metrics.id_switch_rate_per_min:.2f}/min")
        print(f"  IR Loss (p90): {metrics.ir_loss_p90:.4f}")
        print(f"  % Diverged: {metrics.pct_diverged:.1f}%")
        print(f"  Runtime: {metrics.runtime_ms_per_frame:.1f}ms/frame")
        print(f"  Export Size: {metrics.export_size_mb_per_min:.2f} MB/min")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
