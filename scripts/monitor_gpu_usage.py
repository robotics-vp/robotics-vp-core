#!/usr/bin/env python3
"""
Lightweight GPU monitoring script.
Logs GPU usage metrics to JSONL at regular intervals.

Usage:
    python3 scripts/monitor_gpu_usage.py \\
        --interval-seconds 5 \\
        --output results/monitoring/gpu_usage.jsonl \\
        --max-samples 1000
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor GPU usage and log to JSONL"
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=5.0,
        help="Sampling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/monitoring/gpu_usage.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to collect (default: unlimited)",
    )
    return parser.parse_args()


def check_nvidia_smi() -> bool:
    """
    Check if nvidia-smi is available.

    Returns:
        True if nvidia-smi is available, False otherwise
    """
    try:
        subprocess.run(
            ["nvidia-smi", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def query_gpu_stats() -> List[Dict[str, Any]]:
    """
    Query GPU statistics using nvidia-smi.

    Returns:
        List of GPU stat dictionaries (one per GPU)
    """
    try:
        # Query nvidia-smi with CSV output
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )

        # Parse CSV output
        gpu_stats = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue

            try:
                gpu_stat = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                    "temperature_c": int(parts[5]),
                }
                gpu_stats.append(gpu_stat)
            except (ValueError, IndexError):
                continue

        return gpu_stats

    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check nvidia-smi availability
    if not check_nvidia_smi():
        print("ERROR: nvidia-smi not found. Cannot monitor GPU usage.", file=sys.stderr)
        print("This script requires NVIDIA GPU drivers and nvidia-smi utility.", file=sys.stderr)
        return 1

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GPU Usage Monitor")
    print("=" * 80)
    print(f"Interval: {args.interval_seconds}s")
    print(f"Output: {output_path}")
    print(f"Max samples: {args.max_samples or 'unlimited'}")
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring")
    print()

    sample_count = 0

    try:
        with open(output_path, "a") as f:
            while True:
                # Check max samples
                if args.max_samples is not None and sample_count >= args.max_samples:
                    print(f"\nReached max samples ({args.max_samples}). Exiting.")
                    break

                # Query GPU stats
                gpu_stats = query_gpu_stats()

                if not gpu_stats:
                    print("WARNING: No GPU stats available", file=sys.stderr)
                    time.sleep(args.interval_seconds)
                    continue

                # Log each GPU
                for gpu_stat in gpu_stats:
                    # Write to file
                    f.write(json.dumps(gpu_stat) + "\n")
                    f.flush()

                    # Print to stdout
                    print(
                        f"[{gpu_stat['timestamp']}] "
                        f"GPU {gpu_stat['index']}: "
                        f"{gpu_stat['memory_used_mb']:>6}/{gpu_stat['memory_total_mb']:>6} MB "
                        f"({100 * gpu_stat['memory_used_mb'] / gpu_stat['memory_total_mb']:>5.1f}%) | "
                        f"Util: {gpu_stat['utilization_pct']:>3}% | "
                        f"Temp: {gpu_stat['temperature_c']:>3}°C"
                    )

                sample_count += 1

                # Sleep until next sample
                time.sleep(args.interval_seconds)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print(f"Collected {sample_count} samples.")
        print(f"Output written to: {output_path}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
