#!/usr/bin/env python3
"""
Quick GPU environment check script.
Used in Dockerfile health check and for manual environment validation.
"""
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.gpu_env import get_gpu_env_summary, get_gpu_memory_info


def main():
    """Print GPU environment summary."""
    print("=== GPU Environment Check ===")

    summary = get_gpu_env_summary()

    print(f"CUDA Available: {summary['cuda_available']}")
    print(f"Device Count: {summary['device_count']}")
    print(f"Primary Device: {summary['device_name_0'] or 'N/A'}")
    print(f"CUDA_VISIBLE_DEVICES: {summary['visible_devices'] or 'Not set'}")

    if summary['cuda_available'] and summary['device_count'] > 0:
        print("\n=== GPU Memory (Device 0) ===")
        mem_info = get_gpu_memory_info(0)
        if mem_info:
            print(f"Total Memory: {mem_info['total_mb']} MB")
            print(f"Allocated: {mem_info['allocated_mb']} MB")
            print(f"Reserved: {mem_info['reserved_mb']} MB")
            print(f"Free: {mem_info['free_mb']} MB")
        else:
            print("Memory info unavailable")

    print("\n=== Status ===")
    if summary['cuda_available']:
        print("✓ GPU environment ready")
        return 0
    else:
        print("✗ No GPU detected (CPU-only mode)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
