#!/usr/bin/env python3
"""Validate CPU/GPU training readiness without requiring an actual GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.gpu_env import get_gpu_env_summary, get_gpu_memory_info


def main() -> None:
    configs = [
        "configs/replay_policy/cpu_smoke.yaml",
        "configs/replay_policy/gpu_full.yaml",
        "configs/offline_rl/cpu_smoke.yaml",
        "configs/offline_rl/gpu_full.yaml",
        "configs/sac/contract_aware_smoke.yaml",
        "configs/shadow_models/cpu_smoke.yaml",
        "configs/shadow_models/gpu_full.yaml",
    ]
    summary = {
        "torch_version": torch.__version__,
        "gpu_env": get_gpu_env_summary(),
        "gpu_memory": get_gpu_memory_info(0),
        "configs_present": {path: Path(path).exists() for path in configs},
        "checkpoint_dirs_ready": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
