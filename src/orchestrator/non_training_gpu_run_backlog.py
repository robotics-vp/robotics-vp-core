"""Dedicated non-training GPU run backlog wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

from .loop_run_backlog import (
    LoopRunAssessment,
    LoopRunBacklogItem,
    collect_host_capabilities,
    evaluate_loop_run_backlog,
    load_loop_run_backlog,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NON_TRAINING_GPU_RUN_BACKLOG_PATH = (
    REPO_ROOT / "scripts" / "NON_TRAINING_GPU_RUN_BACKLOG.json"
)


def load_non_training_gpu_run_backlog(
    path: Optional[Path] = None,
) -> list[LoopRunBacklogItem]:
    return load_loop_run_backlog(path or DEFAULT_NON_TRAINING_GPU_RUN_BACKLOG_PATH)


def evaluate_non_training_gpu_run_backlog(
    *,
    backlog_items: Optional[Sequence[LoopRunBacklogItem]] = None,
    backlog_path: Optional[Path] = None,
    host_capabilities: Optional[Mapping[str, object]] = None,
) -> list[LoopRunAssessment]:
    return evaluate_loop_run_backlog(
        backlog_items=backlog_items,
        backlog_path=backlog_path or DEFAULT_NON_TRAINING_GPU_RUN_BACKLOG_PATH,
        host_capabilities=host_capabilities,
    )


__all__ = [
    "DEFAULT_NON_TRAINING_GPU_RUN_BACKLOG_PATH",
    "LoopRunAssessment",
    "LoopRunBacklogItem",
    "collect_host_capabilities",
    "evaluate_non_training_gpu_run_backlog",
    "load_non_training_gpu_run_backlog",
]
