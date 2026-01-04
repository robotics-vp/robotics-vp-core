"""Motion hierarchy learner (MHN) for vectorized trajectories."""

from src.vision.motion_hierarchy.config import MotionHierarchyConfig
from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode
from src.vision.motion_hierarchy.datasets import SyntheticChainDataset, TrajectoryDatasetAdapter, MotionHierarchyBatch
from src.vision.motion_hierarchy.trainer import MotionHierarchyTrainer, run_synthetic_training
from src.vision.motion_hierarchy.metrics import (
    MotionHierarchySummary,
    compute_motion_hierarchy_summary_from_raw,
    compute_motion_hierarchy_summary_from_stats,
    compute_motion_plausibility_flags,
    motion_quality_score,
)

__all__ = [
    "MotionHierarchyConfig",
    "MotionHierarchyNode",
    "SyntheticChainDataset",
    "TrajectoryDatasetAdapter",
    "MotionHierarchyBatch",
    "MotionHierarchyTrainer",
    "run_synthetic_training",
    "MotionHierarchySummary",
    "compute_motion_hierarchy_summary_from_raw",
    "compute_motion_hierarchy_summary_from_stats",
    "compute_motion_plausibility_flags",
    "motion_quality_score",
]
