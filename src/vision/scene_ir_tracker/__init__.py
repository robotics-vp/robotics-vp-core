"""
Scene IR Tracker Module.

Provides unified perception and tracking combining SAM3D-Body,
SAM3D-Objects as upstream priors and INRTracker-style inverse
rendering refinement with Kalman tracking.
"""

from src.vision.scene_ir_tracker.config import (
    SceneIRTrackerConfig,
    IRRefinerConfig,
    TrackingConfig,
)
from src.vision.scene_ir_tracker.types import (
    SceneEntity3D,
    SceneTracks,
    SceneTrackerMetrics,
)
from src.vision.scene_ir_tracker.tracker import (
    SceneIRTracker,
    create_scene_ir_tracker,
)

__all__ = [
    # Config
    "SceneIRTrackerConfig",
    "IRRefinerConfig",
    "TrackingConfig",
    # Types
    "SceneEntity3D",
    "SceneTracks",
    "SceneTrackerMetrics",
    # Main interface
    "SceneIRTracker",
    "create_scene_ir_tracker",
]
