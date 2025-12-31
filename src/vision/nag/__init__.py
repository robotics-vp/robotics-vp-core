"""
Neural Atlas-Graph (NAG) visual representation layer.

Provides a view-specific, object-centric atlas representation that:
- Decomposes rendered sequences into per-object planes + atlases
- Supports precise, structured edits in atlas space
- Re-renders counterfactual videos
- Emits structured "edit vectors" back into the econ stack

Works on top of the LSD vector scene stack (3D geometry + dynamics).
"""

from src.vision.nag.types import (
    CameraParams,
    NAGNodeId,
    PlaneParams,
    PoseSplineParams,
)
from src.vision.nag.plane_node import NAGPlaneNode
from src.vision.nag.scene import NAGScene
from src.vision.nag.renderer import render_scene
from src.vision.nag.fitter import fit_nag_to_sequence
from src.vision.nag.editor import (
    edit_texture_from_rgba,
    edit_pose,
    duplicate_node,
    render_clip,
)

__all__ = [
    # Types
    "CameraParams",
    "NAGNodeId",
    "PlaneParams",
    "PoseSplineParams",
    # Core
    "NAGPlaneNode",
    "NAGScene",
    # Functions
    "render_scene",
    "fit_nag_to_sequence",
    "edit_texture_from_rgba",
    "edit_pose",
    "duplicate_node",
    "render_clip",
]
