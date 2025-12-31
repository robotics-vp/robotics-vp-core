"""
Neural Atlas-Graph (NAG) visual representation layer.

Provides a view-specific, object-centric atlas representation that:
- Decomposes rendered sequences into per-object planes + atlases
- Supports precise, structured edits in atlas space
- Re-renders counterfactual videos
- Emits structured "edit vectors" back into the econ stack

Works on top of the LSD vector scene stack (3D geometry + dynamics).

Rendering modes:
    1. Concrete renderer: Use SplattingGaussianRenderer for real 3DGS rendering
    2. Pre-rendered: Pass pre_rendered_frames to bypass rendering
    3. Stub mode: Set use_stub_renderer=True in config for testing only
"""

from src.vision.nag.types import (
    CameraParams,
    NAGEditVector,
    NAGNodeId,
    PlaneParams,
    PoseSplineParams,
    make_node_id,
)
from src.vision.nag.plane_node import NAGPlaneNode
from src.vision.nag.scene import NAGScene
from src.vision.nag.renderer import render_scene
from src.vision.nag.fitter import fit_nag_to_sequence, FitterConfig, FitStats
from src.vision.nag.editor import (
    edit_texture_from_rgba,
    edit_pose,
    duplicate_node,
    remove_node,
    apply_color_shift,
    render_clip,
    NAGEditPolicy,
    apply_random_edits,
)
from src.vision.nag.gaussian_renderer import (
    SplattingGaussianRenderer,
    SplattingRendererConfig,
    create_default_renderer,
)
from src.vision.nag.integration_lsd_backend import (
    NAGFromLSDConfig,
    NAGEditPolicyConfig,
    NAGDatapack,
    build_nag_scene_from_lsd_rollout,
    generate_nag_counterfactuals_for_lsd_episode,
    create_camera_from_lsd_config,
)

__all__ = [
    # Types
    "CameraParams",
    "NAGEditVector",
    "NAGNodeId",
    "PlaneParams",
    "PoseSplineParams",
    "make_node_id",
    # Core
    "NAGPlaneNode",
    "NAGScene",
    # Fitter
    "fit_nag_to_sequence",
    "FitterConfig",
    "FitStats",
    # Renderer
    "render_scene",
    "SplattingGaussianRenderer",
    "SplattingRendererConfig",
    "create_default_renderer",
    # Editor
    "edit_texture_from_rgba",
    "edit_pose",
    "duplicate_node",
    "remove_node",
    "apply_color_shift",
    "render_clip",
    "NAGEditPolicy",
    "apply_random_edits",
    # LSD Integration
    "NAGFromLSDConfig",
    "NAGEditPolicyConfig",
    "NAGDatapack",
    "build_nag_scene_from_lsd_rollout",
    "generate_nag_counterfactuals_for_lsd_episode",
    "create_camera_from_lsd_config",
]
